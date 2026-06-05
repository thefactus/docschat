"""Query planning, intent routing, and history-aware query rewriting.

plan_details() is the single LLM call that:
  1. Classifies intent: greeting | meta | doc_question
  2. Rewrites the query to be self-contained using conversation history
  3. Optionally decomposes into retrieval sub-queries

retrieve_with_plan() runs retrieval given a pre-computed plan.
retrieve_with_planning() is the convenience wrapper used by the eval harness.

_bound_history and _parse_planner_output are extracted as testable pure helpers.
"""

import json
from dataclasses import dataclass
from typing import Literal

import structlog
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.retrieval.pipeline import Chunk, retrieve

log = structlog.get_logger()

# ── Canned responses for short-circuit intents ───────────────────────────────

GREETING_ANSWER = (
    "Hey! I'm DocChat. Upload your documents, then ask me anything — "
    "I'll find the relevant passages and give you a grounded answer with citations."
)

META_ANSWER = (
    "I'm DocChat, a document-grounded assistant. "
    "Upload PDFs, TXT, or Markdown files, then ask questions — "
    "I'll search your documents and answer with inline citations like [filename, p.N]. "
    "I only answer from your uploaded content; I won't use outside knowledge or guess."
)

CANNED: dict[str, str] = {'greeting': GREETING_ANSWER, 'meta': META_ANSWER}

# ── Data type ─────────────────────────────────────────────────────────────────

@dataclass
class IntentPlanDetails:
    intent: Literal['greeting', 'meta', 'doc_question']
    standalone_query: str
    queries: list[str]      # retrieval sub-queries (empty for greeting/meta)
    decomposed: bool
    rewritten: bool         # standalone_query was contextualised from history
    reason: str


# ── LLM prompt ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a query planning component for a document-grounded RAG system.
Given the conversation history (if any) and the user's latest message, classify intent
and prepare retrieval queries. Return JSON only.

Intent categories:
- "greeting": pure salutation (hi/hello/hey/oi/good morning etc.) with no information request.
- "meta": asks what this system is, what it can do, what documents it knows, or similar.
- "doc_question": everything else — questions about documents, follow-ups, off-topic questions.

Output schema (JSON only, no markdown):
{
  "intent": "greeting" | "meta" | "doc_question",
  "standalone_query": "<self-contained user message>",
  "sub_queries": ["<retrieval query 1>"],
  "reason": "string"
}

Rules for standalone_query (doc_question only):
- If the user refers to a topic from history via pronouns ("it", "that", "they"), ellipsis
  ("how many?", "and the carryover?"), or implicit reference, rewrite to include the topic.
- If the question is already self-contained, copy it verbatim.
- Preserve all critical terms: product names, error codes, dates, named entities.
- Do not introduce facts not present in the message or history.

Rules for sub_queries:
- For greeting or meta: return [].
- For doc_question: 1-4 focused retrieval queries; always include standalone_query first.
  Decompose only when the question contains multiple independent information needs.

reason: "greeting" | "meta" | "unchanged" | "contextualized" | "refined" |
  "decomposed — <why>" | "fallback"\
"""


class _PlannerOutput(BaseModel):
    intent: Literal['greeting', 'meta', 'doc_question'] = 'doc_question'
    standalone_query: str = ''
    sub_queries: list[str] = []
    reason: str = ''


# ── Pure helpers (imported by unit tests) ─────────────────────────────────────

_HISTORY_LIMIT = 6


def _bound_history(history: list[dict], limit: int = _HISTORY_LIMIT) -> list[dict]:
    """Return the `limit` most-recent messages."""
    if not history:
        return []
    return list(history[-limit:])


def _parse_planner_output(content: str, question: str) -> IntentPlanDetails:
    """Parse and validate a planner LLM response string.

    Raises ValueError on parsing/validation failure — caller should catch and fall back.
    """
    try:
        parsed = _PlannerOutput.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"planner output invalid: {exc}") from exc

    queries = [q.strip() for q in parsed.sub_queries if q.strip()][:4]
    standalone = parsed.standalone_query.strip() or question

    if parsed.intent == 'doc_question':
        if not queries:
            queries = [standalone]
        decomposed = len(queries) > 1 or parsed.reason.startswith("decomposed")
        rewritten = standalone.lower() != question.lower().strip()
    else:
        queries = []
        decomposed = False
        rewritten = False

    return IntentPlanDetails(
        intent=parsed.intent,
        standalone_query=standalone,
        queries=queries,
        decomposed=decomposed,
        rewritten=rewritten,
        reason=parsed.reason or parsed.intent,
    )


# ── Main planning function ────────────────────────────────────────────────────

def plan_details(
    question: str,
    history: list[dict] | None = None,
) -> IntentPlanDetails:
    """Intent-routing + query-planning LLM call.

    Falls back to {intent: doc_question, queries: [question]} on any failure.
    With no history the output is equivalent to the previous single-turn planner.
    """
    bounded = _bound_history(history or [])

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for m in bounded:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model=settings.planner_model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        plan = _parse_planner_output(content, question)
        log.info(
            "planner.done",
            intent=plan.intent,
            rewritten=plan.rewritten,
            reason=plan.reason,
            n_queries=len(plan.queries),
        )
        return plan

    except Exception as exc:
        log.warning("planner.fallback", error=str(exc))
        return IntentPlanDetails(
            intent='doc_question',
            standalone_query=question,
            queries=[question],
            decomposed=False,
            rewritten=False,
            reason="fallback",
        )


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _retrieve_from_queries(
    queries: list[str],
    document_ids: list[str] | None,
) -> list[Chunk]:
    seen_ids: set[str] = set()
    all_chunks: list[Chunk] = []

    for q in queries:
        for chunk in retrieve(q, document_ids=document_ids):
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                all_chunks.append(chunk)

    all_chunks.sort(key=lambda c: c.raw_vec_score, reverse=True)
    result = all_chunks[: settings.retrieval_top_k]
    log.info(
        "planner.retrieve_done",
        n_subqueries=len(queries),
        total_candidates=len(all_chunks),
        returned=len(result),
    )
    return result


def retrieve_with_plan(
    plan: IntentPlanDetails,
    document_ids: list[str] | None = None,
) -> list[Chunk]:
    """Retrieve chunks from a pre-computed plan. Returns [] for greeting/meta."""
    if plan.intent != 'doc_question':
        return []
    return _retrieve_from_queries(plan.queries, document_ids)


def retrieve_with_planning(
    question: str,
    document_ids: list[str] | None = None,
    history: list[dict] | None = None,
) -> list[Chunk]:
    """Convenience wrapper: plan + retrieve in one call. Used by eval harness."""
    plan = plan_details(question, history=history)
    return retrieve_with_plan(plan, document_ids)
