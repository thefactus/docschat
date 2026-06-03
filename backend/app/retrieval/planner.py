import json

import structlog
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.retrieval.pipeline import Chunk, retrieve

log = structlog.get_logger()

_SYSTEM_PROMPT = """\
You are a query planning component for a document-grounded RAG system.
Your job is to transform the user question into retrieval queries — not to answer it.

Rules:
- Return JSON only.
- Preserve critical terms exactly: product names, clause names, version numbers,
  error codes, acronyms, dates, and named entities.
- Do not introduce facts, entities, or constraints not present or clearly implied
  in the user question.
- If the question is simple or focused, return the original question as the only query.
- If the question contains multiple independent intents, decompose into at most 4
  focused retrieval queries. Each sub-query should be independently searchable.
- Always include the original user question in "original_query".
- Keep retrieval queries short and searchable.
- Include a brief "reason" field: "unchanged", "refined", or "decomposed — <why>".

Output schema (JSON only, no markdown):
{
  "original_query": "string",
  "queries": ["string"],
  "reason": "string"
}\
"""


class _PlannerOutput(BaseModel):
    original_query: str
    queries: list[str]
    reason: str


def _plan(question: str) -> list[str]:
    """Call the planner LLM to decompose or refine the question.

    Returns a list of retrieval query strings (max 4).
    Falls back to [question] on any failure — LLM error, invalid JSON, schema
    mismatch, or empty queries list. Fallback is logged for observability.
    """
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.planner_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = _PlannerOutput.model_validate(json.loads(content))
        # Enforce max 4 in code regardless of what the prompt says
        queries = [q.strip() for q in parsed.queries if q.strip()][:4]
        if not queries:
            queries = [question]
        log.info("planner.done", reason=parsed.reason, n_queries=len(queries))
        return queries
    except (ValidationError, json.JSONDecodeError, Exception) as exc:
        log.warning("planner.fallback", error=str(exc))
        return [question]


def retrieve_with_planning(
    question: str,
    document_ids: list[str] | None = None,
) -> list[Chunk]:
    """Decompose the question into sub-queries, retrieve per sub-query, merge and dedup.

    Merge strategy: dedup by chunk.id (a chunk retrieved by multiple sub-queries
    keeps its first occurrence). Re-sort by raw_vec_score (an absolute cosine
    similarity) rather than the normalized fused score — fused scores are only
    comparable within a single retrieve() call, not across calls with different
    sub-queries. Return at most retrieval_top_k chunks.
    """
    queries = _plan(question)

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
        question=question[:80],
        n_subqueries=len(queries),
        total_candidates=len(all_chunks),
        returned=len(result),
    )
    return result
