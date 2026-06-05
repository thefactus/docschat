from dataclasses import dataclass

import structlog
from openai import AsyncOpenAI, OpenAI

from app.config import settings

log = structlog.get_logger()

_HISTORY_LIMIT = 6

_SYSTEM_PROMPT = """\
You are a document assistant. Answer questions using ONLY the document excerpts provided.

Rules:
- Base your answer exclusively on the provided excerpts. Do not use outside knowledge.
- Cite sources inline using [filename, p.N] or [filename] when no page number is available.
- If the excerpts do not contain enough information, respond exactly:
  "I don't have enough information in the uploaded documents to answer this question."
- Do not speculate, infer beyond what is stated, or mix document content with general knowledge.
- Be concise and accurate.\
"""


@dataclass
class GenerationResult:
    answer: str
    tokens_used: int


def _build_messages(
    question: str,
    chunks: list,
    history: list[dict],
) -> list[dict]:
    """Assemble the messages array: system → bounded history → grounded user turn."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        page_ref = f", p.{chunk.page}" if chunk.page is not None else ""
        context_parts.append(f"[{i}] {chunk.filename}{page_ref}:\n{chunk.content}")
    context = "\n\n---\n\n".join(context_parts)
    user_message = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"

    bounded = history[-_HISTORY_LIMIT:] if len(history) > _HISTORY_LIMIT else history
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for m in bounded:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


async def generate(
    question: str,
    chunks: list,
    history: list[dict] | None = None,
) -> GenerationResult:
    """Build a grounded prompt from retrieved chunks and call the OpenAI chat model."""
    max_raw = max((c.raw_vec_score for c in chunks), default=0.0)

    if max_raw < settings.low_confidence_threshold:
        log.info(
            "generation.low_confidence",
            max_raw_vec_score=round(max_raw, 3),
            threshold=settings.low_confidence_threshold,
        )
        return GenerationResult(
            answer=(
                "I don't have enough information in the uploaded documents"
                " to answer this question."
            ),
            tokens_used=0,
        )

    messages = _build_messages(question, chunks, history or [])

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.generation_model,
        messages=messages,
        temperature=0.0,
    )

    answer = response.choices[0].message.content or ""
    tokens_used = response.usage.total_tokens if response.usage else 0

    log.info(
        "generation.done",
        tokens_used=tokens_used,
        max_raw_vec_score=round(max_raw, 3),
        chunks_used=len(chunks),
    )

    return GenerationResult(answer=answer, tokens_used=tokens_used)


async def generate_stream(
    question: str,
    chunks: list,
    history: list[dict] | None = None,
):
    """Streaming generation — async generator yielding token dicts then a done dict.

    Yields: {"type": "token", "text": str}
            {"type": "done", "tokens_used": int}
    """
    messages = _build_messages(question, chunks, history or [])

    async_client = AsyncOpenAI(api_key=settings.openai_api_key)
    tokens_used = 0

    stream = await async_client.chat.completions.create(
        model=settings.generation_model,
        messages=messages,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk_resp in stream:
        if chunk_resp.choices and chunk_resp.choices[0].delta.content:
            yield {"type": "token", "text": chunk_resp.choices[0].delta.content}
        if chunk_resp.usage:
            tokens_used = chunk_resp.usage.total_tokens

    yield {"type": "done", "tokens_used": tokens_used}
