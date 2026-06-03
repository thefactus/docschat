from dataclasses import dataclass

import structlog
from openai import OpenAI

from app.config import settings

log = structlog.get_logger()

# Chunks whose best raw cosine similarity (pre-normalization) is below this
# threshold are too semantically distant to ground a reliable answer.
# Checked against raw_vec_score, not the normalized fused score — the fused
# score always peaks near 1.0 after normalization and is useless as a gate.
_LOW_CONFIDENCE_THRESHOLD = 0.30

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


async def generate(question: str, chunks: list) -> GenerationResult:
    """Build a grounded prompt from retrieved chunks and call the OpenAI chat model.

    chunks is list[Chunk] from retrieval/pipeline.py — accessed duck-typed to
    avoid a circular import between generation and retrieval packages.
    """
    max_raw = max((c.raw_vec_score for c in chunks), default=0.0)

    if max_raw < _LOW_CONFIDENCE_THRESHOLD:
        log.info(
            "generation.low_confidence",
            max_raw_vec_score=round(max_raw, 3),
            threshold=_LOW_CONFIDENCE_THRESHOLD,
        )
        return GenerationResult(
            answer="I don't have enough information in the uploaded documents to answer this question.",
            tokens_used=0,
        )

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        page_ref = f", p.{chunk.page}" if chunk.page is not None else ""
        context_parts.append(f"[{i}] {chunk.filename}{page_ref}:\n{chunk.content}")
    context = "\n\n---\n\n".join(context_parts)

    user_message = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.generation_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
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
