"""Async SSE pipeline for POST /query/stream.

Runs the same retrieval + generation pipeline as /query but as an async generator
that emits one SSE event per stage. All pipeline primitives (planner, retrieve,
fusion, guardrail, generate) are reused — only the orchestration is new.

/query is never touched by this module.
"""
import asyncio
import json
import time

import structlog

from app.config import settings

log = structlog.get_logger()

_REFUSAL = (
    "I don't have enough information in the uploaded documents"
    " to answer this question."
)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def query_stream(question: str, document_ids: list[str] | None):
    """Async generator that yields SSE-formatted strings for /query/stream.

    Emits events: planner, retrieval (one per sub-query), fusion, guardrail,
    generating, token (one per LLM delta), done.  On any exception: error.
    """
    t0 = time.perf_counter()

    try:
        # ── Planner ────────────────────────────────────────────────────────
        from app.retrieval.planner import plan_details

        plan = await asyncio.to_thread(plan_details, question)
        yield _sse(
            "planner",
            {
                "decomposed": plan.decomposed,
                "sub_queries": plan.queries,
                "reason": plan.reason,
            },
        )

        # ── Retrieval (per sub-query) ───────────────────────────────────────
        from app.retrieval.pipeline import retrieve_detailed

        seen_ids: set[str] = set()
        all_chunks: list = []

        for q in plan.queries:
            detail = await asyncio.to_thread(retrieve_detailed, q, document_ids)
            yield _sse(
                "retrieval",
                {
                    "sub_query": q,
                    "vector_hits": detail.vec_hits,
                    "fts_hits": detail.fts_hits,
                    "fused": detail.fused_count,
                },
            )
            for chunk in detail.chunks:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    all_chunks.append(chunk)

        # Merge + trim identical to retrieve_with_planning
        all_chunks.sort(key=lambda c: c.raw_vec_score, reverse=True)
        final_chunks = all_chunks[: settings.retrieval_top_k]

        # ── Fusion summary ──────────────────────────────────────────────────
        top_sources = [
            {
                "filename": c.filename,
                "page": c.page,
                "score": round(c.raw_vec_score, 4),
            }
            for c in final_chunks[:5]
        ]
        yield _sse(
            "fusion",
            {
                "weights": f"{settings.fusion_vector_weight}/{settings.fusion_fts_weight}",
                "top": top_sources,
            },
        )

        # ── Guardrail ───────────────────────────────────────────────────────
        max_score = max((c.raw_vec_score for c in final_chunks), default=0.0)
        proceed = bool(final_chunks) and max_score >= settings.low_confidence_threshold
        yield _sse(
            "guardrail",
            {
                "max_score": round(max_score, 4),
                "threshold": settings.low_confidence_threshold,
                "decision": "proceed" if proceed else "refuse",
            },
        )

        if not proceed:
            yield _sse(
                "done",
                {
                    "answer": _REFUSAL,
                    "sources": [],
                    "tokens_used": 0,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                },
            )
            return

        # ── Generating ──────────────────────────────────────────────────────
        yield _sse("generating", {"chunks_used": len(final_chunks)})

        from app.generation.openai import generate_stream

        answer_parts: list[str] = []
        tokens_used = 0

        async for evt in generate_stream(question, final_chunks):
            if evt["type"] == "token":
                answer_parts.append(evt["text"])
                yield _sse("token", {"text": evt["text"]})
            elif evt["type"] == "done":
                tokens_used = evt["tokens_used"]

        answer = "".join(answer_parts)
        sources = [
            c.to_source().model_dump()
            for c in final_chunks
            if c.raw_vec_score > 0.0
        ]

        yield _sse(
            "done",
            {
                "answer": answer,
                "sources": sources,
                "tokens_used": tokens_used,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        )

    except Exception as exc:
        log.exception("stream.error", error=str(exc))
        yield _sse("error", {"message": str(exc)})
