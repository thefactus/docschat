from dataclasses import dataclass

import structlog
from openai import OpenAI

from app.config import settings
from app.models import ChunkSource

log = structlog.get_logger()


@dataclass
class Chunk:
    id: str
    document_id: str
    filename: str
    page: int | None
    chunk_index: int
    content: str
    score: float
    raw_vec_score: float = 0.0  # pre-norm cosine similarity; generation confidence guardrail

    def to_source(self) -> ChunkSource:
        # Expose raw_vec_score as the user-facing score: it is an absolute cosine
        # similarity, comparable across planner sub-query calls, and matches the
        # sort order applied in retrieve_with_planning. The per-call fused score is
        # only comparable within a single retrieve() call and would disagree with
        # the ordering, producing a visibly unsorted sources panel.
        return ChunkSource(
            document_id=self.document_id,
            filename=self.filename,
            page=self.page,
            chunk_index=self.chunk_index,
            score=round(self.raw_vec_score, 4),
        )


def retrieve(question: str, document_ids: list[str] | None = None) -> list[Chunk]:
    """Embed the question, run both retrieval arms, fuse, return top-k Chunk objects.

    Query planning is intentionally absent here — retrieve() is a single-query
    retrieval primitive. The planner (a separate layer) calls retrieve() once per
    subquery and merges results before returning to the caller.
    """
    from app.retrieval.fts import fts_search
    from app.retrieval.fusion import fuse
    from app.retrieval.vector import vector_search

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.embeddings.create(model=settings.embedding_model, input=[question])
    query_embedding = sorted(resp.data, key=lambda x: x.index)[0].embedding

    # Fetch more than top_k from each arm so fusion has enough candidates to
    # promote chunks that score well in both arms above single-arm winners.
    fetch_k = max(settings.retrieval_top_k * 3, 20)

    vec_results = vector_search(query_embedding, document_ids, fetch_k)
    fts_results = fts_search(question, document_ids, fetch_k)

    log.debug(
        "retrieve.arms",
        question=question[:80],
        vec_hits=len(vec_results),
        fts_hits=len(fts_results),
    )

    fused = fuse(
        vec_results,
        fts_results,
        settings.retrieval_top_k,
        settings.fusion_vector_weight,
        settings.fusion_fts_weight,
    )

    return [
        Chunk(
            id=r["id"],
            document_id=r["document_id"],
            filename=r["filename"],
            page=r["page"],
            chunk_index=r["chunk_index"],
            content=r["content"],
            score=r["score"],
            raw_vec_score=r["raw_vec_score"],
        )
        for r in fused
    ]
