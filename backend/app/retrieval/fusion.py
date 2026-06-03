def fuse(
    vector_results: list[dict],
    fts_results: list[dict],
    top_k: int,
    vec_weight: float,
    fts_weight: float,
) -> list[dict]:
    """Merge vector and FTS results with weighted fusion after per-arm normalization.

    Why normalize before fusing: pgvector cosine similarity and PostgreSQL ts_rank
    live on incompatible scales (cosine ~[0,1], ts_rank typically ~[0.01,0.1]).
    Adding raw scores would collapse the FTS signal. Min-max normalization maps each
    arm independently to [0,1] before applying weights.

    Why weighted over RRF: fusion_vector_weight/fusion_fts_weight in config are
    meaningful tuning knobs — 0.7/0.3 reflects the intuition that semantic similarity
    carries more signal than keyword overlap for most queries. RRF collapses all scores
    to rank position, making those config values meaningless. Score magnitude matters
    here: a chunk at cosine=0.98 should score higher than one at 0.62, not just "rank 1
    vs rank 2".

    Edge case: if all scores in an arm are identical (max == min), every chunk in that
    arm gets normalized score = 1.0 (all equally relevant; penalizing them would be wrong).

    Chunks that appear in only one arm get 0.0 for the missing arm — the arm's weight
    still applies, so a strong vector match with no FTS signal gets vec_weight * norm.
    """
    vec_scores: dict[str, float] = {r["id"]: float(r["score"]) for r in vector_results}
    fts_scores: dict[str, float] = {r["id"]: float(r["score"]) for r in fts_results}

    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        if hi == lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    vec_norm = _normalize(vec_scores)
    fts_norm = _normalize(fts_scores)

    all_ids = set(vec_scores) | set(fts_scores)
    fused_scores = {
        id_: vec_weight * vec_norm.get(id_, 0.0) + fts_weight * fts_norm.get(id_, 0.0)
        for id_ in all_ids
    }

    # Row data: prefer the vector result row (always has an embedding match);
    # fall back to FTS row for chunks that only appeared in the FTS arm.
    row_by_id: dict[str, dict] = {r["id"]: r for r in fts_results}
    row_by_id.update({r["id"]: r for r in vector_results})

    top_ids = sorted(fused_scores, key=lambda id_: fused_scores[id_], reverse=True)[:top_k]

    return [
        {**row_by_id[id_], "score": round(fused_scores[id_], 4)}
        for id_ in top_ids
    ]
