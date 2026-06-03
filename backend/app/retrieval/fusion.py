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

    What min-max preserves: RELATIVE SPACING within each candidate set, not absolute
    magnitude. The best candidate in each arm always normalizes to 1.0, regardless of
    its raw score — a query whose top vector hit is cosine=0.98 and one whose top hit
    is cosine=0.40 both produce a winner at 1.0. The consequence: fused scores are not
    comparable across separate retrieve() calls (e.g., in the planner), so cross-call
    merging must sort by raw_vec_score, not fused score.

    Known weakness: a near-uniform arm (e.g., 20 FTS hits with ts_rank 0.050–0.051)
    gets stretched to the full [0,1] range, amplifying noise into full-weight signal.
    The hi==lo→1.0 rule has the same effect for a degenerate arm. RRF doesn't have
    this failure mode. Accepted trade-off: the config knobs remain meaningful and the
    relative ordering within each arm is preserved.

    Why weighted over RRF: fusion_vector_weight/fusion_fts_weight in config are
    meaningful tuning knobs — 0.7/0.3 reflects that semantic similarity carries more
    signal than keyword overlap for most queries. RRF collapses all scores to rank
    position, making those config values meaningless.

    Chunks that appear in only one arm get 0.0 for the missing arm — the arm's weight
    still applies, so a strong vector match with no FTS signal gets vec_weight * norm.

    raw_vec_score: the original (pre-normalization) cosine similarity from the vector
    arm, carried through for the confidence guardrail in generation. Chunks that only
    appeared in the FTS arm get raw_vec_score=0.0.
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
        {
            **row_by_id[id_],
            "score": round(fused_scores[id_], 4),
            "raw_vec_score": vec_scores.get(id_, 0.0),
        }
        for id_ in top_ids
    ]
