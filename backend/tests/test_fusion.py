"""
Unit tests for fuse(). Pure function — no DB, no API calls.
"""

import pytest

from app.retrieval.fusion import fuse


def _vec(id_: str, score: float) -> dict:
    return {
        "id": id_,
        "document_id": "doc1",
        "filename": "test.pdf",
        "page": 1,
        "chunk_index": 0,
        "content": f"content {id_}",
        "score": score,
    }


def _fts(id_: str, score: float) -> dict:
    return {**_vec(id_, score), "page": None}


# ---------------------------------------------------------------------------
# Normalization correctness
# ---------------------------------------------------------------------------


def test_normalization_maps_min_to_0_and_max_to_1():
    # Three vector results — min and max must land exactly at 0 and 1.
    vec = [_vec("a", 0.9), _vec("b", 0.6), _vec("c", 0.3)]
    result = fuse(vec, [], top_k=3, vec_weight=1.0, fts_weight=0.0)

    by_id = {r["id"]: r["score"] for r in result}
    assert by_id["a"] == pytest.approx(1.0), "Max score should normalize to 1.0"
    assert by_id["c"] == pytest.approx(0.0), "Min score should normalize to 0.0"
    # Middle: (0.6 - 0.3) / (0.9 - 0.3) = 0.5
    assert by_id["b"] == pytest.approx(0.5)


def test_degenerate_arm_all_equal_scores_no_div_by_zero():
    # hi == lo → every chunk gets 1.0 (not a ZeroDivisionError).
    vec = [_vec("a", 0.5), _vec("b", 0.5), _vec("c", 0.5)]
    result = fuse(vec, [], top_k=3, vec_weight=1.0, fts_weight=0.0)

    for r in result:
        assert r["score"] == pytest.approx(1.0), f"id={r['id']} should have score 1.0"


# ---------------------------------------------------------------------------
# Single-arm chunks
# ---------------------------------------------------------------------------


def test_chunk_only_in_vector_arm_gets_zero_fts_contribution():
    # "only_vec" is in vector only. Score = vec_weight * vec_norm + fts_weight * 0.0
    vec = [_vec("only_vec", 0.8), _vec("both", 0.4)]
    fts = [_fts("both", 0.05)]

    result = fuse(vec, fts, top_k=3, vec_weight=0.7, fts_weight=0.3)
    by_id = {r["id"]: r for r in result}

    # "only_vec" is vec arm max → vec_norm=1.0; fts_norm=0.0
    # score = 0.7 * 1.0 + 0.3 * 0.0 = 0.7
    assert by_id["only_vec"]["score"] == pytest.approx(0.7)


def test_chunk_only_in_fts_arm_gets_zero_vec_contribution():
    # "only_fts" is in FTS only. Score = vec_weight * 0.0 + fts_weight * fts_norm
    vec = [_vec("both", 0.8)]
    fts = [_fts("only_fts", 0.1), _fts("both", 0.05)]

    result = fuse(vec, fts, top_k=3, vec_weight=0.7, fts_weight=0.3)
    by_id = {r["id"]: r for r in result}

    # "only_fts" is fts arm max → fts_norm=1.0; vec_norm=0.0
    # score = 0.7 * 0.0 + 0.3 * 1.0 = 0.3
    assert by_id["only_fts"]["score"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Weights actually change ordering
# ---------------------------------------------------------------------------


def test_weights_change_result_ordering():
    # "vec_winner": high vector, low FTS. "fts_winner": low vector, high FTS.
    vec = [_vec("vec_winner", 0.9), _vec("fts_winner", 0.2)]
    fts = [_fts("fts_winner", 0.9), _fts("vec_winner", 0.1)]

    # Vector-heavy weights → vec_winner ranks first
    result_vec = fuse(vec, fts, top_k=2, vec_weight=0.9, fts_weight=0.1)
    assert result_vec[0]["id"] == "vec_winner"

    # FTS-heavy weights → fts_winner ranks first
    result_fts = fuse(vec, fts, top_k=2, vec_weight=0.1, fts_weight=0.9)
    assert result_fts[0]["id"] == "fts_winner"


# ---------------------------------------------------------------------------
# top_k truncation
# ---------------------------------------------------------------------------


def test_top_k_returns_exactly_k_results():
    vec = [_vec(str(i), i / 10) for i in range(10)]
    result = fuse(vec, [], top_k=4, vec_weight=1.0, fts_weight=0.0)
    assert len(result) == 4


def test_results_sorted_highest_fused_score_first():
    vec = [_vec("low", 0.2), _vec("high", 0.8), _vec("mid", 0.5)]
    result = fuse(vec, [], top_k=3, vec_weight=1.0, fts_weight=0.0)
    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# raw_vec_score propagation
# ---------------------------------------------------------------------------


def test_raw_vec_score_carried_forward_for_vector_chunks():
    vec = [_vec("a", 0.75)]
    result = fuse(vec, [], top_k=1, vec_weight=1.0, fts_weight=0.0)
    assert result[0]["raw_vec_score"] == pytest.approx(0.75)


def test_raw_vec_score_is_zero_for_fts_only_chunks():
    fts = [_fts("fts_only", 0.05)]
    result = fuse([], fts, top_k=1, vec_weight=0.0, fts_weight=1.0)
    assert result[0]["raw_vec_score"] == pytest.approx(0.0)
