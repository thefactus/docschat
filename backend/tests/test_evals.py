"""
Unit tests for eval scoring logic.

All tested functions are pure — no DB, no OpenAI calls, no Docker required.
Run: python -m pytest tests/test_evals.py

Importing evals.run_evals triggers its module-level app imports (app.config,
app.retrieval.*, app.generation.*). In the Docker environment all packages are
installed; none of those imports opens a DB connection or API client at import
time, so this file is safe to run with --no-deps (no live DB, no key).
"""
import sys
from pathlib import Path

# Ensure both backend/ (for app.*) and the test directory are importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.run_evals import (
    ItemResult,
    Source,
    _contains_lexical_fact,
    _score,
    compute_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    type: str = "answerable",
    answer: str = "The answer is here.",
    tokens_used: int = 100,
    refused: bool = False,
    sources: list | None = None,
) -> ItemResult:
    return ItemResult(
        id="test-item",
        type=type,
        question="Test question?",
        answer=answer,
        tokens_used=tokens_used,
        refused=refused,
        sources=sources or [],
    )


def _make_source(filename: str = "doc.pdf", page: int | None = 1, score: float = 0.7) -> Source:
    return Source(filename=filename, page=page, score=score, content="some content")


# ---------------------------------------------------------------------------
# _contains_lexical_fact
# ---------------------------------------------------------------------------

class TestContainsLexicalFact:
    def test_direct_substring_match(self):
        assert _contains_lexical_fact("Employees get 25 days of leave.", "25")

    def test_case_insensitive(self):
        assert _contains_lexical_fact("pgvector is used for embeddings.", "pgVector")

    def test_missing_returns_false(self):
        assert not _contains_lexical_fact("We use Elasticsearch.", "BM25")

    def test_compact_strips_spaces(self):
        # "Lang Graph" compacts to "langgraph" which contains compact("LangGraph")
        assert _contains_lexical_fact("We use Lang Graph for orchestration.", "LangGraph")

    def test_number_alias_digit_in_answer(self):
        assert _contains_lexical_fact("Employees receive 25 days annually.", "25")

    def test_number_alias_word_in_answer(self):
        # "twenty-five" → compact "twentyfive"; alias for "25" is "twentyfive"
        assert _contains_lexical_fact("Employees receive twenty-five days annually.", "25")

    def test_number_alias_word_no_hyphen(self):
        # "twenty five" → compact "twentyfive" → alias matches
        assert _contains_lexical_fact("Employees receive twenty five days annually.", "25")

    def test_number_alias_does_not_fire_for_unlisted_numbers(self):
        # "99" has no alias entry
        assert not _contains_lexical_fact("ninety-nine problems.", "99")

    def test_empty_answer_returns_false(self):
        assert not _contains_lexical_fact("", "pgvector")

    def test_empty_expected_matches_anything(self):
        # Empty string is always a substring
        assert _contains_lexical_fact("any answer", "")


# ---------------------------------------------------------------------------
# _score — partial-credit answer_match
# ---------------------------------------------------------------------------

class TestScorePartialCredit:
    def test_all_keywords_matched_gives_1(self):
        item = {
            "type": "answerable",
            "expect_documents": [],
            "expect_answer_contains": ["pgvector", "Pinecone"],
        }
        result = _make_result(answer="We support pgvector and Pinecone.")
        _score(result, item)
        assert result.answer_hit == 1.0
        assert not result.failures

    def test_half_keywords_matched_gives_0_5(self):
        item = {
            "type": "answerable",
            "expect_documents": [],
            "expect_answer_contains": ["BM25", "Elasticsearch"],
        }
        result = _make_result(answer="Elasticsearch and Solr are mentioned.")
        _score(result, item)
        assert result.answer_hit == 0.5
        assert any("BM25" in f for f in result.failures)

    def test_two_of_four_keywords_gives_0_5(self):
        item = {
            "type": "answerable",
            "expect_documents": [],
            "expect_answer_contains": ["pgvector", "Pinecone", "Weaviate", "Chroma"],
        }
        # Only pgvector and Pinecone are in the answer → 2/4 = 0.5
        result = _make_result(answer="We use pgvector and Pinecone.")
        _score(result, item)
        assert result.answer_hit == 0.5

    def test_no_keywords_matched_gives_0(self):
        item = {
            "type": "answerable",
            "expect_documents": [],
            "expect_answer_contains": ["LangGraph", "AutoGen"],
        }
        result = _make_result(answer="I don't have enough information.")
        _score(result, item)
        assert result.answer_hit == 0.0

    def test_no_expect_strings_leaves_answer_hit_none(self):
        item = {"type": "answerable", "expect_documents": []}
        result = _make_result(answer="Any answer.")
        _score(result, item)
        assert result.answer_hit is None

    def test_failure_note_shows_partial_count(self):
        item = {
            "type": "answerable",
            "expect_documents": [],
            "expect_answer_contains": ["BM25", "Elasticsearch"],
        }
        result = _make_result(answer="Elasticsearch only.")
        _score(result, item)
        assert any("1/2" in f for f in result.failures)


# ---------------------------------------------------------------------------
# _score — refusal classification
# ---------------------------------------------------------------------------

class TestScoreRefusal:
    def test_refuse_item_correctly_refused_has_no_failure(self):
        item = {"type": "refuse"}
        result = _make_result(type="refuse", tokens_used=0, refused=True)
        _score(result, item)
        assert not result.failures

    def test_refuse_item_incorrectly_answered_flags_failure(self):
        item = {"type": "refuse"}
        result = _make_result(
            type="refuse", tokens_used=50, refused=False,
            answer="The boiling point of helium is -269°C."
        )
        _score(result, item)
        assert any("unexpected_answer" in f for f in result.failures)

    def test_answerable_item_unexpectedly_refused_flags_failure(self):
        item = {
            "type": "answerable",
            "expect_documents": [],
            "expect_answer_contains": ["25"],
        }
        result = _make_result(
            answer="I don't have enough information.", tokens_used=0, refused=True
        )
        _score(result, item)
        assert any("unexpected_refusal" in f for f in result.failures)


# ---------------------------------------------------------------------------
# _score — retrieval hit detection
# ---------------------------------------------------------------------------

class TestScoreRetrievalHit:
    def test_expected_document_present_is_hit(self):
        item = {
            "type": "answerable",
            "expect_documents": ["policy.pdf"],
            "expect_answer_contains": [],
        }
        result = _make_result(sources=[_make_source("policy.pdf")])
        _score(result, item)
        assert result.retrieval_hit is True
        assert not any("retrieval_miss" in f for f in result.failures)

    def test_expected_document_absent_is_miss(self):
        item = {
            "type": "answerable",
            "expect_documents": ["policy.pdf"],
            "expect_answer_contains": [],
        }
        result = _make_result(sources=[_make_source("other.pdf")])
        _score(result, item)
        assert result.retrieval_hit is False
        assert any("retrieval_miss" in f for f in result.failures)

    def test_all_expected_documents_must_be_present(self):
        item = {
            "type": "answerable",
            "expect_documents": ["a.pdf", "b.pdf"],
            "expect_answer_contains": [],
        }
        result = _make_result(sources=[_make_source("a.pdf")])
        _score(result, item)
        assert result.retrieval_hit is False


# ---------------------------------------------------------------------------
# _score — scoping leak detection
# ---------------------------------------------------------------------------

class TestScoreScoping:
    def test_all_sources_in_scope_passes(self):
        item = {
            "type": "answerable",
            "scope": ["doc1.pdf"],
            "expect_documents": ["doc1.pdf"],
            "expect_answer_contains": [],
        }
        result = _make_result(sources=[_make_source("doc1.pdf")])
        _score(result, item)
        assert result.scoping_ok is True

    def test_out_of_scope_source_detected(self):
        item = {
            "type": "answerable",
            "scope": ["doc1.pdf"],
            "expect_documents": ["doc1.pdf"],
            "expect_answer_contains": [],
        }
        result = _make_result(
            sources=[_make_source("doc1.pdf"), _make_source("doc2.pdf")]
        )
        _score(result, item)
        assert result.scoping_ok is False
        assert any("scope_leak" in f for f in result.failures)

    def test_no_scope_key_leaves_scoping_ok_none(self):
        item = {"type": "answerable", "expect_documents": [], "expect_answer_contains": []}
        result = _make_result()
        _score(result, item)
        assert result.scoping_ok is None


# ---------------------------------------------------------------------------
# compute_metrics — partial-credit aggregation
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def _make_answerable(self, answer_hit: float | None = 1.0) -> ItemResult:
        r = _make_result(type="answerable")
        r.retrieval_hit = True
        r.answer_hit = answer_hit
        return r

    def _make_refuse(self, refused: bool) -> ItemResult:
        return _make_result(type="refuse", refused=refused, tokens_used=0 if refused else 50)

    def test_answer_match_mean_partial_credit(self):
        # 1.0 + 0.5 + 1.0 = 2.5 / 3 ≈ 0.8333
        results = [
            self._make_answerable(1.0),
            self._make_answerable(0.5),
            self._make_answerable(1.0),
        ]
        m = compute_metrics(results)
        assert abs(m["answer_match"]["mean"] - round(2.5 / 3, 4)) < 1e-4
        assert m["answer_match"]["n"] == 3

    def test_answer_match_excludes_none_items(self):
        # Items with no expect_answer_contains have answer_hit=None; excluded from mean
        results = [
            self._make_answerable(1.0),
            self._make_answerable(None),
        ]
        m = compute_metrics(results)
        assert m["answer_match"]["n"] == 1
        assert m["answer_match"]["mean"] == 1.0

    def test_refusal_accuracy_counts(self):
        results = [
            self._make_refuse(True),     # correctly refused
            self._make_refuse(False),    # incorrectly answered
            self._make_answerable(1.0),  # answerable, not refused
        ]
        m = compute_metrics(results)
        ra = m["refusal_accuracy"]
        assert ra["refuse_correct"] == 1
        assert ra["refuse_total"] == 2
        assert ra["answerable_not_refused"] == 1
        assert ra["answerable_total"] == 1

    def test_retrieval_hit_rate(self):
        r1 = self._make_answerable()
        r1.retrieval_hit = True
        r2 = self._make_answerable()
        r2.retrieval_hit = False
        m = compute_metrics([r1, r2])
        assert m["retrieval_hit_rate"]["hits"] == 1
        assert m["retrieval_hit_rate"]["n"] == 2
        assert m["retrieval_hit_rate"]["rate"] == 0.5

    def test_empty_results_no_crash(self):
        m = compute_metrics([])
        assert m["answer_match"]["mean"] == 0.0
        assert m["answer_match"]["n"] == 0
        assert m["retrieval_hit_rate"]["rate"] == 0.0
