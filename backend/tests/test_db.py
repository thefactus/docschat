"""
Token-free integration tests for the SQL retrieval layer.

Spins up pgvector/pg16 via testcontainers — requires Docker.
No OpenAI calls: embeddings are deterministic random vectors.
We are testing SQL wiring (::vector cast, numnode guard, FTS ranking),
not model behaviour.
"""
import random
import uuid

import pytest
from _pytest.monkeypatch import MonkeyPatch
from testcontainers.postgres import PostgresContainer

PGVECTOR_IMAGE = "pgvector/pgvector:pg16"
EMBEDDING_DIM = 1536


def _fake_embedding(seed: int = 0) -> list[float]:
    """Deterministic random vector. Normalisation not required for pgvector."""
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def module_monkeypatch() -> MonkeyPatch:
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def db_url(module_monkeypatch: MonkeyPatch) -> str:
    """Spin up a pgvector container, patch settings, and init the schema."""
    with PostgresContainer(image=PGVECTOR_IMAGE) as pg:
        url = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")

        from app.config import settings
        module_monkeypatch.setattr(settings, "database_url", url)

        from app.retrieval.store import init_db
        init_db()

        yield url


@pytest.fixture(scope="module")
def seeded_chunks(db_url: str) -> list[dict]:
    """Insert two known chunks; return the rows for assertion use."""
    from app.retrieval.store import store_chunks

    rows = [
        {
            "id": str(uuid.uuid4()),
            "document_id": "doc-policy",
            "filename": "policy.pdf",
            "page": 1,
            "chunk_index": 0,
            "content": "Employees are entitled to 20 days of annual leave per year.",
            "embedding": _fake_embedding(seed=1),
        },
        {
            "id": str(uuid.uuid4()),
            "document_id": "doc-policy",
            "filename": "policy.pdf",
            "page": 2,
            "chunk_index": 1,
            "content": "Sick leave of up to 10 days per year is provided without a doctor note.",
            "embedding": _fake_embedding(seed=2),
        },
    ]
    store_chunks(rows)
    return rows


# ---------------------------------------------------------------------------
# vector_search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    def test_returns_seeded_chunks(self, seeded_chunks: list[dict], db_url: str) -> None:
        from app.retrieval.vector import vector_search

        results = vector_search(_fake_embedding(seed=99), document_ids=None, top_k=10)
        assert len(results) >= 2

    def test_exact_vector_scores_one(self, seeded_chunks: list[dict], db_url: str) -> None:
        """Querying with a chunk's own embedding → cosine similarity ≈ 1.0."""
        from app.retrieval.vector import vector_search

        results = vector_search(
            _fake_embedding(seed=1),
            document_ids=["doc-policy"],
            top_k=10,
        )
        by_index = {r["chunk_index"]: r["score"] for r in results}
        assert by_index[0] == pytest.approx(1.0, abs=1e-4)

    def test_document_id_filter_excludes_others(
        self, seeded_chunks: list[dict], db_url: str
    ) -> None:
        from app.retrieval.vector import vector_search

        results = vector_search(_fake_embedding(), document_ids=["nonexistent-doc"], top_k=10)
        assert results == []

    def test_top_k_limits_results(self, seeded_chunks: list[dict], db_url: str) -> None:
        from app.retrieval.vector import vector_search

        results = vector_search(_fake_embedding(), document_ids=None, top_k=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# fts_search
# ---------------------------------------------------------------------------


class TestFtsSearch:
    def test_keyword_match_returns_correct_chunk(
        self, seeded_chunks: list[dict], db_url: str
    ) -> None:
        from app.retrieval.fts import fts_search

        results = fts_search("annual leave", document_ids=None, top_k=10)
        assert len(results) >= 1
        assert any("annual leave" in r["content"] for r in results)

    def test_stopword_only_query_returns_empty(
        self, seeded_chunks: list[dict], db_url: str
    ) -> None:
        """numnode guard: all-stopword tsquery must not raise, must return []."""
        from app.retrieval.fts import fts_search

        # "the" is an English stopword; websearch_to_tsquery returns empty tsquery.
        assert fts_search("the", document_ids=None, top_k=10) == []

    def test_document_id_filter_excludes_others(
        self, seeded_chunks: list[dict], db_url: str
    ) -> None:
        from app.retrieval.fts import fts_search

        assert fts_search("leave", document_ids=["nonexistent-doc"], top_k=10) == []


# ---------------------------------------------------------------------------
# store_chunks round-trip
# ---------------------------------------------------------------------------


class TestStoreRoundtrip:
    def test_inserted_chunk_retrieved_by_own_vector(self, db_url: str) -> None:
        """store_chunks writes correctly enough that vector_search can find it."""
        from app.retrieval.store import store_chunks
        from app.retrieval.vector import vector_search

        chunk_id = str(uuid.uuid4())
        vec = _fake_embedding(seed=77)
        store_chunks(
            [
                {
                    "id": chunk_id,
                    "document_id": "doc-rt",
                    "filename": "rt.txt",
                    "page": None,
                    "chunk_index": 0,
                    "content": "round trip verification content",
                    "embedding": vec,
                }
            ]
        )

        results = vector_search(vec, document_ids=["doc-rt"], top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == chunk_id
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)
