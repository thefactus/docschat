import psycopg2
import structlog
from pgvector.psycopg2 import register_vector
from psycopg2.extras import RealDictCursor, execute_values

from app.config import settings
from app.models import DocumentInfo

log = structlog.get_logger()


def get_conn():
    return psycopg2.connect(settings.database_url)


def init_db() -> None:
    """Create pgvector extension and chunks table if they don't exist."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id          TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    filename    TEXT NOT NULL,
                    page        INTEGER,
                    chunk_index INTEGER NOT NULL,
                    content     TEXT NOT NULL,
                    embedding   vector(1536),
                    fts         tsvector GENERATED ALWAYS AS
                                    (to_tsvector('english', content)) STORED
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks USING hnsw (embedding vector_cosine_ops)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_fts_idx
                ON chunks USING gin (fts)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_document_id_idx
                ON chunks (document_id)
            """)
        conn.commit()
    log.info("db.init_done")


def store_chunks(rows: list[dict]) -> None:
    """Insert chunk rows into PostgreSQL. register_vector is called here (not in get_conn)
    to avoid a chicken-and-egg issue: init_db creates the extension first."""
    with get_conn() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # No deduplication: each ingest call assigns a fresh uuid4 as id, so the same
            # file uploaded twice creates two separate document_ids with independent chunks.
            execute_values(
                cur,
                """
                INSERT INTO chunks
                    (id, document_id, filename, page, chunk_index, content, embedding)
                VALUES %s
                """,
                [
                    (
                        r["id"],
                        r["document_id"],
                        r["filename"],
                        r["page"],
                        r["chunk_index"],
                        r["content"],
                        r["embedding"],
                    )
                    for r in rows
                ],
            )
        conn.commit()


def list_documents() -> list[DocumentInfo]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT document_id, filename, COUNT(*) AS chunks
                FROM chunks
                GROUP BY document_id, filename
                ORDER BY filename
            """)
            rows = cur.fetchall()
    return [
        DocumentInfo(document_id=r["document_id"], filename=r["filename"], chunks=r["chunks"])
        for r in rows
    ]
