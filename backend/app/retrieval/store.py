import psycopg2
import structlog
from psycopg2.extras import RealDictCursor

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
                    fts         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
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
