from pgvector.psycopg2 import register_vector
from psycopg2.extras import RealDictCursor

from app.retrieval.store import get_conn


def vector_search(
    query_embedding: list[float],
    document_ids: list[str] | None,
    top_k: int,
) -> list[dict]:
    """Cosine similarity search via pgvector <=> operator.

    Returns rows with score = 1 - cosine_distance, so higher is more similar.
    document_ids filtering is done in SQL to avoid fetching unnecessary rows.
    """
    with get_conn() as conn:
        register_vector(conn)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if document_ids:
                cur.execute(
                    """
                    SELECT id, document_id, filename, page, chunk_index, content,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    WHERE document_id = ANY(%s)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, document_ids, query_embedding, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT id, document_id, filename, page, chunk_index, content,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, top_k),
                )
            return [dict(r) for r in cur.fetchall()]
