from psycopg2.extras import RealDictCursor

from app.retrieval.store import get_conn


def fts_search(
    query: str,
    document_ids: list[str] | None,
    top_k: int,
) -> list[dict]:
    """Full-text search using PostgreSQL tsvector + websearch_to_tsquery.

    websearch_to_tsquery is used over plainto_tsquery because it handles boolean
    operators (AND/OR/NOT) and phrase search, which matter for technical documents
    with exact clause names, product names, and acronyms.

    An empty tsquery (all stop-words, empty string, etc.) is detected via
    tsquery_numnode before running the main query; returns [] in that case.
    document_ids filtering is done in SQL.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Guard: websearch_to_tsquery returns an empty tsquery for all-stopword
            # or empty input. Running fts @@ empty_tsquery raises a PostgreSQL error.
            cur.execute(
                "SELECT tsquery_numnode(websearch_to_tsquery('english', %s)) AS n",
                (query,),
            )
            if cur.fetchone()["n"] == 0:
                return []

            if document_ids:
                cur.execute(
                    """
                    SELECT id, document_id, filename, page, chunk_index, content,
                           ts_rank(fts, websearch_to_tsquery('english', %s)) AS score
                    FROM chunks
                    WHERE fts @@ websearch_to_tsquery('english', %s)
                      AND document_id = ANY(%s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, query, document_ids, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT id, document_id, filename, page, chunk_index, content,
                           ts_rank(fts, websearch_to_tsquery('english', %s)) AS score
                    FROM chunks
                    WHERE fts @@ websearch_to_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, query, top_k),
                )
            return [dict(r) for r in cur.fetchall()]
