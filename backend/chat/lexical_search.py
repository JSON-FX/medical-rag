"""BM25 search over chunk text via SQLite FTS5.

bm25() returns NEGATIVE values where more-negative means a better match, so
results are ordered ascending. The magnitude is never compared to a cosine
similarity — fusion uses rank position only (spec 6.2).
"""
from __future__ import annotations

from django.db import connection

from rag.lexical import build_fts_query

SQL = """
    SELECT c.document_id, c.chunk_index
    FROM chunk_fts f
    JOIN documents_chunk c ON c.id = f.rowid
    WHERE chunk_fts MATCH %s
    ORDER BY bm25(chunk_fts)
    LIMIT %s
"""


def search(question: str, limit: int) -> list[str]:
    if not isinstance(limit, int) or limit <= 0:
        return []
    expression = build_fts_query(question)
    if not expression:
        return []
    with connection.cursor() as cursor:
        cursor.execute(SQL, [expression, limit])
        return [f"{document_id}_{chunk_index}" for document_id, chunk_index in cursor.fetchall()]
