import pytest
from django.db import connection

from documents.models import Chunk, Document

pytestmark = pytest.mark.django_db


def _doc(**kw):
    return Document.objects.create(title=kw.pop("title", "Monograph"), **kw)


def _fts_search(expression: str) -> list[int]:
    with connection.cursor() as cur:
        cur.execute(
            "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH %s ORDER BY bm25(chunk_fts)",
            [expression],
        )
        return [row[0] for row in cur.fetchall()]


def test_vector_id_is_deterministic():
    doc = _doc()
    chunk = Chunk.objects.create(document=doc, chunk_index=3, page_number=1, text="x")
    assert chunk.vector_id == f"{doc.id}_3"


def test_chunk_index_is_unique_per_document():
    doc = _doc()
    Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="a")
    with pytest.raises(Exception):
        Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="b")


def test_insert_trigger_populates_the_fts_index():
    doc = _doc()
    chunk = Chunk.objects.create(
        document=doc, chunk_index=0, page_number=1, text="metformin dosing in adults"
    )
    assert _fts_search('"metformin"') == [chunk.id]


def test_porter_stemming_matches_dose_to_dosing():
    """Justifies the porter tokenizer over plain unicode61."""
    doc = _doc()
    chunk = Chunk.objects.create(
        document=doc, chunk_index=0, page_number=1, text="recommended dosing schedule"
    )
    assert _fts_search('"dose"') == [chunk.id]


def test_delete_trigger_removes_the_row_from_fts():
    doc = _doc()
    chunk = Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="atenolol")
    chunk_id = chunk.id
    chunk.delete()
    assert _fts_search('"atenolol"') == []


def test_deleting_a_document_cascades_to_chunks_and_fts():
    doc = _doc()
    Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="warfarin")
    doc.delete()
    assert Chunk.objects.count() == 0
    assert _fts_search('"warfarin"') == []


def test_bm25_returns_negative_scores():
    """Documents the sign convention that makes rank-only fusion necessary."""
    doc = _doc()
    Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="metformin metformin")
    with connection.cursor() as cur:
        cur.execute("SELECT bm25(chunk_fts) FROM chunk_fts WHERE chunk_fts MATCH '\"metformin\"'")
        assert cur.fetchone()[0] < 0
