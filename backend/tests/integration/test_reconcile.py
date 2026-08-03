import io

import pytest
from django.core.management import call_command

from documents.models import Chunk, Document

pytestmark = pytest.mark.django_db


@pytest.fixture
def store(tmp_path, monkeypatch):
    import documents.services as services
    from rag.vectorstore import ChromaStore

    s = ChromaStore(path=str(tmp_path / "chroma"), collection_name="test")
    monkeypatch.setattr(services, "get_store", lambda: s)
    return s


def _run(*args) -> str:
    out = io.StringIO()
    call_command("reconcile_vectors", *args, stdout=out)
    return out.getvalue()


def test_reports_clean_when_stores_agree(store):
    doc = Document.objects.create(title="d", status="ready")
    chunk = Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="metformin")
    store.upsert([chunk.vector_id], [[0.5] * 768], [{"document_id": doc.id, "chunk_index": 0}])
    assert "no drift" in _run().lower()


def test_detects_chunk_without_a_vector(store):
    doc = Document.objects.create(title="d", status="ready")
    Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="orphan row")
    output = _run()
    assert "1 chunk" in output.lower()
    assert "missing" in output.lower()


def test_detects_vector_without_a_chunk(store):
    store.upsert(["99_0"], [[0.5] * 768], [{"document_id": 99, "chunk_index": 0}])
    output = _run()
    assert "1 vector" in output.lower()
    assert "orphan" in output.lower()


def test_fix_removes_orphaned_vectors(store):
    store.upsert(["99_0"], [[0.5] * 768], [{"document_id": 99, "chunk_index": 0}])
    _run("--fix")
    assert store.count() == 0


def test_fix_marks_documents_with_missing_vectors_as_failed(store):
    doc = Document.objects.create(title="d", status="ready")
    Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="orphan row")
    _run("--fix")
    doc.refresh_from_db()
    assert doc.status == "failed"
    assert "re-upload" in doc.error_message.lower()


def test_fix_preserves_valid_vectors_when_removing_an_orphan(store):
    """The repair must never be more destructive than the drift it repairs."""
    doc = Document.objects.create(title="d", status="ready")
    kept = [
        Chunk.objects.create(document=doc, chunk_index=i, page_number=1, text=f"chunk {i}")
        for i in range(2)
    ]
    store.upsert(
        ids=[c.vector_id for c in kept] + [f"{doc.id}_99"],
        embeddings=[[0.5] * 768] * 3,
        metadatas=[{"document_id": doc.id, "chunk_index": i} for i in (0, 1, 99)],
    )

    _run("--fix")

    assert store.all_ids() == {c.vector_id for c in kept}, "valid vectors destroyed"
    doc.refresh_from_db()
    assert doc.status == "ready", "a repaired document was wrongly marked failed"


def test_fix_survives_a_malformed_vector_id_and_still_marks_documents(store):
    """One bad row must not defeat the safety net for unrelated documents."""
    broken = Document.objects.create(title="broken", status="ready")
    Chunk.objects.create(document=broken, chunk_index=0, page_number=1, text="no vector")
    store.upsert(["not-an-int_0"], [[0.5] * 768], [{"document_id": 1, "chunk_index": 0}])

    _run("--fix")

    broken.refresh_from_db()
    assert broken.status == "failed", "missing-vector repair was skipped"
    assert "not-an-int_0" not in store.all_ids()


def test_fix_is_convergent(store):
    """Running twice must reach a clean state, not crash or oscillate."""
    store.upsert(["77_0"], [[0.5] * 768], [{"document_id": 77, "chunk_index": 0}])
    _run("--fix")
    assert "no drift" in _run().lower()
