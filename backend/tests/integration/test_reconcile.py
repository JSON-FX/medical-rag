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
