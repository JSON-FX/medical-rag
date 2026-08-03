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


# --- ghost ready documents: status=ready, chunk_count>0, zero chunk rows ---
#
# Reachable via a crash between the chunk delete and the document delete in
# the DELETE view (now wrapped in transaction.atomic(), but this command must
# still be able to see the state if it arises any other way). Neither the
# missing-vector nor the orphan-vector check can catch it: with zero actual
# chunk rows there is nothing to be missing a vector and nothing to orphan,
# so pre-fix this reported "No drift" over a document that is completely
# unsearchable.


def test_detects_a_ready_document_with_no_actual_chunk_rows(store):
    Document.objects.create(title="ghost", status="ready", chunk_count=3)
    output = _run()
    assert "no drift" not in output.lower()
    assert "1 document" in output.lower()
    assert "ready" in output.lower()


def test_clean_report_is_unaffected_by_a_ready_document_with_zero_chunk_count(store):
    """A `ready` document that legitimately has no chunks (chunk_count == 0,
    e.g. never finished ingesting) is not a ghost and must not be flagged."""
    Document.objects.create(title="not a ghost", status="ready", chunk_count=0)
    assert "no drift" in _run().lower()


def test_fix_marks_a_ghost_ready_document_as_failed_with_the_reupload_message(store):
    doc = Document.objects.create(title="ghost", status="ready", chunk_count=3)
    _run("--fix")
    doc.refresh_from_db()
    assert doc.status == "failed"
    assert "re-upload" in doc.error_message.lower()


def test_ghost_ready_repair_does_not_touch_an_unrelated_healthy_document(store):
    """The repair must be scoped to the broken document only."""
    healthy = Document.objects.create(title="healthy", status="ready", chunk_count=1)
    chunk = Chunk.objects.create(document=healthy, chunk_index=0, page_number=1, text="ok")
    store.upsert(
        [chunk.vector_id], [[0.5] * 768], [{"document_id": healthy.id, "chunk_index": 0}]
    )
    ghost = Document.objects.create(title="ghost", status="ready", chunk_count=3)

    _run("--fix")

    healthy.refresh_from_db()
    ghost.refresh_from_db()
    assert healthy.status == "ready"
    assert ghost.status == "failed"


def test_ghost_ready_detection_runs_independently_of_missing_and_orphan_repair(store):
    """One kind of drift must not block detection or repair of another."""
    Chunk.objects.create(
        document=Document.objects.create(title="missing-vec", status="ready"),
        chunk_index=0, page_number=1, text="no vector",
    )
    store.upsert(["99_0"], [[0.5] * 768], [{"document_id": 99, "chunk_index": 0}])
    ghost = Document.objects.create(title="ghost", status="ready", chunk_count=1)

    _run("--fix")

    ghost.refresh_from_db()
    assert ghost.status == "failed"


def test_fix_is_convergent_for_ghost_ready_documents(store):
    """Once marked failed, status is no longer `ready`, so a second run must
    not re-flag it and must reach a clean report."""
    Document.objects.create(title="ghost", status="ready", chunk_count=3)
    _run("--fix")
    assert "no drift" in _run().lower()
