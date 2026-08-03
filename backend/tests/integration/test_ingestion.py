import pytest
from django.core.files.base import ContentFile

from documents.ingestion import cleanup_document, extract_pages, ingest_document
from documents.models import Chunk, Document
from rag.config import load_config
from tests.conftest import ExplodingEmbedder
from tests.fixtures.make_fixture_pdf import make_blank_pdf, make_pdf

pytestmark = pytest.mark.django_db

CFG = load_config(env={"CHUNK_SIZE": "120", "CHUNK_OVERLAP": "20"})


@pytest.fixture
def pdf_doc(tmp_path):
    path = make_pdf(tmp_path / "mono.pdf", ["Metformin adult dose is 500mg.", "Atenolol 50mg daily."])
    doc = Document.objects.create(title="mono.pdf")
    doc.file.save("mono.pdf", ContentFile(path.read_bytes()), save=True)
    return doc


def test_extract_pages_returns_one_entry_per_page(tmp_path):
    path = make_pdf(tmp_path / "two.pdf", ["page one text", "page two text"])
    pages = extract_pages(str(path))
    assert [p.page_number for p in pages] == [1, 2]
    assert "page one" in pages[0].text


def test_successful_ingest_marks_ready_with_counts(pdf_doc, chroma_store, fake_embedder):
    doc = ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    assert doc.status == "ready"
    assert doc.page_count == 2
    assert doc.chunk_count > 0
    assert doc.chunk_count == Chunk.objects.filter(document=doc).count()


def test_vectors_and_chunks_agree_after_ingest(pdf_doc, chroma_store, fake_embedder):
    doc = ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    assert chroma_store.count() == doc.chunk_count
    assert chroma_store.all_ids() == {c.vector_id for c in Chunk.objects.filter(document=doc)}


def test_chunks_carry_real_page_numbers(pdf_doc, chroma_store, fake_embedder):
    ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    assert set(Chunk.objects.values_list("page_number", flat=True)) == {1, 2}


def test_embeddings_are_batched_in_one_call(pdf_doc, chroma_store, fake_embedder):
    ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    assert fake_embedder.document_batches == 1


def test_pdf_with_no_extractable_text_fails_with_a_useful_message(
    tmp_path, chroma_store, fake_embedder
):
    path = make_blank_pdf(tmp_path / "scanned.pdf")
    doc = Document.objects.create(title="scanned.pdf")
    doc.file.save("scanned.pdf", ContentFile(path.read_bytes()), save=True)

    result = ingest_document(doc, fake_embedder, chroma_store, CFG)
    assert result.status == "failed"
    assert "no extractable text" in result.error_message.lower()
    assert "ocr" in result.error_message.lower()
    assert Chunk.objects.count() == 0
    assert chroma_store.count() == 0


def test_embedding_failure_leaves_no_orphans(pdf_doc, chroma_store):
    result = ingest_document(pdf_doc, ExplodingEmbedder(), chroma_store, CFG)
    assert result.status == "failed"
    assert "ollama exploded" in result.error_message
    assert Chunk.objects.count() == 0
    assert chroma_store.count() == 0


def test_reingesting_the_same_document_does_not_duplicate(
    pdf_doc, chroma_store, fake_embedder
):
    first = ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    count = first.chunk_count
    second = ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    assert second.chunk_count == count
    assert chroma_store.count() == count
    assert Chunk.objects.filter(document=pdf_doc).count() == count


def test_cleanup_removes_from_both_stores(pdf_doc, chroma_store, fake_embedder):
    ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    cleanup_document(pdf_doc.id, chroma_store)
    assert Chunk.objects.filter(document=pdf_doc).count() == 0
    assert chroma_store.count() == 0


def test_failed_reingest_preserves_a_working_document(pdf_doc, chroma_store, fake_embedder):
    """A transient Ollama outage during re-ingest must not cost the user a
    document that is currently ready and searchable."""
    first = ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    assert first.status == "ready"
    original_count = first.chunk_count
    original_ids = chroma_store.all_ids()

    result = ingest_document(pdf_doc, ExplodingEmbedder(), chroma_store, CFG)

    assert result.status == "ready", "a working document was destroyed by a failed re-ingest"
    assert result.chunk_count == original_count
    assert Chunk.objects.filter(document=pdf_doc).count() == original_count
    assert chroma_store.all_ids() == original_ids
    assert "ollama exploded" in result.error_message


def test_first_ingest_failure_still_marks_failed(pdf_doc, chroma_store):
    """A document that was never ready has nothing to preserve."""
    result = ingest_document(pdf_doc, ExplodingEmbedder(), chroma_store, CFG)
    assert result.status == "failed"
    assert result.chunk_count == 0
    assert Chunk.objects.count() == 0
    assert chroma_store.count() == 0


def test_cleanup_failure_does_not_strand_document_in_processing(pdf_doc, chroma_store, fake_embedder):
    """If Chroma is unreachable during cleanup, the document must still get a
    terminal status and a message — not sit in `processing` forever."""
    ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)

    class BrokenStore:
        def delete_document(self, document_id):
            raise RuntimeError("chroma unreachable")
        def upsert(self, *a, **k):
            raise RuntimeError("chroma unreachable")
        def count(self):
            return 0
        def all_ids(self):
            return set()

    result = ingest_document(pdf_doc, fake_embedder, BrokenStore(), CFG)
    assert result.status in {"ready", "failed"}, "document stranded in processing"
    assert result.status != "processing"
    assert result.error_message
