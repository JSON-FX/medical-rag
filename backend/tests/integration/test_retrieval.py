import pytest

from chat.retrieval import retrieve
from documents.models import Chunk, Document
from rag.config import load_config

pytestmark = pytest.mark.django_db

CFG = load_config(env={})


@pytest.fixture
def seeded(chroma_store, fake_embedder):
    doc = Document.objects.create(title="Monograph", status="ready")
    chunks = [
        Chunk.objects.create(document=doc, chunk_index=0, page_number=1,
                             text="Metformin adult starting dose is 500mg twice daily."),
        Chunk.objects.create(document=doc, chunk_index=1, page_number=2,
                             text="Atenolol is a beta blocker for hypertension."),
    ]
    chroma_store.upsert(
        ids=[c.vector_id for c in chunks],
        embeddings=fake_embedder.embed_documents([c.text for c in chunks]),
        metadatas=[{"document_id": doc.id, "chunk_index": c.chunk_index} for c in chunks],
    )
    return doc


def test_empty_corpus_short_circuits_before_retrieval(chroma_store, fake_embedder):
    result = retrieve("anything", fake_embedder, chroma_store, CFG)
    assert result.decision.proceed is False
    assert result.decision.reason == "empty_corpus"
    assert result.chunks == []


def test_on_topic_question_proceeds_with_hydrated_chunks(seeded, chroma_store, fake_embedder):
    result = retrieve("metformin dose", fake_embedder, chroma_store, CFG)
    assert result.decision.proceed is True
    assert result.chunks
    assert "Metformin" in result.chunks[0].text
    assert result.chunks[0].title == "Monograph"
    assert result.chunks[0].page_number == 1


def test_off_domain_question_declines(seeded, chroma_store, fake_embedder):
    result = retrieve("what is the capital of france", fake_embedder, chroma_store, CFG)
    assert result.decision.proceed is False
    assert result.decision.reason == "off_domain"


def test_gate_signals_are_populated_for_observability(seeded, chroma_store, fake_embedder):
    result = retrieve("metformin dose", fake_embedder, chroma_store, CFG)
    assert set(result.decision.signals) == {
        "top_similarity", "mean_similarity", "lexical_support", "corpus_empty"
    }


def test_chunks_are_limited_to_top_k(seeded, chroma_store, fake_embedder):
    result = retrieve("metformin", fake_embedder, chroma_store, CFG)
    assert len(result.chunks) <= CFG.retrieval.top_k


def test_vector_ids_with_no_sqlite_row_are_dropped_not_crashed(
    seeded, chroma_store, fake_embedder
):
    """An orphaned vector must not break a query (spec 10)."""
    chroma_store.upsert(
        ["999_0"],
        fake_embedder.embed_documents(["metformin"]),
        [{"document_id": 999, "chunk_index": 0}],
    )
    result = retrieve("metformin dose", fake_embedder, chroma_store, CFG)
    assert all(c.chunk_id != "999_0" for c in result.chunks)


def test_lexical_support_does_not_depend_on_document_id_ordering(chroma_store, fake_embedder):
    """Regression: when the legs disagree, RRF ties and the winner was decided
    by chunk_id string sort, so this signal flipped with document numbering."""
    from rag.gate import GateSignals

    # Two documents whose ids sort in opposite directions relative to each other.
    low = Document.objects.create(title="Low", status="ready")
    high = Document.objects.create(title="High", status="ready")
    for doc, text in ((low, "metformin dosing guidance"), (high, "atenolol dosing guidance")):
        chunk = Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text=text)
        chroma_store.upsert(
            [chunk.vector_id],
            fake_embedder.embed_documents([text]),
            [{"document_id": doc.id, "chunk_index": 0}],
        )

    result = retrieve("metformin dosing", fake_embedder, chroma_store, CFG)
    assert result.decision.signals["lexical_support"] is True, (
        "a chunk found by the lexical leg was delivered, so support must be True"
    )


def test_lexical_support_is_false_when_no_delivered_chunk_was_found_lexically(
    chroma_store, fake_embedder
):
    """A question whose terminology appears nowhere in the delivered text."""
    doc = Document.objects.create(title="Mono", status="ready")
    chunk = Chunk.objects.create(
        document=doc, chunk_index=0, page_number=1, text="Atenolol is a beta blocker."
    )
    chroma_store.upsert(
        [chunk.vector_id],
        fake_embedder.embed_documents([chunk.text]),
        [{"document_id": doc.id, "chunk_index": 0}],
    )
    result = retrieve("zzzquux nonexistent terminology", fake_embedder, chroma_store, CFG)
    assert result.decision.signals["lexical_support"] is False
