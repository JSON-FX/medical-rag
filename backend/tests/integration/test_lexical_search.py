import pytest

from chat.lexical_search import search
from documents.models import Chunk, Document

pytestmark = pytest.mark.django_db


@pytest.fixture
def seeded():
    doc = Document.objects.create(title="Monograph", status="ready")
    Chunk.objects.create(document=doc, chunk_index=0, page_number=1,
                         text="Metformin adult starting dose is 500mg twice daily.")
    Chunk.objects.create(document=doc, chunk_index=1, page_number=2,
                         text="Atenolol is a beta blocker used for hypertension.")
    return doc


def test_search_returns_vector_ids_best_first(seeded):
    assert search("metformin dose", limit=5)[0] == f"{seeded.id}_0"


def test_search_matches_stemmed_terms(seeded):
    assert f"{seeded.id}_0" in search("dosing", limit=5)


def test_search_respects_the_limit(seeded):
    assert len(search("metformin atenolol dose", limit=1)) == 1


def test_punctuation_heavy_question_does_not_raise(seeded):
    assert search("What's the max dose (mg/kg)?", limit=5)


def test_question_with_no_usable_terms_returns_empty(seeded):
    assert search("??? !!!", limit=5) == []


def test_search_on_empty_corpus_returns_empty():
    assert search("metformin", limit=5) == []
