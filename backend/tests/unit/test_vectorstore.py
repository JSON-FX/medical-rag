import pytest

from rag.vectorstore import ChromaStore, VectorHit


@pytest.fixture
def store(tmp_path):
    return ChromaStore(path=str(tmp_path / "chroma"), collection_name="test_docs")


def _vec(seed: float) -> list[float]:
    return [seed] * 8


def test_collection_uses_cosine_space_not_the_l2_default():
    """Chroma defaults to L2. The whole gate depends on cosine (spec 6.4).

    Verified empirically on chromadb 1.5.9: with no configuration, two opposite
    unit vectors sit at distance 4.0 (L2 squared); with cosine they sit at 2.0.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        assert ChromaStore(path=tmp, collection_name="spacecheck").space == "cosine"


def test_opposite_vectors_are_distance_two_proving_cosine_not_l2(store):
    """The behavioural counterpart to the config assertion above: L2 would
    give 4.0 here, so this fails loudly if the space silently reverts."""
    store.upsert(
        ids=["1_0", "1_1"],
        embeddings=[_vec(1.0), [-1.0] + [0.0] * 7],
        metadatas=[{"document_id": 1, "chunk_index": 0}, {"document_id": 1, "chunk_index": 1}],
    )
    hits = store.query([1.0] + [0.0] * 7, n_results=2)
    assert hits[1].distance == pytest.approx(2.0, abs=1e-3)


def test_upsert_then_query_returns_nearest_first(store):
    store.upsert(
        ids=["1_0", "1_1"],
        embeddings=[_vec(1.0), _vec(-1.0)],
        metadatas=[{"document_id": 1, "chunk_index": 0}, {"document_id": 1, "chunk_index": 1}],
    )
    hits = store.query(_vec(1.0), n_results=2)
    assert [h.chunk_id for h in hits] == ["1_0", "1_1"]
    assert isinstance(hits[0], VectorHit)


def test_cosine_distance_of_identical_vectors_is_about_zero(store):
    store.upsert(["1_0"], [_vec(1.0)], [{"document_id": 1, "chunk_index": 0}])
    hit = store.query(_vec(1.0), n_results=1)[0]
    assert hit.distance == pytest.approx(0.0, abs=1e-4)


def test_upsert_is_idempotent_for_the_same_id(store):
    meta = [{"document_id": 1, "chunk_index": 0}]
    store.upsert(["1_0"], [_vec(1.0)], meta)
    store.upsert(["1_0"], [_vec(0.5)], meta)
    assert store.count() == 1


def test_delete_document_removes_only_that_documents_vectors(store):
    store.upsert(
        ids=["1_0", "2_0"],
        embeddings=[_vec(1.0), _vec(0.5)],
        metadatas=[{"document_id": 1, "chunk_index": 0}, {"document_id": 2, "chunk_index": 0}],
    )
    store.delete_document(1)
    assert store.count() == 1
    assert [h.chunk_id for h in store.query(_vec(0.5), n_results=5)] == ["2_0"]


def test_query_on_empty_collection_returns_empty_list(store):
    assert store.query(_vec(1.0), n_results=4) == []


def test_upsert_with_no_ids_is_a_noop(store):
    store.upsert([], [], [])
    assert store.count() == 0


def test_query_with_non_positive_n_results_returns_empty_not_crash(store):
    """Chroma raises TypeError on n_results<=0; callers get [] instead."""
    store.upsert(["1_0"], [_vec(1.0)], [{"document_id": 1, "chunk_index": 0}])
    assert store.query(_vec(1.0), n_results=0) == []
    assert store.query(_vec(1.0), n_results=-1) == []


def test_query_tolerates_n_results_larger_than_collection(store):
    """Verified against chromadb 1.5.9: over-asking returns fewer, never raises."""
    store.upsert(["1_0"], [_vec(1.0)], [{"document_id": 1, "chunk_index": 0}])
    hits = store.query(_vec(1.0), n_results=50)
    assert len(hits) == 1


def test_delete_document_returns_number_removed(store):
    store.upsert(
        ids=["7_0", "7_1", "8_0"],
        embeddings=[_vec(1.0), _vec(0.9), _vec(0.5)],
        metadatas=[
            {"document_id": 7, "chunk_index": 0},
            {"document_id": 7, "chunk_index": 1},
            {"document_id": 8, "chunk_index": 0},
        ],
    )
    assert store.delete_document(7) == 2
    assert store.all_ids() == {"8_0"}


def test_delete_document_for_unknown_id_returns_zero(store):
    assert store.delete_document(999) == 0
