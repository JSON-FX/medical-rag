import pytest

from rag.config import OllamaConfig
from rag.embeddings import OllamaEmbedder

CFG = OllamaConfig()


class SpyTransport:
    """Captures the payload the embedder sends."""

    def __init__(self, dims=768):
        self.payloads = []
        self.dims = dims

    def __call__(self, url, payload):
        self.payloads.append(payload)
        return {"embeddings": [[0.1] * self.dims for _ in payload["input"]]}


def test_documents_get_the_search_document_prefix():
    spy = SpyTransport()
    OllamaEmbedder(CFG, transport=spy).embed_documents(["metformin 500mg"])
    assert spy.payloads[0]["input"] == ["search_document: metformin 500mg"]


def test_queries_get_the_search_query_prefix():
    spy = SpyTransport()
    OllamaEmbedder(CFG, transport=spy).embed_query("what is the dose?")
    assert spy.payloads[0]["input"] == ["search_query: what is the dose?"]


def test_documents_are_sent_as_one_batch_not_n_requests():
    spy = SpyTransport()
    OllamaEmbedder(CFG, transport=spy).embed_documents(["a", "b", "c"])
    assert len(spy.payloads) == 1
    assert len(spy.payloads[0]["input"]) == 3


def test_embed_query_returns_a_flat_vector_not_a_list_of_one():
    vec = OllamaEmbedder(CFG, transport=SpyTransport()).embed_query("q")
    assert isinstance(vec[0], float)
    assert len(vec) == 768


def test_empty_document_list_makes_no_request():
    spy = SpyTransport()
    assert OllamaEmbedder(CFG, transport=spy).embed_documents([]) == []
    assert spy.payloads == []


def test_dimension_mismatch_is_rejected_loudly():
    """A wrong embed model silently produces wrong-width vectors; fail fast."""
    with pytest.raises(ValueError, match="768"):
        OllamaEmbedder(CFG, transport=SpyTransport(dims=384)).embed_documents(["a"])
