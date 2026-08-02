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


def test_short_embedding_response_raises_rather_than_misaligning():
    """A count mismatch would pair chunk text with the wrong vector in Task 7's
    ingestion, silently poisoning the store. Fail loudly instead."""

    class ShortTransport:
        def __call__(self, url, payload):
            return {"embeddings": [[0.1] * 768]}  # 1 vector for N inputs

    with pytest.raises(ValueError, match="misalign"):
        OllamaEmbedder(CFG, transport=ShortTransport()).embed_documents(["a", "b", "c"])


def test_missing_embeddings_key_raises_rather_than_returning_empty():
    class EmptyTransport:
        def __call__(self, url, payload):
            return {"model": "nomic-embed-text"}  # no embeddings key at all

    with pytest.raises(ValueError):
        OllamaEmbedder(CFG, transport=EmptyTransport()).embed_documents(["a"])


def test_non_json_response_becomes_ollama_unavailable(monkeypatch):
    """A 200 with an HTML body must not leak json.JSONDecodeError to callers
    that only catch OllamaError."""
    import httpx

    from rag.embeddings import _http_transport
    from rag.ollama import OllamaUnavailable

    def fake_post(url, *a, **k):
        # A real httpx.post() always attaches the request to its response;
        # raise_for_status() requires that regardless of status code, so the
        # fake must mirror it or it fails for an unrelated reason.
        return httpx.Response(200, text="<html>gateway error</html>", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(OllamaUnavailable, match="not valid JSON"):
        _http_transport("http://x/api/embed", {}, 5.0)
