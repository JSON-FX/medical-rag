"""Unit tests for the thin Ollama HTTP client (rag/ollama.py).

`/api/health/`'s entire job is to never 500. `resp.json()` and the
`m["name"]` list comprehension in `list_models()` used to sit outside the
`try`, so a 200 with a non-JSON body or a renamed response key escaped as a
raw JSONDecodeError/KeyError instead of the documented `OllamaUnavailable` —
the same class of escape already closed in embeddings.py and generation.py.
"""
import httpx
import pytest

from rag.config import OllamaConfig
from rag.ollama import OllamaClient, OllamaUnavailable

CFG = OllamaConfig()


def _response(url, status=200, json_body=None, text=None):
    kwargs = {"json": json_body} if json_body is not None else {"text": text}
    # A real httpx.get() always attaches the request to its response;
    # raise_for_status() requires that regardless of status code, so the
    # fake must mirror it or it fails for an unrelated reason.
    return httpx.Response(status, request=httpx.Request("GET", url), **kwargs)


def test_list_models_returns_names_from_a_well_formed_response(monkeypatch):
    def fake_get(url, *a, **k):
        return _response(url, json_body={"models": [{"name": "llama3.1:8b"}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    assert OllamaClient(CFG).list_models() == ["llama3.1:8b"]


def test_empty_models_list_returns_empty_not_an_error(monkeypatch):
    def fake_get(url, *a, **k):
        return _response(url, json_body={"models": []})

    monkeypatch.setattr(httpx, "get", fake_get)
    assert OllamaClient(CFG).list_models() == []


def test_connection_failure_becomes_ollama_unavailable(monkeypatch):
    def fake_get(url, *a, **k):
        raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(OllamaUnavailable):
        OllamaClient(CFG).list_models()


def test_non_json_200_body_becomes_ollama_unavailable_not_a_500(monkeypatch):
    """A 200 with an HTML body (e.g. a proxy error page) must not leak
    json.JSONDecodeError to callers that only catch OllamaError."""

    def fake_get(url, *a, **k):
        return _response(url, text="<html>gateway error</html>")

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(OllamaUnavailable, match="malformed response"):
        OllamaClient(CFG).list_models()


def test_missing_name_key_becomes_ollama_unavailable_not_a_500(monkeypatch):
    """A renamed response key used to raise a raw KeyError from the list
    comprehension, which sat outside the try block entirely."""

    def fake_get(url, *a, **k):
        return _response(url, json_body={"models": [{"id": "llama3.1:8b"}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(OllamaUnavailable, match="malformed response"):
        OllamaClient(CFG).list_models()


def test_non_dict_model_entry_becomes_ollama_unavailable_not_a_500(monkeypatch):
    """A "models" entry that is a bare string (or any non-mapping) raises
    TypeError on subscript rather than KeyError — both must be caught."""

    def fake_get(url, *a, **k):
        return _response(url, json_body={"models": ["llama3.1:8b"]})

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(OllamaUnavailable, match="malformed response"):
        OllamaClient(CFG).list_models()


def test_is_reachable_is_false_rather_than_raising_on_a_malformed_response(monkeypatch):
    def fake_get(url, *a, **k):
        return _response(url, text="not json")

    monkeypatch.setattr(httpx, "get", fake_get)
    assert OllamaClient(CFG).is_reachable() is False
