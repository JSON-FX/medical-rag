import json
import pytest
from django.test import Client

from rag.config import load_config


class FakeOllama:
    """Stands in for OllamaClient. Keep the signature identical."""

    def __init__(self, models, reachable=True):
        self._models = models
        self._reachable = reachable

    def is_reachable(self):
        return self._reachable

    def list_models(self):
        if not self._reachable:
            raise ConnectionError("unreachable")
        return self._models


@pytest.fixture
def client():
    return Client()


def test_health_reports_all_present(client, monkeypatch):
    import chat.views as views
    monkeypatch.setattr(
        views, "build_client", lambda cfg: FakeOllama(["llama3.1:8b", "nomic-embed-text:latest"])
    )
    resp = client.get("/api/health/")
    body = json.loads(resp.content)
    assert resp.status_code == 200
    assert body["ollama_reachable"] is True
    assert body["models"]["chat"] is True
    assert body["models"]["embed"] is True


def test_health_matches_model_tags_ignoring_latest_suffix(client, monkeypatch):
    """`ollama list` reports `nomic-embed-text:latest` for a `nomic-embed-text` pull."""
    import chat.views as views
    monkeypatch.setattr(
        views, "build_client", lambda cfg: FakeOllama(["llama3.1:8b", "nomic-embed-text:latest"])
    )
    body = json.loads(client.get("/api/health/").content)
    assert body["models"]["embed"] is True


def test_health_reports_missing_model(client, monkeypatch):
    import chat.views as views
    monkeypatch.setattr(views, "build_client", lambda cfg: FakeOllama(["nomic-embed-text:latest"]))
    body = json.loads(client.get("/api/health/").content)
    assert body["models"]["chat"] is False
    assert body["models"]["embed"] is True


def test_health_reports_unreachable_without_raising(client, monkeypatch):
    import chat.views as views
    monkeypatch.setattr(views, "build_client", lambda cfg: FakeOllama([], reachable=False))
    resp = client.get("/api/health/")
    body = json.loads(resp.content)
    assert resp.status_code == 200          # health must not 500 when Ollama is down
    assert body["ollama_reachable"] is False
    assert body["models"]["chat"] is False
