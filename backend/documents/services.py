"""Process-wide singletons for the Chroma store and embedder.

Separate module so tests can monkeypatch these without importing views.
"""
from __future__ import annotations

from django.conf import settings

from rag.config import load_config
from rag.embeddings import OllamaEmbedder
from rag.vectorstore import ChromaStore

_store: ChromaStore | None = None


def get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore(path=str(settings.CHROMA_PATH))
    return _store


def get_embedder() -> OllamaEmbedder:
    return OllamaEmbedder(load_config().ollama)
