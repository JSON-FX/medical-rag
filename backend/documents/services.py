"""Process-wide singletons for the Chroma store and embedder.

Separate module so tests can monkeypatch these without importing views.
"""
from __future__ import annotations

import threading

from django.conf import settings

from rag.config import load_config
from rag.embeddings import OllamaEmbedder
from rag.vectorstore import ChromaStore

_store: ChromaStore | None = None
_store_lock = threading.Lock()


def get_store() -> ChromaStore:
    """Process-wide Chroma handle.

    Double-checked locking because sync views run in uvicorn's threadpool:
    concurrent first requests would otherwise race chromadb's tenant
    initialisation, which fails loudly and non-deterministically.
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ChromaStore(path=str(settings.CHROMA_PATH))
    return _store


def get_embedder() -> OllamaEmbedder:
    return OllamaEmbedder(load_config().ollama)
