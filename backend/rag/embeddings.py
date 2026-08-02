"""Ollama embedding client.

`nomic-embed-text` is a prefixed model: indexed text needs `search_document: `
and queries need `search_query: `. Omitting them degrades retrieval silently,
so the prefixes are applied here and are not callable parameters (spec 6.3).
"""
from __future__ import annotations

from typing import Callable

import httpx

from .config import OllamaConfig
from .ollama import OllamaUnavailable

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def _http_transport(url: str, payload: dict) -> dict:
    try:
        resp = httpx.post(url, json=payload, timeout=120.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaUnavailable(f"embed request failed: {exc}") from exc
    return resp.json()


class OllamaEmbedder:
    def __init__(self, cfg: OllamaConfig, transport: Callable[[str, dict], dict] | None = None):
        self.cfg = cfg
        self._transport = transport or _http_transport

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        body = self._transport(
            f"{self.cfg.host}/api/embed", {"model": self.cfg.embed_model, "input": inputs}
        )
        vectors = body.get("embeddings", [])
        for vector in vectors:
            if len(vector) != self.cfg.embed_dimensions:
                raise ValueError(
                    f"expected {self.cfg.embed_dimensions}-dim embeddings from "
                    f"{self.cfg.embed_model}, got {len(vector)}"
                )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([DOCUMENT_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([QUERY_PREFIX + text])[0]
