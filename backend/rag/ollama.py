"""Thin Ollama HTTP client. No Django imports."""
from __future__ import annotations

import httpx

from .config import OllamaConfig


class OllamaError(RuntimeError):
    """Base for Ollama transport failures."""


class OllamaUnavailable(OllamaError):
    """Ollama is not reachable at the configured host."""


class OllamaClient:
    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg

    def list_models(self) -> list[str]:
        try:
            resp = httpx.get(f"{self.cfg.host}/api/tags", timeout=5.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(str(exc)) from exc
        return [m["name"] for m in resp.json().get("models", [])]

    def is_reachable(self) -> bool:
        try:
            self.list_models()
            return True
        except OllamaUnavailable:
            return False
