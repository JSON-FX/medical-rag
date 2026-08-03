"""Thin Ollama HTTP client. No Django imports."""
from __future__ import annotations

import httpx

from .config import OllamaConfig


class OllamaError(RuntimeError):
    """Base for Ollama transport failures."""


class OllamaUnavailable(OllamaError):
    """Ollama is not reachable at the configured host."""


class OllamaProtocolError(OllamaError, ValueError):
    """Ollama responded, but the payload was not usable.

    Subclasses both `OllamaError` (so callers that correctly catch the base
    transport-failure type also catch this) and `ValueError` (so existing
    `pytest.raises(ValueError, ...)` assertions on the count/dimension checks
    in `embeddings.py` keep passing unmodified — this genuinely is a bad
    value, just one that also belongs to the Ollama error family).
    """


class OllamaClient:
    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg

    def list_models(self) -> list[str]:
        try:
            resp = httpx.get(f"{self.cfg.host}/api/tags", timeout=5.0)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(str(exc)) from exc
        except (ValueError, KeyError, TypeError) as exc:
            # ValueError: resp.json() on a non-JSON 200 body (json.JSONDecodeError
            # subclasses ValueError). KeyError: an entry missing "name". TypeError:
            # a "models" entry that isn't a mapping. Health's entire job is to
            # never 500, so a 200 with a malformed payload must surface the same
            # way an unreachable host does, not escape as a raw parsing error —
            # the same class of escape already closed in embeddings.py and
            # generation.py.
            raise OllamaUnavailable(f"malformed response from Ollama: {exc}") from exc

    def is_reachable(self) -> bool:
        try:
            self.list_models()
            return True
        except OllamaUnavailable:
            return False
