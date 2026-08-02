"""Every tunable in one place. No Django imports (see spec 3.1)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    chat_model: str = "llama3.1:8b"      # NOT llama3.1:8b-instruct (invalid tag)
    embed_model: str = "nomic-embed-text"
    embed_dimensions: int = 768
    request_timeout_s: int = 120


@dataclass(frozen=True)
class ChunkConfig:
    size: int = 1000
    overlap: int = 150


@dataclass(frozen=True)
class RetrievalConfig:
    per_leg: int = 10     # candidates pulled from each of the vector and lexical legs
    top_k: int = 4        # chunks kept after fusion
    rrf_k: int = 60       # reciprocal rank fusion constant


@dataclass(frozen=True)
class GateConfig:
    # PLACEHOLDERS until the Phase 3 eval sweep. See spec 6.5 and 17.
    tau_abstain: float = 0.30
    tau_strong: float = 0.45


@dataclass(frozen=True)
class RagConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    max_upload_mb: int = 15
    history_messages: int = 4


def load_config(env: Mapping[str, str] | None = None) -> RagConfig:
    e = os.environ if env is None else env

    def _f(key: str, default: float) -> float:
        return float(e.get(key, default))

    def _i(key: str, default: int) -> int:
        return int(e.get(key, default))

    def _s(key: str, default: str) -> str:
        return e.get(key, default)

    return RagConfig(
        ollama=OllamaConfig(
            host=_s("OLLAMA_HOST", "http://127.0.0.1:11434"),
            chat_model=_s("CHAT_MODEL", "llama3.1:8b"),
            embed_model=_s("EMBED_MODEL", "nomic-embed-text"),
            request_timeout_s=_i("OLLAMA_TIMEOUT_S", 120),
        ),
        chunk=ChunkConfig(size=_i("CHUNK_SIZE", 1000), overlap=_i("CHUNK_OVERLAP", 150)),
        retrieval=RetrievalConfig(
            per_leg=_i("RETRIEVE_N", 10), top_k=_i("TOP_K", 4), rrf_k=_i("RRF_K", 60)
        ),
        gate=GateConfig(tau_abstain=_f("TAU_ABSTAIN", 0.30), tau_strong=_f("TAU_STRONG", 0.45)),
        max_upload_mb=_i("MAX_UPLOAD_MB", 15),
        history_messages=_i("HISTORY_MESSAGES", 4),
    )
