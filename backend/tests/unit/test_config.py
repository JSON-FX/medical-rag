import os
from rag.config import load_config, RagConfig


def test_defaults_match_spec():
    cfg = load_config(env={})
    assert cfg.ollama.host == "http://127.0.0.1:11434"
    assert cfg.ollama.chat_model == "llama3.1:8b"
    assert cfg.ollama.embed_model == "nomic-embed-text"
    assert cfg.chunk.size == 1000
    assert cfg.chunk.overlap == 150
    assert cfg.retrieval.per_leg == 10
    assert cfg.retrieval.top_k == 4
    assert cfg.retrieval.rrf_k == 60
    # Measured by the Phase 3 eval sweep — see evals/eval_results.md. These
    # assertions are what stops the thresholds drifting silently.
    assert cfg.gate.tau_abstain == 0.70
    assert cfg.gate.tau_strong == 0.75
    assert cfg.max_upload_mb == 15
    assert cfg.history_messages == 4


def test_env_overrides_are_typed():
    cfg = load_config(env={"TAU_ABSTAIN": "0.5", "CHUNK_SIZE": "800", "CHAT_MODEL": "other:7b"})
    assert cfg.gate.tau_abstain == 0.5
    assert isinstance(cfg.gate.tau_abstain, float)
    assert cfg.chunk.size == 800
    assert isinstance(cfg.chunk.size, int)
    assert cfg.ollama.chat_model == "other:7b"


def test_config_is_frozen():
    cfg = load_config(env={})
    try:
        cfg.gate.tau_abstain = 0.9
    except Exception:
        return
    raise AssertionError("GateConfig must be frozen")
