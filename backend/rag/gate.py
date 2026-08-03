"""Stage-1 confidence gate.

Runs before the LLM, so a clearly off-domain question costs nothing. Pure:
no I/O, no logging, no clock — which is what lets the eval harness sweep it
offline over thousands of parameter combinations (spec 6.5).
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import GateConfig

REASONS = ("ok", "off_domain", "weak_unsupported", "empty_corpus")


@dataclass(frozen=True)
class GateSignals:
    top_similarity: float
    mean_similarity: float
    lexical_support: bool
    corpus_empty: bool = False

    def as_dict(self) -> dict:
        return {
            "top_similarity": self.top_similarity,
            "mean_similarity": self.mean_similarity,
            "lexical_support": self.lexical_support,
            "corpus_empty": self.corpus_empty,
        }


@dataclass(frozen=True)
class GateDecision:
    proceed: bool
    reason: str
    signals: dict


def evaluate_gate(signals: GateSignals, cfg: GateConfig) -> GateDecision:
    payload = signals.as_dict()

    if signals.corpus_empty:
        return GateDecision(False, "empty_corpus", payload)

    if signals.top_similarity < cfg.tau_abstain:
        return GateDecision(False, "off_domain", payload)

    # Middle band: semantically plausible but not confident. Require that the
    # question's actual terminology appears in the retrieved text. This is what
    # separates "metformin dosing" (documented) from "metformin pediatric
    # dosing" (absent) — both sit at nearly identical cosine distance.
    if signals.top_similarity < cfg.tau_strong and not signals.lexical_support:
        return GateDecision(False, "weak_unsupported", payload)

    return GateDecision(True, "ok", payload)
