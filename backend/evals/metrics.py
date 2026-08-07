"""Metric computation over cached signals. Pure — no I/O, no network.

Simulating the full two-stage outcome from a cached record is valid because
neither stage's raw behaviour depends on the thresholds: retrieval signals come
from the question and corpus, and whether the model emitted the sentinel came
from the question and the retrieved chunks. Only the gate's decision RULE reads
tau_abstain and tau_strong (spec 4.1).
"""
from __future__ import annotations

from dataclasses import dataclass

from rag.config import GateConfig
from rag.gate import GateSignals, evaluate_gate


@dataclass(frozen=True)
class OperatingPoint:
    tau_abstain: float
    tau_strong: float
    declined: int
    correct_declines: int
    should_decline: int
    false_declines: int
    stage1_declines: int
    stage2_declines: int
    llm_calls_avoided: int
    near_miss_stage1: int
    near_miss_stage2: int

    @property
    def precision(self) -> float:
        return self.correct_declines / self.declined if self.declined else 0.0

    @property
    def recall(self) -> float:
        return self.correct_declines / self.should_decline if self.should_decline else 0.0


def score_point(records: list[dict], cfg: GateConfig) -> OperatingPoint:
    declined = correct = false_declines = 0
    stage1 = stage2 = avoided = nm1 = nm2 = 0
    should_decline = sum(1 for r in records if r["expected"] == "decline")

    for r in records:
        signals = GateSignals(
            top_similarity=r["top_similarity"],
            mean_similarity=r["mean_similarity"],
            lexical_support=r["lexical_support"],
            corpus_empty=r["corpus_empty"],
        )
        decision = evaluate_gate(signals, cfg)

        if not decision.proceed:
            outcome, stage = "decline", 1
        elif r["sentinel_fired"]:
            outcome, stage = "decline", 2
        else:
            outcome, stage = "answer", 0

        if outcome == "decline":
            declined += 1
            if stage == 1:
                stage1 += 1
                avoided += 1          # the LLM was never called
            else:
                stage2 += 1
            if r["expected"] == "decline":
                correct += 1
                if r["bucket"] == "near_miss":
                    nm1 += stage == 1
                    nm2 += stage == 2
            else:
                false_declines += 1   # refused a question it could answer

    return OperatingPoint(
        tau_abstain=cfg.tau_abstain,
        tau_strong=cfg.tau_strong,
        declined=declined,
        correct_declines=correct,
        should_decline=should_decline,
        false_declines=false_declines,
        stage1_declines=stage1,
        stage2_declines=stage2,
        llm_calls_avoided=avoided,
        near_miss_stage1=nm1,
        near_miss_stage2=nm2,
    )


def choose_best(points: list[OperatingPoint]) -> OperatingPoint:
    """Rank lexicographically, not by a blended score (spec 4.5).

    Zero false declines first: a system that refuses questions it can answer is
    broken in the way users notice first, and an F1-style objective would
    happily trade those away to buy decline recall.
    """
    clean = [p for p in points if p.false_declines == 0] or points
    return max(clean, key=lambda p: (p.recall, p.llm_calls_avoided, p.precision))
