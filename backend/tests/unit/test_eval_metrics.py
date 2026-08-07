import pytest

from evals.metrics import OperatingPoint, choose_best, score_point
from rag.config import GateConfig


def rec(bucket, expected, top, lex, sentinel, rid="x"):
    return {
        "id": rid, "bucket": bucket, "expected": expected,
        "top_similarity": top, "mean_similarity": top - 0.05,
        "lexical_support": lex, "corpus_empty": False,
        "sentinel_fired": sentinel,
    }


CFG = GateConfig(tau_abstain=0.30, tau_strong=0.45)


def test_answerable_that_answers_is_a_true_negative():
    p = score_point([rec("answerable", "answer", 0.9, True, False)], CFG)
    assert p.false_declines == 0
    assert p.declined == 0


def test_answerable_wrongly_declined_by_the_gate_counts_as_a_false_decline():
    p = score_point([rec("answerable", "answer", 0.10, True, False)], CFG)
    assert p.false_declines == 1
    assert p.stage1_declines == 1


def test_answerable_wrongly_declined_by_the_sentinel_also_counts():
    """A false decline is a false decline whichever stage produced it."""
    p = score_point([rec("answerable", "answer", 0.9, True, True)], CFG)
    assert p.false_declines == 1
    assert p.stage2_declines == 1


def test_off_domain_declined_by_the_gate_avoids_an_llm_call():
    p = score_point([rec("off_domain", "decline", 0.10, False, False)], CFG)
    assert p.stage1_declines == 1
    assert p.llm_calls_avoided == 1


def test_near_miss_caught_by_the_sentinel_costs_an_llm_call():
    p = score_point([rec("near_miss", "decline", 0.80, True, True)], CFG)
    assert p.stage2_declines == 1
    assert p.llm_calls_avoided == 0
    assert p.correct_declines == 1


def test_precision_and_recall_arithmetic():
    records = [
        rec("answerable", "answer", 0.90, True, False, "a1"),    # answered, correct
        rec("answerable", "answer", 0.10, True, False, "a2"),    # declined, WRONG
        rec("off_domain", "decline", 0.10, False, False, "d1"),  # declined, correct
        rec("near_miss", "decline", 0.80, True, True, "n1"),     # declined, correct
        rec("off_domain", "decline", 0.90, True, False, "d2"),   # answered, WRONG
    ]
    p = score_point(records, CFG)
    assert p.declined == 3
    assert p.correct_declines == 2
    assert p.should_decline == 3
    assert p.precision == pytest.approx(2 / 3)
    assert p.recall == pytest.approx(2 / 3)
    assert p.false_declines == 1


def test_precision_is_zero_not_an_error_when_nothing_is_declined():
    p = score_point([rec("answerable", "answer", 0.9, True, False)], CFG)
    assert p.precision == 0.0
    assert p.recall == 0.0


def test_choose_best_rejects_any_point_with_a_false_decline():
    """Refusing a question the system can answer is the failure users notice
    first, so it is ranked ahead of recall rather than traded against it."""
    bad = OperatingPoint(0.5, 0.7, declined=9, correct_declines=9, should_decline=9,
                         false_declines=1, stage1_declines=9, stage2_declines=0,
                         llm_calls_avoided=9, near_miss_stage1=0, near_miss_stage2=0)
    good = OperatingPoint(0.4, 0.6, declined=5, correct_declines=5, should_decline=9,
                          false_declines=0, stage1_declines=5, stage2_declines=0,
                          llm_calls_avoided=5, near_miss_stage1=0, near_miss_stage2=0)
    assert choose_best([bad, good]) is good


def test_choose_best_breaks_recall_ties_on_llm_calls_avoided():
    cheap = OperatingPoint(0.5, 0.7, declined=5, correct_declines=5, should_decline=9,
                           false_declines=0, stage1_declines=5, stage2_declines=0,
                           llm_calls_avoided=5, near_miss_stage1=0, near_miss_stage2=0)
    dear = OperatingPoint(0.3, 0.5, declined=5, correct_declines=5, should_decline=9,
                          false_declines=0, stage1_declines=0, stage2_declines=5,
                          llm_calls_avoided=0, near_miss_stage1=0, near_miss_stage2=0)
    assert choose_best([dear, cheap]) is cheap


def test_choose_best_falls_back_when_every_point_has_a_false_decline():
    only = OperatingPoint(0.5, 0.7, declined=9, correct_declines=8, should_decline=9,
                          false_declines=2, stage1_declines=9, stage2_declines=0,
                          llm_calls_avoided=9, near_miss_stage1=0, near_miss_stage2=0)
    assert choose_best([only]) is only


REPLAY_RECORDS = [
    rec("answerable", "answer", 0.90, True, False, "a1"),
    rec("near_miss", "decline", 0.80, True, True, "n1"),
    rec("off_domain", "decline", 0.35, False, False, "d1"),
]


def test_sweep_is_a_pure_replay():
    """The same records must produce identical points every run, or the
    committed results are not reproducible."""
    from evals.sweep import sweep

    assert sweep(REPLAY_RECORDS) == sweep(REPLAY_RECORDS)


def test_sweep_output_is_pinned_to_the_records_alone():
    """Two in-process calls agreeing proves determinism, not independence: a
    sweep that read GateConfig() or an environment variable would still agree
    with itself and still stop reproducing the moment production config moved.
    That is precisely the risk sweep.py's PLACEHOLDERS constant exists to
    guard, so the expected values are written out here as literals.

    The three chosen points also have to disagree with each other, or a
    degenerate sweep would satisfy all of them at once.
    """
    from evals.sweep import sweep

    points = {(p.tau_abstain, p.tau_strong): p for p in sweep(REPLAY_RECORDS)}

    # Permissive: off_domain at 0.35 clears the gate and the sentinel never
    # fires for it, so it is answered — half the declines that should happen.
    permissive = points[(0.20, 0.25)]
    assert (permissive.declined, permissive.correct_declines) == (1, 1)
    assert (permissive.should_decline, permissive.recall) == (2, 0.5)
    assert (permissive.stage1_declines, permissive.stage2_declines) == (0, 1)
    assert permissive.llm_calls_avoided == 0

    # Balanced: off_domain is now below tau_abstain and dies at stage 1, the
    # near-miss still needs the model to catch it.
    balanced = points[(0.40, 0.45)]
    assert (balanced.declined, balanced.correct_declines) == (2, 2)
    assert (balanced.false_declines, balanced.recall) == (0, 1.0)
    assert (balanced.stage1_declines, balanced.stage2_declines) == (1, 1)
    assert (balanced.llm_calls_avoided, balanced.near_miss_stage2) == (1, 1)

    # Strict: tau_abstain now sits above the near-miss too, so stage 1 catches
    # it and the model is never called for either decline.
    strict = points[(0.85, 0.90)]
    assert (strict.declined, strict.false_declines) == (2, 0)
    assert (strict.stage1_declines, strict.stage2_declines) == (2, 0)
    assert (strict.llm_calls_avoided, strict.near_miss_stage1) == (2, 1)


def test_grid_only_contains_valid_threshold_pairs():
    from evals.sweep import sweep

    points = sweep([rec("answerable", "answer", 0.9, True, False)])
    assert all(p.tau_abstain < p.tau_strong for p in points)
    assert len(points) == 120
