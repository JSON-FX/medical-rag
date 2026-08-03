import pytest

from rag.config import GateConfig
from rag.gate import GateDecision, GateSignals, evaluate_gate

CFG = GateConfig(tau_abstain=0.30, tau_strong=0.45)


def signals(top=0.9, mean=0.8, lexical=True, empty=False) -> GateSignals:
    return GateSignals(
        top_similarity=top, mean_similarity=mean, lexical_support=lexical, corpus_empty=empty
    )


def test_empty_corpus_declines_before_any_other_check():
    decision = evaluate_gate(signals(top=0.99, empty=True), CFG)
    assert decision.proceed is False
    assert decision.reason == "empty_corpus"


def test_similarity_below_tau_abstain_is_off_domain():
    decision = evaluate_gate(signals(top=0.10), CFG)
    assert decision.proceed is False
    assert decision.reason == "off_domain"


def test_high_similarity_proceeds():
    decision = evaluate_gate(signals(top=0.80), CFG)
    assert decision.proceed is True
    assert decision.reason == "ok"


def test_middle_band_without_lexical_support_declines():
    decision = evaluate_gate(signals(top=0.40, lexical=False), CFG)
    assert decision.proceed is False
    assert decision.reason == "weak_unsupported"


def test_middle_band_with_lexical_support_proceeds():
    assert evaluate_gate(signals(top=0.40, lexical=True), CFG).proceed is True


def test_lexical_support_cannot_rescue_below_tau_abstain():
    """Off-domain is off-domain even if a stray word matched."""
    decision = evaluate_gate(signals(top=0.05, lexical=True), CFG)
    assert decision.reason == "off_domain"


@pytest.mark.parametrize("top,expected", [(0.2999, "off_domain"), (0.30, "weak_unsupported")])
def test_tau_abstain_boundary_is_inclusive_upward(top, expected):
    assert evaluate_gate(signals(top=top, lexical=False), CFG).reason == expected


@pytest.mark.parametrize("top,expected", [(0.4499, "weak_unsupported"), (0.45, "ok")])
def test_tau_strong_boundary_is_inclusive_upward(top, expected):
    assert evaluate_gate(signals(top=top, lexical=False), CFG).reason == expected


def test_mean_similarity_does_not_affect_the_decision():
    """Pins the documented v1 behaviour (spec 6.5). If mean_similarity is
    ever promoted to a rule, this test must fail rather than the change
    passing unnoticed."""
    low = evaluate_gate(signals(top=0.80, mean=0.01), CFG)
    high = evaluate_gate(signals(top=0.80, mean=0.79), CFG)
    assert low.proceed is True and high.proceed is True
    assert low.reason == high.reason


def test_signals_are_carried_on_the_decision_for_observability():
    decision = evaluate_gate(signals(top=0.8, mean=0.7, lexical=True), CFG)
    assert decision.signals["top_similarity"] == 0.8
    assert decision.signals["mean_similarity"] == 0.7
    assert decision.signals["lexical_support"] is True


def test_decision_is_frozen():
    decision = evaluate_gate(signals(), CFG)
    assert isinstance(decision, GateDecision)
    with pytest.raises(Exception):
        decision.proceed = False
