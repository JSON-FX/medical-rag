import json
import math
import pathlib

import pytest
import yaml

EVALS = pathlib.Path(__file__).resolve().parents[2] / "evals"
SIGNALS = EVALS / "signals.json"
QUESTIONS = EVALS / "questions.yaml"


@pytest.fixture(scope="module")
def records():
    if not SIGNALS.exists():
        pytest.skip("signals.json not collected yet - run `python -m evals.collect`")
    return json.loads(SIGNALS.read_text())


def test_every_question_has_a_record(records):
    questions = yaml.safe_load(QUESTIONS.read_text())
    assert {r["id"] for r in records} == {q["id"] for q in questions}


def test_every_record_carries_the_fields_the_sweep_needs(records):
    required = {"id", "bucket", "expected", "top_similarity", "mean_similarity",
                "lexical_support", "corpus_empty", "sentinel_fired"}
    for r in records:
        assert required <= set(r), f"{r['id']} missing {required - set(r)}"


def test_similarities_are_finite_and_in_range(records):
    for r in records:
        for key in ("top_similarity", "mean_similarity"):
            value = r[key]
            assert math.isfinite(value), f"{r['id']} {key} is not finite"
            assert -1.0 <= value <= 1.0, f"{r['id']} {key}={value} outside [-1, 1]"


def test_the_corpus_was_not_empty(records):
    assert not any(r["corpus_empty"] for r in records), "corpus failed to ingest"


def test_answerable_retrieved_more_strongly_than_off_domain(records):
    # If this inverts, the corpus or the questions are wrong and no threshold
    # chosen from this data would mean anything.
    answerable = [r["top_similarity"] for r in records if r["bucket"] == "answerable"]
    off_domain = [r["top_similarity"] for r in records if r["bucket"] == "off_domain"]
    assert min(answerable) > max(off_domain), (
        f"answerable min {min(answerable):.3f} <= off_domain max {max(off_domain):.3f}"
    )


def test_answerable_questions_were_not_refused_by_the_model(records):
    # A sentinel on an answerable question means it is not actually answerable
    # from the corpus. It would count as a false decline at EVERY operating
    # point, silently dragging down the whole sweep.
    refused = [r["id"] for r in records if r["bucket"] == "answerable" and r["sentinel_fired"]]
    assert not refused, f"answerable questions the model refused: {refused}"


def test_lexical_support_carries_signal(records):
    # The gate's middle band depends on this signal distinguishing corpus
    # content from unrelated questions.
    off_domain_lex = sum(r["lexical_support"] for r in records if r["bucket"] == "off_domain")
    answerable_lex = sum(r["lexical_support"] for r in records if r["bucket"] == "answerable")
    assert answerable_lex > off_domain_lex, "lexical_support carries no signal"
