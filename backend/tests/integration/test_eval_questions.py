import pathlib

import pytest
import yaml

from evals.axes import NEAR_MISS_AXES
from evals.corpus import load_manifest

QUESTIONS = pathlib.Path(__file__).resolve().parents[2] / "evals" / "questions.yaml"
BUCKETS = {"answerable", "near_miss", "off_corpus_medical", "off_domain"}


@pytest.fixture(scope="module")
def questions():
    return yaml.safe_load(QUESTIONS.read_text())


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def test_every_question_has_the_required_fields(questions):
    for q in questions:
        assert set(q) >= {"id", "bucket", "question", "expected"}, q
        assert q["bucket"] in BUCKETS, q
        assert q["expected"] in {"answer", "decline"}, q


def test_ids_are_unique(questions):
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))


def test_expected_matches_bucket(questions):
    for q in questions:
        want = "answer" if q["bucket"] == "answerable" else "decline"
        assert q["expected"] == want, f"{q['id']} bucket/expected mismatch"


def test_near_miss_targets_a_verified_absent_pair(questions, manifest):
    """THE test that stops the mistake this phase made twice: a near-miss whose
    answer is actually in the corpus would silently corrupt the headline result."""
    for q in questions:
        if q["bucket"] != "near_miss":
            continue
        drug, axis = q.get("drug"), q.get("axis")
        assert drug in manifest["drugs"], f"{q['id']} names unknown drug {drug!r}"
        assert axis in NEAR_MISS_AXES, f"{q['id']} names unknown axis {axis!r}"
        absent = manifest["drugs"][drug]["verified_absent"]
        assert axis in absent, (
            f"{q['id']} is labelled near_miss but {axis!r} is PRESENT in the "
            f"{drug} corpus — it is actually answerable"
        )


def test_answerable_names_a_drug_in_the_corpus(questions, manifest):
    for q in questions:
        if q["bucket"] == "answerable":
            assert q.get("drug") in manifest["drugs"], f"{q['id']} names unknown drug"


def test_off_corpus_questions_do_not_name_a_corpus_drug(questions, manifest):
    """An off-corpus question about a drug we actually have is not off-corpus."""
    corpus_drugs = set(manifest["drugs"])
    for q in questions:
        if q["bucket"] != "off_corpus_medical":
            continue
        named = {d for d in corpus_drugs if d in q["question"].lower()}
        assert not named, f"{q['id']} names corpus drug(s) {named}"


def test_off_corpus_questions_do_not_name_a_drug_class_the_corpus_belongs_to(questions, manifest):
    """Naming a drug NAME is the obvious collision; naming its CLASS is the
    subtle one. Amoxicillin self-identifies as "a penicillin-class antibacterial",
    so an off-corpus question mentioning penicillin partly retrieves corpus text."""
    from evals.corpus import corpus_text, load_drug

    corpus = " ".join(
        corpus_text(slug.title(), load_drug(slug)["included"]).lower() for slug in manifest["drugs"]
    )
    class_terms = ["penicillin", "beta-lactam", "biguanide", "beta blocker", "cephalosporin"]
    present = [t for t in class_terms if t in corpus]
    for q in questions:
        if q["bucket"] != "off_corpus_medical":
            continue
        named = [t for t in present if t in q["question"].lower()]
        assert not named, f"{q['id']} names corpus drug class(es) {named}"


def test_bucket_counts_are_balanced(questions):
    counts = {b: sum(1 for q in questions if q["bucket"] == b) for b in BUCKETS}
    assert counts["answerable"] >= 12, counts
    assert counts["near_miss"] >= 8, counts
    assert counts["off_corpus_medical"] >= 8, counts
    assert counts["off_domain"] >= 6, counts
    assert sum(counts.values()) >= 38, counts


def test_manifest_absence_still_holds(manifest):
    """Re-derive absence from the committed corpus. If a corpus change
    reintroduces withheld content, fail here rather than silently
    reclassifying near-misses as answerable."""
    from evals.axes import verified_absent_axes
    from evals.corpus import assemble_text, load_drug

    for slug, entry in manifest["drugs"].items():
        recomputed = verified_absent_axes(assemble_text(load_drug(slug)["included"]))
        assert recomputed == entry["verified_absent"], (
            f"{slug}: manifest says {entry['verified_absent']}, corpus says {recomputed}"
        )
