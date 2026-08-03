# Phase 3 — Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the confidence gate's placeholder thresholds with values measured against a real labelled question set, and commit the sweep that justifies them.

**Architecture:** Three real FDA drug labels form a fixture corpus whose withheld content is *verified absent by keyword scan*, not assumed absent from section names. A collect pass calls the LLM once per question — unconditionally, regardless of what the gate would say — and caches retrieval signals plus sentinel outcome. A sweep pass then replays that cache across a 120-point threshold grid in milliseconds, because the gate is a pure function.

**Tech Stack:** Python 3.12, `pyyaml` (new dependency), existing `rag/` library, Ollama (`llama3.1:8b`, `nomic-embed-text`), pytest.

**Spec:** [`docs/superpowers/specs/2026-08-03-eval-harness-design.md`](../specs/2026-08-03-eval-harness-design.md). Section references below (§N) point there.

---

## Global Constraints

- **No authored clinical content.** Every medical fact in the corpus comes from a real FDA label. Questions are authored *about* that text; the text itself is never invented.
- **`evals/` is a measurement tool, not production code.** It may import from `rag/` and Django, but nothing in `rag/`, `documents/` or `chat/` may import from `evals/`.
- **The three labels are pinned by `set_id`** and must not be changed:
  - metformin `011de1a5-1ac0-4831-9e8d-26ec79ba2205`
  - atenolol `09b21985-1818-449d-9b29-98f733cf7b9f`
  - amoxicillin `00fbd46e-05fd-4f8a-9f59-a7a4d01c8e54`
- **Included sections:** `indications_and_usage`, `dosage_and_administration`, `contraindications`, `adverse_reactions`, `drug_interactions`.
- **Withheld sections:** `pediatric_use`, `overdosage`, `pregnancy`.
- **Absence is measured, never assumed.** The manifest's `verified_absent` list is the authority for which near-miss questions are legitimate (§2.2.1).
- **`fetch_fixtures.py` is the only component that touches the network.** Everything else runs offline except for Ollama calls in `collect.py`.
- **`signals.json` and `eval_results.md` are committed**, so the sweep is reproducible without Ollama.
- Run tests with `uv run --no-sync pytest` from inside `backend/`. Current baseline: **231 passed / 3 deselected**.

**Measured ground truth** (from §2.2.1 — the plan was written against this, do not re-derive):

| axis | metformin | atenolol | amoxicillin |
|---|---|---|---|
| pediatric | present | **absent** | present |
| overdose | **absent** | **absent** | present |
| pregnancy | **absent** | **absent** | present |
| geriatric | **absent** | present | **absent** |
| hepatic | **absent** | **absent** | **absent** |
| renal dosing | present | present | present |

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/evals/__init__.py` | package marker |
| `backend/evals/axes.py` | near-miss axis definitions + `verified_absent_axes()` — pure, no I/O |
| `backend/evals/fetch_fixtures.py` | one-time openFDA fetch; writes raw JSON + manifest |
| `backend/evals/corpus.py` | build paginated PDFs from fixture text |
| `backend/evals/fixtures/*.json` | committed raw label sections |
| `backend/evals/fixtures/manifest.json` | committed: included/withheld/verified_absent per drug |
| `backend/evals/questions.yaml` | ~40 hand-authored labelled questions |
| `backend/evals/collect.py` | pass 1 — ingest corpus, run questions, cache signals |
| `backend/evals/signals.json` | committed pass-1 output |
| `backend/evals/metrics.py` | pure metric computation + operating-point selection |
| `backend/evals/sweep.py` | pass 2 — grid over cached signals, writes results |
| `backend/evals/eval_results.md` | **the artifact this phase exists to produce** |
| `backend/tests/unit/test_eval_axes.py` | axis scanning is correct |
| `backend/tests/unit/test_eval_metrics.py` | metric arithmetic + operating-point ranking |
| `backend/tests/integration/test_eval_questions.py` | question set validates against the manifest |

---

## Task 1: Fixture corpus with verified-absent axes

**Files:**
- Create: `backend/evals/__init__.py`, `backend/evals/axes.py`, `backend/evals/fetch_fixtures.py`, `backend/evals/corpus.py`
- Modify: `backend/tests/fixtures/make_fixture_pdf.py` (add PDF string escaping)
- Test: `backend/tests/unit/test_eval_axes.py`

**Interfaces:**
- Consumes: nothing
- Produces: `evals.axes.NEAR_MISS_AXES`, `evals.axes.verified_absent_axes(text) -> list[str]`;
  `evals.corpus.build_pdf(path, title, sections) -> Path`;
  committed `evals/fixtures/{metformin,atenolol,amoxicillin}.json` and `manifest.json`

- [ ] **Step 1: Add the `pyyaml` dependency**

```bash
cd backend && uv add pyyaml
```

- [ ] **Step 2: Write the failing test**

`backend/tests/unit/test_eval_axes.py`:

```python
from evals.axes import NEAR_MISS_AXES, verified_absent_axes


def test_axis_present_when_any_keyword_appears():
    text = "Pediatric Dosage: starting dose 500 mg orally twice a day."
    assert "pediatric" not in verified_absent_axes(text)


def test_axis_absent_when_no_keyword_appears():
    text = "Adult dosage: 50 mg once daily for hypertension."
    absent = verified_absent_axes(text)
    assert "pediatric" in absent
    assert "pregnancy" in absent


def test_scan_is_case_insensitive():
    assert "pregnancy" not in verified_absent_axes("PREGNANT women should not take this.")


def test_every_axis_is_reported_one_way_or_the_other():
    absent = set(verified_absent_axes("some unrelated text"))
    assert absent == set(NEAR_MISS_AXES), "an axis went unreported"


def test_real_label_phrasings_are_detected():
    """These exact phrasings appear in the pinned FDA labels."""
    cases = [
        ("pediatric", "In Pediatric Patients over 3 Months of Age, 20 to 45 mg/kg/day"),
        ("pediatric", "The safety and effectiveness in children have not been established"),
        ("overdose", "Overdose of metformin hydrochloride has occurred"),
        ("pregnancy", "Limited data with metformin in pregnant women"),
        ("geriatric", "Atenolol is excreted by the kidneys; elderly patients"),
        ("hepatic", "No dosage adjustment is needed for hepatic impairment"),
    ]
    for axis, text in cases:
        assert axis not in verified_absent_axes(text), f"{axis!r} not detected in {text!r}"


def test_absent_axes_are_sorted_for_stable_manifests():
    assert verified_absent_axes("x") == sorted(verified_absent_axes("x"))
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run --no-sync pytest tests/unit/test_eval_axes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evals'`

- [ ] **Step 4: Write `evals/axes.py`**

```python
"""Near-miss axis detection.

A near-miss question is only legitimate if the corpus genuinely lacks the
answer. Withholding a label SECTION is not enough to guarantee that: metformin's
`dosage_and_administration` carries a full "Pediatric Dosage" paragraph even
with `pediatric_use` withheld, so a pediatric question about metformin is
answerable from the shipped text. Absence is therefore measured over the
assembled corpus, not inferred from section names (spec 2.2.1).
"""
from __future__ import annotations

import re

NEAR_MISS_AXES: dict[str, list[str]] = {
    "pediatric": [r"pediatric", r"children", r"\bchild\b", r"infant", r"neonate", r"adolescent"],
    "overdose": [r"overdos", r"toxicity", r"ingestion of amounts"],
    "pregnancy": [r"pregnan", r"lactation", r"nursing", r"breast-?feed", r"teratogen"],
    "geriatric": [r"geriatric", r"elderly", r"older patients"],
    "hepatic": [r"hepatic impairment", r"liver impairment", r"hepatic dysfunction"],
}


def verified_absent_axes(text: str) -> list[str]:
    """Axes with no keyword anywhere in `text`, sorted for stable manifests."""
    lowered = text.lower()
    return sorted(
        axis
        for axis, patterns in NEAR_MISS_AXES.items()
        if not any(re.search(p, lowered) for p in patterns)
    )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run --no-sync pytest tests/unit/test_eval_axes.py -v`
Expected: PASS — 6 tests

- [ ] **Step 6: Fix PDF string escaping in the shared fixture generator**

`backend/tests/fixtures/make_fixture_pdf.py` builds a PDF string literal as `({text})`, with no
escaping. FDA label text is full of parentheses — `(2.1)`, `(eGFR below 30 mL/min/1.73 m 2)` — and an
unbalanced paren corrupts the PDF. This is a latent bug that never fired because existing fixtures
used simple text.

Add this helper and use it where the content stream is built:

```python
def _escape(text: str) -> str:
    """PDF string literals need \\, ( and ) escaped. FDA label text is full of
    parentheses, and an unbalanced one corrupts the file."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
```

Then change the content-stream line from `f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"` to use
`_escape(text)`.

- [ ] **Step 7: Verify escaping with a test**

Add to `backend/tests/unit/test_chunking.py` — or a new `tests/unit/test_fixture_pdf.py`:

```python
def test_generated_pdf_survives_parentheses_and_backslashes(tmp_path):
    """FDA label text is full of parens; an unbalanced one corrupts the PDF."""
    from pypdf import PdfReader
    from tests.fixtures.make_fixture_pdf import make_pdf

    tricky = r"Starting dose 500 mg (2.1) with meals \ see (5.1) and (eGFR below 30)"
    path = make_pdf(tmp_path / "tricky.pdf", [tricky])
    text = PdfReader(str(path)).pages[0].extract_text() or ""
    assert "2.1" in text
    assert "eGFR" in text
```

Run: `uv run --no-sync pytest tests/unit -k "fixture_pdf or chunking" -v`
Expected: PASS

- [ ] **Step 8: Write `evals/corpus.py`**

```python
"""Build the eval corpus from committed fixture text.

Pagination is deliberate: real page numbers make citation assertions meaningful
and exercise the page-aware chunker the way an uploaded document would.
"""
from __future__ import annotations

import json
import pathlib
import re

from tests.fixtures.make_fixture_pdf import make_pdf

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CHARS_PER_PAGE = 1200

SECTION_TITLES = {
    "indications_and_usage": "INDICATIONS AND USAGE",
    "dosage_and_administration": "DOSAGE AND ADMINISTRATION",
    "contraindications": "CONTRAINDICATIONS",
    "adverse_reactions": "ADVERSE REACTIONS",
    "drug_interactions": "DRUG INTERACTIONS",
}


def _paginate(text: str, chars_per_page: int = CHARS_PER_PAGE) -> list[str]:
    words, pages, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > chars_per_page:
            pages.append(current.strip())
            current = ""
        current += word + " "
    if current.strip():
        pages.append(current.strip())
    return pages


def assemble_text(included: dict[str, str]) -> str:
    """One normalised string over the included sections, in a stable order."""
    parts = []
    for key, title in SECTION_TITLES.items():
        body = included.get(key)
        if body:
            parts.append(f"{title}\n{re.sub(r'\\s+', ' ', body).strip()}")
    return "\n\n".join(parts)


def build_pdf(path: pathlib.Path, title: str, included: dict[str, str]) -> pathlib.Path:
    body = f"{title}\n\n{assemble_text(included)}"
    return make_pdf(path, _paginate(body))


def load_manifest() -> dict:
    return json.loads((FIXTURES / "manifest.json").read_text())


def load_drug(slug: str) -> dict:
    return json.loads((FIXTURES / f"{slug}.json").read_text())
```

- [ ] **Step 9: Write `evals/fetch_fixtures.py`**

```python
"""One-time openFDA fetch. The ONLY component that touches the network.

Labels are pinned by set_id so a re-fetch returns the same document, and the
extracted text is committed so the eval is reproducible offline and an upstream
change cannot silently move the results.

US federal government works are public domain. No clinical content is authored
for this project.
"""
from __future__ import annotations

import json
import pathlib
import urllib.parse
import urllib.request

from evals.axes import verified_absent_axes
from evals.corpus import FIXTURES, assemble_text

DRUGS = {
    "metformin": "011de1a5-1ac0-4831-9e8d-26ec79ba2205",
    "atenolol": "09b21985-1818-449d-9b29-98f733cf7b9f",
    "amoxicillin": "00fbd46e-05fd-4f8a-9f59-a7a4d01c8e54",
}
INCLUDE = [
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "adverse_reactions",
    "drug_interactions",
]
WITHHOLD = ["pediatric_use", "overdosage", "pregnancy"]


def fetch(set_id: str) -> dict:
    query = urllib.parse.quote(f'set_id:"{set_id}"')
    url = f"https://api.fda.gov/drug/label.json?search={query}&limit=1"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)["results"][0]


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "openFDA drug label API (US federal work, public domain)", "drugs": {}}

    for slug, set_id in DRUGS.items():
        record = fetch(set_id)
        included = {s: record[s][0] for s in INCLUDE if record.get(s)}
        withheld = {s: record[s][0] for s in WITHHOLD if record.get(s)}
        (FIXTURES / f"{slug}.json").write_text(
            json.dumps({"set_id": set_id, "included": included, "withheld": withheld}, indent=1)
        )
        manifest["drugs"][slug] = {
            "set_id": set_id,
            "included_sections": sorted(included),
            "withheld_sections": sorted(withheld),
            "included_chars": sum(len(v) for v in included.values()),
            # Measured over the text that actually ships, not inferred from
            # which sections were withheld (spec 2.2.1).
            "verified_absent": verified_absent_axes(assemble_text(included)),
        }
        print(f"{slug:12} {manifest['drugs'][slug]['included_chars']:6} chars  "
              f"absent={manifest['drugs'][slug]['verified_absent']}")

    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {FIXTURES / 'manifest.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Run the fetch**

```bash
cd backend && uv run --no-sync python -m evals.fetch_fixtures
```

Expected output — these are the measured values from §2.2.1, and they must match:

```
metformin      9930 chars  absent=['geriatric', 'hepatic', 'overdose', 'pregnancy']
atenolol      17155 chars  absent=['hepatic', 'overdose', 'pediatric', 'pregnancy']
amoxicillin   14596 chars  absent=['geriatric', 'hepatic']
```

If the character counts differ, openFDA has changed the label — STOP and report it rather than
proceeding, because every question in Task 2 was authored against these exact documents.

- [ ] **Step 11: Commit**

```bash
git add backend/evals backend/tests/fixtures/make_fixture_pdf.py backend/tests/unit/ backend/pyproject.toml backend/uv.lock
git commit -m "feat: add eval fixture corpus with verified-absent axes

Three real FDA labels pinned by set_id, public domain. Absence of near-miss
content is measured over the assembled text rather than inferred from which
sections were withheld — withholding pediatric_use does not remove the
pediatric dosing paragraph that lives in dosage_and_administration.

Also escapes PDF string literals: FDA text is full of parentheses and an
unbalanced one corrupts the file."
```

---

## Task 2: Labelled question set

**Files:**
- Create: `backend/evals/questions.yaml`
- Test: `backend/tests/integration/test_eval_questions.py`

**Interfaces:**
- Consumes: `evals.corpus.load_manifest()`, `evals.axes.NEAR_MISS_AXES`
- Produces: `evals/questions.yaml` — a list of records with `id`, `bucket`, `question`, `expected`, and optional `drug` / `axis`

- [ ] **Step 1: Write the failing validity test**

`backend/tests/integration/test_eval_questions.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --no-sync pytest tests/integration/test_eval_questions.py -v`
Expected: FAIL — `questions.yaml` does not exist

- [ ] **Step 3: Write `evals/questions.yaml`**

Every `answerable` question below is answerable from the committed fixture text; every `near_miss`
targets a `(drug, axis)` pair the manifest lists as verified absent.

```yaml
# Labelled eval question set. See spec 3.
#
# answerable         -> a fact present in the included corpus text
# near_miss          -> a (drug, axis) pair the manifest lists as verified_absent
# off_corpus_medical -> a real medical question about a drug NOT in the corpus
# off_domain         -> not medical at all
#
# Near-miss pairs are validated against fixtures/manifest.json by
# tests/integration/test_eval_questions.py. Do not add one without checking.

# ---------- answerable (14) ----------
- {id: a01, bucket: answerable, expected: answer, drug: metformin,
   question: "What is the adult starting dose of metformin?"}
- {id: a02, bucket: answerable, expected: answer, drug: metformin,
   question: "What is the maximum daily dose of metformin?"}
- {id: a03, bucket: answerable, expected: answer, drug: metformin,
   question: "In which patients is metformin contraindicated?"}
- {id: a04, bucket: answerable, expected: answer, drug: metformin,
   question: "At what eGFR is metformin contraindicated?"}
- {id: a05, bucket: answerable, expected: answer, drug: metformin,
   question: "How often should the metformin dose be increased?"}
- {id: a06, bucket: answerable, expected: answer, drug: atenolol,
   question: "What is the initial dose of atenolol for hypertension?"}
- {id: a07, bucket: answerable, expected: answer, drug: atenolol,
   question: "What is the maximum useful daily dose of atenolol for hypertension?"}
- {id: a08, bucket: answerable, expected: answer, drug: atenolol,
   question: "When is atenolol contraindicated?"}
- {id: a09, bucket: answerable, expected: answer, drug: atenolol,
   question: "What is atenolol indicated for?"}
- {id: a10, bucket: answerable, expected: answer, drug: amoxicillin,
   question: "What is the adult daily dose range of amoxicillin?"}
- {id: a11, bucket: answerable, expected: answer, drug: amoxicillin,
   question: "What is the amoxicillin triple therapy regimen for H. pylori?"}
- {id: a12, bucket: answerable, expected: answer, drug: amoxicillin,
   question: "Who should not receive amoxicillin?"}
- {id: a13, bucket: answerable, expected: answer, drug: amoxicillin,
   question: "How often is amoxicillin given in divided doses?"}
- {id: a14, bucket: answerable, expected: answer, drug: metformin,
   question: "What drug interactions are described for metformin?"}

# ---------- near_miss (9) — every pair verified absent in the manifest ----------
- {id: n01, bucket: near_miss, expected: decline, drug: metformin, axis: hepatic,
   question: "How should metformin be dosed in patients with hepatic impairment?"}
- {id: n02, bucket: near_miss, expected: decline, drug: atenolol, axis: hepatic,
   question: "Does atenolol require a dose adjustment in liver impairment?"}
- {id: n03, bucket: near_miss, expected: decline, drug: amoxicillin, axis: hepatic,
   question: "What is the amoxicillin dose for a patient with hepatic dysfunction?"}
- {id: n04, bucket: near_miss, expected: decline, drug: metformin, axis: overdose,
   question: "What should be done in the event of a metformin overdose?"}
- {id: n05, bucket: near_miss, expected: decline, drug: atenolol, axis: overdose,
   question: "How is an atenolol overdose managed?"}
- {id: n06, bucket: near_miss, expected: decline, drug: metformin, axis: pregnancy,
   question: "Is metformin safe to use during pregnancy?"}
- {id: n07, bucket: near_miss, expected: decline, drug: atenolol, axis: pregnancy,
   question: "Can atenolol be taken while breastfeeding?"}
- {id: n08, bucket: near_miss, expected: decline, drug: atenolol, axis: pediatric,
   question: "What is the pediatric dose of atenolol?"}
- {id: n09, bucket: near_miss, expected: decline, drug: metformin, axis: geriatric,
   question: "How should metformin be dosed in elderly patients?"}

# ---------- off_corpus_medical (9) — real drugs, none in the corpus ----------
- {id: o01, bucket: off_corpus_medical, expected: decline,
   question: "What are the contraindications for warfarin?"}
- {id: o02, bucket: off_corpus_medical, expected: decline,
   question: "What is the starting dose of lisinopril for hypertension?"}
- {id: o03, bucket: off_corpus_medical, expected: decline,
   question: "How is insulin glargine titrated?"}
- {id: o04, bucket: off_corpus_medical, expected: decline,
   question: "What is the maximum daily dose of ibuprofen?"}
- {id: o05, bucket: off_corpus_medical, expected: decline,
   question: "What are the side effects of atorvastatin?"}
- {id: o06, bucket: off_corpus_medical, expected: decline,
   question: "When should azithromycin be used instead of penicillin?"}
- {id: o07, bucket: off_corpus_medical, expected: decline,
   question: "What is the reversal agent for heparin?"}
- {id: o08, bucket: off_corpus_medical, expected: decline,
   question: "How should levothyroxine be monitored?"}
- {id: o09, bucket: off_corpus_medical, expected: decline,
   question: "What is the recommended dose of prednisone for an asthma exacerbation?"}

# ---------- off_domain (8) ----------
- {id: d01, bucket: off_domain, expected: decline,
   question: "What is the capital of France?"}
- {id: d02, bucket: off_domain, expected: decline,
   question: "How do I bake sourdough bread?"}
- {id: d03, bucket: off_domain, expected: decline,
   question: "Who won the 1998 World Cup?"}
- {id: d04, bucket: off_domain, expected: decline,
   question: "Write me a Python function that reverses a string."}
- {id: d05, bucket: off_domain, expected: decline,
   question: "What is the weather forecast for tomorrow?"}
- {id: d06, bucket: off_domain, expected: decline,
   question: "Explain how a diesel engine works."}
- {id: d07, bucket: off_domain, expected: decline,
   question: "What are the best restaurants in Lisbon?"}
- {id: d08, bucket: off_domain, expected: decline,
   question: "How much should I invest in index funds?"}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --no-sync pytest tests/integration/test_eval_questions.py -v`
Expected: PASS — 8 tests

If `test_near_miss_targets_a_verified_absent_pair` fails, the manifest and the question set disagree.
The manifest is the authority — fix the question, never the manifest.

- [ ] **Step 5: Commit**

```bash
git add backend/evals/questions.yaml backend/tests/integration/test_eval_questions.py
git commit -m "feat: add labelled eval question set

40 questions across four buckets. Every near_miss targets a (drug, axis)
pair the manifest lists as verified absent, enforced by test — the mistake
this phase made twice was labelling a near-miss whose answer was in the
corpus all along."
```

---

## Task 3: Collect pass

**Files:**
- Create: `backend/evals/collect.py`
- Test: `backend/tests/integration/test_eval_signals.py`

**Interfaces:**
- Consumes: `evals.corpus.build_pdf/load_drug/load_manifest`, `documents.ingestion.ingest_document`, `chat.retrieval.retrieve`, `rag.generation.stream_chat/filter_sentinel`, `rag.prompts.build_messages`
- Produces: `evals/signals.json` — a list of records:
  `{id, bucket, expected, drug, axis, top_similarity, mean_similarity, lexical_support, corpus_empty, gate_reason_at_defaults, sentinel_fired, answer, retrieved}`

- [ ] **Step 1: Write `evals/collect.py`**

```python
"""Pass 1 — the expensive one. Runs once per corpus.

For each question: ingest-time retrieval signals, plus the LLM called
UNCONDITIONALLY regardless of what the gate would say at any threshold.

That unconditional call is the point. Without it the sweep cannot answer
"what would stage 2 have done if a lower tau_abstain had let this through?",
which is exactly what the near_miss bucket exists to measure (spec 4.2).
"""
from __future__ import annotations

import json
import pathlib
import tempfile

import django
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "medical_rag.settings")
django.setup()

import yaml  # noqa: E402
from django.core.files.base import ContentFile  # noqa: E402

from chat.retrieval import retrieve  # noqa: E402
from documents.ingestion import ingest_document  # noqa: E402
from documents.models import Document  # noqa: E402
from evals.corpus import FIXTURES, build_pdf, load_drug, load_manifest  # noqa: E402
from rag.config import load_config  # noqa: E402
from rag.embeddings import OllamaEmbedder  # noqa: E402
from rag.generation import filter_sentinel, stream_chat  # noqa: E402
from rag.prompts import build_messages  # noqa: E402
from rag.vectorstore import ChromaStore  # noqa: E402

HERE = pathlib.Path(__file__).parent
QUESTIONS = HERE / "questions.yaml"
SIGNALS = HERE / "signals.json"


def ingest_corpus(store, embedder, cfg) -> int:
    """Build a PDF per drug and ingest it. Returns the chunk total."""
    total = 0
    for slug in load_manifest()["drugs"]:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = build_pdf(pathlib.Path(tmp) / f"{slug}.pdf", slug.title(),
                            load_drug(slug)["included"])
            document = Document.objects.create(title=f"{slug}.pdf")
            document.file.save(f"{slug}.pdf", ContentFile(pdf.read_bytes()), save=True)
        document = ingest_document(document, embedder, store, cfg)
        if document.status != "ready":
            raise SystemExit(f"{slug} failed to ingest: {document.error_message}")
        print(f"  {slug:12} {document.chunk_count:3} chunks")
        total += document.chunk_count
    return total


def run_llm(question: str, chunks, cfg) -> tuple[bool, str]:
    """Call the model and report whether the sentinel fired."""
    messages = build_messages(question, chunks, history=[], max_history=cfg.history_messages)
    collected: list[str] = []
    for kind, text in filter_sentinel(stream_chat(cfg.ollama, messages)):
        if kind == "declined":
            return True, ""
        collected.append(text)
    return False, "".join(collected)


def main() -> None:
    cfg = load_config()
    questions = yaml.safe_load(QUESTIONS.read_text())

    with tempfile.TemporaryDirectory() as chroma_dir:
        store = ChromaStore(path=chroma_dir, collection_name="eval_corpus")
        embedder = OllamaEmbedder(cfg.ollama)

        print("ingesting corpus...")
        total = ingest_corpus(store, embedder, cfg)
        print(f"  {total} chunks total\n")

        records = []
        for i, q in enumerate(questions, start=1):
            result = retrieve(q["question"], embedder, store, cfg)
            signals = result.decision.signals

            # Unconditional: we need stage 2's behaviour even where the gate
            # declined, so the sweep can model a lower threshold letting it by.
            if result.chunks:
                sentinel_fired, answer = run_llm(q["question"], result.chunks, cfg)
            else:
                sentinel_fired, answer = False, ""

            records.append({
                "id": q["id"],
                "bucket": q["bucket"],
                "expected": q["expected"],
                "drug": q.get("drug"),
                "axis": q.get("axis"),
                "top_similarity": signals["top_similarity"],
                "mean_similarity": signals["mean_similarity"],
                "lexical_support": signals["lexical_support"],
                "corpus_empty": signals["corpus_empty"],
                "gate_reason_at_defaults": result.decision.reason,
                "sentinel_fired": sentinel_fired,
                "answer": answer[:400],
                "retrieved": [c.chunk_id for c in result.chunks],
            })
            print(f"  [{i:2}/{len(questions)}] {q['id']} {q['bucket']:19} "
                  f"top={signals['top_similarity']:.4f} lex={signals['lexical_support']!s:5} "
                  f"sentinel={sentinel_fired}")

    SIGNALS.write_text(json.dumps(records, indent=1))
    print(f"\nwrote {SIGNALS} ({len(records)} records)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the collect pass**

Ollama must be running with both models pulled. This takes several minutes.

```bash
cd backend && uv run --no-sync python -m evals.collect
```

Expected: three drugs ingest (roughly 12–20 chunks each, ~45 total), then one line per question, then
`signals.json` written with 40 records.

- [ ] **Step 3: Sanity-check the collected data**

```bash
cd backend && uv run --no-sync python -c "
import json, statistics as st
rs = json.load(open('evals/signals.json'))
print(f'{len(rs)} records')
for b in ('answerable','near_miss','off_corpus_medical','off_domain'):
    sims = [r['top_similarity'] for r in rs if r['bucket']==b]
    lex  = sum(r['lexical_support'] for r in rs if r['bucket']==b)
    sent = sum(r['sentinel_fired'] for r in rs if r['bucket']==b)
    print(f'  {b:19} n={len(sims):2} top_sim {min(sims):.3f}-{max(sims):.3f} '
          f'median {st.median(sims):.3f}  lexical={lex}  sentinel_fired={sent}')
"
```

Expected shape: `answerable` similarities highest, `off_domain` lowest, `sentinel_fired` concentrated
in `near_miss` and `off_corpus_medical`. If `answerable` shows a high `sentinel_fired` count the
corpus or questions are wrong — STOP and report rather than proceeding to the sweep.

- [ ] **Step 4: Pin the sanity check as a real test**

Step 3's eyeball check has genuine stop conditions, so make them assertions. A corrupt collect pass
would otherwise surface only as puzzling sweep numbers, after the expensive pass has already run.

`backend/tests/integration/test_eval_signals.py`:

```python
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
```

Run: `uv run --no-sync pytest tests/integration/test_eval_signals.py -v`
Expected: PASS - 7 tests

If `test_answerable_questions_were_not_refused_by_the_model` fails, STOP: a question labelled
answerable is not answerable from the corpus. Fix the question or the corpus before running the
sweep - do not weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add backend/evals/collect.py backend/evals/signals.json backend/tests/integration/test_eval_signals.py
git commit -m "feat: add eval collect pass with cached signals

Calls the LLM unconditionally per question, regardless of what the gate
would say, so the sweep can model what stage 2 would have done had a lower
threshold let a question through. signals.json is committed so the sweep is
reproducible without Ollama."
```

---

## Task 4: Metrics and sweep

**Files:**
- Create: `backend/evals/metrics.py`, `backend/evals/sweep.py`
- Test: `backend/tests/unit/test_eval_metrics.py`

**Interfaces:**
- Consumes: `evals/signals.json`, `rag.gate.evaluate_gate`, `rag.config.GateConfig`
- Produces: `evals.metrics.OperatingPoint`, `evals.metrics.score_point(records, cfg) -> OperatingPoint`, `evals.metrics.choose_best(points) -> OperatingPoint`; `evals/eval_results.md`

- [ ] **Step 1: Write the failing test**

`backend/tests/unit/test_eval_metrics.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --no-sync pytest tests/unit/test_eval_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evals.metrics'`

- [ ] **Step 3: Write `evals/metrics.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --no-sync pytest tests/unit/test_eval_metrics.py -v`
Expected: PASS — 10 tests

- [ ] **Step 5: Write `evals/sweep.py`**

```python
"""Pass 2 — the cheap one. Replays cached signals across a threshold grid.

Runs in milliseconds because the gate is a pure function, which is what
spec 6.5's no-I/O constraint was for.
"""
from __future__ import annotations

import json
import pathlib

from evals.metrics import OperatingPoint, choose_best, score_point
from rag.config import GateConfig

HERE = pathlib.Path(__file__).parent
SIGNALS = HERE / "signals.json"
RESULTS = HERE / "eval_results.md"

GRID = [round(0.20 + i * 0.05, 2) for i in range(16)]  # 0.20 .. 0.95


def sweep(records: list[dict]) -> list[OperatingPoint]:
    return [
        score_point(records, GateConfig(tau_abstain=a, tau_strong=s))
        for a in GRID
        for s in GRID
        if a < s
    ]


def bucket_table(records: list[dict]) -> str:
    rows = ["| bucket | n | top_similarity range | lexical_support | sentinel fired |",
            "|---|---|---|---|---|"]
    for b in ("answerable", "near_miss", "off_corpus_medical", "off_domain"):
        rs = [r for r in records if r["bucket"] == b]
        if not rs:
            continue
        sims = [r["top_similarity"] for r in rs]
        rows.append(
            f"| `{b}` | {len(rs)} | {min(sims):.3f} – {max(sims):.3f} | "
            f"{sum(r['lexical_support'] for r in rs)}/{len(rs)} | "
            f"{sum(r['sentinel_fired'] for r in rs)}/{len(rs)} |"
        )
    return "\n".join(rows)


def main() -> None:
    records = json.loads(SIGNALS.read_text())
    points = sweep(records)
    best = choose_best(points)
    default = score_point(records, GateConfig())

    top = sorted(points, key=lambda p: (p.false_declines, -p.recall, -p.llm_calls_avoided))[:12]
    rows = ["| tau_abstain | tau_strong | precision | recall | false declines | LLM calls avoided | stage1 | stage2 |",
            "|---|---|---|---|---|---|---|---|"]
    for p in top:
        rows.append(
            f"| {p.tau_abstain:.2f} | {p.tau_strong:.2f} | {p.precision:.2f} | {p.recall:.2f} | "
            f"{p.false_declines} | {p.llm_calls_avoided} | {p.stage1_declines} | {p.stage2_declines} |"
        )

    RESULTS.write_text(f"""# Eval Results

Generated by `evals/sweep.py` from `evals/signals.json` ({len(records)} questions,
{len(points)} operating points). Regenerate with `uv run python -m evals.sweep`.

## Corpus

Three real FDA drug labels (metformin, atenolol, amoxicillin), public domain, pinned by `set_id`.
Near-miss questions target `(drug, axis)` pairs measured absent from the shipped text — see
`fixtures/manifest.json`.

## Signal distribution

{bucket_table(records)}

## Shipped defaults (tau_abstain={default.tau_abstain}, tau_strong={default.tau_strong})

- precision {default.precision:.2f}, recall {default.recall:.2f}
- false declines on answerable questions: **{default.false_declines}**
- LLM calls avoided by stage 1: **{default.llm_calls_avoided}** of {default.should_decline} declines
- near-misses caught: stage 1 {default.near_miss_stage1}, stage 2 {default.near_miss_stage2}

## Best operating points

Ranked by: zero false declines first, then recall, then LLM calls avoided (spec 4.5).

{chr(10).join(rows)}

## Chosen operating point

**`tau_abstain = {best.tau_abstain}`, `tau_strong = {best.tau_strong}`**

- precision {best.precision:.2f}, recall {best.recall:.2f}
- false declines on answerable questions: **{best.false_declines}**
- LLM calls avoided by stage 1: **{best.llm_calls_avoided}**
- near-misses caught: stage 1 {best.near_miss_stage1}, stage 2 {best.near_miss_stage2}

## Caveats

- Three drug labels from one document type is a narrow basis. These thresholds are calibrated for
  this corpus and should be re-measured against any substantially different one.
- {len(records)} questions is a small sample; precision and recall move meaningfully with a handful
  of reclassifications. Raw counts are reported alongside rates for that reason.
- `nomic-embed-text` similarities cluster high — unrelated text bottoms out near 0.37, not near zero,
  which is why the grid starts at 0.20 rather than 0.0.
""")
    print(f"wrote {RESULTS}")
    print(f"chosen: tau_abstain={best.tau_abstain} tau_strong={best.tau_strong} "
          f"recall={best.recall:.2f} false_declines={best.false_declines} "
          f"avoided={best.llm_calls_avoided}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Add a determinism test**

Add to `backend/tests/unit/test_eval_metrics.py`:

```python
def test_sweep_is_a_pure_replay():
    """The same records must produce identical points every run, or the
    committed results are not reproducible."""
    from evals.sweep import sweep

    records = [
        rec("answerable", "answer", 0.90, True, False, "a1"),
        rec("near_miss", "decline", 0.80, True, True, "n1"),
        rec("off_domain", "decline", 0.35, False, False, "d1"),
    ]
    assert sweep(records) == sweep(records)


def test_grid_only_contains_valid_threshold_pairs():
    from evals.sweep import sweep

    points = sweep([rec("answerable", "answer", 0.9, True, False)])
    assert all(p.tau_abstain < p.tau_strong for p in points)
    assert len(points) == 120
```

- [ ] **Step 7: Run the tests**

Run: `uv run --no-sync pytest tests/unit/test_eval_metrics.py -v`
Expected: PASS — 12 tests

- [ ] **Step 8: Commit**

```bash
git add backend/evals/metrics.py backend/evals/sweep.py backend/tests/unit/test_eval_metrics.py
git commit -m "feat: add eval metrics and threshold sweep

Replays cached signals across 120 operating points. The operating point is
ranked lexicographically with zero false declines first, because an
F1-style objective would trade away exactly the answerable questions that
justify the system."
```

---

## Task 5: Run the sweep and adopt the measured thresholds

**Files:**
- Create: `backend/evals/eval_results.md`
- Modify: `backend/rag/config.py` (`GateConfig` defaults), `backend/tests/unit/test_config.py`
- Modify: `docs/superpowers/specs/2026-08-02-medical-rag-design.md` (§14 config table)

**Interfaces:**
- Consumes: everything above
- Produces: committed `eval_results.md`; `GateConfig` defaults derived from measurement

- [ ] **Step 1: Run the sweep**

```bash
cd backend && uv run --no-sync python -m evals.sweep
```

Expected: `eval_results.md` written, and a chosen operating point printed.

- [ ] **Step 2: Read `eval_results.md` and sanity-check the choice**

Confirm before adopting:
- `false declines` at the chosen point is **0**. If no point achieves zero, STOP and report — that
  means some answerable question is not actually answerable, and the question set needs fixing before
  any threshold is adopted.
- The chosen `tau_abstain` is above the highest `off_domain` similarity in the distribution table.
- `LLM calls avoided` is greater than at the shipped defaults. If not, stage 1 is still doing no
  work and that is the headline finding.

- [ ] **Step 3: Update `GateConfig` defaults**

In `backend/rag/config.py`, replace the placeholder block with the measured values.

`<chosen>` below is not a plan placeholder — it is the number this entire phase exists to produce,
and it does not exist until Step 1 runs. Substitute the two values printed by the sweep:

```python
@dataclass(frozen=True)
class GateConfig:
    # Measured by the Phase 3 eval sweep over 40 labelled questions against a
    # three-label FDA corpus. See evals/eval_results.md for the curve these were
    # chosen from and the caveats on how far they generalise.
    tau_abstain: float = <chosen>
    tau_strong: float = <chosen>
```

- [ ] **Step 4: Update the config test**

`backend/tests/unit/test_config.py` asserts `tau_abstain == 0.30` and `tau_strong == 0.45`. Update
both to the measured values. Do NOT delete the assertions — they are what stops the thresholds
drifting silently.

- [ ] **Step 5: Re-run the full suite**

Run: `uv run --no-sync pytest -q`
Expected: PASS. The whole suite must be green with the new defaults.

If a gate test fails, it was asserting behaviour tied to the old thresholds. Read it before changing
it — a genuine behaviour change needs the test updated deliberately, not adjusted until green.

- [ ] **Step 6: Re-run the sweep to confirm determinism**

```bash
cd backend && cp evals/eval_results.md /tmp/first.md && uv run --no-sync python -m evals.sweep && diff /tmp/first.md evals/eval_results.md && echo "DETERMINISTIC"
```

Expected: `DETERMINISTIC` — no diff.

- [ ] **Step 7: Update spec §14**

In `docs/superpowers/specs/2026-08-02-medical-rag-design.md`, the §14 config table lists
`TAU_ABSTAIN / TAU_STRONG` as `0.30 / 0.45` with source **"placeholders until §13 runs"**. Replace
the values with the measured ones and the source with `measured — evals/eval_results.md`.

- [ ] **Step 8: Commit**

```bash
git add backend/evals/eval_results.md backend/rag/config.py backend/tests/unit/test_config.py docs/
git commit -m "feat: adopt measured gate thresholds from the eval sweep

Replaces the placeholder constants with values chosen from 40 labelled
questions across 120 operating points, ranked with zero false declines on
answerable questions first. eval_results.md is committed with the curve and
the caveats on how far these generalise.

Closes the open question named in PRD 17."
```

---

## Verification Checklist

After Task 5, all of the following must hold:

- [ ] `uv run --no-sync pytest` — full suite green with the new defaults
- [ ] `evals/eval_results.md` is committed and shows a chosen operating point with **0 false declines**
- [ ] `evals/signals.json` is committed, so the sweep reproduces without Ollama
- [ ] Re-running `python -m evals.sweep` produces a byte-identical `eval_results.md`
- [ ] `GateConfig` defaults match the chosen point, and `test_config.py` asserts them
- [ ] Spec §14 no longer describes the thresholds as placeholders
- [ ] `test_near_miss_targets_a_verified_absent_pair` passes — no near-miss is secretly answerable
