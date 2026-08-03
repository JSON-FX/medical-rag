# Phase 3 — Eval Harness Design

**Date:** 2026-08-03
**Status:** Approved for planning
**Extends:** [`2026-08-02-medical-rag-design.md`](2026-08-02-medical-rag-design.md) §13

This is an addendum, not a replacement. §13 settles the structure — four buckets, a threshold
sweep, a committed `eval_results.md`. This document settles the two things §13 left open: where the
corpus comes from, and how the sweep runs without taking hours.

---

## 1. Why this phase exists

PRD §17 names the unmeasured `DISTANCE_THRESHOLD` as the project's biggest open question. Phase 2
shipped placeholder constants and a preview measurement that showed them to be wrong:

| bucket | top_similarity | lexical_support | gate says |
|---|---|---|---|
| answerable | 0.8768 / 0.9212 | ✅ | ok |
| near-miss | 0.7958 | ✅ | ok → stage 2 caught it |
| off-corpus medical | 0.5884 | ❌ | ok |
| off-domain | 0.4552 / 0.4236 / 0.3729 | ❌ | ok / weak_unsupported |

`tau_abstain = 0.30` sits below every question in that set, so the stage-1 `off_domain` branch never
fires and every off-topic question pays for a full LLM call. That preview was seven questions against
a two-chunk corpus — enough to prove the constants wrong, not enough to set new ones. This phase
replaces it with a measurement large enough to justify a number.

**Success criterion:** `GateConfig`'s defaults are chosen from committed measured data, and
`eval_results.md` shows the curve they were chosen from.

---

## 2. Corpus

### 2.1 Source

Three real FDA drug labels from openFDA. US federal government works are public domain, and these are
genuine drug monographs — the document type the PRD names. No clinical content is authored for this
project; fabricated doses in a medical repository are a hazard even in a demo, and an eval measured
against invented text measures nothing.

Labels are pinned by `set_id` so a re-fetch returns the same document:

| Drug | `set_id` | source chars |
|---|---|---|
| Metformin hydrochloride | `011de1a5-1ac0-4831-9e8d-26ec79ba2205` | 11,549 |
| Atenolol | `09b21985-1818-449d-9b29-98f733cf7b9f` | 18,924 |
| Amoxicillin | `00fbd46e-05fd-4f8a-9f59-a7a4d01c8e54` | 16,766 |

Three drugs from three distinct classes (antidiabetic, beta blocker, antibiotic) so that off-corpus
questions about an unrelated drug are unambiguously off-corpus.

### 2.2 Section selection — the mechanism that makes near-misses principled

**This is the load-bearing decision of the phase.**

The original §13 near-miss example assumed "only adult dosing documented." Real FDA labels have a
`pediatric_use` section. Metformin's runs 1036 characters and opens *"The safety and effectiveness of
metformin hydrochloride tablets for the treatment of type 2 diabetes mellitus…"* — so against a full
label, "what is the pediatric dose of metformin?" **is** answerable, and labelling it `near_miss`
would have silently corrupted the most important bucket in the eval.

The fixture builder therefore selects sections, and records what it selected:

**Included — the documented surface**
- `indications_and_usage`
- `dosage_and_administration`
- `contraindications`
- `adverse_reactions`
- `drug_interactions`

**Withheld — the near-miss surface**
- `pediatric_use`
- `overdosage`
- `pregnancy`

Each withheld section is a near-miss axis across all three drugs: nine questions whose absence from
the corpus is a **property of the build**, not an assumption about the source. The builder writes
`fixtures/manifest.json` recording per drug which sections were included, which were withheld, and
their character counts, so any label in `questions.yaml` can be audited against it.

A near-miss question is valid only if it targets a withheld section of a drug that IS in the corpus.
That is checkable, and §5 makes it a test.

### 2.3 Reproducibility — one network boundary

`fetch_fixtures.py` is the only component that touches the network. It runs once, writes raw JSON,
extracted per-drug text, and the manifest — all committed. `run_eval.py` needs no network except
Ollama.

The committed text is the provenance: anyone can see exactly what the eval measured without
re-fetching, and openFDA changing upstream cannot silently move the results.

Expected corpus: ~42,000 characters → **~45 chunks** at 1000/150. Comfortably above `per_leg = 10`,
which also makes it the corpus that can finally pin the `lexical_support` regression noted as a gap
in the Phase 2 notes.

---

## 3. Question set

`evals/questions.yaml`, roughly 40 questions across the four §13 buckets:

| Bucket | Construction | Expected | Target |
|---|---|---|---|
| `answerable` | Facts in an included section | answer | ~14 |
| `near_miss` | Facts in a **withheld** section of a corpus drug | decline | ~9 |
| `off_corpus_medical` | Real medical questions about drugs not in the corpus (warfarin, insulin, …) | decline | ~9 |
| `off_domain` | Non-medical entirely | decline | ~8 |

Each entry carries `id`, `bucket`, `question`, `expected` (`answer` \| `decline`), and — for
`answerable` and `near_miss` — the `drug` and `section` it targets, which is what makes §5's
validation possible.

Questions are authored by hand against the committed fixture text. They are not generated, because a
generated question set would be measuring the generator.

---

## 4. Sweep architecture

### 4.1 The observation that makes it fast

**The raw behaviour of both stages is threshold-independent.** Retrieval signals depend on the
question and the corpus. Whether the model emits the sentinel depends on the question and the
retrieved chunks. Only the gate's *decision rule* consumes `tau_abstain` and `tau_strong`.

So the expensive work is done once and the grid replays it.

### 4.2 Pass 1 — collect (expensive, once per corpus)

For each question: embed, retrieve, record `GateSignals`, and **call the LLM unconditionally** —
regardless of what the gate would say at any particular threshold — recording whether the sentinel
fired and what the answer was.

The unconditional call is essential. Without it the sweep cannot answer *"what would stage 2 have
done if a lower `tau_abstain` had let this question through?"*, which is precisely the question the
near-miss bucket exists to ask.

Writes `evals/signals.json`: one record per question with signals, sentinel outcome, retrieved chunk
ids, and the model's answer. Committed, so the sweep is reproducible without Ollama.

Cost: ~40 LLM calls, a few minutes.

### 4.3 Pass 2 — sweep (cheap, thousands of times)

Replay the cached signals through `evaluate_gate` across a `tau_abstain × tau_strong` grid
(0.20…0.95, step 0.05, `tau_abstain < tau_strong`). Pure function, no I/O, no network — this is
exactly what §6.5's purity constraint was for.

For each operating point, simulate the full two-stage outcome:
- gate declines → decline, attributed to **stage 1**
- gate proceeds and the cached sentinel fired → decline, attributed to **stage 2**
- gate proceeds and no sentinel → answer

### 4.4 Metrics

Per operating point:

- **decline precision** — of questions declined, how many should have been
- **decline recall** — of questions that should decline, how many did
- **false declines on answerable questions** — the metric a naive threshold search quietly
  destroys, reported separately for that reason
- **LLM calls avoided** — how many questions stage 1 rejected without invoking the model. This is
  the efficiency claim from PRD §6, currently unrealised at `tau_abstain = 0.30`.
- **stage attribution** — how many near-misses stage 1 caught versus stage 2

### 4.5 Choosing the operating point

Ranked lexicographically, not by a blended score:

1. **Zero false declines on `answerable`.** A system that refuses questions it can answer is broken
   in the way users notice first.
2. Maximise decline recall.
3. Break ties by LLM calls avoided.

Stated explicitly because "maximise F1" would happily trade away answerable questions, and in this
system those are the ones that justify its existence.

---

## 5. Testing

The harness is a measurement tool, not production code, but three properties are worth pinning:

- **Question-set validity** — every `near_miss` targets a section the manifest records as withheld,
  for a drug that is in the corpus; every `answerable` targets an included section. A near-miss that
  is secretly answerable would corrupt the headline result silently.
- **Sweep determinism** — the same `signals.json` produces byte-identical `eval_results.md`. The
  sweep must be a pure replay.
- **Metric arithmetic** — precision/recall computed against a hand-built confusion matrix.

Contract-marked (`@pytest.mark.ollama`) coverage is deliberately absent: pass 1 already exercises the
real stack, and the existing contract suite covers the clients.

---

## 6. Deliverables

| Artifact | Committed | Purpose |
|---|---|---|
| `evals/fetch_fixtures.py` | ✅ | One-time openFDA fetch, pinned by `set_id` |
| `evals/fixtures/*.json`, `*.txt` | ✅ | Raw labels and extracted text — the provenance |
| `evals/fixtures/manifest.json` | ✅ | Which sections were included/withheld per drug |
| `evals/questions.yaml` | ✅ | ~40 hand-authored labelled questions |
| `evals/collect.py` | ✅ | Pass 1 — signals + sentinel outcomes |
| `evals/signals.json` | ✅ | Cached pass-1 output |
| `evals/sweep.py` | ✅ | Pass 2 — grid, metrics, operating-point choice |
| `evals/eval_results.md` | ✅ | **The artifact this phase exists to produce** |
| `rag/config.py` | modified | `GateConfig` defaults replaced with measured values |

---

## 7. Risks

- **The chosen operating point may be corpus-specific.** Three drug labels from one document type is
  a narrow basis; the numbers will not generalise to arbitrary medical corpora. `eval_results.md`
  must say so rather than presenting the thresholds as universal.
- **Stage 1 may turn out to contribute little.** If the sweep shows most near-misses caught by stage
  2, that is the honest result and gets reported. It would still leave stage 1 doing real work on
  off-domain questions, which is where its efficiency claim lives.
- **`nomic-embed-text` similarities cluster high** — unrelated text bottoms out near 0.37, not near
  zero. The grid starts at 0.20 rather than 0.0 for that reason, but the useful range may prove
  narrower still.
- **~40 questions is a small sample.** Precision and recall will move meaningfully with a handful of
  reclassifications. `eval_results.md` reports raw counts alongside the rates so the reader can see
  the sample size behind each number.
