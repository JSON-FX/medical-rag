# Implementation Notes — Backend (Phases 0–2)

Findings worth keeping from building the backend. The spec says what the system *is*;
this says what building it *taught*, including where the spec was wrong.

Every item here was found by review or measurement, not by the tests passing.

---

## The bugs that mattered

### Streaming didn't stream, and the end-to-end test passed anyway

The spec chose ASGI (`uvicorn`) for "well-defined generator teardown on client disconnect."
`StreamingHttpResponse` cannot async-iterate a **sync** generator, so Django fell back to draining
the entire generator in a threadpool before sending a byte.

| | ASGI | WSGI |
|---|---|---|
| `meta` frame | 9.48s | 0.01s |
| first token | 9.48s | 0.72s |
| token spread | **0.00s** | 2.89s |

The end-to-end check passed because it verified frame **order**, and order survives buffering.
It took timing the frames to see it.

The original rationale was also backwards: ASGI never delivers `GeneratorExit` to a sync generator
precisely *because* it drains rather than pauses it. WSGI calls `close()` on disconnect. ASGI
provided neither property it was chosen for. See spec §11.1.

### "What happens when Ollama isn't running?" returned a stack trace

`retrieve(...)` was called one line *above* the response generator, outside any `try`. With Ollama
down the endpoint returned **HTTP 500 with a Django traceback**, while the spec's error table
promised `{"type":"error","code":"ollama_unavailable"}`. It also left an orphan user message that
session replay would show forever.

Sixteen task reviews missed it because each looked *inside* the generator. Only the whole-branch
review looked above it. This is the first question anyone asks a local-LLM app.

### The confidence gate failed open on NaN

`nan < threshold` is `False` for every threshold, so a NaN similarity fell through both declining
branches and reached `proceed=True, reason="ok"` — the most permissive outcome from the most
degenerate input, in the one component whose job is to decline when uncertain. A zero-norm embedding
produces exactly that (0/0). Now fails closed.

### The lexical tokenizer couldn't tell 0.5mg from 5mg

`\w+` split `"0.5mg"` into `["0", "5mg"]` and dropped the orphaned `"0"` as too short, so
`build_fts_query("0.5mg")` and `build_fts_query("5mg")` produced **byte-identical** queries. Against
a real corpus both returned the same rows in the same order — a pediatric dose lexically
indistinguishable from an adult one.

This is the clearest answer to "why is medical RAG different from RAG," and it lived in the least
glamorous file in the repo.

### A safety signal decided by string sort

`lexical_support` was `top_ids[0] in lexical_ids`. When the two retrieval legs disagree completely,
RRF gives both rank-1 items the identical score `1/(k+1)`, so the fused winner breaks on `chunk_id`
lexicographic order. Measured: `vector=['1_0']/lexical=['2_0']` → `False`; the structurally identical
`vector=['2_0']/lexical=['1_0']` → `True`.

Neither component was wrong alone. RRF *should* break ties deterministically or the threshold sweep
isn't reproducible. The gate *should* ask whether the question's terminology appears in the retrieved
text. The defect was that "deterministic" meant "sorted by id," and that arbitrary ordering became
the answer to a semantic question. Now an intersection over the delivered chunks.

### The repair tool was more destructive than the damage

`reconcile_vectors --fix` deleted by `document_id`, so a document with two healthy chunks and one
orphaned vector lost all three — converting wasted disk space into a `ready`-but-unsearchable
document, the worse of the two drift states it exists to detect. It then reported success, because
the damage was computed before the loop that caused it. A malformed id also aborted the run
permanently, since retrying hit the same crash forever.

Both dissolved by deleting the specific orphaned **ids** instead of whole documents, which also
removed the id parsing that caused the crash.

### A failed re-ingest destroyed a working document

Cleanup ran before the failure-prone steps had succeeded, so a `ready` document re-ingested during a
transient Ollama outage ended `failed` with zero chunks. A network blip cost the user a working,
searchable document. Cleanup now runs only after embedding succeeds; a previously-ready document
survives and keeps its error message.

---

## What the environment actually did, versus what was assumed

- **Chroma defaults to L2, not cosine** — confirmed behaviourally: opposite unit vectors sit at
  distance 4.0 under L2 and 2.0 under cosine. Every gate threshold assumes cosine. The constructor
  asserts the configured space, and a test pre-creates an L2 collection to prove the assertion fires.
- **chromadb 1.5.9 rejects collection names under 3 characters**, and its `query()` tolerates both an
  empty collection and an over-large `n_results`. The plan's defensive clamp guarded failures that
  don't exist while *creating* the only real one (`n_results=0`, which Chroma does reject). Removing
  the defence fixed the bug.
- **FTS5 auxiliary functions don't resolve table aliases** — `bm25(f)` raises `no such column: f`;
  only the literal table name works.
- **SQLite's `bm25()` returns negative values**, more-negative being better. Rank fusion never
  compares raw scores, so the sign never enters the arithmetic — which is a large part of why rank
  fusion was chosen over score blending.
- **`nomic-embed-text` requires task prefixes.** Omitting `search_document:` / `search_query:`
  produces no error, just worse retrieval. Verified against the live model: identical text embedded
  as document vs query yields different vectors (cosine 0.8744).
- **Django's `bulk_create` does fire SQL triggers**, so chunks written in bulk reach the FTS index.
  Worth confirming rather than assuming — silently unindexed chunks would be invisible to lexical
  search forever.

---

## Measured threshold calibration (input to Phase 3)

Real similarities against a small corpus, `nomic-embed-text`:

| bucket | top_similarity | lexical_support | gate says |
|---|---|---|---|
| answerable | 0.8768 / 0.9212 | ✅ | ok |
| near-miss | 0.7958 | ✅ | ok → stage 2 caught it |
| off-corpus medical | 0.5884 | ❌ | ok |
| off-domain | 0.4552 / 0.4236 / 0.3729 | ❌ | ok / weak_unsupported |

**`tau_abstain = 0.30` sits below every question in the set**, so the stage-1 `off_domain` branch
never fires and every off-topic question pays for a full LLM call. All three declines in the live
demo came from stage 2, not the gate. `nomic-embed-text` similarities cluster far higher than the
placeholder assumed — unrelated text bottoms out near 0.37, not near zero.

Separation is nonetheless clean, and `lexical_support` is the sharpest signal in the table: `True`
for every answerable question, `False` for everything off-corpus. An operating point near
`tau_abstain ≈ 0.50 / tau_strong ≈ 0.75` would let 4 of 7 questions skip the LLM entirely.

**The constants were deliberately left unchanged.** Seven questions against a two-chunk corpus is a
preview, not a measurement, and setting constants from it is exactly the unmeasured guess this design
exists to avoid. Phase 3's labelled sweep owns this.

---

## Known gaps

- **`lexical_support` has no red→green integration test.** The fix is verified at the formula level,
  but a small corpus makes the vector leg return everything, so the legs can't be forced disjoint
  through `retrieve()`. Pinning it needs either a corpus larger than `per_leg` or a unit test over the
  signal computation with hand-built leg lists.
- **Stage-1 decline has no disconnect persistence.** The window is between two adjacent statements
  with no I/O and the lost artifact is canned copy, so it was judged not worth destabilising the
  generator that had just been restructured.
- **`chat/views.py` retrieval-failure handler catches bare `Exception`**, so a non-Ollama failure is
  labelled `ollama_unavailable`, and it forwards `str(exc)` unredacted where the documents endpoint
  redacts. Acceptable under the stated localhost-only scope; worth tightening.
- **`rag/ollama.py` and `rag/generation.py` still hardcode timeouts** while `embeddings.py` reads
  config. The streaming one legitimately needs a longer budget than `request_timeout_s`, so it wants
  its own field rather than a copy-paste fix.
- **Test fixtures write real PDFs into `backend/media/`** and accumulate; DB rollback doesn't undo
  disk writes. Wants a `MEDIA_ROOT` override fixture.
- **No citation validation.** A model emitting `[3]` when two chunks were supplied renders a dangling
  citation.
