# Medical RAG — Design Specification

**Date:** 2026-08-02
**Status:** Approved for planning
**Source PRD:** [`docs/MEDICAL_RAG_PRD.md`](../../MEDICAL_RAG_PRD.md)
**Audience target:** full-stack engineering interview; open-ended timeline, portfolio quality

---

## 1. Purpose & Scope

A locally-run RAG chat application scoped to a user's own uploaded medical documents. Everything — LLM, embeddings, vector store — runs on the user's machine. No external API calls, no keys, no data leaving the device.

This document specifies the design agreed during brainstorming. It extends the PRD rather than replacing it: the PRD's stack, non-goals, and responsible-use boundaries all stand. Section 16 lists every deliberate deviation.

**The design centre of gravity is grounding** — guaranteeing that an answer is either supported by uploaded material or is a clean, explicit decline. In a medical context a confidently wrong dosage is materially worse than "I don't know," so the architecture treats refusal as a first-class output path rather than an error case.

### 1.1 Success criteria

Inherited from PRD §2, with two additions:

- [ ] A PDF uploads and reaches `ready` with a chunk count.
- [ ] A question answerable from the uploaded document returns a grounded, streamed answer with a page-level citation.
- [ ] A question outside the uploaded material returns a clean decline, not a hallucination.
- [ ] Zero external API calls or keys.
- [ ] **A near-miss question — same topic, absent fact — declines rather than extrapolating.**
- [ ] **The gate's operating point is chosen from measured data, with the sweep committed to the repo.**

---

## 2. Verified Environment

Confirmed on the target machine 2026-08-02, not assumed:

| Item | State |
|---|---|
| Hardware | Apple M3 Pro, 36 GB unified memory, 18 GPU cores, ~395 GB free |
| Ollama | **v0.32.5, installed and running**, API 200 OK on `127.0.0.1:11434` |
| Ollama CLI on `PATH` | **No** — binary is at `/Applications/Ollama.app/Contents/Resources/ollama` |
| Models pulled | **None** at spec time; `nomic-embed-text` + `llama3.1:8b` pulling |
| Python | System 3.9.6 only — **too old for Django 5.x** |
| `uv` | Available at `~/.local/bin/uv` |
| Node | v26.0.0 |
| Git | Repo initialised for this spec |

Two consequences for Phase 0:

- **Python 3.12 via `uv`.** Django 5.x requires ≥3.10. `uv python install 3.12` and pin with `.python-version`.
- **Ollama CLI needs a `PATH` entry or an absolute path** in any setup script. The server being up does not mean `ollama` resolves in a shell.

The 36 GB / 18-core machine runs an 8B model comfortably with fast streaming, and leaves headroom to compare a larger model later (PRD §15 Phase 4).

### 2.1 Model tag correction

The PRD specifies `llama3.1:8b-instruct`. **That is not a valid Ollama tag** — `llama3.1` instruct builds always carry a quantization suffix (`8b-instruct-q4_K_M`, `8b-instruct-fp16`, …). The bare tag `llama3.1:8b` already resolves to the instruction-tuned `q4_K_M` build. Use `llama3.1:8b`.

---

## 3. Architecture

Two pipelines over three stores, with the interesting logic isolated in a framework-free library.

```
INGESTION
  PDF ──► pypdf (per page) ──► chunk ──► embed ──► ┌─► Chroma   (vectors)
                                                    └─► SQLite   (chunk text + FTS5)

QUERY
  question ──┬─► embed ──► Chroma  ──► [(chunk_id, distance)]  ──┐
             └─► FTS5 MATCH ──────────► [(chunk_id, bm25)]     ──┤
                                                                 ▼
                                                    reciprocal rank fusion
                                                                 ▼
                                                    ┌────────────────────┐
                                                    │ STAGE 1 — gate     │  no LLM call
                                                    └─────────┬──────────┘
                                                   decline ◄──┤──► proceed
                                                                 ▼
                                                    ┌────────────────────┐
                                                    │ STAGE 2 — sentinel │  buffered
                                                    └─────────┬──────────┘
                                                   decline ◄──┤──► stream answer + citations
```

### 3.1 The central boundary

**`rag/` is a pure Python library with zero Django imports.** No models, no `HttpRequest`, no ORM. It accepts text and configuration; it returns chunks, scores, and decisions. Django apps orchestrate it.

This is deliberate and load-bearing:

- The logic worth testing (chunking, fusion, the gate, sentinel detection) tests without a database, a network, or a test client.
- The eval harness drives the same code the request path drives, without booting Django.
- It is the clearest structural evidence in the repo of thinking in boundaries.

The rule is mechanically checkable: a test asserts no module under `rag/` imports `django`.

### 3.2 Store responsibilities

| Store | Owns | Does not own |
|---|---|---|
| **SQLite** | Document/chunk/session/message metadata, **chunk text**, FTS5 lexical index | Vectors |
| **Chroma** | Embeddings keyed by deterministic chunk id, minimal metadata `{document_id, chunk_index}` | Chunk text |
| **Filesystem** | Uploaded PDF originals under `media/` | — |

Inverting the PRD here — chunk text in SQLite rather than Chroma — is what makes hybrid retrieval nearly free (see §6.2).

---

## 4. Repository Structure

```
medical-rag/
├── backend/
│   ├── pyproject.toml            # uv-managed, Python 3.12
│   ├── .python-version
│   ├── manage.py
│   ├── medical_rag/              # settings, urls, asgi
│   ├── documents/                # Document + Chunk models, ingestion views
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── ingestion.py          # orchestrates rag/ + persistence
│   │   └── migrations/           # includes FTS5 setup migration
│   ├── chat/                     # ChatSession + ChatMessage, query view
│   │   ├── models.py
│   │   ├── views.py
│   │   └── streaming.py          # NDJSON frame assembly
│   ├── rag/                      # ◄── pure library, no Django
│   │   ├── config.py             # frozen dataclasses, all tunables
│   │   ├── chunking.py
│   │   ├── embeddings.py         # Ollama /api/embed client
│   │   ├── vectorstore.py        # Chroma adapter
│   │   ├── lexical.py            # FTS5 query builder + sanitiser
│   │   ├── fusion.py             # reciprocal rank fusion
│   │   ├── gate.py               # confidence gate (pure function)
│   │   ├── prompts.py            # system prompt + context assembly
│   │   └── generation.py         # Ollama /api/chat streaming
│   ├── evals/
│   │   ├── questions.yaml        # labelled question set
│   │   ├── run_eval.py           # threshold sweep
│   │   ├── fixtures/             # sample public-domain PDFs
│   │   └── eval_results.md       # committed output
│   └── tests/
│       ├── unit/
│       ├── integration/
│       ├── contract/             # @pytest.mark.ollama, skipped by default
│       └── fake_ollama.py        # stub server for E2E
├── frontend/
│   ├── app/
│   │   ├── page.tsx              # chat
│   │   ├── documents/page.tsx
│   │   └── components/
│   ├── lib/
│   │   ├── api.ts
│   │   └── ndjson.ts             # stream reader
│   ├── e2e/                      # Playwright
│   └── package.json
├── docs/
└── README.md
```

---

## 5. Data Model

Django models. **`Chunk` is new relative to PRD §8**; the rest is as specified there.

```python
class Document(models.Model):
    STATUS_CHOICES = [("processing", "Processing"), ("ready", "Ready"), ("failed", "Failed")]
    title         = models.CharField(max_length=255)
    file          = models.FileField(upload_to="documents/")
    status        = models.CharField(max_length=20, choices=STATUS_CHOICES, default="processing")
    page_count    = models.IntegerField(null=True, blank=True)
    chunk_count   = models.IntegerField(default=0)
    uploaded_at   = models.DateTimeField(auto_now_add=True)
    error_message = models.TextField(blank=True)


class Chunk(models.Model):
    document    = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.IntegerField()
    page_number = models.IntegerField()
    text        = models.TextField()

    class Meta:
        unique_together = [("document", "chunk_index")]
        indexes = [models.Index(fields=["document", "chunk_index"])]

    @property
    def vector_id(self) -> str:
        return f"{self.document_id}_{self.chunk_index}"


class ChatSession(models.Model):
    id         = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title      = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class ChatMessage(models.Model):
    ROLE_CHOICES = [("user", "User"), ("assistant", "Assistant")]
    session           = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role              = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content           = models.TextField()
    retrieved_sources = models.JSONField(default=list, blank=True)
    was_declined      = models.BooleanField(default=False)
    decline_reason    = models.CharField(max_length=32, blank=True)  # off_domain | weak_unsupported | insufficient_context | empty_corpus
    truncated         = models.BooleanField(default=False)           # client disconnect / mid-stream failure
    gate_signals      = models.JSONField(default=dict, blank=True)   # observability + eval replay
    created_at        = models.DateTimeField(auto_now_add=True)
```

`decline_reason` and `gate_signals` exist so every decision is inspectable after the fact — in the Django admin, in the eval harness, and when explaining a surprising decline in a live demo.

### 5.1 FTS5 index

A dedicated migration creates an external-content FTS5 table over `Chunk.text`, kept in sync by triggers:

```sql
CREATE VIRTUAL TABLE chunk_fts USING fts5(
    text,
    content='documents_chunk',
    content_rowid='id',
    tokenize='porter unicode61'
);
-- + AFTER INSERT / AFTER DELETE / AFTER UPDATE triggers on documents_chunk
```

Porter stemming matters for clinical prose (`dosing`/`dose`/`dosage`). FTS5 ships with SQLite and `bm25()` is built in, so the lexical half of hybrid retrieval costs **zero new dependencies** and survives restarts.

*Availability check:* a Phase 0 task asserts `sqlite3` was compiled with FTS5 (`SELECT 1 FROM pragma_compile_options WHERE compile_options='ENABLE_FTS5'`). If absent, fall back to `rank_bm25` held in memory and rebuilt on startup; the `lexical.py` interface is identical either way.

---

## 6. RAG Library

### 6.1 Chunking (`chunking.py`)

Page-aware, so citations are precise:

1. `pypdf` extracts text per page.
2. Each page splits independently into ~1000-character chunks with ~150-character overlap, using a recursive separator cascade (`\n\n` → `\n` → `. ` → ` `).
3. **Chunks never span a page boundary**, so every chunk carries an exact `page_number`.
4. Whitespace-only pages are skipped without consuming a `chunk_index`.

Hand-rolled; no LangChain. The whole splitter is ~60 lines and the mechanics are worth being able to explain.

Signature: `chunk_pages(pages: list[PageText], cfg: ChunkConfig) -> list[ChunkDraft]`.

### 6.2 Retrieval — hybrid, fused by rank

```
question
  ├─ embed("search_query: " + q) ──► Chroma  top-10 ──► [(chunk_id, cosine_distance)]
  └─ sanitise(q) ──► FTS5 MATCH  ──► top-10 ──► [(chunk_id, bm25)]
                                    ▼
                    reciprocal rank fusion, k = 60
                                    ▼
                            top 4 chunk_ids ──► hydrate text from SQLite (single query)
```

**Why hybrid.** Medical text is dense with exact tokens — drug names, dosages, ICD codes — where lexical match outperforms dense similarity. "Metformin" appearing verbatim is a stronger signal than any embedding provides, and embeddings systematically blur near-neighbour drug names.

**Why RRF specifically.** `score = Σ 1/(60 + rank_i)` compares only *positions in two ranked lists*. It never has to reconcile a cosine distance with a BM25 score — two scales with no principled normalization between them. It also sidesteps a real SQLite gotcha: `bm25()` returns **negative** values where more-negative means a better match. Under RRF the sign never enters the arithmetic; we `ORDER BY bm25(chunk_fts)` ascending and use rank position only.

**FTS5 input sanitising is mandatory.** Raw user questions contain characters FTS5 parses as query syntax — quotes, `*`, `-`, `:`, `NEAR`, `AND`. Passing a question straight into `MATCH` raises `fts5: syntax error` on ordinary input like `What's the max dose?`. `lexical.py` tokenises to alphanumeric terms, drops terms under 2 characters, wraps each in double quotes, and joins with `OR`. This has a dedicated unit test with adversarial inputs.

### 6.3 Embeddings (`embeddings.py`)

Ollama `POST /api/embed`, which accepts batched `input` arrays — used to embed a whole document's chunks in batches rather than one HTTP round trip per chunk.

**Task prefixes are required.** `nomic-embed-text` is a prefixed model:

- Indexed chunks: `search_document: {text}`
- Queries: `search_query: {text}`

Embedding both sides unprefixed silently degrades retrieval — no error, just worse results. This is the most commonly skipped detail in `nomic-embed-text` integrations and it is enforced at the API boundary of `embeddings.py` (separate `embed_documents()` and `embed_query()` functions, prefixes applied internally, never passed by callers).

Dimensionality: 768. Asserted in the contract test.

### 6.4 Vector store (`vectorstore.py`)

Chroma `PersistentClient`, single collection `medical_documents`.

**Cosine space must be configured explicitly.** Chroma defaults to **L2 (squared Euclidean)**, not cosine. The PRD's `0.35` threshold is meaningless until the space is pinned, and the two spaces produce different orderings. The collection is created with cosine explicitly, and a test asserts the configured space at startup so a silent default can never take hold.

*Version note:* the configuration API has moved across Chroma releases (`metadata={"hnsw:space": "cosine"}` vs `configuration={"hnsw": {"space": "cosine"}}`). Pin `chromadb` in `pyproject.toml` and confirm the correct form for the pinned version during Phase 1. The startup assertion catches a wrong guess immediately.

With cosine space, `similarity = 1 - distance`, distance ∈ [0, 2].

### 6.5 Confidence gate (`gate.py`) — Stage 1

A pure function. No I/O, no logging, no clock.

```python
@dataclass(frozen=True)
class GateConfig:
    tau_abstain: float = 0.30   # below → clearly off-domain
    tau_strong:  float = 0.45   # above → confident
    top_k:       int   = 4

@dataclass(frozen=True)
class GateSignals:
    top_similarity:  float
    mean_similarity: float
    lexical_support: bool
    corpus_empty:    bool

@dataclass(frozen=True)
class GateDecision:
    proceed: bool
    reason:  str    # ok | off_domain | weak_unsupported | empty_corpus
    signals: dict


def evaluate_gate(s: GateSignals, cfg: GateConfig) -> GateDecision:
    if s.corpus_empty:
        return GateDecision(False, "empty_corpus", ...)
    if s.top_similarity < cfg.tau_abstain:
        return GateDecision(False, "off_domain", ...)
    if s.top_similarity < cfg.tau_strong and not s.lexical_support:
        return GateDecision(False, "weak_unsupported", ...)
    return GateDecision(True, "ok", ...)
```

`lexical_support` is true when the top fused chunk also appeared in the BM25 leg — meaning the question's actual terminology occurs in the retrieved text, not merely something semantically adjacent. This is the signal that distinguishes "metformin dosing" (present) from "metformin **pediatric** dosing" (absent) better than distance alone, because both queries land at nearly identical cosine distance from the same adult-dosing chunk.

**`mean_similarity` is recorded but deliberately not part of the v1 decision rule.** The intuition — one lucky chunk looks different from a genuinely covered topic, so a large gap between top and mean suggests thin support — is plausible but unmeasured. Adding a third threshold on a hunch would mean shipping a constant nobody can defend, which is the exact failure this design is trying to avoid. It is captured in `GateSignals` and persisted in `ChatMessage.gate_signals` so the Phase 3 sweep can test whether it separates the buckets. If it does, it earns a rule then; if not, it stays observability. Same reasoning applies to the deferred graded-overlap score in §17.

The starting threshold values are **guesses**, exactly as PRD §17 warns. The design's contribution is making them *measurable* (§13), not pretending they are known.

### 6.6 Sentinel (`generation.py` + `prompts.py`) — Stage 2

Stage 1 cannot catch the near-miss case where retrieval legitimately succeeds but the retrieved text does not contain the answer. Stage 2 hands that judgement to the model, with a machine-detectable output.

The system prompt instructs: if the context does not contain enough information, reply with exactly `INSUFFICIENT_CONTEXT` and nothing else.

The server **buffers the first 40 characters** of the stream before flushing anything to the client:

- Buffer starts with the sentinel → discard, emit canned decline, `was_declined=True`, `decline_reason="insufficient_context"`.
- Otherwise → flush the buffer, stream the remainder normally.

Buffering is not optional: a sentinel cannot be streamed to the browser and then retracted. The added latency is a handful of tokens and is imperceptible.

**The sentinel may arrive split across two stream deltas** (`INSUFF` + `ICIENT_CONTEXT`). Detection accumulates until it holds ≥ `len(sentinel)` characters or the stream ends. This has an explicit unit test, because it is precisely how this check breaks in practice.

### 6.7 Prompt assembly (`prompts.py`)

Builds on PRD §10's draft, plus the sentinel instruction and citation markers. Context chunks are rendered with an index the model can reference:

```
[1] (Tenormin monograph, p. 3)
{chunk text}

[2] (Tenormin monograph, p. 4)
{chunk text}
```

Recent history capped at the last 4 messages per PRD §10. History is truncated to a character budget as well as a message count, so two long turns cannot crowd out the retrieved context in an 8B model's window.

### 6.8 Decline copy

Declines are generated by the server, never by the model — that is what makes them consistent and testable. Each `decline_reason` maps to distinct copy, because collapsing them into one message destroys the user's ability to tell "I haven't uploaded anything yet" apart from "your question is off-topic":

| `decline_reason` | Copy |
|---|---|
| `empty_corpus` | "No documents have been uploaded yet. Upload a medical reference document and I'll answer questions grounded in it." |
| `off_domain` | "I can only answer questions grounded in the medical documents you've uploaded, and this question doesn't relate to them." |
| `weak_unsupported` | "I found some possibly related material, but not close enough to answer this reliably. Try rephrasing, or upload a document that covers it." |
| `insufficient_context` | "Your uploaded documents cover this topic, but don't contain enough detail to answer this specific question. I'd rather decline than guess." |

`insufficient_context` deliberately names the near-miss situation. In a medical tool, "the document is about this drug but doesn't state that dose" is far more useful to the user than a generic refusal — it tells them the gap is in the source material, not in their phrasing.

Copy lives in `rag/prompts.py` as constants so the eval harness asserts against the same strings the UI renders.

---

## 7. API

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/api/health/` | GET | — | `{ollama_reachable, models: {chat, embed}, documents_ready}` |
| `/api/documents/` | POST | multipart `file` | `{id, title, status, page_count, chunk_count}` (201) |
| `/api/documents/` | GET | — | `[{id, title, status, page_count, chunk_count, uploaded_at, error_message}]` |
| `/api/documents/{id}/` | DELETE | — | 204 |
| `/api/chat/` | POST | `{session_id: str \| null, question: str}` | NDJSON stream |
| `/api/chat/sessions/{id}/messages/` | GET | — | `[{role, content, retrieved_sources, was_declined, decline_reason, created_at}]` |

Plain Django views, no DRF, per PRD §5.

`/api/health/` is beyond the PRD and earns its place: the frontend polls it once on load, so a cold Ollama or a missing model produces *"Ollama isn't running — start it and retry"* instead of a mystery 500 thirty seconds into a live demo.

### 7.1 Streaming protocol — NDJSON

The PRD specifies a plain-text body plus an `X-Session-Id` header. That cannot carry citations, the declined flag, or a mid-stream Ollama failure. One JSON object per line solves all three, still works with `fetch()` + `response.body.getReader()`, and still accepts a POST body — the reason `EventSource` was ruled out in PRD §9 is unaffected.

```
{"type":"meta","session_id":"9f3c…"}
{"type":"sources","items":[{"document_id":1,"title":"Tenormin monograph","page":3,"snippet":"…","score":0.0312}]}
{"type":"token","text":"Met"}
{"type":"token","text":"formin"}
{"type":"done","message_id":42,"was_declined":false,"decline_reason":null,"truncated":false}
```

`done` always carries the same keys; `decline_reason` is `null` on answered turns and one of the §6.8 values on declines. A fixed shape means the client parses one frame type rather than branching on which keys are present.

**Frame order carries meaning.** `meta` is emitted immediately so the client can persist the session id even if the request later fails. **`sources` is emitted only after both gates clear.** Sending citations up front means a Stage-2 decline leaves the UI displaying sources for an answer that never arrives — the exact false-confidence the design exists to prevent.

Decline path:

```
{"type":"meta","session_id":"9f3c…"}
{"type":"token","text":"I can only answer from …"}
{"type":"done","message_id":43,"was_declined":true,"decline_reason":"off_domain"}
```

Failure path:

```
{"type":"error","code":"ollama_unavailable","message":"Ollama is not responding on 127.0.0.1:11434"}
```

`Content-Type: application/x-ndjson`, `X-Accel-Buffering: no`, `Cache-Control: no-cache`.

---

## 8. Frontend

Next.js App Router, client components, per PRD §11.

- **`/`** — chat. Message list, input, incrementally rendered streamed tokens, citation chips under each answer showing document title and page.
- **`/documents`** — upload form, document list with status and chunk count, delete.
- **Persistent disclaimer banner** — PRD §11 wording, always visible.
- **Health gate** — a non-blocking banner when `/api/health/` reports Ollama down or a model missing, naming the exact fix.

Two UI behaviours the protocol makes possible and that carry the design's intent:

- **Declines render as a visually distinct card**, not as an assistant message with no sources. The user should never have to infer from absent citations that the system refused.
- **Citation chips appear only on answered turns**, guaranteed by frame ordering rather than by frontend conditionals.

`lib/ndjson.ts` owns stream decoding: a `TextDecoder` with `{stream: true}`, a line buffer, and per-line `JSON.parse`. **A JSON object can be split across two network chunks**, so the reader buffers until a newline rather than parsing each chunk — a standard bug in hand-written NDJSON readers, and unit-tested here.

---

## 9. Ingestion Flow

Synchronous inside `POST /api/documents/`, per PRD §9. 15 MB cap, checked against `request.FILES["file"].size` before any parsing, returning 413.

1. Validate content type and size → 400/413 on failure.
2. Create `Document(status="processing")`, commit — visible in the list immediately.
3. Extract pages with `pypdf`. **Zero extractable text → `failed`** with a message naming the likely cause (scanned/image-only PDF) and suggesting OCR.
4. Chunk (§6.1).
5. Embed in batches (§6.3).
6. Upsert vectors into Chroma using deterministic ids.
7. In one `transaction.atomic()`: `bulk_create` chunks, set `status="ready"`, `page_count`, `chunk_count`.
8. On any failure in 3–7: `cleanup(document_id)` deletes by `document_id` from Chroma, cascades chunks in SQLite, sets `status="failed"` with the real error.

---

## 10. Consistency Between Two Stores

Chroma and SQLite cannot participate in a shared transaction. The design addresses this explicitly rather than hoping:

- **Deterministic ids** (`{document_id}_{chunk_index}`) make Chroma writes idempotent upserts — a retry after partial failure converges rather than duplicating.
- **Vectors written before the SQLite transaction.** An orphaned vector is invisible to users (retrieval hydrates text from SQLite; an id with no row is dropped from results). An orphaned SQLite row is *worse* — it appears in the document list as `ready` while being unsearchable.
- **Compensating cleanup** on any failure, deleting from both sides.
- **`reconcile_vectors` management command** detects drift in both directions and repairs it — the honest answer to "what happens if it crashes mid-ingest."

---

## 11. Error Handling

Every case below has a defined user-visible outcome. None produce a bare 500.

| Case | Behaviour |
|---|---|
| Scanned / image-only PDF (no extractable text) | `failed`; message names the cause and suggests OCR. Common, and silently produces an empty index otherwise. |
| Encrypted or malformed PDF | `failed` with the parser error surfaced |
| Non-PDF upload | 400 before parsing |
| Upload > 15 MB | 413 before parsing |
| Ollama unreachable at query time | Stage 1 still runs; if it would proceed, `error` frame `ollama_unavailable` |
| Model not pulled | `/api/health/` reports it by name; chat returns `error` frame `model_missing` |
| Ollama dies mid-stream | `error` frame appended after partial content; message persisted with `truncated=True` |
| **No documents uploaded yet** | Short-circuit before retrieval, `decline_reason="empty_corpus"`, distinct copy. Without this, every question declines as `off_domain` and a healthy empty state is indistinguishable from a broken gate. |
| Client disconnects mid-stream | `finally` persists accumulated content with `truncated=True` — closes PRD §17's open item |
| FTS5 syntax error from user input | Cannot occur; input is sanitised (§6.2) and unit-tested |

### 11.1 WSGI, not ASGI — corrected by measurement

Run under **WSGI**. This reverses the original draft of this section, which specified `uvicorn`/ASGI for "well-defined generator teardown on disconnect". Measured against real `llama3.1:8b` during implementation:

| | ASGI (uvicorn) | WSGI |
|---|---|---|
| `meta` frame | 9.48s | **0.01s** |
| first token | 9.48s | **0.72s** |
| token spread | **0.00s** | **2.89s** |

`StreamingHttpResponse` cannot async-iterate a **sync** generator, so under ASGI Django falls back to draining the entire generator in a threadpool before sending a byte — the response is fully buffered and nothing streams. Frame *order* survives that buffering, which is why an end-to-end check that verified ordering rather than timing passed against a broken stream.

The teardown rationale was also backwards: ASGI never delivers `GeneratorExit` to a sync generator, precisely because it drains rather than pauses it. WSGI calls `close()` on client disconnect, so the `finally`-block persistence only works there. ASGI provided neither property it was chosen for.

Views stay **synchronous**, so ORM usage is unchanged. The async alternative — an async generator with an async Ollama client and async ORM — is real but disproportionate for a single-user local app.

`ATOMIC_REQUESTS` stays off (Django's default). A request-wrapping transaction would remain open for the full duration of a stream. Transactions are explicit and narrow.

---

## 12. Testing Strategy

Full pyramid. The pure-library boundary (§3.1) is what lets the base of it be large and fast.

### Unit — `pytest`, no DB, no network

- **Chunking:** page boundaries never crossed; overlap correctness; a page shorter than one chunk; empty pages skipped without consuming an index; `chunk_index` monotonic across pages.
- **Fusion:** RRF against hand-computed rankings; disjoint lists; identical lists; one empty leg.
- **Gate:** table-driven matrix over `(top_similarity, lexical_support, corpus_empty)` covering each `reason` and both sides of each threshold, plus a test pinning the documented v1 behaviour that **`mean_similarity` does not affect the decision** — so if it is later promoted to a rule, that test fails loudly rather than the change passing unnoticed.
- **Sentinel:** clean sentinel; **sentinel split across two deltas**; sentinel-like prefix that is not the sentinel; stream ending mid-buffer.
- **FTS5 sanitiser:** apostrophes, quotes, `*`, `-`, `NEAR`, `AND`, empty-after-sanitising, unicode.
- **Prompt assembly:** history truncation by both message count and character budget.
- **NDJSON framing:** frame order invariants — `sources` never precedes gate clearance.

### Integration — `pytest-django`, real SQLite + real Chroma in `tmp_path`, Ollama faked

- Ingest a fixture PDF end to end; assert chunk count, page numbers, vector count, `ready`.
- Retrieval against a seeded store returns expected ids.
- **Full NDJSON frame sequence for all four paths:** answered, Stage-1 decline, Stage-2 decline, Ollama down.
- Delete cascades both stores; `reconcile_vectors` repairs injected drift.
- Ingestion failure at each step leaves no orphans.

### Contract — `@pytest.mark.ollama`, deselected by default

A handful of tests against **real Ollama**, proving the client speaks the actual API: embedding dimensionality is 768, `/api/embed` batch shape, `/api/chat` streaming delta shape. Without these, a fully mocked suite stays green while the real integration is broken — the standard failure mode of heavy mocking. Run locally and documented in the README; not in CI.

### E2E — Playwright against a **fake Ollama**

`tests/fake_ollama.py` is a ~50-line stub honouring the real API with scripted responses. Running an 8B model in CI is how E2E becomes slow and flaky and then gets deleted.

Two flows:
1. Upload → `ready` → answerable question → tokens stream in → citation chip with page number.
2. "What's the capital of France?" → decline card, **no citation chips**.

---

## 13. Eval Harness

**Not a test.** A measurement tool, and the artifact that converts PRD §17's open question into a defended number.

`evals/questions.yaml` — ~40 labelled questions over the fixture corpus, in four buckets:

| Bucket | Example | Expected |
|---|---|---|
| `answerable` | "What is the adult starting dose of metformin?" | answer |
| `near_miss` | "What is the **pediatric** dose of metformin?" (only adult dosing documented) | decline |
| `off_corpus_medical` | "What are the contraindications for warfarin?" (no warfarin document) | decline |
| `off_domain` | "What's the capital of France?" | decline |

`near_miss` is the bucket that justifies the whole two-stage design. A single-threshold gate scores well on the other three and fails here, which is exactly the point the sweep will demonstrate.

`run_eval.py`:
1. Ingests the fixture PDFs into a throwaway store.
2. Runs every question, recording gate signals and the final decision.
3. Sweeps `tau_abstain × tau_strong` over a grid.
4. Reports, per operating point: **decline precision**, **decline recall**, and **false-decline count on genuinely answerable questions** — the last being the metric a naive threshold search quietly destroys.
5. Writes `eval_results.md` with the table, the chosen operating point, and the reasoning.

Stage-2 behaviour is reported separately, since the sentinel is what should be catching `near_miss` once Stage 1 has let it through.

`eval_results.md` is committed. Fixture PDFs are public-domain reference material only, never real patient data (PRD §13).

---

## 14. Configuration

All tunables in `rag/config.py` as frozen dataclasses, overridable by environment variable, with defaults matching the eval-chosen operating point:

| Setting | Default | Source |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | verified |
| `CHAT_MODEL` | `llama3.1:8b` | §2.1 correction |
| `EMBED_MODEL` | `nomic-embed-text` | PRD §5 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | PRD §10 |
| `RETRIEVE_N` / `TOP_K` | 10 per leg / 4 fused | §6.2 |
| `RRF_K` | 60 | conventional |
| `TAU_ABSTAIN` / `TAU_STRONG` | 0.70 / 0.75 | measured — `evals/eval_results.md` |
| `MAX_UPLOAD_MB` | 15 | PRD §9 |
| `HISTORY_MESSAGES` | 4 | PRD §10 |

---

## 15. Build Phases

Extends PRD §15. Each phase ends in a working, demoable state.

| Phase | Scope |
|---|---|
| **0 — Setup** | `uv` + Python 3.12; Django + Next.js scaffolds; CORS; `uvicorn`; Ollama `PATH` handling; **pull both models**; FTS5 availability check; `/api/health/` |
| **1 — Ingestion** | `Document`/`Chunk` models; FTS5 migration + triggers; chunking; embeddings; Chroma adapter with cosine assertion; upload/list/delete; cleanup + `reconcile_vectors`; documents UI |
| **2 — Retrieval & gate** | FTS5 search + sanitiser; RRF; gate; prompts; Ollama streaming; sentinel buffering; NDJSON view; **backend answer/decline path complete and `curl`-able** |
| **3 — Eval** | Fixture corpus; `questions.yaml`; `run_eval.py`; threshold sweep; commit `eval_results.md`; **set real defaults** |
| **4 — Chat UI** | NDJSON reader; streaming message list; citation chips; decline card; disclaimer; health banner; loading/error states |
| **5 — Hardening** | E2E; contract suite; session replay endpoint; README with architecture narrative |
| **6 — Stretch** | Medical-tuned model comparison (PRD §5 note); larger-model comparison on the 36 GB machine |

**Eval precedes the chat UI deliberately.** `run_eval.py` drives the `rag/` library directly and needs nothing from the frontend, so it is unblocked the moment Phase 2 lands. Building chat UI against unmeasured thresholds is the costly ordering: every odd decline is then ambiguous between a frontend bug and a bad constant, and the thresholds move underneath the UI once the sweep finally runs. Fixing the operating point first gives the frontend stable backend behaviour to build against.

---

## 16. Deviations from the PRD

Every intentional difference, with justification:

| # | PRD | This design | Why |
|---|---|---|---|
| 1 | Single distance threshold | Two-stage: multi-signal gate + sentinel | Catches near-miss questions, which a single threshold structurally cannot. Chosen in brainstorming. |
| 2 | Pure vector retrieval | Hybrid BM25 + vector, RRF-fused | Medical text is exact-token-dense; lexical match is a strong complementary signal |
| 3 | Chunk text in Chroma | Chunk text in SQLite; Chroma holds vectors only | Makes FTS5 hybrid retrieval dependency-free; exact-snippet citations without a Chroma round trip; SQL cascade on delete |
| 4 | One `chat` app | `documents` + `chat` apps + framework-free `rag/` library | Cleaner seams; the interesting logic tests without Django |
| 5 | Plain-text stream + `X-Session-Id` | NDJSON frames | Plain text cannot carry citations, the declined flag, or mid-stream errors |
| 6 | (unspecified) | Chroma cosine space configured explicitly | Chroma defaults to L2; the PRD's 0.35 is undefined without pinning the space |
| 7 | (unspecified) | `search_document:` / `search_query:` prefixes | `nomic-embed-text` requires them; omitting them silently degrades retrieval |
| 8 | (unspecified) | `/api/health/` | Turns a cold-Ollama demo failure into an actionable message |
| 9 | `llama3.1:8b-instruct` | `llama3.1:8b` | Not a valid Ollama tag; bare tag is already instruction-tuned |
| 10 | WSGI dev server implied | `uvicorn` (ASGI), sync views | Well-defined generator teardown on disconnect |
| 11 | §17: disconnect loses message | `finally`-block persistence, `truncated` flag | Closes a stated open item |
| 12 | No testing section | Full pyramid + eval harness | Requested; the eval harness resolves §17's threshold question |

Unchanged from the PRD: stack, non-goals, chunk sizing, `top_k=4`, history cap, 15 MB limit, synchronous ingestion, single-user scope, responsible-use boundaries.

---

## 17. Risks & Open Questions

**Carried from the PRD:**

- **Thresholds are measured, on a narrow corpus.** Phase 3 replaced the placeholders with `0.70 / 0.75`, chosen from 120 operating points over 40 labelled questions with zero false declines (`evals/eval_results.md`). The residual risk moved rather than closed: three FDA labels of one document type is a thin basis, and `tau_strong` was never binding on that corpus, so it rests on no evidence at all. Re-measure against any substantially different corpus.
- **No single layer is airtight.** Two stages reduce failure rate; they do not eliminate it. An 8B model can still stray. Worth stating plainly in an interview rather than overselling.

**New to this design:**

- **The sentinel depends on instruction-following.** An 8B model may occasionally answer instead of emitting `INSUFFICIENT_CONTEXT`. Stage 1 remains the stronger layer; Stage 2 is defence in depth. The eval harness measures how often Stage 2 fires correctly, so the claim stays evidence-backed.
- **`lexical_support` is a coarse boolean, and the sweep showed it is not binding.** Every one of the 40 questions that cleared `tau_abstain` also had lexical support, so the middle band never ruled on a single case and a graded score would have changed nothing. That is a fact about this corpus — FTS5 finds *some* term overlap for almost any medically-phrased question against medical text — not a general finding. Still deferred, now for a measured reason.
- **Chroma's configuration API has shifted across versions.** Mitigated by pinning and a startup assertion.
- **FTS5 compilation is assumed, not guaranteed.** Phase 0 checks; `rank_bm25` is the fallback behind an identical interface.
- **RRF discards score magnitude**, which is what feeds `top_similarity`. The gate therefore reads similarity from the *vector leg directly*, not from fused output — a subtle ordering constraint in the query path worth calling out during implementation.
