# Medical RAG

A local-only Retrieval-Augmented Generation system for asking questions against medical reference
PDFs you upload yourself. Django + Next.js + Ollama + Chroma + SQLite/FTS5. Nothing in the request
path calls a hosted API — the documents, the embeddings, and the model all stay on your machine.

Upload a PDF, ask a question, and get an answer grounded in what you uploaded with a citation to the
page it came from — or a decline, when the answer genuinely isn't in there.

> **For informational reference only — not a substitute for professional medical judgment.**
> Do not put real patient data (PHI) into this. See [Scope and limits](#scope-and-limits).

---

## The idea: "I don't know" is a feature

The central bet is that the model should never answer from its own general knowledge. Every answer
must trace back to a retrieved chunk of a document you uploaded, and refusing has to be a normal,
common outcome rather than a failure mode. That's enforced twice:

**Stage 1 — before the LLM runs.** Hybrid retrieval (cosine similarity over local embeddings, fused
with SQLite FTS5/BM25 lexical search via reciprocal rank fusion) produces a `top_similarity` score
and a `lexical_support` signal. A confidence gate declines outright — `off_domain`,
`weak_unsupported`, or `empty_corpus` — without ever spending an LLM call. This is what makes an
obviously off-topic question cheap.

**Stage 2 — inside the LLM call.** The system prompt says that if the retrieved context doesn't
support an answer, the entire response must be one exact sentinel token. The server buffers the
start of the stream, detects that sentinel server-side, and substitutes its own decline copy — so
neither the raw token nor an ungrounded answer ever reaches the browser. Decline wording is always
server-authored, never model-authored, which keeps it consistent and testable.

Every gate decision is persisted per message and inspectable in Django's admin.

### The thresholds are measured, not guessed

The gate's two thresholds were originally hand-picked. An eval harness (`backend/evals/`) replaced
them with measured values: it builds a corpus from three real FDA drug labels, asks 40 hand-labelled
questions across four buckets, and sweeps 120 threshold pairs over cached retrieval signals.

The finding that mattered wasn't the numbers — it was that precision and recall were already a
perfect 1.00 at the old thresholds, while stage 1 was declining only 2 of 26 questions it should
have. The other 24 off-topic questions each paid for a full LLM call before stage 2 caught them. The
gate was correct all along; it just wasn't doing the job it exists for. Measured, that goes to 14
of 26 with zero false declines.

See [`backend/evals/eval_results.md`](backend/evals/eval_results.md) for the curve and its caveats —
including the honest one, that `tau_strong` is not constrained by this corpus at all.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Backend | Django, plain views (no DRF) | A handful of endpoints don't justify a serializer layer; free admin for inspecting gate decisions |
| Frontend | Next.js App Router, Tailwind, shadcn/ui | Streams NDJSON straight from Django — no proxy hop |
| LLM runtime | Ollama, local | No API keys, no data leaves the machine |
| Chat model | `llama3.1:8b` | Reliable instruction-following at laptop size |
| Embeddings | `nomic-embed-text` (768-dim) | Small, fast, purpose-built for RAG |
| Vector store | Chroma, embedded `PersistentClient` | Zero infra — no server or Docker to run |
| Lexical search | SQLite FTS5 / BM25 | Catches exact terminology cosine similarity blurs |
| PDF parsing | `pypdf` | Dependency-light |

---

## Installation

### Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — the repo pins Python 3.12; `uv` fetches it automatically.
- **[Node.js](https://nodejs.org)** 20 or newer, with npm.
- **[Ollama](https://ollama.com)** running locally, with both models pulled:
  ```bash
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ```
- **SQLite with FTS5** — bundled with Python's stdlib `sqlite3` on macOS and most Linux distros.

### Set up

```bash
git clone https://github.com/JSON-FX/medical-rag.git
cd medical-rag
```

Backend:

```bash
cd backend
uv run python manage.py migrate
```

Frontend:

```bash
cd frontend
npm install
```

### Run it

Ollama must be running first. Then start the backend, in its own terminal:

```bash
cd backend && uv run python manage.py runserver
```

And the frontend, in another:

```bash
cd frontend && npm run dev
```

Open **http://localhost:3000**. Upload a PDF on the Documents page, then ask about it on the Chat
page. If something isn't wired up, the health banner at the top tells you which of the three things
is wrong — Ollama not running, a model not pulled, or no documents uploaded yet.

> The backend is served under **WSGI**, deliberately. Do not point an ASGI server at it —
> `StreamingHttpResponse` can't async-iterate a sync generator, so ASGI silently drains the whole
> response before sending a byte (9.5s to first token, versus 0.7s). See
> [`backend/README.md`](backend/README.md#wsgi-not-asgi--this-is-load-bearing-not-a-style-preference).

### Try it without a PDF of your own

The eval harness can generate a real FDA drug label as a PDF:

```bash
cd backend
uv run python -c "
import pathlib
from evals.corpus import build_pdf, load_drug
build_pdf(pathlib.Path('metformin.pdf'), 'Metformin', load_drug('metformin')['included'])
print('wrote metformin.pdf')
"
```

Upload that, then try "What is the adult starting dose of metformin?" (answers, with a citation)
against "What should be done in the event of a metformin overdose?" (declines — that section is
deliberately withheld from the fixture) and "What is the capital of France?" (declines before the
model is ever called).

---

## API

Mounted under `/api/` on port 8000:

| Endpoint | Purpose |
|---|---|
| `GET /api/health/` | Ollama reachability, which models are present, documents ready |
| `GET/POST /api/documents/` | List, or upload (multipart, field `file`, PDF only, 15 MB) |
| `DELETE /api/documents/<id>/` | Delete a document and its vectors |
| `POST /api/chat/` | NDJSON stream: `meta` → `sources`? → `token`* → `done` |
| `GET /api/chat/sessions/<id>/messages/` | Replay a session |

A `sources` frame arrives **if and only if** the turn will be an answer — both decline paths emit
none. That invariant is what lets the UI render a refusal as a refusal from its first character,
rather than showing it as an answer and restyling when the stream ends.

---

## Tests

```bash
cd backend && uv run pytest -q      # 261 passed, 3 deselected
cd frontend && npm test             # 31 passed
```

The 3 deselected are contract tests that need a live Ollama; run them with `uv run pytest -m ollama`.

Frontend tests cover the two modules with real logic — NDJSON frame reassembly across stream chunk
boundaries, and the pure reducer that turns frames into chat state. End-to-end browser coverage is
not here yet.

---

## Scope and limits

This is a single-user local tool, not a deployable service. Read this before pointing it at anything
real:

- **No authentication anywhere.** Every endpoint, including document upload and delete, is open to
  whoever can reach the port. It is built for one person talking to `127.0.0.1`.
- **`DEBUG` is on by default.**
- **Do not put real patient data (PHI) into this.** It has not been built or reviewed against HIPAA
  or any other compliance standard. Use public or synthetic reference documents.
- **Answers are informational reference only** — never a substitute for professional medical
  judgment. The system prompt tells the model as much, and the UI says so on every page.
- **Thresholds are calibrated to a narrow corpus** — three FDA labels of one document type. A very
  different corpus may need them re-measured.
- **No single layer is airtight.** Two stages reduce the failure rate; they don't eliminate it. An
  8B model can still occasionally stray.

---

## Repository layout

```
backend/          Django: ingestion, hybrid retrieval, gate, streaming chat
  rag/            Pure library — chunking, embeddings, fusion, gate, prompts
  evals/          Eval harness: FDA fixture corpus, labelled questions, threshold sweep
frontend/         Next.js chat and documents UI
docs/             PRD, design specs, implementation plans, and notes
```

[`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md) is the most useful document here if
you want the honest version: what building this taught, including where the spec turned out to be
wrong. The streaming bug whose test passed anyway, the confidence gate that failed open on NaN, and
the lexical tokenizer that couldn't tell `0.5mg` from `5mg` are all in there.

## License

No license specified — all rights reserved. The FDA label text under `backend/evals/fixtures/` is
US federal government work and in the public domain.
