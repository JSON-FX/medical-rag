# Medical RAG Backend

A local-only Retrieval-Augmented Generation backend for asking questions against medical reference
PDFs you upload yourself. Django + SQLite (FTS5 for lexical search) + Chroma (vector search) +
Ollama (local embeddings and local chat generation — both models run on the same machine). Nothing
in the request path calls a hosted API: upload a PDF, ask a question, get an answer grounded in what
you uploaded, or a decline if the answer isn't actually in there.

## Two-stage grounding

The design's central bet is that the model should never be allowed to answer from its own general
knowledge — every answer must trace back to a retrieved chunk of an uploaded document, and "I don't
know" has to be a normal, common outcome rather than a failure mode. That is enforced twice:

- **Stage 1, before the LLM ever runs:** hybrid retrieval (cosine similarity over Ollama embeddings,
  fused with SQLite FTS5/BM25 lexical search via reciprocal rank fusion) produces a `top_similarity`
  and a `lexical_support` signal. A confidence gate (`rag/gate.py`) uses two thresholds
  (`tau_abstain`, `tau_strong`) to decline outright — `off_domain`, `weak_unsupported`, or
  `empty_corpus` — without spending an LLM call. This is what makes an obviously off-topic question
  cheap.
- **Stage 2, inside the LLM call:** the system prompt instructs the model that if the retrieved
  context doesn't support an answer, its entire response must be one exact sentinel token
  (`INSUFFICIENT_CONTEXT`). The server buffers the start of the stream, detects the sentinel
  server-side (tolerating a short preamble a small instruct model may add anyway), and substitutes
  its own decline copy instead of ever letting the raw token — or an ungrounded answer — reach the
  client. Decline wording is always server-authored, never model-authored, so it stays consistent
  and testable.

Every gate decision (`was_declined`, `decline_reason`, and the raw `gate_signals`) is persisted on
the `ChatMessage` row and visible in `/admin/`.

The gate's two thresholds (`rag/config.py`) are measured, not guessed: `evals/` builds a corpus from
three real FDA labels, asks 40 labelled questions, and sweeps 120 threshold pairs over the cached
signals — see [`evals/eval_results.md`](evals/eval_results.md) for the curve and its caveats. The
sweep's own finding was that the previous hand-picked values left stage 1 declining almost nothing,
so 24 of 26 off-topic questions still cost a full LLM call. `mean_similarity` is computed and
recorded on every turn but still deliberately not wired into the gate decision.

Re-run the sweep without Ollama (`signals.json` is committed):

```
uv run python -m evals.sweep
```

## Scope — read this before pointing it at anything real

This is a single-user, local-only tool, not a deployable service:

- Built for exactly one local user talking to `127.0.0.1`. There is no authentication anywhere —
  every endpoint, including document upload and delete, is open to whoever can reach the port.
- `DEBUG` is on by default (see Configuration below).
- **Do not put real patient data (PHI) into this.** It has not been built or reviewed against any
  compliance standard (HIPAA or otherwise). Use synthetic or public reference documents only.
- Answers are informational reference only, never a substitute for professional medical judgment —
  the system prompt says as much to the model, too.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/). This repo pins Python 3.12 (`.python-version`); `uv` fetches
  it automatically if it isn't already installed.
- [Ollama](https://ollama.com) running locally at `http://127.0.0.1:11434` (the default — override
  with `OLLAMA_HOST`), with both models pulled:
  ```
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ```
- SQLite built with FTS5 support (bundled with Python's stdlib `sqlite3` on macOS and most Linux
  distributions) — this is what backs lexical/BM25 search.

## Running it

```
uv run python manage.py migrate
uv run python manage.py runserver
```

Serves on `http://127.0.0.1:8000` under **WSGI** — see the note below, it matters. The API is
mounted under `/api/`:

- `GET  /api/health/` — Ollama reachability, which models are present, how many documents are ready
- `GET /api/documents/`, `POST /api/documents/` (multipart, field `file`, PDF only), `DELETE /api/documents/<id>/`
- `POST /api/chat/` — NDJSON streaming response: a `meta` frame first, then `sources`/`token`/`error`
  frames as appropriate, always ending in one `done` frame
- `GET  /api/chat/sessions/<id>/messages/`

The browser UI lives in [`../frontend`](../frontend) and expects this server on port 8000.

`/admin/` has `Document`, `Chunk`, `ChatSession` and `ChatMessage` registered (create a superuser
first: `uv run python manage.py createsuperuser`) — `ChatMessage`'s list view surfaces
`was_declined` / `decline_reason` directly, so gate decisions are inspectable without a shell.

To check (and optionally repair) drift between SQLite chunk rows and Chroma vectors:
```
uv run python manage.py reconcile_vectors [--fix]
```

### Configuration

Everything tunable reads from an env var with a working local default — see `rag/config.py` for the
full list (chunk size/overlap, retrieval `top_k`/`per_leg`, gate thresholds, history length, upload
size cap). The ones most worth knowing about:

| Env var | Default |
|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` |
| `CHAT_MODEL` | `llama3.1:8b` |
| `EMBED_MODEL` | `nomic-embed-text` |
| `DJANGO_SECRET_KEY` | an insecure local-dev-only key |
| `DJANGO_DEBUG` | `1` (on) |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1,testserver` |

## Tests

```
uv run pytest -q            # fast suite: fakes/mocks only, no live Ollama needed
uv run pytest -m ollama -q  # contract tests against a REAL, running Ollama
```

The `ollama`-marked contract tests are deselected by default (`pytest.ini`:
`addopts = -m "not ollama"`) because they need Ollama actually running with both models pulled — they
hit the real HTTP API and check response shape: real embedding dimensions, that document/query
prefixes actually produce different embeddings, that chat streaming yields real content.

## WSGI, not ASGI — this is load-bearing, not a style preference

`manage.py runserver` and `medical_rag/wsgi.py` are what this app is meant to be served under. Do
not point an ASGI server (`uvicorn medical_rag.asgi:application`) at it.

The chat endpoint streams via a plain synchronous Python generator wrapped in Django's
`StreamingHttpResponse`. `StreamingHttpResponse` cannot async-iterate a sync generator, so under
ASGI Django silently falls back to draining the *entire* generator in a threadpool before sending a
single byte. Measured against a real model: every token arrives at once at 9.5s under ASGI, versus
progressive delivery starting at 0.7s under WSGI. The failure is silent — no error, no warning, just
a response that is identical in shape and is not, in fact, streamed.

WSGI is also what makes exactly-once persistence of the assistant's reply reliable: it calls
`close()` on client disconnect, which raises `GeneratorExit` inside the generator and lets the
`finally` block persist whatever was collected so far. ASGI never delivers `GeneratorExit` to a sync
generator on disconnect, so that guarantee would simply not hold. If this is ever made to run under
ASGI, the Ollama client and the streaming view need to be rewritten as native async code first — the
current implementation actively depends on WSGI's semantics, not just tested against them.
