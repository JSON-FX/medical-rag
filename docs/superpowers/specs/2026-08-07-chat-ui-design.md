# Phase 4 — Chat UI Design

**Date:** 2026-08-07
**Status:** Approved for planning
**Extends:** [`2026-08-02-medical-rag-design.md`](2026-08-02-medical-rag-design.md), PRD §11

The backend is finished and measured. This document settles how the browser consumes it: the
NDJSON reader, how frames become rendered state, and what the UI does when the answer is a refusal
rather than an answer — which, in a system designed so "I don't know" is a normal outcome, is not
an error path but a first-class one.

---

## 1. Why this phase exists

PRD §11 specifies a two-page Next.js frontend. Today there is none: the API is exercised by `curl`
and `/admin/`. Nothing about the grounding design is visible — a decline looks identical to an
answer at the shell, and the citations that justify every answer are never shown next to it.

**Success criterion:** a user can upload a PDF, ask a question, watch the answer stream in with its
sources, ask something the corpus does not cover, and see a decline that reads as a deliberate
outcome rather than a failure.

---

## 2. Stack and constraints

| Decision | Choice | Why |
|---|---|---|
| Framework | Next.js App Router, client components | PRD §5 |
| Styling | Tailwind + shadcn/ui | Radix accessibility defaults; the app has real dialogs, badges and toasts |
| Data access | Browser → Django directly | `CORS_ALLOWED_ORIGINS = ["http://localhost:3000"]` already configured |
| Streaming | `fetch()` + `ReadableStream` | PRD §11; `EventSource` cannot POST |
| Sessions | `session_id` in React state, ephemeral | PRD §15 lists reload-history as a stretch item |
| Tests | Vitest over pure modules | Playwright E2E is Phase 5 |

### 2.1 Why not a Next.js proxy

Routing the stream through a Next route handler would remove the CORS config and hide the backend
URL. Neither is worth anything to a single local user, and the cost is real: the response must be
re-streamed through a second runtime, which is where buffering bugs live.

This project has already shipped that bug once. Under ASGI, `StreamingHttpResponse` drained the
entire generator before sending a byte — 9.48s to first token against 0.72s under WSGI — and the
end-to-end test passed anyway, because it asserted frame **order**, and order survives buffering
(see `IMPLEMENTATION_NOTES.md`). A proxy hop reintroduces exactly that failure mode, invisible to
exactly that kind of assertion. The browser talks to Django directly.

---

## 3. The frame contract

Read off `chat/views.py` and pinned by existing backend tests. Every response opens with `meta` and
closes with exactly one `done`.

| Turn | Frames |
|---|---|
| Answer | `meta` → `sources` → `token`… → `done{was_declined:false}` |
| Decline (stage 1 or 2) | `meta` → `token`(decline copy) → `done{was_declined:true}` |
| Error | `meta` → `error{code,message}` → `done{truncated:true}` |

### 3.1 The invariant the UI is built on

**A `sources` frame arrives if and only if the turn will be an answer.**

Stage 1 declines before the LLM is called, so no sources are emitted. Stage 2 breaks out of the
token loop on the sentinel *before* the `sources` frame is yielded, so a sentinel decline emits none
either. On the answer path, `sources` is always yielded before the first `token`.

This is not an accident of ordering — it is already pinned by three tests in
`tests/integration/test_chat_view.py`:

- `test_answered_question_emits_meta_sources_tokens_done_in_order`
- `test_off_domain_question_declines_without_sources`
- `test_sentinel_response_becomes_a_decline_with_no_sources_leaked`

The consequence is the whole reason the UI is clean: a decline can be rendered *as* a decline from
its first character. The alternative — render tokens as a normal message, then restyle the bubble
when `done` reveals `was_declined` — makes every refusal visibly flicker from answer to non-answer.
No backend change is required for Phase 4.

---

## 4. Modules

### 4.1 `lib/ndjson.ts`

```ts
async function* readFrames(body: ReadableStream<Uint8Array>): AsyncIterable<Frame>
```

Decodes the stream and yields one parsed object per `\n`-delimited line.

`ReadableStream` chunks do not respect line boundaries. A single frame can arrive split across two
reads, several frames can arrive in one, and the final frame may have no trailing newline. The
reader keeps a buffer, splits on `\n`, retains the trailing partial for the next chunk, and flushes
what remains at close.

A naive `chunk.split("\n").map(JSON.parse)` works on short answers and throws on long ones — a bug
that ships green whenever fixtures are small. The split-frame case is therefore a required test, not
an optional one.

### 4.2 `lib/chatReducer.ts`

```ts
function chatReducer(state: ChatState, frame: Frame): ChatState
```

Pure. No fetch, no DOM, no clock. Every decision the UI makes about a turn lives here:

- `meta` — record `session_id` for subsequent turns
- `sources` — mark the pending turn `kind: "answer"`, attach citations
- `token` — append text; if the turn is still unclassified, this is a decline (§3.1), so mark it
  `kind: "decline"`
- `done` — finalise; `decline_reason` selects the decline copy, `truncated` flags an incomplete answer
- `error` — map `code` to recovery text

Keeping this pure is deliberate mirroring of the backend. `rag/gate.py` is a pure function, and that
single property is what made a 120-point threshold sweep possible offline. The same choice here
means decline classification is testable without a browser.

### 4.3 `lib/api.ts`

One typed function per endpoint: `getHealth`, `listDocuments`, `uploadDocument`, `deleteDocument`,
`streamChat`. Base URL from `NEXT_PUBLIC_API_BASE`, defaulting to `http://localhost:8000`.

---

## 5. Components

| Component | Responsibility |
|---|---|
| `AppShell` | Nav rail + `StatusBar`, wrapping every page |
| `StatusBar` | Disclaimer and health in one line; re-checks on navigation |
| `ChatWindow` | Owns `useReducer(chatReducer)`, drives `streamChat`, holds evidence selection |
| `MessageBubble` | One turn, dispatching to answer / decline / error treatments |
| `AnswerText` | Parses `[n]` markers into buttons that select a source |
| `EvidencePanel` | The passages behind the selected answer, always visible |
| `DocumentUploader` | PDF-only, 15 MB, real pending state |
| `DocumentTable` | Title, status, pages, chunks, delete |

Superseded by the Clinical Workbench redesign: `DisclaimerBanner` and `HealthBanner` merged into
`StatusBar`; `SourceChips` and `DeclineCard` folded into `EvidencePanel` and `MessageBubble`;
`MessageList` collapsed into `ChatWindow`.

### 5.0 Why the evidence is a panel, not a tooltip

Every answer is supposed to trace back to retrieved text, and the first build put that text in a
`title` attribute — technically present, practically invisible. A claim nobody can check is
indistinguishable from a claim that is merely asserted, which is the failure this whole system is
built to avoid. The passages now sit beside the answer, and each `[n]` in the text selects one.

That mapping is only sound because `format_context` numbers chunks from 1 in the same order
`_sources_payload` serialises them, so `[n]` is always `sources[n-1]`. A marker past the end of the
list renders as plain text rather than a dead button — small models do invent citation numbers, and
an affordance that does nothing is worse than none.

### 5.1 Declines are not errors

A decline is the system working. It gets its own visual treatment — neutral, explanatory, distinct
from both a normal answer and a red error state — and names which kind it was: nothing relevant
found (`off_domain`), found something too weak to trust (`weak_unsupported`), no documents uploaded
(`empty_corpus`), or retrieved context that did not actually support an answer
(`insufficient_context`). Decline copy is server-authored and rendered as received; the UI chooses
presentation, never wording.

---

## 6. Health and error states

`/api/health/` already separates three conditions with three different fixes:

| Condition | Banner |
|---|---|
| `ollama_reachable: false` | Ollama is not running — start it |
| `models.chat` or `models.embed` false | Name the missing model and its `ollama pull` command |
| `documents_ready: 0` | Nothing uploaded yet — link to `/documents` |

The third matters most. With an empty corpus every question declines with `empty_corpus`, and
without the banner that reads as a broken app rather than an empty one.

Mid-stream, an `error` frame maps `ollama_unavailable` and `model_missing` to their own recovery
text. A `done` frame with `truncated: true` marks the answer visibly incomplete rather than letting
a half-answer pass as whole.

---

## 7. Testing

Vitest over `lib/ndjson.ts` and `lib/chatReducer.ts`, fed synthetic streams — no network, no Ollama.

Required cases:

- A frame split across two chunks reassembles
- Several frames in one chunk all parse
- A final frame with no trailing newline is not dropped
- `sources` → `token` classifies as an answer with citations attached
- `token` with no preceding `sources` classifies as a decline (§3.1)
- Each `decline_reason` selects its own copy
- `error` frames produce recovery text and end the turn
- `truncated: true` marks the answer incomplete

No component tests. E2E against a fake Ollama is Phase 5.

---

## 8. Out of scope

Markdown rendering of answers, copy-to-clipboard, dark mode, session switcher, optimistic UI,
auth. Session reload from `/api/chat/sessions/<id>/messages/` is deliberately deferred — the
endpoint exists, and the UI does not use it in Phase 4.

---

## 9. Risks

- **The `sources`-implies-answer invariant is load-bearing.** It is pinned by three backend tests
  today. If someone later emits sources before the sentinel check, declines will render as answers
  mid-stream. The reducer's decline test documents the dependency from the frontend side.
- **Synchronous ingestion blocks the upload request.** A large PDF can hold the connection for a
  long time with no progress signal. The pending state must not look hung; a spinner with an
  explanatory line is the mitigation, and background jobs remain the real fix (PRD §16).
- **`nomic-embed-text` similarities cluster high.** Thresholds are calibrated to a three-label FDA
  corpus. A user uploading very different documents may see more declines than expected, and the UI
  should make the reason legible rather than merely refusing.
