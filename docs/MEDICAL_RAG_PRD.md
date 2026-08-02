# Medical RAG — Product Requirements Document

**Status:** Draft v1
**Owner:** Jayson
**Purpose:** Portfolio / interview-demo project — a locally-run RAG chat app scoped to a user's own uploaded medical documents.

---

## 1. Overview

Medical RAG is a local, privacy-preserving chat application that lets a user upload medical reference documents (drug monographs, SOAP note templates, clinical guidelines) and ask questions that are answered *only* from what's been uploaded — not from the model's general training knowledge, and not about anything outside the medical domain.

Everything runs on the user's own machine: the LLM (via Ollama), the embeddings, and the vector store. No documents or questions leave the device.

## 2. Goals & Success Criteria

Primary goal: a working, demoable RAG pipeline with a clearly explainable design, especially around domain-scoping — this is the part worth walking an interviewer through.

A demo is successful if:
- [ ] A PDF can be uploaded and shows a "ready" status once processed.
- [ ] A question answerable from the uploaded document returns a grounded, streamed answer.
- [ ] A question *not* covered by the uploaded document (e.g. "what's the capital of France?", or a medical question outside the uploaded material) returns a clean decline — not a hallucinated answer.
- [ ] The whole thing runs with zero external API calls or keys.

## 3. Non-Goals (out of scope for v1)

- Multi-user accounts / auth
- HIPAA-grade security, encryption at rest, audit logging (see §13 for why this matters and what it would take)
- Multi-document cross-referencing beyond simple top-k retrieval
- Mobile app / responsive polish beyond "looks fine on a laptop"
- Editing or versioning uploaded documents
- Background job queue (Celery/RQ) — ingestion is synchronous for v1

## 4. Users & Use Case

Single local user (Jayson, in the demo). Use case: upload a handful of medical reference PDFs, then chat with an assistant that answers strictly from that material — a plausible shape for internal clinical reference tools, SOP lookup, or formulary Q&A.

## 5. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Backend | Django (plain views, no DRF) | Two endpoints don't justify serializer/viewset overhead; free admin panel for inspecting documents |
| Frontend | Next.js (App Router, client components) | Matches existing skill set, fastest path to a working chat UI |
| LLM runtime | Ollama, local | No API keys, no data leaves the machine |
| Chat model | `llama3.1:8b-instruct` (default) | Reliable instruction-following at a size that runs on a laptop |
| Embedding model | `nomic-embed-text` | Small, fast, purpose-built for RAG |
| Vector store | Chroma, `PersistentClient` (embedded, on-disk) | Zero infra — no server/Docker to run the night before a demo |
| PDF parsing | `pypdf` | Simple, dependency-light |
| Relational DB | SQLite (Django default) | Stores document/session/message metadata only — not vectors |

**Note on medical-tuned models:** Ollama also hosts domain-tuned options (`meditron`, `cniongolo/biomistral`) trained on PubMed/clinical corpora. Worth mentioning as something evaluated, but not the v1 default — published write-ups have flagged `meditron:7b` for inconsistent instruction-following, and in a RAG system, grounding comes from retrieval, not the base model's label. Listed as a stretch swap in §15.

## 6. System Architecture

Two independent pipelines, as diagrammed earlier in this conversation:

**Ingestion (upload → searchable):**
`PDF upload → extract & chunk text (pypdf + splitter) → generate embeddings (Ollama, nomic-embed-text) → store in Chroma`

**Query (question → answer):**
`User question → embed & retrieve top-k (Ollama + Chroma) → [confidence gate] → either "canned decline" (low similarity) or "Ollama chat model with system prompt + context" (good match) → response streamed to UI`

The confidence gate is the core design decision: retrieval quality decides *before* the LLM is ever called whether the question is answerable from what's been uploaded. This is more reliable than leaning on the system prompt alone, and cheaper, since a weak match skips the LLM call entirely.

## 7. Functional Requirements

**Document management**
- Upload a PDF (single file per request)
- List uploaded documents with status (`processing` / `ready` / `failed`) and chunk count
- Delete a document (removes its DB row and its vectors from Chroma)

**Chat**
- Start a new conversation or continue an existing one (session-scoped)
- Ask a question, receive a streamed, token-by-token response
- See which source document(s) an answer drew from
- Receive a clear decline (not a guess) when the question falls outside the uploaded material or outside the medical domain

## 8. Data Model (Django)

Vectors live in Chroma, keyed by chunk id. Django/SQLite stores metadata only.

```python
class Document(models.Model):
    STATUS_CHOICES = [("processing", "Processing"), ("ready", "Ready"), ("failed", "Failed")]
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="documents/")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="processing")
    page_count = models.IntegerField(null=True, blank=True)
    chunk_count = models.IntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    error_message = models.TextField(blank=True)

class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class ChatMessage(models.Model):
    ROLE_CHOICES = [("user", "User"), ("assistant", "Assistant")]
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    retrieved_sources = models.JSONField(default=list, blank=True)  # [{document_id, title, chunk_index, score}]
    was_declined = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
```

Chroma collection: single collection `medical_documents`. Each chunk stored with id `f"{document_id}_{chunk_index}"` and metadata `{document_id, chunk_index, page_number, source_title}`.

## 9. API Design

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/api/documents/` | POST | multipart: `file` | `{id, title, status, page_count, chunk_count}` (201) |
| `/api/documents/` | GET | — | `[{id, title, status, page_count, chunk_count, uploaded_at}]` |
| `/api/documents/{id}/` | DELETE | — | 204 |
| `/api/chat/` | POST | `{session_id: str \| null, question: str}` | streamed `text/plain` body; `X-Session-Id` response header |
| `/api/chat/sessions/{id}/messages/` | GET | — | `[{role, content, retrieved_sources, was_declined, created_at}]` (stretch) |

**Streaming detail:** the chat response is a plain streamed text body via `StreamingHttpResponse`, not SSE — the frontend uses `fetch()` + `response.body.getReader()` rather than `EventSource`, since `EventSource` can't send a POST body for the question. Session id comes back via a response header since the body itself carries no JSON envelope.

Ingestion runs synchronously inside the POST `/api/documents/` request for v1. Cap uploads at ~15MB to keep this from timing out a live demo.

## 10. RAG Pipeline Detail

**Chunking:** ~1000 characters per chunk, ~150 character overlap. Simple recursive splitter — no need for a full LangChain dependency for this alone.

**Retrieval:** `top_k = 4`. Chroma returns *distances* (lower = closer), not similarities — this matters for the gate's comparison direction.

**Confidence gate (pseudocode):**
```python
results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
best_distance = results["distances"][0][0]

if best_distance > DISTANCE_THRESHOLD:   # too far from anything uploaded
    return CANNED_DECLINE, was_declined=True
else:
    context = format_chunks(results)
    stream_from_ollama(system_prompt, context, recent_history, question)
```
`DISTANCE_THRESHOLD` starting value: `0.35` — this is a placeholder, not a measured number. It needs to be tuned empirically against your actual embedding model and sample documents once you have real data to test against; don't treat it as calibrated out of the box.

**System prompt (starting draft):**
```
You are a clinical reference assistant for the user's uploaded medical documents.
Answer only using the context provided below, drawn from documents the user has uploaded.
If the context does not contain enough information to answer, say so directly — do not guess.
If the question is unrelated to medicine or to the uploaded documents, decline, and explain
that you can only answer questions grounded in the uploaded material.
Do not draw on general knowledge beyond what's in the context.
When relevant, remind the user your answers are for informational/reference purposes only
and are not a substitute for professional medical judgment.

Context:
{retrieved_chunks}

Recent conversation:
{last_4_messages}
```

Recent history is capped at the last 4 messages (2 turns) to keep the prompt small for a local 7-8B model.

## 11. Frontend Requirements (Next.js)

- `/` — chat interface: message list, input box, streamed assistant responses rendered incrementally, source document shown under each answer
- `/documents` — upload form + list of documents with status
- Persistent disclaimer banner: "For informational reference only — not a substitute for professional medical judgment."
- Streaming consumed via `fetch()` + `ReadableStream`, not `EventSource` (see §9)

## 12. Non-Functional Requirements

- Runs fully offline/local — no external API dependency
- First streamed token should arrive within a few seconds on a modern laptop CPU/GPU, using the default 7-8B model class
- No auth, no rate limiting, no horizontal scale — single local user only

## 13. Responsible Use & Data Handling

Real SOAP notes and patient records contain PHI. For this project:
- Use publicly available reference material (drug monographs, published guidelines) or synthetic/de-identified sample SOAP notes — never real patient data, even for a local demo.
- If this were ever taken past a portfolio project, it would need encryption at rest, access controls, and audit logging before touching real clinical data — explicitly out of scope here, but worth naming in an interview as a conscious boundary, not an oversight.

## 14. Repo Structure

```
medical-rag/
├── backend/
│   ├── manage.py
│   ├── medical_rag/          # project settings, urls
│   ├── chat/                 # app: models, views, urls
│   │   └── services/
│   │       ├── ingest.py     # extract, chunk, embed, store
│   │       ├── retrieve.py   # embed query, similarity search, gate
│   │       └── ollama_client.py
│   ├── chroma_db/            # gitignored
│   ├── media/                # gitignored
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   ├── documents/page.tsx
│   │   └── components/
│   │       ├── ChatWindow.tsx
│   │       ├── MessageBubble.tsx
│   │       └── DocumentUploader.tsx
│   ├── lib/api.ts
│   └── package.json
└── README.md
```

## 15. Phased Build Plan

| Phase | Scope | Notes |
|---|---|---|
| 0 — Setup | Ollama installed, models pulled, Django + Next.js projects scaffolded, CORS wired | |
| 1 — Ingestion | Upload endpoint, pypdf extraction, chunking, embeddings, Chroma storage, document list UI | |
| 2 — Chat | Retrieval, confidence gate, Ollama streaming call, Next.js streaming chat UI | Core demo path — everything else is polish |
| 3 — Polish | Disclaimer banner, source citations in UI, loading/error states | |
| 4 — Stretch | Swap in a medical-tuned model, delete documents from UI, persist/reload session history | |

## 16. Trade-offs & Future Considerations

| Decision | Trade-off | Revisit if... |
|---|---|---|
| Embedded Chroma vs. dedicated vector DB | Zero infra, but single-process only | Multi-user or production scale |
| Synchronous ingestion vs. background jobs | Simple, but blocks the request during processing | Large documents or many concurrent uploads |
| Prompt + confidence gate vs. fine-tuned classifier | Fast to build, reasonably reliable, but not bulletproof — a local 7-8B model can still occasionally stray outside context despite instructions | Guardrail failures show up in testing; would need an eval set of adversarial questions |
| Local models vs. cloud APIs | No data leaves the device, no cost, but capped quality vs. frontier models | Quality ceiling becomes the limiting factor |

## 17. Open Questions / Risks

- `DISTANCE_THRESHOLD` needs empirical tuning against real documents — flag this explicitly if asked about it in an interview, rather than presenting it as a solved number.
- Local models are not guaranteed to perfectly respect the system prompt's scope restriction 100% of the time — the retrieval gate is the stronger enforcement layer, but worth being upfront that no single layer is airtight.
- No persistence guarantee if a client disconnects mid-stream in v1 — the message is only saved to the DB after the generator completes.
