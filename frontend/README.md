# Medical RAG — Frontend

Next.js chat UI for the local medical RAG backend. Talks to Django directly at
`http://localhost:8000` (override with `NEXT_PUBLIC_API_BASE`); there is no proxy layer, so the
NDJSON stream reaches the browser unbuffered.

## Running it

The backend must be running first — see `../backend/README.md`.

```
npm install
npm run dev
```

Serves on `http://localhost:3000`, which is the origin Django's `CORS_ALLOWED_ORIGINS` already
permits.

## Pages

- `/` — chat. Answers stream token by token with source chips; questions the corpus can't support
  render as decline cards rather than errors.
- `/documents` — upload (PDF, 15 MB) and delete. Ingestion is synchronous, so upload holds until the
  document is `ready` or `failed`.

## Tests

```
npm test
```

Vitest over `lib/` only. The two modules with real logic are pure: `ndjson.ts` (frame reassembly
across stream chunks) and `chatReducer.ts` (frames to state, including decline classification). End
-to-end coverage is Phase 5.
