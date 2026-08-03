# Medical RAG Backend (Phases 0–2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete backend answer/decline path for a local medical RAG system — upload a PDF, ask a question, get a grounded streamed answer or an explicit decline — verifiable end-to-end with `curl`.

**Architecture:** A framework-free `rag/` library holds all retrieval and grounding logic (chunking, hybrid search, fusion, the confidence gate, sentinel detection) with zero Django imports, so it tests without a database or network. Two thin Django apps orchestrate it: `documents` (ingestion) and `chat` (query). Chunk text and a SQLite FTS5 lexical index live in SQLite; Chroma holds only vectors. Grounding is two-stage — a multi-signal gate runs *before* the LLM, and a server-detected refusal sentinel catches near-miss questions the gate lets through.

**Tech Stack:** Python 3.12 (via `uv`), Django 5.x, `uvicorn` (ASGI), Chroma (`PersistentClient`), SQLite + FTS5, `pypdf`, Ollama (`llama3.1:8b`, `nomic-embed-text`), `pytest` + `pytest-django`.

**Spec:** [`docs/superpowers/specs/2026-08-02-medical-rag-design.md`](../specs/2026-08-02-medical-rag-design.md). Section references below (§N) point there.

**Scope:** Phases 0–2 only. Eval harness (Phase 3), chat UI (Phase 4), and hardening (Phase 5) get their own plans. This plan ends with a working, `curl`-able backend.

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.12** via `uv`. System Python is 3.9.6 and cannot run Django 5.x.
- **Ollama CLI is NOT on `PATH`.** It lives at `/Applications/Ollama.app/Contents/Resources/ollama`. The server *is* running on `http://127.0.0.1:11434`.
- **Chat model tag is `llama3.1:8b`** — NOT `llama3.1:8b-instruct`, which is not a valid Ollama tag. The bare tag is already instruction-tuned q4_K_M.
- **Embedding model is `nomic-embed-text`**, 768 dimensions (verified).
- **Embedding prefixes are mandatory:** `search_document: ` for indexed chunks, `search_query: ` for queries. Applied inside `rag/embeddings.py`, never by callers.
- **Chroma must be configured for cosine space explicitly.** It defaults to L2. With cosine, `similarity = 1 - distance`.
- **`rag/` must not import `django`.** Enforced by a test.
- **SQLite FTS5 is available** (verified: `ENABLE_FTS5`, SQLite 3.51.0). Tokenizer is `porter unicode61`.
- **Resolved dependency versions** (installed in Task 1): Python 3.12.13, Django 5.2.16, chromadb 1.5.9.
- **Chroma collection names must be ≥3 characters** — chromadb 1.5.9 raises `InvalidArgumentError` otherwise.
- **`bm25()` returns negative values** where more-negative is a better match (verified). Never compare it to a similarity; use rank position only.
- **All views are synchronous `def`.** Django runs them in a threadpool under ASGI; ORM usage is unchanged.
- **`ATOMIC_REQUESTS` stays off.** Transactions are explicit and narrow, or a stream holds one open for its full duration.
- **Upload cap 15 MB**, checked before parsing.
- **Decline copy is server-generated**, never model-generated, and lives in `rag/prompts.py` as `DECLINE_COPY` (§6.8).
- **Sentinel string is exactly `INSUFFICIENT_CONTEXT`.**
- **Chunk vector ids are deterministic:** `f"{document_id}_{chunk_index}"`.

---

## File Structure

**`rag/` — pure library, no Django (§3.1)**

| File | Responsibility |
|---|---|
| `rag/config.py` | Frozen dataclasses for every tunable; env-var overrides |
| `rag/chunking.py` | Page-aware recursive splitting; `PageText` → `ChunkDraft` |
| `rag/embeddings.py` | Ollama `/api/embed` client; owns task prefixes |
| `rag/vectorstore.py` | Chroma adapter; cosine assertion; upsert/query/delete |
| `rag/lexical.py` | FTS5 query construction and input sanitising |
| `rag/fusion.py` | Reciprocal rank fusion |
| `rag/gate.py` | Stage-1 confidence gate; pure function |
| `rag/prompts.py` | System prompt, context assembly, sentinel, decline copy |
| `rag/generation.py` | Ollama `/api/chat` streaming; sentinel filtering |

**Django**

| File | Responsibility |
|---|---|
| `medical_rag/settings.py` | Config, CORS, DB |
| `documents/models.py` | `Document`, `Chunk` |
| `documents/migrations/0002_fts5.py` | FTS5 virtual table + sync triggers |
| `documents/ingestion.py` | Orchestrates `rag/` + persistence + cleanup |
| `documents/views.py` | Upload / list / delete |
| `documents/management/commands/reconcile_vectors.py` | Drift repair |
| `chat/models.py` | `ChatSession`, `ChatMessage` |
| `chat/retrieval.py` | Hybrid retrieval + gate orchestration |
| `chat/streaming.py` | NDJSON frame assembly |
| `chat/views.py` | Chat endpoint, health endpoint |

---

## Task 1: Backend scaffold and configuration

**Files:**
- Create: `backend/pyproject.toml`, `backend/.python-version`, `backend/manage.py`, `backend/medical_rag/{__init__,settings,urls,asgi}.py`, `backend/rag/{__init__,config}.py`, `backend/pytest.ini`, `backend/tests/__init__.py`
- Test: `backend/tests/unit/test_config.py`, `backend/tests/unit/test_rag_purity.py`

**Interfaces:**
- Consumes: nothing
- Produces: `rag.config.RagConfig`, `ChunkConfig`, `GateConfig`, `OllamaConfig`, `RetrievalConfig`, `load_config() -> RagConfig`

- [ ] **Step 1: Create the project and install dependencies**

```bash
cd backend
uv python install 3.12
uv init --python 3.12 --no-workspace .
uv add "django>=5.0,<6.0" "chromadb>=0.5.0" pypdf httpx "django-cors-headers" uvicorn
uv add --dev pytest pytest-django
echo "3.12" > .python-version
uv run django-admin startproject medical_rag .
```

- [ ] **Step 2: Write the failing tests**

`backend/tests/unit/test_config.py`:

```python
import os
from rag.config import load_config, RagConfig


def test_defaults_match_spec():
    cfg = load_config(env={})
    assert cfg.ollama.host == "http://127.0.0.1:11434"
    assert cfg.ollama.chat_model == "llama3.1:8b"
    assert cfg.ollama.embed_model == "nomic-embed-text"
    assert cfg.chunk.size == 1000
    assert cfg.chunk.overlap == 150
    assert cfg.retrieval.per_leg == 10
    assert cfg.retrieval.top_k == 4
    assert cfg.retrieval.rrf_k == 60
    assert cfg.gate.tau_abstain == 0.30
    assert cfg.gate.tau_strong == 0.45
    assert cfg.max_upload_mb == 15
    assert cfg.history_messages == 4


def test_env_overrides_are_typed():
    cfg = load_config(env={"TAU_ABSTAIN": "0.5", "CHUNK_SIZE": "800", "CHAT_MODEL": "other:7b"})
    assert cfg.gate.tau_abstain == 0.5
    assert isinstance(cfg.gate.tau_abstain, float)
    assert cfg.chunk.size == 800
    assert isinstance(cfg.chunk.size, int)
    assert cfg.ollama.chat_model == "other:7b"


def test_config_is_frozen():
    cfg = load_config(env={})
    try:
        cfg.gate.tau_abstain = 0.9
    except Exception:
        return
    raise AssertionError("GateConfig must be frozen")
```

`backend/tests/unit/test_rag_purity.py` — enforces the §3.1 boundary:

```python
import pathlib
import re

RAG_DIR = pathlib.Path(__file__).resolve().parents[2] / "rag"


def test_rag_library_never_imports_django():
    offenders = []
    for path in RAG_DIR.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(import|from)\s+django", source, re.MULTILINE):
            offenders.append(path.name)
    assert offenders == [], f"rag/ must stay framework-free, but these import django: {offenders}"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.config'`

- [ ] **Step 4: Write `rag/config.py`**

```python
"""Every tunable in one place. No Django imports (see spec 3.1)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    chat_model: str = "llama3.1:8b"      # NOT llama3.1:8b-instruct (invalid tag)
    embed_model: str = "nomic-embed-text"
    embed_dimensions: int = 768
    request_timeout_s: int = 120


@dataclass(frozen=True)
class ChunkConfig:
    size: int = 1000
    overlap: int = 150


@dataclass(frozen=True)
class RetrievalConfig:
    per_leg: int = 10     # candidates pulled from each of the vector and lexical legs
    top_k: int = 4        # chunks kept after fusion
    rrf_k: int = 60       # reciprocal rank fusion constant


@dataclass(frozen=True)
class GateConfig:
    # PLACEHOLDERS until the Phase 3 eval sweep. See spec 6.5 and 17.
    tau_abstain: float = 0.30
    tau_strong: float = 0.45


@dataclass(frozen=True)
class RagConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    max_upload_mb: int = 15
    history_messages: int = 4


def load_config(env: Mapping[str, str] | None = None) -> RagConfig:
    e = os.environ if env is None else env

    def _f(key: str, default: float) -> float:
        return float(e.get(key, default))

    def _i(key: str, default: int) -> int:
        return int(e.get(key, default))

    def _s(key: str, default: str) -> str:
        return e.get(key, default)

    return RagConfig(
        ollama=OllamaConfig(
            host=_s("OLLAMA_HOST", "http://127.0.0.1:11434"),
            chat_model=_s("CHAT_MODEL", "llama3.1:8b"),
            embed_model=_s("EMBED_MODEL", "nomic-embed-text"),
            request_timeout_s=_i("OLLAMA_TIMEOUT_S", 120),
        ),
        chunk=ChunkConfig(size=_i("CHUNK_SIZE", 1000), overlap=_i("CHUNK_OVERLAP", 150)),
        retrieval=RetrievalConfig(
            per_leg=_i("RETRIEVE_N", 10), top_k=_i("TOP_K", 4), rrf_k=_i("RRF_K", 60)
        ),
        gate=GateConfig(tau_abstain=_f("TAU_ABSTAIN", 0.30), tau_strong=_f("TAU_STRONG", 0.45)),
        max_upload_mb=_i("MAX_UPLOAD_MB", 15),
        history_messages=_i("HISTORY_MESSAGES", 4),
    )
```

- [ ] **Step 5: Configure pytest**

`backend/pytest.ini`:

```ini
[pytest]
DJANGO_SETTINGS_MODULE = medical_rag.settings
python_files = test_*.py
testpaths = tests
markers =
    ollama: contract tests requiring a live Ollama (deselected by default)
addopts = -m "not ollama"
```

- [ ] **Step 6: Configure Django settings**

In `medical_rag/settings.py`, add to `INSTALLED_APPS`: `"corsheaders"`, `"documents"`, `"chat"` (apps created in Tasks 6 and 15 — comment them out until then). Add `"corsheaders.middleware.CorsMiddleware"` as the **first** middleware. Then append:

```python
CORS_ALLOWED_ORIGINS = ["http://localhost:3000"]
CHROMA_PATH = BASE_DIR / "chroma_db"
MEDIA_ROOT = BASE_DIR / "media"
MEDIA_URL = "/media/"
# ATOMIC_REQUESTS deliberately left off: a request-wrapping transaction would
# stay open for the entire duration of a streamed response (spec 11.1).
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/unit -v`
Expected: PASS — 4 tests

- [ ] **Step 8: Commit**

```bash
git add backend/
git commit -m "feat: scaffold Django backend with rag config module

Pins Python 3.12 (system 3.9.6 cannot run Django 5.x) and establishes
the rag/ purity boundary with an enforcing test."
```

---

## Task 2: Ollama client and health endpoint

**Files:**
- Create: `backend/rag/ollama.py`, `backend/chat/__init__.py`, `backend/chat/apps.py`, `backend/chat/views.py`, `backend/chat/urls.py`
- Modify: `backend/medical_rag/urls.py`
- Test: `backend/tests/unit/test_health.py`

**Interfaces:**
- Consumes: `rag.config.OllamaConfig`
- Produces: `rag.ollama.OllamaClient(cfg)` with `.list_models() -> list[str]`, `.is_reachable() -> bool`; `chat.views.health`

- [ ] **Step 1: Write the failing test**

`backend/tests/unit/test_health.py`:

```python
import json
import pytest
from django.test import Client

from rag.config import load_config


class FakeOllama:
    """Stands in for OllamaClient. Keep the signature identical."""

    def __init__(self, models, reachable=True):
        self._models = models
        self._reachable = reachable

    def is_reachable(self):
        return self._reachable

    def list_models(self):
        if not self._reachable:
            raise ConnectionError("unreachable")
        return self._models


@pytest.fixture
def client():
    return Client()


def test_health_reports_all_present(client, monkeypatch):
    import chat.views as views
    monkeypatch.setattr(
        views, "build_client", lambda cfg: FakeOllama(["llama3.1:8b", "nomic-embed-text:latest"])
    )
    resp = client.get("/api/health/")
    body = json.loads(resp.content)
    assert resp.status_code == 200
    assert body["ollama_reachable"] is True
    assert body["models"]["chat"] is True
    assert body["models"]["embed"] is True


def test_health_matches_model_tags_ignoring_latest_suffix(client, monkeypatch):
    """`ollama list` reports `nomic-embed-text:latest` for a `nomic-embed-text` pull."""
    import chat.views as views
    monkeypatch.setattr(
        views, "build_client", lambda cfg: FakeOllama(["llama3.1:8b", "nomic-embed-text:latest"])
    )
    body = json.loads(client.get("/api/health/").content)
    assert body["models"]["embed"] is True


def test_health_reports_missing_model(client, monkeypatch):
    import chat.views as views
    monkeypatch.setattr(views, "build_client", lambda cfg: FakeOllama(["nomic-embed-text:latest"]))
    body = json.loads(client.get("/api/health/").content)
    assert body["models"]["chat"] is False
    assert body["models"]["embed"] is True


def test_a_different_tag_of_the_same_model_is_not_a_match(client, monkeypatch):
    """`llama3.1:70b` must NOT satisfy a requirement for `llama3.1:8b`.

    Reporting it present would send the user into a demo whose chat endpoint
    404s on a tag health already vouched for.
    """
    import chat.views as views
    monkeypatch.setattr(
        views,
        "build_client",
        lambda cfg: FakeOllama(["llama3.1:70b", "llama3.1:8b-instruct-q4_0"]),
    )
    body = json.loads(client.get("/api/health/").content)
    assert body["models"]["chat"] is False


def test_health_reports_unreachable_without_raising(client, monkeypatch):
    import chat.views as views
    monkeypatch.setattr(views, "build_client", lambda cfg: FakeOllama([], reachable=False))
    resp = client.get("/api/health/")
    body = json.loads(resp.content)
    assert resp.status_code == 200          # health must not 500 when Ollama is down
    assert body["ollama_reachable"] is False
    assert body["models"]["chat"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_health.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chat'`

- [ ] **Step 3: Write `rag/ollama.py`**

```python
"""Thin Ollama HTTP client. No Django imports."""
from __future__ import annotations

import httpx

from .config import OllamaConfig


class OllamaError(RuntimeError):
    """Base for Ollama transport failures."""


class OllamaUnavailable(OllamaError):
    """Ollama is not reachable at the configured host."""


class OllamaClient:
    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg

    def list_models(self) -> list[str]:
        try:
            resp = httpx.get(f"{self.cfg.host}/api/tags", timeout=5.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(str(exc)) from exc
        return [m["name"] for m in resp.json().get("models", [])]

    def is_reachable(self) -> bool:
        try:
            self.list_models()
            return True
        except OllamaUnavailable:
            return False
```

- [ ] **Step 4: Create the `chat` app and health view**

```bash
cd backend && uv run python manage.py startapp chat
```

`backend/chat/views.py`:

```python
import json

from django.http import JsonResponse
from django.views.decorators.http import require_GET

from rag.config import load_config
from rag.ollama import OllamaClient, OllamaUnavailable


def build_client(cfg):
    """Indirection so tests can substitute a fake."""
    return OllamaClient(cfg.ollama)


def _has_model(available: list[str], wanted: str) -> bool:
    """`ollama list` reports `name:latest` for an untagged pull.

    Only the `:latest` suffix is normalised. Stripping the tag wholesale would
    make `llama3.1:70b` satisfy a request for `llama3.1:8b`, so health would
    report the model present and the failure would resurface later as an
    unexplained 404 from the chat endpoint — the precise false confidence this
    endpoint exists to prevent.
    """

    def normalise(name: str) -> str:
        return name[: -len(":latest")] if name.endswith(":latest") else name

    target = normalise(wanted)
    return any(normalise(name) == target for name in available)


@require_GET
def health(request):
    cfg = load_config()
    client = build_client(cfg)
    try:
        models = client.list_models()
        reachable = True
    except (OllamaUnavailable, ConnectionError):
        models, reachable = [], False

    # Imported inside the try because the `documents` app does not exist yet at
    # this point in the build order, and health must never be the thing that 500s.
    try:
        from documents.models import Document

        ready = Document.objects.filter(status="ready").count()
    except Exception:
        ready = 0

    return JsonResponse(
        {
            "ollama_reachable": reachable,
            "host": cfg.ollama.host,
            "models": {
                "chat": _has_model(models, cfg.ollama.chat_model),
                "embed": _has_model(models, cfg.ollama.embed_model),
            },
            "expected": {"chat": cfg.ollama.chat_model, "embed": cfg.ollama.embed_model},
            "documents_ready": ready,
        }
    )
```

`backend/chat/urls.py`:

```python
from django.urls import path

from . import views

urlpatterns = [path("health/", views.health, name="health")]
```

In `medical_rag/urls.py`:

```python
from django.urls import include, path

urlpatterns = [
    path("api/", include("chat.urls")),
]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_health.py -v`
Expected: PASS — 4 tests

- [ ] **Step 6: Verify against real Ollama**

```bash
uv run uvicorn medical_rag.asgi:application --port 8000 &
curl -s localhost:8000/api/health/ | python3 -m json.tool
```

Expected: `"ollama_reachable": true`, `"models": {"chat": true, "embed": true}`. If `chat` is `false`, the `llama3.1:8b` pull has not finished.

- [ ] **Step 7: Commit**

```bash
git add backend/
git commit -m "feat: add Ollama client and /api/health/ endpoint

Health never 500s when Ollama is down; it reports the state so the UI
can show an actionable message instead of a mystery failure."
```

---

## Task 3: Page-aware chunking

**Files:**
- Create: `backend/rag/chunking.py`
- Test: `backend/tests/unit/test_chunking.py`

**Interfaces:**
- Consumes: `rag.config.ChunkConfig`
- Produces: `PageText(page_number: int, text: str)`, `ChunkDraft(chunk_index: int, page_number: int, text: str)`, `chunk_pages(pages: list[PageText], cfg: ChunkConfig) -> list[ChunkDraft]`

- [ ] **Step 1: Write the failing tests**

`backend/tests/unit/test_chunking.py`:

```python
import pytest

from rag.chunking import PageText, ChunkDraft, chunk_pages
from rag.config import ChunkConfig

CFG = ChunkConfig(size=100, overlap=20)


def test_short_page_becomes_one_chunk():
    chunks = chunk_pages([PageText(1, "Metformin 500mg twice daily.")], CFG)
    assert len(chunks) == 1
    assert chunks[0] == ChunkDraft(chunk_index=0, page_number=1, text="Metformin 500mg twice daily.")


def test_chunks_never_span_a_page_boundary():
    pages = [PageText(1, "alpha " * 40), PageText(2, "beta " * 40)]
    chunks = chunk_pages(pages, CFG)
    for c in chunks:
        assert not ("alpha" in c.text and "beta" in c.text)


def test_page_number_is_preserved_per_chunk():
    pages = [PageText(7, "gamma " * 40), PageText(8, "delta " * 40)]
    chunks = chunk_pages(pages, CFG)
    assert {c.page_number for c in chunks} == {7, 8}
    assert all(c.page_number == 7 for c in chunks if "gamma" in c.text)


def test_chunk_index_is_monotonic_across_pages():
    pages = [PageText(1, "one " * 40), PageText(2, "two " * 40), PageText(3, "three " * 40)]
    chunks = chunk_pages(pages, CFG)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_blank_pages_are_skipped_without_consuming_an_index():
    pages = [PageText(1, "content here"), PageText(2, "   \n  "), PageText(3, "more content")]
    chunks = chunk_pages(pages, CFG)
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert [c.page_number for c in chunks] == [1, 3]


def test_overlap_carries_tail_of_previous_chunk():
    page = PageText(1, "".join(f"sentence{i}. " for i in range(40)))
    chunks = chunk_pages([page], ChunkConfig(size=100, overlap=20))
    assert len(chunks) > 1
    tail = chunks[0].text[-20:]
    assert chunks[1].text.startswith(tail)


def test_no_chunk_greatly_exceeds_configured_size():
    page = PageText(1, "x" * 1000)  # no separators at all
    chunks = chunk_pages([page], ChunkConfig(size=100, overlap=0))
    assert all(len(c.text) <= 100 for c in chunks)


def test_empty_document_yields_no_chunks():
    assert chunk_pages([], CFG) == []
    assert chunk_pages([PageText(1, "")], CFG) == []


def test_overlap_is_added_on_top_of_size_not_carved_out_of_it():
    """Documents the size contract: max chunk length is size + overlap.

    Pinned deliberately. If this ever changes, chunk boundaries shift and the
    Phase 3 threshold sweep is no longer comparable to earlier runs.
    """
    page = PageText(1, "x" * 250)
    chunks = chunk_pages([page], ChunkConfig(size=100, overlap=20))
    assert max(len(c.text) for c in chunks) == 120
    assert all(len(c.text) <= 100 + 20 for c in chunks)


def test_non_positive_chunk_size_raises_rather_than_dropping_text():
    """A negative size once made _split_recursive return [] — silently losing
    the entire page. Losing text is unrecoverable: retrieval can never find it."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            chunk_pages([PageText(1, "content that must not vanish")], ChunkConfig(size=bad, overlap=0))
```

Note `test_no_chunk_greatly_exceeds_configured_size` uses `overlap=0`, which is why it does not conflict with the size+overlap contract above.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_chunking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.chunking'`

- [ ] **Step 3: Write `rag/chunking.py`**

```python
"""Page-aware recursive text splitting.

Chunks never span a page boundary, so every chunk carries an exact page
number for citation (spec 6.1).
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import ChunkConfig

SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class ChunkDraft:
    chunk_index: int
    page_number: int
    text: str


def _split_recursive(text: str, size: int, seps: list[str]) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    if not seps:
        # No separators left: hard-slice so a pathological page still chunks.
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    out: list[str] = []
    buf = ""
    for part in parts:
        candidate = part if not buf else f"{buf}{sep}{part}"
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            out.append(buf)
            buf = ""
        if len(part) > size:
            out.extend(_split_recursive(part, size, rest))
        else:
            buf = part
    if buf:
        out.append(buf)
    return [c.strip() for c in out if c.strip()]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        out.append(previous[-overlap:] + current)
    return out


def chunk_pages(pages: list[PageText], cfg: ChunkConfig) -> list[ChunkDraft]:
    """Split pages into chunks that never span a page boundary.

    Size contract: each chunk holds up to ``cfg.size`` characters of new text,
    plus up to ``cfg.overlap`` characters repeated from the tail of the previous
    chunk on the same page. The effective maximum length is therefore
    ``cfg.size + cfg.overlap``, not ``cfg.size``. At the real configuration
    (1000/150) that is 1150 characters, roughly 300 tokens — far inside the
    embedding model's window.
    """
    if cfg.size <= 0:
        # A negative size made range() empty, so _split_recursive returned []
        # and dropped the page silently. Text that vanishes can never be
        # retrieved, and no test downstream would notice.
        raise ValueError(f"chunk size must be positive, got {cfg.size}")

    drafts: list[ChunkDraft] = []
    index = 0
    for page in pages:
        pieces = _apply_overlap(_split_recursive(page.text, cfg.size, SEPARATORS), cfg.overlap)
        for piece in pieces:
            drafts.append(ChunkDraft(chunk_index=index, page_number=page.page_number, text=piece))
            index += 1
    return drafts
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_chunking.py -v`
Expected: PASS — 8 tests

- [ ] **Step 5: Commit**

```bash
git add backend/rag/chunking.py backend/tests/unit/test_chunking.py
git commit -m "feat: add page-aware recursive chunking

Chunks never cross a page boundary so citations can cite an exact page."
```

---

## Task 4: Embeddings client with mandatory task prefixes

**Files:**
- Create: `backend/rag/embeddings.py`
- Test: `backend/tests/unit/test_embeddings.py`, `backend/tests/contract/test_ollama_contract.py`

**Interfaces:**
- Consumes: `rag.config.OllamaConfig`
- Produces: `OllamaEmbedder(cfg)` with `.embed_documents(texts: list[str]) -> list[list[float]]` and `.embed_query(text: str) -> list[float]`

- [ ] **Step 1: Write the failing tests**

`backend/tests/unit/test_embeddings.py`:

```python
import pytest

from rag.config import OllamaConfig
from rag.embeddings import OllamaEmbedder

CFG = OllamaConfig()


class SpyTransport:
    """Captures the payload the embedder sends."""

    def __init__(self, dims=768):
        self.payloads = []
        self.dims = dims

    def __call__(self, url, payload):
        self.payloads.append(payload)
        return {"embeddings": [[0.1] * self.dims for _ in payload["input"]]}


def test_documents_get_the_search_document_prefix():
    spy = SpyTransport()
    OllamaEmbedder(CFG, transport=spy).embed_documents(["metformin 500mg"])
    assert spy.payloads[0]["input"] == ["search_document: metformin 500mg"]


def test_queries_get_the_search_query_prefix():
    spy = SpyTransport()
    OllamaEmbedder(CFG, transport=spy).embed_query("what is the dose?")
    assert spy.payloads[0]["input"] == ["search_query: what is the dose?"]


def test_documents_are_sent_as_one_batch_not_n_requests():
    spy = SpyTransport()
    OllamaEmbedder(CFG, transport=spy).embed_documents(["a", "b", "c"])
    assert len(spy.payloads) == 1
    assert len(spy.payloads[0]["input"]) == 3


def test_embed_query_returns_a_flat_vector_not_a_list_of_one():
    vec = OllamaEmbedder(CFG, transport=SpyTransport()).embed_query("q")
    assert isinstance(vec[0], float)
    assert len(vec) == 768


def test_empty_document_list_makes_no_request():
    spy = SpyTransport()
    assert OllamaEmbedder(CFG, transport=spy).embed_documents([]) == []
    assert spy.payloads == []


def test_dimension_mismatch_is_rejected_loudly():
    """A wrong embed model silently produces wrong-width vectors; fail fast."""
    with pytest.raises(ValueError, match="768"):
        OllamaEmbedder(CFG, transport=SpyTransport(dims=384)).embed_documents(["a"])
```

`backend/tests/contract/test_ollama_contract.py` — runs only with `-m ollama`:

```python
import pytest

from rag.config import OllamaConfig
from rag.embeddings import OllamaEmbedder

pytestmark = pytest.mark.ollama


def test_real_ollama_returns_768_dimensional_vectors():
    vectors = OllamaEmbedder(OllamaConfig()).embed_documents(["metformin dosing", "atenolol"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 768


def test_real_ollama_query_and_document_embeddings_differ():
    embedder = OllamaEmbedder(OllamaConfig())
    doc = embedder.embed_documents(["metformin dosing"])[0]
    query = embedder.embed_query("metformin dosing")
    assert doc != query, "prefixes must actually reach the model"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.embeddings'`

- [ ] **Step 3: Write `rag/embeddings.py`**

```python
"""Ollama embedding client.

`nomic-embed-text` is a prefixed model: indexed text needs `search_document: `
and queries need `search_query: `. Omitting them degrades retrieval silently,
so the prefixes are applied here and are not callable parameters (spec 6.3).
"""
from __future__ import annotations

from typing import Callable

import httpx

from .config import OllamaConfig
from .ollama import OllamaUnavailable

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def _http_transport(url: str, payload: dict, timeout: float) -> dict:
    try:
        resp = httpx.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        raise OllamaUnavailable(f"embed request failed: {exc}") from exc
    except ValueError as exc:  # json.JSONDecodeError subclasses ValueError
        raise OllamaUnavailable(f"embed response was not valid JSON: {exc}") from exc


class OllamaEmbedder:
    def __init__(self, cfg: OllamaConfig, transport: Callable[[str, dict], dict] | None = None):
        self.cfg = cfg
        # The lambda keeps the injectable seam at two arguments while still
        # wiring cfg.request_timeout_s, which was otherwise unreachable.
        self._transport = transport or (
            lambda url, payload: _http_transport(url, payload, cfg.request_timeout_s)
        )

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        body = self._transport(
            f"{self.cfg.host}/api/embed", {"model": self.cfg.embed_model, "input": inputs}
        )
        vectors = body.get("embeddings", [])
        if len(vectors) != len(inputs):
            # Count mismatches are more dangerous than they look: ingestion passes
            # ids and embeddings to Chroma positionally, so a short response pairs
            # chunk text with the wrong vector and silently poisons the store.
            raise ValueError(
                f"{self.cfg.embed_model} returned {len(vectors)} embeddings for "
                f"{len(inputs)} inputs. Refusing to continue: mismatched counts would "
                f"misalign chunk text with vectors and silently poison the store."
            )
        for vector in vectors:
            if len(vector) != self.cfg.embed_dimensions:
                raise ValueError(
                    f"expected {self.cfg.embed_dimensions}-dim embeddings from "
                    f"{self.cfg.embed_model}, got {len(vector)}"
                )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([DOCUMENT_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([QUERY_PREFIX + text])[0]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_embeddings.py -v`
Expected: PASS — 6 tests

- [ ] **Step 5: Run the contract tests against real Ollama**

Run: `uv run pytest tests/contract -m ollama -v`
Expected: PASS — 2 tests

- [ ] **Step 6: Commit**

```bash
git add backend/rag/embeddings.py backend/tests/
git commit -m "feat: add Ollama embeddings client with task prefixes

nomic-embed-text requires search_document:/search_query: prefixes.
Applied internally so callers cannot omit them."
```

---

## Task 5: Chroma vector store adapter

**Files:**
- Create: `backend/rag/vectorstore.py`
- Test: `backend/tests/unit/test_vectorstore.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `VectorHit(chunk_id: str, distance: float)`, `ChromaStore(path, collection_name="medical_documents")` with `.upsert(ids, embeddings, metadatas)`, `.query(embedding, n_results) -> list[VectorHit]`, `.delete_document(document_id: int) -> int`, `.count() -> int`, `.space -> str`

- [ ] **Step 1: Write the failing tests**

`backend/tests/unit/test_vectorstore.py`:

```python
import pytest

from rag.vectorstore import ChromaStore, VectorHit


@pytest.fixture
def store(tmp_path):
    return ChromaStore(path=str(tmp_path / "chroma"), collection_name="test_docs")


def _vec(seed: float) -> list[float]:
    return [seed] * 8


def test_collection_uses_cosine_space_not_the_l2_default():
    """Chroma defaults to L2. The whole gate depends on cosine (spec 6.4).

    Verified empirically on chromadb 1.5.9: with no configuration, two opposite
    unit vectors sit at distance 4.0 (L2 squared); with cosine they sit at 2.0.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        assert ChromaStore(path=tmp, collection_name="spacecheck").space == "cosine"


def test_opposite_vectors_are_distance_two_proving_cosine_not_l2(store):
    """The behavioural counterpart to the config assertion above: L2 would
    give 4.0 here, so this fails loudly if the space silently reverts."""
    store.upsert(
        ids=["1_0", "1_1"],
        embeddings=[_vec(1.0), [-1.0] + [0.0] * 7],
        metadatas=[{"document_id": 1, "chunk_index": 0}, {"document_id": 1, "chunk_index": 1}],
    )
    hits = store.query([1.0] + [0.0] * 7, n_results=2)
    assert hits[1].distance == pytest.approx(2.0, abs=1e-3)


def test_upsert_then_query_returns_nearest_first(store):
    store.upsert(
        ids=["1_0", "1_1"],
        embeddings=[_vec(1.0), _vec(-1.0)],
        metadatas=[{"document_id": 1, "chunk_index": 0}, {"document_id": 1, "chunk_index": 1}],
    )
    hits = store.query(_vec(1.0), n_results=2)
    assert [h.chunk_id for h in hits] == ["1_0", "1_1"]
    assert isinstance(hits[0], VectorHit)


def test_cosine_distance_of_identical_vectors_is_about_zero(store):
    store.upsert(["1_0"], [_vec(1.0)], [{"document_id": 1, "chunk_index": 0}])
    hit = store.query(_vec(1.0), n_results=1)[0]
    assert hit.distance == pytest.approx(0.0, abs=1e-4)


def test_upsert_is_idempotent_for_the_same_id(store):
    meta = [{"document_id": 1, "chunk_index": 0}]
    store.upsert(["1_0"], [_vec(1.0)], meta)
    store.upsert(["1_0"], [_vec(0.5)], meta)
    assert store.count() == 1


def test_delete_document_removes_only_that_documents_vectors(store):
    store.upsert(
        ids=["1_0", "2_0"],
        embeddings=[_vec(1.0), _vec(0.5)],
        metadatas=[{"document_id": 1, "chunk_index": 0}, {"document_id": 2, "chunk_index": 0}],
    )
    store.delete_document(1)
    assert store.count() == 1
    assert [h.chunk_id for h in store.query(_vec(0.5), n_results=5)] == ["2_0"]


def test_query_on_empty_collection_returns_empty_list(store):
    assert store.query(_vec(1.0), n_results=4) == []


def test_upsert_with_no_ids_is_a_noop(store):
    store.upsert([], [], [])
    assert store.count() == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_vectorstore.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.vectorstore'`

- [ ] **Step 3: Write `rag/vectorstore.py`**

```python
"""Chroma adapter.

Chroma defaults to L2 (squared Euclidean). Cosine must be requested
explicitly or every distance threshold in the gate is meaningless
(spec 6.4). The configured space is asserted at construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    distance: float          # cosine distance in [0, 2]; similarity = 1 - distance


class ChromaStore:
    def __init__(self, path: str, collection_name: str = "medical_documents"):
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        if self.space != "cosine":
            raise RuntimeError(
                f"collection '{collection_name}' is using space '{self.space}', not cosine. "
                "An existing collection created with a different space must be deleted."
            )

    @property
    def space(self) -> str:
        """Read back the configured space.

        `configuration_json` is authoritative on chromadb 1.5.9 — verified to
        report the real space whichever form created the collection — whereas
        `metadata` is populated only by the metadata form. Fall back to
        metadata for older releases.
        """
        config = getattr(self._collection, "configuration_json", None) or {}
        space = (config.get("hnsw") or {}).get("space")
        if space:
            return space
        return (self._collection.metadata or {}).get("hnsw:space", "unknown")

    def upsert(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        if not ids:
            return
        self._collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query(self, embedding: list[float], n_results: int) -> list[VectorHit]:
        """Nearest neighbours, closest first.

        No count() guard or clamp here deliberately. Verified against chromadb
        1.5.9: querying an empty collection returns [[]] and an n_results larger
        than the collection simply returns fewer rows — neither raises. The only
        input Chroma rejects is n_results <= 0, which is what this guards.
        Clamping with count() previously added two backend round-trips per query
        and a TOCTOU window that could drive n_results to zero and crash.
        """
        if n_results <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [VectorHit(chunk_id=i, distance=float(d)) for i, d in zip(ids, distances)]

    def delete_document(self, document_id: int) -> int:
        """Delete every vector for a document, returning how many went.

        Uses Chroma's atomic {"deleted": N} result rather than diffing count()
        before and after, which is wrong under any interleaved mutation.
        """
        result = self._collection.delete(where={"document_id": document_id})
        return (result or {}).get("deleted", 0)

    def delete_ids(self, ids: list[str]) -> int:
        """Delete exactly these vector ids.

        reconcile_vectors needs this rather than delete_document: an orphaned
        vector usually sits alongside valid ones for the same document, and
        deleting by document_id would take the valid ones with it — turning a
        harmless orphan into an unsearchable `ready` document.

        The returned count is Chroma's and can overcount (deleting an id that
        does not exist still reports 1), so callers needing an accurate figure
        should verify against all_ids().
        """
        if not ids:
            return 0
        result = self._collection.delete(ids=ids)
        return (result or {}).get("deleted", 0)

    def all_ids(self) -> set[str]:
        """Used by reconcile_vectors (Task 9)."""
        return set(self._collection.get(include=[])["ids"])

    def count(self) -> int:
        return self._collection.count()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_vectorstore.py -v`
Expected: PASS — 8 tests

**Verified against the installed chromadb 1.5.9 before this plan was executed** — you should not need to adjust anything, but if the assertion fails, adjust the `get_or_create_collection` call and the `space` property together and **do not delete the assertion**; it exists precisely to catch a silent revert to L2.

Measured facts for this version:

| Creation form | `configuration_json.hnsw.space` | `metadata` | Opposite-vector distance |
|---|---|---|---|
| `metadata={"hnsw:space": "cosine"}` | `cosine` | populated | 2.0 ✅ |
| `configuration={"hnsw": {"space": "cosine"}}` | `cosine` | empty | 2.0 ✅ |
| **omitted (default)** | **`l2`** | empty | **4.0** ❌ |

Also note: **chromadb 1.5.9 rejects collection names shorter than 3 characters** with `InvalidArgumentError`. Every collection name in this plan (`medical_documents`, `test_docs`, `test`, `spacecheck`) satisfies that.

- [ ] **Step 5: Commit**

```bash
git add backend/rag/vectorstore.py backend/tests/unit/test_vectorstore.py
git commit -m "feat: add Chroma adapter with enforced cosine space

Chroma defaults to L2; a silent default would invalidate every gate
threshold, so the configured space is asserted at construction."
```

---

## Task 6: Document and Chunk models with FTS5 index

**Files:**
- Create: `backend/documents/` app, `backend/documents/models.py`, `backend/documents/migrations/0002_fts5.py`
- Modify: `backend/medical_rag/settings.py` (uncomment `documents`)
- Test: `backend/tests/integration/test_models_fts.py`

**Interfaces:**
- Consumes: nothing
- Produces: `documents.models.Document`, `documents.models.Chunk` with `.vector_id` property; FTS5 table `chunk_fts` kept in sync by triggers

- [ ] **Step 1: Create the app**

```bash
cd backend && uv run python manage.py startapp documents
```

Add `"documents"` to `INSTALLED_APPS`.

- [ ] **Step 2: Write the failing tests**

`backend/tests/integration/test_models_fts.py`:

```python
import pytest
from django.db import connection

from documents.models import Chunk, Document

pytestmark = pytest.mark.django_db


def _doc(**kw):
    return Document.objects.create(title=kw.pop("title", "Monograph"), **kw)


def _fts_search(expression: str) -> list[int]:
    with connection.cursor() as cur:
        cur.execute(
            "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH %s ORDER BY bm25(chunk_fts)",
            [expression],
        )
        return [row[0] for row in cur.fetchall()]


def test_vector_id_is_deterministic():
    doc = _doc()
    chunk = Chunk.objects.create(document=doc, chunk_index=3, page_number=1, text="x")
    assert chunk.vector_id == f"{doc.id}_3"


def test_chunk_index_is_unique_per_document():
    doc = _doc()
    Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="a")
    with pytest.raises(Exception):
        Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="b")


def test_insert_trigger_populates_the_fts_index():
    doc = _doc()
    chunk = Chunk.objects.create(
        document=doc, chunk_index=0, page_number=1, text="metformin dosing in adults"
    )
    assert _fts_search('"metformin"') == [chunk.id]


def test_porter_stemming_matches_dose_to_dosing():
    """Justifies the porter tokenizer over plain unicode61."""
    doc = _doc()
    chunk = Chunk.objects.create(
        document=doc, chunk_index=0, page_number=1, text="recommended dosing schedule"
    )
    assert _fts_search('"dose"') == [chunk.id]


def test_delete_trigger_removes_the_row_from_fts():
    doc = _doc()
    chunk = Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="atenolol")
    chunk_id = chunk.id
    chunk.delete()
    assert _fts_search('"atenolol"') == []


def test_deleting_a_document_cascades_to_chunks_and_fts():
    doc = _doc()
    Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="warfarin")
    doc.delete()
    assert Chunk.objects.count() == 0
    assert _fts_search('"warfarin"') == []


def test_bm25_returns_negative_scores():
    """Documents the sign convention that makes rank-only fusion necessary."""
    doc = _doc()
    Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="metformin metformin")
    with connection.cursor() as cur:
        cur.execute("SELECT bm25(chunk_fts) FROM chunk_fts WHERE chunk_fts MATCH '\"metformin\"'")
        assert cur.fetchone()[0] < 0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/integration/test_models_fts.py -v`
Expected: FAIL — `ImportError: cannot import name 'Chunk'`

- [ ] **Step 4: Write `documents/models.py`**

```python
from django.db import models


class Document(models.Model):
    STATUS_CHOICES = [
        ("processing", "Processing"),
        ("ready", "Ready"),
        ("failed", "Failed"),
    ]
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="documents/")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="processing")
    page_count = models.IntegerField(null=True, blank=True)
    chunk_count = models.IntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self) -> str:
        return f"{self.title} ({self.status})"


class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.IntegerField()
    page_number = models.IntegerField()
    text = models.TextField()

    class Meta:
        unique_together = [("document", "chunk_index")]
        indexes = [models.Index(fields=["document", "chunk_index"])]
        ordering = ["document_id", "chunk_index"]

    @property
    def vector_id(self) -> str:
        """Deterministic id shared with Chroma (spec 10)."""
        return f"{self.document_id}_{self.chunk_index}"
```

- [ ] **Step 5: Create the model migration**

```bash
uv run python manage.py makemigrations documents
```

- [ ] **Step 6: Write the FTS5 migration**

`backend/documents/migrations/0002_fts5.py`:

```python
from django.db import migrations

CREATE = [
    """
    CREATE VIRTUAL TABLE chunk_fts USING fts5(
        text,
        content='documents_chunk',
        content_rowid='id',
        tokenize='porter unicode61'
    );
    """,
    """
    CREATE TRIGGER chunk_fts_ai AFTER INSERT ON documents_chunk BEGIN
        INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
    END;
    """,
    """
    CREATE TRIGGER chunk_fts_ad AFTER DELETE ON documents_chunk BEGIN
        INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', old.id, old.text);
    END;
    """,
    """
    CREATE TRIGGER chunk_fts_au AFTER UPDATE ON documents_chunk BEGIN
        INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', old.id, old.text);
        INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
    END;
    """,
]

DROP = [
    "DROP TRIGGER IF EXISTS chunk_fts_au;",
    "DROP TRIGGER IF EXISTS chunk_fts_ad;",
    "DROP TRIGGER IF EXISTS chunk_fts_ai;",
    "DROP TABLE IF EXISTS chunk_fts;",
]


class Migration(migrations.Migration):
    dependencies = [("documents", "0001_initial")]

    operations = [
        migrations.RunSQL(sql=CREATE, reverse_sql=DROP),
    ]
```

The `'delete'` command rows are how external-content FTS5 tables stay consistent — a plain `DELETE FROM chunk_fts` corrupts the index.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/integration/test_models_fts.py -v`
Expected: PASS — 7 tests

- [ ] **Step 8: Commit**

```bash
git add backend/documents/ backend/tests/integration/
git commit -m "feat: add Document and Chunk models with FTS5 index

Chunk text lives in SQLite so the lexical half of hybrid retrieval
needs no new dependency. Triggers keep the external-content FTS5
table in sync."
```

---

## Task 7: PDF extraction and ingestion pipeline

**Files:**
- Create: `backend/documents/ingestion.py`, `backend/tests/conftest.py`
- Test: `backend/tests/integration/test_ingestion.py`, `backend/tests/fixtures/make_fixture_pdf.py`

**Interfaces:**
- Consumes: `rag.chunking.chunk_pages`, `rag.embeddings.OllamaEmbedder`, `rag.vectorstore.ChromaStore`, `documents.models.Document`, `documents.models.Chunk`
- Produces: `extract_pages(path) -> list[PageText]`, `ingest_document(document, embedder, store, cfg) -> Document`, `cleanup_document(document_id, store)`
- Produces (test infrastructure, used by Tasks 8, 15, 16): `tests/conftest.py` exporting `FakeEmbedder`, `ExplodingEmbedder`, and the `fake_embedder` / `chroma_store` fixtures

- [ ] **Step 1: Create a fixture PDF generator**

`backend/tests/fixtures/make_fixture_pdf.py`:

```python
"""Builds deterministic test PDFs without adding a dependency.

pypdf cannot author content streams, so this writes minimal raw PDF.
"""
from __future__ import annotations

import pathlib


def make_pdf(path: pathlib.Path, pages: list[str]) -> pathlib.Path:
    objects: list[bytes] = []
    page_ids = [4 + i * 2 for i in range(len(pages))]

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for i, text in enumerate(pages):
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + i * 2} 0 R >>".encode()
        )
        objects.append(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()

    path.write_bytes(bytes(out))
    return path


def make_blank_pdf(path: pathlib.Path) -> pathlib.Path:
    """A page with no text operators — stands in for a scanned PDF."""
    return make_pdf(path, [""])
```

- [ ] **Step 2: Write the shared test fixtures**

`backend/tests/conftest.py` — the single definition of the fake embedder, imported by every later test module. Tasks 8, 15, and 16 use these fixtures rather than redefining stubs.

```python
"""Shared test fixtures.

One fake embedder serves every test module. Known terms map to fixed
orthogonal axes so cosine distances are predictable — "france" is maximally
far from "metformin" — and everything unrecognised lands on the unrelated
axis. Width matches the real model (768) so tests cannot pass against a
dimensionality the production path would reject.
"""
import pytest

from rag.vectorstore import ChromaStore

DIMENSIONS = 768


class FakeEmbedder:
    AXES = {"metformin": 0, "atenolol": 1}
    UNRELATED_AXIS = 2

    def __init__(self):
        self.document_batches = 0      # lets tests assert batching behaviour

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        lowered = text.lower()
        for term, axis in self.AXES.items():
            if term in lowered:
                vector[axis] = 1.0
                return vector
        vector[self.UNRELATED_AXIS] = 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class ExplodingEmbedder(FakeEmbedder):
    """Simulates Ollama failing partway through ingestion."""

    def embed_documents(self, texts):
        raise RuntimeError("ollama exploded")


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def chroma_store(tmp_path):
    return ChromaStore(path=str(tmp_path / "chroma"), collection_name="test")
```

- [ ] **Step 3: Write the failing tests**

`backend/tests/integration/test_ingestion.py`:

```python
import pytest
from django.core.files.base import ContentFile

from documents.ingestion import cleanup_document, extract_pages, ingest_document
from documents.models import Chunk, Document
from rag.config import load_config
from tests.conftest import ExplodingEmbedder
from tests.fixtures.make_fixture_pdf import make_blank_pdf, make_pdf

pytestmark = pytest.mark.django_db

CFG = load_config(env={"CHUNK_SIZE": "120", "CHUNK_OVERLAP": "20"})


@pytest.fixture
def pdf_doc(tmp_path):
    path = make_pdf(tmp_path / "mono.pdf", ["Metformin adult dose is 500mg.", "Atenolol 50mg daily."])
    doc = Document.objects.create(title="mono.pdf")
    doc.file.save("mono.pdf", ContentFile(path.read_bytes()), save=True)
    return doc


def test_extract_pages_returns_one_entry_per_page(tmp_path):
    path = make_pdf(tmp_path / "two.pdf", ["page one text", "page two text"])
    pages = extract_pages(str(path))
    assert [p.page_number for p in pages] == [1, 2]
    assert "page one" in pages[0].text


def test_successful_ingest_marks_ready_with_counts(pdf_doc, chroma_store, fake_embedder):
    doc = ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    assert doc.status == "ready"
    assert doc.page_count == 2
    assert doc.chunk_count > 0
    assert doc.chunk_count == Chunk.objects.filter(document=doc).count()


def test_vectors_and_chunks_agree_after_ingest(pdf_doc, chroma_store, fake_embedder):
    doc = ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    assert chroma_store.count() == doc.chunk_count
    assert chroma_store.all_ids() == {c.vector_id for c in Chunk.objects.filter(document=doc)}


def test_chunks_carry_real_page_numbers(pdf_doc, chroma_store, fake_embedder):
    ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    assert set(Chunk.objects.values_list("page_number", flat=True)) == {1, 2}


def test_embeddings_are_batched_in_one_call(pdf_doc, chroma_store, fake_embedder):
    ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    assert fake_embedder.document_batches == 1


def test_pdf_with_no_extractable_text_fails_with_a_useful_message(
    tmp_path, chroma_store, fake_embedder
):
    path = make_blank_pdf(tmp_path / "scanned.pdf")
    doc = Document.objects.create(title="scanned.pdf")
    doc.file.save("scanned.pdf", ContentFile(path.read_bytes()), save=True)

    result = ingest_document(doc, fake_embedder, chroma_store, CFG)
    assert result.status == "failed"
    assert "no extractable text" in result.error_message.lower()
    assert "ocr" in result.error_message.lower()
    assert Chunk.objects.count() == 0
    assert chroma_store.count() == 0


def test_embedding_failure_leaves_no_orphans(pdf_doc, chroma_store):
    result = ingest_document(pdf_doc, ExplodingEmbedder(), chroma_store, CFG)
    assert result.status == "failed"
    assert "ollama exploded" in result.error_message
    assert Chunk.objects.count() == 0
    assert chroma_store.count() == 0


def test_reingesting_the_same_document_does_not_duplicate(
    pdf_doc, chroma_store, fake_embedder
):
    first = ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    count = first.chunk_count
    second = ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    assert second.chunk_count == count
    assert chroma_store.count() == count
    assert Chunk.objects.filter(document=pdf_doc).count() == count


def test_cleanup_removes_from_both_stores(pdf_doc, chroma_store, fake_embedder):
    ingest_document(pdf_doc, fake_embedder, chroma_store, CFG)
    cleanup_document(pdf_doc.id, chroma_store)
    assert Chunk.objects.filter(document=pdf_doc).count() == 0
    assert chroma_store.count() == 0
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/integration/test_ingestion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'documents.ingestion'`

- [ ] **Step 5: Write `documents/ingestion.py`**

```python
"""Ingestion orchestration.

Order matters (spec 10): vectors are written before the SQLite transaction.
An orphaned vector is invisible to users because retrieval hydrates text from
SQLite and drops ids with no row. An orphaned SQLite row is worse — the
document would appear `ready` while being unsearchable.
"""
from __future__ import annotations

import logging

from django.db import transaction
from pypdf import PdfReader

from rag.chunking import PageText, chunk_pages
from rag.config import RagConfig

from .models import Chunk, Document

logger = logging.getLogger(__name__)

NO_TEXT_MESSAGE = (
    "This PDF contains no extractable text. It is most likely a scanned or "
    "image-only document — run it through OCR and upload the result."
)


def extract_pages(path: str) -> list[PageText]:
    reader = PdfReader(path)
    return [
        PageText(page_number=i, text=(page.extract_text() or ""))
        for i, page in enumerate(reader.pages, start=1)
    ]


def cleanup_document(document_id: int, store) -> None:
    """Compensating delete across both stores. Safe to call repeatedly."""
    store.delete_document(document_id)
    Chunk.objects.filter(document_id=document_id).delete()


def _safe_cleanup(document_id: int, store) -> None:
    """Cleanup that cannot itself abort the caller's error handling.

    If Chroma is unreachable, a raising cleanup inside an except block would
    propagate and skip the status write entirely, stranding the document in
    `processing` forever with no error message. Residual orphans are the
    lesser evil and are what `reconcile_vectors` exists to repair.
    """
    try:
        cleanup_document(document_id, store)
    except Exception:
        logger.exception(
            "cleanup failed for document %s; orphans may remain, run reconcile_vectors",
            document_id,
        )


def ingest_document(document: Document, embedder, store, cfg: RagConfig) -> Document:
    previous_status = document.status
    destroyed_previous = False

    try:
        # Everything failure-prone happens BEFORE any stored data is touched, so
        # a transient Ollama outage during re-ingest cannot destroy a document
        # that is currently ready and searchable.
        pages = extract_pages(document.file.path)
        page_count = len(pages)

        drafts = chunk_pages(pages, cfg.chunk)
        if not drafts:
            raise ValueError(NO_TEXT_MESSAGE)

        embeddings = embedder.embed_documents([d.text for d in drafts])

        # From here on the old state is gone; re-ingest must converge, not accumulate.
        cleanup_document(document.id, store)
        destroyed_previous = True

        store.upsert(
            ids=[f"{document.id}_{d.chunk_index}" for d in drafts],
            embeddings=embeddings,
            metadatas=[
                {"document_id": document.id, "chunk_index": d.chunk_index} for d in drafts
            ],
        )

        with transaction.atomic():
            Chunk.objects.bulk_create(
                [
                    Chunk(
                        document=document,
                        chunk_index=d.chunk_index,
                        page_number=d.page_number,
                        text=d.text,
                    )
                    for d in drafts
                ]
            )
            document.page_count = page_count
            document.chunk_count = len(drafts)
            document.status = "ready"
            document.error_message = ""
            document.save(
                update_fields=["page_count", "chunk_count", "status", "error_message"]
            )

    except Exception as exc:
        logger.exception("ingestion failed for document %s", document.id)

        if destroyed_previous:
            # We had already torn down the old state, so a partial new state is
            # all that can remain. Clear it and mark the document failed.
            _safe_cleanup(document.id, store)
            document.status = "failed"
            document.chunk_count = 0
        else:
            # Nothing stored was touched. A document that was already ready is
            # still complete and searchable — a transient Ollama outage during
            # re-ingest must not cost the user a working document.
            document.status = "failed" if previous_status != "ready" else "ready"

        document.error_message = str(exc)
        document.save(update_fields=["status", "error_message", "chunk_count"])

    return document
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/integration/test_ingestion.py -v`
Expected: PASS — 9 tests

- [ ] **Step 7: Commit**

```bash
git add backend/documents/ingestion.py backend/tests/
git commit -m "feat: add PDF extraction and ingestion pipeline

Vectors written before the SQLite transaction so a partial failure
leaves invisible orphans rather than unsearchable ready documents.
Compensating cleanup runs on every failure path."
```

---

## Task 8: Document endpoints

**Files:**
- Create: `backend/documents/views.py`, `backend/documents/urls.py`, `backend/documents/services.py`
- Modify: `backend/medical_rag/urls.py`
- Test: `backend/tests/integration/test_document_views.py`

**Interfaces:**
- Consumes: `documents.ingestion.ingest_document`, `cleanup_document`
- Produces: `documents.services.get_store()`, `get_embedder()`; endpoints `POST/GET /api/documents/`, `DELETE /api/documents/<id>/`

- [ ] **Step 1: Write the failing tests**

`backend/tests/integration/test_document_views.py`:

```python
import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client

from documents.models import Chunk, Document
from tests.fixtures.make_fixture_pdf import make_pdf

pytestmark = pytest.mark.django_db


@pytest.fixture
def client():
    return Client()


@pytest.fixture(autouse=True)
def isolated_services(chroma_store, fake_embedder, monkeypatch):
    """Point the views at a throwaway Chroma and the shared fake embedder."""
    import documents.services as services

    monkeypatch.setattr(services, "get_store", lambda: chroma_store)
    monkeypatch.setattr(services, "get_embedder", lambda: fake_embedder)
    return chroma_store


def _pdf_upload(tmp_path, name="mono.pdf"):
    path = make_pdf(tmp_path / name, ["Metformin adult dose is 500mg daily."])
    return SimpleUploadedFile(name, path.read_bytes(), content_type="application/pdf")


def test_upload_returns_201_and_ready_status(client, tmp_path):
    resp = client.post("/api/documents/", {"file": _pdf_upload(tmp_path)})
    body = json.loads(resp.content)
    assert resp.status_code == 201
    assert body["status"] == "ready"
    assert body["chunk_count"] > 0
    assert body["page_count"] == 1
    assert body["title"] == "mono.pdf"


def test_upload_rejects_non_pdf_with_400(client):
    bad = SimpleUploadedFile("notes.txt", b"hello", content_type="text/plain")
    resp = client.post("/api/documents/", {"file": bad})
    assert resp.status_code == 400
    assert "pdf" in json.loads(resp.content)["error"].lower()


def test_upload_rejects_oversized_file_with_413(client, settings):
    big = SimpleUploadedFile("big.pdf", b"%PDF-1.4\n" + b"0" * (16 * 1024 * 1024), "application/pdf")
    resp = client.post("/api/documents/", {"file": big})
    assert resp.status_code == 413
    assert Document.objects.count() == 0


def test_upload_with_no_file_returns_400(client):
    assert client.post("/api/documents/", {}).status_code == 400


def test_list_returns_documents_newest_first(client, tmp_path):
    client.post("/api/documents/", {"file": _pdf_upload(tmp_path, "a.pdf")})
    client.post("/api/documents/", {"file": _pdf_upload(tmp_path, "b.pdf")})
    body = json.loads(client.get("/api/documents/").content)
    assert [d["title"] for d in body] == ["b.pdf", "a.pdf"]
    assert set(body[0]) >= {"id", "title", "status", "page_count", "chunk_count", "uploaded_at"}


def test_delete_removes_document_chunks_and_vectors(client, tmp_path, isolated_services):
    created = json.loads(client.post("/api/documents/", {"file": _pdf_upload(tmp_path)}).content)
    assert isolated_services.count() > 0

    resp = client.delete(f"/api/documents/{created['id']}/")
    assert resp.status_code == 204
    assert Document.objects.count() == 0
    assert Chunk.objects.count() == 0
    assert isolated_services.count() == 0


def test_delete_missing_document_returns_404(client):
    assert client.delete("/api/documents/999/").status_code == 404
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/integration/test_document_views.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'documents.services'`

- [ ] **Step 3: Write `documents/services.py`**

```python
"""Process-wide singletons for the Chroma store and embedder.

Separate module so tests can monkeypatch these without importing views.
"""
from __future__ import annotations

from django.conf import settings

from rag.config import load_config
from rag.embeddings import OllamaEmbedder
from rag.vectorstore import ChromaStore

import threading

_store: ChromaStore | None = None
_store_lock = threading.Lock()


def get_store() -> ChromaStore:
    """Process-wide Chroma handle.

    Double-checked locking because sync views run in uvicorn's threadpool:
    concurrent first requests would otherwise race chromadb's tenant
    initialisation, which fails loudly and non-deterministically.
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ChromaStore(path=str(settings.CHROMA_PATH))
    return _store


def get_embedder() -> OllamaEmbedder:
    return OllamaEmbedder(load_config().ollama)
```

- [ ] **Step 4: Write `documents/views.py`**

```python
import logging

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt

from rag.config import load_config

from . import services
from .ingestion import cleanup_document, ingest_document
from .models import Document

logger = logging.getLogger(__name__)


def _redact(message: str) -> str:
    """Strip absolute server paths before an error reaches a client.

    Full detail stays in the server log; the client gets the cause without the
    filesystem layout.
    """
    if not message:
        return message
    for root, label in ((str(settings.MEDIA_ROOT), "<media>"), (str(settings.BASE_DIR), "<app>")):
        message = message.replace(root, label)
    return message


def _serialize(doc: Document) -> dict:
    return {
        "id": doc.id,
        "title": doc.title,
        "status": doc.status,
        "page_count": doc.page_count,
        "chunk_count": doc.chunk_count,
        "uploaded_at": doc.uploaded_at.isoformat(),
        "error_message": _redact(doc.error_message),
    }


@csrf_exempt
def documents_collection(request):
    if request.method == "GET":
        return JsonResponse([_serialize(d) for d in Document.objects.all()], safe=False)
    if request.method == "POST":
        return _upload(request)
    return JsonResponse({"error": "method not allowed"}, status=405)


def _upload(request):
    cfg = load_config()
    upload = request.FILES.get("file")
    if upload is None:
        return JsonResponse({"error": "no file provided"}, status=400)
    if not upload.name.lower().endswith(".pdf"):
        return JsonResponse({"error": "only PDF files are supported"}, status=400)
    if upload.size > cfg.max_upload_mb * 1024 * 1024:
        return JsonResponse(
            {"error": f"file exceeds the {cfg.max_upload_mb}MB limit"}, status=413
        )

    # Resolve dependencies BEFORE creating the row. If Chroma or the embedder
    # cannot be constructed, no Document should exist at all — a row created
    # here would be stranded in `processing` with nothing left to advance it.
    try:
        store = services.get_store()
        embedder = services.get_embedder()
    except Exception as exc:
        logger.exception("could not initialise ingestion services")
        return JsonResponse({"error": f"ingestion services unavailable: {exc}"}, status=503)

    document = Document.objects.create(title=upload.name, file=upload)
    document = ingest_document(document, embedder, store, cfg)
    return JsonResponse(_serialize(document), status=201)


@csrf_exempt
def document_detail(request, document_id: int):
    if request.method != "DELETE":
        return JsonResponse({"error": "method not allowed"}, status=405)
    document = get_object_or_404(Document, pk=document_id)
    cleanup_document(document.id, services.get_store())
    document.delete()
    return JsonResponse({}, status=204)
```

`backend/documents/urls.py`:

```python
from django.urls import path

from . import views

urlpatterns = [
    path("documents/", views.documents_collection, name="documents"),
    path("documents/<int:document_id>/", views.document_detail, name="document-detail"),
]
```

In `medical_rag/urls.py`, add `path("api/", include("documents.urls"))`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/integration/test_document_views.py -v`
Expected: PASS — 7 tests

- [ ] **Step 6: Commit**

```bash
git add backend/documents/ backend/medical_rag/urls.py backend/tests/
git commit -m "feat: add document upload, list, and delete endpoints"
```

---

## Task 9: reconcile_vectors management command

**Files:**
- Create: `backend/documents/management/__init__.py`, `backend/documents/management/commands/__init__.py`, `backend/documents/management/commands/reconcile_vectors.py`
- Test: `backend/tests/integration/test_reconcile.py`

**Interfaces:**
- Consumes: `documents.services.get_store`, `documents.models.Chunk`
- Produces: `python manage.py reconcile_vectors [--fix]`

- [ ] **Step 1: Write the failing tests**

`backend/tests/integration/test_reconcile.py`:

```python
import io

import pytest
from django.core.management import call_command

from documents.models import Chunk, Document

pytestmark = pytest.mark.django_db


@pytest.fixture
def store(tmp_path, monkeypatch):
    import documents.services as services
    from rag.vectorstore import ChromaStore

    s = ChromaStore(path=str(tmp_path / "chroma"), collection_name="test")
    monkeypatch.setattr(services, "get_store", lambda: s)
    return s


def _run(*args) -> str:
    out = io.StringIO()
    call_command("reconcile_vectors", *args, stdout=out)
    return out.getvalue()


def test_reports_clean_when_stores_agree(store):
    doc = Document.objects.create(title="d", status="ready")
    chunk = Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="metformin")
    store.upsert([chunk.vector_id], [[0.5] * 768], [{"document_id": doc.id, "chunk_index": 0}])
    assert "no drift" in _run().lower()


def test_detects_chunk_without_a_vector(store):
    doc = Document.objects.create(title="d", status="ready")
    Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="orphan row")
    output = _run()
    assert "1 chunk" in output.lower()
    assert "missing" in output.lower()


def test_detects_vector_without_a_chunk(store):
    store.upsert(["99_0"], [[0.5] * 768], [{"document_id": 99, "chunk_index": 0}])
    output = _run()
    assert "1 vector" in output.lower()
    assert "orphan" in output.lower()


def test_fix_removes_orphaned_vectors(store):
    store.upsert(["99_0"], [[0.5] * 768], [{"document_id": 99, "chunk_index": 0}])
    _run("--fix")
    assert store.count() == 0


def test_fix_marks_documents_with_missing_vectors_as_failed(store):
    doc = Document.objects.create(title="d", status="ready")
    Chunk.objects.create(document=doc, chunk_index=0, page_number=1, text="orphan row")
    _run("--fix")
    doc.refresh_from_db()
    assert doc.status == "failed"
    assert "re-upload" in doc.error_message.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/integration/test_reconcile.py -v`
Expected: FAIL — `CommandError: Unknown command: 'reconcile_vectors'`

- [ ] **Step 3: Write the command**

`backend/documents/management/commands/reconcile_vectors.py`:

```python
"""Detects and repairs drift between SQLite chunks and Chroma vectors.

Two stores cannot share a transaction (spec 10), so a crash mid-ingest can
leave either side ahead. This is the repair path.
"""
from django.core.management.base import BaseCommand

from documents.models import Chunk, Document
from documents.services import get_store

REUPLOAD_MESSAGE = "Vectors are missing for this document. Please delete and re-upload it."


class Command(BaseCommand):
    help = "Report (and optionally repair) drift between SQLite chunks and Chroma vectors."

    def add_arguments(self, parser):
        parser.add_argument("--fix", action="store_true", help="Repair the drift, not just report it.")

    def handle(self, *args, **options):
        store = get_store()
        vector_ids = store.all_ids()
        chunks = {
            c.vector_id: c
            for c in Chunk.objects.only("id", "document_id", "chunk_index")
        }

        missing_vectors = set(chunks) - vector_ids     # chunk row, no vector
        orphan_vectors = vector_ids - set(chunks)      # vector, no chunk row

        if not missing_vectors and not orphan_vectors:
            self.stdout.write(self.style.SUCCESS("No drift: SQLite and Chroma agree."))
            return

        self.stdout.write(f"{len(missing_vectors)} chunk(s) missing a vector")
        self.stdout.write(f"{len(orphan_vectors)} orphan vector(s) with no chunk row")

        if not options["fix"]:
            self.stdout.write(self.style.WARNING("Run with --fix to repair."))
            return

        # Delete exactly the orphaned ids — never a whole document. Deleting by
        # document_id would take that document's valid vectors with it, turning
        # a harmless orphan into an unsearchable `ready` document: strictly
        # worse than the drift being repaired. It also removes the id parse,
        # so a malformed id can no longer abort the run.
        if orphan_vectors:
            store.delete_ids(sorted(orphan_vectors))
            still_present = store.all_ids() & orphan_vectors
            self.stdout.write(
                f"removed {len(orphan_vectors) - len(still_present)} orphan vector(s)"
            )
            if still_present:
                self.stdout.write(
                    self.style.WARNING(f"{len(still_present)} orphan(s) could not be removed")
                )

        # Independent of the orphan cleanup above. A document that reads `ready`
        # while being unsearchable is the more damaging drift, and must not be
        # left unmarked because orphan removal had a problem.
        affected = {chunks[v].document_id for v in missing_vectors}
        if affected:
            Document.objects.filter(id__in=affected).update(
                status="failed", error_message=REUPLOAD_MESSAGE
            )
            self.stdout.write(f"marked {len(affected)} document(s) failed")

        self.stdout.write(self.style.SUCCESS("Repair complete."))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/integration/test_reconcile.py -v`
Expected: PASS — 5 tests

- [ ] **Step 5: Commit**

```bash
git add backend/documents/management/ backend/tests/integration/test_reconcile.py
git commit -m "feat: add reconcile_vectors command for two-store drift repair"
```

---

## Task 10: FTS5 query sanitising and lexical search

**Files:**
- Create: `backend/rag/lexical.py`, `backend/chat/lexical_search.py`
- Test: `backend/tests/unit/test_lexical.py`, `backend/tests/integration/test_lexical_search.py`

**Interfaces:**
- Consumes: `documents.models.Chunk`
- Produces: `rag.lexical.build_fts_query(question: str) -> str`; `chat.lexical_search.search(question: str, limit: int) -> list[str]` returning **vector ids** (`"{document_id}_{chunk_index}"`) in best-first order

- [ ] **Step 1: Write the failing tests**

`backend/tests/unit/test_lexical.py`:

```python
import pytest

from rag.lexical import build_fts_query


def test_simple_question_becomes_or_of_quoted_terms():
    assert build_fts_query("metformin dose") == '"metformin" OR "dose"'


@pytest.mark.parametrize(
    "question",
    [
        "What's the max dose?",
        'He said "take two"',
        "dose*",
        "metformin - adult",
        "dose NEAR adult",
        "metformin AND atenolol",
        "a OR b NOT c",
        "50% w/v (10:1)",
        "^caret $dollar",
    ],
)
def test_fts_syntax_characters_never_survive_sanitising(question):
    """Raw questions raise `fts5: syntax error`; this is the guard."""
    result = build_fts_query(question)
    unquoted = result.replace('"', "").replace(" OR ", " ")
    assert all(ch.isalnum() or ch.isspace() for ch in unquoted), result


def test_reserved_words_are_quoted_so_they_are_literals():
    result = build_fts_query("dose NEAR adult")
    assert '"near"' in result
    assert " NEAR " not in result


def test_single_character_terms_are_dropped_as_noise():
    assert build_fts_query("a b metformin") == '"metformin"'


def test_terms_are_deduplicated_preserving_first_occurrence():
    assert build_fts_query("dose dose metformin dose") == '"dose" OR "metformin"'


def test_empty_or_punctuation_only_question_yields_empty_string():
    assert build_fts_query("") == ""
    assert build_fts_query("???  !!!") == ""
    assert build_fts_query("a") == ""


def test_unicode_terms_are_preserved():
    assert '"naïve"' in build_fts_query("naïve dosing")


def test_numbers_are_kept_because_dosages_matter():
    assert '"500mg"' in build_fts_query("is it 500mg")
```

`backend/tests/integration/test_lexical_search.py`:

```python
import pytest

from chat.lexical_search import search
from documents.models import Chunk, Document

pytestmark = pytest.mark.django_db


@pytest.fixture
def seeded():
    doc = Document.objects.create(title="Monograph", status="ready")
    Chunk.objects.create(document=doc, chunk_index=0, page_number=1,
                         text="Metformin adult starting dose is 500mg twice daily.")
    Chunk.objects.create(document=doc, chunk_index=1, page_number=2,
                         text="Atenolol is a beta blocker used for hypertension.")
    return doc


def test_search_returns_vector_ids_best_first(seeded):
    assert search("metformin dose", limit=5)[0] == f"{seeded.id}_0"


def test_search_matches_stemmed_terms(seeded):
    assert f"{seeded.id}_0" in search("dosing", limit=5)


def test_search_respects_the_limit(seeded):
    assert len(search("metformin atenolol dose", limit=1)) == 1


def test_punctuation_heavy_question_does_not_raise(seeded):
    assert search("What's the max dose (mg/kg)?", limit=5)


def test_question_with_no_usable_terms_returns_empty(seeded):
    assert search("??? !!!", limit=5) == []


def test_search_on_empty_corpus_returns_empty():
    assert search("metformin", limit=5) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_lexical.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.lexical'`

- [ ] **Step 3: Write `rag/lexical.py`**

```python
"""FTS5 query construction.

Raw user questions contain characters FTS5 parses as query syntax — quotes,
`*`, `-`, `:`, and bare keywords like NEAR/AND/OR/NOT. Passing a question
straight into MATCH raises `fts5: syntax error` on input as ordinary as
"What's the max dose?" (spec 6.2). Every term is reduced to alphanumerics
and quoted, which also neutralises the reserved words.
"""
from __future__ import annotations

import re

# A decimal number with an optional unit suffix is ONE token. Splitting
# "0.5mg" into ["0", "5mg"] and dropping the orphaned "0" made a pediatric
# 0.5mg question byte-identical to an adult 5mg one — a dosage confusion
# originating in the tokenizer. The decimal branch must come first.
TOKEN_RE = re.compile(r"\d+(?:\.\d+)+\w*|\w+", re.UNICODE)
MIN_TERM_LENGTH = 2
MAX_TERMS = 24

# Function words carry no retrieval signal but make every question match almost
# every chunk once OR-joined. That would leave the confidence gate's
# `lexical_support` signal permanently True and collapse its middle band.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have
    how i if in into is it its may might must of on or shall should that the their
    them then there these they this to was were what when where which who why will
    with would you your
    """.split()
)


def build_fts_query(question: str) -> str:
    seen: list[str] = []
    for raw in TOKEN_RE.findall(question.lower()):
        if len(raw) < MIN_TERM_LENGTH or raw in STOPWORDS or raw in seen:
            continue
        seen.append(raw)
        if len(seen) >= MAX_TERMS:
            break
    return " OR ".join(f'"{term}"' for term in seen)
```

- [ ] **Step 4: Write `chat/lexical_search.py`**

```python
"""BM25 search over chunk text via SQLite FTS5.

bm25() returns NEGATIVE values where more-negative means a better match, so
results are ordered ascending. The magnitude is never compared to a cosine
similarity — fusion uses rank position only (spec 6.2).
"""
from __future__ import annotations

from django.db import connection

from rag.lexical import build_fts_query

SQL = """
    SELECT c.document_id, c.chunk_index
    FROM chunk_fts f
    JOIN documents_chunk c ON c.id = f.rowid
    WHERE chunk_fts MATCH %s
    ORDER BY bm25(chunk_fts)
    LIMIT %s
"""


def search(question: str, limit: int) -> list[str]:
    # FTS5 auxiliary functions (bm25/highlight/snippet) do not resolve table
    # aliases — only the literal table name — so `bm25(f)` raises
    # "no such column: f". The `f` alias is fine on the JOIN's f.rowid.
    if not isinstance(limit, int) or limit <= 0:
        return []
    expression = build_fts_query(question)
    if not expression:
        return []
    with connection.cursor() as cursor:
        cursor.execute(SQL, [expression, limit])
        return [f"{document_id}_{chunk_index}" for document_id, chunk_index in cursor.fetchall()]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_lexical.py tests/integration/test_lexical_search.py -v`
Expected: PASS — 22 tests (16 unit including 9 parametrised cases, 6 integration)

- [ ] **Step 6: Commit**

```bash
git add backend/rag/lexical.py backend/chat/lexical_search.py backend/tests/
git commit -m "feat: add FTS5 lexical search with input sanitising

Unsanitised questions raise fts5 syntax errors on ordinary punctuation."
```

---

## Task 11: Reciprocal rank fusion

**Files:**
- Create: `backend/rag/fusion.py`
- Test: `backend/tests/unit/test_fusion.py`

**Interfaces:**
- Consumes: nothing
- Produces: `FusedHit(chunk_id: str, score: float)`, `reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[FusedHit]`

- [ ] **Step 1: Write the failing tests**

`backend/tests/unit/test_fusion.py`:

```python
import pytest

from rag.fusion import FusedHit, reciprocal_rank_fusion


def test_identical_rankings_preserve_order():
    ranking = ["a", "b", "c"]
    assert [h.chunk_id for h in reciprocal_rank_fusion([ranking, ranking])] == ["a", "b", "c"]


def test_scores_match_the_rrf_formula():
    hits = reciprocal_rank_fusion([["a", "b"]], k=60)
    assert hits[0].score == pytest.approx(1 / 61)
    assert hits[1].score == pytest.approx(1 / 62)


def test_appearing_in_both_legs_beats_a_better_rank_in_one():
    """The whole point of hybrid retrieval: agreement outranks a single strong hit."""
    vector = ["x", "agreed"]
    lexical = ["y", "agreed"]
    order = [h.chunk_id for h in reciprocal_rank_fusion([vector, lexical], k=60)]
    assert order[0] == "agreed"


def test_disjoint_rankings_interleave_by_rank():
    order = [h.chunk_id for h in reciprocal_rank_fusion([["a", "b"], ["c", "d"]], k=60)]
    assert set(order[:2]) == {"a", "c"}
    assert set(order[2:]) == {"b", "d"}


def test_one_empty_leg_returns_the_other_unchanged():
    assert [h.chunk_id for h in reciprocal_rank_fusion([["a", "b"], []])] == ["a", "b"]


def test_all_empty_returns_empty():
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([]) == []


def test_ties_break_deterministically_by_chunk_id():
    first = [h.chunk_id for h in reciprocal_rank_fusion([["b", "a"], ["a", "b"]])]
    second = [h.chunk_id for h in reciprocal_rank_fusion([["b", "a"], ["a", "b"]])]
    assert first == second


def test_returns_fused_hit_instances():
    assert isinstance(reciprocal_rank_fusion([["a"]])[0], FusedHit)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_fusion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.fusion'`

- [ ] **Step 3: Write `rag/fusion.py`**

```python
"""Reciprocal rank fusion.

RRF compares only positions in ranked lists, never raw scores. This matters
because the two legs produce incompatible scales — cosine distance in [0, 2]
and BM25 in negative arbitrary units — with no principled normalisation
between them (spec 6.2).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

DEFAULT_K = 60


@dataclass(frozen=True)
class FusedHit:
    chunk_id: str
    score: float


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = DEFAULT_K) -> list[FusedHit]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [FusedHit(chunk_id=chunk_id, score=score) for chunk_id, score in ordered]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_fusion.py -v`
Expected: PASS — 8 tests

- [ ] **Step 5: Commit**

```bash
git add backend/rag/fusion.py backend/tests/unit/test_fusion.py
git commit -m "feat: add reciprocal rank fusion for hybrid retrieval"
```

---

## Task 12: Stage-1 confidence gate

**Files:**
- Create: `backend/rag/gate.py`
- Test: `backend/tests/unit/test_gate.py`

**Interfaces:**
- Consumes: `rag.config.GateConfig`
- Produces: `GateSignals(top_similarity, mean_similarity, lexical_support, corpus_empty)`, `GateDecision(proceed, reason, signals)`, `evaluate_gate(signals, cfg) -> GateDecision`, `REASONS`

- [ ] **Step 1: Write the failing tests**

`backend/tests/unit/test_gate.py`:

```python
import pytest

from rag.config import GateConfig
from rag.gate import GateDecision, GateSignals, evaluate_gate

CFG = GateConfig(tau_abstain=0.30, tau_strong=0.45)


def signals(top=0.9, mean=0.8, lexical=True, empty=False) -> GateSignals:
    return GateSignals(
        top_similarity=top, mean_similarity=mean, lexical_support=lexical, corpus_empty=empty
    )


def test_empty_corpus_declines_before_any_other_check():
    decision = evaluate_gate(signals(top=0.99, empty=True), CFG)
    assert decision.proceed is False
    assert decision.reason == "empty_corpus"


def test_similarity_below_tau_abstain_is_off_domain():
    decision = evaluate_gate(signals(top=0.10), CFG)
    assert decision.proceed is False
    assert decision.reason == "off_domain"


def test_high_similarity_proceeds():
    decision = evaluate_gate(signals(top=0.80), CFG)
    assert decision.proceed is True
    assert decision.reason == "ok"


def test_middle_band_without_lexical_support_declines():
    decision = evaluate_gate(signals(top=0.40, lexical=False), CFG)
    assert decision.proceed is False
    assert decision.reason == "weak_unsupported"


def test_middle_band_with_lexical_support_proceeds():
    assert evaluate_gate(signals(top=0.40, lexical=True), CFG).proceed is True


def test_lexical_support_cannot_rescue_below_tau_abstain():
    """Off-domain is off-domain even if a stray word matched."""
    decision = evaluate_gate(signals(top=0.05, lexical=True), CFG)
    assert decision.reason == "off_domain"


@pytest.mark.parametrize("top,expected", [(0.2999, "off_domain"), (0.30, "weak_unsupported")])
def test_tau_abstain_boundary_is_inclusive_upward(top, expected):
    assert evaluate_gate(signals(top=top, lexical=False), CFG).reason == expected


@pytest.mark.parametrize("top,expected", [(0.4499, "weak_unsupported"), (0.45, "ok")])
def test_tau_strong_boundary_is_inclusive_upward(top, expected):
    assert evaluate_gate(signals(top=top, lexical=False), CFG).reason == expected


def test_mean_similarity_does_not_affect_the_decision():
    """Pins the documented v1 behaviour (spec 6.5). If mean_similarity is
    ever promoted to a rule, this test must fail rather than the change
    passing unnoticed."""
    low = evaluate_gate(signals(top=0.80, mean=0.01), CFG)
    high = evaluate_gate(signals(top=0.80, mean=0.79), CFG)
    assert low.proceed is True and high.proceed is True
    assert low.reason == high.reason


def test_signals_are_carried_on_the_decision_for_observability():
    decision = evaluate_gate(signals(top=0.8, mean=0.7, lexical=True), CFG)
    assert decision.signals["top_similarity"] == 0.8
    assert decision.signals["mean_similarity"] == 0.7
    assert decision.signals["lexical_support"] is True


def test_decision_is_frozen():
    decision = evaluate_gate(signals(), CFG)
    assert isinstance(decision, GateDecision)
    with pytest.raises(Exception):
        decision.proceed = False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.gate'`

- [ ] **Step 3: Write `rag/gate.py`**

```python
"""Stage-1 confidence gate.

Runs before the LLM, so a clearly off-domain question costs nothing. Pure:
no I/O, no logging, no clock — which is what lets the eval harness sweep it
offline over thousands of parameter combinations (spec 6.5).
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import GateConfig

REASONS = ("ok", "off_domain", "weak_unsupported", "empty_corpus")


@dataclass(frozen=True)
class GateSignals:
    top_similarity: float
    mean_similarity: float
    lexical_support: bool
    corpus_empty: bool = False

    def as_dict(self) -> dict:
        return {
            "top_similarity": self.top_similarity,
            "mean_similarity": self.mean_similarity,
            "lexical_support": self.lexical_support,
            "corpus_empty": self.corpus_empty,
        }


@dataclass(frozen=True)
class GateDecision:
    proceed: bool
    reason: str
    signals: dict


def evaluate_gate(signals: GateSignals, cfg: GateConfig) -> GateDecision:
    payload = signals.as_dict()

    if signals.corpus_empty:
        return GateDecision(False, "empty_corpus", payload)

    if signals.top_similarity < cfg.tau_abstain:
        return GateDecision(False, "off_domain", payload)

    # Middle band: semantically plausible but not confident. Require that the
    # question's actual terminology appears in the retrieved text. This is what
    # separates "metformin dosing" (documented) from "metformin pediatric
    # dosing" (absent) — both sit at nearly identical cosine distance.
    if signals.top_similarity < cfg.tau_strong and not signals.lexical_support:
        return GateDecision(False, "weak_unsupported", payload)

    return GateDecision(True, "ok", payload)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_gate.py -v`
Expected: PASS — 13 tests

- [ ] **Step 5: Commit**

```bash
git add backend/rag/gate.py backend/tests/unit/test_gate.py
git commit -m "feat: add stage-1 confidence gate

Pure function over retrieval signals, run before the LLM. Thresholds are
placeholders until the Phase 3 eval sweep."
```

---

## Task 13: Prompts, sentinel, and decline copy

**Files:**
- Create: `backend/rag/prompts.py`
- Test: `backend/tests/unit/test_prompts.py`

**Interfaces:**
- Consumes: nothing
- Produces: `SENTINEL`, `DECLINE_COPY: dict[str, str]`, `ContextChunk(chunk_id, title, page_number, text)`, `build_messages(question, chunks, history) -> list[dict]`, `decline_text(reason) -> str`

- [ ] **Step 1: Write the failing tests**

`backend/tests/unit/test_prompts.py`:

```python
import pytest

from rag.gate import REASONS
from rag.prompts import (
    DECLINE_COPY,
    SENTINEL,
    ContextChunk,
    build_messages,
    decline_text,
)


def _chunks(n=2):
    return [
        ContextChunk(chunk_id=f"1_{i}", title="Monograph", page_number=i + 1, text=f"body {i}")
        for i in range(n)
    ]


def test_sentinel_is_the_exact_documented_string():
    assert SENTINEL == "INSUFFICIENT_CONTEXT"


def test_every_declining_gate_reason_has_copy():
    for reason in REASONS:
        if reason == "ok":
            continue
        assert reason in DECLINE_COPY
    assert "insufficient_context" in DECLINE_COPY


def test_decline_copy_is_distinct_per_reason():
    values = list(DECLINE_COPY.values())
    assert len(values) == len(set(values)), "each reason needs distinguishable copy"


def test_decline_text_falls_back_without_raising():
    assert decline_text("some_unknown_reason")


def test_empty_corpus_copy_mentions_uploading():
    assert "upload" in DECLINE_COPY["empty_corpus"].lower()


def test_insufficient_context_copy_blames_the_source_not_the_user():
    copy = DECLINE_COPY["insufficient_context"].lower()
    assert "don't contain" in copy or "not contain" in copy


def test_system_message_instructs_the_sentinel():
    messages = build_messages("q", _chunks(), history=[])
    assert messages[0]["role"] == "system"
    assert SENTINEL in messages[0]["content"]


def test_context_chunks_are_numbered_with_title_and_page():
    system = build_messages("q", _chunks(2), history=[])[0]["content"]
    assert "[1]" in system and "[2]" in system
    assert "Monograph" in system
    assert "p. 1" in system


def test_question_is_the_final_user_message():
    messages = build_messages("what is the dose?", _chunks(), history=[])
    assert messages[-1] == {"role": "user", "content": "what is the dose?"}


def test_history_is_included_between_system_and_question():
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]
    messages = build_messages("now", _chunks(), history=history)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "earlier"


def test_history_is_capped_by_message_count():
    history = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    messages = build_messages("now", _chunks(), history=history, max_history=4)
    assert len(messages) == 1 + 4 + 1
    assert messages[1]["content"] == "m6"      # keeps the most recent 4


def test_history_is_also_capped_by_character_budget():
    """Two long turns must not crowd out retrieved context in an 8B window."""
    history = [{"role": "user", "content": "x" * 5000} for _ in range(4)]
    messages = build_messages("now", _chunks(), history=history, max_history=4, history_chars=1000)
    body = "".join(m["content"] for m in messages[1:-1])
    assert len(body) <= 1000


def test_system_prompt_carries_the_medical_disclaimer():
    system = build_messages("q", _chunks(), history=[])[0]["content"]
    assert "not a substitute" in system.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.prompts'`

- [ ] **Step 3: Write `rag/prompts.py`**

```python
"""Prompt assembly, the refusal sentinel, and decline copy.

Declines are generated by the server, never by the model — that is what makes
them consistent, testable, and identical between the eval harness and the UI
(spec 6.8).
"""
from __future__ import annotations

from dataclasses import dataclass

SENTINEL = "INSUFFICIENT_CONTEXT"

DECLINE_COPY = {
    "empty_corpus": (
        "No documents have been uploaded yet. Upload a medical reference document "
        "and I'll answer questions grounded in it."
    ),
    "off_domain": (
        "I can only answer questions grounded in the medical documents you've "
        "uploaded, and this question doesn't relate to them."
    ),
    "weak_unsupported": (
        "I found some possibly related material, but not close enough to answer "
        "this reliably. Try rephrasing, or upload a document that covers it."
    ),
    "insufficient_context": (
        "Your uploaded documents cover this topic, but don't contain enough detail "
        "to answer this specific question. I'd rather decline than guess."
    ),
}

FALLBACK_DECLINE = (
    "I can't answer that from the documents you've uploaded."
)

SYSTEM_TEMPLATE = """You are a clinical reference assistant for the user's uploaded medical documents.

Answer only using the context below, drawn from documents the user has uploaded.
Do not draw on general knowledge beyond what is in the context.

If the context does not contain enough information to answer the question, reply with
exactly {sentinel} and nothing else. Do not apologise, explain, or add any other text.

When you do answer, cite the numbered sources you used, like [1] or [2].
Your answers are for informational reference only and are not a substitute for
professional medical judgment.

Context:
{context}"""


@dataclass(frozen=True)
class ContextChunk:
    chunk_id: str
    title: str
    page_number: int
    text: str


def decline_text(reason: str) -> str:
    return DECLINE_COPY.get(reason, FALLBACK_DECLINE)


def format_context(chunks: list[ContextChunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.title}, p. {c.page_number})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    )


def _trim_history(history: list[dict], max_history: int, history_chars: int) -> list[dict]:
    recent = history[-max_history:] if max_history else []
    kept: list[dict] = []
    budget = history_chars
    for message in reversed(recent):
        cost = len(message["content"])
        if cost > budget:
            break
        budget -= cost
        kept.append(message)
    return list(reversed(kept))


def build_messages(
    question: str,
    chunks: list[ContextChunk],
    history: list[dict],
    max_history: int = 4,
    history_chars: int = 2000,
) -> list[dict]:
    system = SYSTEM_TEMPLATE.format(sentinel=SENTINEL, context=format_context(chunks))
    return [
        {"role": "system", "content": system},
        *_trim_history(history, max_history, history_chars),
        {"role": "user", "content": question},
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_prompts.py -v`
Expected: PASS — 13 tests

- [ ] **Step 5: Commit**

```bash
git add backend/rag/prompts.py backend/tests/unit/test_prompts.py
git commit -m "feat: add prompt assembly, refusal sentinel, and decline copy

Declines are server-generated so the eval harness and UI assert the
same strings."
```

---

## Task 14: Ollama chat streaming and sentinel filtering

**Files:**
- Create: `backend/rag/generation.py`
- Test: `backend/tests/unit/test_generation.py`, add to `backend/tests/contract/test_ollama_contract.py`

**Interfaces:**
- Consumes: `rag.config.OllamaConfig`, `rag.prompts.SENTINEL`
- Produces: `stream_chat(cfg, messages, transport=None) -> Iterator[str]`, `filter_sentinel(deltas, sentinel=SENTINEL, buffer_chars=40) -> Iterator[tuple[str, str | None]]` yielding `("token", text)` or a single `("declined", None)`

- [ ] **Step 1: Write the failing tests**

`backend/tests/unit/test_generation.py`:

```python
import pytest

from rag.config import OllamaConfig
from rag.generation import filter_sentinel, stream_chat
from rag.prompts import SENTINEL


def events(deltas, **kw):
    return list(filter_sentinel(iter(deltas), **kw))


def test_normal_stream_emits_all_text():
    result = events(["Metformin ", "is ", "a biguanide."])
    assert [kind for kind, _ in result] == ["token"] * len(result)
    assert "".join(text for _, text in result) == "Metformin is a biguanide."


def test_clean_sentinel_yields_a_single_declined_event():
    assert events([SENTINEL]) == [("declined", None)]


def test_sentinel_split_across_two_deltas_is_still_detected():
    """This is exactly how the check breaks in practice."""
    assert events(["INSUFF", "ICIENT_CONTEXT"]) == [("declined", None)]


def test_sentinel_split_across_many_tiny_deltas_is_detected():
    assert events(list(SENTINEL)) == [("declined", None)]


def test_sentinel_with_leading_whitespace_is_detected():
    assert events(["\n\n", SENTINEL]) == [("declined", None)]


def test_sentinel_like_prefix_that_is_not_the_sentinel_streams_normally():
    deltas = ["INSUFFICIENT data ", "was available in the monograph."]
    result = events(deltas)
    assert ("declined", None) not in result
    assert "".join(t for _, t in result) == "".join(deltas)


def test_text_mentioning_the_sentinel_later_is_not_a_decline():
    deltas = ["The adult dose is 500mg. Note: not " + SENTINEL]
    result = events(deltas)
    assert ("declined", None) not in result


def test_short_stream_ending_before_the_buffer_fills_is_flushed():
    assert events(["ok"]) == [("token", "ok")]


def test_empty_stream_yields_nothing():
    assert events([]) == []


def test_buffered_prefix_is_emitted_exactly_once():
    deltas = ["a" * 10, "b" * 40, "c" * 10]
    joined = "".join(text for _, text in events(deltas))
    assert joined == "".join(deltas)


def test_stream_chat_extracts_message_content_deltas():
    lines = [
        {"message": {"role": "assistant", "content": "Met"}, "done": False},
        {"message": {"role": "assistant", "content": "formin"}, "done": False},
        {"done": True},
    ]
    out = list(stream_chat(OllamaConfig(), [{"role": "user", "content": "q"}],
                           transport=lambda url, payload: iter(lines)))
    assert out == ["Met", "formin"]


def test_stream_chat_ignores_lines_without_content():
    lines = [{"done": False}, {"message": {"content": "x"}, "done": False}, {"done": True}]
    out = list(stream_chat(OllamaConfig(), [], transport=lambda url, payload: iter(lines)))
    assert out == ["x"]
```

Add to `backend/tests/contract/test_ollama_contract.py`:

```python
def test_real_ollama_chat_streams_content_deltas():
    from rag.generation import stream_chat

    deltas = list(
        stream_chat(
            OllamaConfig(),
            [{"role": "user", "content": "Reply with exactly: hello"}],
        )
    )
    assert len(deltas) >= 1
    assert "hello" in "".join(deltas).lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_generation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.generation'`

- [ ] **Step 3: Write `rag/generation.py`**

```python
"""Ollama chat streaming and stage-2 sentinel filtering.

The sentinel cannot be streamed to the browser and then retracted, so output
is buffered until there is enough text to decide (spec 6.6). The buffer costs
a few tokens of latency and is imperceptible.
"""
from __future__ import annotations

import json
from typing import Callable, Iterable, Iterator

import httpx

from .config import OllamaConfig
from .ollama import OllamaUnavailable
from .prompts import SENTINEL

BUFFER_CHARS = 40


def _http_stream(url: str, payload: dict) -> Iterator[dict]:
    try:
        with httpx.stream("POST", url, json=payload, timeout=300.0) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.strip():
                    yield json.loads(line)
    except httpx.HTTPError as exc:
        raise OllamaUnavailable(f"chat request failed: {exc}") from exc


def stream_chat(
    cfg: OllamaConfig,
    messages: list[dict],
    transport: Callable[[str, dict], Iterator[dict]] | None = None,
) -> Iterator[str]:
    send = transport or _http_stream
    payload = {"model": cfg.chat_model, "messages": messages, "stream": True}
    for chunk in send(f"{cfg.host}/api/chat", payload):
        content = (chunk.get("message") or {}).get("content")
        if content:
            yield content


def filter_sentinel(
    deltas: Iterable[str],
    sentinel: str = SENTINEL,
    buffer_chars: int = BUFFER_CHARS,
) -> Iterator[tuple[str, str | None]]:
    """Yield ('token', text) events, or exactly one ('declined', None).

    The sentinel commonly arrives split across deltas, so the decision waits
    until the buffer holds enough characters to be conclusive.
    """
    threshold = max(len(sentinel), buffer_chars)
    buffer = ""
    decided = False

    for delta in deltas:
        if decided:
            yield ("token", delta)
            continue
        buffer += delta
        if len(buffer) >= threshold:
            decided = True
            if buffer.lstrip().startswith(sentinel):
                yield ("declined", None)
                return
            yield ("token", buffer)
            buffer = ""

    if not decided and buffer:
        if buffer.lstrip().startswith(sentinel):
            yield ("declined", None)
        else:
            yield ("token", buffer)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_generation.py -v`
Expected: PASS — 12 tests

- [ ] **Step 5: Run the contract tests**

Run: `uv run pytest tests/contract -m ollama -v`
Expected: PASS — 3 tests. Requires `llama3.1:8b` to be pulled.

- [ ] **Step 6: Commit**

```bash
git add backend/rag/generation.py backend/tests/
git commit -m "feat: add Ollama chat streaming with sentinel filtering

Buffers the stream head so a sentinel is never emitted then retracted,
and detects it when split across deltas."
```

---

## Task 15: Chat models and retrieval orchestration

**Files:**
- Create: `backend/chat/models.py`, `backend/chat/retrieval.py`
- Test: `backend/tests/integration/test_retrieval.py`

**Interfaces:**
- Consumes: everything from Tasks 3–14
- Produces: `chat.models.ChatSession`, `chat.models.ChatMessage`; `chat.retrieval.retrieve(question, embedder, store, cfg) -> RetrievalResult(decision, chunks)` where `chunks: list[ContextChunk]`

- [ ] **Step 1: Write `chat/models.py`**

```python
import uuid

from django.db import models


class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class ChatMessage(models.Model):
    ROLE_CHOICES = [("user", "User"), ("assistant", "Assistant")]

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    retrieved_sources = models.JSONField(default=list, blank=True)
    was_declined = models.BooleanField(default=False)
    decline_reason = models.CharField(max_length=32, blank=True)
    truncated = models.BooleanField(default=False)
    gate_signals = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]
```

Then: `uv run python manage.py makemigrations chat`

- [ ] **Step 2: Write the failing tests**

`backend/tests/integration/test_retrieval.py`:

```python
import pytest

from chat.retrieval import retrieve
from documents.models import Chunk, Document
from rag.config import load_config

pytestmark = pytest.mark.django_db

CFG = load_config(env={})


@pytest.fixture
def seeded(chroma_store, fake_embedder):
    doc = Document.objects.create(title="Monograph", status="ready")
    chunks = [
        Chunk.objects.create(document=doc, chunk_index=0, page_number=1,
                             text="Metformin adult starting dose is 500mg twice daily."),
        Chunk.objects.create(document=doc, chunk_index=1, page_number=2,
                             text="Atenolol is a beta blocker for hypertension."),
    ]
    chroma_store.upsert(
        ids=[c.vector_id for c in chunks],
        embeddings=fake_embedder.embed_documents([c.text for c in chunks]),
        metadatas=[{"document_id": doc.id, "chunk_index": c.chunk_index} for c in chunks],
    )
    return doc


def test_empty_corpus_short_circuits_before_retrieval(chroma_store, fake_embedder):
    result = retrieve("anything", fake_embedder, chroma_store, CFG)
    assert result.decision.proceed is False
    assert result.decision.reason == "empty_corpus"
    assert result.chunks == []


def test_on_topic_question_proceeds_with_hydrated_chunks(seeded, chroma_store, fake_embedder):
    result = retrieve("metformin dose", fake_embedder, chroma_store, CFG)
    assert result.decision.proceed is True
    assert result.chunks
    assert "Metformin" in result.chunks[0].text
    assert result.chunks[0].title == "Monograph"
    assert result.chunks[0].page_number == 1


def test_off_domain_question_declines(seeded, chroma_store, fake_embedder):
    result = retrieve("what is the capital of france", fake_embedder, chroma_store, CFG)
    assert result.decision.proceed is False
    assert result.decision.reason == "off_domain"


def test_gate_signals_are_populated_for_observability(seeded, chroma_store, fake_embedder):
    result = retrieve("metformin dose", fake_embedder, chroma_store, CFG)
    assert set(result.decision.signals) == {
        "top_similarity", "mean_similarity", "lexical_support", "corpus_empty"
    }


def test_chunks_are_limited_to_top_k(seeded, chroma_store, fake_embedder):
    result = retrieve("metformin", fake_embedder, chroma_store, CFG)
    assert len(result.chunks) <= CFG.retrieval.top_k


def test_vector_ids_with_no_sqlite_row_are_dropped_not_crashed(
    seeded, chroma_store, fake_embedder
):
    """An orphaned vector must not break a query (spec 10)."""
    chroma_store.upsert(
        ["999_0"],
        fake_embedder.embed_documents(["metformin"]),
        [{"document_id": 999, "chunk_index": 0}],
    )
    result = retrieve("metformin dose", fake_embedder, chroma_store, CFG)
    assert all(c.chunk_id != "999_0" for c in result.chunks)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/integration/test_retrieval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chat.retrieval'`

- [ ] **Step 4: Write `chat/retrieval.py`**

```python
"""Hybrid retrieval plus the stage-1 gate.

The gate reads `top_similarity` from the VECTOR leg directly, not from fused
output: RRF deliberately discards score magnitude, so fused scores carry no
similarity information (spec 17).
"""
from __future__ import annotations

from dataclasses import dataclass

from documents.models import Chunk
from rag.config import RagConfig
from rag.fusion import reciprocal_rank_fusion
from rag.gate import GateDecision, GateSignals, evaluate_gate
from rag.prompts import ContextChunk

from .lexical_search import search as lexical_search


@dataclass(frozen=True)
class RetrievalResult:
    decision: GateDecision
    chunks: list[ContextChunk]


def _hydrate(vector_ids: list[str]) -> list[ContextChunk]:
    """Map vector ids back to chunk rows in one query, preserving rank order."""
    pairs = []
    for vector_id in vector_ids:
        document_id, _, chunk_index = vector_id.partition("_")
        if chunk_index.isdigit():
            pairs.append((int(document_id), int(chunk_index)))
    if not pairs:
        return []

    rows = Chunk.objects.filter(
        document_id__in={d for d, _ in pairs}, chunk_index__in={i for _, i in pairs}
    ).select_related("document")
    by_key = {(r.document_id, r.chunk_index): r for r in rows}

    hydrated = []
    for document_id, chunk_index in pairs:
        row = by_key.get((document_id, chunk_index))
        if row is None:
            continue          # orphaned vector: drop it silently
        hydrated.append(
            ContextChunk(
                chunk_id=row.vector_id,
                title=row.document.title,
                page_number=row.page_number,
                text=row.text,
            )
        )
    return hydrated


def retrieve(question: str, embedder, store, cfg: RagConfig) -> RetrievalResult:
    if store.count() == 0:
        signals = GateSignals(0.0, 0.0, lexical_support=False, corpus_empty=True)
        return RetrievalResult(evaluate_gate(signals, cfg.gate), [])

    vector_hits = store.query(embedder.embed_query(question), cfg.retrieval.per_leg)
    lexical_ids = lexical_search(question, cfg.retrieval.per_leg)

    similarities = [1.0 - hit.distance for hit in vector_hits]
    top_similarity = max(similarities) if similarities else 0.0
    mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    fused = reciprocal_rank_fusion(
        [[h.chunk_id for h in vector_hits], lexical_ids], k=cfg.retrieval.rrf_k
    )
    top_ids = [hit.chunk_id for hit in fused[: cfg.retrieval.top_k]]

    signals = GateSignals(
        top_similarity=top_similarity,
        mean_similarity=mean_similarity,
        lexical_support=bool(top_ids) and top_ids[0] in set(lexical_ids),
        corpus_empty=False,
    )
    decision = evaluate_gate(signals, cfg.gate)
    return RetrievalResult(decision, _hydrate(top_ids) if decision.proceed else [])
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/integration/test_retrieval.py -v`
Expected: PASS — 6 tests

- [ ] **Step 6: Commit**

```bash
git add backend/chat/ backend/tests/integration/test_retrieval.py
git commit -m "feat: add chat models and hybrid retrieval orchestration

Gate reads similarity from the vector leg directly since RRF discards
score magnitude."
```

---

## Task 16: NDJSON streaming chat endpoint

**Files:**
- Create: `backend/chat/streaming.py`
- Modify: `backend/chat/views.py`, `backend/chat/urls.py`
- Test: `backend/tests/unit/test_streaming.py`, `backend/tests/integration/test_chat_view.py`

**Interfaces:**
- Consumes: `chat.retrieval.retrieve`, `rag.generation.stream_chat`, `filter_sentinel`, `rag.prompts.decline_text`, `build_messages`
- Produces: `chat.streaming.frame(kind, **fields) -> str`; endpoint `POST /api/chat/`, `GET /api/chat/sessions/<uuid>/messages/`

- [ ] **Step 1: Write the failing unit tests**

`backend/tests/unit/test_streaming.py`:

```python
import json

from chat.streaming import frame


def test_frame_is_one_json_object_terminated_by_newline():
    line = frame("token", text="hi")
    assert line.endswith("\n")
    assert line.count("\n") == 1
    assert json.loads(line) == {"type": "token", "text": "hi"}


def test_frame_escapes_newlines_inside_values():
    """A literal newline in content must not split the frame."""
    line = frame("token", text="line one\nline two")
    assert line.count("\n") == 1
    assert json.loads(line)["text"] == "line one\nline two"


def test_frame_escapes_unicode_safely():
    assert json.loads(frame("token", text="naïve — 50µg"))["text"] == "naïve — 50µg"
```

- [ ] **Step 2: Write the failing integration tests**

`backend/tests/integration/test_chat_view.py`:

```python
import json

import pytest
from django.test import Client

from chat.models import ChatMessage, ChatSession
from documents.models import Chunk, Document
from rag.ollama import OllamaUnavailable
from rag.prompts import DECLINE_COPY, SENTINEL

pytestmark = pytest.mark.django_db


def read_frames(response) -> list[dict]:
    body = b"".join(response.streaming_content).decode()
    return [json.loads(line) for line in body.splitlines() if line.strip()]


@pytest.fixture
def wired(chroma_store, fake_embedder, monkeypatch):
    """Wire the view to a throwaway store, the shared fake embedder, and a
    scripted LLM. Mutate `script` in a test to change what the model returns."""
    import chat.views as views
    import documents.services as services

    monkeypatch.setattr(services, "get_store", lambda: chroma_store)
    monkeypatch.setattr(services, "get_embedder", lambda: fake_embedder)

    script = {"deltas": ["The adult dose ", "is 500mg [1]."]}

    def fake_stream(cfg, messages, transport=None):
        if script.get("raises"):
            raise OllamaUnavailable("connection refused")
        yield from script["deltas"]

    monkeypatch.setattr(views, "stream_chat", fake_stream)
    return chroma_store, script


@pytest.fixture
def seeded(wired, fake_embedder):
    store, _ = wired
    doc = Document.objects.create(title="Monograph", status="ready")
    chunk = Chunk.objects.create(
        document=doc, chunk_index=0, page_number=3,
        text="Metformin adult starting dose is 500mg twice daily.",
    )
    store.upsert(
        [chunk.vector_id],
        fake_embedder.embed_documents([chunk.text]),
        [{"document_id": doc.id, "chunk_index": 0}],
    )
    return doc


def _ask(question, session_id=None):
    payload = {"question": question, "session_id": session_id}
    return Client().post("/api/chat/", data=json.dumps(payload), content_type="application/json")


# --- path 1: answered ---------------------------------------------------

def test_answered_question_emits_meta_sources_tokens_done_in_order(seeded):
    frames = read_frames(_ask("metformin dose"))
    kinds = [f["type"] for f in frames]
    assert kinds[0] == "meta"
    assert kinds[1] == "sources"
    assert kinds[-1] == "done"
    assert "token" in kinds
    assert kinds.index("sources") < kinds.index("token")


def test_answered_response_carries_citation_metadata(seeded):
    sources = next(f for f in read_frames(_ask("metformin dose")) if f["type"] == "sources")
    assert sources["items"][0]["title"] == "Monograph"
    assert sources["items"][0]["page"] == 3
    assert sources["items"][0]["snippet"]


def test_answered_turn_persists_both_messages(seeded):
    _ask("metformin dose")
    roles = list(ChatMessage.objects.values_list("role", flat=True))
    assert roles == ["user", "assistant"]
    assistant = ChatMessage.objects.get(role="assistant")
    assert assistant.was_declined is False
    assert assistant.content == "The adult dose is 500mg [1]."
    assert assistant.gate_signals["top_similarity"] > 0


def test_done_frame_shape_is_fixed(seeded):
    done = read_frames(_ask("metformin dose"))[-1]
    assert set(done) == {"type", "message_id", "was_declined", "decline_reason", "truncated"}
    assert done["decline_reason"] is None


# --- path 2: stage-1 decline -------------------------------------------

def test_off_domain_question_declines_without_sources(seeded):
    frames = read_frames(_ask("what is the capital of france"))
    assert not any(f["type"] == "sources" for f in frames)
    assert frames[-1]["was_declined"] is True
    assert frames[-1]["decline_reason"] == "off_domain"


def test_stage_one_decline_never_calls_the_llm(seeded, wired):
    _, script = wired
    script["deltas"] = ["THIS MUST NOT APPEAR"]
    text = "".join(f.get("text", "") for f in read_frames(_ask("capital of france")))
    assert "MUST NOT APPEAR" not in text
    assert text == DECLINE_COPY["off_domain"]


def test_empty_corpus_declines_with_its_own_copy(wired):
    frames = read_frames(_ask("metformin dose"))
    assert frames[-1]["decline_reason"] == "empty_corpus"
    assert "".join(f.get("text", "") for f in frames) == DECLINE_COPY["empty_corpus"]


# --- path 3: stage-2 decline -------------------------------------------

def test_sentinel_response_becomes_a_decline_with_no_sources_leaked(seeded, wired):
    _, script = wired
    script["deltas"] = [SENTINEL]
    frames = read_frames(_ask("metformin pediatric dose"))
    assert frames[-1]["was_declined"] is True
    assert frames[-1]["decline_reason"] == "insufficient_context"
    assert "".join(f.get("text", "") for f in frames) == DECLINE_COPY["insufficient_context"]
    assert not any(f["type"] == "sources" for f in frames)


def test_sentinel_split_across_deltas_is_still_a_decline(seeded, wired):
    _, script = wired
    script["deltas"] = ["INSUFF", "ICIENT_CONTEXT"]
    assert read_frames(_ask("metformin pediatric dose"))[-1]["decline_reason"] == "insufficient_context"


def test_stage_two_decline_persists_as_declined(seeded, wired):
    _, script = wired
    script["deltas"] = [SENTINEL]
    _ask("metformin pediatric dose")
    assistant = ChatMessage.objects.get(role="assistant")
    assert assistant.was_declined is True
    assert assistant.retrieved_sources == []


# --- path 4: ollama down ------------------------------------------------

def test_ollama_failure_emits_an_error_frame(seeded, wired):
    _, script = wired
    script["raises"] = True
    frames = read_frames(_ask("metformin dose"))
    error = next(f for f in frames if f["type"] == "error")
    assert error["code"] == "ollama_unavailable"


def test_ollama_failure_persists_a_truncated_message(seeded, wired):
    _, script = wired
    script["raises"] = True
    _ask("metformin dose")
    assert ChatMessage.objects.get(role="assistant").truncated is True


# --- sessions -----------------------------------------------------------

def test_new_session_is_created_and_returned_in_meta(seeded):
    meta = read_frames(_ask("metformin dose"))[0]
    assert ChatSession.objects.filter(id=meta["session_id"]).exists()


def test_existing_session_is_reused(seeded):
    first = read_frames(_ask("metformin dose"))[0]["session_id"]
    second = read_frames(_ask("metformin dose", session_id=first))[0]["session_id"]
    assert first == second
    assert ChatSession.objects.count() == 1


def test_messages_endpoint_replays_history(seeded):
    session_id = read_frames(_ask("metformin dose"))[0]["session_id"]
    body = json.loads(Client().get(f"/api/chat/sessions/{session_id}/messages/").content)
    assert [m["role"] for m in body] == ["user", "assistant"]
    assert set(body[0]) >= {"role", "content", "retrieved_sources", "was_declined", "created_at"}


def test_blank_question_returns_400(seeded):
    assert _ask("   ").status_code == 400


def test_response_content_type_is_ndjson(seeded):
    assert _ask("metformin dose")["Content-Type"] == "application/x-ndjson"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_streaming.py tests/integration/test_chat_view.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chat.streaming'`

- [ ] **Step 4: Write `chat/streaming.py`**

```python
"""NDJSON frame construction.

One JSON object per line. `json.dumps` escapes embedded newlines, so answer
text containing line breaks cannot split a frame.
"""
from __future__ import annotations

import json


def frame(kind: str, **fields) -> str:
    return json.dumps({"type": kind, **fields}, ensure_ascii=False) + "\n"
```

- [ ] **Step 5: Write the chat view**

Append to `backend/chat/views.py`:

```python
import json

from django.http import Http404, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt

from documents import services
from rag.config import load_config
from rag.generation import filter_sentinel, stream_chat
from rag.ollama import OllamaError
from rag.prompts import build_messages, decline_text

from .models import ChatMessage, ChatSession
from .retrieval import retrieve
from .streaming import frame

SNIPPET_CHARS = 240


def _sources_payload(chunks) -> list[dict]:
    return [
        {
            "chunk_id": c.chunk_id,
            "document_id": int(c.chunk_id.split("_")[0]),
            "title": c.title,
            "page": c.page_number,
            "snippet": c.text[:SNIPPET_CHARS],
        }
        for c in chunks
    ]


def _history(session: ChatSession, limit: int) -> list[dict]:
    recent = session.messages.order_by("-created_at", "-id")[:limit]
    return [{"role": m.role, "content": m.content} for m in reversed(list(recent))]


def _persist(session, content, sources, declined, reason, signals, truncated) -> ChatMessage:
    return ChatMessage.objects.create(
        session=session,
        role="assistant",
        content=content,
        retrieved_sources=sources,
        was_declined=declined,
        decline_reason=reason or "",
        gate_signals=signals,
        truncated=truncated,
    )


@csrf_exempt
def chat(request):
    if request.method != "POST":
        return JsonResponse({"error": "method not allowed"}, status=405)

    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid JSON body"}, status=400)

    question = (payload.get("question") or "").strip()
    if not question:
        return JsonResponse({"error": "question is required"}, status=400)

    cfg = load_config()
    session_id = payload.get("session_id")
    session = (
        ChatSession.objects.filter(id=session_id).first() if session_id else None
    ) or ChatSession.objects.create(title=question[:80])

    history = _history(session, cfg.history_messages)
    ChatMessage.objects.create(session=session, role="user", content=question)

    result = retrieve(question, services.get_embedder(), services.get_store(), cfg)

    def generate():
        yield frame("meta", session_id=str(session.id))

        signals = result.decision.signals

        # Stage 1 declined: the LLM is never called.
        if not result.decision.proceed:
            text = decline_text(result.decision.reason)
            yield frame("token", text=text)
            message = _persist(session, text, [], True, result.decision.reason, signals, False)
            yield frame(
                "done",
                message_id=message.id,
                was_declined=True,
                decline_reason=result.decision.reason,
                truncated=False,
            )
            return

        messages = build_messages(
            question,
            result.chunks,
            history,
            max_history=cfg.history_messages,
        )

        collected: list[str] = []
        declined = False
        truncated = False
        sources_sent = False

        try:
            for kind, text in filter_sentinel(stream_chat(cfg.ollama, messages)):
                if kind == "declined":
                    declined = True
                    break
                if not sources_sent:
                    # Emitted only now: both gates have cleared (spec 7.1).
                    yield frame("sources", items=_sources_payload(result.chunks))
                    sources_sent = True
                collected.append(text)
                yield frame("token", text=text)
        except OllamaError as exc:
            truncated = True
            # Ollama answers 404 "model ... not found" when the tag was never
            # pulled — a different fix for the user than a dead server (spec 11).
            code = (
                "model_missing" if "not found" in str(exc).lower() else "ollama_unavailable"
            )
            yield frame("error", code=code, message=str(exc))
        finally:
            if declined:
                body = decline_text("insufficient_context")
                yield frame("token", text=body)
                message = _persist(
                    session, body, [], True, "insufficient_context", signals, False
                )
                yield frame(
                    "done",
                    message_id=message.id,
                    was_declined=True,
                    decline_reason="insufficient_context",
                    truncated=False,
                )
            else:
                # Runs even on client disconnect, so partial answers survive.
                message = _persist(
                    session,
                    "".join(collected),
                    _sources_payload(result.chunks) if sources_sent else [],
                    False,
                    "",
                    signals,
                    truncated,
                )
                yield frame(
                    "done",
                    message_id=message.id,
                    was_declined=False,
                    decline_reason=None,
                    truncated=truncated,
                )

    response = StreamingHttpResponse(generate(), content_type="application/x-ndjson")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def session_messages(request, session_id):
    session = get_object_or_404(ChatSession, pk=session_id)
    return JsonResponse(
        [
            {
                "role": m.role,
                "content": m.content,
                "retrieved_sources": m.retrieved_sources,
                "was_declined": m.was_declined,
                "decline_reason": m.decline_reason,
                "created_at": m.created_at.isoformat(),
            }
            for m in session.messages.all()
        ],
        safe=False,
    )
```

Update `backend/chat/urls.py`:

```python
from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.health, name="health"),
    path("chat/", views.chat, name="chat"),
    path("chat/sessions/<uuid:session_id>/messages/", views.session_messages, name="session-messages"),
]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_streaming.py tests/integration/test_chat_view.py -v`
Expected: PASS — 21 tests

- [ ] **Step 7: Run the whole suite**

Run: `uv run pytest -v`
Expected: PASS — all tests, `ollama`-marked deselected.

Run: `uv run pytest -m ollama -v`
Expected: PASS — 3 contract tests against real Ollama.

- [ ] **Step 8: Verify end-to-end with curl**

```bash
uv run python manage.py migrate
uv run python -c "
import pathlib, sys; sys.path.insert(0, 'tests')
from fixtures.make_fixture_pdf import make_pdf
make_pdf(pathlib.Path('/tmp/sample.pdf'), [
  'Metformin: the adult starting dose is 500mg taken twice daily with meals.',
  'Atenolol: 50mg once daily for hypertension in adults.',
])"
uv run uvicorn medical_rag.asgi:application --port 8000 &
curl -s -F "file=@/tmp/sample.pdf" localhost:8000/api/documents/ | python3 -m json.tool
curl -N -s localhost:8000/api/chat/ -H 'Content-Type: application/json' \
     -d '{"question":"what is the adult dose?","session_id":null}'
curl -N -s localhost:8000/api/chat/ -H 'Content-Type: application/json' \
     -d '{"question":"what is the capital of France?","session_id":null}'
```

Expected: the first question streams `token` frames after a `sources` frame; the second returns the `off_domain` decline with no `sources` frame. **This is the Phase 2 exit criterion.**

- [ ] **Step 9: Commit**

```bash
git add backend/chat/ backend/tests/
git commit -m "feat: add NDJSON streaming chat endpoint

Frame order enforces the design's core invariant: sources are emitted
only after both gates clear, so a stage-2 decline can never leave
citations on screen for an answer that never arrives."
```

---

## Verification Checklist

After Task 16, all of the following must hold:

- [ ] `uv run pytest` — all tests pass, `ollama` tests deselected
- [ ] `uv run pytest -m ollama` — 3 contract tests pass against real Ollama
- [ ] `curl` upload returns `status: "ready"` with a non-zero `chunk_count`
- [ ] An answerable question streams `meta` → `sources` → `token`… → `done`
- [ ] "Capital of France" streams `meta` → `token` → `done` with **no** `sources` frame
- [ ] `python manage.py reconcile_vectors` reports no drift
- [ ] `/api/health/` reports both models present

---

## Next Plans

- **Phase 3 — Eval:** fixture corpus, `questions.yaml`, `run_eval.py` threshold sweep, commit `eval_results.md`, replace the placeholder `tau_abstain`/`tau_strong` defaults with measured values.
- **Phase 4 — Chat UI:** Next.js NDJSON reader, streaming message list, citation chips, decline card, health banner.
- **Phase 5 — Hardening:** Playwright E2E against a fake Ollama, README architecture narrative.
