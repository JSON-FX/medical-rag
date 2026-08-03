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
    read_frames(_ask("metformin dose"))
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
    read_frames(_ask("metformin pediatric dose"))
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
    read_frames(_ask("metformin dose"))
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
