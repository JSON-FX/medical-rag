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


def test_answered_turn_persists_assistant_message_exactly_once(seeded):
    """Guards the `finally`-block persistence in `chat.views.generate`: a
    naive fix could persist twice (once from a flag set post-yield, once as
    a disconnect safety net). Only one call site exists, so this must hold."""
    read_frames(_ask("metformin dose"))
    assert ChatMessage.objects.filter(role="assistant").count() == 1


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


def test_stage_two_decline_persists_assistant_message_exactly_once(seeded, wired):
    """Same guard as the answered path, for the branch that used to build the
    decline text and persist inside a `finally` that also yielded."""
    _, script = wired
    script["deltas"] = [SENTINEL]
    read_frames(_ask("metformin pediatric dose"))
    assert ChatMessage.objects.filter(role="assistant").count() == 1


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


# --- path 0: retrieval failure (Ollama unreachable while embedding) ----
#
# retrieve() used to run in the synchronous part of the view, outside any
# try, so an unreachable Ollama surfaced as a raw 500 Django debug page
# (reproduced live) instead of a well-formed NDJSON stream. It now runs
# inside generate(), after `meta` is already on the wire.


def _unreachable(*args, **kwargs):
    raise OllamaUnavailable("connection refused")


def test_retrieval_failure_returns_200_with_well_formed_frames(wired, monkeypatch):
    import chat.views as views

    monkeypatch.setattr(views, "retrieve", _unreachable)
    response = _ask("metformin dose")
    assert response.status_code == 200
    frames = read_frames(response)  # must not raise: no exception escapes
    assert frames[0]["type"] == "meta"
    assert frames[-1]["type"] == "done"


def test_retrieval_failure_emits_an_ollama_unavailable_error_frame(wired, monkeypatch):
    import chat.views as views

    monkeypatch.setattr(views, "retrieve", _unreachable)
    frames = read_frames(_ask("metformin dose"))
    error = next(f for f in frames if f["type"] == "error")
    assert error["code"] == "ollama_unavailable"
    assert frames[-1]["was_declined"] is False
    assert frames[-1]["truncated"] is True


def test_retrieval_failure_reports_model_missing_when_the_message_says_so(wired, monkeypatch):
    import chat.views as views

    def not_found(*args, **kwargs):
        raise OllamaUnavailable("model 'nomic-embed-text' not found, try pulling it first")

    monkeypatch.setattr(views, "retrieve", not_found)
    error = next(f for f in read_frames(_ask("metformin dose")) if f["type"] == "error")
    assert error["code"] == "model_missing"


def test_retrieval_failure_persists_a_truncated_assistant_message_exactly_once(wired, monkeypatch):
    import chat.views as views

    monkeypatch.setattr(views, "retrieve", _unreachable)
    read_frames(_ask("metformin dose"))
    assert ChatMessage.objects.filter(role="assistant").count() == 1
    assistant = ChatMessage.objects.get(role="assistant")
    assert assistant.truncated is True
    assert assistant.was_declined is False


def test_retrieval_failure_still_leaves_a_coherent_session_history(wired, monkeypatch):
    """The user turn is committed before generate() runs; a retrieval failure
    must not leave it orphaned with no matching assistant turn, since
    `session_messages` replays history in `created_at` order forever."""
    import chat.views as views

    monkeypatch.setattr(views, "retrieve", _unreachable)
    read_frames(_ask("metformin dose"))
    roles = list(ChatMessage.objects.values_list("role", flat=True))
    assert roles == ["user", "assistant"]


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


# --- malformed input ------------------------------------------------------
#
# These exercise input validation in `chat()`, which runs before retrieval or
# the LLM is ever touched, so they need neither `wired` nor `seeded` — a real
# `load_config()` runs, but it only reads env defaults (no network, no DB).

def test_non_string_question_returns_400():
    """`{"question": 5}` used to reach `.strip()` on an int and 500."""
    assert _ask(5).status_code == 400


def test_boolean_question_returns_400():
    """`isinstance(True, int)` is True but `isinstance(True, str)` is False,
    so this must be rejected the same way as any other non-string question."""
    assert _ask(True).status_code == 400


def test_non_dict_json_body_returns_400():
    """A bare JSON array has no `.get`, so this used to 500 with an
    AttributeError before reaching the `question` check at all."""
    response = Client().post(
        "/api/chat/", data=json.dumps([1, 2, 3]), content_type="application/json"
    )
    assert response.status_code == 400


def test_non_uuid_session_id_returns_400():
    """A `session_id` that isn't a UUID used to blow up inside the queryset
    filter with `ValueError: badly formed hexadecimal UUID string`."""
    assert _ask("metformin dose", session_id="not-a-uuid").status_code == 400
