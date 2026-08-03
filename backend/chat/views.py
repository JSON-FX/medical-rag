import json

from django.http import Http404, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

from documents import services
from rag.config import load_config
from rag.generation import filter_sentinel, stream_chat
from rag.ollama import OllamaClient, OllamaError, OllamaUnavailable
from rag.prompts import build_messages, decline_text

from .models import ChatMessage, ChatSession
from .retrieval import retrieve
from .streaming import frame

SNIPPET_CHARS = 240


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
