import json
import uuid

from django.http import JsonResponse, StreamingHttpResponse
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

    if not isinstance(payload, dict):
        return JsonResponse({"error": "body must be a JSON object"}, status=400)

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return JsonResponse({"error": "question must be a non-empty string"}, status=400)
    question = question.strip()

    cfg = load_config()
    session_id = payload.get("session_id")
    session = None
    if session_id is not None:
        try:
            session = ChatSession.objects.filter(id=uuid.UUID(str(session_id))).first()
        except (ValueError, AttributeError, TypeError):
            return JsonResponse({"error": "session_id must be a valid UUID"}, status=400)
    if session is None:
        session = ChatSession.objects.create(title=question[:80])

    history = _history(session, cfg.history_messages)
    ChatMessage.objects.create(session=session, role="user", content=question)

    def generate():
        yield frame("meta", session_id=str(session.id))

        try:
            # Runs INSIDE the generator, after `meta` is already on the wire.
            # retrieve() embeds the query via Ollama before doing anything
            # else, so an unreachable Ollama (or a missing model) surfaces
            # here first. By this point Django can no longer fall back to a
            # 500 page — headers are already committed — so an uncaught
            # exception here would just truncate the connection with no
            # valid NDJSON at all, which is worse than the debug page this
            # used to show when the call sat outside the generator.
            result = retrieve(question, services.get_embedder(), services.get_store(), cfg)
        except Exception as exc:
            # Ollama answers 404 "model ... not found" when the tag was never
            # pulled — a different fix for the user than a dead server (spec 11).
            code = (
                "model_missing" if "not found" in str(exc).lower() else "ollama_unavailable"
            )
            yield frame("error", code=code, message=str(exc))
            # History must stay coherent: the user turn above was already
            # committed, so the session needs a matching assistant turn even
            # though no answer was produced.
            message = _persist(session, "", [], False, "", {}, True)
            yield frame(
                "done",
                message_id=message.id,
                was_declined=False,
                decline_reason=None,
                truncated=True,
            )
            return

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
        decline_body = ""
        message = None

        try:
            for kind, text in filter_sentinel(stream_chat(cfg.ollama, messages)):
                if kind == "declined":
                    declined = True
                    decline_body = decline_text("insufficient_context")
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
            # Persistence lives here, and ONLY here, so it runs exactly once no
            # matter which of the four ways generate() ends: normal completion,
            # a stage-2 sentinel decline (the `break` above), a handled
            # OllamaError (the `except` above), or a client disconnect that
            # closes this generator mid-stream. A `finally` block runs exactly
            # once per generator teardown, there is a single `_persist` call
            # site here, and this block never yields — so there is no flag to
            # race and no second write to trigger. (A `persisted` flag set
            # *after* yielding the frames would not be safe: if the generator
            # is closed at exactly that yield, GeneratorExit fires before the
            # flag is set, `finally` still runs, and sees the flag unset —
            # a second, spurious persist. Doing the write only here, before
            # anything downstream can observe or interrupt it, avoids that
            # race entirely.) On disconnect, Python throws GeneratorExit at
            # whatever `yield` was in flight; that propagates through this
            # `finally` (which neither catches it nor yields) straight out of
            # generate(), so the code below never runs and no frame is
            # written after teardown has started.
            if declined:
                message = _persist(
                    session, decline_body, [], True, "insufficient_context", signals, False
                )
            else:
                message = _persist(
                    session,
                    "".join(collected),
                    _sources_payload(result.chunks) if sources_sent else [],
                    False,
                    "",
                    signals,
                    truncated,
                )

        # Only reached if the try/finally above completed without propagating
        # an exception. On a client disconnect it is skipped entirely (see the
        # comment in `finally`) — correctly, since nobody is listening on a
        # closed connection.
        if declined:
            yield frame("token", text=decline_body)
            yield frame(
                "done",
                message_id=message.id,
                was_declined=True,
                decline_reason="insufficient_context",
                truncated=False,
            )
        else:
            yield frame(
                "done",
                message_id=message.id,
                was_declined=False,
                decline_reason=None,
                truncated=truncated,
            )

    # Served under WSGI. StreamingHttpResponse cannot async-iterate a SYNC
    # generator, so under ASGI Django drains the whole generator in a threadpool
    # before sending anything — measured: every token arriving at once at 9.5s
    # versus progressive delivery starting at 0.7s under WSGI. WSGI also calls
    # close() on client disconnect, which is what makes the persistence below
    # reliable; ASGI never delivers GeneratorExit for a sync generator.
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
