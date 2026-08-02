import json

from django.http import JsonResponse
from django.views.decorators.http import require_GET

from rag.config import load_config
from rag.ollama import OllamaClient, OllamaUnavailable


def build_client(cfg):
    """Indirection so tests can substitute a fake."""
    return OllamaClient(cfg.ollama)


def _has_model(available: list[str], wanted: str) -> bool:
    """`ollama list` reports `name:latest` for an untagged pull."""
    wanted_base = wanted.split(":")[0]
    return any(name == wanted or name.split(":")[0] == wanted_base for name in available)


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
