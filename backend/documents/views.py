import logging

from django.conf import settings
from django.db import transaction
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

    Full detail stays in the server log; the client gets the cause without
    the filesystem layout.
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
        return JsonResponse(
            {"error": f"ingestion services unavailable: {exc}"}, status=503
        )

    document = Document.objects.create(title=upload.name, file=upload)
    document = ingest_document(document, embedder, store, cfg)
    return JsonResponse(_serialize(document), status=201)


@csrf_exempt
def document_detail(request, document_id: int):
    if request.method != "DELETE":
        return JsonResponse({"error": "method not allowed"}, status=405)
    document = get_object_or_404(Document, pk=document_id)

    # The chunk delete and the document delete must commit together or not at
    # all. Without this, a crash between the two leaves a `ready` Document
    # with chunk_count > 0 but zero actual Chunk rows in SQLite — invisible
    # to reconcile_vectors, since no chunk rows means no missing vectors and
    # no orphans either. Chroma is a separate store that cannot share this
    # transaction (spec 10); if its delete fails or the process dies before
    # this block commits, SQLite rolls back to the pre-delete state instead
    # of the previously-possible split state, which reconcile_vectors can
    # already see as "chunks missing a vector".
    with transaction.atomic():
        cleanup_document(document.id, services.get_store())
        document.delete()

    # FileField.delete() is NOT called by Document.delete() (Django >= 1.3
    # stopped doing that automatically), so without this the PDF is orphaned
    # on disk forever — the wrong wart for an app whose selling point is
    # "nothing leaves your machine". Kept outside the transaction above: this
    # is filesystem I/O, not a DB write, so it cannot be rolled back and
    # would gain nothing from being inside one. save=False because the row is
    # already gone — saving would INSERT a new row with a blank file field
    # rather than update anything. Guarded for a Document with no file
    # attached and for a file already missing from disk (FieldFile.delete()
    # and FileSystemStorage.delete() already no-op on both, but explicit here
    # rather than relied upon).
    if document.file:
        document.file.delete(save=False)

    return JsonResponse({}, status=204)
