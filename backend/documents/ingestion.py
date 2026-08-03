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
