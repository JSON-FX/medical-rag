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


def ingest_document(document: Document, embedder, store, cfg: RagConfig) -> Document:
    try:
        pages = extract_pages(document.file.path)
        page_count = len(pages)

        drafts = chunk_pages(pages, cfg.chunk)
        if not drafts:
            raise ValueError(NO_TEXT_MESSAGE)

        embeddings = embedder.embed_documents([d.text for d in drafts])

        # Re-ingest of an existing document must converge, not accumulate.
        cleanup_document(document.id, store)

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
        cleanup_document(document.id, store)
        document.status = "failed"
        document.error_message = str(exc)
        document.chunk_count = 0
        document.save(update_fields=["status", "error_message", "chunk_count"])

    return document
