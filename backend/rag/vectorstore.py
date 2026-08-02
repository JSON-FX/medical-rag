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

    def all_ids(self) -> set[str]:
        """Used by reconcile_vectors (Task 9)."""
        return set(self._collection.get(include=[])["ids"])

    def count(self) -> int:
        return self._collection.count()
