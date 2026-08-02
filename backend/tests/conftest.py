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
