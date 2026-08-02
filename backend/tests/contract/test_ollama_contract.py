import pytest

from rag.config import OllamaConfig
from rag.embeddings import OllamaEmbedder

pytestmark = pytest.mark.ollama


def test_real_ollama_returns_768_dimensional_vectors():
    vectors = OllamaEmbedder(OllamaConfig()).embed_documents(["metformin dosing", "atenolol"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 768


def test_real_ollama_query_and_document_embeddings_differ():
    embedder = OllamaEmbedder(OllamaConfig())
    doc = embedder.embed_documents(["metformin dosing"])[0]
    query = embedder.embed_query("metformin dosing")
    assert doc != query, "prefixes must actually reach the model"
