"""Hybrid retrieval plus the stage-1 gate.

The gate reads `top_similarity` from the VECTOR leg directly, not from fused
output: RRF deliberately discards score magnitude, so fused scores carry no
similarity information (spec 17).
"""
from __future__ import annotations

from dataclasses import dataclass

from documents.models import Chunk
from rag.config import RagConfig
from rag.fusion import reciprocal_rank_fusion
from rag.gate import GateDecision, GateSignals, evaluate_gate
from rag.prompts import ContextChunk

from .lexical_search import search as lexical_search


@dataclass(frozen=True)
class RetrievalResult:
    decision: GateDecision
    chunks: list[ContextChunk]


def _hydrate(vector_ids: list[str]) -> list[ContextChunk]:
    """Map vector ids back to chunk rows in one query, preserving rank order."""
    pairs = []
    for vector_id in vector_ids:
        document_id, _, chunk_index = vector_id.partition("_")
        if chunk_index.isdigit():
            pairs.append((int(document_id), int(chunk_index)))
    if not pairs:
        return []

    rows = Chunk.objects.filter(
        document_id__in={d for d, _ in pairs}, chunk_index__in={i for _, i in pairs}
    ).select_related("document")
    by_key = {(r.document_id, r.chunk_index): r for r in rows}

    hydrated = []
    for document_id, chunk_index in pairs:
        row = by_key.get((document_id, chunk_index))
        if row is None:
            continue          # orphaned vector: drop it silently
        hydrated.append(
            ContextChunk(
                chunk_id=row.vector_id,
                title=row.document.title,
                page_number=row.page_number,
                text=row.text,
            )
        )
    return hydrated


def retrieve(question: str, embedder, store, cfg: RagConfig) -> RetrievalResult:
    if store.count() == 0:
        signals = GateSignals(0.0, 0.0, lexical_support=False, corpus_empty=True)
        return RetrievalResult(evaluate_gate(signals, cfg.gate), [])

    vector_hits = store.query(embedder.embed_query(question), cfg.retrieval.per_leg)
    lexical_ids = lexical_search(question, cfg.retrieval.per_leg)

    similarities = [1.0 - hit.distance for hit in vector_hits]
    top_similarity = max(similarities) if similarities else 0.0

    fused = reciprocal_rank_fusion(
        [[h.chunk_id for h in vector_hits], lexical_ids], k=cfg.retrieval.rrf_k
    )
    top_ids = [hit.chunk_id for hit in fused[: cfg.retrieval.top_k]]

    # Recorded for Phase 3's eval sweep; the gate itself never reads this
    # (pinned by test_mean_similarity_does_not_affect_the_decision in
    # tests/unit/test_gate.py — do not wire it in here without updating that
    # pin deliberately). Averaged over the chunks actually DELIVERED to the
    # model (top_ids, after fusion) rather than the full per_leg candidate
    # pool: the pool measures ANN-neighbourhood breadth, which is the wrong
    # population for judging delivered-context quality — a chunk ranked 9th
    # of 10 by cosine similarity but never fused into the top_k should not
    # pull this average down, since the model never saw it. A delivered
    # chunk found only by the lexical leg has no vector-space similarity at
    # all and is excluded rather than treated as 0, which would understate a
    # lexically-rescued answer instead of just reflecting that fewer of the
    # top_k have a similarity opinion.
    similarity_by_id = dict(zip((h.chunk_id for h in vector_hits), similarities))
    delivered_similarities = [similarity_by_id[i] for i in top_ids if i in similarity_by_id]
    mean_similarity = (
        sum(delivered_similarities) / len(delivered_similarities)
        if delivered_similarities
        else 0.0
    )

    signals = GateSignals(
        top_similarity=top_similarity,
        mean_similarity=mean_similarity,
        # "Does the question's terminology appear in the text we are about to
        # show the model?" — measured across all delivered chunks, not just the
        # top one. Using top_ids[0] alone made this hinge on RRF tie-breaking:
        # when the legs disagree completely, both rank-1 items score 1/(k+1) and
        # the winner is decided by lexicographic chunk_id sort, so the same
        # disagreement produced either answer depending on document numbering.
        lexical_support=bool(set(top_ids) & set(lexical_ids)),
        corpus_empty=False,
    )
    decision = evaluate_gate(signals, cfg.gate)
    return RetrievalResult(decision, _hydrate(top_ids) if decision.proceed else [])
