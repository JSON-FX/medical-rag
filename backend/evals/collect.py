"""Pass 1 — the expensive one. Runs once per corpus.

For each question: ingest-time retrieval signals, plus the LLM called
UNCONDITIONALLY regardless of what the gate would say at any threshold.

That unconditional call is the point. Without it the sweep cannot answer
"what would stage 2 have done if a lower tau_abstain had let this through?",
which is exactly what the near_miss bucket exists to measure (spec 4.2).

Getting that literally right needs one deviation from the obvious code:
`retrieve()` returns no chunks when its gate declines, so calling it with the
shipped defaults would silently skip the LLM for precisely the questions the
sweep most needs stage-2 data about, and record them as `sentinel_fired: false`
— making a permissive threshold look like it answers questions that stage 2
would in fact have refused. Retrieval therefore runs with an open gate
(tau_abstain = tau_strong = 0.0) so chunks always hydrate, and the shipped
gate's verdict is computed afterwards from the same signals. That is sound for
the same reason the whole sweep is: the signals are a property of the question
and the corpus, not of the thresholds (spec 4.1).
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import tempfile

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "medical_rag.settings")
django.setup()

import yaml  # noqa: E402
from django.core.files.base import ContentFile  # noqa: E402

from chat.retrieval import retrieve  # noqa: E402
from documents.ingestion import ingest_document  # noqa: E402
from documents.models import Document  # noqa: E402
from evals.corpus import build_pdf, load_drug, load_manifest  # noqa: E402
from rag.config import GateConfig, load_config  # noqa: E402
from rag.gate import GateSignals, evaluate_gate  # noqa: E402
from rag.embeddings import OllamaEmbedder  # noqa: E402
from rag.generation import filter_sentinel, stream_chat  # noqa: E402
from rag.prompts import build_messages  # noqa: E402
from rag.vectorstore import ChromaStore  # noqa: E402

HERE = pathlib.Path(__file__).parent
QUESTIONS = HERE / "questions.yaml"
SIGNALS = HERE / "signals.json"

OPEN_GATE = GateConfig(tau_abstain=0.0, tau_strong=0.0)


def ingest_corpus(store, embedder, cfg) -> tuple[int, list[int]]:
    """Build a PDF per drug and ingest it. Returns (chunk total, document ids)."""
    total = 0
    document_ids = []
    for slug in load_manifest()["drugs"]:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = build_pdf(pathlib.Path(tmp) / f"{slug}.pdf", slug.title(),
                            load_drug(slug)["included"])
            document = Document.objects.create(title=f"{slug}.pdf")
            document.file.save(f"{slug}.pdf", ContentFile(pdf.read_bytes()), save=True)
        document = ingest_document(document, embedder, store, cfg)
        if document.status != "ready":
            raise SystemExit(f"{slug} failed to ingest: {document.error_message}")
        print(f"  {slug:12} {document.chunk_count:3} chunks")
        total += document.chunk_count
        document_ids.append(document.id)
    return total, document_ids


def run_llm(question: str, chunks, cfg) -> tuple[bool, str]:
    """Call the model and report whether the sentinel fired."""
    messages = build_messages(question, chunks, history=[], max_history=cfg.history_messages)
    collected: list[str] = []
    for kind, text in filter_sentinel(stream_chat(cfg.ollama, messages)):
        if kind == "declined":
            return True, ""
        collected.append(text)
    return False, "".join(collected)


def _signals_from(payload: dict, question_id: str) -> GateSignals:
    """Rebuild GateSignals from the recorded dict.

    `as_dict` writes None for a non-finite similarity, which json can round-trip
    but the sweep's arithmetic cannot. Real cosine distances never produce one;
    if one appears the corpus is degenerate and the record would poison every
    operating point, so fail loudly rather than cache a hole.
    """
    for key in ("top_similarity", "mean_similarity"):
        if payload[key] is None:
            raise SystemExit(f"{question_id}: {key} is non-finite — refusing to cache it")
    return GateSignals(
        top_similarity=payload["top_similarity"],
        mean_similarity=payload["mean_similarity"],
        lexical_support=payload["lexical_support"],
        corpus_empty=payload["corpus_empty"],
    )


def main() -> None:
    cfg = load_config()
    # Retrieval must hydrate chunks for every question, including ones the
    # shipped gate would refuse — see the module docstring.
    open_cfg = dataclasses.replace(cfg, gate=OPEN_GATE)
    questions = yaml.safe_load(QUESTIONS.read_text())

    with tempfile.TemporaryDirectory() as chroma_dir:
        store = ChromaStore(path=chroma_dir, collection_name="eval_corpus")
        embedder = OllamaEmbedder(cfg.ollama)

        print("ingesting corpus...")
        total, document_ids = ingest_corpus(store, embedder, cfg)
        print(f"  {total} chunks total\n")

        records = []
        try:
            for i, q in enumerate(questions, start=1):
                result = retrieve(q["question"], embedder, store, open_cfg)
                signals = _signals_from(result.decision.signals, q["id"])

                # Unconditional: we need stage 2's behaviour even where the
                # shipped gate would have declined, so the sweep can model a
                # lower threshold letting it by.
                if result.chunks:
                    sentinel_fired, answer = run_llm(q["question"], result.chunks, cfg)
                else:
                    sentinel_fired, answer = False, ""

                shipped = evaluate_gate(signals, cfg.gate)
                records.append({
                    "id": q["id"],
                    "bucket": q["bucket"],
                    "expected": q["expected"],
                    "drug": q.get("drug"),
                    "axis": q.get("axis"),
                    "top_similarity": signals.top_similarity,
                    "mean_similarity": signals.mean_similarity,
                    "lexical_support": signals.lexical_support,
                    "corpus_empty": signals.corpus_empty,
                    "gate_reason_at_defaults": shipped.reason,
                    "sentinel_fired": sentinel_fired,
                    "answer": answer[:400],
                    "retrieved": [c.chunk_id for c in result.chunks],
                })
                print(f"  [{i:2}/{len(questions)}] {q['id']} {q['bucket']:19} "
                      f"top={signals.top_similarity:.4f} lex={signals.lexical_support!s:5} "
                      f"sentinel={sentinel_fired}")
        finally:
            # The eval corpus is scratch data in the developer's own database.
            # Chroma lives in a temp dir and vanishes on its own; the SQLite
            # rows would not, and would leave the dev corpus quietly non-empty.
            # FileField.delete() is not called by Document.delete(), so the
            # generated PDFs need removing explicitly — same reasoning as the
            # delete view in documents/views.py.
            for document in Document.objects.filter(id__in=document_ids):
                if document.file:
                    document.file.delete(save=False)
                document.delete()

    SIGNALS.write_text(json.dumps(records, indent=1))
    print(f"\nwrote {SIGNALS} ({len(records)} records)")


if __name__ == "__main__":
    main()
