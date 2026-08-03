"""Pass 1 — the expensive one. Runs once per corpus.

For each question: ingest-time retrieval signals, plus the LLM called
UNCONDITIONALLY regardless of what the gate would say at any threshold.

That unconditional call is the point. Without it the sweep cannot answer
"what would stage 2 have done if a lower tau_abstain had let this through?",
which is exactly what the near_miss bucket exists to measure (spec 4.2).
"""
from __future__ import annotations

import json
import pathlib
import tempfile

import django
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "medical_rag.settings")
django.setup()

import yaml  # noqa: E402
from django.core.files.base import ContentFile  # noqa: E402

from chat.retrieval import retrieve  # noqa: E402
from documents.ingestion import ingest_document  # noqa: E402
from documents.models import Document  # noqa: E402
from evals.corpus import FIXTURES, build_pdf, load_drug, load_manifest  # noqa: E402
from rag.config import load_config  # noqa: E402
from rag.embeddings import OllamaEmbedder  # noqa: E402
from rag.generation import filter_sentinel, stream_chat  # noqa: E402
from rag.prompts import build_messages  # noqa: E402
from rag.vectorstore import ChromaStore  # noqa: E402

HERE = pathlib.Path(__file__).parent
QUESTIONS = HERE / "questions.yaml"
SIGNALS = HERE / "signals.json"


def ingest_corpus(store, embedder, cfg) -> int:
    """Build a PDF per drug and ingest it. Returns the chunk total."""
    total = 0
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
    return total


def run_llm(question: str, chunks, cfg) -> tuple[bool, str]:
    """Call the model and report whether the sentinel fired."""
    messages = build_messages(question, chunks, history=[], max_history=cfg.history_messages)
    collected: list[str] = []
    for kind, text in filter_sentinel(stream_chat(cfg.ollama, messages)):
        if kind == "declined":
            return True, ""
        collected.append(text)
    return False, "".join(collected)


def main() -> None:
    cfg = load_config()
    questions = yaml.safe_load(QUESTIONS.read_text())

    with tempfile.TemporaryDirectory() as chroma_dir:
        store = ChromaStore(path=chroma_dir, collection_name="eval_corpus")
        embedder = OllamaEmbedder(cfg.ollama)

        print("ingesting corpus...")
        total = ingest_corpus(store, embedder, cfg)
        print(f"  {total} chunks total\n")

        records = []
        for i, q in enumerate(questions, start=1):
            result = retrieve(q["question"], embedder, store, cfg)
            signals = result.decision.signals

            # Unconditional: we need stage 2's behaviour even where the gate
            # declined, so the sweep can model a lower threshold letting it by.
            if result.chunks:
                sentinel_fired, answer = run_llm(q["question"], result.chunks, cfg)
            else:
                sentinel_fired, answer = False, ""

            records.append({
                "id": q["id"],
                "bucket": q["bucket"],
                "expected": q["expected"],
                "drug": q.get("drug"),
                "axis": q.get("axis"),
                "top_similarity": signals["top_similarity"],
                "mean_similarity": signals["mean_similarity"],
                "lexical_support": signals["lexical_support"],
                "corpus_empty": signals["corpus_empty"],
                "gate_reason_at_defaults": result.decision.reason,
                "sentinel_fired": sentinel_fired,
                "answer": answer[:400],
                "retrieved": [c.chunk_id for c in result.chunks],
            })
            print(f"  [{i:2}/{len(questions)}] {q['id']} {q['bucket']:19} "
                  f"top={signals['top_similarity']:.4f} lex={signals['lexical_support']!s:5} "
                  f"sentinel={sentinel_fired}")

    SIGNALS.write_text(json.dumps(records, indent=1))
    print(f"\nwrote {SIGNALS} ({len(records)} records)")


if __name__ == "__main__":
    main()
