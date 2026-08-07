"""The collect pass must measure the eval corpus and nothing else."""
import pytest

from documents.models import Document

pytestmark = pytest.mark.django_db


def test_collect_refuses_to_run_against_a_populated_database():
    """Chroma is sandboxed to a temp dir for the collect pass, but the lexical
    leg is not: chat/lexical_search.py reads the chunk_fts table in the shared
    db.sqlite3. Chunks from anything a developer uploaded through /documents
    would match eval questions, enter RRF fusion, set `lexical_support`, and be
    hydrated into the LLM's context — corrupting the measurement the whole
    phase exists to produce, without a word in the output to say so.

    Asserting on main() rather than on the helper is deliberate: the guard has
    to fire before the config load, the ingest and the first Ollama call.
    """
    from evals.collect import main

    Document.objects.create(title="a-doc-i-uploaded-earlier.pdf", status="ready")

    with pytest.raises(SystemExit) as raised:
        main()
    assert "refusing to run" in str(raised.value)
    assert "chunk_fts" in str(raised.value), "the message must explain WHY"


def test_the_guard_is_silent_on_an_empty_database():
    from evals.collect import require_empty_database

    require_empty_database()
