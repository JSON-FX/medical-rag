import pathlib
import re

RAG_DIR = pathlib.Path(__file__).resolve().parents[2] / "rag"


def test_rag_library_never_imports_django():
    offenders = []
    for path in RAG_DIR.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(import|from)\s+django", source, re.MULTILINE):
            offenders.append(path.name)
    assert offenders == [], f"rag/ must stay framework-free, but these import django: {offenders}"
