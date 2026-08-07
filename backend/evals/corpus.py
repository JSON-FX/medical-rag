"""Build the eval corpus from committed fixture text.

Pagination is deliberate: real page numbers make citation assertions meaningful
and exercise the page-aware chunker the way an uploaded document would.
"""
from __future__ import annotations

import json
import pathlib
import re

from tests.fixtures.make_fixture_pdf import make_pdf

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CHARS_PER_PAGE = 1200

SECTION_TITLES = {
    "indications_and_usage": "INDICATIONS AND USAGE",
    "dosage_and_administration": "DOSAGE AND ADMINISTRATION",
    "contraindications": "CONTRAINDICATIONS",
    "adverse_reactions": "ADVERSE REACTIONS",
    "drug_interactions": "DRUG INTERACTIONS",
}


def _paginate(text: str, chars_per_page: int = CHARS_PER_PAGE) -> list[str]:
    words, pages, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > chars_per_page:
            pages.append(current.strip())
            current = ""
        current += word + " "
    if current.strip():
        pages.append(current.strip())
    return pages


def assemble_text(included: dict[str, str]) -> str:
    """One normalised string over the included sections, in a stable order.

    Whitespace is collapsed before the axis scan runs over this text. A
    multi-word keyword like "hepatic impairment" broken across a newline in the
    source label would otherwise go undetected, marking an axis absent when it
    is present — the exact near-miss corruption spec 2.2.1 exists to prevent.
    """
    parts = []
    for key, title in SECTION_TITLES.items():
        body = included.get(key)
        if body:
            parts.append(f"{title}\n{re.sub(r'\s+', ' ', body).strip()}")
    return "\n\n".join(parts)


def build_pdf(path: pathlib.Path, title: str, included: dict[str, str]) -> pathlib.Path:
    body = f"{title}\n\n{assemble_text(included)}"
    return make_pdf(path, _paginate(body))


def load_manifest() -> dict:
    return json.loads((FIXTURES / "manifest.json").read_text())


def load_drug(slug: str) -> dict:
    return json.loads((FIXTURES / f"{slug}.json").read_text())
