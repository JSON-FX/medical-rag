"""FTS5 query construction.

Raw user questions contain characters FTS5 parses as query syntax — quotes,
`*`, `-`, `:`, and bare keywords like NEAR/AND/OR/NOT. Passing a question
straight into MATCH raises `fts5: syntax error` on input as ordinary as
"What's the max dose?" (spec 6.2). Every term is reduced to alphanumerics
and quoted, which also neutralises the reserved words.
"""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MIN_TERM_LENGTH = 2
MAX_TERMS = 24


def build_fts_query(question: str) -> str:
    seen: list[str] = []
    for raw in TOKEN_RE.findall(question.lower()):
        if len(raw) < MIN_TERM_LENGTH or raw in seen:
            continue
        seen.append(raw)
        if len(seen) >= MAX_TERMS:
            break
    return " OR ".join(f'"{term}"' for term in seen)
