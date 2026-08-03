"""FTS5 query construction.

Raw user questions contain characters FTS5 parses as query syntax — quotes,
`*`, `-`, `:`, and bare keywords like NEAR/AND/OR/NOT. Passing a question
straight into MATCH raises `fts5: syntax error` on input as ordinary as
"What's the max dose?" (spec 6.2). Every term is reduced to alphanumerics
and quoted, which also neutralises the reserved words.
"""
from __future__ import annotations

import re

# A decimal number with an optional unit suffix is ONE token. Splitting
# "0.5mg" into ["0", "5mg"] and dropping the orphaned "0" made a pediatric
# 0.5mg question byte-identical to an adult 5mg one — a dosage confusion
# originating in the tokenizer.
TOKEN_RE = re.compile(r"\d+(?:\.\d+)+\w*|\w+", re.UNICODE)
MIN_TERM_LENGTH = 2
MAX_TERMS = 24

# Function words carry no retrieval signal but make every question match
# almost every chunk once OR-joined. That would leave the confidence gate's
# `lexical_support` signal permanently True and collapse its middle band.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have
    how i if in into is it its may might must of on or shall should that the their
    them then there these they this to was were what when where which who why will
    with would you your
    """.split()
)


def build_fts_query(question: str) -> str:
    seen: list[str] = []
    for raw in TOKEN_RE.findall(question.lower()):
        if len(raw) < MIN_TERM_LENGTH or raw in STOPWORDS or raw in seen:
            continue
        seen.append(raw)
        if len(seen) >= MAX_TERMS:
            break
    return " OR ".join(f'"{term}"' for term in seen)
