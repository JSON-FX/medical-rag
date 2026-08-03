"""NDJSON frame construction.

One JSON object per line. `json.dumps` escapes embedded newlines, so answer
text containing line breaks cannot split a frame.
"""
from __future__ import annotations

import json


def frame(kind: str, **fields) -> str:
    return json.dumps({"type": kind, **fields}, ensure_ascii=False) + "\n"
