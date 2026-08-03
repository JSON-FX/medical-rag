"""Ollama chat streaming and stage-2 sentinel filtering.

The sentinel cannot be streamed to the browser and then retracted, so output
is buffered until there is enough text to decide (spec 6.6). The buffer costs
a few tokens of latency and is imperceptible.
"""
from __future__ import annotations

import json
from typing import Callable, Iterable, Iterator

import httpx

from .config import OllamaConfig
from .ollama import OllamaUnavailable
from .prompts import SENTINEL

BUFFER_CHARS = 40
PREAMBLE_TOLERANCE = 24


def _http_stream(url: str, payload: dict) -> Iterator[dict]:
    try:
        with httpx.stream("POST", url, json=payload, timeout=300.0) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.strip():
                    yield json.loads(line)
    except httpx.HTTPError as exc:
        raise OllamaUnavailable(f"chat request failed: {exc}") from exc


def stream_chat(
    cfg: OllamaConfig,
    messages: list[dict],
    transport: Callable[[str, dict], Iterator[dict]] | None = None,
) -> Iterator[str]:
    send = transport or _http_stream
    payload = {"model": cfg.chat_model, "messages": messages, "stream": True}
    for chunk in send(f"{cfg.host}/api/chat", payload):
        content = (chunk.get("message") or {}).get("content")
        if content:
            yield content


def _is_sentinel(buffer: str, sentinel: str = SENTINEL) -> bool:
    """True when the buffered head is a refusal rather than an answer.

    Tolerates a short conversational preamble. The prompt forbids one, but an
    8B instruct model may still emit "Sure! INSUFFICIENT_CONTEXT", and a missed
    refusal is doubly bad: the model answers when it should have declined, and
    the raw sentinel token leaks into the user's visible stream. A real answer
    that happens to contain the sentinel this early is not a plausible output.
    """
    stripped = buffer.lstrip()
    if stripped.startswith(sentinel):
        return True
    position = stripped.find(sentinel)
    return 0 <= position <= PREAMBLE_TOLERANCE


def filter_sentinel(
    deltas: Iterable[str],
    sentinel: str = SENTINEL,
    buffer_chars: int = BUFFER_CHARS,
) -> Iterator[tuple[str, str | None]]:
    """Yield ('token', text) events, or exactly one ('declined', None).

    The sentinel commonly arrives split across deltas, so the decision waits
    until the buffer holds enough characters to be conclusive.
    """
    threshold = max(len(sentinel), buffer_chars)
    buffer = ""
    decided = False

    for delta in deltas:
        if decided:
            yield ("token", delta)
            continue
        buffer += delta
        if len(buffer) >= threshold:
            decided = True
            if _is_sentinel(buffer, sentinel):
                yield ("declined", None)
                return
            yield ("token", buffer)
            buffer = ""

    if not decided and buffer:
        if _is_sentinel(buffer, sentinel):
            yield ("declined", None)
        else:
            yield ("token", buffer)
