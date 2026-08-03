import pytest

from rag.config import OllamaConfig
from rag.generation import filter_sentinel, stream_chat
from rag.prompts import SENTINEL


def events(deltas, **kw):
    return list(filter_sentinel(iter(deltas), **kw))


def test_normal_stream_emits_all_text():
    result = events(["Metformin ", "is ", "a biguanide."])
    assert [kind for kind, _ in result] == ["token"] * len(result)
    assert "".join(text for _, text in result) == "Metformin is a biguanide."


def test_clean_sentinel_yields_a_single_declined_event():
    assert events([SENTINEL]) == [("declined", None)]


def test_sentinel_split_across_two_deltas_is_still_detected():
    """This is exactly how the check breaks in practice."""
    assert events(["INSUFF", "ICIENT_CONTEXT"]) == [("declined", None)]


def test_sentinel_split_across_many_tiny_deltas_is_detected():
    assert events(list(SENTINEL)) == [("declined", None)]


def test_sentinel_with_leading_whitespace_is_detected():
    assert events(["\n\n", SENTINEL]) == [("declined", None)]


def test_sentinel_like_prefix_that_is_not_the_sentinel_streams_normally():
    deltas = ["INSUFFICIENT data ", "was available in the monograph."]
    result = events(deltas)
    assert ("declined", None) not in result
    assert "".join(t for _, t in result) == "".join(deltas)


def test_text_mentioning_the_sentinel_later_is_not_a_decline():
    deltas = ["The adult dose is 500mg. Note: not " + SENTINEL]
    result = events(deltas)
    assert ("declined", None) not in result


def test_sentinel_after_a_short_preamble_is_still_a_decline():
    """The prompt forbids a preamble, but an 8B model may emit one anyway, and a
    missed refusal leaks the raw sentinel into the user's answer."""
    assert events([f"Sure! {SENTINEL}"]) == [("declined", None)]
    assert events(["Of course. ", SENTINEL]) == [("declined", None)]


def test_sentinel_far_into_a_real_answer_is_not_a_decline():
    """A genuine answer mentioning the token late must still stream."""
    long_answer = "The adult starting dose is 500mg twice daily with meals, per the monograph. "
    result = events([long_answer + SENTINEL])
    assert ("declined", None) not in result


def test_short_stream_ending_before_the_buffer_fills_is_flushed():
    assert events(["ok"]) == [("token", "ok")]


def test_empty_stream_yields_nothing():
    assert events([]) == []


def test_buffered_prefix_is_emitted_exactly_once():
    deltas = ["a" * 10, "b" * 40, "c" * 10]
    joined = "".join(text for _, text in events(deltas))
    assert joined == "".join(deltas)


def test_stream_chat_extracts_message_content_deltas():
    lines = [
        {"message": {"role": "assistant", "content": "Met"}, "done": False},
        {"message": {"role": "assistant", "content": "formin"}, "done": False},
        {"done": True},
    ]
    out = list(stream_chat(OllamaConfig(), [{"role": "user", "content": "q"}],
                           transport=lambda url, payload: iter(lines)))
    assert out == ["Met", "formin"]


def test_stream_chat_ignores_lines_without_content():
    lines = [{"done": False}, {"message": {"content": "x"}, "done": False}, {"done": True}]
    out = list(stream_chat(OllamaConfig(), [], transport=lambda url, payload: iter(lines)))
    assert out == ["x"]


def test_preamble_split_across_deltas_near_the_buffer_boundary_still_declines():
    """Regression: the decision once fired at 40 chars while a tolerated
    preamble plus the sentinel needs 44, locking in a false negative and
    leaking the raw sentinel to the user."""
    for preamble_length in (20, 21, 22, 23, 24):
        preamble = "A" * preamble_length
        stream = list(preamble + SENTINEL)          # one character per delta
        assert events(stream) == [("declined", None)], (
            f"preamble of {preamble_length} chars leaked the sentinel"
        )


def test_preamble_split_into_small_chunks_still_declines():
    text = "A" * 23 + SENTINEL
    chunks = [text[i : i + 3] for i in range(0, len(text), 3)]
    assert events(chunks) == [("declined", None)]


def test_a_preamble_longer_than_tolerance_is_not_a_decline():
    """Beyond the tolerance it is treated as a real answer, by design."""
    stream = list("B" * 60 + SENTINEL)
    assert ("declined", None) not in events(stream)


def test_malformed_json_line_becomes_ollama_unavailable(monkeypatch):
    """A truncated NDJSON line must not leak JSONDecodeError to callers that
    only catch OllamaError."""
    import httpx

    from rag.generation import _http_stream
    from rag.ollama import OllamaUnavailable

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield '{"message": {"content": "ok"}, "done": false}'
            yield '{"message": {"content": tru'          # truncated

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: FakeStream())
    with pytest.raises(OllamaUnavailable, match="invalid JSON"):
        list(_http_stream("http://x/api/chat", {}))
