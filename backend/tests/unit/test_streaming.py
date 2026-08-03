import json

from chat.streaming import frame


def test_frame_is_one_json_object_terminated_by_newline():
    line = frame("token", text="hi")
    assert line.endswith("\n")
    assert line.count("\n") == 1
    assert json.loads(line) == {"type": "token", "text": "hi"}


def test_frame_escapes_newlines_inside_values():
    """A literal newline in content must not split the frame."""
    line = frame("token", text="line one\nline two")
    assert line.count("\n") == 1
    assert json.loads(line)["text"] == "line one\nline two"


def test_frame_escapes_unicode_safely():
    assert json.loads(frame("token", text="naïve — 50µg"))["text"] == "naïve — 50µg"
