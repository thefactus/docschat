"""Unit tests for SSE formatting and the _sse helper.

Pure functions — no DB, no API calls, no Docker required.
"""

import json

import pytest

from app.retrieval.stream import _sse

# ── SSE format correctness ──────────────────────────────────────────────────


def test_sse_starts_with_event_line():
    result = _sse("planner", {"decomposed": False})
    assert result.startswith("event: planner\n")


def test_sse_ends_with_double_newline():
    result = _sse("done", {"answer": "hi"})
    assert result.endswith("\n\n")


def test_sse_data_is_valid_json():
    data = {"decomposed": False, "sub_queries": ["test q"], "reason": "unchanged"}
    result = _sse("planner", data)
    data_line = next(ln for ln in result.split("\n") if ln.startswith("data:"))
    parsed = json.loads(data_line[len("data: "):])
    assert parsed == data


def test_sse_preserves_nested_structures():
    data = {
        "weights": "0.7/0.3",
        "top": [{"filename": "a.pdf", "page": 1, "score": 0.85}],
    }
    result = _sse("fusion", data)
    data_line = next(ln for ln in result.split("\n") if ln.startswith("data:"))
    parsed = json.loads(data_line[len("data: "):])
    assert parsed["top"][0]["filename"] == "a.pdf"
    assert parsed["top"][0]["score"] == pytest.approx(0.85)


def test_sse_handles_special_chars_in_token():
    text = "Hello, world!\nNew line & <tag>"
    result = _sse("token", {"text": text})
    data_line = next(ln for ln in result.split("\n") if ln.startswith("data:"))
    parsed = json.loads(data_line[len("data: "):])
    assert parsed["text"] == text


def test_sse_event_and_data_on_separate_lines():
    result = _sse("guardrail", {"decision": "proceed"})
    lines = result.split("\n")
    assert any(ln.startswith("event:") for ln in lines)
    assert any(ln.startswith("data:") for ln in lines)


def test_sse_exactly_two_trailing_newlines():
    result = _sse("error", {"message": "oops"})
    assert result[-2:] == "\n\n"
    assert result[-3] != "\n"
