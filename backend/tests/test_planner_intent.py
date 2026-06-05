"""Unit tests for planner intent helpers.

Pure functions — no DB, no API calls, no Docker required.
Tests: _bound_history, _parse_planner_output, fallback behaviour.
"""

import json

import pytest

from app.retrieval.planner import (
    _bound_history,
    _parse_planner_output,
)

# ── _bound_history ─────────────────────────────────────────────────────────────

def test_bound_history_empty():
    assert _bound_history([]) == []


def test_bound_history_under_limit():
    history = [{"role": "user", "content": "hi"}]
    assert _bound_history(history) == history


def test_bound_history_exact_limit():
    history = [{"role": "user", "content": str(i)} for i in range(6)]
    assert _bound_history(history) == history


def test_bound_history_over_limit_trims_oldest():
    history = [{"role": "user", "content": str(i)} for i in range(10)]
    result = _bound_history(history, limit=6)
    assert len(result) == 6
    assert result[0]["content"] == "4"
    assert result[-1]["content"] == "9"


def test_bound_history_custom_limit():
    history = [{"role": "user", "content": str(i)} for i in range(4)]
    assert _bound_history(history, limit=2) == history[-2:]


# ── _parse_planner_output — valid inputs ──────────────────────────────────────

def test_parse_greeting():
    content = json.dumps({
        "intent": "greeting",
        "standalone_query": "hi",
        "sub_queries": [],
        "reason": "greeting",
    })
    plan = _parse_planner_output(content, "hi")
    assert plan.intent == "greeting"
    assert plan.queries == []
    assert not plan.decomposed
    assert not plan.rewritten


def test_parse_meta():
    content = json.dumps({
        "intent": "meta",
        "standalone_query": "what can you do?",
        "sub_queries": [],
        "reason": "meta",
    })
    plan = _parse_planner_output(content, "what can you do?")
    assert plan.intent == "meta"
    assert plan.queries == []


def test_parse_doc_question_unchanged():
    q = "How many sick days do employees get?"
    content = json.dumps({
        "intent": "doc_question",
        "standalone_query": q,
        "sub_queries": [q],
        "reason": "unchanged",
    })
    plan = _parse_planner_output(content, q)
    assert plan.intent == "doc_question"
    assert plan.queries == [q]
    assert not plan.rewritten
    assert not plan.decomposed


def test_parse_doc_question_rewritten():
    original = "how many can they carry over?"
    rewritten = "How many unused leave days can be carried over to the next year?"
    content = json.dumps({
        "intent": "doc_question",
        "standalone_query": rewritten,
        "sub_queries": [rewritten],
        "reason": "contextualized",
    })
    plan = _parse_planner_output(content, original)
    assert plan.rewritten is True
    assert plan.standalone_query == rewritten


def test_parse_decomposed():
    content = json.dumps({
        "intent": "doc_question",
        "standalone_query": "q",
        "sub_queries": ["q1", "q2"],
        "reason": "decomposed — two topics",
    })
    plan = _parse_planner_output(content, "q")
    assert plan.decomposed is True
    assert len(plan.queries) == 2


def test_parse_fills_empty_sub_queries_from_standalone():
    content = json.dumps({
        "intent": "doc_question",
        "standalone_query": "How many annual leave days?",
        "sub_queries": [],
        "reason": "unchanged",
    })
    plan = _parse_planner_output(content, "How many annual leave days?")
    assert plan.queries == ["How many annual leave days?"]


def test_parse_caps_sub_queries_at_four():
    content = json.dumps({
        "intent": "doc_question",
        "standalone_query": "q",
        "sub_queries": ["a", "b", "c", "d", "e"],
        "reason": "decomposed",
    })
    plan = _parse_planner_output(content, "q")
    assert len(plan.queries) == 4


# ── _parse_planner_output — failure / fallback inputs ─────────────────────────

def test_parse_raises_on_bad_json():
    with pytest.raises(ValueError, match="planner output invalid"):
        _parse_planner_output("not json", "q")


def test_parse_raises_on_wrong_intent_value():
    content = json.dumps({"intent": "unknown_value", "standalone_query": "q", "sub_queries": []})
    with pytest.raises(ValueError):
        _parse_planner_output(content, "q")


def test_parse_uses_question_when_standalone_empty():
    content = json.dumps({
        "intent": "doc_question",
        "standalone_query": "",
        "sub_queries": [],
        "reason": "unchanged",
    })
    plan = _parse_planner_output(content, "fallback question")
    assert plan.standalone_query == "fallback question"
    assert plan.queries == ["fallback question"]
