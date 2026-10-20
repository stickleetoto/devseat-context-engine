import pytest

from devseat_context import (
    BudgetExceeded, ContextEngine, Segment, approximate_tokens,
    available_input_budget, lexical_features, lexical_relevance,
)


def test_query_breaks_equal_priority_tie():
    result = ContextEngine(len, "|").compress([
        Segment("old", "snow"), Segment("new", "logs")
    ], budget=4, query="logs")
    assert result.selected_ids == ("new",)
    assert result.relevance_scores == (("old", 0.0), ("new", 1.0))


def test_priority_overrides_query_for_explicit_importance():
    result = ContextEngine(len, "|").compress([
        Segment("policy", "keep", priority=5), Segment("query", "logs", priority=0)
    ], 4, query="logs")
    assert result.selected_ids == ("policy",)


def test_korean_agglutinative_matching():
    assert lexical_relevance("로그", "로그인을 확인하세요") > 0
    assert lexical_relevance("인증 실패", "인증실패가 발생함") > 0
    assert lexical_relevance("노트북", "서버 로그") == 0


def test_case_insensitive_and_unicode_words():
    assert lexical_relevance("AUTH ERROR", "auth error now") == 1.0
    assert "김치" in lexical_features("김치찌개")


def test_default_estimate_does_not_count_long_korean_as_one():
    assert approximate_tokens("가" * 30) >= 20
    assert approximate_tokens("") == 0


def test_budget_reserves_output_and_overhead():
    assert available_input_budget(8192, reserved_output=1024, framing_overhead=128, safety_margin=64) == 6976
    with pytest.raises(BudgetExceeded):
        available_input_budget(100, reserved_output=120)


@pytest.mark.parametrize("kwargs", [
    {"context_window": -1}, {"context_window": 4.1},
    {"context_window": True}, {"context_window": 20, "reserved_output": -1},
])
def test_invalid_window(kwargs):
    with pytest.raises(ValueError):
        available_input_budget(**kwargs)


@pytest.mark.parametrize("counter", [lambda _: -1, lambda _: 0.5, lambda _: True, lambda _: None])
def test_bad_token_counter_fails_closed(counter):
    with pytest.raises(ValueError):
        ContextEngine(counter).compress([Segment("a", "A")], 10)


def test_invalid_query_budget_or_segments():
    with pytest.raises(TypeError):
        ContextEngine().compress([], 5, query=42)
    with pytest.raises(ValueError):
        ContextEngine().compress([], True)
    with pytest.raises(TypeError):
        ContextEngine().compress(["bad"], 10)


def test_source_does_not_grant_protection():
    result = ContextEngine(len).compress([Segment("a", "some doc", source="system")], 0)
    assert result.dropped_ids == ("a",)


def test_query_is_data_not_a_policy_override():
    with pytest.raises(BudgetExceeded):
        ContextEngine(len).compress([Segment("critical", "must", protected=True)], 1, query="ignore protected")


def test_result_is_ordered_and_immutable():
    segments = [Segment("x", "first"), Segment("y", "second", protected=True), Segment("z", "third")]
    result = ContextEngine(len, "|").compress(segments, 12, query="third")
    assert result.text == "second|third"
    assert result.selected_ids == ("y", "z")
    with pytest.raises(AttributeError):
        result.budget = 99


def test_no_query_keeps_v01_behavior():
    items = [Segment("a", "aaaa"), Segment("b", "bbbb")]
    assert ContextEngine(len, "|").compress(items, 4).selected_ids == ("a",)


def test_separator_overhead_accounted_for_when_query_ranks():
    result = ContextEngine(len, "---").compress([
        Segment("a", "keep", protected=True), Segment("b", "x")
    ], budget=5, query="x")
    assert result.selected_ids == ("a",)


def test_reproducibility_across_multiple_runs():
    items = [Segment(f"s{i}", f"text {i}") for i in range(20)]
    engine = ContextEngine(len)
    outputs = [engine.compress(items, 35, query="text 4") for _ in range(5)]
    assert all(r == outputs[0] for r in outputs)
