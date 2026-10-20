import pytest

from devseat_context import BudgetExceeded, ContextEngine, Segment


def chars(text: str) -> int:
    return len(text)


def test_preserves_protected_and_original_order():
    result = ContextEngine(chars, "|").compress([
        Segment("low", "abc", priority=0),
        Segment("must", "KEEP", protected=True),
        Segment("high", "xyz", priority=10),
    ], budget=8)
    assert result.text == "KEEP|xyz"
    assert result.selected_ids == ("must", "high")
    assert result.dropped_ids == ("low",)


def test_protected_overflow_fails_closed():
    with pytest.raises(BudgetExceeded):
        ContextEngine(chars).compress([Segment("p", "important", protected=True)], 3)


def test_no_mutation_when_within_budget():
    result = ContextEngine(chars).compress([Segment("one", "abc"), Segment("two", "def")], 10)
    assert result.text == "abc\n\ndef"
    assert result.reduction_ratio == 0


def test_priority_tie_is_stable():
    result = ContextEngine(chars, "|").compress([Segment("a", "aaa"), Segment("b", "bbb")], 3)
    assert result.selected_ids == ("a",)


def test_empty_and_zero_budget():
    result = ContextEngine(chars).compress([], 0)
    assert result.text == ""
    assert result.reduction_ratio == 0


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        ContextEngine().compress([Segment("x", "a"), Segment("x", "b")], 100)


def test_estimate_indicator():
    assert ContextEngine().compress([Segment("x", "hello")], 100).estimated_tokens
    assert not ContextEngine(chars).compress([Segment("x", "hello")], 100).estimated_tokens


def test_no_partial_segment_truncation():
    result = ContextEngine(chars).compress([Segment("a", "12345"), Segment("b", "6")], 3)
    assert result.text == "6"


def test_separator_is_counted():
    result = ContextEngine(chars, "--").compress([Segment("a", "a", protected=True), Segment("b", "b")], 3)
    assert result.text == "a"


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        ContextEngine().compress([], -1)
