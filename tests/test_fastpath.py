"""Exact-character fast path must match generic len callback."""
import random

import pytest

from devseat_context import BudgetExceeded, ContextEngine, Segment, compress_document, verify_source_alignment


def outcome(function, *args, **kwargs):
    try:
        return ("ok", function(*args, **kwargs))
    except BudgetExceeded:
        return ("budget", BudgetExceeded)


@pytest.mark.parametrize("seed", range(40))
def test_segment_fast_equals_generic(seed):
    rand = random.Random(seed)
    options = ("A", "한국어", "E401", "", "a b", "x=12", "traceback")
    sep = rand.choice(("\n", " | ", "", "\n\n"))
    segments = [
        Segment(str(i), rand.choice(options), priority=rand.randrange(-2, 4),
                protected=rand.randrange(8) == 0)
        for i in range(rand.randrange(1, 22))
    ]
    query = rand.choice(("", "E401", "한국어", "x"))
    for budget in (0, 1, 8, 35, 150):
        fast = outcome(ContextEngine(len, sep).compress, segments, budget, query=query)
        slow = outcome(ContextEngine(lambda text: len(text), sep).compress,
                       segments, budget, query=query)
        assert fast == slow


@pytest.mark.parametrize("seed", range(40))
def test_document_fast_equals_generic(seed):
    rand = random.Random(seed + 1000)
    lines = ("일치합니다.", "Error E401.", "timeout=300.", "No changes.",
             "Current status is stable.", "Never remove logs.")
    source = " ".join(rand.choice(lines) for _ in range(rand.randrange(2, 25)))
    query = rand.choice(("", "E401", "status", "timeout", "로그"))
    sep = rand.choice(("\n", "|", " "))
    protected = () if seed % 2 else ((0, len(source.split(" ")[0])),)
    for budget in (0, 2, 17, 45, 100, 500):
        fast = outcome(compress_document, source, budget, query=query,
                       required_spans=protected, token_counter=len, separator=sep)
        slow = outcome(compress_document, source, budget, query=query,
                       required_spans=protected, token_counter=lambda t: len(t), separator=sep)
        assert fast == slow
        if fast[0] == "ok":
            assert verify_source_alignment(source, fast[1], separator=sep)


def test_protected_overflow_still_fails_closed():
    with pytest.raises(BudgetExceeded):
        ContextEngine(len).compress([Segment("p", "DO NOT DROP", protected=True)], 3)
    with pytest.raises(BudgetExceeded):
        compress_document("Never delete logs. Weather fine.", 3,
                          required_literals=("delete logs",), token_counter=len)


def test_nonadditive_counter_falls_back_safely():
    def count(t):
        return len(t) + (50 if "A" in t and "B" in t else 0)
    segs = [Segment("a", "A"), Segment("b", "B")]
    result = ContextEngine(count, separator="|").compress(segs, 3)
    assert count(result.text) <= 3 and len(result.selected_ids) < 2
    result2 = compress_document("A. B. C.", 4, token_counter=count)
    assert count(result2.text) <= 4


def test_empty_segments_and_separators_are_exact():
    seq = [Segment("a", ""), Segment("b", ""), Segment("c", "")]
    for budget in range(6):
        assert ContextEngine(len, "@@").compress(seq, budget) == (
            ContextEngine(lambda s: len(s), "@@").compress(seq, budget))


def test_fifty_thousand_segments_budget_safe_smoke():
    seq = [Segment(f"r{i}", f"record-{i:05d}", priority=i % 3)
           for i in range(50_000)]
    r = ContextEngine(len, "\n").compress(seq, 10_000, query="record")
    assert r.output_tokens <= 10_000
    assert len(r.selected_ids) + len(r.dropped_ids) == 50_000
