"""Independently annotated exact evidence quality tests, not semantic scoring."""

from dataclasses import replace

import pytest

from devseat_context import compress_document
from devseat_context.quality import (
    GoldEvidence, audit_quality, locate_gold,
)


def test_relevant_span_is_retained():
    src = "Rain in Seoul. Login timeout after 30ms. Another reminder."
    res = compress_document(src, 26, query="Login timeout", token_counter=len)
    gold = locate_gold(src, "Login timeout", "incident")
    q = audit_quality(src, res, [gold])
    assert q.source_aligned
    assert q.retained_count == 1
    assert q.exact_span_recall == 1.0


def test_missed_gold_is_reported_even_if_code_repeated():
    src = "E401 happened. Unrelated text. E401 happened."
    res = compress_document(src, 15, query="Unrelated", token_counter=len)
    a = locate_gold(src, "E401", "old")
    b = locate_gold(src, "E401", "new", occurrence=1)
    q = audit_quality(src, res, [a, b])
    assert 0 <= q.retained_count < 2
    assert q.evidence_count == 2
    assert len(q.missed_labels) == 2 - q.retained_count


def test_no_gold_has_defined_recall():
    src = "Note. Another."
    res = compress_document(src, 100, token_counter=len)
    q = audit_quality(src, res, [])
    assert q.exact_span_recall == 1.0 and not q.missed_labels


def test_pass_through_preserves_exact_spans():
    src = " First. Second.\n"
    res = compress_document(src, 200, token_counter=len)
    q = audit_quality(src, res, [locate_gold(src, "Second.", "second")])
    assert q.source_aligned
    assert q.exact_span_recall == 1.0


def test_tampered_document_is_not_counted_as_faithful():
    src = "First. Second."
    res = compress_document(src, 10, query="Second", token_counter=len)
    tampered = replace(res, text="fabricated output")
    q = audit_quality(src, tampered, [locate_gold(src, "Second", "grounded")])
    assert not q.source_aligned
    assert q.exact_span_recall == 0.0


def test_gold_labels_must_be_unique_and_offsets_valid():
    src = "Error E401."
    res = compress_document(src, 50, token_counter=len)
    a = locate_gold(src, "E401", "error")
    with pytest.raises(ValueError, match="distinct"):
        audit_quality(src, res, [a, a])
    for candidate in (
        GoldEvidence("bad", -1, 4),
        GoldEvidence("bad", 5, 2),
        GoldEvidence("bad", 0, 999),
        GoldEvidence("bad", True, 1),
    ):
        with pytest.raises(ValueError, match="offset"):
            audit_quality(src, res, [candidate])


def test_locate_gold_checks_which_occurrence():
    src = "id=X id=X id=Y"
    first = locate_gold(src, "id=X", "first")
    second = locate_gold(src, "id=X", "second", occurrence=1)
    assert first.start < second.start
    with pytest.raises(ValueError):
        locate_gold(src, "id=X", "third", occurrence=2)
    with pytest.raises(ValueError):
        locate_gold(src, "id=X", "negative", occurrence=-1)


def test_quality_reports_measured_not_alleged_token_reduction():
    src = "Short. Longer detail."
    res = compress_document(src, 10, query="Short", token_counter=len)
    q = audit_quality(src, res, [locate_gold(src, "Short", "short")])
    assert q.output_tokens <= q.input_tokens
    assert not q.estimated_tokens
    assert q.observed_reduction >= 0.0
