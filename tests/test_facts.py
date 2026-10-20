"""Fact-aware v0.4 regression tests; no external model dependency."""
from dataclasses import replace

import pytest

from devseat_context import (
    BudgetExceeded, EvidenceSpan, MissingEvidence,
    audit_facts, candidate_conflicts, compress_document, compress_fact_aware,
    extract_facts, verify_source_alignment,
)


def test_assignment_spans_exact_and_unmodified():
    text = "timeout_ms=300. timeout_ms=500. \n error: E401"
    facts = extract_facts(text)
    assigns = [f for f in facts if f.kind == "assignment"]
    assert {f.key for f in assigns} == {"timeout_ms", "error"}
    assert {f.value for f in assigns} >= {"timeout_ms=300", "timeout_ms=500"}
    assert all(text[f.start:f.end] == f.value for f in facts)


def test_urls_not_parsed_as_assignments():
    facts = extract_facts("Open https://example.com/x. timeout_ms=500.")
    assert not any(f.kind == "assignment" and f.key == "https" for f in facts)


def test_conflict_report_does_not_decide_winner():
    text = "timeout_ms=300\nTimeout_MS=500\nlimit=2"
    groups = candidate_conflicts(extract_facts(text))
    assert len(groups) == 1
    assert groups[0].key == "timeout_ms"
    assert len(groups[0].facts) == 2
    assert {f.value for f in groups[0].facts} == {"timeout_ms=300", "Timeout_MS=500"}


def test_no_false_conflict_when_values_equal():
    assert candidate_conflicts(extract_facts("port=443\nport=443")) == ()


def test_scanner_dates_identifiers_quantities_and_korean_negation():
    src = "2026-10-20 ORD-7781 512MB. 삭제해서는 안 됩니다."
    kinds = {f.kind for f in extract_facts(src)}
    assert {"date", "identifier", "quantity", "negation"} <= kinds


def test_english_negation_contains_the_whole_sentence():
    src = "Do not delete files. This is only an example."
    negations = [f for f in extract_facts(src) if f.kind == "negation"]
    assert len(negations) == 1
    assert negations[0].value == "Do not delete files."


def test_strict_defaults_preserve_assignments_and_negation():
    src = "Unrelated details. timeout_ms=300. Do not delete files. Unrelated history."
    result = compress_fact_aware(src, 44, query="unrelated", token_counter=len)
    assert "timeout_ms=300" in result.text
    assert "Do not delete files." in result.text
    assert all(f.kind not in result.required_kinds for f in result.audit.dropped)
    assert verify_source_alignment(src, result.document)


def test_strict_protection_fails_closed_when_cannot_fit():
    src = "timeout_ms=300. Do not delete files."
    with pytest.raises(BudgetExceeded):
        compress_fact_aware(src, 10, token_counter=len)


def test_audit_only_can_drop_nonmandatory_fact_and_reports_it():
    src = "server1=100. Weather remains sunny."
    result = compress_fact_aware(src, 22, query="weather", token_counter=len,
                                 required_kinds=frozenset())
    assert "Weather" in result.text
    assert any(f.kind == "assignment" for f in result.audit.dropped)


def test_required_fact_kinds_works_with_identifier():
    src = "Old report. Ticket ORD-7781 was created. Unrelated notes."
    result = compress_fact_aware(src, 33, query="unrelated", token_counter=len,
                                 required_kinds=frozenset({"identifier"}))
    assert "ORD-7781" in result.text
    assert all(f.kind != "identifier" for f in result.audit.dropped)


def test_required_spans_preserve_duplicate_occurrences_not_just_first():
    src = "status=BAD. Middle note. status=BAD."
    points = tuple((f.start, f.end) for f in extract_facts(src) if f.kind == "assignment")
    assert len(points) == 2
    result = compress_document(src, 28, required_spans=points, token_counter=len)
    assert result.text.count("status=BAD") == 2
    assert verify_source_alignment(src, result)


def test_repeat_required_fact_does_not_silently_drop_second_occurrence():
    src = "status=BAD. " + ("lots of filler " * 8) + "status=BAD."
    with pytest.raises(BudgetExceeded):
        compress_fact_aware(src, 12, token_counter=len)


def test_cross_sentence_required_span_keeps_exact_literal():
    src = "Old note. Error E401. Do not retry. Other."
    p = src.index("E401")
    q = src.index("retry") + len("retry")
    result = compress_document(src, 27, token_counter=len, required_spans=((p, q),))
    assert "Error E401. Do not retry." in result.text
    assert verify_source_alignment(src, result)


@pytest.mark.parametrize("offsets", [
    ((-1, 3),), ((2, 2),), ((0, 99),),
    ((True, 1),), ((2, 1),), ([1, 2],),
])
def test_invalid_offsets_are_rejected(offsets):
    with pytest.raises(ValueError):
        compress_document("ABC", 10, token_counter=len, required_spans=offsets)


def test_required_spans_wrong_outer_type():
    with pytest.raises(TypeError):
        compress_document("abc", 10, required_spans=[(0, 1)])


def test_required_span_with_whitespace_only_fails_closed():
    with pytest.raises(BudgetExceeded):
        compress_document("A.   B.", 2, token_counter=len, required_spans=((2, 4),))


def test_alignment_verifier_rejects_tampered_passthrough_spans():
    text = " First sentence. Second sentence."
    result = compress_document(text, 999, token_counter=len)
    assert verify_source_alignment(text, result)
    tampered = replace(result, selected_spans=(EvidenceSpan(0, len(text), "forged"),))
    assert not verify_source_alignment(text, tampered)


def test_alignment_audit_uses_offsets_even_when_text_repeats():
    text = "x=1. Something irrelevant. x=1."
    facts = extract_facts(text)
    assignments = [f for f in facts if f.kind == "assignment"]
    sel = (EvidenceSpan(assignments[0].start, assignments[0].end, "test"),)
    audited = audit_facts(tuple(assignments), sel)
    assert len(audited.retained) == 1
    assert len(audited.dropped) == 1
    assert audited.fact_recall == 0.5


def test_empty_document_safe():
    result = compress_fact_aware("", 0, token_counter=len)
    assert result.text == ""
    assert result.audit.fact_recall == 1.0


def test_nonadditive_token_counter_checked_on_whole_output():
    src = "a=1. Misc. b=2."
    def count(s):
        return len(s) + (30 if "a=1" in s and "b=2" in s else 0)
    with pytest.raises(BudgetExceeded):
        compress_fact_aware(src, 15, token_counter=count)


def test_invalid_kind_policy_fails():
    for kinds in ({"assignment"}, frozenset({"unrecognized"})):
        with pytest.raises(ValueError):
            compress_fact_aware("x=1", 100, required_kinds=kinds)


def test_caller_missing_exact_required_literal_is_rejected():
    with pytest.raises(MissingEvidence):
        compress_fact_aware("x=1", 100, required_literals=("not here",))


def test_fact_objects_are_immutable():
    f = extract_facts("x=1")[0]
    with pytest.raises(AttributeError):
        f.value = "x=2"


def test_spans_can_cover_facts_inside_atomic_code_fence():
    src = "Before.\n```py\nlimit=3\nlimit=5\n```\nAfter."
    result = compress_fact_aware(src, 29, token_counter=len)
    assert "limit=3" in result.text and "limit=5" in result.text
    assert any(c.key == "limit" for c in result.audit.conflicts)
