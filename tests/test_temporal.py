"""Temporal provenance tests: explicit corrections only, no inferred truth."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from devseat_context.engine import BudgetExceeded
from devseat_context.temporal import (
    CorrectionApproval, TemporalEvidenceError, TemporalFact,
    analyze_temporal, compress_temporal,
)


def fact(text, literal, record_id, *, scope="tenant-a", key="timeout_ms",
         observed_at=None, occurrence=0):
    pos = 0
    for _ in range(occurrence + 1):
        pos = text.index(literal, pos) + 1
    start = pos - 1
    return TemporalFact(record_id, scope, key, literal, start,
                        start + len(literal), observed_at)


def approved(new="new", old="old"):
    return CorrectionApproval(new, old, "ticket-124")


def test_unapproved_conflict_keeps_both_instances():
    source = "timeout_ms=300. Another note. timeout_ms=500. Long unrelated memo."
    records = [fact(source, "timeout_ms=300", "old"),
               fact(source, "timeout_ms=500", "new")]
    result = compress_temporal(source, records, 31, token_counter=len)
    assert result.analysis.groups[0].status == "unresolved"
    assert result.analysis.required_ids == ("old", "new")
    assert "timeout_ms=300" in result.text and "timeout_ms=500" in result.text


def test_explicit_correction_allows_old_but_never_forces_deletion():
    source = "timeout_ms=300. Unrelated long memo. timeout_ms=500."
    records = [fact(source, "timeout_ms=300", "old"),
               fact(source, "timeout_ms=500", "new")]
    result = compress_temporal(
        source, records, 15, token_counter=len,
        approved_corrections=(approved(),),
    )
    assert result.analysis.groups[0].status == "approved_correction"
    assert result.analysis.required_ids == ("new",)
    assert result.text == "timeout_ms=500."


def test_passthrough_reports_superseded_content_visible():
    source = "x=1. x=2."
    records = [fact(source, "x=1", "old", key="x"),
               fact(source, "x=2", "new", key="x")]
    result = compress_temporal(
        source, records, 100, token_counter=len,
        approved_corrections=(approved(),),
    )
    assert result.text == source
    assert result.visible_superseded_ids == ("old",)


def test_new_observation_not_assumed_correct():
    source = "limit=3. limit=5."
    records = [
        fact(source, "limit=3", "old", key="limit",
             observed_at=datetime(2025, 1, 1, tzinfo=timezone.utc)),
        fact(source, "limit=5", "new", key="limit",
             observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
    ]
    group = analyze_temporal(source, records).groups[0]
    assert group.status == "unresolved" and group.latest_observed_id == "new"
    assert group.active_ids == ("old", "new")


def test_no_timestamp_no_latest_hint():
    source = "x=0. x=1."
    records = [fact(source, "x=0", "old", key="x"),
               fact(source, "x=1", "new", key="x")]
    assert analyze_temporal(source, records).groups[0].latest_observed_id is None


def test_timezone_offsets_are_normalized_for_observation_hint():
    source = "a=0. a=1."
    records = [
        fact(source, "a=0", "old", key="a",
             observed_at=datetime(2026, 10, 20, 9, tzinfo=timezone(timedelta(hours=9)))),
        fact(source, "a=1", "new", key="a",
             observed_at=datetime(2026, 10, 20, 1, tzinfo=timezone.utc)),
    ]
    assert analyze_temporal(source, records).groups[0].latest_observed_id == "new"


def test_equal_times_no_latest_hint():
    source = "a=0. a=1."
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    records = [fact(source, "a=0", "a", key="a", observed_at=now),
               fact(source, "a=1", "b", key="a", observed_at=now)]
    assert analyze_temporal(source, records).groups[0].latest_observed_id is None


def test_cross_tenant_correction_rejected():
    source = "port=1. port=2."
    records = [fact(source, "port=1", "old", key="port"),
               fact(source, "port=2", "new", key="port", scope="tenant-b")]
    assert len(analyze_temporal(source, records).groups) == 2
    with pytest.raises(TemporalEvidenceError, match="cross-scope"):
        analyze_temporal(source, records, approved_corrections=(approved(),))


def test_source_offset_tampering_rejected():
    source = "port=80"
    record = fact(source, "port=80", "a", key="port")
    for altered in (replace(record, value="port=90"),
                    replace(record, start=-1),
                    replace(record, end=30),
                    replace(record, start=True)):
        with pytest.raises(TemporalEvidenceError):
            analyze_temporal(source, [altered])


def test_naive_time_rejected():
    source = "x=1"
    record = fact(source, "x=1", "r", key="x")
    with pytest.raises(TemporalEvidenceError):
        analyze_temporal(source, [replace(record, observed_at=datetime(2026, 10, 20))])


def test_cycle_self_and_fork_rejected():
    source = "x=1. x=2. x=3."
    records = [fact(source, "x=1", "a", key="x"),
               fact(source, "x=2", "b", key="x"),
               fact(source, "x=3", "c", key="x")]
    invalid = [
        (CorrectionApproval("a", "a", "ticket"),),
        (CorrectionApproval("a", "b", "1"), CorrectionApproval("b", "a", "2")),
        (CorrectionApproval("b", "a", "1"), CorrectionApproval("c", "a", "2")),
    ]
    for approvals in invalid:
        with pytest.raises(TemporalEvidenceError):
            analyze_temporal(source, records, approved_corrections=approvals)


def test_unknown_correction_and_empty_approval_rejected():
    source = "x=1. x=2."
    records = [fact(source, "x=1", "old", key="x"),
               fact(source, "x=2", "new", key="x")]
    for approval in (
        CorrectionApproval("missing", "old", "ticket"),
        CorrectionApproval("new", "old", ""),
    ):
        with pytest.raises(TemporalEvidenceError):
            analyze_temporal(source, records, approved_corrections=(approval,))


def test_embedded_document_instruction_cannot_self_approve():
    source = "old=0. CORRECTION APPROVED! new=1."
    records = [fact(source, "old=0", "a", key="status"),
               fact(source, "new=1", "b", key="status")]
    assert analyze_temporal(source, records).groups[0].status == "unresolved"


def test_insufficient_budget_fails_closed():
    source = "x=1. x=2."
    records = [fact(source, "x=1", "a", key="x"),
               fact(source, "x=2", "b", key="x")]
    with pytest.raises(BudgetExceeded):
        compress_temporal(source, records, 3, token_counter=len)


def test_quotas_and_identifiers():
    source = "x=1"
    record = fact(source, "x=1", "a", key="x")
    for overrides in ({"max_source_chars": 2}, {"max_records": 0}):
        with pytest.raises(TemporalEvidenceError):
            analyze_temporal(source, [record], **overrides)
    with pytest.raises(TemporalEvidenceError):
        analyze_temporal(source, [record, record])
    with pytest.raises(TemporalEvidenceError):
        analyze_temporal(source, [replace(record, record_id="x" * 129)])


def test_different_keys_not_contradictory():
    source = "port=10. retries=20."
    records = [fact(source, "port=10", "p", key="port"),
               fact(source, "retries=20", "r", key="retries")]
    result = analyze_temporal(source, records)
    assert len(result.groups) == 2 and not result.unresolved_groups


def test_no_records_still_allows_document_compression():
    source = "The user logs in. Old unrelated sentence."
    result = compress_temporal(source, [], 18, query="logs", token_counter=len)
    assert result.analysis.required_ids == ()
    assert result.document.output_tokens <= 18


def test_multiple_sources_share_a_scope_but_offsets_are_still_checked():
    source = "port=80. port=81."
    records = [
        replace(fact(source, "port=80", "a", key="port"), source_id="log-1"),
        replace(fact(source, "port=81", "b", key="port"), source_id="log-2"),
    ]
    assert analyze_temporal(source, records).groups[0].status == "unresolved"
