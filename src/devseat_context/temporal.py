"""Time-and-scope-aware evidence policy, not semantic truth inference.

Facts are exact substrings of a caller-provided document. Correction approvals
MUST be separate trusted inputs: never infer them from text or timestamps.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence

from .document import DocumentResult, compress_document
from .engine import BudgetExceeded


class TemporalEvidenceError(ValueError):
    """Invalid source metadata or ambiguous correction graph."""


@dataclass(frozen=True, slots=True)
class TemporalFact:
    record_id: str
    scope: str
    key: str
    value: str
    start: int
    end: int
    observed_at: datetime | None = None
    valid_at: datetime | None = None
    source_id: str = "document"


@dataclass(frozen=True, slots=True)
class CorrectionApproval:
    """Caller-attested correction; approval_id is NOT cryptographic authority."""

    newer_id: str
    older_id: str
    approval_id: str


@dataclass(frozen=True, slots=True)
class TemporalGroup:
    scope: str
    key: str
    record_ids: tuple[str, ...]
    active_ids: tuple[str, ...]
    superseded_ids: tuple[str, ...]
    status: str
    latest_observed_id: str | None


@dataclass(frozen=True, slots=True)
class TemporalAnalysis:
    groups: tuple[TemporalGroup, ...]
    required_ids: tuple[str, ...]
    superseded_ids: tuple[str, ...]
    approvals: tuple[CorrectionApproval, ...]

    @property
    def unresolved_groups(self) -> tuple[TemporalGroup, ...]:
        return tuple(g for g in self.groups if g.status == "unresolved")


@dataclass(frozen=True, slots=True)
class TemporalCompressionResult:
    document: DocumentResult
    analysis: TemporalAnalysis
    retained_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    visible_superseded_ids: tuple[str, ...]

    @property
    def text(self) -> str:
        return self.document.text


def _aware(value: datetime | None, label: str) -> None:
    if value is not None and (
        not isinstance(value, datetime) or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise TemporalEvidenceError(f"{label} must be timezone-aware datetime or None")


def _validate(text: str, facts: tuple[TemporalFact, ...], max_records: int,
              max_source_chars: int) -> dict[str, TemporalFact]:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if type(max_records) is not int or max_records < 0:
        raise ValueError("max_records must be a nonnegative integer")
    if type(max_source_chars) is not int or max_source_chars < 0:
        raise ValueError("max_source_chars must be a nonnegative integer")
    if len(text) > max_source_chars or len(facts) > max_records:
        raise TemporalEvidenceError("source or record limit exceeded")
    by_id = {}
    for fact in facts:
        if not isinstance(fact, TemporalFact):
            raise TypeError("facts must contain TemporalFact")
        for name in ("record_id", "scope", "key", "source_id", "value"):
            v = getattr(fact, name)
            if not isinstance(v, str) or not v or len(v) > max_source_chars:
                raise TemporalEvidenceError(f"invalid {name}")
        if len(fact.record_id) > 128 or len(fact.scope) > 128 or len(fact.key) > 128:
            raise TemporalEvidenceError("record identifier/scope/key too long")
        if fact.record_id in by_id:
            raise TemporalEvidenceError("duplicate record_id")
        if (type(fact.start) is not int or type(fact.end) is not int or
            not 0 <= fact.start < fact.end <= len(text) or
            text[fact.start:fact.end] != fact.value):
            raise TemporalEvidenceError("fact value is not exact source span")
        _aware(fact.observed_at, "observed_at")
        _aware(fact.valid_at, "valid_at")
        by_id[fact.record_id] = fact
    return by_id


def analyze_temporal(
    text: str,
    records: Sequence[TemporalFact],
    *,
    approved_corrections: Sequence[CorrectionApproval] = (),
    max_records: int = 256,
    max_source_chars: int = 100_000,
) -> TemporalAnalysis:
    """Do not choose truth based on recency or untrusted document instructions.

    Correction approvals are supplied by trusted integration code; cycles,
    forks, unknown targets and cross-scope edges are rejected.
    """
    facts = tuple(records)
    approvals = tuple(approved_corrections)
    by_id = _validate(text, facts, max_records, max_source_chars)
    superseded: set[str] = set()
    corrected_by: dict[str, str] = {}
    seen_edges: set[tuple[str, str]] = set()
    for edge in approvals:
        if not isinstance(edge, CorrectionApproval):
            raise TypeError("approvals must contain CorrectionApproval")
        if not isinstance(edge.approval_id, str) or not edge.approval_id.strip():
            raise TemporalEvidenceError("correction requires external approval_id")
        if edge.newer_id not in by_id or edge.older_id not in by_id:
            raise TemporalEvidenceError("unknown correction record")
        if edge.newer_id == edge.older_id:
            raise TemporalEvidenceError("self correction is invalid")
        newer, older = by_id[edge.newer_id], by_id[edge.older_id]
        if newer.scope != older.scope or newer.key.casefold() != older.key.casefold():
            raise TemporalEvidenceError("cross-scope or cross-key correction")
        pair = (edge.newer_id, edge.older_id)
        if pair in seen_edges or edge.older_id in corrected_by:
            raise TemporalEvidenceError("duplicate or forked correction")
        seen_edges.add(pair)
        corrected_by[edge.older_id] = edge.newer_id
        superseded.add(edge.older_id)
    for start in corrected_by:
        visited: set[str] = set()
        node = start
        while node in corrected_by:
            if node in visited:
                raise TemporalEvidenceError("cyclic correction graph")
            visited.add(node)
            node = corrected_by[node]

    groups: dict[tuple[str, str], list[TemporalFact]] = {}
    for fact in facts:
        groups.setdefault((fact.scope, fact.key.casefold()), []).append(fact)
    analyzed: list[TemporalGroup] = []
    for (scope, key), members in groups.items():
        ids = tuple(f.record_id for f in members)
        active = tuple(f.record_id for f in members if f.record_id not in superseded)
        corrected = tuple(f.record_id for f in members if f.record_id in superseded)
        active_values = {by_id[i].value for i in active}
        if len(active_values) > 1:
            status = "unresolved"
        elif corrected and len({f.value for f in members}) > 1:
            status = "approved_correction"
        else:
            status = "consistent"
        # Observation time is informational, not a truth or correction rule.
        # Missing/tied timestamps suppress even this hint.
        latest = None
        if members and all(f.observed_at is not None for f in members):
            sorted_members = sorted(
                members, key=lambda f: f.observed_at.astimezone(timezone.utc),
                reverse=True,
            )
            if (len(sorted_members) == 1 or
                sorted_members[0].observed_at > sorted_members[1].observed_at):
                latest = sorted_members[0].record_id
        analyzed.append(TemporalGroup(scope, key, ids, active, corrected, status, latest))
    required = tuple(f.record_id for f in facts if f.record_id not in superseded)
    old = tuple(f.record_id for f in facts if f.record_id in superseded)
    return TemporalAnalysis(tuple(analyzed), required, old, approvals)


def compress_temporal(
    text: str,
    records: Sequence[TemporalFact],
    budget: int,
    *,
    query: str = "",
    approved_corrections: Sequence[CorrectionApproval] = (),
    token_counter: Callable[[str], int] | None = None,
    max_records: int = 256,
    max_source_chars: int = 100_000,
) -> TemporalCompressionResult:
    """Preserve active facts and report any old approved-corrected facts still shown.

    An approved-corrected record becomes *optional*, not secretly erased:
    returned context may include it when budget allows. Check
    visible_superseded_ids before delivering the compressed context.
    """
    seq = tuple(records)
    analysis = analyze_temporal(
        text, seq, approved_corrections=approved_corrections,
        max_records=max_records, max_source_chars=max_source_chars,
    )
    mandatory = set(analysis.required_ids)
    spans = tuple((f.start, f.end) for f in seq if f.record_id in mandatory)
    document = compress_document(
        text, budget, query=query, required_spans=spans,
        token_counter=token_counter,
    )

    def covered(fact: TemporalFact) -> bool:
        return any(s.start <= fact.start and s.end >= fact.end
                   for s in document.selected_spans)

    retained = tuple(f.record_id for f in seq if covered(f))
    dropped = tuple(f.record_id for f in seq if not covered(f))
    if any(i in mandatory for i in dropped):
        raise BudgetExceeded("active/unresolved temporal evidence lost")
    visible_old = tuple(i for i in retained if i in analysis.superseded_ids)
    return TemporalCompressionResult(document, analysis, retained, dropped, visible_old)
