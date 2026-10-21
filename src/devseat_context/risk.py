"""Conservative source-offset risk signals for extractive context selection.

Signals are *heuristic candidates*, not a semantic entailment oracle and not
proof of trust. Required protection is an EXPLICIT trusted caller decision.
"""

from collections import Counter
from dataclasses import dataclass
import re
from typing import Callable, Literal

from .document import DocumentResult, compress_document, split_document, verify_source_alignment
from .engine import BudgetExceeded
from .facts import extract_facts, candidate_conflicts
from .learning import LinearCPURanker

# Intentionally broad for review: a matching clause may be irrelevant or
# adversarial. Matching alone MUST NOT promote a document to higher authority.
_NEGATION = re.compile(
    r"\b(?:never|do\s+not|don't|must\s+not|should\s+not|"
    r"cannot|can't|not\s+(?:allowed|enabled|permitted|authorized|supported))\b"
    r"|(?:해서는\s*안|하지\s*말|금지|불가|않습니다|않는다|않음|안\s*됩니다)",
    flags=re.IGNORECASE,
)
_REVISION = re.compile(
    r"\b(?:previously|formerly|updated|revised|corrected|superseded|"
    r"deprecated|overridden|reopened|revoked)\b"
    r"|(?:기존|이전|정정|수정|변경|폐기|취소|철회|재개)",
    flags=re.IGNORECASE,
)
_KINDS = frozenset({"negation", "revision_marker", "assignment_conflict", "repeated_identifier"})


class RiskLimitExceeded(ValueError):
    """Caller-specified scan quota exceeded; nothing was silently ignored."""


@dataclass(frozen=True, slots=True)
class RiskSignal:
    kind: str
    start: int
    end: int
    key: str | None = None


@dataclass(frozen=True, slots=True)
class RiskReport:
    signals: tuple[RiskSignal, ...]
    retained: tuple[RiskSignal, ...]
    dropped: tuple[RiskSignal, ...]
    source_aligned: bool
    policy: str

    @property
    def signal_recall(self) -> float:
        return len(self.retained) / len(self.signals) if self.signals else 1.0

    @property
    def review_required(self) -> bool:
        # A risk signal can still be semantically altered by an omission.
        return bool(self.dropped) or not self.source_aligned


@dataclass(frozen=True, slots=True)
class RiskCompressionResult:
    document: DocumentResult
    report: RiskReport

    @property
    def text(self) -> str:
        return self.document.text


def detect_risks(
    source: str,
    *,
    max_source_chars: int = 100_000,
    max_signals: int = 256,
) -> tuple[RiskSignal, ...]:
    """Find exact risk candidates, including repeated occurrences.

    No external LLM, source-supplied instructions or user-owned relevance
    labels influence classification.
    """
    if not isinstance(source, str):
        raise TypeError("source must be str")
    if type(max_source_chars) is not int or max_source_chars < 0:
        raise ValueError("max_source_chars must be nonnegative integer")
    if type(max_signals) is not int or max_signals < 0:
        raise ValueError("max_signals must be nonnegative integer")
    if len(source) > max_source_chars:
        raise RiskLimitExceeded("source size exceeds scan quota")
    out: list[RiskSignal] = []

    def append(kind: str, start: int, end: int, key: str | None = None) -> None:
        if len(out) >= max_signals:
            raise RiskLimitExceeded("risk signal count exceeds scan quota")
        out.append(RiskSignal(kind, start, end, key))

    # Prefer whole-clause protection for negations and revision notices.
    for span in split_document(source):
        text = source[span.start:span.end]
        if _NEGATION.search(text):
            append("negation", span.start, span.end)
        if _REVISION.search(text):
            append("revision_marker", span.start, span.end)

    facts = extract_facts(source)
    for group in candidate_conflicts(facts):
        for fact in group.facts:
            append("assignment_conflict", fact.start, fact.end, group.key)

    identifiers = [fact for fact in facts if fact.kind == "identifier"]
    occurrences = Counter(f.value for f in identifiers)
    for fact in identifiers:
        if occurrences[fact.value] > 1:
            append("repeated_identifier", fact.start, fact.end, fact.value)

    return tuple(sorted(out, key=lambda s: (s.start, s.end, s.kind, s.key or "")))


def audit_risks(
    source: str,
    document: DocumentResult,
    signals: tuple[RiskSignal, ...],
    *,
    policy: str = "audit",
    separator: str = "\n",
) -> RiskReport:
    """Audit by source offsets, preventing duplicate literal inflation."""
    if not isinstance(source, str):
        raise TypeError("source must be str")
    if not isinstance(document, DocumentResult):
        raise TypeError("document must be DocumentResult")
    if not isinstance(signals, tuple) or any(not isinstance(s, RiskSignal) for s in signals):
        raise TypeError("signals must be tuple[RiskSignal, ...]")
    if policy not in ("audit", "require"):
        raise ValueError("unknown risk policy")
    if not isinstance(separator, str):
        raise TypeError("separator must be str")
    for item in signals:
        if (item.kind not in _KINDS or type(item.start) is not int or type(item.end) is not int
                or not 0 <= item.start < item.end <= len(source)):
            raise ValueError("invalid risk signal")
    aligned = verify_source_alignment(source, document, separator=separator)

    def retained(signal: RiskSignal) -> bool:
        return aligned and any(
            span.start <= signal.start and span.end >= signal.end
            for span in document.selected_spans
        )
    saved = tuple(s for s in signals if retained(s))
    dropped = tuple(s for s in signals if not retained(s))
    return RiskReport(signals, saved, dropped, aligned, policy)


def compress_risk_aware(
    source: str,
    budget: int,
    *,
    query: str = "",
    policy: Literal["audit", "require"] = "audit",
    protected_kinds: frozenset[str] = _KINDS,
    required_spans: tuple[tuple[int, int], ...] = (),
    token_counter: Callable[[str], int] | None = None,
    separator: str = "\n",
    max_source_chars: int = 100_000,
    max_signals: int = 256,
    ranker: LinearCPURanker | None = None,
) -> RiskCompressionResult:
    """Opt-in risk protection for explicitly trusted and pre-classified data.

    audit   : informational risk report, no extra selection privileges.
    require : all signals of protected_kinds are mandatory; raise on overflow.

    A retrieved article or log can contain adversarial 'never...' and
    'key=value' strings. Callers MUST NOT enable require for arbitrary
    untrusted input without authorization, quotas and failure handling.
    """
    if policy not in ("audit", "require"):
        raise ValueError("policy must be audit or require")
    if not isinstance(protected_kinds, frozenset) or not protected_kinds <= _KINDS:
        raise ValueError("protected_kinds must be a frozenset of supported risk types")
    signals = detect_risks(
        source, max_source_chars=max_source_chars, max_signals=max_signals
    )
    required = required_spans
    if policy == "require":
        if not isinstance(required_spans, tuple):
            raise TypeError("required_spans must be tuple")
        required = required_spans + tuple(
            (s.start, s.end) for s in signals if s.kind in protected_kinds
        )
    document = compress_document(
        source, budget, query=query, required_spans=required,
        token_counter=token_counter, separator=separator, ranker=ranker,
    )
    report = audit_risks(source, document, signals, policy=policy, separator=separator)
    if policy == "require" and any(s.kind in protected_kinds for s in report.dropped):
        raise BudgetExceeded("risk evidence required by trusted caller was dropped")
    return RiskCompressionResult(document, report)
