"""Heuristic fact detection and conservative, offset-audited compression.

Recognized fact types are deliberately narrow. This is NOT natural-language
inference, contradiction resolution, or a guarantee of factual completeness.
"""

from dataclasses import dataclass
import re
from typing import Callable

from .document import (
    DocumentResult, EvidenceSpan, compress_document, split_document,
)
from .engine import BudgetExceeded

# Avoid URLs such as https://... being parsed as key:value assignments.
_ASSIGNMENT = re.compile(
    r'(?<![\w])(?P<key>[A-Za-z_][A-Za-z0-9_.-]{0,63})\s*'
    r'(?:=|:(?!//))\s*'
    r'(?P<value>"[^"\n]{1,128}"|'
    r"'[^'\n]{1,128}'|[A-Za-z0-9_/-](?:[A-Za-z0-9_./-]{0,126}[A-Za-z0-9_/-])?)"
    r'(?=$|[.\s,;!?)\]}])',
)
_DATE = re.compile(r'(?<![\d])\d{4}-\d{2}-\d{2}(?![\d])')
_IDENTIFIER = re.compile(r'(?<![\w])(?:[A-Z]{1,8}[-_]?\d{2,}(?:[-_][A-Za-z0-9]+)*)(?![\w])')
_QUANTITY = re.compile(r'(?<![\w])\d+(?:\.\d+)?(?:ms|GB|MB|KiB|MiB|%|회|번|원)(?![\w])')
_NEGATION = re.compile(
    r'\b(?:must\s+not|should\s+not|do\s+not|never|forbidden|prohibited)\b'
    r'|(?:해서는\s*안|하지\s*말|금지|않는다|않음|불가)',
    flags=re.IGNORECASE,
)
_KINDS = frozenset({'assignment', 'date', 'identifier', 'quantity', 'negation'})
_DEFAULT_REQUIRED_KINDS = frozenset({'assignment', 'negation'})


@dataclass(frozen=True, slots=True)
class Fact:
    kind: str
    value: str
    start: int
    end: int
    key: str | None = None

    @property
    def span(self) -> tuple[int, int]:
        return self.start, self.end


@dataclass(frozen=True, slots=True)
class AssignmentConflict:
    """A candidate conflicting key with distinct literal values; no ordering inferred."""

    key: str
    facts: tuple[Fact, ...]


@dataclass(frozen=True, slots=True)
class FactAudit:
    retained: tuple[Fact, ...]
    dropped: tuple[Fact, ...]
    conflicts: tuple[AssignmentConflict, ...]
    total: int

    @property
    def fact_recall(self) -> float:
        """Heuristic recognized-fact span recall, NOT answer accuracy."""
        return len(self.retained) / self.total if self.total else 1.0


@dataclass(frozen=True, slots=True)
class FactAwareResult:
    document: DocumentResult
    audit: FactAudit
    required_kinds: frozenset[str]

    @property
    def text(self) -> str:
        return self.document.text


def extract_facts(text: str) -> tuple[Fact, ...]:
    """Find exact fact candidates with character offsets; never rewrite values."""
    if not isinstance(text, str):
        raise TypeError('text must be str')
    out: list[Fact] = []
    for match in _ASSIGNMENT.finditer(text):
        # Conservative: multiline untrusted documents can contain incidental
        # key-value-like strings. This scanner identifies candidates only.
        out.append(Fact('assignment', match.group(0), *match.span(),
                        key=match.group('key').casefold()))
    for kind, pattern in (('date', _DATE), ('identifier', _IDENTIFIER),
                          ('quantity', _QUANTITY)):
        for match in pattern.finditer(text):
            out.append(Fact(kind, match.group(0), *match.span()))
    for sentence in split_document(text):
        sentence_text = text[sentence.start:sentence.end]
        if _NEGATION.search(sentence_text):
            out.append(Fact('negation', sentence_text, sentence.start, sentence.end))
    # Unambiguous stable ordering; identical offsets of different kinds are
    # intentionally separate typed observations, not silently deduplicated.
    return tuple(sorted(out, key=lambda f: (f.start, f.end, f.kind)))


def candidate_conflicts(facts: tuple[Fact, ...]) -> tuple[AssignmentConflict, ...]:
    """Signal literal value disagreement for same normalized key, no auto-resolution."""
    if not isinstance(facts, tuple) or any(not isinstance(f, Fact) for f in facts):
        raise TypeError('facts must be tuple[Fact, ...]')
    buckets: dict[str, list[Fact]] = {}
    for fact in facts:
        if fact.kind == 'assignment' and fact.key:
            buckets.setdefault(fact.key, []).append(fact)
    conflicts = []
    for key, members in buckets.items():
        # Values are compared as exact literals: different quoting/casing may
        # cause a false positive, which must be reviewed by the integrator.
        values = set()
        for fact in members:
            m = _ASSIGNMENT.fullmatch(fact.value)
            if m:
                values.add(m.group('value'))
        if len(values) > 1:
            conflicts.append(AssignmentConflict(key, tuple(members)))
    return tuple(conflicts)


def audit_facts(
    facts: tuple[Fact, ...],
    selected_spans: tuple[EvidenceSpan, ...],
) -> FactAudit:
    """Check exact source-offset coverage, not string occurrence in output.

    This prevents duplicate literals elsewhere from falsely counting as
    retained evidence.
    """
    if any(not isinstance(f, Fact) for f in facts):
        raise TypeError('facts must be Fact instances')
    if any(not isinstance(s, EvidenceSpan) for s in selected_spans):
        raise TypeError('selected_spans must be EvidenceSpan instances')
    def covered(fact: Fact) -> bool:
        return any(s.start <= fact.start and s.end >= fact.end for s in selected_spans)
    retained = tuple(f for f in facts if covered(f))
    dropped = tuple(f for f in facts if not covered(f))
    return FactAudit(retained, dropped, candidate_conflicts(facts), len(facts))


def compress_fact_aware(
    text: str,
    budget: int,
    *,
    query: str = '',
    required_kinds: frozenset[str] = _DEFAULT_REQUIRED_KINDS,
    required_literals: tuple[str, ...] = (),
    token_counter: Callable[[str], int] | None = None,
    separator: str = '\n',
) -> FactAwareResult:
    """Protect recognized assignment/negation spans by default; fail closed.

    Other kinds are audited but not automatically protected. Pass a subset
    of known kinds in required_kinds, including empty frozenset for audit-only.
    Do not use heuristic fact extraction to infer trusted instructions.
    """
    if not isinstance(required_kinds, frozenset) or not required_kinds <= _KINDS:
        raise ValueError('required_kinds must be a frozenset of recognized kinds')
    facts = extract_facts(text)
    required_spans = tuple(f.span for f in facts if f.kind in required_kinds)
    doc = compress_document(
        text, budget, query=query, required_literals=required_literals,
        required_spans=required_spans, token_counter=token_counter,
        separator=separator,
    )
    audit = audit_facts(facts, doc.selected_spans)
    if any(f.kind in required_kinds for f in audit.dropped):
        raise BudgetExceeded('mandatory recognized fact span was not retained')
    return FactAwareResult(doc, audit, required_kinds)
