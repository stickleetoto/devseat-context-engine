"""Audit exact-source evidence retention independently of the compressor.

This can evaluate literal/source-offset coverage. It cannot verify semantic
equivalence, negation scope, correctness of answers, or logical entailment.
"""
from dataclasses import dataclass
from typing import Sequence

from .document import DocumentResult, verify_source_alignment


@dataclass(frozen=True, slots=True)
class GoldEvidence:
    label: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class QualityResult:
    source_aligned: bool
    evidence_count: int
    retained_count: int
    missed_labels: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    estimated_tokens: bool

    @property
    def exact_span_recall(self) -> float:
        return self.retained_count / self.evidence_count if self.evidence_count else 1.0

    @property
    def observed_reduction(self) -> float:
        return 0.0 if self.input_tokens == 0 else 1 - self.output_tokens / self.input_tokens


def audit_quality(
    source: str,
    result: DocumentResult,
    gold: Sequence[GoldEvidence],
    *,
    separator: str = "\n",
) -> QualityResult:
    """Evaluate labeled source-offset coverage, including duplicate literals.

    gold spans are used ONLY by evaluation, never passed to compression.
    """
    if not isinstance(source, str):
        raise TypeError("source must be string")
    if not isinstance(result, DocumentResult):
        raise TypeError("result must be DocumentResult")
    labels: set[str] = set()
    checked = []
    for fact in gold:
        if not isinstance(fact, GoldEvidence):
            raise TypeError("gold must contain GoldEvidence")
        if (not isinstance(fact.label, str) or not fact.label
                or fact.label in labels):
            raise ValueError("gold labels must be distinct nonempty strings")
        if (type(fact.start) is not int or type(fact.end) is not int
                or not 0 <= fact.start < fact.end <= len(source)):
            raise ValueError("invalid gold span offsets")
        labels.add(fact.label)
        checked.append(fact)
    aligned = verify_source_alignment(source, result, separator=separator)
    missing = tuple(
        f.label for f in checked
        if not aligned or not any(
            s.start <= f.start and s.end >= f.end
            for s in result.selected_spans
        )
    )
    return QualityResult(
        source_aligned=aligned,
        evidence_count=len(checked),
        retained_count=len(checked) - len(missing),
        missed_labels=missing,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        estimated_tokens=result.estimated_tokens,
    )


def locate_gold(
    source: str, literal: str, label: str, *, occurrence: int = 0,
) -> GoldEvidence:
    """Exact test fixture annotation helper; does not normalize content."""
    if not isinstance(source, str) or not isinstance(literal, str) or not literal:
        raise ValueError("source/literal must be valid text")
    if type(occurrence) is not int or occurrence < 0:
        raise ValueError("occurrence must be a nonnegative integer")
    start = 0
    for _ in range(occurrence + 1):
        found = source.find(literal, start)
        if found < 0:
            raise ValueError(f"gold literal not found at occurrence {occurrence}: {label}")
        start = found + len(literal)
    return GoldEvidence(label, found, found + len(literal))
