"""Auditable, source-aligned, extractive compression for individual documents.

No text is paraphrased. This module deliberately does NOT claim semantic
faithfulness: discarded statements can include important facts.
"""

from dataclasses import dataclass
import re
from typing import Callable

from .engine import BudgetExceeded, ContextEngine
from .relevance import lexical_relevance

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_PROSE_BREAK = re.compile(r"(?<=[.!?。！？])(?=\s|$)")
_OPAQUE = re.compile(
    r"^\s*(?:def |async def |class |import |from |Traceback |File |at |>>> |\$ |PS |"
    r"\{|\[|#include |//|INFO\b|WARN\b|ERROR\b|[\w./\\-]+:\d+:)"
)


class MissingEvidence(ValueError):
    """The caller requires a literal absent from the source document."""


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    start: int
    end: int
    kind: str


@dataclass(frozen=True, slots=True)
class DocumentResult:
    text: str
    selected_spans: tuple[EvidenceSpan, ...]
    dropped_spans: tuple[EvidenceSpan, ...]
    required_literals: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    budget: int
    estimated_tokens: bool

    @property
    def reduction_ratio(self) -> float:
        return 0.0 if self.input_tokens == 0 else 1 - self.output_tokens / self.input_tokens


def _span(text: str, start: int, end: int, kind: str) -> EvidenceSpan | None:
    """Trim only outside whitespace, retaining exact character offsets."""
    piece = text[start:end]
    left = len(piece) - len(piece.lstrip())
    right = len(piece.rstrip())
    if right <= left:
        return None
    return EvidenceSpan(start + left, start + right, kind)


def split_document(text: str) -> tuple[EvidenceSpan, ...]:
    """Return exact non-overlapping text spans, preserving fenced code atomically.

    Line-based heuristics leave logs and source-code-looking lines intact.
    Language boundaries are best-effort, not grammatical sentence parsing.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    lines = text.splitlines(keepends=True)
    spans: list[EvidenceSpan] = []
    position = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("Traceback (most recent call last):"):
            start = position
            position += len(line)
            i += 1
            # Stack traces are an evidence unit: isolate through the next blank
            # line so file locations and the final exception stay together.
            while i < len(lines) and lines[i].strip():
                position += len(lines[i])
                i += 1
            found = _span(text, start, position, "traceback")
            if found:
                spans.append(found)
            continue
        match = _FENCE.match(line)
        if match:
            start = position
            marker = match.group(1)
            position += len(line)
            i += 1
            closer = re.compile(r"^\s*" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*$")
            while i < len(lines):
                candidate = lines[i]
                position += len(candidate)
                i += 1
                if closer.match(candidate):
                    break
            found = _span(text, start, position, "code")
            if found:
                spans.append(found)
            continue
        start = position
        position += len(line)
        i += 1
        if _OPAQUE.match(line):
            found = _span(text, start, position, "opaque")
            if found:
                spans.append(found)
            continue
        cursor = 0
        for boundary in _PROSE_BREAK.finditer(line):
            found = _span(text, start + cursor, start + boundary.start(), "prose")
            if found:
                spans.append(found)
            cursor = boundary.start()
        found = _span(text, start + cursor, position, "prose")
        if found:
            spans.append(found)
    return tuple(spans)


def _validate_literals(required_literals: tuple[str, ...], text: str) -> None:
    for literal in required_literals:
        if not isinstance(literal, str) or not literal or literal.isspace():
            raise ValueError("required literals must be nonempty non-whitespace strings")
        if literal not in text:
            raise MissingEvidence(f"required literal missing from source: {literal[:80]!r}")


def compress_document(
    text: str,
    budget: int,
    *,
    query: str = "",
    required_literals: tuple[str, ...] = (),
    required_spans: tuple[tuple[int, int], ...] = (),
    token_counter: Callable[[str], int] | None = None,
    separator: str = "\n",
) -> DocumentResult:
    """Choose exact source spans with lexical query relevance and hard literals.

    A required literal that crosses chunk boundaries is preserved by making a
    merged source span; if the merged required content exceeds budget, fail.
    Caller-owned trust policy is needed: source text never grants itself safety.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(required_literals, tuple):
        raise TypeError("required_literals must be a tuple")
    if not isinstance(required_spans, tuple):
        raise TypeError("required_spans must be a tuple")
    for pair in required_spans:
        if (not isinstance(pair, tuple) or len(pair) != 2
                or any(type(n) is not int for n in pair)
                or not 0 <= pair[0] < pair[1] <= len(text)):
            raise ValueError("required_spans must contain valid (start,end) offsets")
    if not isinstance(separator, str):
        raise TypeError("separator must be a string")
    engine = ContextEngine(token_counter=token_counter, separator=separator)
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    _validate_literals(required_literals, text)
    chunks = list(split_document(text))
    input_tokens = engine._count(text)
    if input_tokens <= budget:
        return DocumentResult(text, tuple(chunks), (), required_literals,
                              input_tokens, input_tokens, budget, token_counter is None)

    # Merge the minimal consecutive units intersected by each required literal.
    # This retains the exact original literal even across sentence boundaries.
    groups: list[tuple[int, int]] = []
    required_ranges = list(required_spans)
    for literal in dict.fromkeys(required_literals):
        begin = text.find(literal)
        end = begin + len(literal)
        required_ranges.append((begin, end))
    for begin, end in required_ranges:
        affected = [i for i, s in enumerate(chunks) if s.start < end and s.end > begin]
        if not affected:
            raise BudgetExceeded("required literal could not be mapped to a source span")
        groups.append((affected[0], affected[-1]))
    # Union of overlapping index intervals; untouched chunks remain singletons.
    merged: list[EvidenceSpan] = []
    mandatory: set[int] = set()
    idx = 0
    while idx < len(chunks):
        end_index = idx
        required = False
        changed = True
        while changed:
            changed = False
            for a, b in groups:
                if a <= end_index and b >= idx:
                    required = True
                    if b > end_index:
                        end_index = b
                        changed = True
        first, last = chunks[idx], chunks[end_index]
        begin = first.start
        finish = last.end
        # Extend to literal's exact endpoints when surrounding whitespace is part
        # of the literal, without overlapping adjacent groups.
        for p, q in required_ranges:
            if p < finish and q > begin:
                begin = min(begin, p)
                finish = max(finish, q)
        if merged and begin < merged[-1].end:
            raise BudgetExceeded("required literal overlaps incompatible source spans")
        merged.append(EvidenceSpan(begin, finish, "required" if required else first.kind))
        if required:
            mandatory.add(len(merged) - 1)
        idx = end_index + 1

    def render(indices: set[int]) -> str:
        return separator.join(text[sp.start:sp.end] for i, sp in enumerate(merged) if i in indices)

    selected = set(mandatory)
    if engine._count(render(selected)) > budget:
        raise BudgetExceeded("required literal content exceeds budget")
    # A small proximity boost retains nearby explanatory sentences when they
    # fit. Only source adjacency is used; this is not semantic inference.
    relevance = [lexical_relevance(query, text[s.start:s.end]) for s in merged]
    def score(i: int) -> float:
        near = max((relevance[j] for j in (i - 1, i + 1)
                    if 0 <= j < len(merged)), default=0.0)
        return relevance[i] + 0.2 * near

    candidates = sorted((i for i in range(len(merged)) if i not in mandatory),
                        key=lambda i: (-score(i), i))
    for i in candidates:
        attempt = selected | {i}
        if engine._count(render(attempt)) <= budget:
            selected = attempt
    output = render(selected)
    if engine._count(output) > budget or any(lit not in output for lit in required_literals):
        raise BudgetExceeded("compressed output exceeds budget or breaks required evidence")
    return DocumentResult(
        output,
        tuple(sp for i, sp in enumerate(merged) if i in selected),
        tuple(sp for i, sp in enumerate(merged) if i not in selected),
        required_literals, input_tokens, engine._count(output), budget, token_counter is None,
    )


def verify_source_alignment(source: str, result: DocumentResult, *, separator: str = "\n") -> bool:
    """Verify exact selected substrings and order; never proves semantic accuracy."""
    if not isinstance(source, str):
        return False
    previous_end = -1
    fragments = []
    for span in result.selected_spans:
        if span.start < previous_end or span.start < 0 or span.end > len(source) or span.start >= span.end:
            return False
        fragments.append(source[span.start:span.end])
        previous_end = span.end
    expected = separator.join(fragments)
    # A verbatim pass-through has gaps between the trimmed source units.
    # Verify the entire expected span map, rather than trusting equality alone.
    passthrough = (
        result.text == source and not result.dropped_spans
        and result.selected_spans == split_document(source)
    )
    if result.text == source:
        return passthrough and all(lit in result.text for lit in result.required_literals)
    return expected == result.text and all(
        lit in result.text for lit in result.required_literals
    )
