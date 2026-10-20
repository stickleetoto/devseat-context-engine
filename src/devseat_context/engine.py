"""Conservative extractive context selection, not semantic summarization.

Only trusted callers can designate protected segments. A model tokenizer should
be supplied for strict context windows; the built-in estimate is not exact.
"""

from dataclasses import dataclass
import math
from typing import Callable, Sequence

from .relevance import lexical_relevance


class BudgetExceeded(ValueError):
    """Mandatory content does not fit within the configured input budget."""


@dataclass(frozen=True, slots=True)
class Segment:
    id: str
    text: str
    priority: int = 0
    protected: bool = False
    source: str | None = None


@dataclass(frozen=True, slots=True)
class ContextResult:
    text: str
    selected_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    budget: int
    estimated_tokens: bool
    relevance_scores: tuple[tuple[str, float], ...] = ()

    @property
    def reduction_ratio(self) -> float:
        return 0.0 if self.input_tokens == 0 else 1 - self.output_tokens / self.input_tokens


def approximate_tokens(text: str) -> int:
    """Byte-based rough estimate, including multi-byte text; never an exact limit.

    A short Korean sentence must not misleadingly count as one word/token.
    """
    return math.ceil(len(text.encode("utf-8")) / 3) if text else 0


def available_input_budget(context_window: int, *, reserved_output: int = 0,
                           framing_overhead: int = 0, safety_margin: int = 0) -> int:
    """Compute usable input budget after externally estimated reservations."""
    values = (context_window, reserved_output, framing_overhead, safety_margin)
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError("all budget components must be nonnegative integers")
    remaining = context_window - reserved_output - framing_overhead - safety_margin
    if remaining < 0:
        raise BudgetExceeded("output/framing/safety reservations exceed context window")
    return remaining


class ContextEngine:
    def __init__(self, token_counter: Callable[[str], int] | None = None, separator: str = "\n\n"):
        if token_counter is not None and not callable(token_counter):
            raise TypeError("token_counter must be callable")
        if not isinstance(separator, str):
            raise TypeError("separator must be a string")
        self._counter = token_counter or approximate_tokens
        self._estimated = token_counter is None
        self._separator = separator

    def _count(self, text: str) -> int:
        value = self._counter(text)
        if type(value) is not int or value < 0:
            raise ValueError("token_counter must return a nonnegative integer")
        return value

    def compress(self, segments: Sequence[Segment], budget: int, *, query: str | None = None) -> ContextResult:
        """Select full segments; priority wins, then lexical relevance, then input order.

        Protected segments are mandatory. Input order and exact text are retained
        for selected segments. ``source`` is metadata, not a trust designation.
        """
        if type(budget) is not int or budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        if query is not None and not isinstance(query, str):
            raise TypeError("query must be a string or None")
        seq = tuple(segments)
        if any(not isinstance(s, Segment) for s in seq):
            raise TypeError("segments must contain Segment instances")
        ids = [s.id for s in seq]
        if any(not isinstance(s.id, str) or not s.id for s in seq) or len(ids) != len(set(ids)):
            raise ValueError("segment ids must be unique nonempty strings")
        if any(type(s.priority) is not int for s in seq):
            raise ValueError("priority must be an integer")
        if any(not isinstance(s.text, str) for s in seq):
            raise TypeError("segment text must be a string")
        if any(type(s.protected) is not bool for s in seq):
            raise TypeError("protected must be a boolean")

        def render(indices: set[int]) -> str:
            return self._separator.join(s.text for i, s in enumerate(seq) if i in indices)

        all_indices = set(range(len(seq)))
        original = render(all_indices)
        original_tokens = self._count(original)
        protected = {i for i, seg in enumerate(seq) if seg.protected}
        if self._count(render(protected)) > budget:
            raise BudgetExceeded("protected content exceeds budget; nothing was silently truncated")
        scores = tuple((s.id, lexical_relevance(query, s.text)) for s in seq) if query else ()
        if original_tokens <= budget:
            selected = all_indices
        else:
            selected = set(protected)
            by_id = dict(scores)
            candidates = sorted(all_indices - protected,
                                key=lambda i: (-seq[i].priority, -by_id.get(seq[i].id, 0.0), i))
            for i in candidates:
                attempt = selected | {i}
                if self._count(render(attempt)) <= budget:
                    selected = attempt
        output = render(selected)
        output_tokens = self._count(output)
        if output_tokens > budget:
            raise BudgetExceeded("output exceeds budget according to supplied counter")
        return ContextResult(
            text=output,
            selected_ids=tuple(s.id for i, s in enumerate(seq) if i in selected),
            dropped_ids=tuple(s.id for i, s in enumerate(seq) if i not in selected),
            input_tokens=original_tokens,
            output_tokens=output_tokens,
            budget=budget,
            estimated_tokens=self._estimated,
            relevance_scores=scores,
        )
