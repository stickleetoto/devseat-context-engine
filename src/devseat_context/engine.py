"""Conservative extractive context selection, not semantic summarization.

The caller controls trust, mandatory status, and token counting. This engine
never modifies source text, and it fails closed if protected content cannot fit.
"""

from dataclasses import dataclass
import re
from typing import Callable, Sequence


class BudgetExceeded(ValueError):
    """Mandatory content does not fit within the configured token budget."""


@dataclass(frozen=True, slots=True)
class Segment:
    id: str
    text: str
    priority: int = 0
    protected: bool = False


@dataclass(frozen=True, slots=True)
class ContextResult:
    text: str
    selected_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    budget: int
    estimated_tokens: bool

    @property
    def reduction_ratio(self) -> float:
        return 0.0 if self.input_tokens == 0 else 1 - self.output_tokens / self.input_tokens


def approximate_tokens(text: str) -> int:
    """Rough budget estimate only; not safe for hard provider limits."""
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


class ContextEngine:
    def __init__(self, token_counter: Callable[[str], int] | None = None, separator: str = "\n\n"):
        self._count = token_counter or approximate_tokens
        self._estimated = token_counter is None
        self._separator = separator

    def compress(self, segments: Sequence[Segment], budget: int) -> ContextResult:
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        ids = [s.id for s in segments]
        if any(not s.id for s in segments) or len(ids) != len(set(ids)):
            raise ValueError("segment ids must be unique and nonempty")
        if any(not isinstance(s.priority, int) for s in segments):
            raise ValueError("priority must be an integer")

        def render(indices: set[int]) -> str:
            return self._separator.join(s.text for i, s in enumerate(segments) if i in indices)

        all_indices = set(range(len(segments)))
        original = render(all_indices)
        protected = {i for i, seg in enumerate(segments) if seg.protected}
        if self._count(render(protected)) > budget:
            raise BudgetExceeded("protected content exceeds budget; nothing was silently truncated")
        if self._count(original) <= budget:
            selected = all_indices
        else:
            selected = set(protected)
            candidates = sorted(all_indices - protected, key=lambda i: (-segments[i].priority, i))
            for i in candidates:
                candidate = selected | {i}
                if self._count(render(candidate)) <= budget:
                    selected = candidate
        output = render(selected)
        return ContextResult(
            text=output,
            selected_ids=tuple(s.id for i, s in enumerate(segments) if i in selected),
            dropped_ids=tuple(s.id for i, s in enumerate(segments) if i not in selected),
            input_tokens=self._count(original),
            output_tokens=self._count(output),
            budget=budget,
            estimated_tokens=self._estimated,
        )
