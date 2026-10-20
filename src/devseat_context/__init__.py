"""Deterministic, CPU-first context selection."""

from .engine import (
    BudgetExceeded,
    ContextEngine,
    ContextResult,
    Segment,
    approximate_tokens,
    available_input_budget,
)
from .relevance import lexical_features, lexical_relevance

__all__ = [
    "BudgetExceeded", "ContextEngine", "ContextResult", "Segment",
    "approximate_tokens", "available_input_budget",
    "lexical_features", "lexical_relevance",
]
