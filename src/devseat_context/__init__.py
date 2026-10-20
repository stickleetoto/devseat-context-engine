"""Deterministic, CPU-first context selection."""

from .engine import BudgetExceeded, ContextEngine, ContextResult, Segment

__all__ = ["BudgetExceeded", "ContextEngine", "ContextResult", "Segment"]
