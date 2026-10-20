"""CPU-first source-aligned context selection and fact auditing."""

from .document import (
    DocumentResult,
    EvidenceSpan,
    MissingEvidence,
    compress_document,
    split_document,
    verify_source_alignment,
)
from .engine import (
    BudgetExceeded,
    ContextEngine,
    ContextResult,
    Segment,
    approximate_tokens,
    available_input_budget,
)
from .facts import (
    AssignmentConflict,
    Fact,
    FactAudit,
    FactAwareResult,
    audit_facts,
    candidate_conflicts,
    compress_fact_aware,
    extract_facts,
)
from .relevance import lexical_features, lexical_relevance
from .temporal import (
    CorrectionApproval,
    TemporalAnalysis,
    TemporalCompressionResult,
    TemporalEvidenceError,
    TemporalFact,
    TemporalGroup,
    analyze_temporal,
    compress_temporal,
)

__all__ = [
    "AssignmentConflict",
    "BudgetExceeded",
    "CorrectionApproval",
    "ContextEngine",
    "ContextResult",
    "DocumentResult",
    "EvidenceSpan",
    "Fact",
    "FactAudit",
    "FactAwareResult",
    "MissingEvidence",
    "Segment",
    "TemporalAnalysis",
    "TemporalCompressionResult",
    "TemporalEvidenceError",
    "TemporalFact",
    "TemporalGroup",
    "approximate_tokens",
    "analyze_temporal",
    "audit_facts",
    "available_input_budget",
    "candidate_conflicts",
    "compress_document",
    "compress_fact_aware",
    "compress_temporal",
    "extract_facts",
    "lexical_features",
    "lexical_relevance",
    "split_document",
    "verify_source_alignment",
]
