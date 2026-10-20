"""Evidence-integrity gate for source-aligned extractive compression.

Exact source obligations are caller-provided metadata, not text instructions.
The gate can guarantee literal span retention ON SUCCESS, not that an
unannotated document's meaning, logical implications, or answers survive.
"""

from dataclasses import dataclass
from typing import Callable, Literal

from .document import DocumentResult, compress_document, verify_source_alignment
from .engine import BudgetExceeded
from .risk import RiskReport, audit_risks, detect_risks

_RISK_KINDS = frozenset(
    {"negation", "revision_marker", "assignment_conflict", "repeated_identifier"}
)


class EvidenceContractError(ValueError):
    """Invalid trusted-caller evidence metadata or source mismatch."""


class EvidenceReviewRequired(BudgetExceeded):
    """A recognized risk was omitted; output must not be treated as approved."""


@dataclass(frozen=True, slots=True)
class EvidenceObligation:
    """Source-offset reference with explicit expected bytes-as-text and stable ID.

    required=False means *audit only*. It is NOT a statement that evidence
    can safely be omitted. The caller owns that decision.
    """

    label: str
    start: int
    end: int
    expected_text: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """All named pieces are linked and *all required* if this bundle is given.

    Useful for caller-verified old/new values, a prohibition plus its
    exception, and repeated identifiers with different source occurrences.
    """

    label: str
    member_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IntegrityResult:
    document: DocumentResult
    obligations: tuple[EvidenceObligation, ...]
    required_labels: tuple[str, ...]
    retained_labels: tuple[str, ...]
    dropped_labels: tuple[str, ...]
    risk_report: RiskReport
    risk_mode: str

    @property
    def text(self) -> str:
        return self.document.text

    @property
    def review_required(self) -> bool:
        return bool(self.dropped_labels) or self.risk_report.review_required


def _validate_contract(
    source: str,
    obligations: tuple[EvidenceObligation, ...],
    bundles: tuple[EvidenceBundle, ...],
    *,
    max_source_chars: int,
    max_obligations: int,
    max_bundles: int,
) -> set[str]:
    if not isinstance(source, str):
        raise TypeError("source must be str")
    if type(max_source_chars) is not int or max_source_chars < 0:
        raise ValueError("max_source_chars must be nonnegative int")
    if type(max_obligations) is not int or max_obligations < 0:
        raise ValueError("max_obligations must be nonnegative int")
    if type(max_bundles) is not int or max_bundles < 0:
        raise ValueError("max_bundles must be nonnegative int")
    if (len(source) > max_source_chars or len(obligations) > max_obligations
            or len(bundles) > max_bundles):
        raise EvidenceContractError("evidence contract resource limit exceeded")
    if any(not isinstance(o, EvidenceObligation) for o in obligations):
        raise TypeError("obligations must contain EvidenceObligation")
    by_label = {}
    for item in obligations:
        if (not isinstance(item.label, str) or not item.label
                or len(item.label) > 128 or item.label in by_label):
            raise EvidenceContractError("invalid or duplicate obligation label")
        if (type(item.start) is not int or type(item.end) is not int
                or not 0 <= item.start < item.end <= len(source)
                or not isinstance(item.expected_text, str)
                or source[item.start:item.end] != item.expected_text):
            raise EvidenceContractError("evidence does not match exact source offsets")
        if type(item.required) is not bool:
            raise EvidenceContractError("required must be bool")
        by_label[item.label] = item
    required = {o.label for o in obligations if o.required}
    bundle_names = set()
    for bundle in bundles:
        if not isinstance(bundle, EvidenceBundle):
            raise TypeError("bundles must contain EvidenceBundle")
        if (not isinstance(bundle.label, str) or not bundle.label
                or len(bundle.label) > 128 or bundle.label in bundle_names):
            raise EvidenceContractError("invalid or duplicate bundle label")
        bundle_names.add(bundle.label)
        members = bundle.member_labels
        if (not isinstance(members, tuple) or len(members) < 2
                or any(not isinstance(v, str) for v in members)
                or len(set(members)) != len(members)
                or any(v not in by_label for v in members)):
            raise EvidenceContractError("bundle members must be distinct known labels")
        required.update(members)
    return required


def compress_with_integrity(
    source: str,
    budget: int,
    *,
    query: str = "",
    obligations: tuple[EvidenceObligation, ...] = (),
    bundles: tuple[EvidenceBundle, ...] = (),
    risk_mode: Literal["audit", "review", "require"] = "review",
    risk_kinds: frozenset[str] = _RISK_KINDS,
    token_counter: Callable[[str], int] | None = None,
    separator: str = "\n",
    max_source_chars: int = 100_000,
    max_obligations: int = 256,
    max_bundles: int = 256,
    max_signals: int = 256,
) -> IntegrityResult:
    """Compress with verified source obligations and explicit risk policy.

    audit: return risk/obligation omissions in audit, no implicit approval.
    review: fail closed on omitted recognized selected-risk *candidates*.
    require: protect selected recognized risk spans only on caller-authorized
             trusted scope; untrusted documents could exhaust the budget.

    An arbitrary source sentence claiming "system" or "corrected" never
    becomes an EvidenceObligation or EvidenceBundle.
    """
    if not isinstance(obligations, tuple) or not isinstance(bundles, tuple):
        raise TypeError("obligations and bundles must be tuples")
    if risk_mode not in ("audit", "review", "require"):
        raise ValueError("risk_mode must be audit/review/require")
    if not isinstance(risk_kinds, frozenset) or not risk_kinds <= _RISK_KINDS:
        raise ValueError("risk_kinds must be a frozenset of supported kinds")
    required = _validate_contract(
        source, obligations, bundles,
        max_source_chars=max_source_chars, max_obligations=max_obligations,
        max_bundles=max_bundles,
    )
    signals = detect_risks(
        source, max_source_chars=max_source_chars, max_signals=max_signals,
    )
    required_spans = tuple(
        (o.start, o.end) for o in obligations if o.label in required
    )
    if risk_mode == "require":
        required_spans += tuple(
            (s.start, s.end) for s in signals if s.kind in risk_kinds
        )
    document = compress_document(
        source, budget, query=query,
        required_spans=required_spans,
        token_counter=token_counter, separator=separator,
    )
    if not verify_source_alignment(source, document, separator=separator):
        raise EvidenceContractError("output is not source-aligned")
    retained = tuple(
        item.label for item in obligations
        if any(span.start <= item.start and span.end >= item.end
               for span in document.selected_spans)
    )
    retained_set = set(retained)
    dropped = tuple(
        item.label for item in obligations if item.label not in retained_set
    )
    if any(label in required for label in dropped):
        raise BudgetExceeded("required source evidence omitted by compressor")
    audit = audit_risks(
        source, document, signals, policy="require" if risk_mode == "require" else "audit",
        separator=separator,
    )
    lost_protected = any(s.kind in risk_kinds for s in audit.dropped)
    if risk_mode == "review" and lost_protected:
        raise EvidenceReviewRequired("recognized risk evidence requires review")
    if risk_mode == "require" and lost_protected:
        raise BudgetExceeded("required risk candidate was omitted")
    return IntegrityResult(
        document=document, obligations=obligations,
        required_labels=tuple(o.label for o in obligations if o.label in required),
        retained_labels=retained, dropped_labels=dropped,
        risk_report=audit, risk_mode=risk_mode,
    )
