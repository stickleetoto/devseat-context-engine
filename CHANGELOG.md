## 0.8.0 — exact-character CPU selection optimization (2026-10-20)

- Builtin-len exact-character fast paths for segment and document span selection.
- Generic/tokenizer-based counters keep full-render accounting with nonadditive safety.
- Deterministic randomized equivalence tests against lambda-len baseline.
- 50,000 segment budget smoke case and CPU p50/p95/peak Python allocation profiler.
- No semantic rewrite, no trust-policy changes or production throughput guarantee.

## 0.7.0 — opt-in risk-aware preservation (2026-10-20)

- Exact source-offset candidate signals for negation, revisions, assignment conflict and repeated IDs.
- Audit mode leaves source selection unchanged; require mode uses caller-trusted risk kinds to preserve signals or fail closed.
- Include source size and signal count quotas and independent-offset verification.
- Compare risk protection vs plain v0.6 with explicit refusal counts.
- Still CPU-only and does not automatically infer source trust or fact correctness.

## 0.6.0 — tokenizer and evidence audit (2026-10-20)

- Optional tiktoken encoding/model adapters, with explicit failure on unrecognized mappings.
- RenderedTokenCounter and FinalBudget checks with output/framing reservations.
- Standalone independent-labeled source-offset quality auditor, including duplicate evidence.
- Twelve public synthetic Korean/English/code/log/date/negation examples and command-line evaluation harness.
- Expanded cross-version CI and local optional-tokenizer test coverage.
- No claims of message-level provider-accurate counts, semantic preservation, or representative LLM task performance.

## 0.5.0 (experimental) — explicit temporal provenance (2026-10-20)

- Introduced TemporalFact with exact source offsets, source and scope labels, and optional timezone-aware observed/valid timestamps.
- Added CorrectionApproval: explicit caller-attested replacement edges; unknown, cyclic, forked and cross-scope approvals are rejected.
- Added TemporalAnalysis: unresolved/consistent/approved-correction candidate groups, active IDs and optional latest-observation hint (never a truth decision).
- Added compress_temporal: protects all active/unresolved evidence as required source spans; budget failure is explicit, previously approved-corrected facts remain auditable if still visible.
- Added strict input bounds and adversarial regression tests; CPU-only with no new runtime dependencies.
- This is not semantic entailment, logical resolution, or automatically trusted correction inference.

## 0.4.0 — experimental fact-aware compression (2026-10-20)

- Exact-offset heuristic scanning: assignments, dates, identifiers, quantities, negation clauses.
- Candidate key-value conflict discovery, with no automatic winner or truth claim.
- Fail-closed required-fact source spans and offset-based FactAudit.
- Optional required_spans protects multiple occurrences of identical literals.
- Hardened source-alignment verification on original pass-through documents.
- Public fact-retention micro-benchmark and additional CI checks.

# Changelog

## 0.3.0 (2026-10-20) — experimental

- Added CPU-only document-span extractor with source offsets, fenced-code and traceback atomic blocks, lexical query ranking and limited neighbor selection.
- Added explicit trusted `required_literals` and fail-closed overflow/missing-evidence behavior; span-alignment verifier.
- Added a four-case synthetic, character-budget benchmark with separately annotated literal recall, plus research memo and provenance-focused regression tests.
- Maintained v0.2 `ContextEngine` behavior and runtime dependency-free installation.

## 0.2.0 (2026-10-20)

- Query-aware deterministic tie-breaking with Unicode/Hangul lexical overlap.
- Caller-supplied provenance metadata, scored selection audit, and input budget reservation helper.
- UTF-8-oriented rough token estimates (not provider-accurate); validated token-count callbacks.

## 0.1.0

- Atomic segment selection, explicit priorities, mandatory protected segments, token budget tracking.