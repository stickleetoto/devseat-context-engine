# v0.5 — Temporal Provenance, scoped correction edges

Date: 2026-10-20. Status: EXPERIMENTAL. Runtime: Python 3.10+, CPU-only.

## Problem

The v0.4 regex scanner can flag two assignments with a matching key and
different values. That does not establish which value is true. The values
may refer to different organizations, environments, or points in time.
Selecting the last occurrence as truth would silently corrupt context.

## Model

- `TemporalFact`: source-indexed immutable evidence with explicit `scope`,
  `key`, `source_id`, `observed_at`, `valid_at` and exact source offsets.
  `observed_at` and `valid_at` are independent; neither is inferred from
  text or used to declare a winner.
- `CorrectionApproval`: external auditable edge `newer_id -> older_id`.
  The approval ID is only a *label*, not a signature or authorization check.
  Only a trusted caller may create/authorize these edges. Untrusted source
  material mentioning 'correction' is never parsed into an approval.
- `TemporalGroup`: precise scope plus casefolded key. Different live literal
  values are marked unresolved; an explicitly approved predecessor becomes
  superseded. Equal literal values are consistent at the syntactic level only.
- `compress_temporal`: includes all live/unsuperseded facts as exact required
  spans using the v0.4 document extractor. A budget overflow raises an error.
  Correction-approved older evidence becomes optional and is never silently
  rewritten. If still in output, its record IDs appear in
  `visible_superseded_ids`.

## Isolation and integrity requirements

1. `text[start:end] == value` for every evidence record. No fabricated
   values or mismatched offsets may be accepted.
2. Scope string equality, key normalization and unique record IDs are required
   for any correction edge. Cross-tenant edges are refused.
3. No cycles, self-edges, forks or unknown correction targets.
4. Time values must be timezone-aware datetimes. UTC order is only a hint:
   a later observation is *not* automatically a correction.
5. Resource guards default to 100,000 source characters and 256 records.
   Both are adjustable caller quotas, not comprehensive denial-of-service
   protection. Untrusted records/approvals must be rejected upstream.

## Test strategy

- Regression tests cover unknown/unapproved corrections, cycles, forks,
  cross-scope links, bad offsets, duplicate identities, timezone offsets,
  missing timestamps, budget overflow, and source-embedded instruction
  spoofing.
- CI must pass on Python 3.10, 3.12 and 3.13, with correctness lint enabled.
- No claims are made about actual LLM task accuracy, token saving, semantic
  correctness, or large corpus CPU performance.

## Open issues

- Source scope/ownership authorization must remain outside this library.
- Claim normalization and multilingual semantic equivalence are absent.
- The approval graph is memory-resident and lacks signed attestation,
  persistent event history, retraction and delegation.
- The temporal extractor only protects supplied records; other potentially
  critical source facts, negations and unannotated corrections can be omitted.
- Approved old evidence may still be returned when the source fits the
  budget, so callers must inspect the returned audit before making claims.
- Real provider tokenizers, contradiction reasoning and downstream LLM
  evaluation are future work.

## Next gate

Build a held-out temporal correctness dataset with human-reviewed historical
corrections, scopes, valid-time intervals, and mixed Korean/English examples.
Compare raw context, v0.4, and v0.5 on 2+ model tasks with error categories.
