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