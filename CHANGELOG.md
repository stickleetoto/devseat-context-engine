# Changelog

## 0.2.0 (2026-10-20)

- Query-aware deterministic tie-breaking with Unicode and Hangul lexical overlap.
- Caller-supplied provenance metadata, scored selection audit, and explicit input budget reservation helper.
- UTF-8-oriented rough token estimates (not provider-accurate); validated token-count callbacks.
- Extra regression/security-oriented tests and Python CI matrix.
- Backwards-compatible `compress(segments, budget)` call.

## 0.1.0

- Atomic segment selection, explicit priorities, mandatory protected segments, token budget tracking.
