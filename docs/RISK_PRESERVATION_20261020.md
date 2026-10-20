# v0.7 Risk-Preservation Research

Date: 2026-10-20. Status: Experimental.

## Motivation
Earlier 12 synthetic test documents gave 14/18 reference source spans retained.
Misses clustered around negations, repeated IDs and configuration corrections.
This is not a population-level measure or model-task accuracy claim.

## Design
RiskSignal tracks exact character offsets of regex/lexical candidates:
negation (entire clause), revision marker (whole statement), assignment
conflict (each occurrence for one normalized key), and repeated identifier
(each occurrence). Repeated text in another location does not count as
preservation.

The default audit mode cannot change selection. Explicit require mode adds
caller-selected candidate risk spans to the document compressor's required
spans. Oversized required context raises BudgetExceeded, not an incomplete
compressed result. Source size and signal-count quotas reject large scans.

## Security
Regex matches are NOT trusted authority. A malicious retrieved page could
fill a prompt with fake negations and cause budget exhaustion if its content
is blindly promoted to required. Only authorized integration code should
activate strict protection on trusted source scopes. Privacy remains under
caller control; no automatic remote model calls or logging.

## Evaluation
Risk benchmark uses the same previously inspected v0.6 dev fixtures and
scores held-out-from-compressor labeled offsets only AFTER compression.
These fixtures are not unseen by developers and must not be represented
as held-out generalization evidence. Failures and successes are reported
separately. Benchmarks use character budgets, not exact model token budgets.

## Open gates
Independent manual >100 examples, false positive and false negative study,
CPU p95/memory scalability, provider-level budget accounting, and downstream
two-model task accuracy vs raw context all remain incomplete.
