# DevSeat Context Engine — v1.0 readiness and Codex handoff plan

Decision: GPT directly implements and researches through v1.0. Once v1.0
is validated and frozen, new feature development moves to Codex. This is
a workflow decision, not an estimate of timeline.

## Current stable branch

main receives only CI-verified and reviewed changes. Versions under 1.0
remain experimental and may fail closed on untrusted or incomplete input.

## Proposed gates

### v0.6 (this PR)
- Optional provider-associated text BPE token counting.
- No silent tokenizer fallbacks.
- Independent source-offset evidence audit and synthetic examples.
- Pytest, correctness lint and cross-version CI passing.

### v0.7 (next)
- Mixed Korean/English lexical improvements and structured negation / status
  change risk detectors, with offset traceability.
- Evidence-loss risk profiles, plus explicit review/fail-closed policy.
- Extend datasets past 100 *held-out*, manually verified cases; report all
  misses instead of optimizing only a demo benchmark.
- Respect trusted caller scope; never promote attacker-supplied content.

### v0.8 (experimental exact-character fast path)
- CPU p50/p95, memory and complexity tests at 1k/10k/50k spans.
- Fast stable selection with non-additive tokenizer edge cases tested.
- Robust streaming/multi-turn interfaces, persistence-free by default.
- Reproduce developer-machine benchmark settings, hardware and seeds.

### v0.8.2 (experimental evidence-integrity gate)

- Caller-authorized exact source-offset obligations and all-required bundles.
- Default risk review refuses detected omissions; declared required evidence
  is always retained on success or the compressor fails closed.
- 150 synthetic generated cases; independent human-reviewed held-out
  evaluation and downstream model tests still open.

### v0.9.0 — experimental optional CPU learning ranker

- Tiny immutable offline-trained scorer used only for relevance order;
  default OFF, with no ability to authorize sources or override protection.
- Group-disjoint handcrafted training/development cases; not independent
  human-held-out, not model-answer quality.
- Deterministic retraining, split leakage validation and opt-in CPU profiles.
- Remaining v0.9 adapter integrations and authorization tests still open.

### v0.9.1 — offline agent replay and shadow evaluation

- Caller-authorized bounded local replay suites with human-reviewed segment
  labels and deterministic baseline vs opt-in trained ranker comparisons.
- Aggregate-only reports, explicit refusal accounting and privacy guards.
- No automatic reading of Codex transcripts, self-learning from unverified
  completions, actual model-task execution or source authorization.
- Production-ready task-success and true held-out measurements are still open.

### v0.9 — remaining adapter/security work
- Well-specified adapter to DevSeat Agent Harness, typed BIO retrieval
  provenance, and optional Agent-X-Ray tracing.
- Audit leakage prevention, invalid scope isolation, crash/cancel/fallback.
- Optional real model-based semantics behind explicit opt-in and
  independent comparison; CPU-only core remains functional.

### v1.0 release gate (must pass before Codex handoff)
- Core public API frozen with versioned compatibility and examples.
- Security model / source trust / secret-handling review documented.
- 100+ held-out scenarios including corrections and adversarial negatives.
- Explicit required facts: 100% retained on successful returns or
  BudgetExceeded instead of silent omission, verified across tests.
- Downstream tasks with at least two models compared to raw-context
  baselines, reporting task success, exact retention and failure counts
  separately. No promise of fixed compression ratios.
- Provider token budget verified with documented serialization assumptions.
- Windows and Linux CI; Python supported version matrix.
- Documented CPU p50/p95 and peak memory in realistic workloads.
- Reproducible release artifact, changelog, license, handoff document
  with API contracts, known debt and evidence links.

## After v1.0

Codex: implement iterative features, refactoring, cross-provider adapters,
benchmarks and CI, without silently changing trust boundaries or tests.
GPT: architecture/experiment review, counterexamples, validation gates and
release decisions. All merges use reviewed PRs and gated CI.

No automatic model invocation, user data export, or permission escalation
is authorized by this plan.
