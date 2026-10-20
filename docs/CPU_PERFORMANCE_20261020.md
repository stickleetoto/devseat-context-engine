# v0.8 CPU Fast Path — Research and correctness notes

Date: 2026-10-20. Status: experimental, not a production p95 claim.

## Bottleneck

The v0.7 selectors rejoined all previously accepted fragments for every
new candidate to measure its full token count. Under many source units this
can generate quadratic text work, even for an additive character counter.

## Optimization and safety boundary

Only when the supplied callable IS Python's built-in len (identity checked),
the result length can be calculated by exact fragment character lengths
plus the separator length for each boundary. A selected result with k>0
fragments has exactly SUM(len(fragment)) + (k-1)*len(separator) characters.
This preserves score ordering, original rendering order and protected units.

Tiktoken and other BPE tokenizers are not additive across fragment boundaries.
For arbitrary user callbacks, even callbacks that happen to return len(text),
the implementation retains the old full-render/recount algorithm. No
tokenizer heuristic or hidden fallback is introduced.

## Evaluation

Tests compare builtin len and equivalent lambda-len over deterministic
randomized segment/document cases, mandatory span overflow, zero-length
segments, Korean/English text, nonadditive counters and 50k segments.
Tests must match the entire immutable returned result or fail with the
same BudgetExceeded type.

The performance harness accepts size, repetitions and legacy-max switches:

  python -m devseat_context.perf_benchmark --sizes 1000 10000 50000 --repetitions 5

It records p50/p95 wall milliseconds using nearest-rank aggregation,
median of tracemalloc's peak Python allocations, platform and Python
version. Beware tracer overhead and tiny repetition counts. CI runners
are shared hardware, so do not use their numbers as latency SLAs.

## Open risks

- Document required-span merging scans each required span against chunks.
  Extremely many mandatory ranges can still become superlinear.
- Lexical relevance computation and other regex scans remain CPU work.
- In-memory context assembly can still require substantial allocations.
- Generic BPE counters can still trigger whole-context re-rendering
  and need a separate, explicitly justified tokenizer-aware strategy.
- Chunked/streaming ingestion, cancellation/backpressure, large realistic
  corpus benchmarking and bounded application memory are not complete.
- Current benchmark is on synthetic fragments, not BIO retrieval,
  private user conversations or realistic downstream agent tasks.

## Release gate

Continue to v0.9 only after validating semantic/risk regressions, API
compatibility, source-aligned proof, CPU measurements on target hardware
and isolated provider-token policy. v1.0 remains gated separately.
