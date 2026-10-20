# Prepared query relevance — CPU hotspot investigation (2026-10-21)

Status: experimentally validated for deterministic score equivalence and synthetic performance. **Not** a provider-level or production SLA.

## Observed user environment (v0.8)

Windows 11, Python 3.13.4, 5 repetitions, tracemalloc enabled:
- 1,000 fragments: p50 31.616ms; p95 33.799ms
- 10,000 fragments: p50 334.103ms; p95 593.405ms
- 50,000 fragments: p50 3411.949ms; p95 3694.358ms
- 50,000 peak Python-traced allocation median 19.247MiB

The source code calculates lexical_relevance(query, segment) for every segment, which reparses the same query each time, and repeatedly tests Hangul patterns even for Latin-only terms.

## Isolated profiling (separate Python environment; faithful algorithm reproduction)

10,000 synthetic fragments: cProfile total 227ms. Around 193ms cumulative in lexical_relevance, 176ms in lexical_features, 12ms in sorting. Because cProfile introduces overhead and this is a separate CPU, these proportions are *hotspot evidence*, not measurements on the user's laptop.

Independent same-process micro-benchmark, 5 repetitions, no tracer, 50,000 fragments:
- Old algorithm reproduction: median 362.57ms
- Prepared-query version reproduction: median 145.46ms
- ~2.49x speedup, with output text and exact score tuple equality.

## Change and invariants

- Precompute frozenset query features once per compress call.
- The prepared scorer tests full word matches and, only when query features contain Hangul bigrams, relevant Hangul bigram matches. Duplicate feature matches count once.
- No globally cached private queries or model calls.
- Retain the public lexical_relevance implementation as an independent baseline.
- Document/segment selection, source offsets, protected content, dedup semantics, ordering, API, and strict BPE-token accounting stay unchanged.
- Tests compare scores exactly on Korean, English, Unicode casefold, Hangul, empty query, punctuation, and many seeded mixed-language inputs. End-to-end ContextResult and DocumentResult are compared with the legacy relevance function.

## GitHub CI Linux benchmark reference

Prior v0.8 (3 runs with tracemalloc):
- 50k fragments: p50 2559.794ms / p95 2567.460ms, Python-traced peak 19.257MiB.

v0.8.1 candidate in a later hosted runner (3 runs with tracemalloc):
- 50k fragments: p50 988.193ms / p95 1004.334ms, Python-traced peak 19.248MiB.

The improvement is consistent with the local A/B experiment but the hosted-runner comparison is NOT hardware-pinned or randomized. Do not infer a Windows p50 below 2 seconds until the laptop is retested. Synthetic logs repeat fixed query terms; real-world gains depend on query complexity, language mix, number of unique fragments and competing processes.

## Relevant prior internal research

Reused the design principle of deterministic, audit-aligned reduction, but **not** any private source code. Real semantic reliability and down-stream model-answer fidelity remain separate unfinished v1.0 release gates.

## Remaining bottlenecks

- Each unique text fragment is still casefolded/regex-scanned (linear in text size).
- Scores for every fragment are exposed publicly; cannot silently switch to approximate early top-k without changing result metadata.
- Sorting all candidate priority/score tuples is O(n log n).
- Large batches of required source ranges in document.py may still show quadratic merging work.
- Tokenizer/non-additive budget counters still use the safe full-render recount path.
- No long-lived global cache, as avoiding retention of sensitive user queries is deliberate.
