# Fact-Aware Compression v0.4 — Research Notes (2026-10-20)

## Aim and transfer

v0.4 expands CPU-first extractive selection with deterministic, source-aligned,
regex-detected fact candidates. The approach favors auditable evidence over
speculative paraphrasing. The design builds on v0.3's evidence alignment;
it does **not** reproduce learned methods such as LLMLingua-2.

## Technical additions

1. Recognize selected assignment, date, identifier, quantity, and
   English/Korean prohibition patterns, retaining exact source offsets.
2. Group assignments with the same case-folded key; if literal values differ,
   mark a *candidate* conflict and expose all original instances. A candidate
   is NOT a semantic contradiction or evidence that the last value is correct.
3. Preserve explicitly protected fact kinds at individual exact source
   offsets, including repeated equal values. Raise BudgetExceeded if required
   content cannot fit; never quietly drop a protected instance.
4. Audit retention by source-offset coverage rather than text equality.
   Record dropped recognized facts and all candidate conflicts.
5. Verify uncompressed passthrough span maps instead of accepting arbitrary
   remapped spans merely because the output matches the original.

## Benchmark methodology

- Five tiny synthetic fact-preservation examples under character budgets.
- Separate v0.3 span extraction and v0.4 strict fact protection; both receive
  only source, query and budget. Gold literals are read *after* compression.
- Record retained literal counts, budget failure counts and conflict candidates.
- Do not report successful-case recall without also disclosing fail-closed
  cases. No estimate of typical LLM accuracy or token savings is supported.
- Benchmark code and input examples are reproducible in the repository.

## Known failure modes

- Regex misses implicit facts and may incorrectly classify irrelevant strings.
- Negation-pattern recognition is not a grammar or logical inference engine.
- Two assignments with the same key could refer to different scopes, dates,
  tenants or environments; automatic conflict resolution would be unsafe.
- Different-source duplicates and edits are not normalized or disambiguated.
- Strict preservation can be defeated as an availability mechanism by an
  untrusted document packed with apparent facts; enforce caller-side quotas
  and trusted classification before using as an agent tool.
- Sensitive keys may expose credentials; callers must handle output/audit
  under appropriate retention and privacy rules.
- No provider-specific tokenizer, multilingual semantic entailment, temporal
  provenance or downstream LLM task evaluation yet.
- Current merging/ranking can show superlinear performance with many fragments.

## Next gates

1. Held-out 100+ source documents with scope/negation/correction cases.
2. Explicit source, namespace, timestamp metadata for conflict candidates.
3. CPU time and memory p50/p95 vs source length and fact count.
4. Provider-accurate tokenizers and message envelope reservation tests.
5. Two independent LLM downstream-task evaluations versus uncompressed input.
