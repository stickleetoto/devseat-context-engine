# Tokenizer adapters and evidence retention — v0.6 research

Date: 2026-10-20. Development stage: experimental.

## Problem

Prior versions used approximations such as UTF-8 bytes/3 or a caller-supplied
character counter. Neither is a precise text token count for a given
language-model tokenizer. Also, token reduction cannot be equated with
downstream answer correctness. v0.6 separates three distinct concerns.

## 1. Text encoding counts

- TiktokenCounter.for_encoding(name): count token IDs encoded by the
  caller-selected BPE vocabulary for the given text.
- TiktokenCounter.for_model(name): require a known tiktoken model-to-encoding
  mapping; fail explicitly when unknown; no invented fallback.
- The optional dependency loads lazily. CPU-only core use requires no
  tiktoken installation. Encoding data may be downloaded on initial use
  and subsequently cached, so this is not always first-run offline.
- Treat strings resembling special ChatML markers as ordinary source text
  via disallowed_special=().

Reference: https://github.com/openai/tiktoken
Reference: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken

## 2. Total-window reservations

- FinalBudget models a CALLER-SPECIFIED window, output reserve, message
  overhead reserve, and safety margin. Remaining input tokens are calculated
  using v0.5's existing validated budget function.
- RenderedTokenCounter accepts a caller-owned serialization function and
  a text-token counter. This supports adding an explicitly known wrapper
  to the counted data, but cannot recreate unknown provider internals.
- verify_final_budget fails if the full supplied counted text exceeds the
  declared input limit. Users must not assume this proves entire API request
  conformance (especially tool schemas, multimodal inputs and hidden
  messages). Accurate text tokenization != precise server accounting.
- Watch for envelope double-counting: if the renderer incorporates some
  wrappers, do not add the same wrappers again in framing_overhead.

## 3. Independent reference evidence labels

- GoldEvidence specifies source character offsets, not content text
  inferred by a model. Duplicate literals are tracked independently.
- audit_quality checks source alignment and exact output span coverage.
- QualityResult contains evidence count, retained count, missing labels and
  input/output counts under the supplied counter.
- The compressor does not receive gold labels in public benchmark code.
  This helps avoid data leakage in ranking evaluation.

## Experimental baseline

Twelve small hand-authored synthetic cases with eighteen labeled spans:
- All twelve cases returned a compression result under caller's
  CHARACTER-counted budget.
- Fourteen of eighteen annotated spans survived (77.8%).
- All completed outputs passed exact-source alignment.
- Cases include English/Korean negations, amended setting values,
  repeated IDs, code, traceback-like errors, dates and paths.
- This is not a representative success rate and makes no claim about
  model token savings or user-visible answer quality. It is a smoke baseline
  that deliberately exposes loss cases for v0.7.

## Failure modes and upcoming work

- An otherwise source-aligned summary can omit one side of a correction
  or prohibition. This must be addressed by future risk gating and trusted
  required-span selection, not by claiming semantic equivalence.
- Running tokenization on each candidate whole output is potentially
  superlinear on long prompts; benchmark at 1k/10k/50k fragments.
- Third-party tokenizer mapping and provider message accounting can change.
- Realistically held-out human-annotated evaluation, adversarial content,
  independent model task correctness and CPU p50/p95 remain outstanding.
