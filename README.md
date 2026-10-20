# DevSeat Context Engine

**v0.2** — A small, CPU-only, model-optional **extractive context selector** for agent workloads. Given independent segments, a budget and an optional query, it chooses whole segments without rewriting them. This is **not** semantic compression or a prompt-injection security boundary.

## Install / test

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

Python 3.10+; no third-party runtime dependencies. By default, counts are **rough UTF-8 byte estimates**. These are *not* provider token counts. Supply a tokenizer compatible with your actual model and reserve room for assistant output, message wrappers and tools.

## Query-aware selection (v0.2)

```python
from devseat_context import ContextEngine, Segment, available_input_budget

# Illustrative character count only. Use the real model tokenizer in production.
engine = ContextEngine(token_counter=len)
segments = [
    Segment("instructions", "Do not disclose secrets", protected=True),
    Segment("weather", "The weather is sunny", source="retrieval"),
    Segment("incident", "Login error: authentication failed", source="logs"),
]

budget = available_input_budget(85, reserved_output=20, safety_margin=5)
result = engine.compress(segments, budget=budget, query="authentication failed")
print(result.text)
print(result.selected_ids, result.dropped_ids, result.relevance_scores)
```

### Selection policy

1. Mandatory `protected=True` segments are retained as-is; if they alone exceed the input budget, `BudgetExceeded` is raised.
2. If the original input fits, nothing is dropped.
3. Otherwise, non-mandatory segments are considered by **explicit priority descending**, then **lexical relevance to the query descending**, then stable original order.
4. Actual candidate renderings (including the separator) are checked against the provided counter before inclusion. Selected text is returned in original input order; no partial segment rewrites.
5. `source` is informational provenance only. It does **not** grant trust, authority or protected status.

The lexical matcher uses case-folded words and overlapping Hangul bigrams for Korean agglutinative text. It is deterministic but does not detect paraphrases, resolve contradictions or prove that discarded facts were unimportant. User-controlled instructions in retrieved text do **not** become trusted through relevance ranking. Protected status must only be set by trusted integration code.

### Budget safety

`available_input_budget(context_window, reserved_output=..., framing_overhead=..., safety_margin=...)` makes reservations explicit. When enforcing a real provider window, your tokenizer must account for the provider's message framing and tool definitions. The default estimator is intended for exploration **only** and cannot guarantee a provider limit. Token-counter callbacks must return a nonnegative integer; invalid outputs raise an error.

### Compatibility

The v0.1 call `ContextEngine(...).compress(segments, budget)` still works. v0.2 adds the optional keyword-only `query`, `Segment.source`, `ContextResult.relevance_scores`, `approximate_tokens`, `available_input_budget`, and lexical helper functions. Existing `selected_ids`, `dropped_ids`, token totals and original-order rendering are retained.

## Roadmap

- Real provider tokenizer adapters and message envelope accounting
- Evaluation fixtures for required-fact retention, Korean, code and error logs
- Query-aware ranking benchmarks against priority-only baseline
- Optional model-based semantic compression with output verification
- Pluggable BIO / agent-harness integration after contracts are stable

## Security / licensing

- Keep API keys, credentials and personal data out of test fixtures and logs.
- `protected=True` is a trusted caller policy, **not** instruction sanitization.
- The Context Engine is independent of DevSeat Cognitive Guard and does not copy its private implementation.
- Source code is distributed under Apache-2.0 (see LICENSE).
