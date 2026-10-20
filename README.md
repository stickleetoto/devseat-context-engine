# DevSeat Context Engine

A CPU-first, model-optional **extractive context selector**. v0.1 selects whole caller-defined segments under a budget, never rewrites text, and raises an explicit error if protected segments cannot fit.

## Install and test

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Example

```python
from devseat_context import ContextEngine, Segment

engine = ContextEngine(token_counter=len)  # DEMO: characters, not model tokens
result = engine.compress([
    Segment("system", "Never disclose secrets", protected=True),
    Segment("history", "Old conversation", priority=0),
    Segment("task", "Fix the current bug", priority=10),
], budget=45)
print(result.text, result.dropped_ids)
```

## Scope and limitations

- Deterministic priority-based whole-segment selection, original-order output
- Protected segments retained or `BudgetExceeded` raised; no silent truncation
- Auditable selected/dropped IDs and token counts
- Zero external runtime dependencies; CPU-only
- Default token counter is **only a rough estimate**, not safe for strict model budgets. Provide a provider-accurate tokenizer and allow for message framing/output reservations.
- `protected` is a trusted caller policy, not a defense against prompt injection. Never promote untrusted content to trusted instructions.
- No semantic summarization or factual preservation guarantee for dropped segments.
- No Cognitive Guard proprietary implementation copied.

Future work: precise tokenization, multilingual relevance, contradiction handling, provenance, and quality benchmarks.
