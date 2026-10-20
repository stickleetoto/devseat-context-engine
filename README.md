# DevSeat Context Engine

**v0.3 experimental** — A CPU-first, model-optional context **selection + document span extraction** toolkit. This is not model-based semantic compression, lossless encoding, or a prompt-injection sanitizer.

## Quick start

```powershell
git clone https://github.com/stickleetoto/devseat-context-engine.git
cd devseat-context-engine
python -m pip install -e ".[dev]"
python -m pytest -q
python -m devseat_context.benchmark benchmarks/fixtures.json
```

Runs with Python 3.10+. The runtime uses only the standard library. `pytest` is a development-only dependency.

### v0.3: exact-span document compression

```python
from devseat_context import compress_document, verify_source_alignment

source = "Old unrelated note. Login error E401 occurred at 10:42. Another irrelevant note."
result = compress_document(
    source, budget=49, query="login error", required_literals=("E401",),
    token_counter=len,  # demo character budget, NOT real model tokens
)
print(result.text)
print(result.selected_spans, result.reduction_ratio)
assert verify_source_alignment(source, result)
```

- Uses sentence-like text units, with fenced code and stack traces kept atomic.
- Ranks units by query lexical overlap plus limited adjacent-context retention.
- `required_literals` are exact, **trusted caller-supplied** strings that must survive or the operation raises `BudgetExceeded`; missing source literals raise `MissingEvidence`.
- Selected span offsets point to **literal substrings of the original source**. The alignment verifier catches fabricated/altered selected text; it does **not** prove that omissions preserve meaning.
- The original input is returned *verbatim* when within budget.
- Source references, required literals and compressed content may contain secrets. Do not log them without an explicit privacy policy.

### v0.2: whole-segment selection (unchanged)

```python
from devseat_context import ContextEngine, Segment, available_input_budget
engine = ContextEngine(token_counter=len)
segments = [
    Segment("instructions", "Never disclose secrets", protected=True),
    Segment("history", "Old unrelated conversation"),
    Segment("incident", "Login failed with E401"),
]
budget = available_input_budget(100, reserved_output=20, safety_margin=10)
result = engine.compress(segments, budget=budget, query="login failed")
print(result.text, result.selected_ids, result.dropped_ids)
```

The caller must mark trusted mandatory sections `protected=True`. Source content, query content and `source` metadata cannot self-promote to trust. The `Segment` API is backwards compatible with v0.2.

### Evaluation

`python -m devseat_context.benchmark` compares **paragraph-level v0.2 selection** against **span-level v0.3 extraction** over four synthetic Korean/English/code/ID fixtures. Gold expected literals are used **only in evaluation**, not given to the compressor. Outputs are measured in **character budgets**, not provider tokens. This is *not* a downstream LLM quality score or representative performance claim; see [research memo](docs/RESEARCH_20261020.md).

### Budget disclaimer

The default tokenizer is only a rough UTF-8 estimate. For production provide a counter matching the actual model and reserve output/framing/token overhead. Token counts can be non-additive, so candidate renderings are recounted; no hard window guarantee is made for the fallback estimate.

### License / status

Apache-2.0. This is a new independent implementation; no proprietary Cognitive Guard code was imported. Experimental modules are not yet benchmark-qualified for general deployment.

## v0.4 experimental — Fact-Aware Compression

This release adds **heuristic fact detection, candidate assignment conflicts,
and auditable exact-offset fact protection**, not semantic rewriting or a claim
of knowing which fact is true.

~~~python
from devseat_context import compress_fact_aware

source = (
    "Old note. timeout_ms=300. New note. timeout_ms=500. "
    "Do not delete logs. Weather is sunny."
)
result = compress_fact_aware(
    source, budget=55, query="timeout failure",
    token_counter=len,  # character budget demonstration, NOT model tokens
)
print(result.text)
print(result.audit.conflicts)
print(result.audit.dropped)
print(result.audit.fact_recall)
~~~

- Fact kinds: assignment (key=value and limited key: value), ISO-like date,
  identifier, quantity, whole negation/prohibition clause. All detections
  include original source string offsets.
- Different literal values under the same normalized assignment key are
  **candidate conflicts only**. No recency or truth is inferred.
- Default strict policy requires all recognized assignments and negation
  clauses; budget overflow raises BudgetExceeded.
- Other fact kinds are audited but not automatically mandatory.
  Pass required_kinds=frozenset() for audit-only ranking, or specify
  required_kinds=frozenset({"assignment","negation","identifier"}).
- FactAudit counts precise source offsets, not matching text elsewhere.
- compress_document accepts required_spans=((start,end), ...) to guard
  individual repeated occurrences.
- The v0.3 public API remains usable; no private Guard or BIO code copied.

**Security:** strict default fact protection over arbitrary *untrusted* text
is susceptible to budget-exhaustion denial of service if that text contains
many fabricated key-values or prohibitions. Upstream integrations must
classify trust, impose resource quotas and only promote approved source data
to a required status. Output/logs can contain personal data or credentials.

**Validation:**

~~~powershell
python -m pytest -q
python -m ruff check src tests
python -m devseat_context.fact_benchmark
~~~

The fact benchmark uses just **five synthetic examples** and character
budgets. Gold literals are kept out of the compression input. This measures
literal retention and fail-closed errors, not LLM answer correctness, semantic
faithfulness, tokenizer-exact savings, or representative system performance.
See [v0.4 research memo](docs/FACT_AWARE_RESEARCH_20261020.md).
