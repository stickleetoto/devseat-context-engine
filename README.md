# DevSeat Context Engine

**v0.9.1 experimental** — A CPU-first, model-optional context toolkit for query-aware selection, source-aligned extraction, fact auditing, and explicit temporal corrections. It does not provide semantic equivalence guarantees, automatically determine truth, or sanitize prompt injection.

## Quick start

```powershell
git clone https://github.com/stickleetoto/devseat-context-engine.git
cd devseat-context-engine
python -m pip install -e ".[dev]"
python -m pytest -q
python -m devseat_context.benchmark benchmarks/fixtures.json
python -m devseat_context.fact_benchmark
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

## v0.9.1 experimental — offline Agent Replay & Shadow Evaluation

This is the first step toward *real-world Codex-driven tuning*, **not** a
connector that secretly accesses or controls Codex. The tool replays context
selection on a caller-provided, manually reviewed local JSON suite. It
compares the unchanged lexical baseline with an explicitly selected,
locally trained CPU ranker, and exports only aggregate evidence-retention
and refusal metrics. It never trains automatically.

~~~powershell
# Smoke test bundled artificial sample without a learned ranker
python -m devseat_context.agent_replay --suite benchmarks/replay_synthetic.json

# Compare both modes with an existing, explicitly approved local weights file
python -m devseat_context.agent_replay --suite benchmarks/replay_synthetic.json --model cpu-ranker.json

# In Windows with restricted system Temp permissions
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-local
~~~

To start evaluating actual development work, review/authorize one Codex
task yourself, **remove secrets and private content before writing it**,
and create a local suite in .replay-private/. Do not automatically
copy agent transcripts, tool outputs, email, account data, code from private
repositories, tokens, or users' prompts into training/evaluation data.
The tool requires an explicit_approval flag for local evaluation only,
but a JSON flag cannot verify consent or data labeling quality.

Suite schema version 1:
- schema: 1; data_use: local_evaluation_only; explicit_approval: true
- cases: entries with id, group, query, budget_characters, segments, gold_ids
- each segment: id, text, priority (integer), protected (boolean)
- gold_ids: caller-reviewed exact segment IDs needed for the task, not
  learner-generated labels or self-asserted instructions from source text

There are strict file, case, segment and text size bounds and schema checks.
Reports omit all original identifiers, text, questions and per-case guesses.
A baseline refusal and a missed gold label are separately counted.
A model score never changes protected-segment policy. Results reflect
**character budgets** and caller-provided evidence labels, NOT agent
task completion, real provider-token usage, validated semantics or
independently tested generalization.

For the capture/evaluation protocol see
[Agent replay research](docs/AGENT_REPLAY_V091_20261021.md).

## v0.9.0 experimental — opt-in learned CPU ranker

The optional CPU ranker is a small logistic regression model implemented
entirely in standard-library Python. It trains locally on explicitly labeled
query/candidate pairs and scores ranking only. No pretrained LLM, sklearn,
GPU or external API is required. The default lexical scorer is unchanged.

~~~python
from devseat_context import (
    ContextEngine, Segment, TrainingExample, train_cpu_ranker,
    LinearCPURanker,
)

rows = [
    TrainingExample("why login failed", "E401 authentication failure", True, "a"),
    TrainingExample("why login failed", "Lunch reminder", False, "a"),
    TrainingExample("오류 파일 위치", r"C:\services\auth.py", True, "b"),
    TrainingExample("오류 파일 위치", "Meeting notes", False, "b"),
]
model = train_cpu_ranker(rows, epochs=80, seed=17)
model.save("cpu-ranker.json")
model = LinearCPURanker.load("cpu-ranker.json")

items = [
    Segment("note", "Team lunch meeting"),
    Segment("log", "Request failed with E401"),
]
result = ContextEngine(token_counter=len).compress(
    items, budget=35, query="why did login fail", ranker=model,
)
print(result.selected_ids, result.output_tokens)
~~~

The ranker parameter is supported as explicit opt-in by ContextEngine,
compress_document, compress_risk_aware, and compress_with_integrity.
It cannot elevate untrusted source contents, bypass protected segments,
ignore caller-required exact spans, or avoid final budget checks.
The serialized model contains weights only, not examples. A checksum
detects accidental changes, NOT an attacker with model-file access.
Load only models approved by your own application.

~~~powershell
python -m devseat_context.ranker_benchmark
python -m devseat_context.ranker_benchmark --model-out cpu-ranker.json
python -m devseat_context.ranker_perf --sizes 1000 10000 --repetitions 5
~~~

The 72 example pairs are small handcrafted synthetic development data
(48 training rows in 24 groups, 24 evaluation rows in 12 other groups).
The old lexical scorer selected the correct item in 9 of 12 dev groups;
the learned scorer did so in 12 of 12. This is NOT independent held-out
generalization, semantic QA correctness or a safe production promotion.
The data was authored while the features were being developed.

Read the [research memo](docs/CPU_LEARNING_RANKER_20261021.md) before using
the model in an agent. Keep the default rule-based path as a fallback.

## v0.8.2 experimental — exact evidence-integrity and review gate

This adds an explicit, caller-owned evidence contract. Important facts are
tracked by exact source character offsets, not by occurrences of the same
string elsewhere. Source documents do not grant themselves authority.

~~~python
from devseat_context import (
    EvidenceObligation, EvidenceBundle, BudgetExceeded,
    EvidenceReviewRequired, compress_with_integrity,
)

source = "timeout_ms=100. Other notes. timeout_ms=500."
old_text = "timeout_ms=100"
new_text = "timeout_ms=500"
pos = source.index(new_text)
old = EvidenceObligation("old", 0, len(old_text), old_text)
new = EvidenceObligation("new", pos, pos + len(new_text), new_text)
try:
    result = compress_with_integrity(
        source, budget=33, query="notes", token_counter=len,
        obligations=(old, new),
        bundles=(EvidenceBundle("history", ("old", "new")),),
        risk_mode="review",
    )
except EvidenceReviewRequired:
    print("Recognized risk omitted: manual review required")
except BudgetExceeded:
    print("Required evidence cannot fit: increase budget")
else:
    print(result.text, result.retained_labels)
~~~

Three risk modes are available: audit (report losses only), review (default;
fail closed if recognized selected-risk kinds are omitted) and require
(caller-authorized automatic preservation of chosen heuristic risk kinds).
The latter must not be used indiscriminately on untrusted retrieved text:
an attacker could make the entire document appear mandatory.

Every successful call retains all explicit mandatory obligations and
members of evidence bundles. A refused call is NOT counted as success.
Source-character, obligation, bundle and scanner-signal quotas apply.

This does not establish semantic entailment, complete detection of critical
content, factual truth or answer equivalence. No model or GPU is required.

~~~powershell
python -m pytest -q
python -m devseat_context.integrity_benchmark --count 150 --seed 8042
~~~

The benchmark is 150 code-generated cases, not independent human-reviewed
held-out work. In the current CI run, the plain selector kept 74/225
annotated source spans; default review refused 104/150 cases and the
trust-supplied contract refused 76/150, preserving all required spans in
its 74 successful cases. The detector itself only recognized 150/225
benchmark reference spans (none of the error/date or path references). The trusted-contract experimental arm intentionally receives
the ground-truth source offsets; its success rate evaluates the contract,
not automatic detection. Reports separate refusals and evidence retention.

See [Evidence integrity analysis](docs/EVIDENCE_INTEGRITY_20261021.md).

## v0.8.1 experimental — Prepared lexical relevance (CPU performance)

The v0.8 50,000-fragment Windows benchmark showed p50 = 3,411.9ms,
and a 10,000-fragment controlled cProfile reproduction attributed ~85%
of measured cumulative time to lexical relevance scoring. The same query
was being tokenized once *per fragment*, even when it never changed.

v0.8.1 computes query features **once per compression call**, then
scores each fragment against that prepared set. For English-only queries,
it skips Hangul-bigram scanning that cannot affect the score. This
preserves all original scores, ordering, protected evidence and budgets.

- No global query cache: temporary lexical state does not persist between
  requests or tenants.
- No learned model, network calls, GPU, or public API change required.
- Original `lexical_relevance(query, text)` remains available and identical;
  the prepared scorer is a reusable helper in `devseat_context.relevance`.
- The exact-character optimization from v0.8 and conservative provider
  tokenizer path are unchanged.

To reproduce on your Windows laptop:

~~~powershell
python -m devseat_context.perf_profile --segments 10000 --top 15
python -m devseat_context.perf_benchmark --sizes 1000 10000 50000 --repetitions 5
~~~

The profiler runs only synthetic text and prints function timings rather
than private prompt data. Profiler timing is inflated by cProfile overhead;
use the benchmark CLI for real latency comparisons. Verify on identical
hardware and power conditions. See
[CPU hotspot investigation](docs/LEXICAL_HOTSPOT_20261021.md).

## v0.8 experimental: additive CPU fast path

When the caller explicitly supplies Python's built-in len as the counter, the
whole-segment and source-span selectors now track exact fragment and separator
lengths without repeatedly assembling the entire candidate output.

This is mathematically exact for CHARACTER budgets only. For BPE tokenizers,
UTF-8 estimates, and other custom counters, the original conservative full
render-and-recount path remains unchanged. The fast path does not make
provider-token quotas accurate. Existing protected spans, source order,
deterministic ranking, errors and audit metadata are preserved.

To compare on your own CPU, with recorded Python/platform, sampled p50/p95
wall time, and peak Python allocation tracking:

~~~powershell
python -m devseat_context.perf_benchmark --sizes 1000 10000 50000 --repetitions 5
~~~

The legacy generic-lambda baseline is only profiled for small inputs by
default (1000 segments), because its repeated re-rendering scales poorly.
The profiler uses tracemalloc, which affects timing; run on an idle machine
with the same options when comparing hardware. A 5-run p95 is still noisy.
These are NOT whole-agent latency, GPU latency, or exact model-token savings.

See [v0.8 CPU research](docs/CPU_PERFORMANCE_20261020.md).

## v0.7 experimental: risk detection and opt-in protection

Detect candidate source spans for English/Korean negation, revision markers,
different literal values for the same assignment key and repeated IDs.
This is not a semantic truth engine and does not approve corrections.

~~~python
from devseat_context import compress_risk_aware, BudgetExceeded

source = "timeout_ms=100. timeout_ms=500. Never delete backup."
try:
    result = compress_risk_aware(
        source, 65, query="timeout", token_counter=len,
        policy="require",
        protected_kinds=frozenset({"negation", "assignment_conflict"})
    )
except BudgetExceeded:
    print("Protected evidence did not fit")
else:
    print(result.text, result.report.review_required)
~~~

Default audit mode retains the v0.6 selection algorithm and reports candidate
loss. Require mode is ONLY for caller-approved trusted contexts and either
retains every protected source occurrence or raises BudgetExceeded. Inputs
have source-character and signal-count quotas. Untrusted content can fabricate
important-looking statements to cause denial of service: do not elevate them
to trusted rules.

~~~powershell
python -m pytest -q
python -m devseat_context.risk_benchmark
~~~

The benchmark compares previously inspected synthetic development cases. Gold
reference offsets are used for evaluation only, and explicit refusals are
reported separately. It does not demonstrate real LLM task success.

See [risk policy notes](docs/RISK_PRESERVATION_20261020.md).

## v0.6 experimental — explicit tokenizer and retention audit

**New:** Optional real BPE tokenizer counter and independently labeled source-offset
retention evaluation. The runtime remains pure Python; tiktoken is an OPTIONAL
extra and no LLM API key is required.

~~~powershell
python -m pip install -e ".[dev,tokenizers]"
python -m pytest -q
python -m devseat_context.quality_benchmark
~~~

~~~python
from devseat_context import (
    TiktokenCounter, FinalBudget, verify_final_budget,
    compress_document, audit_quality, locate_gold,
)

counter = TiktokenCounter.for_encoding("cl100k_base")
source = "Unrelated memo. E401 authentication failure. Other notes."
budget = FinalBudget(
    context_window=100,
    output_reservation=30, framing_overhead=10, safety_margin=10
)
result = compress_document(
    source, budget=budget.input_limit,
    query="authentication failure", token_counter=counter,
)
verify_final_budget(result.text, counter, budget)
gold = [locate_gold(source, "E401", "error-id")]
print(audit_quality(source, result, gold).exact_span_recall)
~~~

- TiktokenCounter.for_encoding(name) uses an explicitly verified BPE
  encoding; for_model(name) uses tiktoken's known mapping. Unknown names
  are rejected, NOT silently mapped to another model. Initial tiktoken
  encoding-data load may require internet access before cache is populated.
- Only **plain text encoding** is measured. Entire provider API requests may
  count roles, chat delimiters, tools, images and additional messages.
  FinalBudget reservations are caller estimates and do not claim API-level
  exactness. RenderedTokenCounter can count a caller-serialized plain-text
  envelope when the caller knows that format; it is not a turnkey chat
  serializer. Do not count known wrapper tokens twice.
- Independent GoldEvidence labels and source offsets are used for evaluation,
  NEVER passed to the compressor. QualityResult measures exact span coverage
  and source alignment, NOT truth, semantics or task accuracy.
- 12 synthetic fixture cases with 18 annotated spans provide a regression
  smoke baseline. Their **14/18** retained-span outcome is limited to a tiny
  hand-built dataset under character-based budgets; it is NOT a broad metric.
  In particular, negations and corrected values may be dropped by the
  plain document selector. Use trusted required-span policy when a known
  critical fact must be preserved.

See [tokenization/quality engineering notes](docs/TOKENIZATION_EVALUATION_20261020.md)
and [v1 readiness gates](docs/V1_READINESS.md).

## v0.5 experimental — Temporal Provenance

The **temporal evidence policy** distinguishes observation time from truth.
It uses exact source offsets and explicit, externally verified correction
approvals to avoid silently discarding unresolved conflicts.

```python
from devseat_context import TemporalFact, CorrectionApproval, compress_temporal

source = "timeout_ms=300. Unrelated memo. timeout_ms=500."
old = TemporalFact("old", "tenant-a", "timeout_ms", "timeout_ms=300", 0, 14)
pos = source.index("timeout_ms=500")
new = TemporalFact("new", "tenant-a", "timeout_ms", "timeout_ms=500",
                   pos, pos + len("timeout_ms=500"))

# Approval must come from a trusted caller or audited system, NOT source text.
result = compress_temporal(
    source, [old, new], budget=15,
    approved_corrections=(CorrectionApproval("new", "old", "ticket-124"),),
    token_counter=len,   # demonstration character counter, not model tokens
)
print(result.text)                          # newest approved evidence
print(result.analysis.groups[0].status)     # "approved_correction"
print(result.visible_superseded_ids)        # previously corrected facts still visible
```

Without explicit approval, **both differing values remain required** and an
insufficient budget raises `BudgetExceeded`. Timestamps are hints about
observation order only; they never automatically overwrite older facts.

The record metadata and correction approval must originate in trustworthy
integration code. The engine validates that `value` is the **exact substring**
at `[start:end]` and that approvals connect the same explicit scope and key.
It rejects cycles, correction forks, cross-scope edges, naive timestamps and
unknown record IDs. Supported metadata includes aware `datetime` objects for
`observed_at` and `valid_at`; parsing arbitrary natural-language dates is
not implemented.

**Caveat:** approved-corrected source text is eligible for dropping but not
automatically erased. When it remains visible (e.g. source fits the budget),
consult `visible_superseded_ids` before sending it to an LLM. Other
unannotated facts and prohibitions are NOT automatically protected by this
temporal-specific API. Use the v0.4 fact policy separately where appropriate.
Every declared active/unresolved temporal record is protected, so untrusted
record metadata could exhaust the budget. Validate approvals, scopes, record
count, permissions and source trust at the API boundary.

See [temporal design and risk notes](docs/TEMPORAL_PROVENANCE_20261020.md).

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
