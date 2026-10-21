# v0.9.0 experimental offline CPU learning ranker

Date: 2026-10-21. Not production qualified.

## Motivation

Current v0.8.2 exact-offset evidence obligation API preserves explicitly
required source material or refuses compression. The heuristic risk detector
recognized 150/225 annotated offsets in its own synthetic fixture and
the review policy refused 104/150 synthetic cases. This is not a reason
to bypass integrity protection: instead, optional ranking can help retain
more query-relevant *optional* source material within a limited budget.

## Mechanism

- Tiny frozen logistic regression model (pure Python, no additional
  runtime dependencies, no external inference API).
- Fourteen bounded features: lexical query overlap, content error/date/
  path/assignment/negation/ticket cues, six query-intent interactions and
  content length fraction.
- Offline seeded stochastic gradient descent; epochs, learning rate and
  L2 penalties bounded. Model has finite numeric weights and a schema.
- Serialization is JSON with SHA256 checksum for accidental corruption.
  SHA256 is NOT authentication; the caller must trust file provenance.
- Query preparation caches only within a single call, no global cross-
  user query cache. No training examples or user text in model export.
- Model cannot elevate untrusted content to an EvidenceObligation, approve
  corrections, turn off risk modes, or override mandatory source offsets.
- Model is opt-in and disabled by default on all four entry points.

## Development benchmark

72 hand-authored example pairs in six categories (error, date, file path,
config assignment, prohibition, ticket). 48 train examples in 24 groups;
24 dev examples in 12 different groups. One positive and one negative
per query group. Dataset loader rejects group leakage, duplicate query/
text pairs and invalid labels. Split is group-disjoint but is NOT an
independent held-out benchmark; it was authored alongside the model
features and is vulnerable to overfitting.

Linux Python CI run on this development fixture:
- Legacy lexical top-1: 9 / 12 (75%).
- Optional trained ranker top-1: 12 / 12 (100%).

This metric is *only* ranking the correct of two examples under a
character-count budget. It does not measure actual LLM answers, unseen
queries, recall@k on multiple passages, provider-reported token counts,
NDCG, corrections or source truth. No production quality claim.

## Threat model and limitations

User-provided query and retrieved text are DATA, not instructions or
authorization evidence. Model features may match malicious text, which
could raise its rank, but protected content and exact obligations remain
mandatory. App integrations must verify source ownership and avoid
promoting model scores to trust policies.

Do not train directly on raw private conversations or generated labels
without user consent, a privacy plan and strict group-separation tests.
This version does not support learning online from unchecked success
signals. As seen in prior DevSeat context research, a failed compressed
task does not prove that every omitted item caused the failure.

## Release-gate follow-ups

- Independently sourced >100 human-reviewed real agent contexts, with
  group splits by actual session/repo and adversarial distractors.
- Real-world ranking NDCG/Recall@k, calibration, false-positive rate,
  important fact omissions and increased refusal rate.
- Measure inference p50/p95 on Windows with the candidate model enabled,
  plus Python process RAM and actual provider BPE token counts.
- Paired raw-context vs compressed-context downstream tasks on two
  LLMs. Require no statistically credible reliability regression
  before allowing default ranker promotion.
- BIO and Agent Harness adapters still belong to the remaining v0.9
  integration and source isolation phase.

## Reproduce

~~~powershell
python -m pytest -q
python -m devseat_context.ranker_benchmark
python -m devseat_context.ranker_benchmark --model-out cpu-ranker.json
python -m devseat_context.ranker_perf --sizes 1000 10000 50000 --repetitions 5
~~~

Training runs synchronously and locally and is not scheduled or
background work. The model-out option is explicit.
