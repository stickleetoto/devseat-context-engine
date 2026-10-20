# Evidence-integrity gate and honesty about accuracy

Date: 2026-10-21. Status: Experimental.

## Problem

The original engine is extractive: its selected strings match source
substrings. Nevertheless, removal can change meaning by dropping
prohibitions, qualifying exceptions, old/new values or duplicated IDs.
The small v0.7 benchmark retained 16/18 annotated spans with one
budget refusal. That is not semantic or model-answer accuracy.

## Mechanism

EvidenceObligation supplies a stable label, expected literal and exact
source start/end. It must be issued by trusted caller-side application
logic. The engine validates source exactness, identity uniqueness,
boolean requirement flag, offset types and resource limits.
EvidenceBundle makes two or more member obligations jointly required
even when individual declarations are audit-only.

compress_with_integrity supports:
- audit: return omitted evidence/risk metadata without changing selection.
- review (default): raise EvidenceReviewRequired when a detected risk
  candidate of configured kinds is dropped. This is an explicit refusal,
  not a source-authorized protected instruction.
- require: add detected risk kinds to mandatory source spans, ONLY when
  a trusted caller authorizes the selected scope.

A successful required obligation always has exact original source coverage.
Required content that cannot fit raises BudgetExceeded. The API also
checks source-aligned reconstruction after compression.

## Limitations and failure modes

Regex risk scanning cannot identify all negations, implication boundaries
or false context and may flag irrelevant content. A sentence can be
source-aligned but misleading without omitted qualifying context.
No source truth, authorization, identity isolation or correction approval
is inferred. Authenticated provenance must come from the integration
layer. Malicious untrusted content can trigger review refusals (DoS);
limit size, count, retries, source trust and log exposure upstream.

## 150-case experiment (seed 8042)

Six seeded synthetic families: Korean/English prohibition, config old/new
pairs, repeated ticket ID, date/error and Windows source path. All
gold annotations are exact source occurrences. The baseline and review
arms see only source/query/budget. The trusted-contract arm is
deliberately given the exact gold offsets as caller metadata; this
evaluates mandatory preservation behavior, not automatic discovery.

Summary fields:
- successful and refused case counts
- exact-span recall among successful compressions
- effective reference coverage counting refusals as zero.

Oracle-assisted successful retention is expected to equal 100% by design,
but the fraction of refused calls can rise under strict budgets.
No provider token measurements, human-reviewed held-out scenarios or
paired raw-vs-compressed real model tasks are present.

## v1 release blockers

Independently reviewed 100+ held-out cases, corrections/negation scope,
per-tenant evidence contracts, actual downstream model task comparison
using two or more models, provider reported token usage, production CPU
budgets and recovery/authorization validation are required.
