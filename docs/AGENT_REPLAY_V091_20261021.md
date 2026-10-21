# v0.9.1 — Agent Replay & Learning Harness, first offline phase

Date: 2026-10-21. Status: Experimental / not connected to Codex.

## Scope of completed implementation

A human/operator creates a **local, consented** test suite of questions,
candidate context segments, hard character budgets and independently
reviewed gold segment IDs. This is passed into an offline local CLI.
The harness evaluates the existing lexical ranking and, if a user
explicitly provides an approved local model file, the optional
v0.9.0 learned CPU ranker against **the same** input and budget.
It reports per-mode successful/refused cases, retained gold IDs,
retention conditioned on success, refusal-adjusted coverage,
protected-segment violations and paired wins/ties.

It does NOT read existing Codex session files or connect to a Codex
API, navigate the user desktop, synthesize labels, train from replay,
run live coding tasks, prove source authorization, or measure
actual LLM downstream task success. Group fields are accepted for
manual split tracking, but are not returned in aggregates.
Known gold IDs are human annotations; the engine does not discover them.

## Privacy and governance

- Never collect agent or project logs automatically.
- Explicit opt-in to evaluation-only processing is required in the
  local schema. The flag itself is **not proof** of valid consent.
- Review and redact secrets, private identifiers, third-party code
  and confidential user materials BEFORE writing local JSON.
- Keep local suites under .replay-private/ (gitignored) and reports
  under .replay-reports/ (also gitignored).
- Output JSON contains only total counts, ratios and generic descriptions,
  never raw prompt, snippet contents, case IDs, gold IDs or per-case results.
- No network access, telemetry, persistence, model training or auto-uploads.
- File size <= 2MB; 500 cases, 500 items/case, 10k segments total;
  each text <= 20k chars and query <= 2,048 chars.
- The benchmark reference is character length, not true provider
  BPE tokens or an agent end-to-end success score.

## How to create a real evaluation sample safely

1. Choose ONE user-authorized, non-sensitive task from actual Codex
   development work. Do not export existing sessions wholesale.
2. Independently review the candidate snippets and select which
   original *segment IDs* are truly necessary to solve the task.
   Confirm this from tests, fixes and inspected source; don't label
   by whether a ranker picked it.
3. Locally compose a JSON suite matching benchmarks/replay_synthetic.json,
   with explicit_approval true for local evaluation only.
4. Run the replay command on the local file and compare aggregate
   evidence recall/refusal. Use a second model file only if trusted.
5. Keep new tasks out of training until evidence IDs have been
   independently verified and a project/session-disjoint evaluation
   partition is frozen.
6. For genuine task-quality claims, add a future harness capable of
   executing comparable tasks with original vs compressed context,
   capturing test outcomes, run failures and provider-reported tokens.
   That feature is NOT present in v0.9.1.

## Reproduce

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-local
.\.venv\Scripts\python.exe -m devseat_context.agent_replay --suite benchmarks/replay_synthetic.json
.\.venv\Scripts\python.exe -m devseat_context.agent_replay --suite benchmarks/replay_synthetic.json --model cpu-ranker.json
~~~

CI runs the included synthetic fixture on Windows/Linux. This checks
the harness plumbing, NOT performance with genuine Codex workloads.
The default ranker remains unchanged and mandatory protected inputs
still fail closed when a budget cannot accommodate them.
