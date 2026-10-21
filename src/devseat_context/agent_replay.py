"""Explicit opt-in, offline shadow replay of labeled agent-context selections.

No agent instrumentation, Codex log access, background collection, training,
network calls or raw prompt export is performed. Input is provided by a caller
who has separately reviewed its provenance and disclosure permissions.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .engine import BudgetExceeded, ContextEngine, Segment
from .learning import LinearCPURanker

_MAX_FILE_BYTES = 2_000_000
_MAX_CASES = 500
_MAX_SEGMENTS = 500
_MAX_TOTAL_SEGMENTS = 10_000
_MAX_TEXT = 20_000
_MAX_QUERY = 2_048
_MODES = ("lexical", "learned_cpu")


class ReplayValidationError(ValueError):
    """Malformed or non-consented input; evaluation does not proceed."""


@dataclass(frozen=True, slots=True)
class ReplayCase:
    id: str
    group: str
    query: str
    budget_characters: int
    segments: tuple[Segment, ...]
    gold_ids: tuple[str, ...]


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 128


def parse_replay_suite(data: Any) -> tuple[ReplayCase, ...]:
    """Require explicit suite-level evaluation consent and bounded inputs.

    Gold IDs must have been independently reviewed by the caller. Merely
    setting the consent flag cannot prove review or correct labels.
    """
    if (not isinstance(data, dict) or set(data) != {
        "schema", "data_use", "explicit_approval", "cases"
    } or type(data["schema"]) is not int or data["schema"] != 1
            or data["data_use"] != "local_evaluation_only"
            or data["explicit_approval"] is not True
            or not isinstance(data["cases"], list)
            or not 1 <= len(data["cases"]) <= _MAX_CASES):
        raise ReplayValidationError("suite schema, purpose, consent or case limit invalid")
    cases = []
    ids = set()
    total_segments = 0
    for entry in data["cases"]:
        if (not isinstance(entry, dict) or set(entry) != {
            "id", "group", "query", "budget_characters", "segments", "gold_ids"
        } or not _identifier(entry["id"]) or entry["id"] in ids
                or not _identifier(entry["group"])
                or not isinstance(entry["query"], str)
                or len(entry["query"]) > _MAX_QUERY
                or type(entry["budget_characters"]) is not int
                or entry["budget_characters"] < 0
                or not isinstance(entry["segments"], list)
                or not 1 <= len(entry["segments"]) <= _MAX_SEGMENTS
                or not isinstance(entry["gold_ids"], list)
                or not entry["gold_ids"]):
            raise ReplayValidationError("invalid replay case")
        ids.add(entry["id"])
        total_segments += len(entry["segments"])
        if total_segments > _MAX_TOTAL_SEGMENTS:
            raise ReplayValidationError("total segment limit exceeded")
        parts = []
        part_ids = set()
        for part in entry["segments"]:
            if (not isinstance(part, dict)
                    or set(part) != {"id", "text", "priority", "protected"}
                    or not _identifier(part["id"])
                    or part["id"] in part_ids
                    or not isinstance(part["text"], str)
                    or len(part["text"]) > _MAX_TEXT
                    or type(part["priority"]) is not int
                    or not -100 <= part["priority"] <= 100
                    or type(part["protected"]) is not bool):
                raise ReplayValidationError("invalid segment or duplicate segment ID")
            part_ids.add(part["id"])
            parts.append(Segment(
                part["id"], part["text"], part["priority"], part["protected"]
            ))
        gold = entry["gold_ids"]
        if (any(not _identifier(item) or item not in part_ids for item in gold)
                or len(gold) != len(set(gold))):
            raise ReplayValidationError("gold IDs must reference distinct original segments")
        cases.append(ReplayCase(
            entry["id"], entry["group"], entry["query"],
            entry["budget_characters"], tuple(parts), tuple(gold),
        ))
    return tuple(cases)


def read_replay_suite(path: str | Path) -> tuple[ReplayCase, ...]:
    file = Path(path)
    if file.stat().st_size > _MAX_FILE_BYTES:
        raise ReplayValidationError("suite file too large")
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReplayValidationError("invalid UTF-8 JSON replay file") from exc
    return parse_replay_suite(data)


def evaluate_replay(
    cases: tuple[ReplayCase, ...],
    *,
    ranker: LinearCPURanker | None = None,
) -> dict:
    """Compare selection against reviewed IDs; no external agent tasks run.

    The returned report is aggregate only: never serialize source fragments,
    prompts, case IDs, group names, selected IDs or per-case predictions.
    """
    if (not isinstance(cases, tuple) or not cases
            or any(not isinstance(case, ReplayCase) for case in cases)):
        raise ValueError("nonempty ReplayCase tuple required")
    if ranker is not None and not isinstance(ranker, LinearCPURanker):
        raise TypeError("ranker must be a checked LinearCPURanker or None")
    selected = {}
    summary = {}
    for mode in (_MODES if ranker is not None else ("lexical",)):
        totals = {
            "cases": len(cases), "successful": 0, "refused": 0,
            "gold_spans": 0, "gold_retained": 0,
            "successful_gold_spans": 0, "total_input_characters": 0,
            "total_output_characters_on_success": 0,
            "protected_violations": 0,
        }
        decisions = []
        for case in cases:
            gold = set(case.gold_ids)
            totals["gold_spans"] += len(gold)
            totals["total_input_characters"] += len(
                "\n".join(seg.text for seg in case.segments)
            )
            try:
                result = ContextEngine(len, separator="\n").compress(
                    case.segments, case.budget_characters,
                    query=case.query, ranker=ranker if mode == "learned_cpu" else None,
                )
            except BudgetExceeded:
                totals["refused"] += 1
                decisions.append(None)
                continue
            protected = {seg.id for seg in case.segments if seg.protected}
            chosen = set(result.selected_ids)
            if not protected <= chosen or result.output_tokens > case.budget_characters:
                # This should be unreachable: never report unsafe success.
                raise AssertionError("replay detected a protected/budget regression")
            totals["successful"] += 1
            totals["total_output_characters_on_success"] += result.output_tokens
            retained = len(gold & chosen)
            totals["successful_gold_spans"] += len(gold)
            totals["gold_retained"] += retained
            decisions.append((retained, len(gold)))
        summary[mode] = {
            **totals,
            "recall_on_success": (
                totals["gold_retained"] / totals["successful_gold_spans"]
                if totals["successful_gold_spans"] else None
            ),
            "effective_recall_including_refusals": (
                totals["gold_retained"] / totals["gold_spans"]
            ),
        }
        selected[mode] = decisions
    comparisons = None
    if ranker is not None:
        comparisons = {
            "both_success": 0, "both_refused": 0, "lexical_only_success": 0,
            "learned_only_success": 0, "lexical_more_gold": 0,
            "learned_more_gold": 0, "equal_gold": 0,
        }
        for legacy, trained in zip(selected["lexical"], selected["learned_cpu"]):
            if legacy is None and trained is None:
                comparisons["both_refused"] += 1
            elif legacy is None:
                comparisons["learned_only_success"] += 1
            elif trained is None:
                comparisons["lexical_only_success"] += 1
            else:
                comparisons["both_success"] += 1
                if legacy[0] > trained[0]:
                    comparisons["lexical_more_gold"] += 1
                elif legacy[0] < trained[0]:
                    comparisons["learned_more_gold"] += 1
                else:
                    comparisons["equal_gold"] += 1
    return {
        "schema": 1,
        "task_success_measured": False,
        "counter": "Python len, character budgets, not provider tokens",
        "source": "caller-supplied, explicitly approved local replay cases",
        "modes": summary, "paired_comparison": comparisons,
        "warning": (
            "Gold IDs are caller-provided; source review is not independently "
            "verified. No Codex run, model-answer evaluation, or training occurred."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None,
                        help="approved local model JSON; omitted = lexical only")
    parser.add_argument("--report", type=Path, default=None,
                        help="optional aggregate-only JSON result")
    args = parser.parse_args()
    cases = read_replay_suite(args.suite)
    model = LinearCPURanker.load(args.model) if args.model else None
    output = evaluate_replay(cases, ranker=model)
    result = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)
    if args.report:
        args.report.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
