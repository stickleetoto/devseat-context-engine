"""Compare ordinary extractive selection and an opt-in risk-protection policy.

Gold annotations are used only AFTER candidate compression. A refusal is
reported as fail_closed, not turned into a successful retention score.
Fixtures are small development smoke tests, not unseen downstream workloads.
"""
import argparse
import json
from pathlib import Path

from .document import compress_document
from .engine import BudgetExceeded
from .quality import audit_quality, locate_gold
from .risk import RiskLimitExceeded, compress_risk_aware


def compare_risk_cases(cases: list[dict]) -> dict:
    records = []
    names = set()
    for case in cases:
        name, source, query, budget = (
            case["id"], case["source"], case["query"], case["budget_characters"]
        )
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("case IDs must be unique and nonempty")
        names.add(name)
        if not isinstance(source, str) or not isinstance(query, str):
            raise TypeError("source/query must be strings")
        if type(budget) is not int or budget < 0:
            raise ValueError("budget must be nonnegative int")
        gold = tuple(locate_gold(
            source, item["literal"], item["label"],
            occurrence=item.get("occurrence", 0)
        ) for item in case["evidence"])
        for mode in ("v0.6_plain", "v0.7_risk_require"):
            try:
                if mode == "v0.6_plain":
                    result = compress_document(
                        source, budget, query=query, token_counter=len
                    )
                    dropped_signal_count = 0
                else:
                    evaluated = compress_risk_aware(
                        source, budget, query=query, policy="require",
                        # By design, revision markers alone are weak and remain
                        # audit-only. Conflicts, repetitions and negations
                        # are protected by trusted policy in this experiment.
                        protected_kinds=frozenset({
                            "negation", "assignment_conflict", "repeated_identifier"
                        }),
                        token_counter=len,
                    )
                    result = evaluated.document
                    dropped_signal_count = len(evaluated.report.dropped)
                quality = audit_quality(source, result, gold)
                if not quality.source_aligned or result.output_tokens > budget:
                    raise AssertionError("source alignment or hard budget failed")
                records.append({
                    "id": name, "mode": mode, "status": "ok",
                    "retained": quality.retained_count, "total": len(gold),
                    "dropped_signals": dropped_signal_count,
                    "output_characters": result.output_tokens,
                    "input_characters": len(source),
                })
            except (BudgetExceeded, RiskLimitExceeded):
                records.append({
                    "id": name, "mode": mode, "status": "fail_closed",
                    "retained": 0, "total": len(gold),
                    "dropped_signals": None,
                    "output_characters": 0, "input_characters": len(source),
                })
    summary = {}
    for mode in ("v0.6_plain", "v0.7_risk_require"):
        group = [r for r in records if r["mode"] == mode]
        hits = sum(r["retained"] for r in group)
        expected = sum(r["total"] for r in group)
        summary[mode] = {
            "successful_samples": sum(r["status"] == "ok" for r in group),
            "fail_closed_samples": sum(r["status"] == "fail_closed" for r in group),
            "retained_exact_spans": hits,
            "all_gold_spans": expected,
            "retention_including_failures": hits / expected if expected else 1.0,
        }
    return {"summary": summary, "records": records}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("fixtures", nargs="?", default=str(
        Path(__file__).resolve().parents[2] / "benchmarks" / "quality_cases.json"
    ))
    a = p.parse_args()
    cases = json.loads(Path(a.fixtures).read_text(encoding="utf-8"))
    data = compare_risk_cases(cases)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print("CAUTION: hand-built character-budget smoke examples; "
          "no semantic truth, model quality, or representative success claim.")


if __name__ == "__main__":
    main()
