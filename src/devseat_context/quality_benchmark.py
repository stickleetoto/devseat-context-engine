"""Run: python -m devseat_context.quality_benchmark

Tiny held-out-label smoke benchmark, NOT downstream language-model quality.
The labeled gold evidence is never passed to the compressor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .document import compress_document
from .engine import BudgetExceeded
from .quality import audit_quality, locate_gold


def run_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(cases, list):
        raise TypeError("cases must be a list")
    rows = []
    seen: set[str] = set()
    for case in cases:
        name = case["id"]
        source = case["source"]
        query = case["query"]
        budget = case["budget_characters"]
        evidence = case["evidence"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("case id must be unique nonempty string")
        seen.add(name)
        if not isinstance(source, str) or not isinstance(query, str):
            raise ValueError("case source/query must be strings")
        if type(budget) is not int or budget < 0:
            raise ValueError("budget must be nonnegative integer")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("case must have nonempty gold evidence list")
        annotations = tuple(
            locate_gold(source, f["literal"], f["label"],
                        occurrence=f.get("occurrence", 0))
            for f in evidence
        )
        # The compressor intentionally receives ONLY source, query and budget.
        try:
            result = compress_document(
                source, budget, query=query, token_counter=len
            )
        except BudgetExceeded:
            rows.append({"id": name, "status": "fail_closed", "retained": 0,
                         "total": len(annotations), "source_aligned": False,
                         "original_characters": len(source), "output_characters": 0})
            continue
        report = audit_quality(source, result, annotations)
        rows.append({"id": name, "status": "ok",
                     "retained": report.retained_count,
                     "total": report.evidence_count,
                     "source_aligned": report.source_aligned,
                     "original_characters": report.input_tokens,
                     "output_characters": report.output_tokens})
    completed = [row for row in rows if row["status"] == "ok"]
    hits = sum(row["retained"] for row in completed)
    attempted = sum(row["total"] for row in rows)
    return {
        "samples": len(rows),
        "successful_samples": len(completed),
        "fail_closed_samples": len(rows) - len(completed),
        "exact_evidence_hits": hits,
        "all_labeled_spans": attempted,
        "micro_retention_including_failures": hits / attempted if attempted else 1.0,
        "all_completed_source_aligned": all(row["source_aligned"] for row in completed),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixtures", nargs="?",
        default=str(Path(__file__).resolve().parents[2] /
                    "benchmarks" / "quality_cases.json")
    )
    args = parser.parse_args()
    cases = json.loads(Path(args.fixtures).read_text(encoding="utf-8"))
    print(json.dumps(run_cases(cases), ensure_ascii=False, indent=2))
    print("WARNING: hand-built source-offset literal retention only; "
          "character budgets, no semantic correctness or LLM task score.")


if __name__ == "__main__":
    main()
