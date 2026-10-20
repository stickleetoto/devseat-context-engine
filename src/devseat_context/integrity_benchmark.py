"""Offline integrity benchmark with independent source-offset reference labels.

Reference labels are deliberately supplied as *trusted caller metadata* only
for the contract experiment. That is an oracle-assisted contract test, NOT a
claim of automatic evidence discovery or downstream model answer accuracy.
"""

import argparse
import json
import random
from pathlib import Path

from .document import compress_document
from .engine import BudgetExceeded
from .integrity import (
    EvidenceBundle, EvidenceObligation, compress_with_integrity,
)
from .quality import audit_quality, locate_gold

FAMILIES = (
    "en_prohibition", "ko_prohibition", "config_revision",
    "ticket_history", "error_and_date", "source_path",
)


def generate_cases(count: int = 150, seed: int = 8042) -> list[dict]:
    if type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError("count must be positive int and seed must be int")
    rng = random.Random(seed)
    cases = []
    for i in range(count):
        family = FAMILIES[i % len(FAMILIES)]
        ticket = f"INC-{1000 + i}"
        code = f"E{400 + i}"
        number = 100 + i
        if family == "en_prohibition":
            needle = f"Never delete backup-{number}"
            source = f"Weather was clear. {needle}. The next sprint starts tomorrow."
            gold = [(needle, "prohibition", 0)]
        elif family == "ko_prohibition":
            needle = f"로그-{number}는 삭제해서는 안 됩니다"
            source = f"오후에는 회의가 있습니다. {needle}. 점심 기록을 남겼습니다."
            gold = [(needle, "prohibition", 0)]
        elif family == "config_revision":
            old = f"timeout_ms={number}"
            new = f"timeout_ms={number + 400}"
            source = f"{old}. Previously applied setting. {new}. Revised setting is recorded."
            gold = [(old, "old", 0), (new, "new", 0)]
        elif family == "ticket_history":
            source = (f"Ticket {ticket} was opened. "
                      f"Unrelated memo. Ticket {ticket} was closed.")
            gold = [(ticket, "open", 0), (ticket, "closed", 1)]
        elif family == "error_and_date":
            date = f"2026-10-{(i % 27) + 1:02d}"
            source = f"Morning update. Incident {code} occurred on {date}. Afternoon review."
            gold = [(code, "code", 0), (date, "date", 0)]
        else:
            path = f"C:\\services\\app-{i}\\core.py"
            source = f"All documentation updated. Broken file at {path}. Meeting later."
            gold = [(path, "path", 0)]
        # Each case is reproducible but varied and limited in budget.
        budget = max(1, int(len(source) * rng.choice((0.25, 0.42, 0.68))))
        cases.append({
            "id": f"{family}-{i:03d}", "family": family,
            "source": source, "query": rng.choice((
                "meeting update", "settings incident", "backup logs",
                "timeout errors", "deployment", "오후 회의")),
            "budget_characters": budget, "gold": gold,
            "group_all": family in ("config_revision", "ticket_history"),
        })
    return cases


def evaluate_cases(cases: list[dict]) -> dict:
    rows = []
    ids = set()
    for case in cases:
        if case["id"] in ids:
            raise ValueError("duplicate case id")
        ids.add(case["id"])
        source = case["source"]
        gold = tuple(
            locate_gold(source, literal, label, occurrence=occurrence)
            for literal, label, occurrence in case["gold"]
        )
        obligations = tuple(
            EvidenceObligation(g.label, g.start, g.end, source[g.start:g.end])
            for g in gold
        )
        bundles = (
            (EvidenceBundle("linked-history", tuple(x.label for x in gold)),)
            if case["group_all"] else ()
        )
        for mode in ("plain", "review", "trusted_contract"):
            try:
                if mode == "plain":
                    doc = compress_document(source, case["budget_characters"],
                                            query=case["query"], token_counter=len)
                else:
                    result = compress_with_integrity(
                        source, case["budget_characters"], query=case["query"],
                        obligations=obligations if mode == "trusted_contract" else (),
                        bundles=bundles if mode == "trusted_contract" else (),
                        risk_mode="audit" if mode == "trusted_contract" else "review",
                        token_counter=len,
                    )
                    doc = result.document
                audit = audit_quality(source, doc, gold)
                assert audit.source_aligned and doc.output_tokens <= case["budget_characters"]
                if mode == "trusted_contract":
                    assert audit.retained_count == len(gold)
                rows.append({
                    "id": case["id"], "mode": mode, "status": "ok",
                    "retained": audit.retained_count, "total": len(gold),
                })
            except BudgetExceeded:
                rows.append({
                    "id": case["id"], "mode": mode, "status": "fail_closed",
                    "retained": 0, "total": len(gold),
                })
    summary = {}
    for mode in ("plain", "review", "trusted_contract"):
        subset = [row for row in rows if row["mode"] == mode]
        completed = [row for row in subset if row["status"] == "ok"]
        success_gold = sum(row["total"] for row in completed)
        hits = sum(row["retained"] for row in completed)
        total_gold = sum(row["total"] for row in subset)
        summary[mode] = {
            "cases": len(subset),
            "successful": len(completed),
            "refused": len(subset) - len(completed),
            "gold_spans": total_gold,
            "retained_on_success": hits,
            "gold_on_success": success_gold,
            "recall_on_success": hits / success_gold if success_gold else None,
            "effective_recall_with_refusals": hits / total_gold if total_gold else None,
        }
    return {"summary": summary, "sample_count": len(cases),
            "families": sorted(set(case["family"] for case in cases))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=150)
    p.add_argument("--seed", type=int, default=8042)
    p.add_argument("--output", type=Path, default=None,
                   help="optional JSON summary path; no source text is written")
    a = p.parse_args()
    report = evaluate_cases(generate_cases(a.count, a.seed))
    data = json.dumps(report, ensure_ascii=False, indent=2)
    print(data)
    print("WARNING: seeded synthetic, oracle-assisted caller contract. "
          "NOT human-held-out, automatic fact discovery or LLM task accuracy.")
    if a.output:
        a.output.write_text(data + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
