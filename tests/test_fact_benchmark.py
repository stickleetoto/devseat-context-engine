import json
from pathlib import Path

import pytest

from devseat_context.fact_benchmark import evaluate_fact_cases


def test_public_fact_benchmark_runs_and_is_budget_safe():
    cases = json.loads(Path("benchmarks/fact_fixtures.json").read_text(encoding="utf-8"))
    rows = evaluate_fact_cases(cases)
    assert len(rows) == len(cases) * 2
    assert {r["mode"] for r in rows} == {"v0.3_span", "v0.4_fact_strict"}
    assert all(r["output_chars"] <= r["budget_characters"] for r in rows)
    assert any(r["status"] == "fail_closed" for r in rows)
    assert any(r["conflict_candidates"] > 0 for r in rows)


def test_benchmark_does_not_feed_gold_to_engine():
    case = {
        "id": "gold_is_not_a_requirement", "text": "Old note. Ticket E401 happened.",
        "query": "old", "expected_literals": ["E401"], "budget_characters": 9,
    }
    result = evaluate_fact_cases([case])
    assert all(r["literal_hits"] == 0 for r in result if r["status"] == "ok")


def test_benchmark_bad_labels_rejected():
    with pytest.raises(ValueError):
        evaluate_fact_cases([{
            "id": "bad", "text": "hello", "query": "hello", "expected_literals": ["missing"],
            "budget_characters": 5,
        }])
