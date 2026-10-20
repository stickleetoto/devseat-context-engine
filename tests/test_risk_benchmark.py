import json
from pathlib import Path

import pytest

from devseat_context.risk_benchmark import compare_risk_cases


def test_quality_cases_compare_and_report_refusals():
    cases = json.loads(Path("benchmarks/quality_cases.json").read_text(encoding="utf-8"))
    data = compare_risk_cases(cases)
    assert len(data["records"]) == 2 * len(cases)
    for metric in data["summary"].values():
        assert metric["successful_samples"] + metric["fail_closed_samples"] == len(cases)
        assert 0 <= metric["retention_including_failures"] <= 1.0
    assert data["summary"]["v0.7_risk_require"]["fail_closed_samples"] >= 1


def test_gold_never_becomes_required_protection():
    case = {
        "id": "intentional-loss", "source": "Old notes. Ignored gold E713.",
        "query": "old", "budget_characters": 10,
        "evidence": [{"label": "gold", "literal": "E713"}]
    }
    data = compare_risk_cases([case])
    assert all(r["retained"] == 0 for r in data["records"] if r["status"] == "ok")


def test_absent_gold_fails_fixture_validation():
    with pytest.raises(ValueError):
        compare_risk_cases([{
            "id": "bad", "source": "Test", "query": "Test",
            "budget_characters": 5,
            "evidence": [{"label": "bad", "literal": "UNSEEN"}]
        }])
