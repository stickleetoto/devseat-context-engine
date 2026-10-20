import json
from pathlib import Path

import pytest

from devseat_context.quality_benchmark import run_cases


def test_public_quality_suite_is_reproducible():
    data = json.loads(Path("benchmarks/quality_cases.json").read_text(encoding="utf-8"))
    results = run_cases(data)
    assert results["samples"] >= 12
    assert results["all_completed_source_aligned"]
    assert 0 <= results["micro_retention_including_failures"] <= 1
    assert results["successful_samples"] + results["fail_closed_samples"] == results["samples"]


def test_gold_labels_not_given_to_compressor():
    case = {
        "id": "a", "source": "Old blue notes. Error E401.",
        "query": "old blue", "budget_characters": 15,
        "evidence": [{"label": "late", "literal": "E401"}]
    }
    result = run_cases([case])
    assert result["exact_evidence_hits"] == 0


def test_malformed_or_missing_gold_fails():
    case = {
        "id": "a", "source": "old test", "query": "old",
        "budget_characters": 3,
        "evidence": [{"label": "missing", "literal": "E401"}]
    }
    with pytest.raises(ValueError):
        run_cases([case])


def test_fail_closed_is_disclosed_not_hidden():
    # Current extractor doesn't need to fail, but API must report failures
    # if future policies add mandatory evidence under tiny budgets.
    case = {
        "id": "a", "source": "Long data. Another sentence.", "query": "Another",
        "budget_characters": 1,
        "evidence": [{"label": "data", "literal": "Long data"}],
    }
    result = run_cases([case])
    assert "fail_closed_samples" in result
    assert result["all_labeled_spans"] == 1
