"""Synthetic accuracy experiment measures refusals and evidence separately."""
import pytest

from devseat_context.integrity_benchmark import evaluate_cases, generate_cases


def test_seeded_cases_reproduce_and_have_multiple_families():
    data = generate_cases()
    assert data == generate_cases()
    assert len(data) == 150
    assert len(set(c["id"] for c in data)) == 150
    assert len(set(c["family"] for c in data)) == 6


def test_no_undisclosed_reference_loss_on_contract_success():
    report = evaluate_cases(generate_cases(150))
    assert report["sample_count"] == 150
    assert set(report["summary"]) == {"plain", "review", "trusted_contract"}
    for summary in report["summary"].values():
        assert summary["successful"] + summary["refused"] == 150
        assert summary["gold_spans"] >= 150
        assert 0 <= summary["effective_recall_with_refusals"] <= 1
    assert report["risk_detector"]["annotated_spans_total"] == 225
    assert 0 < report["risk_detector"]["annotated_spans_detected"] < 225
    assert len(report["risk_detector"]["by_family"]) == 6
    strict = report["summary"]["trusted_contract"]
    assert strict["successful"] > 0
    assert strict["retained_on_success"] == strict["gold_on_success"]
    assert strict["recall_on_success"] == 1.0
    assert strict["refused"] > 0


def test_review_can_refuse_without_pretending_to_preserve_gold():
    report = evaluate_cases(generate_cases(30, seed=177))
    assert report["summary"]["review"]["refused"] > 0


def test_wrong_generator_params_fail():
    for n in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            generate_cases(n)


def test_duplicate_case_ids_fail():
    examples = generate_cases(2)
    examples[1]["id"] = examples[0]["id"]
    with pytest.raises(ValueError):
        evaluate_cases(examples)
