import pytest

from devseat_context.evaluation import evaluate_cases, load_cases


def test_benchmark_cases_run_with_independent_labels():
    cases = load_cases('benchmarks/fixtures.json')
    result = evaluate_cases(cases)
    assert len(result) == 2 * len(cases)
    assert all(0 <= s.literal_recall <= 1 for s in result)
    assert all(0 <= s.output_characters <= s.input_characters for s in result)


def test_missing_ground_truth_rejected():
    with pytest.raises(ValueError, match='ground-truth'):
        evaluate_cases([{'id': 'x', 'text': 'A', 'query': 'A', 'budget_characters': 1,
                         'expected_literals': ['ABSENT']}])


def test_no_data_leak_in_evaluation():
    case = {'id': 'x', 'text': 'blue sky. red door.', 'query': 'blue',
            'budget_characters': 9, 'expected_literals': ['red door']}
    results = evaluate_cases([case])
    assert any(s.literal_recall < 1.0 for s in results)
