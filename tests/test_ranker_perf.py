import pytest

from devseat_context import TrainingExample, train_cpu_ranker
from devseat_context.ranker_perf import run_cpu_comparison


def example_ranker():
    return train_cpu_ranker([
        TrainingExample("why fail", "Request failed with E401", True, "a"),
        TrainingExample("why fail", "Meeting notes today", False, "a"),
    ], epochs=10)


def test_opt_in_perf_comparison_is_bounded():
    report = run_cpu_comparison(example_ranker(), (10, 20), 2)
    assert len(report["records"]) == 4
    for record in report["records"]:
        assert record["p50_ms"] >= 0
        assert record["p95_ms"] >= record["p50_ms"]
        assert record["selected_count"] >= 1


@pytest.mark.parametrize("sizes,repetitions", [
    ((0,), 3), ((100001,), 3), ((10.1,), 3),
    ([10], 3), ((10,), 0), ((10,), True),
])
def test_perf_rejects_unbounded_benchmarks(sizes, repetitions):
    with pytest.raises(ValueError):
        run_cpu_comparison(example_ranker(), sizes, repetitions)
