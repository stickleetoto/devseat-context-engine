"""Performance profiler smoke testing without thresholds on wall-clock time."""

import pytest

from devseat_context.perf_profile import profile_segments


def test_profiler_reports_safe_summary_and_functions():
    text = profile_segments(100, 10)
    assert "Synthetic fragments=100" in text
    assert "compress" in text
    assert "authentication failed E401" not in text


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_profiler_invalid_workload_rejected(count):
    with pytest.raises(ValueError):
        profile_segments(count)


@pytest.mark.parametrize("top", [0, -2, True])
def test_profiler_invalid_display_count_rejected(top):
    with pytest.raises(ValueError):
        profile_segments(10, top)
