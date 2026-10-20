"""CPU-only synthetic workload profiler with function-level cumulative timing.

Usage:
  python -m devseat_context.perf_profile --segments 10000 --top 16

This is diagnostic instrumentation, not a latency SLA. cProfile overhead is
significant. No application text, secrets or external API calls are required.
"""

import argparse
import cProfile
import io
import pstats
import time

from .engine import ContextEngine, Segment


def profile_segments(segments: int, top: int = 15) -> str:
    """Profile one realistic-ish repetitive synthetic context selection."""
    if type(segments) is not int or segments <= 0:
        raise ValueError("segments must be a positive integer")
    if type(top) is not int or top <= 0:
        raise ValueError("top must be a positive integer")
    items = [
        Segment(str(i), f"log {i:05d}: authentication failed E401.",
                priority=i % 5, protected=i == 0)
        for i in range(segments)
    ]
    budget = len("\n".join(s.text for s in items)) // 5
    engine = ContextEngine(len, separator="\n")

    prof = cProfile.Profile()
    started = time.perf_counter()
    prof.enable()
    try:
        result = engine.compress(items, budget, query="authentication failed")
    finally:
        prof.disable()
    elapsed_ms = (time.perf_counter() - started) * 1000
    stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stream)
    stats.strip_dirs().sort_stats("cumulative").print_stats(top)
    assert result.output_tokens <= budget
    return (
        f"Synthetic fragments={segments}; selected={len(result.selected_ids)}; "
        f"measured_ms_with_profiler={elapsed_ms:.3f}\n"
        "Note: cProfile changes timing; compare function cumulative shares "
        "not these numbers to the unprofiled perf_benchmark results.\n"
        + stream.getvalue()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=10000)
    parser.add_argument("--top", type=int, default=15)
    opts = parser.parse_args()
    print(profile_segments(opts.segments, opts.top))


if __name__ == "__main__":
    main()
