"""Measure CPU latency p50/p95 and tracemalloc allocations.

Usage: python -m devseat_context.perf_benchmark --sizes 1000 10000 50000
Only builtin len is additive. Generic lambda len stands in for slower exact
rendering, not for a model tokenizer. Measurements depend on hardware.
"""
import argparse
import json
import math
import platform
import statistics
import sys
import time
import tracemalloc

from .engine import ContextEngine, Segment


def percentile(values: list[float], percent: float) -> float:
    if not values or not 0 < percent <= 100:
        raise ValueError("invalid percentile")
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * percent / 100) - 1]


def benchmark_case(n: int, repetitions: int, legacy: bool) -> dict:
    if n < 1 or repetitions < 1:
        raise ValueError("n/repetitions must be positive")
    seq = [Segment(str(i), f"log {i:05d}: authentication failed E401.",
                   priority=i % 5, protected=(i == 0)) for i in range(n)]
    budget = len("\n".join(s.text for s in seq)) // 5
    result = {}
    for mode in (("fast_char", "generic_lambda") if legacy else ("fast_char",)):
        counter = len if mode == "fast_char" else lambda text: len(text)
        worker = ContextEngine(counter, separator="\n")
        timings = []
        memory = []
        signature = None
        for _ in range(repetitions):
            tracemalloc.start()
            start = time.perf_counter_ns()
            output = worker.compress(seq, budget, query="authentication failed")
            timings.append((time.perf_counter_ns() - start) / 1_000_000)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory.append(peak / 1048576)
            assert output.output_tokens <= budget
            current = (output.text, output.selected_ids, output.dropped_ids)
            if signature is not None:
                assert signature == current
            signature = current
        result[mode] = {
            "p50_ms": round(percentile(timings, 50), 3),
            "p95_ms": round(percentile(timings, 95), 3),
            "median_peak_alloc_mib": round(statistics.median(memory), 3),
            "selected_segments": len(signature[1]),
        }
    if legacy:
        assert result["fast_char"]["selected_segments"] == result["generic_lambda"]["selected_segments"]
    return {"segments": n, "budget_characters": budget, "modes": result}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000])
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--legacy-max", type=int, default=1000)
    opts = p.parse_args()
    if opts.repetitions < 1 or opts.legacy_max < 0 or any(n < 1 for n in opts.sizes):
        p.error("sizes/repetitions must be positive, legacy-max >=0")
    print(json.dumps({
        "python": sys.version.split()[0], "platform": platform.platform(),
        "counter": "builtin len vs generic lambda len (character budgets)",
        "repetitions": opts.repetitions,
        "results": [
            benchmark_case(n, opts.repetitions, legacy=n <= opts.legacy_max)
            for n in opts.sizes
        ],
    }, indent=2))
    print("CPU and tracemalloc overhead vary. No model-token or production p95 guarantee.")


if __name__ == "__main__":
    main()
