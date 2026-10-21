"""Profile optional CPU-ranker overhead without loading an LLM or GPU.

python -m devseat_context.ranker_perf --sizes 1000 10000 50000
This is an uninstrumented synthetic timing comparison, NOT answer quality,
provider-token accuracy, or a production SLA.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

from .engine import ContextEngine, Segment
from .learning import LinearCPURanker
from .perf_benchmark import percentile
from .ranker_benchmark import read_examples
from .learning import train_cpu_ranker


def run_cpu_comparison(
    model: LinearCPURanker, sizes: tuple[int, ...], repetitions: int = 5,
) -> dict:
    if not isinstance(model, LinearCPURanker):
        raise TypeError("model must be a verified LinearCPURanker")
    if (not isinstance(sizes, tuple) or any(type(n) is not int or n < 1 or n > 100_000
                                          for n in sizes)
            or type(repetitions) is not int or not 1 <= repetitions <= 50):
        raise ValueError("invalid benchmark sizes or repetitions")
    records = []
    for n in sizes:
        items = [
            Segment(str(i), f"Request {i:05d}: authentication failed with E{401 + i % 300}.",
                    priority=i % 3, protected=i == 0)
            for i in range(n)
        ]
        budget = len("\n".join(s.text for s in items)) // 5
        for mode in ("lexical_default", "learned_cpu"):
            engine = ContextEngine(len, "\n")
            times = []
            reference = None
            # Warm once before timed runs.
            engine.compress(items, budget, query="why did authentication fail",
                            ranker=model if mode == "learned_cpu" else None)
            for _ in range(repetitions):
                start = time.perf_counter_ns()
                output = engine.compress(
                    items, budget, query="why did authentication fail",
                    ranker=model if mode == "learned_cpu" else None,
                )
                times.append((time.perf_counter_ns() - start) / 1_000_000)
                assert output.output_tokens <= budget and "0" in output.selected_ids
                signature = (output.text, output.selected_ids)
                if reference is not None:
                    assert signature == reference
                reference = signature
            records.append({
                "segments": n, "mode": mode, "budget_characters": budget,
                "p50_ms": round(percentile(times, 50), 3),
                "p95_ms": round(percentile(times, 95), 3),
                "selected_count": len(reference[1]),
            })
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repetitions": repetitions,
        "counter": "builtin len: character budget (NOT LLM BPE tokens)",
        "records": records,
        "warning": "Synthetic CPU timing; runner power, clocks and input mix vary.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()
    if args.model is None:
        fixture = (Path(__file__).resolve().parents[2] /
                   "benchmarks" / "ranker_examples.json")
        train, _ = read_examples(fixture)
        model = train_cpu_ranker(train, epochs=80, seed=17)
    else:
        model = LinearCPURanker.load(args.model)
    result = run_cpu_comparison(model, tuple(args.sizes), args.repetitions)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
