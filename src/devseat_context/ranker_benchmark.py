"""Reproducible offline CPU-ranker development evaluation; no model API calls.

python -m devseat_context.ranker_benchmark
python -m devseat_context.ranker_benchmark --model-out local-ranker.json

Dev groups are distinct from training groups, but examples are synthetic and
were authored with the features in mind. This is NOT independent held-out
semantic-quality evidence or permission to auto-promote trained weights.
"""
import argparse
import json
from pathlib import Path

from .engine import ContextEngine, Segment
from .learning import TrainingExample, train_cpu_ranker

_MAX_DATASET_BYTES = 2_000_000


def read_examples(path: str | Path) -> tuple[list[TrainingExample], list[TrainingExample]]:
    file = Path(path)
    if file.stat().st_size > _MAX_DATASET_BYTES:
        raise ValueError("dataset exceeds allowed size")
    data = json.loads(file.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("schema") != 1
            or not isinstance(data.get("examples"), list)):
        raise ValueError("invalid dataset schema")
    raw = data["examples"]
    if len(raw) < 8 or len(raw) > 10000:
        raise ValueError("dataset must contain 8..10000 labeled rows")
    train, dev = [], []
    by_group = {}
    seen_pairs = set()
    for row in raw:
        if not isinstance(row, dict) or set(row) != {
            "group", "split", "query", "text", "relevant"
        }:
            raise ValueError("invalid example record")
        split = row["split"]
        if split not in ("train", "dev"):
            raise ValueError("split must be train or dev")
        example = TrainingExample(row["query"], row["text"], row["relevant"], row["group"])
        if (not all(isinstance(v, str) and v for v in
                    (example.query, example.text, example.group))
                or type(example.relevant) is not bool):
            raise ValueError("example fields invalid")
        previous = by_group.setdefault(example.group, (split, example.query))
        if previous != (split, example.query):
            raise ValueError("group leaks across splits or queries")
        pair = (example.query, example.text)
        if pair in seen_pairs:
            raise ValueError("duplicate query/text pairs across dataset")
        seen_pairs.add(pair)
        (train if split == "train" else dev).append(example)
    if not train or not dev:
        raise ValueError("nonempty train and dev required")
    for split_data in (train, dev):
        groups = {}
        for example in split_data:
            groups.setdefault(example.group, []).append(example)
        if any(len(group) != 2 or sum(row.relevant for row in group) != 1
               for group in groups.values()):
            raise ValueError("each group needs two candidates with one positive")
    return train, dev


def evaluate_dev(
    train: list[TrainingExample],
    dev: list[TrainingExample],
    *,
    epochs: int = 80,
    seed: int = 17,
) -> tuple[dict, object]:
    model = train_cpu_ranker(train, epochs=epochs, seed=seed)
    grouped = {}
    for example in dev:
        grouped.setdefault(example.group, []).append(example)
    results = {}
    for mode in ("lexical", "trained_cpu"):
        successes = 0
        picked = []
        for group, rows in grouped.items():
            # A budget equal to the largest single item admits exactly one
            # complete segment, and the protected-policy baseline remains on.
            budget = max(len(item.text) for item in rows)
            segs = [Segment(f"{group}/{i}", item.text) for i, item in enumerate(rows)]
            output = ContextEngine(len, separator="|").compress(
                segs, budget, query=rows[0].query,
                ranker=model if mode == "trained_cpu" else None,
            )
            if len(output.selected_ids) != 1:
                raise AssertionError("paired ranking benchmark must select one candidate")
            i = int(output.selected_ids[0].rsplit("/", 1)[1])
            successes += int(rows[i].relevant)
            picked.append({"group": group, "picked_relevant": rows[i].relevant})
        results[mode] = {
            "evaluated_groups": len(grouped),
            "correct_top1": successes,
            "top1_accuracy": successes / len(grouped),
        }
    return {
        "training_pairs": len(train),
        "dev_pairs": len(dev),
        "split": "group-disjoint synthetic development data",
        "epoch_count": epochs,
        "metrics": results,
        "interpretation": (
            "Scores are only candidate-selection correctness on authored examples; "
            "NOT semantic answer accuracy, production recall or an independent "
            "held-out evaluation."
        ),
    }, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path,
        default=Path(__file__).resolve().parents[2] / "benchmarks" / "ranker_examples.json",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model-out", type=Path, default=None)
    args = parser.parse_args()
    train, dev = read_examples(args.data)
    report, model = evaluate_dev(train, dev, epochs=args.epochs, seed=args.seed)
    print(json.dumps(report, indent=2))
    if args.model_out is not None:
        model.save(args.model_out)
        print("Model written to caller-selected path. Checksum is not a signature.")
    print("WARNING: synthetic development data, NOT automatic accuracy or LLM quality.")


if __name__ == "__main__":
    main()
