"""Train/eval split guardrails and honest metrics for the CPU learning module."""
import json
from pathlib import Path

import pytest

from devseat_context.ranker_benchmark import evaluate_dev, read_examples


def fixture_data():
    return Path(__file__).resolve().parents[1] / "benchmarks" / "ranker_examples.json"


def test_36_separate_groups_and_deterministic_train():
    train, dev = read_examples(fixture_data())
    assert len(train) == 48 and len(dev) == 24
    assert not set(e.group for e in train) & set(e.group for e in dev)
    result, model = evaluate_dev(train, dev)
    again, model2 = evaluate_dev(train, dev)
    assert result == again and model == model2
    assert result["metrics"]["lexical"]["evaluated_groups"] == 12
    assert result["metrics"]["trained_cpu"]["evaluated_groups"] == 12
    assert 0 <= result["metrics"]["trained_cpu"]["top1_accuracy"] <= 1
    assert 0 <= result["metrics"]["lexical"]["top1_accuracy"] <= 1


def test_dev_is_not_used_to_train():
    train, dev = read_examples(fixture_data())
    orig, original_model = evaluate_dev(train, dev)
    mutated = [type(e)(e.query, e.text, not e.relevant, e.group) for e in dev]
    changed, same_model = evaluate_dev(train, mutated)
    assert original_model == same_model
    assert orig["training_pairs"] == changed["training_pairs"]


def test_data_rejects_group_leakage(tmp_path):
    obj = json.loads(fixture_data().read_text("utf8"))
    dev = next(x for x in obj["examples"] if x["split"] == "dev")
    dev["group"] = obj["examples"][0]["group"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(obj), encoding="utf8")
    with pytest.raises(ValueError):
        read_examples(p)


def test_data_rejects_pair_leakage(tmp_path):
    obj = json.loads(fixture_data().read_text("utf8"))
    row = obj["examples"][-1]
    row["query"] = obj["examples"][0]["query"]
    row["text"] = obj["examples"][0]["text"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(obj), encoding="utf8")
    with pytest.raises(ValueError):
        read_examples(p)


def test_bad_schema_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"schema": 9, "examples": []}', encoding="utf8")
    with pytest.raises(ValueError):
        read_examples(p)


def test_missing_training_split_rejected(tmp_path):
    d = json.loads(fixture_data().read_text("utf8"))
    d["examples"] = [e for e in d["examples"] if e["split"] == "dev"]
    p = tmp_path / "no-train.json"
    p.write_text(json.dumps(d), encoding="utf8")
    with pytest.raises(ValueError):
        read_examples(p)
