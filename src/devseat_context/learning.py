"""Optional, CPU-only trainable importance scorer for query/document pairs.

Training runs offline on caller-labeled data. Inference cannot declare any
source trusted, mandatory, corrected, or safe to omit. Default extractive
ranking remains unchanged until a caller opts into a validated model.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Sequence

from .relevance import prepare_lexical_scorer

_SCHEMA = 1
_MAX_MODEL_BYTES = 65_536
_MAX_QUERY_CHARS = 2_048
_MAX_TEXT_CHARS = 20_000
_FEATURE_NAMES = (
    "lexical_overlap", "error", "datetime", "path", "assignment",
    "negation", "ticket", "error_intent", "time_intent", "path_intent",
    "config_intent", "policy_intent", "ticket_intent", "length_fraction",
)
_ERROR = re.compile(r"\b(?:E\d{3,5}|ERR[-_]?\d{2,6}|ERROR[-_]?\d{2,6}|exception|traceback)\b", re.I)
_DATETIME = re.compile(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b")
_PATH = re.compile(r"(?:[A-Za-z]:\\|(?:/[\w.-]+){2,})")
_ASSIGN = re.compile(r"\b[A-Za-z_][A-Za-z_0-9]*\s*=\s*[^ \t\n.;,]+")
_NEGATION = re.compile(r"\b(?:never|cannot|can't|don't|not|forbidden|prohibited)\b|(?:안\s*됩니다|해서는\s*안|금지|불가|않습니다|하지\s*말)", re.I)
_TICKET = re.compile(r"\b(?:INC|TICKET|ISSUE)[-_ ]?\d+\b", re.I)
_INTENTS = (
    ("error", "exception", "fail", "원인", "오류", "에러", "장애", "실패"),
    ("when", "date", "time", "날짜", "언제", "시각", "발생"),
    ("where", "file", "path", "stack", "위치", "파일", "경로", "어디"),
    ("config", "timeout", "setting", "change", "설정", "변경", "수정", "값"),
    ("never", "allowed", "policy", "prohibit", "금지", "허용", "삭제", "정책"),
    ("ticket", "incident", "issue", "티켓", "이슈", "사건", "아이디"),
)


class RankerModelError(ValueError):
    """Invalid, incompatible or modified (not authenticated) model data."""


@dataclass(frozen=True, slots=True)
class TrainingExample:
    query: str
    text: str
    relevant: bool
    group: str


@dataclass(frozen=True, slots=True)
class LinearCPURanker:
    """Small frozen logistic model. score() returns [0,1] for ranking only."""

    weights: tuple[float, ...]
    bias: float
    training_examples: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.weights, tuple)
                or len(self.weights) != len(_FEATURE_NAMES)
                or any(type(w) not in (float, int) or not math.isfinite(w)
                       or abs(w) > 50 for w in self.weights)
                or type(self.bias) not in (float, int)
                or not math.isfinite(self.bias) or abs(self.bias) > 50
                or type(self.training_examples) is not int
                or not 0 <= self.training_examples <= 1_000_000):
            raise RankerModelError("model has invalid shape or unbounded weights")

    def prepare(self, query: str):
        features = prepare_ranker_features(query)
        weights, bias = self.weights, self.bias

        def score(text: str) -> float:
            values = features(text)
            logit = bias + sum(w * x for w, x in zip(weights, values))
            return _sigmoid(logit)

        return score

    def predict(self, query: str, text: str) -> float:
        return self.prepare(query)(text)

    def to_json(self) -> str:
        payload = {
            "schema": _SCHEMA, "features": list(_FEATURE_NAMES),
            "weights": list(self.weights), "bias": self.bias,
            "training_examples": self.training_examples,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
        checksum = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return json.dumps({"payload": payload, "sha256": checksum},
                          ensure_ascii=False, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, data: str) -> "LinearCPURanker":
        if not isinstance(data, str) or len(data.encode("utf-8")) > _MAX_MODEL_BYTES:
            raise RankerModelError("model JSON exceeds size limit or is not text")
        try:
            obj = json.loads(data)
            if not isinstance(obj, dict) or set(obj) != {"payload", "sha256"}:
                raise RankerModelError("invalid model envelope")
            payload, sig = obj["payload"], obj["sha256"]
            if not isinstance(payload, dict) or set(payload) != {
                "schema", "features", "weights", "bias", "training_examples"
            } or not isinstance(sig, str):
                raise RankerModelError("invalid model payload")
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False)
            if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != sig:
                raise RankerModelError("model checksum mismatch")
            if (type(payload["schema"]) is not int or payload["schema"] != _SCHEMA
                    or payload["features"] != list(_FEATURE_NAMES)
                    or not isinstance(payload["weights"], list)):
                raise RankerModelError("incompatible model schema")
            return cls(tuple(payload["weights"]), payload["bias"],
                       payload["training_examples"])
        except (TypeError, ValueError, KeyError) as exc:
            raise RankerModelError("invalid model JSON") from exc

    def save(self, path: str | Path) -> None:
        """Caller must select a trusted local destination; no network access."""
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinearCPURanker":
        path = Path(path)
        if path.stat().st_size > _MAX_MODEL_BYTES:
            raise RankerModelError("model file exceeds size limit")
        return cls.from_json(path.read_text(encoding="utf-8"))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def prepare_ranker_features(query: str):
    """Return a per-request extractor; no persistent or cross-tenant caching."""
    if not isinstance(query, str) or len(query) > _MAX_QUERY_CHARS:
        raise ValueError("query must be text within configured length")
    lexical = prepare_lexical_scorer(query)
    q = query.casefold()
    intents = tuple(any(needle in q for needle in category) for category in _INTENTS)

    def extract(text: str) -> tuple[float, ...]:
        if not isinstance(text, str) or len(text) > _MAX_TEXT_CHARS:
            raise ValueError("segment text must be within feature length limit")
        # Cheap NECESSARY prefilters avoid six regex passes for most short
        # records. They must never exclude text that the corresponding
        # regex could match: otherwise learned scores would silently drift.
        folded = text.casefold()
        negation_hint = any(piece in folded for piece in (
            "never", "cannot", "can't", "don't", "not", "forbidden",
            "prohibited", "안", "해서는", "금지", "불가", "않", "하지",
        ))
        present = (
            float("e" in folded and bool(_ERROR.search(text))),
            float(any(c in text for c in "-/:") and bool(_DATETIME.search(text))),
            float(("\\" in text or "/" in text) and bool(_PATH.search(text))),
            float("=" in text and bool(_ASSIGN.search(text))),
            float(negation_hint and bool(_NEGATION.search(text))),
            float(any(tag in folded for tag in ("inc", "ticket", "issue"))
                  and bool(_TICKET.search(text))),
        )
        return (
            lexical(text), *present,
            *(float(important) * indicator
              for important, indicator in zip(intents, present)),
            min(len(text) / 400.0, 1.0),
        )

    return extract


def train_cpu_ranker(
    examples: Sequence[TrainingExample],
    *,
    epochs: int = 60,
    learning_rate: float = 0.12,
    l2: float = 0.002,
    seed: int = 17,
) -> LinearCPURanker:
    """Deterministic SGD logistic loss on explicitly labeled pairs.

    Caller owns privacy, provenance, group splits, data quality and held-out
    scoring. No training examples, labels, or query contents enter artifacts.
    """
    seq = tuple(examples)
    if (not 2 <= len(seq) <= 100_000 or any(
        not isinstance(e, TrainingExample)
        or not isinstance(e.query, str) or not isinstance(e.text, str)
        or type(e.relevant) is not bool
        or not isinstance(e.group, str) or not e.group or len(e.group) > 128
        for e in seq
    )):
        raise ValueError("need 2..100000 valid labeled training examples")
    if not any(e.relevant for e in seq) or all(e.relevant for e in seq):
        raise ValueError("both positive and negative examples are required")
    if (type(epochs) is not int or not 1 <= epochs <= 500
            or type(learning_rate) not in (float, int)
            or not math.isfinite(learning_rate) or not 0 < learning_rate <= 1
            or type(l2) not in (float, int)
            or not math.isfinite(l2) or not 0 <= l2 <= 0.5
            or type(seed) is not int):
        raise ValueError("invalid bounded training hyperparameters")
    prepared = {}
    rows = []
    for e in seq:
        if e.query not in prepared:
            prepared[e.query] = prepare_ranker_features(e.query)
        rows.append((prepared[e.query](e.text), float(e.relevant)))
    # Bound dataset memory before training; model contains no raw examples.
    weights = [0.0] * len(_FEATURE_NAMES)
    bias = 0.0
    order = list(range(len(rows)))
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(order)
        for i in order:
            features, target = rows[i]
            pred = _sigmoid(bias + sum(w * x for w, x in zip(weights, features)))
            error = pred - target
            for j, feature in enumerate(features):
                weights[j] -= learning_rate * (error * feature + l2 * weights[j])
            bias -= learning_rate * error
        # Avoid permitting a malformed or overtrained model into selection.
        if max(abs(w) for w in weights) > 50 or abs(bias) > 50:
            raise RankerModelError("training diverged")
    return LinearCPURanker(tuple(weights), bias, len(seq))
