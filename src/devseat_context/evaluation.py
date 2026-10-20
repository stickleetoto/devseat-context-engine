"""Tiny transparent lexical-evidence benchmark, not an LLM performance score."""

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .document import compress_document
from .engine import ContextEngine, Segment


@dataclass(frozen=True, slots=True)
class CaseScore:
    case_id: str
    mode: str
    hit_count: int
    expected_count: int
    output_characters: int
    input_characters: int
    elapsed_ms: float

    @property
    def literal_recall(self) -> float:
        return self.hit_count / self.expected_count if self.expected_count else 1.0

    @property
    def character_reduction(self) -> float:
        return 1 - self.output_characters / self.input_characters if self.input_characters else 0.0


def _paragraph_baseline(text: str, query: str, budget: int) -> str:
    # v0.2 selects caller-supplied text segments, typically paragraph-level.
    paragraphs = [x.strip() for x in text.split('\n\n') if x.strip()]
    segs = [Segment(str(i), x) for i, x in enumerate(paragraphs)]
    return ContextEngine(token_counter=len, separator='\n').compress(segs, budget, query=query).text


def evaluate_cases(cases: list[dict[str, Any]]) -> tuple[CaseScore, ...]:
    """Evaluate annotated exact-literal recall WITHOUT giving labels to selector."""
    scores = []
    for case in cases:
        cid = case['id']
        text = case['text']
        query = case['query']
        expected = case['expected_literals']
        budget = case['budget_characters']
        if not isinstance(expected, list) or not expected or any(not isinstance(x, str) or not x for x in expected):
            raise ValueError(f'{cid}: expected_literals must contain strings')
        if any(x not in text for x in expected):
            raise ValueError(f'{cid}: ground-truth literal absent from input')
        for mode in ('v0.2_paragraph_baseline', 'v0.3_document_spans'):
            start = perf_counter()
            if mode == 'v0.2_paragraph_baseline':
                compressed = _paragraph_baseline(text, query, budget)
            else:
                compressed = compress_document(text, budget, query=query, token_counter=len).text
            duration_ms = (perf_counter() - start) * 1000
            if len(compressed) > budget:
                raise AssertionError(f'{cid}: budget exceeded')
            scores.append(CaseScore(cid, mode, sum(x in compressed for x in expected),
                                    len(expected), len(compressed), len(text), duration_ms))
    return tuple(scores)


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding='utf-8'))
