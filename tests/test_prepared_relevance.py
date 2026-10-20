"""Prepared lexical scoring must reproduce the original scoring without model calls."""
import random

import pytest

from devseat_context import ContextEngine, Segment, compress_document
from devseat_context import lexical_relevance
from devseat_context.relevance import prepare_lexical_scorer
import devseat_context.engine as engine_module
import devseat_context.document as document_module


@pytest.mark.parametrize("query", [
    "", " ", "auth failed", "인증 실패", "로그",
    "a_b", "E401", "인증실패", "한", "aß", "İstanbul",
    "한글 English", "mixed 123!", "_", "🚀", "Tokyo 東京 한국어",
])
def test_prepared_matches_original_for_edge_queries(query):
    score = prepare_lexical_scorer(query)
    examples = [
        "", "authentication failed E401", "한국어 인증 실패는 계속됩니다",
        "로그로그", "인증실패", "혼합 한국어 English example 123",
        "Türkçe İSTANBUL", "aß", "🚀", "_", "Tokyo 東京 한국어",
        "한", "한글", "가나", "가나다", "abc한글abc", "A_B", "\t\n",
    ]
    assert [score(t) for t in examples] == [lexical_relevance(query, t) for t in examples]


@pytest.mark.parametrize("seed", range(80))
def test_randomized_mixed_language_scores_identical(seed):
    r = random.Random(seed)
    alphabet = [
        "로그", "인증", "실패", "E401", "a_b", "abc", "İ", "ß",
        "汉字", "한", "한국어", "a", "9", "🚀", ".", " ", "_", "가나다"
    ]
    query = "".join(r.choice(alphabet) for _ in range(r.randrange(0, 20)))
    scorer = prepare_lexical_scorer(query)
    for _ in range(20):
        text = "".join(r.choice(alphabet) for _ in range(r.randrange(0, 35)))
        assert scorer(text) == lexical_relevance(query, text)


def test_no_global_cache_keeps_query_state_local():
    first = prepare_lexical_scorer("E401")
    second = prepare_lexical_scorer("E402")
    assert first("E401") == 1
    assert second("E401") == 0
    assert first("E401") == 1


def test_invalid_query_rejected():
    with pytest.raises(TypeError):
        prepare_lexical_scorer(None)


@pytest.mark.parametrize("seed", range(20))
def test_engine_results_identical_to_legacy_scorer(seed, monkeypatch):
    r = random.Random(seed)
    terms = ["인증", "failed", "A_B", "한글", "İstanbul", "🚀", "", "E401"]
    segments = [
        Segment(str(i), r.choice(terms) + " " + r.choice(terms),
                priority=r.randrange(-2, 4), protected=(r.randrange(20) == 0))
        for i in range(40)
    ]
    q = r.choice(terms)
    kwargs = {"query": q}
    budget = 250
    expected = ContextEngine(len, separator="|").compress(segments, budget, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(engine_module, "prepare_lexical_scorer",
                      lambda query: lambda text: lexical_relevance(query, text))
        legacy = ContextEngine(len, separator="|").compress(segments, budget, **kwargs)
    assert expected == legacy


@pytest.mark.parametrize("seed", range(20))
def test_document_results_identical_to_legacy_scorer(seed, monkeypatch):
    r = random.Random(seed)
    terms = ["인증 실패.", "Error E401.", "Do not delete.", "한글입니다.", "System ok."]
    source = " ".join(r.choice(terms) for _ in range(25))
    q = r.choice(["인증", "E401", "한글", "", "delete"])
    result = compress_document(source, 110, query=q, token_counter=len)
    with monkeypatch.context() as patch:
        patch.setattr(document_module, "prepare_lexical_scorer",
                      lambda query: lambda text: lexical_relevance(query, text))
        baseline = compress_document(source, 110, query=q, token_counter=len)
    assert result == baseline
