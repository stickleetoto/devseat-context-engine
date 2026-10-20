"""Tests for optional tokenizer adapters and rendered-budget enforcement."""

import types

import pytest

import devseat_context.tokenizers as adapters
from devseat_context import BudgetExceeded
from devseat_context.tokenizers import (
    FinalBudget, RenderedTokenCounter, TiktokenCounter,
    TokenizerConfigurationError, TokenizerUnavailable, verify_final_budget,
)


class FakeEncoding:
    name = "fake_base"

    def encode(self, text, *, disallowed_special):
        assert disallowed_special == ()
        return list(text.encode("utf-8"))


def fake_tiktoken():
    return types.SimpleNamespace(
        get_encoding=lambda name: FakeEncoding() if name == "fake_base" else (_ for _ in ()).throw(KeyError(name)),
        encoding_for_model=lambda name: FakeEncoding() if name == "fake-v1" else (_ for _ in ()).throw(KeyError(name)),
    )


def test_explicit_encoding_counts_korean_utf8(monkeypatch):
    monkeypatch.setattr(adapters, "import_module", lambda _: fake_tiktoken())
    counter = TiktokenCounter.for_encoding("fake_base")
    assert counter("한국어") == len("한국어".encode("utf-8"))
    assert counter.encoding_name == "fake_base"


def test_model_mapping_without_silent_fallback(monkeypatch):
    monkeypatch.setattr(adapters, "import_module", lambda _: fake_tiktoken())
    assert TiktokenCounter.for_model("fake-v1")("abc") == 3
    with pytest.raises(TokenizerConfigurationError, match="mapping unknown"):
        TiktokenCounter.for_model("unknown-latest")


def test_unknown_encoding_rejected(monkeypatch):
    monkeypatch.setattr(adapters, "import_module", lambda _: fake_tiktoken())
    with pytest.raises(TokenizerConfigurationError, match="unknown encoding"):
        TiktokenCounter.for_encoding("not-real")


def test_absent_optional_dependency_has_actionable_error(monkeypatch):
    def unavailable(_):
        raise ImportError("not installed")

    monkeypatch.setattr(adapters, "import_module", unavailable)
    with pytest.raises(TokenizerUnavailable, match="tokenizers"):
        TiktokenCounter.for_encoding("cl100k_base")


@pytest.mark.parametrize("name", ["", None, 5])
def test_invalid_tokenizer_names(name):
    with pytest.raises(TokenizerConfigurationError):
        TiktokenCounter.for_encoding(name)
    with pytest.raises(TokenizerConfigurationError):
        TiktokenCounter.for_model(name)


def test_special_looking_text_is_not_treated_as_trusted_special_token(monkeypatch):
    monkeypatch.setattr(adapters, "import_module", lambda _: fake_tiktoken())
    counter = TiktokenCounter.for_encoding("fake_base")
    assert counter("<|im_start|>") > 0


def test_rendered_counter_counts_entire_payload():
    counter = RenderedTokenCounter(len, lambda s: "<user>" + s + "</user>")
    assert counter("hello") == len("<user>hello</user>")
    assert counter("") == len("<user></user>")


def test_renderer_type_and_counter_value_checked():
    with pytest.raises(TypeError):
        RenderedTokenCounter("invalid", lambda s: s)
    with pytest.raises(TypeError):
        RenderedTokenCounter(len, lambda s: None)("abc")
    for bad in (lambda _: -1, lambda _: 1.2, lambda _: True):
        with pytest.raises(ValueError):
            RenderedTokenCounter(bad, lambda s: s)("hello")


def test_final_budget_verification_and_reservations():
    budget = FinalBudget(100, output_reservation=20, framing_overhead=5, safety_margin=5)
    assert budget.input_limit == 70
    verified = verify_final_budget("hello", len, budget)
    assert verified.counted_input_tokens == 5
    assert verified.remaining_tokens == 65
    with pytest.raises(BudgetExceeded):
        verify_final_budget("x" * 71, len, budget)


def test_budget_verifies_rendered_content_not_raw_only():
    final = FinalBudget(20, output_reservation=3)
    wrapped = RenderedTokenCounter(len, lambda s: '{"role":"user","text":' + s + '}')
    with pytest.raises(BudgetExceeded):
        verify_final_budget("x", wrapped, final)


@pytest.mark.parametrize("kwargs", [
    {"context_window": -1}, {"context_window": True},
    {"context_window": 20, "output_reservation": 21},
    {"context_window": 20, "framing_overhead": -1},
])
def test_invalid_budget_reservations(kwargs):
    with pytest.raises((ValueError, BudgetExceeded)):
        _ = FinalBudget(**kwargs).input_limit


def test_final_budget_bad_counter_and_invalid_inputs():
    with pytest.raises(ValueError):
        verify_final_budget("text", lambda _: True, FinalBudget(200))
    with pytest.raises(TypeError):
        verify_final_budget(123, len, FinalBudget(200))
    with pytest.raises(TypeError):
        verify_final_budget("hello", "bad", FinalBudget(200))
    with pytest.raises(TypeError):
        verify_final_budget("hello", len, "bad")


def test_optional_real_tiktoken_if_present():
    """CI extras install makes this an actual integration test; local core stays optional."""
    tiktoken = pytest.importorskip("tiktoken")
    counter = TiktokenCounter.for_encoding("cl100k_base")
    text = "hello world 안녕하세요! <|im_start|>"
    assert counter(text) == len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))
