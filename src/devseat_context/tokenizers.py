"""Optional model-token encoding adapters and explicit final budget checks.

Text tokenizer counts do not include provider chat framing, hidden system
messages, tools, images or multimodal payloads. The caller must account for
those separately or render an accurate full payload using RenderedTokenCounter.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from .engine import BudgetExceeded, available_input_budget


class TokenizerUnavailable(ImportError):
    """Optional tokenizer dependency is not installed."""


class TokenizerConfigurationError(ValueError):
    """Model or encoding is not recognized; never use a silent fallback."""


@dataclass(frozen=True, slots=True)
class TiktokenCounter:
    """Exact for a *particular tiktoken text encoding*, NOT the full API request."""

    encoding_name: str
    _encoding: object

    @classmethod
    def for_encoding(cls, name: str) -> "TiktokenCounter":
        if not isinstance(name, str) or not name:
            raise TokenizerConfigurationError("encoding name must be nonempty")
        try:
            tiktoken = import_module("tiktoken")
        except ImportError as exc:
            raise TokenizerUnavailable(
                'Optional tokenizer missing: pip install "devseat-context-engine[tokenizers]"'
            ) from exc
        try:
            encoding = tiktoken.get_encoding(name)
        except KeyError as exc:
            raise TokenizerConfigurationError(f"unknown encoding: {name}") from exc
        return cls(encoding.name, encoding)

    @classmethod
    def for_model(cls, model: str) -> "TiktokenCounter":
        if not isinstance(model, str) or not model:
            raise TokenizerConfigurationError("model name must be nonempty")
        try:
            tiktoken = import_module("tiktoken")
        except ImportError as exc:
            raise TokenizerUnavailable(
                'Optional tokenizer missing: pip install "devseat-context-engine[tokenizers]"'
            ) from exc
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError as exc:
            raise TokenizerConfigurationError(
                f"model mapping unknown: {model}; specify a verified encoding explicitly"
            ) from exc
        return cls(encoding.name, encoding)

    def __call__(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("token count input must be text")
        # Special-looking source strings are ordinary untrusted content.
        return len(self._encoding.encode(text, disallowed_special=()))


@dataclass(frozen=True, slots=True)
class RenderedTokenCounter:
    """Count a caller-owned serialized envelope, not just the inner text.

    renderer must accurately reflect the real provider payload. No automatic
    wrapper-token assumptions are made by this library.
    """

    text_counter: Callable[[str], int]
    renderer: Callable[[str], str]

    def __post_init__(self) -> None:
        if not callable(self.text_counter) or not callable(self.renderer):
            raise TypeError("text_counter and renderer must be callable")

    def __call__(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        rendered = self.renderer(text)
        if not isinstance(rendered, str):
            raise TypeError("renderer must return a string")
        result = self.text_counter(rendered)
        if type(result) is not int or result < 0:
            raise ValueError("text_counter must return a nonnegative integer")
        return result


@dataclass(frozen=True, slots=True)
class FinalBudget:
    """Caller-declared total window and reservations; not provider discovery."""

    context_window: int
    output_reservation: int = 0
    framing_overhead: int = 0
    safety_margin: int = 0

    @property
    def input_limit(self) -> int:
        return available_input_budget(
            self.context_window,
            reserved_output=self.output_reservation,
            framing_overhead=self.framing_overhead,
            safety_margin=self.safety_margin,
        )


@dataclass(frozen=True, slots=True)
class BudgetVerification:
    counted_input_tokens: int
    input_limit: int
    remaining_tokens: int


def verify_final_budget(
    text: str,
    counter: Callable[[str], int],
    budget: FinalBudget,
) -> BudgetVerification:
    """Fail closed if the final rendered input exceeds caller's hard limit."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not callable(counter):
        raise TypeError("counter must be callable")
    if not isinstance(budget, FinalBudget):
        raise TypeError("budget must be FinalBudget")
    limit = budget.input_limit
    tokens = counter(text)
    if type(tokens) is not int or tokens < 0:
        raise ValueError("counter must return a nonnegative integer")
    if tokens > limit:
        raise BudgetExceeded("rendered input exceeds declared input token limit")
    return BudgetVerification(tokens, limit, limit - tokens)
