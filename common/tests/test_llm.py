"""Tests for common/llm.py. All tests run against MockLLM (LLM_PROVIDER=mock)
per SPEC.md R8 -- no real API keys in CI. Real-provider smoke tests are run
manually by the maintainer before F1 ship.
"""

from __future__ import annotations

import pytest

from common.llm import (
    LLM,
    AnthropicLLM,
    LLMResponse,
    MockLLM,
    OpenAILLM,
    get_anthropic_llm,
    get_llm,
)

# ---------- MockLLM ----------


def test_mock_returns_default_response_and_records_call():
    llm = MockLLM(default_response="hello world")
    response = llm.complete(prompt="anything", model="fake-model")

    assert response.text == "hello world"
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.cached_input_tokens == 0
    assert response.cache_creation_input_tokens == 0
    assert len(llm.calls) == 1
    assert llm.calls[0].prompt == "anything"
    assert llm.calls[0].model == "fake-model"


def test_mock_canned_responses_match_by_substring_first_wins():
    llm = MockLLM(
        default_response="default",
        canned={"receipt": "receipt-response", "invoice": "invoice-response"},
    )
    assert llm.complete("this is a receipt", model="m").text == "receipt-response"
    assert llm.complete("this is an invoice", model="m").text == "invoice-response"
    assert llm.complete("neither word here", model="m").text == "default"


def test_mock_first_match_wins_when_multiple_substrings_present():
    # Insertion order defines priority. "alpha" is checked before "beta".
    llm = MockLLM(default_response="d", canned={"alpha": "A", "beta": "B"})
    assert llm.complete("contains alpha AND beta", model="m").text == "A"


def test_mock_records_every_call_in_order():
    llm = MockLLM()
    llm.complete("first", model="m1", temperature=0.5, max_tokens=100)
    llm.complete("second", model="m2", temperature=0.0, max_tokens=200)
    assert len(llm.calls) == 2
    assert llm.calls[0].prompt == "first"
    assert llm.calls[0].temperature == 0.5
    assert llm.calls[0].max_tokens == 100
    assert llm.calls[1].prompt == "second"
    assert llm.calls[1].model == "m2"


# ---------- MockLLM cache-tier simulation (the important part) ----------


def test_mock_cache_first_call_bills_as_creation():
    """First call with a given cacheable_prefix must bill prefix tokens as
    cache_creation_input_tokens, not cached_input_tokens. Otherwise agents
    using Anthropic caching wouldn't be testable at all."""
    llm = MockLLM()
    prefix = "a" * 400  # ~100 pseudo-tokens
    response = llm.complete("hello", model="m", cacheable_prefix=prefix)

    assert response.cache_creation_input_tokens == 100
    assert response.cached_input_tokens == 0
    # input_tokens must include both the prefix and the suffix portions.
    assert response.input_tokens >= 100


def test_mock_cache_second_call_same_prefix_bills_as_read():
    """Same prefix on a subsequent call must bill as cached_input_tokens
    (the discounted read tier), not creation. This is the tier that makes
    Anthropic prompt caching economical."""
    llm = MockLLM()
    prefix = "a" * 400  # ~100 pseudo-tokens

    llm.complete("first", model="m", cacheable_prefix=prefix)  # write
    second = llm.complete("second", model="m", cacheable_prefix=prefix)  # read

    assert second.cached_input_tokens == 100
    assert second.cache_creation_input_tokens == 0


def test_mock_cache_different_prefix_is_a_fresh_write():
    """A new/different prefix on the same MockLLM instance must be a fresh
    cache write, not a read from the earlier prefix's slot."""
    llm = MockLLM()
    llm.complete("x", model="m", cacheable_prefix="a" * 400)
    # Different prefix -- should be a write again.
    response = llm.complete("y", model="m", cacheable_prefix="b" * 400)

    assert response.cache_creation_input_tokens > 0
    assert response.cached_input_tokens == 0


def test_mock_no_cacheable_prefix_means_no_cache_billing():
    """When cacheable_prefix is not passed, both cache-tier fields must stay
    zero -- otherwise every non-caching call would look like it was using
    caching in cost accounting."""
    llm = MockLLM()
    response = llm.complete("hello", model="m")
    assert response.cached_input_tokens == 0
    assert response.cache_creation_input_tokens == 0


# ---------- Factory / env-var routing ----------


def test_get_llm_returns_mock_when_env_var_set(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    llm = get_llm()
    assert isinstance(llm, MockLLM)


def test_get_llm_returns_mock_when_provider_kwarg_passed():
    llm = get_llm(provider="mock")
    assert isinstance(llm, MockLLM)


def test_get_llm_provider_kwarg_overrides_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    llm = get_llm(provider="mock")
    assert isinstance(llm, MockLLM)


def test_get_llm_rejects_unknown_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm(provider="totally-fake-provider")


def test_get_llm_case_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "MOCK")
    llm = get_llm()
    assert isinstance(llm, MockLLM)


def test_get_anthropic_llm_returns_mock_when_env_var_set(monkeypatch):
    """R8 says CI is mock-only for every provider, not just OpenAI. The
    Anthropic-specific factory must respect LLM_PROVIDER=mock too."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    llm = get_anthropic_llm()
    assert isinstance(llm, MockLLM)


# ---------- Real-adapter smoke (no network, no API key) ----------


def test_openai_llm_raises_helpful_error_without_key(monkeypatch):
    """Verified against openai SDK behavior: the client raises OpenAIError at
    construction time when no API key is present in the arg or env. Test
    documents this so a future SDK change (e.g. deferring to first call)
    would surface as a test failure and force a review.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    with pytest.raises(Exception) as exc_info:
        OpenAILLM()
    # Failure must be about credentials, not an import bug.
    message = str(exc_info.value).lower()
    assert "key" in message or "credential" in message or "auth" in message


def test_anthropic_llm_constructs_without_key(monkeypatch):
    """Verified against anthropic SDK behavior: unlike openai's SDK, the
    anthropic client does NOT raise at construction on a missing key -- it
    defers to the first API call. This test documents the split so a future
    SDK-alignment change (either direction) surfaces as a test failure and
    forces a review of our downstream error-handling assumptions.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Constructor must NOT raise. Only actual .complete() calls need a key.
    llm = AnthropicLLM()
    assert llm is not None


# ---------- Protocol conformance ----------


def test_mock_llm_conforms_to_LLM_protocol():
    """Structural typing check -- MockLLM must be usable anywhere an LLM
    is expected. If this test breaks, the Protocol contract changed."""
    llm: LLM = MockLLM()
    response = llm.complete(prompt="x", model="m")
    assert isinstance(response, LLMResponse)
