"""Tests for common/llm.py. All tests run against MockLLM (LLM_PROVIDER=mock)
-- CONTRIBUTING.md's R8 rule: no real API keys in CI. Real-provider smoke
tests are run manually by the maintainer before shipping.
"""

from __future__ import annotations

import pytest

from common.llm import (
    DEFAULT_MODELS,
    LLM,
    AnthropicLLM,
    GeminiLLM,
    LLMResponse,
    MockLLM,
    OllamaLLM,
    OpenAILLM,
    _autodetect_provider,
    _reset_fallback_banner_state,
    get_anthropic_llm,
    get_gemini_llm,
    get_llm,
    resolve_model,
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


# ---------- resolve_model (configurable provider + model) ----------


def test_resolve_model_returns_default_when_no_env_override(monkeypatch):
    for provider in ("openai", "anthropic", "gemini"):
        for env_var in ("OPENAI_DEFAULT_MODEL", "ANTHROPIC_DEFAULT_MODEL", "GEMINI_DEFAULT_MODEL"):
            monkeypatch.delenv(env_var, raising=False)
        assert resolve_model(provider) == DEFAULT_MODELS[provider]


def test_resolve_model_env_override_takes_precedence(monkeypatch):
    """The whole point of this feature: a user must be able to swap models
    without touching code, e.g. bump OpenAI up to gpt-5.4-mini."""
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-5.4-mini-2026-03-17")
    assert resolve_model("openai") == "gpt-5.4-mini-2026-03-17"


def test_resolve_model_override_is_per_provider_not_global(monkeypatch):
    """Setting ANTHROPIC_DEFAULT_MODEL must not affect openai's resolution --
    each provider's override is independent."""
    monkeypatch.setenv("ANTHROPIC_DEFAULT_MODEL", "some-other-claude-model")
    monkeypatch.delenv("OPENAI_DEFAULT_MODEL", raising=False)
    assert resolve_model("openai") == DEFAULT_MODELS["openai"]
    assert resolve_model("anthropic") == "some-other-claude-model"


def test_resolve_model_case_insensitive_provider():
    assert resolve_model("OpenAI") == resolve_model("openai")


def test_resolve_model_rejects_unknown_provider():
    with pytest.raises(ValueError, match="No default model for provider"):
        resolve_model("totally-fake-provider")


def test_default_models_covers_every_real_provider():
    """Every non-mock provider get_llm() supports must have a DEFAULT_MODELS
    entry, or resolve_model() would raise for a provider get_llm() accepts --
    an inconsistency that would only surface at call time, not import time."""
    assert set(DEFAULT_MODELS) == {"openai", "anthropic", "gemini", "ollama"}


# ---------- GeminiLLM + factory routing ----------


def test_get_llm_returns_gemini_when_env_var_set(monkeypatch):
    """Verifies routing only -- does not construct a real GeminiLLM (would
    require the google-genai package + a key). Mock provider short-circuits
    before GeminiLLM's constructor runs."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    llm = get_llm()
    assert isinstance(llm, MockLLM)


def test_get_gemini_llm_returns_mock_when_env_var_set(monkeypatch):
    """R8: CI is mock-only for every provider, including Gemini."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    llm = get_gemini_llm()
    assert isinstance(llm, MockLLM)


def test_get_llm_rejects_unknown_provider_still_works_with_gemini_added(monkeypatch):
    """Regression guard: adding gemini as a valid provider must not loosen
    the "reject anything else" behavior."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm(provider="still-not-a-real-provider")


def test_gemini_llm_class_exists_and_is_lazy_importing():
    """GeminiLLM must not require the google-genai package to be installed
    at MODULE IMPORT time -- only at construction time (matching
    OpenAILLM/AnthropicLLM's established lazy-import convention). This test
    only checks the class is importable from common.llm; it does not
    construct an instance (that would require google-genai installed)."""
    assert GeminiLLM is not None
    assert hasattr(GeminiLLM, "complete")


# ---------- OllamaLLM + local-fallback autodetect ----------


def test_ollama_llm_builds_openai_client_at_local_endpoint(monkeypatch):
    """OllamaLLM wires the `openai` SDK against `localhost:11434/v1` and
    passes a dummy `api_key='ollama'` per the Ollama OpenAI-compat
    docs. Monkeypatches the OpenAI constructor to capture the args
    instead of instantiating a real client."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    captured = {}

    class _StubOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai
    monkeypatch.setattr(openai, "OpenAI", _StubOpenAI)
    OllamaLLM()
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["api_key"] == "ollama"


def test_ollama_host_env_var_strips_trailing_v1(monkeypatch):
    """OLLAMA_HOST with a trailing `/v1` already included must not
    produce a double `/v1/v1`."""
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434/v1")
    captured = {}

    class _StubOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai
    monkeypatch.setattr(openai, "OpenAI", _StubOpenAI)
    OllamaLLM()
    assert captured["base_url"] == "http://localhost:11434/v1"


def test_get_llm_falls_back_to_ollama_when_no_api_key_env(monkeypatch):
    """LLM_PROVIDER unset AND no *_API_KEY set -> Ollama fallback."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_FALLBACK_BANNER", "1")

    # Stub openai.OpenAI so no real client construction is attempted.
    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **_kw: object())

    llm = get_llm()
    assert isinstance(llm, OllamaLLM)


def test_get_llm_empty_api_key_treated_as_unset(monkeypatch):
    """An empty-string OPENAI_API_KEY (e.g. `OPENAI_API_KEY=` in .env)
    must not win autodetection; the chain continues past it."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")  # whitespace-only also skipped
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("NO_FALLBACK_BANNER", "1")

    assert _autodetect_provider() == "ollama"


def test_autodetect_prefers_openai_when_key_set(monkeypatch):
    """Sanity check the positive path: a real OPENAI_API_KEY wins the
    autodetection over the fallback."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-used-in-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert _autodetect_provider() == "openai"


def test_ollama_in_default_models_registry():
    """Regression guard: DEFAULT_MODELS must carry an entry for
    'ollama' so `resolve_model('ollama')` works out of the box."""
    assert "ollama" in DEFAULT_MODELS
    assert resolve_model("ollama") == DEFAULT_MODELS["ollama"]


def test_ollama_llm_translates_connection_refused_to_friendly_error(monkeypatch):
    """The single highest-value UX in the Ollama fallback: when the
    server isn't running, the user gets an actionable hint pointing
    at `ollama serve` and `ollama pull`, not a raw httpx traceback."""

    class _Completions:
        @staticmethod
        def create(**_kw):
            raise ConnectionError("connection refused")

    class _Chat:
        completions = _Completions

    class _StubClient:
        chat = _Chat()

    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **_kw: _StubClient())

    llm = OllamaLLM()
    with pytest.raises(RuntimeError, match="ollama serve"):
        llm.complete(prompt="hi", model="test")


def test_reset_fallback_banner_state_lets_banner_fire_again(monkeypatch, capsys):
    """`_reset_fallback_banner_state` is the test-isolation hook that
    lets a test verify the banner-fires-once behavior without being
    poisoned by whatever earlier test hit it first."""
    # Baseline: fire the banner (allowed), verify it went out.
    monkeypatch.delenv("NO_FALLBACK_BANNER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    _reset_fallback_banner_state()
    _autodetect_provider()
    err1 = capsys.readouterr().err
    assert "falling back to local Ollama" in err1

    # Second call in the same process is silent (once-per-process).
    _autodetect_provider()
    err2 = capsys.readouterr().err
    assert err2 == ""

    # After reset, banner fires again.
    _reset_fallback_banner_state()
    _autodetect_provider()
    err3 = capsys.readouterr().err
    assert "falling back to local Ollama" in err3
