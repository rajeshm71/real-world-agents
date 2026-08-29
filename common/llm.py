"""LLM adapter for real-world-agents.

`LLM.complete()` returns an `LLMResponse` with text, token counts,
cached-token count (for prompt-caching cost tracking), and latency. Four
concrete implementations ship here: `OpenAILLM`, `AnthropicLLM`, `GeminiLLM`,
and `MockLLM` (deterministic, used in CI per CONTRIBUTING.md's R8 -- CI never
touches a real API key, for any provider).

Provider + model are both user-configurable via env vars (never hardcoded to
one provider) -- this project is OSS and every user brings their own key(s).
`LLM_PROVIDER` picks openai (default) / anthropic / gemini / mock.
`OPENAI_DEFAULT_MODEL` / `ANTHROPIC_DEFAULT_MODEL` / `GEMINI_DEFAULT_MODEL`
override the per-provider default model string -- see `DEFAULT_MODELS` below.
None of this requires an API key to develop or run tests against; a key is
only needed for an actual non-mock call, which is the end user's job, not a
development-time requirement.

Ported from sibling rag-recipes' recipes/llm.py. The main deviation is the env
var name: `LLM_PROVIDER` (this project) instead of `RAG_RECIPES_LLM` -- cleaner
and idiomatic since this project's name is longer than rag-recipes'.

Agents that use a framework's own client (Instructor, LangGraph, CrewAI,
PydanticAI, DSPy) skip this module entirely -- the framework's own client sits
directly against the provider SDK. This adapter exists for agents that DON'T
need a framework (e.g., the contract reviewer, the from-scratch agent loop).
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Protocol

# Per-provider default model, overridable by env var. Values verified
# 2026-08-11 (OpenAI, ported from sibling rag-recipes) and 2026-08-13
# (Anthropic, Gemini) -- see common/pricing.py's module docstring for the
# full verification notes, including the Gemini-has-no-dated-snapshot
# caveat. Re-verify before relying on these for a real run -- model version
# pinning is R7 in CONTRIBUTING.md's hard rules.
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4.1-mini-2025-04-14",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-2.5-flash",
    # Local-Ollama fallback: Gemma-4-E4B is natively multimodal and
    # supports tool calling. Fits in 8 GB VRAM. Users pull the model
    # once with `ollama pull gemma4:e4b`.
    "ollama": "gemma4:e4b",
}

_MODEL_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_DEFAULT_MODEL",
    "anthropic": "ANTHROPIC_DEFAULT_MODEL",
    "gemini": "GEMINI_DEFAULT_MODEL",
    "ollama": "OLLAMA_DEFAULT_MODEL",
}


def resolve_model(provider: str) -> str:
    """Return the model string to use for `provider`: the env-var override
    if the user set one, else DEFAULT_MODELS[provider]. This is the single
    source of truth every agent should call instead of hardcoding a model
    string, so `OPENAI_DEFAULT_MODEL=gpt-5.4-mini-2026-03-17` (or any other
    override) takes effect everywhere without code changes.
    """
    provider = provider.lower()
    env_var = _MODEL_ENV_VARS.get(provider)
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return override
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"No default model for provider {provider!r}. "
            f"Expected one of {sorted(DEFAULT_MODELS)}."
        )
    return DEFAULT_MODELS[provider]


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    # Anthropic-only: tokens written to a FRESH cache entry, billed at a
    # premium (distinct tier from cached_input_tokens' discounted-read rate).
    # Always 0 for OpenAI/Mock, which have no separate write tier.
    cache_creation_input_tokens: int = 0
    latency_ms: float = 0.0


class LLM(Protocol):
    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        # Hint for providers with explicit prompt caching (Anthropic): the
        # portion of the request that's shared/repeated across calls and
        # should be cached. No-op for OpenAILLM (its caching is automatic,
        # no explicit control at our usage tier) and recorded-only for
        # MockLLM (doesn't change the mock response, but drives the cache-
        # tier simulation in MockLLM.complete()).
        cacheable_prefix: str | None = None,
    ) -> LLMResponse: ...


class OpenAILLM:
    """Production adapter. Requires OPENAI_API_KEY."""

    def __init__(self, api_key: str | None = None) -> None:
        # Lazy import so MockLLM-only test runs never need `openai` installed.
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cacheable_prefix: str | None = None,  # no-op: see LLM Protocol docstring
    ) -> LLMResponse:
        start = time.perf_counter()
        base_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
        }
        try:
            response = self._client.chat.completions.create(
                **base_kwargs, temperature=temperature
            )
        except Exception as exc:
            # Some reasoning-capable models (gpt-5.4-mini family) reject
            # temperature entirely rather than clamping/ignoring it. On the
            # specific "unsupported parameter" error, retry once without
            # temperature. Ported behavior from rag-recipes.
            message = str(exc).lower()
            if "temperature" in message and (
                "unsupported" in message or "not supported" in message
            ):
                response = self._client.chat.completions.create(**base_kwargs)
            else:
                raise
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        usage = response.usage
        cached = 0
        if usage is not None and getattr(usage, "prompt_tokens_details", None):
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cached_input_tokens=cached,
            latency_ms=latency_ms,
        )


class AnthropicLLM:
    """Production adapter. Requires ANTHROPIC_API_KEY. Handles the cache tiers
    (cached_input_tokens + cache_creation_input_tokens) correctly per
    Anthropic's usage-semantics: usage.input_tokens EXCLUDES both cache tiers
    (unlike OpenAI's prompt_tokens which includes cached_tokens), so we
    recombine here to keep LLMResponse.input_tokens meaning "total input" across
    providers.
    """

    def __init__(self, api_key: str | None = None) -> None:
        # Lazy import, same convention as OpenAILLM.
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cacheable_prefix: str | None = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if cacheable_prefix:
            kwargs["system"] = [
                {"type": "text", "text": cacheable_prefix, "cache_control": {"type": "ephemeral"}}
            ]
        response = self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        # See docstring: Anthropic's input_tokens excludes cache tiers.
        total_input = usage.input_tokens + cache_read + cache_creation
        text = "".join(block.text for block in response.content if block.type == "text")

        return LLMResponse(
            text=text,
            input_tokens=total_input,
            output_tokens=usage.output_tokens,
            cached_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
            latency_ms=latency_ms,
        )


class GeminiLLM:
    """Production adapter. Requires GEMINI_API_KEY (or GOOGLE_API_KEY, which
    the google-genai SDK also accepts). Uses Google's `google-genai` SDK
    (the current recommended client per ai.google.dev, GA since May 2025).

    NOTE: token-usage attribute names (`usage_metadata.prompt_token_count`
    etc.) are written from documented SDK conventions, not independently
    exercised against a real API call (no GEMINI_API_KEY available). Verify
    before relying on this for a real run.
    """

    def __init__(self, api_key: str | None = None) -> None:
        # Lazy import, same convention as OpenAILLM/AnthropicLLM.
        from google import genai

        self._client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )

    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cacheable_prefix: str | None = None,  # no-op: Gemini's caching has no per-call control at this tier
    ) -> LLMResponse:
        start = time.perf_counter()
        full_prompt = f"{cacheable_prefix}\n\n{prompt}" if cacheable_prefix else prompt
        response = self._client.models.generate_content(
            model=model,
            contents=full_prompt,
            config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        latency_ms = (time.perf_counter() - start) * 1000

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        cached = getattr(usage, "cached_content_token_count", 0) or 0

        return LLMResponse(
            text=response.text or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached,
            latency_ms=latency_ms,
        )


class OllamaLLM:
    """Local-Ollama adapter via Ollama's OpenAI-compatible endpoint.

    No API key required (Ollama runs entirely on the user's machine).
    The `openai` Python SDK is reused with a `base_url` pointed at the
    local server -- no additional runtime dep. Users need to install
    Ollama separately (https://ollama.com/download) and pull the
    default model once: `ollama pull gemma4:e4b`.

    `OLLAMA_HOST` env var overrides the endpoint; if the user sets it
    with a trailing `/v1` already included, that is stripped so the
    final URL never doubles up (`.../v1/v1`)."""

    def __init__(self, host: str | None = None) -> None:
        # Lazy import, same convention as OpenAILLM/AnthropicLLM.
        from openai import OpenAI

        raw_host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        base = raw_host.rstrip("/").removesuffix("/v1")
        self._client = OpenAI(base_url=f"{base}/v1", api_key="ollama")

    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cacheable_prefix: str | None = None,  # no-op: llama.cpp has no caching-tier we can control
    ) -> LLMResponse:
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            # Ollama server not running is the single most common
            # failure mode -- surface a clean hint instead of the raw
            # httpx traceback.
            exc_class = type(exc).__name__.lower()
            msg_lower = str(exc).lower()
            if "connection" in exc_class or "connect" in msg_lower or "refused" in msg_lower:
                raise RuntimeError(
                    "Ollama connection failed. Is `ollama serve` running? "
                    "See https://ollama.com/download to install, then "
                    f"`ollama pull {DEFAULT_MODELS['ollama']}` to fetch the "
                    "default model."
                ) from exc
            raise
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        usage = response.usage
        # Ollama's OpenAI-compat response includes prompt/completion
        # tokens but NOT prompt_tokens_details.cached_tokens (that's
        # OpenAI-specific). Default cached=0 gracefully.
        cached = 0
        if usage is not None and getattr(usage, "prompt_tokens_details", None):
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cached_input_tokens=cached,
            latency_ms=latency_ms,
        )


@dataclass
class _RecordedCall:
    prompt: str
    model: str
    temperature: float
    max_tokens: int
    cacheable_prefix: str | None = None


@dataclass
class MockLLM:
    """Deterministic adapter for tests and CI. Never makes a network call.

    Returns a fixed string by default, or a caller-supplied mapping from prompt
    substrings to canned responses (checked in insertion order, first match
    wins) for tests that need different answers to different prompts. Every
    call is recorded in `.calls` for test assertions.

    Cache-tier simulation: when `cacheable_prefix` is passed, the FIRST call
    with a given prefix (per MockLLM instance) bills its prefix-portion tokens
    as `cache_creation_input_tokens` (write); every subsequent call with the
    SAME prefix bills as `cached_input_tokens` (read). Only activates when a
    caller passes cacheable_prefix -- agents that don't use caching get
    plain input_tokens accounting. Same behavior as rag-recipes' MockLLM,
    verified to produce the correct 2.00x read:write ratio on repeated calls.
    """

    default_response: str = "This is a mock response."
    canned: dict[str, str] = field(default_factory=dict)
    calls: list[_RecordedCall] = field(default_factory=list)
    # Tracks which cacheable_prefix strings this instance has "seen," to
    # drive the cache write-then-read simulation.
    _seen_prefixes: set[str] = field(default_factory=set)

    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cacheable_prefix: str | None = None,
    ) -> LLMResponse:
        self.calls.append(_RecordedCall(prompt, model, temperature, max_tokens, cacheable_prefix))

        # Pick response: canned substring match first, then default.
        text = self.default_response
        for substring, canned_text in self.canned.items():
            if substring in prompt:
                text = canned_text
                break

        # Deterministic pseudo token counts (~4 chars per token). Good enough
        # for cost/metric code to have something to work with in tests.
        output_tokens = max(1, len(text) // 4)

        if cacheable_prefix:
            prefix_tokens = max(1, len(cacheable_prefix) // 4)
            suffix_tokens = max(1, len(prompt) // 4)
            input_tokens = prefix_tokens + suffix_tokens
            if cacheable_prefix in self._seen_prefixes:
                cached_input_tokens = prefix_tokens
                cache_creation_input_tokens = 0
            else:
                cached_input_tokens = 0
                cache_creation_input_tokens = prefix_tokens
                self._seen_prefixes.add(cacheable_prefix)
        else:
            input_tokens = max(1, len(prompt) // 4)
            cached_input_tokens = 0
            cache_creation_input_tokens = 0

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            latency_ms=0.1,
        )


_FALLBACK_BANNER_SHOWN = False


def _autodetect_provider() -> str:
    """Called when `LLM_PROVIDER` is unset. Picks the first cloud
    provider whose API key is a non-empty env value; falls back to
    ollama (local, free, offline) otherwise. Empty-string keys count
    as unset so a `.env` line like `OPENAI_API_KEY=""` doesn't wrongly
    win."""
    for env_var, provider in (
        ("OPENAI_API_KEY", "openai"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("GEMINI_API_KEY", "gemini"),
    ):
        val = (os.environ.get(env_var) or "").strip()
        if val:
            return provider
    _log_ollama_fallback_banner()
    return "ollama"


def _log_ollama_fallback_banner() -> None:
    """Once-per-process stderr banner telling the user we picked the
    local-Ollama fallback because no API keys were set. Suppressed
    when NO_FALLBACK_BANNER=1 (useful for tests / CI-like runs)."""
    global _FALLBACK_BANNER_SHOWN
    if _FALLBACK_BANNER_SHOWN:
        return
    if os.environ.get("NO_FALLBACK_BANNER") == "1":
        _FALLBACK_BANNER_SHOWN = True
        return
    print(
        "[real-world-agents] no *_API_KEY env vars found; falling back to "
        f"local Ollama with model={DEFAULT_MODELS['ollama']!r}. "
        "Set LLM_PROVIDER explicitly to silence this.",
        file=sys.stderr,
    )
    _FALLBACK_BANNER_SHOWN = True


def get_llm(provider: str | None = None) -> LLM:
    """Factory. Respects the `LLM_PROVIDER` env var if `provider` is
    not passed. Values: "openai", "anthropic", "gemini", "ollama",
    "mock". CI always sets LLM_PROVIDER=mock per R8.

    When neither `provider` nor `LLM_PROVIDER` is set, autodetects
    from the *_API_KEY env vars and falls back to local Ollama if
    none are configured (OSS-friendly default -- no paid key needed
    to kick the tires)."""
    if provider is None:
        raw = os.environ.get("LLM_PROVIDER")
        provider = raw if raw is not None else _autodetect_provider()
    provider = provider.lower()
    if provider == "openai":
        return OpenAILLM()
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "gemini":
        return GeminiLLM()
    if provider == "ollama":
        return OllamaLLM()
    if provider == "mock":
        return MockLLM()
    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. Expected 'openai', "
        "'anthropic', 'gemini', 'ollama', or 'mock'."
    )


# Convenience aliases for agents that specifically need one provider (e.g.
# the receipt extractor's vision step). Under mock, every one of these
# returns the same MockLLM class (R8 applies to every provider, not just
# OpenAI) -- `provider` param lets an agent force a non-default provider
# while still honoring LLM_PROVIDER=mock in CI.
def get_anthropic_llm() -> LLM:
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "mock":
        return MockLLM()
    return AnthropicLLM()


def get_gemini_llm() -> LLM:
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "mock":
        return MockLLM()
    return GeminiLLM()


# Helper used by MockLLM in some downstream tests -- kept public so downstream
# agents can build deterministic canned responses that vary by prompt hash.
def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
