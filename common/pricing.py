"""Per-token USD pricing for the models this project uses.

Ported from sibling rag-recipes' `recipes/pricing.py` and kept in sync (both
projects share the same maintained rate table — updating one means updating
the other).

OpenAI prices verified against platform.openai.com/docs/models and
platform.openai.com/docs/pricing on 2026-08-11. Anthropic (claude-sonnet-5)
verified against platform.claude.com/docs/en/about-claude/models/overview
(base rate) and .../build-with-claude/prompt-caching (cache multipliers:
write 5-min TTL = 1.25x base input, read = 0.1x base input) on 2026-08-12.
Re-verify before relying on these for a real spend decision if this file is
more than a few weeks old, per SPEC.md R7 (model version pinning).

All prices are USD per 1,000,000 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float
    cached_input_per_1m: float | None
    output_per_1m: float
    # Anthropic-only: price for tokens written to a fresh cache entry.
    # None for models with no separate cache-write tier (all OpenAI models
    # here — their caching is automatic with no explicit write cost).
    cache_creation_per_1m: float | None = None


# Chat/completion + vision models used across agents.
CHAT_PRICING: dict[str, ModelPricing] = {
    "gpt-4.1-mini-2025-04-14": ModelPricing(
        input_per_1m=0.40, cached_input_per_1m=0.10, output_per_1m=1.60
    ),
    "gpt-5.4-mini-2026-03-17": ModelPricing(
        input_per_1m=0.75, cached_input_per_1m=0.075, output_per_1m=4.50
    ),
    # F1 receipt extractor + F2 contract reviewer use this.
    "claude-sonnet-5": ModelPricing(
        input_per_1m=2.00,
        cached_input_per_1m=0.20,
        output_per_1m=10.00,
        cache_creation_per_1m=2.50,
    ),
}

# Embedding models. No output tokens; cached_input_per_1m is None because
# OpenAI does not offer prompt-caching discounts on embedding calls.
EMBEDDING_PRICING: dict[str, ModelPricing] = {
    "text-embedding-3-small": ModelPricing(
        input_per_1m=0.02, cached_input_per_1m=None, output_per_1m=0.0
    ),
}


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> float:
    """Compute the USD cost of one LLM or embedding call.

    `cached_input_tokens` (discounted cache-read tier) and
    `cache_creation_input_tokens` (premium cache-write tier, Anthropic only)
    are both subsets of `input_tokens`; their sum must not exceed it. The
    remaining, non-cached portion of `input_tokens` is billed at the regular
    input rate.
    """
    pricing = CHAT_PRICING.get(model) or EMBEDDING_PRICING.get(model)
    if pricing is None:
        raise ValueError(
            f"No pricing entry for model {model!r}. Add it to common/pricing.py "
            "after verifying the current rate at the provider's pricing docs."
        )

    if cached_input_tokens + cache_creation_input_tokens > input_tokens:
        raise ValueError(
            "cached_input_tokens + cache_creation_input_tokens cannot exceed input_tokens"
        )

    uncached_input_tokens = input_tokens - cached_input_tokens - cache_creation_input_tokens
    cost = (uncached_input_tokens / 1_000_000) * pricing.input_per_1m
    cost += (output_tokens / 1_000_000) * pricing.output_per_1m

    if cached_input_tokens:
        if pricing.cached_input_per_1m is None:
            raise ValueError(
                f"Model {model!r} has no cached-input rate; caller passed "
                f"cached_input_tokens={cached_input_tokens} in error."
            )
        cost += (cached_input_tokens / 1_000_000) * pricing.cached_input_per_1m

    if cache_creation_input_tokens:
        if pricing.cache_creation_per_1m is None:
            raise ValueError(
                f"Model {model!r} has no cache-creation rate; caller passed "
                f"cache_creation_input_tokens={cache_creation_input_tokens} in error."
            )
        cost += (cache_creation_input_tokens / 1_000_000) * pricing.cache_creation_per_1m

    return cost
