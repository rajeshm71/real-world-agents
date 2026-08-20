"""Tests for common/pricing.py. Same rigor as rag-recipes' pricing tests,
covering: normal billing, cache-tier arithmetic, error paths, and the
important cache_creation_input_tokens sum-must-equal-input constraint that
was the source of a real bug in rag-recipes' pipeline.
"""

from __future__ import annotations

import pytest

from common.pricing import CHAT_PRICING, EMBEDDING_PRICING, cost_usd

# ---------- Normal billing ----------


def test_cost_zero_when_zero_tokens():
    assert cost_usd("gpt-4.1-mini-2025-04-14", input_tokens=0) == 0.0


def test_cost_input_only():
    # gpt-4.1-mini: $0.40 per 1M input
    cost = cost_usd("gpt-4.1-mini-2025-04-14", input_tokens=1_000_000)
    assert cost == pytest.approx(0.40)


def test_cost_input_plus_output():
    # $0.40 input + $1.60 output per 1M
    cost = cost_usd(
        "gpt-4.1-mini-2025-04-14",
        input_tokens=1_000_000,
        output_tokens=500_000,
    )
    assert cost == pytest.approx(0.40 + 0.80)


def test_cost_scales_linearly():
    cost_1k = cost_usd("gpt-4.1-mini-2025-04-14", input_tokens=1000)
    cost_10k = cost_usd("gpt-4.1-mini-2025-04-14", input_tokens=10_000)
    assert cost_10k == pytest.approx(cost_1k * 10)


# ---------- Cached-input tier (all providers) ----------


def test_cost_cached_tokens_use_discounted_rate():
    # gpt-4.1-mini: $0.40 normal, $0.10 cached (4x discount)
    # 1M tokens all cached
    cost_cached = cost_usd(
        "gpt-4.1-mini-2025-04-14",
        input_tokens=1_000_000,
        cached_input_tokens=1_000_000,
    )
    assert cost_cached == pytest.approx(0.10)


def test_cost_mixed_cached_and_uncached_input():
    # Half cached, half regular.
    cost = cost_usd(
        "gpt-4.1-mini-2025-04-14",
        input_tokens=1_000_000,
        cached_input_tokens=500_000,
    )
    # 500k regular @ $0.40/M + 500k cached @ $0.10/M
    expected = (500_000 / 1_000_000) * 0.40 + (500_000 / 1_000_000) * 0.10
    assert cost == pytest.approx(expected)


def test_cost_cached_tokens_error_when_model_has_no_cached_rate():
    """Embedding models have cached_input_per_1m=None -- passing
    cached_input_tokens on them is a caller bug and should raise."""
    with pytest.raises(ValueError, match="no cached-input rate"):
        cost_usd(
            "text-embedding-3-small",
            input_tokens=1000,
            cached_input_tokens=500,
        )


# ---------- Cache-creation tier (Anthropic only) ----------


def test_cost_cache_creation_uses_premium_rate():
    # claude-sonnet-5: $2.00 normal, $2.50 cache write, $0.20 cache read
    cost = cost_usd(
        "claude-sonnet-5",
        input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )
    assert cost == pytest.approx(2.50)


def test_cost_all_three_tiers_together():
    """Real Anthropic response with a fresh cache write, some cache reads,
    and some uncached input. This is what pattern 07 (contextual retrieval)
    produces in the rag-recipes sibling project."""
    # 100k uncached + 200k cache-write + 700k cache-read = 1M input
    cost = cost_usd(
        "claude-sonnet-5",
        input_tokens=1_000_000,
        output_tokens=100_000,
        cached_input_tokens=700_000,
        cache_creation_input_tokens=200_000,
    )
    expected = (
        (100_000 / 1_000_000) * 2.00  # uncached input
        + (700_000 / 1_000_000) * 0.20  # cache read
        + (200_000 / 1_000_000) * 2.50  # cache write
        + (100_000 / 1_000_000) * 10.00  # output
    )
    assert cost == pytest.approx(expected)


def test_cost_cache_creation_error_when_model_has_no_write_tier():
    """OpenAI models don't have a cache-creation rate (their caching is
    automatic, no explicit write cost). Passing cache_creation_input_tokens
    on them is a caller bug and must raise -- otherwise a pattern that
    later switches providers would silently mis-attribute cost."""
    with pytest.raises(ValueError, match="no cache-creation rate"):
        cost_usd(
            "gpt-4.1-mini-2025-04-14",
            input_tokens=1000,
            cache_creation_input_tokens=500,
        )


# ---------- Sum-must-not-exceed-input invariant ----------


def test_cost_error_when_cache_tiers_exceed_input():
    """cached_input_tokens + cache_creation_input_tokens > input_tokens is
    always a caller bug (both are subsets of input_tokens). This invariant
    catches a whole class of accounting mistakes at the API boundary."""
    with pytest.raises(ValueError, match="cannot exceed input_tokens"):
        cost_usd(
            "claude-sonnet-5",
            input_tokens=1000,
            cached_input_tokens=800,
            cache_creation_input_tokens=500,  # 800 + 500 > 1000
        )


def test_cost_ok_when_tiers_exactly_equal_input():
    """Edge case: cached + creation == input (nothing uncached). Should
    price everything at the tier rates, zero at the uncached rate."""
    cost = cost_usd(
        "claude-sonnet-5",
        input_tokens=1000,
        cached_input_tokens=600,
        cache_creation_input_tokens=400,
    )
    expected = (600 / 1_000_000) * 0.20 + (400 / 1_000_000) * 2.50
    assert cost == pytest.approx(expected)


# ---------- Unknown model ----------


def test_cost_error_when_model_not_in_pricing_table():
    with pytest.raises(ValueError, match="No pricing entry"):
        cost_usd("totally-fake-model", input_tokens=100)


# ---------- Embedding models ----------


def test_cost_embedding_model_no_output():
    # text-embedding-3-small: $0.02/1M input, no output.
    cost = cost_usd("text-embedding-3-small", input_tokens=1_000_000)
    assert cost == pytest.approx(0.02)


# ---------- Realistic receipt-extractor sanity check ----------


def test_receipt_extractor_cost_matches_documented_estimate():
    """The receipt extractor's README estimates ~$0.005 per extraction
    (Claude Sonnet-5 with vision: ~1500 input tokens + ~200 output tokens).
    This test pins that estimate -- if the pricing table changes, this test
    catches the drift and forces us to update the README's cost estimate too."""
    cost = cost_usd(
        "claude-sonnet-5",
        input_tokens=1500,
        output_tokens=200,
    )
    # 1500 * $2/1M + 200 * $10/1M = $0.003 + $0.002 = $0.005
    assert cost == pytest.approx(0.005)


# ---------- Rate-table sanity ----------


def test_all_chat_pricing_entries_have_positive_rates():
    """Every model in the rate table should have positive input + output
    rates. Catches accidental zeros/None where they'd cause silent
    under-billing."""
    for model, pricing in CHAT_PRICING.items():
        assert pricing.input_per_1m > 0, f"{model} has non-positive input rate"
        assert pricing.output_per_1m > 0, f"{model} has non-positive output rate"


def test_embedding_pricing_entries_have_no_output_cost():
    """Embedding models produce vectors, not tokens -- output rate is meaningless."""
    for model, pricing in EMBEDDING_PRICING.items():
        assert pricing.output_per_1m == 0.0, f"{model} unexpectedly has output cost"


def test_claude_cache_rates_follow_documented_multipliers():
    """Per Anthropic docs (2026-08-12 verification):
    cache-read = 0.1x base input, cache-write = 1.25x base input.
    This test pins the multiplier semantics so a future rate update
    that violates the documented ratio would fail loudly."""
    claude = CHAT_PRICING["claude-sonnet-5"]
    assert claude.cached_input_per_1m == pytest.approx(claude.input_per_1m * 0.10)
    assert claude.cache_creation_per_1m == pytest.approx(claude.input_per_1m * 1.25)


# ---------- Gemini (multi-provider support, user request) ----------


def test_gemini_flash_pricing_matches_verified_rate():
    # Verified 2026-08-13: $0.30/1M input, $2.50/1M output.
    cost = cost_usd("gemini-2.5-flash", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(0.30 + 2.50)


def test_gemini_flash_lite_pricing_matches_verified_rate():
    # Verified 2026-08-13: $0.10/1M input, $0.40/1M output.
    cost = cost_usd("gemini-2.5-flash-lite", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(0.10 + 0.40)


def test_gemini_has_no_cache_tiers_priced():
    """Gemini entries deliberately have cached_input_per_1m=None (no
    per-call user-facing cache-read rate at this project's usage tier,
    unlike Anthropic's explicit cache_control). Passing cached_input_tokens
    against a Gemini model must raise, same as any other no-cache-tier model."""
    with pytest.raises(ValueError, match="no cached-input rate"):
        cost_usd("gemini-2.5-flash", input_tokens=1000, cached_input_tokens=500)
