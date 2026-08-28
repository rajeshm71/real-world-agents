"""Tiny arithmetic helpers for the sample_repo demo project."""

from __future__ import annotations


def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""
    return a * b


def clamp(value: float, low: float, high: float) -> float:
    """Clamp `value` to the [low, high] range."""
    if low > high:
        raise ValueError("low must be <= high")
    return max(low, min(high, value))
