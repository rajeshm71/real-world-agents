"""Greeting helpers for the sample_repo demo project."""

from __future__ import annotations


def greet(name: str) -> str:
    """Return a friendly greeting for `name`."""
    return f"Hello, {name}!"


def farewell(name: str) -> str:
    """Return a friendly farewell for `name`."""
    return f"Goodbye, {name}."
