"""Sample target module for agent #08's demo. Not real project code.

A tiny temperature-converter class with one class method, one static
method, and one deliberate edge case (division-by-zero potential in
the rate helper) so the generated tests have something meaningful to
cover. Small enough to read in 20 seconds, self-contained (no imports
beyond stdlib), and public enough that `list_public_symbols` sees
three symbols worth generating tests for.
"""

from __future__ import annotations


class TemperatureConverter:
    """Convert between Celsius and Fahrenheit."""

    ABSOLUTE_ZERO_CELSIUS = -273.15
    ABSOLUTE_ZERO_FAHRENHEIT = -459.67

    @classmethod
    def celsius_to_fahrenheit(cls, celsius: float) -> float:
        """Convert Celsius to Fahrenheit. Rejects below-absolute-zero input."""
        if celsius < cls.ABSOLUTE_ZERO_CELSIUS:
            raise ValueError(f"celsius {celsius} is below absolute zero")
        return celsius * 9 / 5 + 32

    @classmethod
    def fahrenheit_to_celsius(cls, fahrenheit: float) -> float:
        """Convert Fahrenheit to Celsius. Rejects below-absolute-zero input."""
        if fahrenheit < cls.ABSOLUTE_ZERO_FAHRENHEIT:
            raise ValueError(f"fahrenheit {fahrenheit} is below absolute zero")
        return (fahrenheit - 32) * 5 / 9

    @staticmethod
    def rate_of_change_per_minute(delta_celsius: float, minutes: float) -> float:
        """Degrees Celsius per minute. Does not guard against zero minutes."""
        return delta_celsius / minutes
