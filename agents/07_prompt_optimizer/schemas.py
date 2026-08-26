"""Pydantic schema for the DSPy prompt optimizer's result.

An OptimizationResult is what the agent produces after running (or, under
mock, faking) a compilation pass: the original hand-written prompt, the
new optimizer-produced prompt, per-field accuracy before/after, and cost
+ wall-time so the reader knows what the run cost.

The per-field accuracy maps are keyed by the field names scored by agent
#01's eval harness -- currently `total`, `line_items`, `subtotal`,
`tax_total`. Kept as an open dict rather than a closed schema so a future
change in the target agent's scored-field list doesn't cascade a schema
change here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class OptimizationResult(BaseModel):
    """One compilation pass's before/after report."""

    original_prompt: str = Field(
        ..., description="The prompt string as it was BEFORE optimization."
    )
    optimized_prompt: str = Field(
        ..., description="The prompt string the optimizer produced."
    )

    baseline_accuracy: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-field accuracy of the ORIGINAL prompt, keyed by field "
            "name -> fraction 0.0-1.0. Only fields the eval harness "
            "actually scores appear here (currently total, line_items, "
            "subtotal, tax_total for agent #01)."
        ),
    )
    optimized_accuracy: dict[str, float] = Field(
        default_factory=dict,
        description="Same shape as baseline_accuracy, but for the optimized prompt.",
    )
    improvement: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-field delta (optimized - baseline), populated by the "
            "model validator. Positive = optimizer improved it, negative "
            "= regressed. Read this before trusting a headline number."
        ),
    )

    n_cases_used: int = Field(
        ..., ge=1, description="Number of eval cases used during compilation."
    )
    total_llm_calls: int = Field(
        ..., ge=0, description="Total LLM calls the optimizer made during compilation."
    )
    estimated_cost_usd: float = Field(
        ..., ge=0.0, description="Rough dollar cost of the compilation run."
    )
    wall_time_seconds: float = Field(
        ..., ge=0.0, description="Wall-clock duration of the compilation run."
    )

    optimizer: str = Field(
        ...,
        description=(
            "Optimizer identity used, e.g. \"dspy.GEPA(auto='light')\". "
            "Kept as a string rather than an enum so a v1.1 that adds a "
            "second optimizer (MIPROv2, BootstrapFewShotWithRandomSearch) "
            "doesn't need a schema migration."
        ),
    )
    provider: str = Field(..., description="LLM_PROVIDER used at compile time.")
    model: str = Field(..., description="Model string used at compile time.")

    @model_validator(mode="after")
    def _compute_improvement(self) -> OptimizationResult:
        """Fill `improvement` from baseline vs optimized. Any field present
        in either map gets a delta; a missing side is treated as 0.0 so
        the delta reflects what actually changed rather than a phantom
        drop from a field that simply wasn't scored on one side."""
        fields = set(self.baseline_accuracy) | set(self.optimized_accuracy)
        self.improvement = {
            f: self.optimized_accuracy.get(f, 0.0) - self.baseline_accuracy.get(f, 0.0)
            for f in fields
        }
        return self
