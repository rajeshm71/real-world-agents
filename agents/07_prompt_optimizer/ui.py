"""Gradio UI for the prompt optimizer.

Kept deliberately minimal: provider + effort selectors on the left, Run
button, then a per-field before/after accuracy table + side-by-side
prompt diff on the right. Under `LLM_PROVIDER=mock` the Run button
returns the canned result instantly; under a real provider it fires a
GEPA compilation, which takes minutes and costs real money.
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

try:
    from .agent import (
        OPTIMIZER_EFFORT_LEVELS,
        PromptOptimizerError,
        optimize_prompt,
        resolve_provider,
    )
except ImportError:
    from agent import (
        OPTIMIZER_EFFORT_LEVELS,
        PromptOptimizerError,
        optimize_prompt,
        resolve_provider,
    )


def _run_optimization(provider: str, effort: str) -> tuple[str, str, str, str]:
    """Wrapper the Gradio Interface calls. Returns
    (accuracy_table_md, before_prompt, after_prompt, status_line)."""
    start = time.perf_counter()
    try:
        result = optimize_prompt(provider=provider, optimizer_effort=effort)
    except PromptOptimizerError as exc:
        return "", "", "", f"**Optimization failed:** {exc.message}"

    elapsed = time.perf_counter() - start
    table_md = _format_accuracy_table(
        baseline=result.baseline_accuracy,
        optimized=result.optimized_accuracy,
        improvement=result.improvement,
    )
    cost_note = (
        f"Compiled in {elapsed:.1f}s using **{result.optimizer}** on "
        f"**{result.provider}/{result.model}** "
        f"across {result.n_cases_used} cases "
        f"(estimated cost ~${result.estimated_cost_usd:.2f})."
    )
    return table_md, result.original_prompt, result.optimized_prompt, cost_note


def _format_accuracy_table(
    *,
    baseline: dict[str, float],
    optimized: dict[str, float],
    improvement: dict[str, float],
) -> str:
    """Render the before/after per-field accuracy as a small markdown
    table. Matches the style used in agent #01's Accuracy section."""
    rows = ["| Field | Baseline | Optimized | Delta |", "|---|---|---|---|"]
    for field in sorted(set(baseline) | set(optimized)):
        b = baseline.get(field, 0.0)
        o = optimized.get(field, 0.0)
        d = improvement.get(field, 0.0)
        rows.append(f"| `{field}` | {b:.1%} | {o:.1%} | {d:+.1%} |")
    return "\n".join(rows)


def _last_run_json_path() -> Path:
    return Path(__file__).parent / "last_run.json"


def _load_last_run_json() -> str:
    """Return the last-run JSON if present, else a placeholder. Shown
    in a code block below the Run button so a reader can inspect the
    full OptimizationResult without leaving the UI."""
    path = _last_run_json_path()
    if not path.exists():
        return "// last_run.json will appear here after the first CLI run"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"// error reading last_run.json: {exc}"


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks app. Kept as a function (not
    module-level) so importing this file doesn't spin up UI state --
    important for tests."""
    provider = resolve_provider()
    is_mock = provider == "mock"

    with gr.Blocks(title="Prompt Optimizer", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Prompt optimizer for agent #01\n"
            "Run DSPy's GEPA optimizer against agent #01's receipt-extractor "
            "prompt + 20-case CORD eval set. Under `LLM_PROVIDER=mock` this "
            "returns a canned result instantly; with a real provider key it "
            "runs a real compile pass "
            f"(~$0.30 light / ~$1.50 medium / ~$5 heavy).\n\n"
            f"**Current provider:** `{provider}`."
        )

        with gr.Row():
            with gr.Column(scale=1):
                provider_choice = gr.Radio(
                    choices=["openai", "anthropic", "gemini", "mock"],
                    value=provider if provider in {"openai", "anthropic", "gemini", "mock"} else "mock",
                    label="Provider (override for this run)",
                )
                effort = gr.Radio(
                    choices=list(OPTIMIZER_EFFORT_LEVELS),
                    value="light",
                    label="Optimizer effort",
                )
                run_btn = gr.Button("Run optimization", variant="primary")
                if not is_mock:
                    gr.Markdown(
                        "*Real compile runs cost real money -- see cost estimates above.*"
                    )
                status = gr.Markdown()

            with gr.Column(scale=2):
                gr.Markdown("### Per-field accuracy")
                accuracy_table = gr.Markdown()
                with gr.Row():
                    before_box = gr.Code(
                        label="Original prompt (baseline)",
                        language=None,
                        lines=15,
                    )
                    after_box = gr.Code(
                        label="Optimized prompt",
                        language=None,
                        lines=15,
                    )

        with gr.Accordion("Last CLI run (full JSON)", open=False):
            last_run_box = gr.Code(
                value=_load_last_run_json(),
                language="json",
                lines=20,
            )
            refresh_btn = gr.Button("Refresh from last_run.json")

        run_btn.click(
            fn=_run_optimization,
            inputs=[provider_choice, effort],
            outputs=[accuracy_table, before_box, after_box, status],
        )
        refresh_btn.click(fn=_load_last_run_json, outputs=last_run_box)

    return app


if __name__ == "__main__":
    build_ui().launch()
