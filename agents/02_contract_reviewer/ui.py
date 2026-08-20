"""Gradio UI for the contract reviewer.

Kept deliberately minimal: upload widget on the left (PDF or plain text),
per-flag output + JSON download + partial-attempt warning banner on the
right. Colour-coded by severity so a non-lawyer can eyeball the review
without reading every clause. If a reader wants to understand the
technique, they should read `agent.py` -- this file is UI glue, not the
load-bearing code (agent.py's `_run_review_loop` is the pedagogical
anchor).

Launched via `uv run python -m agent --ui` or, when deployed to a
HuggingFace Space, as the container's entry point.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import gradio as gr

# Dual-mode import, same rationale as agent.py's schemas import -- when
# agent.py's `--ui` branch does `from .ui import build_ui` (relative,
# when agent is loaded as a top-level module by `python -m agent`), this
# module in turn needs an ABSOLUTE import of `agent` since it's a sibling
# top-level module in that same invocation, not a package member. The
# relative form still resolves when ui.py is imported as a proper package
# submodule (e.g. by a test).
try:
    from .agent import (
        SUPPORTED_PROVIDERS,
        ContractReviewError,
        resolve_provider,
        review_contract,
    )
    from .schemas import ContractReview
except ImportError:
    from agent import (
        SUPPORTED_PROVIDERS,
        ContractReviewError,
        resolve_provider,
        review_contract,
    )
    from schemas import ContractReview

# common.llm is the root workspace package (see agent.py's identical
# import for the full rationale) -- never a relative import.
from common.llm import resolve_model

_EXAMPLES_DIR = Path(__file__).parent / "examples"

# Severity colours -- background tints, chosen for legibility in both
# light and dark Gradio themes.
_SEVERITY_COLOUR = {
    "high": "#ffdddd",    # soft red
    "medium": "#fff3cd",  # soft amber
    "low": "#e2f0e2",     # soft green
}


def _sample_contract_paths() -> list[str]:
    """List example contracts under examples/. Returns [] if the
    directory is empty -- the UI just hides the sample buttons in that
    case. Accepts .pdf and .txt (both are valid `review_contract` inputs)."""
    if not _EXAMPLES_DIR.exists():
        return []
    return sorted(
        str(p) for p in _EXAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}
    )


def _render_flags_html(review: ContractReview) -> str:
    """Render `review.flags` as an HTML block: one severity-tinted card
    per flag. Keeps the Gradio output compact and scannable for a
    non-lawyer, without dropping into a full table framework.

    Returns a placeholder message if the review has zero flags -- an
    empty flags list IS a valid outcome ("no clauses worth flagging"),
    not an error, and the UI shouldn't render as blank in that case.
    """
    if not review.flags:
        return (
            "<div style='padding:1em; border:1px solid #ccc; border-radius:6px;'>"
            "<strong>No flagged clauses.</strong> "
            "The reviewer didn't find any clauses in this contract worth "
            "flagging for a non-lawyer. Read the summary and overall_risk "
            "in the JSON below before signing anyway."
            "</div>"
        )

    cards = []
    for flag in review.flags:
        colour = _SEVERITY_COLOUR.get(flag.severity, "#f0f0f0")
        page_bit = f" (page {flag.page_number})" if flag.page_number is not None else ""
        rec_bit = (
            f"<div><strong>Recommendation:</strong> {html.escape(flag.recommendation)}</div>"
            if flag.recommendation
            else ""
        )
        cards.append(
            f"<div style='padding:0.75em; margin-bottom:0.5em; background:{colour}; "
            f"border-radius:6px; border:1px solid rgba(0,0,0,0.1); color:#111;'>"
            f"<div><strong>{flag.category}</strong> "
            f"<span style='opacity:0.7;'>({flag.severity}{page_bit})</span></div>"
            f"<div style='margin:0.4em 0; font-style:italic;'>"
            f"&ldquo;{html.escape(flag.excerpt)}&rdquo;</div>"
            f"<div>{html.escape(flag.explanation)}</div>"
            f"{rec_bit}"
            f"</div>"
        )
    return "".join(cards)


def _run_review(file_path: str | None, provider_choice: str) -> tuple[str, str, str, str]:
    """Wrapper the Gradio button calls. Returns (flags_html, summary_line,
    json_output, warning_banner). All four are strings because Gradio's
    outputs don't accept None cleanly.

    `provider_choice` is a Radio selection; the empty string "" means
    "use whatever LLM_PROVIDER resolves to" (no per-click override).
    """
    if not file_path:
        return "", "", "", "Upload a contract file first (.pdf or .txt)."

    path = Path(file_path)
    input_bytes = path.read_bytes()

    provider_override = provider_choice or None
    start = time.perf_counter()
    try:
        review = review_contract(input_bytes, provider=provider_override)
    except ContractReviewError as exc:
        warning = f"**Review failed:** {exc.message}"
        if exc.partial is not None:
            warning += f"\n\nRaw model output:\n\n```\n{exc.partial.raw_text}\n```"
        return "", "", "", warning

    elapsed_ms = (time.perf_counter() - start) * 1000
    json_output = json.dumps(review.model_dump(mode="json"), indent=2)

    resolved_provider = (provider_override or resolve_provider()).lower()
    if resolved_provider == "mock":
        summary_line = (
            f"Reviewed in {elapsed_ms:.0f}ms (mock mode -- no real API call). "
            f"**{review.contract_type}** -- overall risk: **{review.overall_risk}**. {review.summary}"
        )
    else:
        model = resolve_model(resolved_provider)
        summary_line = (
            f"Reviewed in {elapsed_ms:.0f}ms with `{model}`. "
            f"**{review.contract_type}** -- overall risk: **{review.overall_risk}**. {review.summary}"
        )

    flags_html = _render_flags_html(review)
    return flags_html, summary_line, json_output, ""


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks app. Kept as a function (not
    module-level) so importing this module doesn't spin up any UI state
    -- important for tests."""
    samples = _sample_contract_paths()
    default_provider = resolve_provider()
    default_model = resolve_model(default_provider) if default_provider != "mock" else "mock"
    provider_choices = ["", *SUPPORTED_PROVIDERS, "mock"]

    with gr.Blocks(title="Contract Reviewer", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Contract review -- flag risky clauses before you sign\n"
            "Upload a PDF or plain-text contract (NDA, MSA, SaaS ToS, "
            "employment, vendor SOW, etc.). Get flagged clauses colour-coded "
            "by severity, plain-English explanations, and a JSON output you "
            "can pipe into other tools. "
            "Powered by a hand-rolled JSON-validate-retry loop over "
            "[OpenAI](https://openai.com/) / "
            "[Anthropic](https://www.anthropic.com/) / "
            "[Gemini](https://ai.google.dev/) -- pick your provider below.\n\n"
            f"**Default provider:** `{default_provider}` -- **default model:** `{default_model}`. "
            "[Source code](https://github.com/rajeshm71/real-world-agents/tree/main/agents/02_contract_reviewer)"
        )

        with gr.Row():
            with gr.Column():
                contract_file = gr.File(
                    label="Contract (PDF or plain text)",
                    file_types=[".pdf", ".txt"],
                    type="filepath",
                )
                provider_radio = gr.Radio(
                    choices=provider_choices,
                    value="",
                    label="Provider override (empty = use LLM_PROVIDER env var)",
                    info="'mock' returns a deterministic canned review with no API call.",
                )
                if samples:
                    gr.Examples(examples=samples, inputs=contract_file, label="Sample contracts")
                review_btn = gr.Button("Review contract", variant="primary")

            with gr.Column():
                warning = gr.Markdown(visible=True)
                summary_line = gr.Markdown()
                flags_html = gr.HTML(label="Flagged clauses")
                json_output = gr.Code(
                    label="Full review as JSON (copy or download)",
                    language="json",
                    lines=15,
                )

        review_btn.click(
            fn=_run_review,
            inputs=[contract_file, provider_radio],
            outputs=[flags_html, summary_line, json_output, warning],
        )

    return app


if __name__ == "__main__":
    build_ui().launch()
