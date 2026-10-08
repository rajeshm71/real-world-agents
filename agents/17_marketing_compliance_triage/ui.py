"""Gradio UI for the marketing compliance triage agent.

Kept deliberately minimal: copy-text + context + provider inputs on the
left, active lenses + severity-colored findings + summary + overall
risk on the right. If a reader wants to understand the technique, they
should read `agent.py` -- this file is UI glue, not the load-bearing
code (agent.py's `_build_graph` is the pedagogical anchor).

Launched via `uv run python -m agent --ui`.
"""

from __future__ import annotations

import html
import time

import gradio as gr

# Dual-mode import (same rationale as agents #01/#02/#03 ui.py).
try:
    from .agent import (
        DEFAULT_MAX_RETRIES,
        SUPPORTED_PROVIDERS,
        TriageError,
        resolve_provider,
        triage_marketing_copy,
    )
    from .schemas import ComplianceReport
except ImportError:
    from agent import (
        DEFAULT_MAX_RETRIES,
        SUPPORTED_PROVIDERS,
        TriageError,
        resolve_provider,
        triage_marketing_copy,
    )
    from schemas import ComplianceReport

from common.llm import resolve_model

_SEVERITY_COLOR = {
    "high": "#ffdddd",
    "medium": "#fff4cc",
    "low": "#e2f0e2",
}

_LENS_LABEL = {
    "health_wellness": "Health / Wellness",
    "financial_earnings": "Financial / Earnings",
    "childrens_privacy": "Children's Privacy",
    "environmental": "Environmental",
}


def _render_findings_html(report: ComplianceReport) -> str:
    """Render findings as severity-colored HTML blocks, mirroring
    agent #03's _render_attempts_html pattern."""
    if not report.findings:
        return "<div><em>No findings -- no active lenses triggered.</em></div>"

    rows = []
    for finding in report.findings:
        colour = _SEVERITY_COLOR.get(finding.severity, "#eeeeee")
        lens_label = _LENS_LABEL.get(finding.lens, finding.lens)
        phrases = ", ".join(f'"{html.escape(p)}"' for p in finding.flagged_phrases)
        rows.append(
            f"<div style='padding:0.6em; margin-bottom:0.5em; background:{colour}; "
            f"border-radius:5px; border:1px solid rgba(0,0,0,0.1); color:#111;'>"
            f"<div><strong>{html.escape(lens_label)}</strong> -- severity: {finding.severity}</div>"
            f"<div style='margin-top:0.3em;'><strong>Flagged:</strong> {phrases or '(none)'}</div>"
            f"<div style='margin-top:0.3em;'><strong>Concern:</strong> {html.escape(finding.concern)}</div>"
            f"<div style='margin-top:0.3em;'><strong>Suggested fix:</strong> {html.escape(finding.suggested_fix)}</div>"
            f"</div>"
        )
    return "".join(rows)


def _run_triage(
    copy_text: str,
    context: str,
    provider_choice: str,
) -> tuple[str, str, str]:
    """Wrapper the Gradio button calls. Returns (summary_md,
    findings_html, warning)."""
    if not copy_text.strip():
        return "", "", "Paste draft marketing copy first."

    provider_override = provider_choice or None
    start = time.perf_counter()
    try:
        report = triage_marketing_copy(
            copy_text, context or None, provider=provider_override
        )
    except TriageError as exc:
        warning = f"**Triage failed:** {exc.message}"
        if exc.partial is not None:
            warning += f"\n\n**Last raw output:**\n```\n{exc.partial.raw_text}\n```"
        return "", "", warning

    elapsed_ms = (time.perf_counter() - start) * 1000

    resolved_provider = (provider_override or resolve_provider()).lower()
    if resolved_provider == "mock":
        provider_bit = "mock mode -- no real API call"
    else:
        model = resolve_model(resolved_provider)
        provider_bit = f"`{model}` via `{resolved_provider}`"

    lens_labels = ", ".join(_LENS_LABEL.get(lens, lens) for lens in report.active_lenses)
    lens_bit = lens_labels if lens_labels else "none"

    summary_md = (
        f"**Overall risk: {report.overall_risk_level.upper()}**\n\n"
        f"{report.summary}\n\n"
        f"*Active lenses: {lens_bit}. Ran in {elapsed_ms:.0f}ms ({provider_bit}).*"
    )
    findings_html = _render_findings_html(report)
    return summary_md, findings_html, ""


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks app. Kept as a function (not
    module-level) so importing this module doesn't spin up any UI
    state -- important for tests."""
    default_provider = resolve_provider()
    default_model = resolve_model(default_provider) if default_provider != "mock" else "mock"
    provider_choices = ["", *SUPPORTED_PROVIDERS, "mock"]

    with gr.Blocks(title="Marketing Compliance Triage", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Marketing copy compliance triage\n"
            "Paste draft ad/marketing copy, get back which regulatory lenses "
            "apply (health/wellness, financial/earnings, children's privacy, "
            "environmental) and specific findings for each. Under the hood: "
            "a LangGraph classify node decides which lenses apply, then "
            "dispatches ONLY the relevant specialist reviewers -- in "
            "parallel when more than one applies -- before merging their "
            "findings into one report. Powered by "
            "[OpenAI](https://openai.com/) / "
            "[Anthropic](https://www.anthropic.com/) / "
            "[Gemini](https://ai.google.dev/) / "
            "[Ollama](https://ollama.com/) -- pick your provider below.\n\n"
            f"**Default provider:** `{default_provider}` -- **default model:** `{default_model}` -- "
            f"**max retries per node:** {DEFAULT_MAX_RETRIES}. "
            "[Source code](https://github.com/rajeshm71/real-world-agents/tree/main/agents/17_marketing_compliance_triage)\n\n"
            "*This is a first-pass triage tool, not a substitute for actual "
            "legal/compliance review.*"
        )

        with gr.Row():
            with gr.Column(scale=1):
                copy_box = gr.Textbox(
                    label="Draft marketing copy",
                    placeholder="Paste the ad/marketing copy to triage...",
                    lines=6,
                )
                context_box = gr.Textbox(
                    label="Product/industry context (optional)",
                    placeholder="e.g. 'supplement brand' or 'fintech app'",
                    lines=1,
                )
                provider_radio = gr.Radio(
                    choices=provider_choices,
                    value="",
                    label="Provider override (empty = use LLM_PROVIDER env var)",
                    info="'mock' returns a deterministic canned report with no API call.",
                )
                triage_btn = gr.Button("Triage", variant="primary")

            with gr.Column(scale=2):
                warning = gr.Markdown(visible=True)
                summary_md = gr.Markdown()
                findings_html = gr.HTML()

        triage_btn.click(
            fn=_run_triage,
            inputs=[copy_box, context_box, provider_radio],
            outputs=[summary_md, findings_html, warning],
        )

    return app


if __name__ == "__main__":
    build_ui().launch()
