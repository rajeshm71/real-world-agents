"""Gradio UI for the research crew.

Kept deliberately minimal: topic input on the left, brief sections +
sources panel on the right. If a reader wants to understand the
multi-agent technique, they should read `agent.py` -- this file is
UI glue, not the load-bearing code (agent.py's `_build_crew` is the
pedagogical anchor).

Launched via `uv run python -m agent --ui` or, when deployed to a
HuggingFace Space, as the container's entry point.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import gradio as gr

# Dual-mode import (same rationale as #01-04's ui.py).
try:
    from .agent import (
        DEFAULT_MAX_ITER,
        ResearchError,
        resolve_provider,
        run_research,
    )
    from .schemas import ResearchBrief
except ImportError:
    from agent import (
        DEFAULT_MAX_ITER,
        ResearchError,
        resolve_provider,
        run_research,
    )
    from schemas import ResearchBrief

from common.llm import resolve_model

_EXAMPLES_DIR = Path(__file__).parent / "examples"

# Curated sample topics that exercise the mock corpus's substring
# matcher (quantum / climate / ai safety) plus a fallback topic to
# demonstrate the graceful-degradation branch.
_SAMPLE_TOPICS = [
    "State of quantum computing in 2026",
    "Global climate adaptation strategies",
    "AI safety landscape for 2026",
    "Emerging market payment infrastructure",  # falls through to fallback source
]


def _render_brief_html(brief: ResearchBrief) -> str:
    """Render the ResearchBrief as an HTML block: summary panel,
    background paragraph, key-findings list, implications paragraph.
    All user-visible strings go through html.escape."""
    findings_html = "".join(
        f"<li>{html.escape(f)}</li>" for f in brief.key_findings
    )
    return (
        f"<div style='padding:0.75em; margin-bottom:0.75em; "
        f"background:#e2f0e2; border-radius:6px; "
        f"border:1px solid rgba(0,0,0,0.1); color:#111;'>"
        f"<strong>Summary</strong><br>{html.escape(brief.summary)}"
        f"</div>"
        f"<div style='margin-bottom:0.75em;'>"
        f"<strong>Background</strong><br>"
        f"<div style='margin-top:0.3em;'>{html.escape(brief.background)}</div>"
        f"</div>"
        f"<div style='margin-bottom:0.75em;'>"
        f"<strong>Key findings</strong>"
        f"<ul style='margin-top:0.3em;'>{findings_html}</ul>"
        f"</div>"
        f"<div style='margin-bottom:0.75em;'>"
        f"<strong>Implications</strong><br>"
        f"<div style='margin-top:0.3em;'>{html.escape(brief.implications)}</div>"
        f"</div>"
    )


def _render_sources_html(brief: ResearchBrief) -> str:
    """Render sources_used as an HTML block. Every source is a
    verified citation (editor's verify_source_citation tool). The
    snippet is what the editor confirmed appears verbatim in the
    source content."""
    if not brief.sources_used:
        return (
            "<div style='padding:1em; border:1px solid #ccc; "
            "border-radius:6px;'>No sources returned.</div>"
        )
    cards = []
    for i, src in enumerate(brief.sources_used, start=1):
        cards.append(
            f"<div style='padding:0.6em; margin-bottom:0.4em; "
            f"background:#f5f5f5; border-radius:5px; "
            f"border:1px solid rgba(0,0,0,0.1); color:#111;'>"
            f"<div><strong>[{i}]</strong> "
            f"<a href='{html.escape(src.url)}' target='_blank'>"
            f"{html.escape(src.title)}</a></div>"
            f"<div style='font-style:italic; margin-top:0.3em; "
            f"font-size:0.9em; opacity:0.85;'>"
            f"&ldquo;{html.escape(src.snippet)}&rdquo;</div>"
            f"</div>"
        )
    return "".join(cards)


def _run_crew(topic: str) -> tuple[str, str, str, str, str]:
    """Wrapper the Gradio button calls. Returns (status_line, brief_
    html, sources_html, full_json, warning). All strings because
    Gradio outputs don't accept None cleanly."""
    if not topic or not topic.strip():
        return "", "", "", "", "Enter a research topic first."

    start = time.perf_counter()
    try:
        brief = run_research(topic)
    except ResearchError as exc:
        warning = f"**Research failed:** {exc.message}"
        if exc.partial is not None:
            if exc.partial.stage:
                warning += f"\n\n*Failed at stage: {exc.partial.stage}*"
            if exc.partial.partial_output:
                warning += (
                    f"\n\n*Partial output:*\n\n```\n"
                    f"{exc.partial.partial_output[:500]}...\n```"
                )
        return "", "", "", "", warning

    elapsed_ms = (time.perf_counter() - start) * 1000
    resolved_provider = resolve_provider()
    if resolved_provider == "mock":
        provider_bit = "mock mode -- no crew invoked, no API call"
    else:
        model = resolve_model(resolved_provider)
        provider_bit = f"`{model}` via `{resolved_provider}` -- 3-agent Sequential crew"

    status_line = (
        f"*Ran in {elapsed_ms:.0f}ms ({provider_bit}). "
        f"Topic: **{html.escape(brief.topic)}**. "
        f"Word count: {brief.word_count}. "
        f"{len(brief.sources_used)} source(s) verified.*"
    )
    brief_html = _render_brief_html(brief)
    sources_html = _render_sources_html(brief)
    full_json = json.dumps(brief.model_dump(mode="json"), indent=2)
    return status_line, brief_html, sources_html, full_json, ""


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks app. Kept as a function
    (not module-level) so importing this module doesn't spin up any
    UI state -- important for tests."""
    default_provider = resolve_provider()
    default_model = (
        resolve_model(default_provider) if default_provider != "mock" else "mock"
    )

    with gr.Blocks(title="Research Crew", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Multi-agent research crew -- topic to structured brief\n"
            "Enter a research topic; three specialized agents (researcher, "
            "writer, editor) collaborate to produce a fact-checked brief "
            "with verified source citations. Powered by [CrewAI](https://"
            "docs.crewai.com/)'s Sequential Process. Every source in the "
            "final brief was verified via a verbatim substring check "
            "before landing in the output.\n\n"
            f"**Provider:** `{default_provider}` -- **model:** "
            f"`{default_model}` -- **max iterations per agent:** "
            f"{DEFAULT_MAX_ITER}. [Source code](https://github.com/"
            "rajeshm71/real-world-agents/tree/main/agents/05_research_crew)"
        )

        with gr.Row():
            with gr.Column(scale=1):
                topic_box = gr.Textbox(
                    label="Research topic",
                    placeholder=(
                        "e.g. 'State of quantum computing in 2026' or "
                        "'AI safety landscape'."
                    ),
                    lines=3,
                )
                sample_dropdown = gr.Dropdown(
                    choices=_SAMPLE_TOPICS,
                    label="Or pick a sample topic",
                    value=None,
                )
                sample_dropdown.change(
                    fn=lambda t: t or "",
                    inputs=[sample_dropdown],
                    outputs=[topic_box],
                )
                research_btn = gr.Button("Run research crew", variant="primary")

            with gr.Column(scale=2):
                warning = gr.Markdown(visible=True)
                status_line = gr.Markdown()
                brief_html = gr.HTML(label="Research brief")
                sources_html = gr.HTML(label="Verified sources")
                full_json = gr.Code(
                    label="Full brief as JSON (copy or download)",
                    language="json",
                    lines=12,
                )

        research_btn.click(
            fn=_run_crew,
            inputs=[topic_box],
            outputs=[status_line, brief_html, sources_html, full_json, warning],
        )

    return app


if __name__ == "__main__":
    build_ui().launch()
