"""Gradio UI for the meeting-notes agent.

Kept deliberately minimal: notes textarea on the left, summary +
action items with priority pills + participants + key decisions on
the right. If a reader wants to understand the ReAct technique,
they should read `agent.py` -- this file is UI glue, not the
load-bearing code (agent.py's `_build_agent` is the pedagogical
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

# Dual-mode import (same rationale as agents #01/#02/#03).
try:
    from .agent import (
        DEFAULT_MAX_TURNS,
        MeetingNotesError,
        extract_action_items,
        resolve_provider,
    )
    from .schemas import MeetingSummary
except ImportError:
    from agent import (
        DEFAULT_MAX_TURNS,
        MeetingNotesError,
        extract_action_items,
        resolve_provider,
    )
    from schemas import MeetingSummary

from common.llm import resolve_model

_EXAMPLES_DIR = Path(__file__).parent / "examples"

# Priority colour pills, chosen for legibility in both light and dark
# Gradio themes.
_PRIORITY_COLOUR = {
    "high": "#ffdddd",    # soft red
    "medium": "#fff3cd",  # soft amber
    "low": "#e2f0e2",     # soft green
}


def _sample_notes_paths() -> list[str]:
    """List example meeting notes under examples/. Returns [] if the
    directory is empty -- the UI hides the sample buttons in that case."""
    if not _EXAMPLES_DIR.exists():
        return []
    return sorted(
        str(p) for p in _EXAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    )


def _load_sample_text(path: str | None) -> str:
    """Read a sample file into the notes textarea. Called via a
    Gradio event when a user clicks a sample."""
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8")


def _render_action_items_html(summary: MeetingSummary) -> str:
    """Render `summary.action_items` as an HTML block: one priority-
    tinted card per item. Placeholder message if empty -- zero
    action items IS a valid outcome (info-share meeting) and the UI
    shouldn't render as blank in that case."""
    if not summary.action_items:
        return (
            "<div style='padding:1em; border:1px solid #ccc; border-radius:6px;'>"
            "<strong>No action items.</strong> "
            "The meeting produced discussion but no commitments this "
            "agent could identify. Read the summary + key decisions for what "
            "changed."
            "</div>"
        )

    cards = []
    for item in summary.action_items:
        colour = _PRIORITY_COLOUR.get(item.priority, "#f0f0f0")
        owner_bit = f" -- <strong>{html.escape(item.owner)}</strong>" if item.owner else ""
        due_bit = (
            f" <span style='opacity:0.7;'>(due: {html.escape(item.due_hint)})</span>"
            if item.due_hint
            else ""
        )
        cards.append(
            f"<div style='padding:0.75em; margin-bottom:0.5em; background:{colour}; "
            f"border-radius:6px; border:1px solid rgba(0,0,0,0.1); color:#111;'>"
            f"<div><strong>[{item.priority}]</strong>{owner_bit}{due_bit}</div>"
            f"<div style='margin:0.4em 0;'>{html.escape(item.description)}</div>"
            f"<div style='font-style:italic; opacity:0.8; font-size:0.9em;'>"
            f"&ldquo;{html.escape(item.context_excerpt)}&rdquo;</div>"
            f"</div>"
        )
    return "".join(cards)


def _render_participants_and_decisions(summary: MeetingSummary) -> str:
    """Render participants + key decisions as a compact HTML block."""
    parts = []
    if summary.participants:
        parts.append(
            "<div><strong>Participants:</strong> "
            + ", ".join(html.escape(p) for p in summary.participants)
            + "</div>"
        )
    if summary.key_decisions:
        decisions_html = "".join(
            f"<li>{html.escape(d)}</li>" for d in summary.key_decisions
        )
        parts.append(
            f"<div style='margin-top:0.5em;'><strong>Key decisions:</strong>"
            f"<ul style='margin-top:0.25em;'>{decisions_html}</ul></div>"
        )
    if not parts:
        return ""
    return "<div style='padding:0.5em 0;'>" + "".join(parts) + "</div>"


def _run_extraction(notes: str) -> tuple[str, str, str, str, str]:
    """Wrapper the Gradio button calls. Returns (summary_line,
    items_html, participants_html, full_json, warning). All strings
    because Gradio outputs don't accept None cleanly."""
    if not notes or not notes.strip():
        return "", "", "", "", "Paste some meeting notes first."

    start = time.perf_counter()
    try:
        summary = extract_action_items(notes)
    except MeetingNotesError as exc:
        warning = f"**Extraction failed:** {exc.message}"
        if exc.partial is not None and exc.partial.turns_used:
            warning += f"\n\n*Turns used before giving up: {exc.partial.turns_used}*"
        return "", "", "", "", warning

    elapsed_ms = (time.perf_counter() - start) * 1000
    resolved_provider = resolve_provider()
    if resolved_provider == "mock":
        provider_bit = "mock mode -- no real API call"
    else:
        model = resolve_model(resolved_provider)
        provider_bit = f"`{model}` via `{resolved_provider}`"

    topic_bit = f"**{summary.meeting_topic}** -- " if summary.meeting_topic else ""
    summary_line = (
        f"{topic_bit}{summary.overall_summary}\n\n"
        f"*Extracted in {elapsed_ms:.0f}ms ({provider_bit}). "
        f"{len(summary.action_items)} action item(s), "
        f"{len(summary.key_decisions)} key decision(s).*"
    )
    items_html = _render_action_items_html(summary)
    participants_html = _render_participants_and_decisions(summary)
    full_json = json.dumps(summary.model_dump(mode="json"), indent=2)
    return summary_line, items_html, participants_html, full_json, ""


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks app. Kept as a function
    (not module-level) so importing this module doesn't spin up any
    UI state -- important for tests."""
    samples = _sample_notes_paths()
    default_provider = resolve_provider()
    default_model = (
        resolve_model(default_provider) if default_provider != "mock" else "mock"
    )

    with gr.Blocks(title="Meeting Notes -> Action Items", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Meeting notes -> action items\n"
            "Paste meeting notes; get a structured list of action items "
            "(with owner + priority + verbatim excerpt), key decisions, "
            "and participants. Under the hood: the OpenAI Agents SDK runs "
            "a ReAct loop -- the model interleaves reasoning with grounded "
            "tool calls (extract_speakers, extract_dates, score_urgency, "
            "verify_excerpt) to prevent fabricated attendees and paraphrased "
            "quotes.\n\n"
            f"**Provider:** `{default_provider}` -- **model:** `{default_model}` -- "
            f"**max ReAct turns:** {DEFAULT_MAX_TURNS}. "
            "[Source code](https://github.com/rajeshm71/real-world-agents/tree/main/agents/04_meeting_notes)"
        )

        with gr.Row():
            with gr.Column(scale=1):
                notes_box = gr.Textbox(
                    label="Meeting notes",
                    placeholder=(
                        "Paste your meeting notes here. Transcript-style "
                        "(Alice: ...) or narrative-style (Alice said ...) "
                        "both work."
                    ),
                    lines=18,
                )
                if samples:
                    sample_dropdown = gr.Dropdown(
                        choices=samples,
                        label="Load a sample",
                        value=None,
                    )
                    sample_dropdown.change(
                        fn=_load_sample_text,
                        inputs=[sample_dropdown],
                        outputs=[notes_box],
                    )
                extract_btn = gr.Button("Extract action items", variant="primary")

            with gr.Column(scale=2):
                warning = gr.Markdown(visible=True)
                summary_line = gr.Markdown()
                items_html = gr.HTML(label="Action items")
                participants_html = gr.HTML()
                full_json = gr.Code(
                    label="Full extraction as JSON (copy or download)",
                    language="json",
                    lines=12,
                )

        extract_btn.click(
            fn=_run_extraction,
            inputs=[notes_box],
            outputs=[summary_line, items_html, participants_html, full_json, warning],
        )

    return app


if __name__ == "__main__":
    build_ui().launch()
