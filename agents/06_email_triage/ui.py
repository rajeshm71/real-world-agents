"""Gradio UI for the email-triage agent.

Kept deliberately minimal: .eml upload + optional
known-contacts / important-domains config on the left, triage
decision with category/priority/action pills + reasoning +
suggested reply on the right. If a reader wants to understand
the type-safe agent technique, they should read `agent.py` --
this file is UI glue, not the load-bearing code
(agent.py's `_build_agent` is the pedagogical anchor).
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import gradio as gr

# Dual-mode import (same rationale as #01-05's ui.py).
try:
    from .agent import (
        EmailTriageError,
        resolve_provider,
        triage_email,
    )
    from .schemas import EmailTriage, TriageDeps
except ImportError:
    from agent import (
        EmailTriageError,
        resolve_provider,
        triage_email,
    )
    from schemas import EmailTriage, TriageDeps

from common.llm import resolve_model

_EXAMPLES_DIR = Path(__file__).parent / "examples"

# Category / priority / action colour pills.
_CATEGORY_COLOUR = {
    "work": "#e6f0ff",
    "personal": "#f0e6ff",
    "newsletter": "#e2f0e2",
    "spam": "#ffdddd",
    "important": "#fff3cd",
    "notification": "#f0f0f0",
    "other": "#f5f5f5",
}
_PRIORITY_COLOUR = {
    "urgent": "#ffdddd",
    "high": "#fff3cd",
    "medium": "#e2f0e2",
    "low": "#f0f0f0",
}
_ACTION_COLOUR = {
    "respond_now": "#ffdddd",
    "respond_later": "#fff3cd",
    "delegate": "#e6f0ff",
    "archive": "#e2f0e2",
    "ignore": "#f0f0f0",
}


def _sample_eml_paths() -> list[str]:
    if not _EXAMPLES_DIR.exists():
        return []
    return sorted(
        str(p) for p in _EXAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".eml"
    )


def _pill(label: str, colour: str) -> str:
    return (
        f"<span style='display:inline-block; padding:0.25em 0.6em; "
        f"margin-right:0.4em; background:{colour}; border-radius:12px; "
        f"border:1px solid rgba(0,0,0,0.1); font-size:0.9em; color:#111;'>"
        f"{html.escape(label)}</span>"
    )


def _render_triage_html(triage: EmailTriage) -> str:
    """Render pills + reasoning + snippet + suggested reply."""
    pills = (
        _pill(triage.category, _CATEGORY_COLOUR.get(triage.category, "#f5f5f5"))
        + _pill(triage.priority, _PRIORITY_COLOUR.get(triage.priority, "#f5f5f5"))
        + _pill(triage.action, _ACTION_COLOUR.get(triage.action, "#f5f5f5"))
    )
    reply_block = ""
    if triage.suggested_reply is not None:
        reply_block = (
            f"<div style='margin-top:0.75em; padding:0.6em; background:#f5f5f5; "
            f"border-left:3px solid #4a90e2; color:#111;'>"
            f"<strong>Suggested reply:</strong><br>"
            f"<div style='margin-top:0.3em;'>{html.escape(triage.suggested_reply)}</div>"
            f"</div>"
        )
    return (
        f"<div style='padding:0.5em 0;'>{pills}</div>"
        f"<div style='margin-bottom:0.5em;'>"
        f"<strong>Reasoning:</strong> {html.escape(triage.reasoning)}"
        f"</div>"
        f"<div style='margin-bottom:0.5em; padding:0.5em; "
        f"background:#fafafa; border-left:3px solid #ccc; "
        f"font-style:italic; color:#111;'>"
        f"&ldquo;{html.escape(triage.key_snippet)}&rdquo;"
        f"</div>"
        f"{reply_block}"
    )


def _parse_config_list(s: str) -> list[str]:
    """Parse a comma-or-newline-separated string into a list."""
    if not s:
        return []
    parts = s.replace(",", "\n").split("\n")
    return [p.strip() for p in parts if p.strip()]


def _run_triage(
    file_path: str | None,
    contacts_text: str,
    domains_text: str,
) -> tuple[str, str, str, str]:
    """Wrapper the Gradio button calls. Returns (status_line,
    triage_html, full_json, warning)."""
    if not file_path:
        return "", "", "", "Upload a .eml file first."

    eml_bytes = Path(file_path).read_bytes()
    deps = TriageDeps(
        known_contacts=_parse_config_list(contacts_text),
        important_domains=_parse_config_list(domains_text),
    )

    start = time.perf_counter()
    try:
        triage = triage_email(eml_bytes, deps=deps)
    except EmailTriageError as exc:
        warning = f"**Triage failed:** {exc.message}"
        if exc.partial is not None:
            warning += f"\n\n*Failed at stage: {exc.partial.stage}*"
        return "", "", "", warning

    elapsed_ms = (time.perf_counter() - start) * 1000
    resolved_provider = resolve_provider()
    if resolved_provider == "mock":
        provider_bit = "mock mode -- no PydanticAI Agent invoked"
    else:
        model = resolve_model(resolved_provider)
        provider_bit = f"`{model}` via `{resolved_provider}`"

    status_line = (
        f"*Triaged in {elapsed_ms:.0f}ms ({provider_bit}). "
        f"Deps: {len(deps.known_contacts)} known contact(s), "
        f"{len(deps.important_domains)} important domain(s).*"
    )
    triage_html = _render_triage_html(triage)
    full_json = json.dumps(triage.model_dump(mode="json"), indent=2)
    return status_line, triage_html, full_json, ""


def build_ui() -> gr.Blocks:
    samples = _sample_eml_paths()
    default_provider = resolve_provider()
    default_model = (
        resolve_model(default_provider) if default_provider != "mock" else "mock"
    )

    with gr.Blocks(title="Email Triage", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Type-safe email triage\n"
            "Upload a `.eml` file; get a structured triage decision "
            "(category, priority, action, suggested reply) grounded in "
            "your own contact list + important domains. Powered by "
            "[PydanticAI](https://ai.pydantic.dev/)'s typed "
            "`Agent[TriageDeps, EmailTriage]` -- dependency injection "
            "means the model's tools have real user context, not "
            "fabricated guesses.\n\n"
            f"**Provider:** `{default_provider}` -- **model:** "
            f"`{default_model}`. [Source code](https://github.com/"
            "rajeshm71/real-world-agents/tree/main/agents/06_email_triage)"
        )

        with gr.Row():
            with gr.Column(scale=1):
                eml_file = gr.File(
                    label="Upload .eml",
                    file_types=[".eml"],
                    type="filepath",
                )
                contacts_text = gr.Textbox(
                    label="Known contacts (one per line, or comma-separated)",
                    placeholder="alice@example.com\nBob Smith\nboss@mycompany.com",
                    lines=3,
                )
                domains_text = gr.Textbox(
                    label="Important domains",
                    placeholder="@mycompany.com\n@client-name.com",
                    lines=2,
                )
                if samples:
                    gr.Examples(examples=samples, inputs=eml_file, label="Sample emails")
                triage_btn = gr.Button("Triage", variant="primary")

            with gr.Column(scale=2):
                warning = gr.Markdown(visible=True)
                status_line = gr.Markdown()
                triage_html = gr.HTML(label="Triage decision")
                full_json = gr.Code(
                    label="Full triage as JSON",
                    language="json",
                    lines=10,
                )

        triage_btn.click(
            fn=_run_triage,
            inputs=[eml_file, contacts_text, domains_text],
            outputs=[status_line, triage_html, full_json, warning],
        )

    return app


if __name__ == "__main__":
    build_ui().launch()
