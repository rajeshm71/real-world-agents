"""Gradio UI for the CSV chat agent.

Kept deliberately minimal: CSV upload + question textbox on the left,
answer + winning SQL + attempts history + result table on the right.
If a reader wants to understand the technique, they should read
`agent.py` -- this file is UI glue, not the load-bearing code
(agent.py's `_build_graph` is the pedagogical anchor).

Launched via `uv run python -m agent --ui` or, when deployed to a
HuggingFace Space, as the container's entry point.
"""

from __future__ import annotations

import html
import time
from pathlib import Path

import gradio as gr

# Dual-mode import (same rationale as agents #01/#02 ui.py).
try:
    from .agent import (
        DEFAULT_MAX_ATTEMPTS,
        SUPPORTED_PROVIDERS,
        ChatCsvError,
        chat_with_csv,
        resolve_provider,
    )
    from .schemas import CsvAnswer
except ImportError:
    from agent import (
        DEFAULT_MAX_ATTEMPTS,
        SUPPORTED_PROVIDERS,
        ChatCsvError,
        chat_with_csv,
        resolve_provider,
    )
    from schemas import CsvAnswer

from common.llm import resolve_model

_EXAMPLES_DIR = Path(__file__).parent / "examples"


def _sample_csv_paths() -> list[str]:
    """List example CSVs under examples/. Returns [] if the directory
    is empty -- the UI just hides the sample buttons in that case."""
    if not _EXAMPLES_DIR.exists():
        return []
    return sorted(
        str(p) for p in _EXAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    )


def _render_attempts_html(answer: CsvAnswer) -> str:
    """Render the SQL-attempts history as an HTML block. Only shown
    when there were multiple attempts (retries happened) -- for the
    common single-attempt success path, the full history would be
    noise (just the winning SQL, which is already shown elsewhere).
    """
    if len(answer.attempts) <= 1:
        return ""

    rows = []
    for i, attempt in enumerate(answer.attempts, start=1):
        is_winner = attempt.error is None
        colour = "#e2f0e2" if is_winner else "#ffdddd"
        outcome = (
            f"succeeded ({attempt.row_count} rows)"
            if is_winner
            else f"error: {html.escape(attempt.error or '')}"
        )
        rows.append(
            f"<div style='padding:0.5em; margin-bottom:0.4em; background:{colour}; "
            f"border-radius:5px; border:1px solid rgba(0,0,0,0.1); color:#111;'>"
            f"<div><strong>Attempt {i}</strong> -- {outcome}</div>"
            f"<div style='font-family:monospace; margin-top:0.3em; font-size:0.9em; "
            f"white-space:pre-wrap;'>{html.escape(attempt.sql)}</div>"
            f"</div>"
        )
    header = (
        f"<div style='margin-bottom:0.5em;'><strong>Retry history</strong> "
        f"({len(answer.attempts)} attempts before landing a working query):</div>"
    )
    return header + "".join(rows)


def _render_result_table(answer: CsvAnswer) -> list[list]:
    """Convert answer.result_sample (list of dicts) into a Gradio
    Dataframe payload: [[header_row], [row1], [row2], ...]. Gradio
    Dataframe expects a list-of-lists with the first row as headers
    when `headers=True` -- actually, Gradio wants headers separately."""
    if not answer.result_sample:
        return []
    return [list(row.values()) for row in answer.result_sample]


def _result_table_headers(answer: CsvAnswer) -> list[str]:
    if not answer.result_sample:
        return []
    return list(answer.result_sample[0].keys())


def _run_chat(
    csv_path: str | None,
    question: str,
    provider_choice: str,
) -> tuple[str, str, str, list[list], list[str], str]:
    """Wrapper the Gradio button calls. Returns (answer_md, sql_md,
    attempts_html, table_data, table_headers, warning). All strings/
    lists because Gradio outputs don't accept None cleanly.

    `provider_choice` is a Radio selection; the empty string "" means
    "use whatever LLM_PROVIDER resolves to."
    """
    if not csv_path:
        return "", "", "", [], [], "Upload a CSV file first."
    if not question.strip():
        return "", "", "", [], [], "Type a question about the data."

    provider_override = provider_choice or None
    start = time.perf_counter()
    try:
        answer = chat_with_csv(csv_path, question, provider=provider_override)
    except ChatCsvError as exc:
        warning = f"**Chat failed:** {exc.message}"
        if exc.partial is not None and exc.partial.attempts:
            warning += "\n\nAttempts before giving up:"
            for i, attempt in enumerate(exc.partial.attempts, start=1):
                err_bit = f" (error: {attempt.error})" if attempt.error else ""
                warning += f"\n\n**[{i}]** `{attempt.sql}`{err_bit}"
        return "", "", "", [], [], warning

    elapsed_ms = (time.perf_counter() - start) * 1000

    resolved_provider = (provider_override or resolve_provider()).lower()
    if resolved_provider == "mock":
        provider_bit = "mock mode -- no real API call"
    else:
        model = resolve_model(resolved_provider)
        provider_bit = f"`{model}` via `{resolved_provider}`"

    answer_md = f"**Answer:** {answer.answer}\n\n*Ran in {elapsed_ms:.0f}ms ({provider_bit}). Total rows returned: {answer.row_count}.*"
    sql_md = f"```sql\n{answer.sql_used}\n```"
    attempts_html = _render_attempts_html(answer)
    table_data = _render_result_table(answer)
    table_headers = _result_table_headers(answer)
    return answer_md, sql_md, attempts_html, table_data, table_headers, ""


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks app. Kept as a function
    (not module-level) so importing this module doesn't spin up any
    UI state -- important for tests."""
    samples = _sample_csv_paths()
    default_provider = resolve_provider()
    default_model = resolve_model(default_provider) if default_provider != "mock" else "mock"
    provider_choices = ["", *SUPPORTED_PROVIDERS, "mock"]

    with gr.Blocks(title="CSV Chat", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# CSV chat -- ask your data questions in plain English\n"
            "Upload a CSV, type a question, get a plain-English answer. "
            "Under the hood: a LangGraph state machine translates your "
            "question to SQL, runs it against an in-memory SQLite "
            "database, retries on SQL errors, then explains the result. "
            "Powered by "
            "[OpenAI](https://openai.com/) / "
            "[Anthropic](https://www.anthropic.com/) / "
            "[Gemini](https://ai.google.dev/) -- pick your provider below.\n\n"
            f"**Default provider:** `{default_provider}` -- **default model:** `{default_model}` -- "
            f"**max SQL retries:** {DEFAULT_MAX_ATTEMPTS}. "
            "[Source code](https://github.com/rajeshm71/real-world-agents/tree/main/agents/03_csv_chat)"
        )

        with gr.Row():
            with gr.Column(scale=1):
                csv_file = gr.File(
                    label="Your CSV",
                    file_types=[".csv"],
                    type="filepath",
                )
                question_box = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. 'How many rows are there?' or 'What's the average of column X grouped by Y?'",
                    lines=2,
                )
                provider_radio = gr.Radio(
                    choices=provider_choices,
                    value="",
                    label="Provider override (empty = use LLM_PROVIDER env var)",
                    info="'mock' returns a deterministic canned answer with no API call.",
                )
                if samples:
                    gr.Examples(examples=samples, inputs=csv_file, label="Sample CSVs")
                ask_btn = gr.Button("Ask", variant="primary")

            with gr.Column(scale=2):
                warning = gr.Markdown(visible=True)
                answer_md = gr.Markdown()
                sql_md = gr.Markdown(label="SQL used")
                attempts_html = gr.HTML()
                result_table = gr.Dataframe(
                    label="Result rows (first 10)",
                    interactive=False,
                    wrap=True,
                )

        # Gradio Dataframe takes value + headers separately; we pass
        # both as outputs.
        def _run_and_split(csv_path, question, provider):
            answer_md_val, sql_md_val, attempts, data, headers, warn = _run_chat(
                csv_path, question, provider
            )
            # Gradio Dataframe accepts a dict {"data": [[...]], "headers": [...]}.
            df_payload = {"data": data, "headers": headers} if data else None
            return answer_md_val, sql_md_val, attempts, df_payload, warn

        ask_btn.click(
            fn=_run_and_split,
            inputs=[csv_file, question_box, provider_radio],
            outputs=[answer_md, sql_md, attempts_html, result_table, warning],
        )

    return app


if __name__ == "__main__":
    build_ui().launch()
