"""Gradio UI for the test generator.

Two input paths (file upload OR pasted code) so the reader can try
either their own module or something they type on the spot. Iteration
+ sandbox-timeout sliders on the left, generated test code + pass/fail
summary + collapsible pytest output on the right. Under
`LLM_PROVIDER=mock` the Run button returns a canned result instantly;
under a real provider it fires a ReAct loop that can take a minute
and costs real money.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import gradio as gr

try:
    from .agent import (
        DEFAULT_MAX_ITERATIONS,
        DEFAULT_SANDBOX_TIMEOUT_SECONDS,
        TestGeneratorError,
        generate_tests,
        resolve_provider,
    )
except ImportError:
    from agent import (
        DEFAULT_MAX_ITERATIONS,
        DEFAULT_SANDBOX_TIMEOUT_SECONDS,
        TestGeneratorError,
        generate_tests,
        resolve_provider,
    )


def _resolve_source_path(uploaded_file, pasted_code: str) -> Path | None:
    """Turn whichever input the user provided (file upload OR pasted
    code textarea) into an on-disk .py path the agent can read. Pasted
    code goes into a temp file; the uploaded file's own path is used
    directly."""
    if uploaded_file:
        return Path(uploaded_file)
    if pasted_code and pasted_code.strip():
        # NamedTemporaryFile because the agent needs a .py path.
        # delete=False so the file survives after the context manager
        # exits; deletion happens later at Python garbage-collect time,
        # which is fine for the duration of one generate_tests call.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(pasted_code)
            tmp_name = tmp.name
        return Path(tmp_name)
    return None


def _run_generation(
    uploaded_file, pasted_code: str, max_iterations: int, sandbox_timeout: float
) -> tuple[str, str, str, str]:
    """Wrapper the Gradio Interface calls. Returns
    (test_code, summary_md, pytest_stdout, pytest_stderr)."""
    source_path = _resolve_source_path(uploaded_file, pasted_code)
    if source_path is None:
        return (
            "",
            "**No input.** Upload a .py file or paste code into the textarea.",
            "",
            "",
        )

    start = time.perf_counter()
    try:
        result = generate_tests(
            source_path,
            max_iterations=int(max_iterations),
            sandbox_timeout_seconds=float(sandbox_timeout),
        )
    except TestGeneratorError as exc:
        return "", f"**Generation failed:** {exc.message}", "", ""

    elapsed = time.perf_counter() - start
    status_emoji = "PASSED" if result.all_passing else "FAILED"
    summary = (
        f"**{status_emoji}** {result.tests_added} tests generated across "
        f"{result.iterations_used} iteration(s) in {elapsed:.1f}s. "
        f"Final: {result.final_result.tests_passed} passed, "
        f"{result.final_result.tests_failed} failed "
        f"(exit code {result.final_result.exit_code})."
    )
    return result.test_code, summary, result.final_result.stdout, result.final_result.stderr


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks app. Kept as a function (not
    module-level) so importing this file doesn't spin up any UI state,
    important for tests."""
    provider = resolve_provider()

    with gr.Blocks(title="Test Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Python test generator (agent #08)\n"
            "Generate a passing pytest suite for any Python module. The "
            "agent iterates on failures using its own subprocess sandbox "
            "as feedback.\n\n"
            f"**Current provider:** `{provider}`. Under `mock` this returns "
            "a canned result instantly; under `openai` it fires a real ReAct "
            "loop (~$0.005-$0.02 per generation)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                uploaded = gr.File(label="Upload a .py file", file_types=[".py"])
                gr.Markdown("*or paste code below:*")
                pasted = gr.Code(label="Paste Python source", language="python", lines=15)
                iterations = gr.Slider(
                    minimum=1, maximum=10, value=DEFAULT_MAX_ITERATIONS, step=1,
                    label="Max iterations",
                )
                timeout = gr.Slider(
                    minimum=5, maximum=120, value=DEFAULT_SANDBOX_TIMEOUT_SECONDS, step=5,
                    label="Sandbox timeout (seconds)",
                )
                run_btn = gr.Button("Generate tests", variant="primary")

            with gr.Column(scale=2):
                summary_md = gr.Markdown()
                test_code_out = gr.Code(
                    label="Generated test code", language="python", lines=20
                )
                with gr.Accordion("pytest output (final run)", open=False):
                    stdout_box = gr.Code(label="stdout", lines=12)
                    stderr_box = gr.Code(label="stderr", lines=8)

        run_btn.click(
            fn=_run_generation,
            inputs=[uploaded, pasted, iterations, timeout],
            outputs=[test_code_out, summary_md, stdout_box, stderr_box],
        )

    return app


if __name__ == "__main__":
    build_ui().launch()
