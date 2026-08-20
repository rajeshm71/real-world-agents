"""Gradio UI for the receipt extractor.

Kept deliberately minimal: upload widget on the left, JSON output + cost
estimate + partial-extraction warning banner (when applicable) on the right.
That's it. If the reader wants to understand the extraction technique, they
should read `agent.py` — this file is UI glue, not the load-bearing code.

Launched via `uv run python -m agent --ui` or, when deployed to a HuggingFace
Space, as the container's entry point.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import gradio as gr

# F1.3 review fix S1: dual-mode import, same rationale as agent.py's
# schemas import -- when agent.py's `--ui` branch does `from .ui import
# build_ui` (relative, when agent is loaded as a top-level module by
# `python -m agent`), this module in turn needs an ABSOLUTE import of
# `agent` since it's a sibling top-level module in that same invocation,
# not a package member. The relative form still resolves when ui.py is
# imported as a proper package submodule (e.g. by a future test).
try:
    from .agent import (
        DEFAULT_MAX_TOKENS,
        ReceiptExtractionError,
        _guess_media_type,
        extract_receipt,
        resolve_provider,
    )
except ImportError:
    from agent import (
        DEFAULT_MAX_TOKENS,
        ReceiptExtractionError,
        _guess_media_type,
        extract_receipt,
        resolve_provider,
    )

# common.llm is the root workspace package (see agent.py's identical import
# for the full rationale) -- never a relative import.
from common.llm import resolve_model

# F1.2 review S3: `logging` import + unused `logger` removed. If we later
# need per-request logging in the UI, add `import logging` back at that
# point.

_EXAMPLES_DIR = Path(__file__).parent / "examples"


def _sample_receipt_paths() -> list[str]:
    """List example receipts (added under examples/). Returns [] if the
    directory is empty — the UI just hides the sample buttons in that case.

    Image types only -- deliberately excludes .pdf even though agent.py
    supports PDF input for openai/anthropic, because the `image` component
    below (gr.Image) only accepts image files. A PDF sample would need its
    own upload widget, not just adding ".pdf" to this suffix set."""
    if not _EXAMPLES_DIR.exists():
        return []
    return sorted(
        str(p) for p in _EXAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def _run_extraction(image_path: str | None) -> tuple[str, str, str]:
    """Wrapper the Gradio Interface calls. Returns (json_output, cost_line,
    warning_banner). All three are strings because Gradio's Textbox output
    doesn't accept None cleanly.
    """
    if not image_path:
        return "", "", "Upload a receipt image first."

    path = Path(image_path)
    image_bytes = path.read_bytes()
    media_type = _guess_media_type(path)

    start = time.perf_counter()
    try:
        result = extract_receipt(image_bytes, media_type=media_type)
    except ReceiptExtractionError as exc:
        # R5 case 3: partial extraction. If we have raw model output, show it
        # in the warning banner so the user isn't left with nothing.
        # F1.2 review S3: emoji removed (violates CLAUDE.md "no emojis unless
        # explicitly requested"). Using bold "**Extraction failed:**" instead
        # -- still visually prominent in Gradio's Markdown component.
        warning = f"**Extraction failed:** {exc.message}"
        if exc.partial is not None:
            warning += f"\n\nRaw model output:\n\n```\n{exc.partial.raw_text}\n```"
        return "", "", warning

    elapsed_ms = (time.perf_counter() - start) * 1000
    json_output = json.dumps(result.model_dump(mode="json"), indent=2, default=str)

    # Cost estimate (mock mode gives $0 — real mode estimates from the SPEC's
    # §13 per-extraction numbers). Under mock we don't have real token counts;
    # the estimate is honest about being an estimate.
    # Must match extract_receipt()'s own resolution (resolve_provider(),
    # default "openai") -- a hardcoded default here previously drifted from
    # that and made the cost line show the wrong model.
    provider = resolve_provider()
    if provider == "mock":
        cost_line = f"Extracted in {elapsed_ms:.0f}ms (mock mode — no real API call)"
    else:
        # SPEC §13: ~$0.005/extraction with Claude Sonnet-5 vision. Using the
        # dated snapshot ballpark rather than tracking real tokens per-call
        # here; if that becomes important we can wire common.pricing.cost_usd
        # to a token-tracking wrapper. F1.5 (accuracy eval) will measure real
        # cost more rigorously.
        model = resolve_model(provider) if provider in {"openai", "anthropic", "gemini"} else provider
        cost_line = f"Extracted in {elapsed_ms:.0f}ms · estimated cost ~$0.005 ({model})"

    return json_output, cost_line, ""


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks app. Kept as a function (not module-
    level) so importing this module doesn't spin up any UI state -- important
    for tests."""
    samples = _sample_receipt_paths()
    provider = resolve_provider()
    model = resolve_model(provider) if provider != "mock" else "mock"

    with gr.Blocks(title="Receipt Extractor", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Receipt / invoice → JSON\n"
            "Upload a receipt image; get clean structured JSON out. "
            "Powered by Anthropic Claude vision + [Instructor](https://python.useinstructor.com/) "
            "for schema-validated extraction.\n\n"
            f"**Model:** `{model}` · **Max tokens:** {DEFAULT_MAX_TOKENS} · "
            "[Source code](https://github.com/rajeshm71/real-world-agents/tree/main/agents/01_receipt_extractor)"
        )

        with gr.Row():
            with gr.Column():
                image = gr.Image(
                    label="Receipt image (JPEG/PNG)",
                    type="filepath",
                    sources=["upload", "clipboard"],
                )
                if samples:
                    gr.Examples(examples=samples, inputs=image, label="Sample receipts")
                extract_btn = gr.Button("Extract", variant="primary")
            with gr.Column():
                warning = gr.Markdown(visible=True)
                cost_line = gr.Markdown()
                json_output = gr.Code(
                    label="Extracted JSON",
                    language="json",
                    lines=20,
                )

        extract_btn.click(
            fn=_run_extraction,
            inputs=[image],
            outputs=[json_output, cost_line, warning],
        )

    return app


if __name__ == "__main__":
    build_ui().launch()
