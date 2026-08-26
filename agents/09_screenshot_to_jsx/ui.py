"""Gradio UI for the screenshot -> JSX reconstructor.

Kept deliberately minimal for Phase A: image upload on the left, JSX
code + metadata on the right. Phase B will add a side-by-side "input
vs. rendered" panel; Phase C will add an iteration-history accordion.
Those are shipped in later versions -- see the README's "Verification
status" table for what's actually covered today.
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

try:
    from .agent import (
        DEFAULT_STYLING,
        ScreenshotToJsxError,
        _guess_media_type,
        reconstruct,
        resolve_provider,
    )
except ImportError:
    from agent import (
        DEFAULT_STYLING,
        ScreenshotToJsxError,
        _guess_media_type,
        reconstruct,
        resolve_provider,
    )

_EXAMPLES_DIR = Path(__file__).parent / "examples"


def _sample_screenshot_paths() -> list[str]:
    """List example screenshots under examples/. Returns [] if empty."""
    if not _EXAMPLES_DIR.exists():
        return []
    return sorted(
        str(p) for p in _EXAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )


def _run_reconstruction(
    image_path: str | None, styling: str
) -> tuple[str, str, str, str, str, str]:
    """Wrapper the Gradio Interface calls. Returns
    (jsx_code, component_name, sections_md, imports_md, notes_md, status_line)."""
    if not image_path:
        return "", "", "", "", "", "Upload a screenshot or pick a sample first."

    path = Path(image_path)
    screenshot_bytes = path.read_bytes()
    media_type = _guess_media_type(path)

    start = time.perf_counter()
    try:
        result = reconstruct(
            screenshot_bytes, media_type=media_type, styling=styling
        )
    except ScreenshotToJsxError as exc:
        warning = f"**Reconstruction failed:** {exc.message}"
        if exc.partial is not None:
            warning += (
                f"\n\nRaw model output (validation failed):\n\n"
                f"```\n{exc.partial.raw_text[:2000]}\n```"
            )
        return "", "", "", "", "", warning

    elapsed_ms = (time.perf_counter() - start) * 1000

    sections_md = (
        "**Detected sections:** "
        + " ".join(f"`{s}`" for s in result.detected_sections)
        if result.detected_sections
        else "*(model reported no top-level sections)*"
    )
    imports_md = (
        "**npm packages to install:** "
        + " ".join(f"`{i}`" for i in result.imports)
        if result.imports
        else "*(no additional npm packages needed beyond `react`)*"
    )
    notes_md = f"**Notes from the model:**\n\n{result.notes}" if result.notes else ""

    status_line = (
        f"Reconstructed in {elapsed_ms:.0f}ms as **{result.component_name}** "
        f"({result.styling_approach})."
    )

    return (
        result.jsx_code,
        result.component_name,
        sections_md,
        imports_md,
        notes_md,
        status_line,
    )


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks app. Kept as a function (not
    module-level) so importing this module doesn't spin up any UI state."""
    samples = _sample_screenshot_paths()
    provider = resolve_provider()

    with gr.Blocks(title="Screenshot to JSX", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Screenshot -> React JSX (agent #09)\n"
            "Upload a full-page UI screenshot; get back a React functional "
            "component that reconstructs the visible structure. Uses "
            "Instructor + vision for multi-provider structured extraction "
            "(same technique as agent #01, different output domain).\n\n"
            f"**Current provider:** `{provider}`. Under `mock` this returns "
            "a canned MockLandingPage instantly; under a real provider "
            "(Anthropic default) it fires a real vision call "
            "(~$0.01-0.03 per generation at Sonnet-5 pricing).\n\n"
            "**Phase A (v0.1):** one-shot generation only. "
            "Rendering + iteration are queued for v0.2 / v1.0 -- see the "
            "README's Verification status section."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(
                    label="Full-page screenshot (PNG/JPG/WebP)",
                    type="filepath",
                    sources=["upload", "clipboard"],
                )
                if samples:
                    gr.Examples(examples=samples, inputs=image, label="Sample screenshots")
                styling_choice = gr.Radio(
                    choices=["tailwind", "inline_styles"],
                    value=DEFAULT_STYLING,
                    label="Styling approach",
                )
                run_btn = gr.Button("Reconstruct", variant="primary")
                status = gr.Markdown()

            with gr.Column(scale=2):
                component_name_box = gr.Textbox(
                    label="Component name", interactive=False
                )
                sections_box = gr.Markdown()
                imports_box = gr.Markdown()
                jsx_output = gr.Code(
                    label="Generated JSX",
                    language="javascript",
                    lines=25,
                )
                notes_box = gr.Markdown()

        run_btn.click(
            fn=_run_reconstruction,
            inputs=[image, styling_choice],
            outputs=[
                jsx_output,
                component_name_box,
                sections_box,
                imports_box,
                notes_box,
                status,
            ],
        )

    return app


if __name__ == "__main__":
    build_ui().launch()
