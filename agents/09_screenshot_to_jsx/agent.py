"""Screenshot to React JSX reconstructor: agent #09 of real-world-agents.

Technique demonstrated: **multimodal structured extraction** applied to
a non-JSON output domain (React component code). Same Instructor +
vision pipeline as agent #01 (receipt extraction), but the schema's
load-bearing field is a code string that has to be syntactically valid
JSX AND coherent with the declared `component_name`.

Why this technique for this use case: producing a full React component
from a screenshot needs the model to make MANY decisions (component
name, layout structure, styling approach, imports, placeholder handling
for unclear assets). Free-form text output would give inconsistent
shape a caller can't reliably consume. A Pydantic schema locks the
output shape so downstream tools (a Gradio UI, a batch script, a code
review pipeline) all get the same structure regardless of provider.

This is v0.1 (Phase A): one-shot generation only. No rendering, no
comparison, no iteration. Phase B (future release) will add Playwright
+ Babel Standalone to render the produced JSX back to a screenshot for
side-by-side comparison. Phase C will add an LLM-as-judge iteration
loop where the model critiques its own output. See README's
"Verification status" section for what's covered today.

Real error handling per R5 (three cases):
1. Zero-byte / malformed screenshot -> ScreenshotToJsxError before any
   LLM call.
2. Missing media_type or unrecognized value -> validated at the public
   API boundary.
3. LLM API failure during Instructor's client.create() -> six-branch
   translator (same class-name -> status -> message priority as agents
   #02-#08).

Provider stance: multi-provider (openai/anthropic/gemini) via
Instructor's `from_provider()` unified interface. Anthropic default
since Claude's vision output for markup is meaningfully better in
practice; every provider is equally supported at the code level.
"""

from __future__ import annotations

import argparse
import base64
import functools
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Dual-mode import: relative for tests-as-package, absolute for
# `python -m agent` from inside this dir. See agent #01's identical
# pattern for the full rationale.
try:
    from .schemas import ReconstructedComponent
except ImportError:
    from schemas import ReconstructedComponent

# common.llm is the root workspace package -- never a relative import.
from common.llm import resolve_model

# --- Provider + constants --------------------------------------------------

SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini")
_INSTRUCTOR_PROVIDER_PREFIX = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google",
}

# Anthropic default: Claude's vision output for structured markup is
# meaningfully better than OpenAI/Gemini's in practice. Every provider
# is still equally supported at the code level.
_DEFAULT_PROVIDER = "anthropic"

DEFAULT_STYLING: Literal["tailwind", "inline_styles"] = "tailwind"
DEFAULT_MAX_TOKENS = 4096  # JSX + notes for a full page runs longer than a receipt
DEFAULT_MAX_RETRIES = 3

# Anthropic caps vision inputs at ~20MB per image; OpenAI + Gemini are
# similar. Reject BEFORE base64-encoding so a caller passing a 50MB
# screenshot gets a clear domain error rather than a cryptic
# "content too large" surfacing from deep inside Instructor.
MAX_SCREENSHOT_BYTES = 20 * 1024 * 1024  # 20 MiB

_VALID_MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")

_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"


# --- Error type ------------------------------------------------------------


@dataclass
class ReconstructionAttempt:
    """Partial state attached to ScreenshotToJsxError when reconstruction
    fails partway through. `raw_text` is whatever the model returned
    before validation rejected it (helps a caller debug a
    schema-mismatch failure)."""

    raw_text: str
    validation_errors: list[str]


class ScreenshotToJsxError(Exception):
    """Raised on any user-facing reconstruction failure: bad screenshot,
    validation failure after retries, or API error during generation."""

    def __init__(self, message: str, partial: ReconstructionAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Provider resolution ---------------------------------------------------


def resolve_provider() -> str:
    """LLM_PROVIDER env var, defaulting to Anthropic (see module
    docstring for why). "mock" is handled by the caller (reconstruct),
    not here."""
    provider = os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER).lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}."
        )
    return provider


@functools.lru_cache(maxsize=1)
def _load_prompt() -> str:
    """Read the system prompt from disk once, cache for subsequent
    calls. `reconstruct()` runs this on every non-mock invocation --
    without the cache, a batch tool processing 100 screenshots reads
    the same file 100 times."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


# --- Public API ------------------------------------------------------------


def reconstruct(
    screenshot_bytes: bytes,
    *,
    media_type: str = "image/png",
    styling: Literal["tailwind", "inline_styles"] = DEFAULT_STYLING,
    provider: str | None = None,
    model: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    _client=None,
) -> ReconstructedComponent:
    """Reconstruct a screenshot as a React JSX component.

    Args: screenshot_bytes (raw image bytes); media_type (image/png |
    image/jpeg | image/webp | image/gif); styling ("tailwind" default
    | "inline_styles"); provider ("anthropic" default / "openai" /
    "gemini" / "mock"); model (override); max_retries (Instructor
    validation-retry cap, default 3); _client (test escape hatch).

    Returns a validated ReconstructedComponent. Raises
    ScreenshotToJsxError on any user-facing failure (bad input,
    oversized, unknown media_type/styling, retries exhausted, API
    error) -- one exception class at the boundary so a caller wrapping
    the call in a single `except` catches everything."""
    # Fail-fast validation BEFORE any base64 encoding or LLM call.
    # All four boundary checks raise the same ScreenshotToJsxError so a
    # caller wrapping the call in a single `except` catches them all.
    if not screenshot_bytes:
        raise ScreenshotToJsxError(
            "screenshot_bytes is empty. Pass a non-zero-byte image."
        )
    if len(screenshot_bytes) > MAX_SCREENSHOT_BYTES:
        raise ScreenshotToJsxError(
            f"screenshot_bytes is {len(screenshot_bytes) / 1024 / 1024:.1f} MB, "
            f"over the {MAX_SCREENSHOT_BYTES / 1024 / 1024:.0f} MB cap. "
            "Vision providers reject inputs above ~20 MB; downscale the "
            "screenshot before passing it in."
        )
    if media_type not in _VALID_MEDIA_TYPES:
        raise ScreenshotToJsxError(
            f"Unknown media_type: {media_type!r}. "
            f"Expected one of {_VALID_MEDIA_TYPES}."
        )
    if styling not in ("tailwind", "inline_styles"):
        raise ScreenshotToJsxError(
            f"Unknown styling: {styling!r}. "
            "Expected 'tailwind' or 'inline_styles'."
        )

    resolved_provider = (provider or resolve_provider()).lower()

    if resolved_provider == "mock":
        return _mock_reconstruction(
            screenshot_bytes=screenshot_bytes, styling=styling
        )

    resolved_model = model or resolve_model(resolved_provider)
    client = _client if _client is not None else _get_real_client(
        resolved_provider, resolved_model
    )
    prompt = _load_prompt() + f"\n\nUse styling_approach = {styling!r}."
    b64_data = base64.b64encode(screenshot_bytes).decode("ascii")
    content = _build_content(resolved_provider, media_type, b64_data, prompt)

    try:
        result = client.create(
            max_tokens=DEFAULT_MAX_TOKENS,
            max_retries=max_retries,
            response_model=ReconstructedComponent,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:
        raise _translate_api_error(exc) from exc

    return result


# --- Provider-branching content construction ------------------------------


def _build_content(
    provider: str, media_type: str, b64_data: str, prompt: str
) -> list[dict]:
    """Build the provider-specific image+text content-block list for one
    Instructor `.create()` call. OpenAI/Anthropic image formats are
    verified against long-stable public API shapes (same #01 uses).
    The Gemini branch relies on Instructor's provider-agnostic message
    normalization (accepting OpenAI-shaped blocks); not independently
    verified against a real Gemini API call in this sandbox."""
    if provider == "anthropic":
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_data,
                },
            },
            {"type": "text", "text": prompt},
        ]
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
        },
        {"type": "text", "text": prompt},
    ]


# --- Client factory --------------------------------------------------------


def _get_real_client(provider: str, model: str):
    """Build an Instructor client via `from_provider()`. Same lazy-import
    pattern agent #01 uses: mock-mode CI doesn't need instructor OR any
    provider SDK installed."""
    import instructor

    instructor_prefix = _INSTRUCTOR_PROVIDER_PREFIX.get(provider)
    if instructor_prefix is None:
        raise ValueError(f"No Instructor provider mapping for {provider!r}")
    return instructor.from_provider(f"{instructor_prefix}/{model}")


# --- Error translation (R5 case 3) -----------------------------------------


def _translate_api_error(exc: Exception) -> ScreenshotToJsxError:
    """Turn an Instructor / provider SDK exception into a user-facing
    ScreenshotToJsxError. Priority: ValidationError first (attaches
    partial), then the standard 6-branch (class-name -> status-code ->
    message-string -> generic) matching agents #02-#08."""
    exc_class_name = type(exc).__name__.lower()
    message_lower = str(exc).lower()
    status = getattr(exc, "status_code", None)

    if "validationerror" in exc_class_name:
        return _build_validation_error(exc)

    if "ratelimiterror" in exc_class_name:
        return _rate_limit_error()
    if "authenticationerror" in exc_class_name or "apikeyerror" in exc_class_name:
        return _auth_error()
    if status == 429:
        return _rate_limit_error()
    if status == 401:
        return _auth_error()
    if "rate limit" in message_lower or "overloaded" in message_lower:
        return _rate_limit_error()
    if "authentication" in message_lower or "api key" in message_lower:
        return _auth_error()

    return ScreenshotToJsxError(
        f"Reconstruction failed: {type(exc).__name__}: {exc}. "
        "This is an unexpected error -- check the agent logs. "
        "The agent's partial state (if any) was not preserved."
    )


def _build_validation_error(exc: Exception) -> ScreenshotToJsxError:
    """Instructor's ValidationError carries `raw_output` (the model's
    raw text before validation) and `errors()` (the Pydantic complaints).
    Attach both as `partial` so a caller inspecting the failure can see
    what the model actually produced."""
    raw = getattr(exc, "raw_output", None) or str(exc)
    errors_attr = getattr(exc, "errors", None)
    if callable(errors_attr):
        errors = errors_attr()
    else:
        errors = [str(exc)]
    error_strs = [str(e) for e in errors] if isinstance(errors, list) else [str(errors)]
    return ScreenshotToJsxError(
        "The model returned JSX that didn't match the expected schema "
        "even after Instructor's retries. Raw output is attached; "
        "check .partial.raw_text to see what was produced.",
        partial=ReconstructionAttempt(
            raw_text=str(raw), validation_errors=error_strs
        ),
    )


def _rate_limit_error() -> ScreenshotToJsxError:
    """Shared message so every path that reaches this case can't drift
    apart. No auto-retry: vision calls are already the expensive part
    of this agent; silent retries could multiply the bill."""
    return ScreenshotToJsxError(
        "The provider is temporarily rate-limited or overloaded. "
        "Wait a minute and try again."
    )


def _auth_error() -> ScreenshotToJsxError:
    return ScreenshotToJsxError(
        "Authentication failed: check that your API key is set correctly "
        "for the provider you selected (ANTHROPIC_API_KEY / OPENAI_API_KEY "
        "/ GEMINI_API_KEY). See .env.example at the repo root."
    )


# --- Mock mode -------------------------------------------------------------


_MOCK_JSX_TEMPLATES = {
    "tailwind": (
        'function MockLandingPage() {\n'
        '  return (\n'
        '    <div className="min-h-screen bg-white">\n'
        '      <header className="p-8 bg-gray-100"><h1>Mock Landing Page</h1></header>\n'
        '      <main className="p-8"><p>Reconstructed by mock mode. No real API call.</p></main>\n'
        '      <footer className="p-8 bg-gray-100"><p>Footer</p></footer>\n'
        "    </div>\n"
        "  );\n"
        "}\n"
    ),
    "inline_styles": (
        "function MockLandingPage() {\n"
        "  return (\n"
        "    <div style={{minHeight: '100vh', background: '#fff'}}>\n"
        "      <header style={{padding: '2rem', background: '#f3f4f6'}}><h1>Mock Landing Page</h1></header>\n"
        "      <main style={{padding: '2rem'}}><p>Reconstructed by mock mode. No real API call.</p></main>\n"
        "      <footer style={{padding: '2rem', background: '#f3f4f6'}}><p>Footer</p></footer>\n"
        "    </div>\n"
        "  );\n"
        "}\n"
    ),
}


def _mock_reconstruction(
    *, screenshot_bytes: bytes, styling: str
) -> ReconstructedComponent:
    """Deterministic canned ReconstructedComponent. Input byte count is
    echoed into `notes` so tests prove the mock saw its input (anti-
    refactor guard convention agent #01 uses)."""
    return ReconstructedComponent(
        component_name="MockLandingPage",
        jsx_code=_MOCK_JSX_TEMPLATES[styling],
        imports=[],
        styling_approach=styling,  # type: ignore[arg-type]
        notes=(
            f"[MOCK] Generated from {len(screenshot_bytes)} bytes of "
            f"screenshot input. Set LLM_PROVIDER to a real provider for "
            "actual reconstruction."
        ),
        detected_sections=["header", "main", "footer"],
    )


# --- CLI entry point -------------------------------------------------------


def main() -> int:
    """CLI: `uv run python -m agent path/to/screenshot.png`. Prints
    a summary + writes full ReconstructedComponent JSON to
    `last_run.json` next to this file (gitignored per
    agents/*/last_run.json)."""
    parser = argparse.ArgumentParser(
        prog="screenshot-to-jsx",
        description=(
            "Reconstruct a full-page screenshot as a React JSX component. "
            "Set LLM_PROVIDER=mock for a canned demo, or supply "
            "ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY for a "
            "real generation."
        ),
    )
    parser.add_argument(
        "screenshot",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a screenshot image (PNG/JPEG/WebP/GIF). Omit when using --ui.",
    )
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI instead of CLI.")
    parser.add_argument(
        "--provider",
        choices=(*SUPPORTED_PROVIDERS, "mock"),
        default=None,
        help="Override LLM_PROVIDER for this run.",
    )
    parser.add_argument("--model", default=None, help="Override the resolved model.")
    parser.add_argument(
        "--styling",
        choices=("tailwind", "inline_styles"),
        default=DEFAULT_STYLING,
        help="Styling approach (default tailwind).",
    )
    args = parser.parse_args()

    if args.ui:
        try:
            from .ui import build_ui
        except ImportError:
            from ui import build_ui
        build_ui().launch()
        return 0

    if args.screenshot is None:
        parser.error("screenshot path is required (or pass --ui to launch the web interface)")
        return 2  # unreachable

    if not args.screenshot.exists():
        print(f"error: screenshot not found: {args.screenshot}", file=sys.stderr)
        return 2

    screenshot_bytes = args.screenshot.read_bytes()
    media_type = _guess_media_type(args.screenshot)

    start = time.perf_counter()
    try:
        result = reconstruct(
            screenshot_bytes,
            media_type=media_type,
            styling=args.styling,
            provider=args.provider,
            model=args.model,
        )
    except ScreenshotToJsxError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.partial is not None:
            print(f"raw model output:\n{exc.partial.raw_text}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    out_path = Path(__file__).parent / "last_run.json"
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    print(f"Component name:     {result.component_name}")
    print(f"Styling:            {result.styling_approach}")
    print(f"Detected sections:  {', '.join(result.detected_sections)}")
    print(f"Imports needed:     {', '.join(result.imports) or '(none)'}")
    print(f"Wall time:          {elapsed:.1f}s")
    if result.notes:
        print()
        print(f"Notes:              {result.notes}")
    print()
    print(f"JSX code written to {out_path} (also full JSON there).")
    return 0


def _guess_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/png")


if __name__ == "__main__":
    raise SystemExit(main())
