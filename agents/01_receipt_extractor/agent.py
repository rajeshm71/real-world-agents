"""Receipt/invoice extractor agent: agent #01 of real-world-agents.

Technique demonstrated: **structured extraction with Instructor + vision, across
multiple providers**. Pydantic schema (see schemas.py) is passed as
`response_model` to Instructor, which validates and retries the LLM's output
until it conforms. The model sees the receipt image directly (vision) plus the
prompt at prompts/extract.txt, and fills in the schema.

Why this technique for this use case: receipts are structured documents that
happen to arrive as images. Instructor collapses "prompt the model + parse JSON
+ validate against Pydantic + retry on schema failure" into one call. Without
Instructor you'd hand-roll all four steps. This is exactly what the pattern is
designed for.

Real error handling (R5 in CONTRIBUTING.md's hard rules): three concrete
failure modes are handled explicitly (see extract_receipt below):
  1. Malformed/unreadable image -> ReceiptExtractionError with friendly message
  2. Rate limit / transient API failure -> exponential backoff via Instructor's
     max_retries, then surface the error rather than looping forever
  3. Partial extraction (validation fails after all retries) -> raise
     ReceiptExtractionError with the raw model output attached so the caller
     can show it as a warning banner

Provider + model are fully user-configurable (this project is OSS -- every
user brings their own key(s), no provider should be hardcoded):
  - LLM_PROVIDER="mock" -> _mock_extraction() (no network, no key, CI default)
  - LLM_PROVIDER="openai" (default) -> gpt-4.1-mini-2025-04-14 (vision-capable,
    cheap). Override the model with OPENAI_DEFAULT_MODEL, e.g. bump up to
    gpt-5.4-mini-2026-03-17 for harder documents.
  - LLM_PROVIDER="anthropic" -> claude-sonnet-5 (override: ANTHROPIC_DEFAULT_MODEL)
  - LLM_PROVIDER="gemini" -> gemini-2.5-flash (override: GEMINI_DEFAULT_MODEL)
Provider/model resolution goes through common/llm.py's `resolve_model()` so
the same env-var contract applies across the whole repo, not just this agent.
Uses `instructor.from_provider("<provider>/<model>")`'s unified interface --
one `.create()` call path works across all three real providers; only the
image content-block format below differs per provider (OpenAI's `image_url`
data-URI format and Anthropic's `image`+`source` format are both long-stable
and verified against public docs; the Gemini path is NOT independently
verified against a real API call).

Accepts images (JPEG/PNG/WebP/GIF) for all three providers, plus PDF for
openai/anthropic via Instructor's cross-provider `PDF` multimodal helper
(gemini raises a clear error for PDF input rather than mishandling it
silently). PDF support is implemented but not yet verified against a real
API call.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Dual-mode import. `.schemas` (relative) resolves when this module is
# loaded AS a submodule of the 01_receipt_extractor package (how
# tests/test_smoke.py imports it, via importlib since the dir name starts
# with a digit). The bare `schemas` (absolute) resolves when this file is
# run directly via `python -m agent` from inside the agent's own directory
# (the documented CLI invocation in README.md) -- in that mode Python has no
# parent-package context at all, so relative imports raise ImportError. Both
# invocation styles must work; this try/except is the standard pattern for a
# script that's also importable as a package member.
try:
    from .schemas import (  # LineItem imported here (not inside _mock_extraction) to keep it a module-level import
        ExtractedReceipt,
        LineItem,
    )
except ImportError:
    from schemas import ExtractedReceipt, LineItem

# common.llm is the root workspace package (declared as a dependency in this
# agent's pyproject.toml via [tool.uv.sources] workspace = true), so this
# import works in both invocation modes without a try/except -- it's never
# a relative import in the first place.
from common.llm import resolve_model

# --- Provider + model resolution ---

SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini", "ollama")
# Maps our LLM_PROVIDER values to Instructor's from_provider() prefix.
# Only "gemini" differs (Instructor calls it "google"). "ollama" is
# not a from_provider prefix -- we route through instructor.from_openai
# with a custom base_url instead (see _get_real_client).
_INSTRUCTOR_PROVIDER_PREFIX = {"openai": "openai", "anthropic": "anthropic", "gemini": "google"}

# Per-provider model defaults. Only ollama differs from resolve_model's
# catalog default (gemma4:e4b) -- that model's vision returns all-nulls
# on CORD receipts (verified against agent #07 as well). qwen2.5vl:7b
# handles receipt OCR reliably at the same 8 GB VRAM budget.
_DEFAULT_MODEL_BY_PROVIDER = {"ollama": "qwen2.5vl:7b"}

DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_RETRIES = 3  # covers rate-limit blips + one schema-validation retry

_PROMPT_PATH = Path(__file__).parent / "prompts" / "extract.txt"


def resolve_provider() -> str:
    """LLM_PROVIDER env var, defaulting to "openai" per project convention
    (OpenAI is the sensible default; every provider is equally supported).
    "mock" is handled by the caller (extract_receipt), not here.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}."
        )
    return provider


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# --- Error type ---


@dataclass
class ExtractionAttempt:
    """What we got back before validation failed. Attached to
    ReceiptExtractionError so the UI can show the raw output as a warning
    banner rather than dropping it silently."""

    raw_text: str
    validation_errors: list[str]


class ReceiptExtractionError(Exception):
    """Raised on any user-facing extraction failure (bad image, API failure,
    schema validation failure). Includes a user-friendly `message` and,
    when relevant, the raw model output via `partial`."""

    def __init__(self, message: str, partial: ExtractionAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Public API ---


def extract_receipt(
    image_bytes: bytes,
    *,
    media_type: str = "image/jpeg",
    provider: str | None = None,
    model: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    _client=None,  # test-injection escape hatch; production callers should not pass this
) -> ExtractedReceipt:
    """Extract structured receipt data from an image.

    Args:
        image_bytes: raw bytes of a JPEG/PNG receipt image or a PDF.
        media_type: MIME type of the input (e.g. "image/jpeg", "image/png",
            "application/pdf"). PDF is only supported for provider "openai"
            or "anthropic" -- "gemini" raises ReceiptExtractionError.
        provider: "openai" (default) / "anthropic" / "gemini" / "mock".
            Defaults to the LLM_PROVIDER env var if not passed (see
            resolve_provider()).
        model: model ID for the resolved provider. Defaults to that
            provider's DEFAULT_MODELS entry, overridable per-provider via
            env var (see common.llm.resolve_model()).
        max_retries: how many Instructor retries on schema-validation failures
            or transient API errors. Default 3 is a sane blend of "handle
            rate-limit blips" and "don't loop forever on a genuinely
            unextractable image."
        _client: injected Instructor client for tests. Production callers
            should leave this None; the client is built from provider/model.

    Returns:
        A validated ExtractedReceipt.

    Raises:
        ReceiptExtractionError: on any of the three R5-required failure modes.
    """
    resolved_provider = (provider or resolve_provider()).lower()
    if resolved_provider == "mock":
        # Bypass Instructor entirely under mock -- see _mock_extraction docstring
        # for why (we don't want tests to depend on Instructor's internals).
        return _mock_extraction(image_bytes)

    if media_type == "application/pdf" and resolved_provider == "gemini":
        # Also enforced inside _build_content(); checked here first to avoid
        # resolving a model / building a client just to throw it away.
        raise _pdf_unsupported_for_gemini_error()

    resolved_model = (
        model
        or _DEFAULT_MODEL_BY_PROVIDER.get(resolved_provider)
        or resolve_model(resolved_provider)
    )
    client = _client if _client is not None else _get_real_client(resolved_provider, resolved_model)
    prompt = _load_prompt()
    b64_data = base64.b64encode(image_bytes).decode("ascii")
    content = _build_content(resolved_provider, media_type, b64_data, prompt)

    # Instructor's unified .create() handles retries on schema validation
    # across every provider (from_provider()'s whole point). Provider API
    # errors (rate limit / auth / 400) surface as exceptions we catch and
    # translate below.
    # Ollama needs more output headroom -- a complex receipt (many
    # line items, or non-English text) pushes the JSON payload past
    # the 2k default. Verified: cord_00 (10+ items) hits 4096; 8192
    # comfortably fits every CORD case tested. Cloud providers do
    # fine at the default; keep them unchanged.
    max_tokens = 8192 if resolved_provider == "ollama" else DEFAULT_MAX_TOKENS
    create_kwargs: dict = {
        "model": resolved_model,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
        "response_model": ExtractedReceipt,
        "messages": [{"role": "user", "content": content}],
    }
    try:
        # instructor.from_provider() bakes the model into the client, so
        # `model=` is normally optional on .create(). But instructor.from_openai
        # (used for the Ollama path) does NOT bake it -- pass it explicitly
        # so both paths work.
        result = client.create(**create_kwargs)
    except Exception as exc:
        raise _translate_api_error(exc) from exc

    return result


def _pdf_unsupported_for_gemini_error() -> ReceiptExtractionError:
    """Shared message for the gemini+PDF rejection, checked in two places
    (extract_receipt() for an early exit, _build_content() as a defensive
    guard against calling it directly) -- factored out so the two call
    sites can't drift apart."""
    return ReceiptExtractionError(
        "PDF input is not yet supported with LLM_PROVIDER=gemini. "
        "Use 'openai' or 'anthropic' for PDF receipts, or convert to an image first."
    )


def _build_content(provider: str, media_type: str, b64_data: str, prompt: str) -> list:
    """Build the provider-specific content list for one Instructor `.create()`
    call. Handles both images (dict content blocks, hand-built per provider)
    and PDFs (Instructor's own cross-provider `PDF` multimodal helper).

    OpenAI and Anthropic image formats are verified against stable,
    long-documented public API shapes. The Gemini image branch is NOT
    independently verified against a real API call (no GEMINI_API_KEY
    available) -- it relies on Instructor's own stated "provider-agnostic"
    design (accepting OpenAI-shaped content blocks and normalizing them
    internally). Verify before relying on this for a real Gemini run.

    PDF support (openai + anthropic only) uses
    `instructor.processing.multimodal.PDF`, verified to exist with a
    `from_base64(data_uri)` classmethod against the installed library
    (instructor>=1.7, this file's pin) via `inspect.signature()` -- NOT
    verified against a real end-to-end API response, since no API key was
    available. Treat the PDF path as unverified the same way, until it's
    been run against a live call.

    Raises:
        ReceiptExtractionError: if `media_type` is "application/pdf" and
            `provider` is "gemini". `extract_receipt()` also checks this
            earlier (to avoid building a client/resolving a model just to
            throw it away), but the guard lives here too so a direct call to
            this function can't silently produce a PDF block for a provider
            that doesn't support it.
    """
    if media_type == "application/pdf":
        if provider == "gemini":
            raise _pdf_unsupported_for_gemini_error()
        from instructor.processing.multimodal import PDF

        return [prompt, PDF.from_base64(f"data:application/pdf;base64,{b64_data}")]

    if provider == "anthropic":
        return [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64_data},
            },
            {"type": "text", "text": prompt},
        ]
    # OpenAI's data-URI image_url format. Used as-is for "openai", and as
    # the (unverified) best-effort format for "gemini" per Instructor's
    # provider-agnostic message normalization claim.
    return [
        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64_data}"}},
        {"type": "text", "text": prompt},
    ]


# --- Error translation (R5: real error handling) ---


def _translate_api_error(exc: Exception) -> ReceiptExtractionError:
    """Turn an Anthropic/Instructor exception into a user-facing
    ReceiptExtractionError. Handles the three R5-required cases; anything
    unrecognized bubbles up as a generic ExtractionError with the original
    exception text preserved for debugging.

    Priority order (exception CLASS is checked before message strings, to
    prevent misclassification when a validation error happens to contain
    words like "overloaded" or "invalid image" in its body):
      1. Validation error (by class name) -> case 3, attach raw output
      2. HTTP status code 400 or bad-image message -> case 1
      3. HTTP status code 429 or rate-limit/overloaded message -> case 2
      4. Legacy validation-shaped message (fallback) -> case 3
      5. Fallback: preserve original exception details for debugging
    """
    exc_class_name = type(exc).__name__.lower()
    message_lower = str(exc).lower()
    status = getattr(exc, "status_code", None)

    # Case 3 (checked FIRST -- see priority-order note above): schema validation failure. Instructor's
    # ValidationError / InstructorValidationError / pydantic's ValidationError
    # all contain "validationerror" in the class name. Message typically
    # includes the raw model output; attach it as `partial` so the UI can
    # surface it in a warning banner.
    if "validationerror" in exc_class_name:
        return _build_validation_error(exc)

    # Case 1: malformed / unreadable image (Anthropic returns 400 with
    # "invalid image" or similar in the body).
    if status == 400 or "invalid image" in message_lower or "could not process" in message_lower:
        return ReceiptExtractionError(
            "Couldn't read this image. Is it a receipt or invoice? "
            "Try a clearer photo or a different file format (JPEG/PNG)."
        )

    # Case 2: rate limit / transient API failure -- Instructor's max_retries
    # already tried; if we're here, retries are exhausted.
    if status == 429 or "rate limit" in message_lower or "overloaded" in message_lower:
        return ReceiptExtractionError(
            "The service is temporarily rate-limited or overloaded. "
            "Wait a minute and try again."
        )

    # Fallback validation-shape detection (rare -- an exception whose class
    # name doesn't include "ValidationError" but whose message does).
    if "validation" in message_lower:
        return _build_validation_error(exc)

    # Ollama down -- surface a clean hint instead of a raw httpx trace.
    from common.llm import OLLAMA_CONNECTION_HINT, is_ollama_connection_error
    if is_ollama_connection_error(exc):
        return ReceiptExtractionError(OLLAMA_CONNECTION_HINT)
    # Ollama's default context (4096 tokens) is too small for image
    # inputs (typical receipt tokenizes to 5-6k). The OpenAI-compat
    # surface ignores per-request options.num_ctx, so the raise has
    # to happen at server startup. Give the user the fix instead of
    # a raw JSON error blob.
    if "exceed_context_size" in message_lower or "n_ctx" in message_lower:
        return ReceiptExtractionError(
            "Ollama's default context (4096 tokens) is too small for "
            "this receipt image. Start Ollama with a larger context: "
            "`OLLAMA_CONTEXT_LENGTH=16384 ollama serve`, then retry. "
            "(This raises num_ctx globally on the Ollama server; it's "
            "the only knob Ollama's OpenAI-compat surface honors.)"
        )

    # Fallback: unknown error class. Preserve original message for debugging.
    return ReceiptExtractionError(
        f"Extraction failed: {type(exc).__name__}: {exc}. "
        "This is an unexpected error -- check the agent logs."
    )


def _build_validation_error(exc: Exception) -> ReceiptExtractionError:
    """Shared by both the class-name path and the message-fallback path in
    _translate_api_error() so they build the same partial-attempt structure.
    Instructor exceptions carry `raw_output` (raw text the model returned)
    and `errors()` (list of validation errors)."""
    raw = getattr(exc, "raw_output", None) or str(exc)
    errors_attr = getattr(exc, "errors", None)
    if callable(errors_attr):
        errors = errors_attr()
    else:
        errors = [str(exc)]
    error_strs = [str(e) for e in errors] if isinstance(errors, list) else [str(errors)]
    return ReceiptExtractionError(
        "Extracted data didn't match the expected receipt schema. "
        "The model's raw output is attached -- you can inspect it below.",
        partial=ExtractionAttempt(raw_text=str(raw), validation_errors=error_strs),
    )


# --- Client factory ---


def _get_real_client(provider: str, model: str):
    """Build an Instructor client for `provider`/`model`. Cloud providers
    (openai/anthropic/gemini) route through Instructor's unified
    `from_provider()`. Ollama routes through `from_openai` with the
    OpenAI client pointed at Ollama's OpenAI-compat surface -- Instructor
    doesn't have a native "ollama/" prefix, but Ollama looks like OpenAI
    over HTTP so from_openai is the right entry point. Imported lazily
    so tests running under LLM_PROVIDER=mock never need `instructor`
    (or any provider SDK) installed.
    """
    import instructor

    if provider == "ollama":
        from openai import OpenAI

        from common.llm import ollama_base_url
        client = OpenAI(base_url=ollama_base_url(), api_key="ollama")
        return instructor.from_openai(client, mode=instructor.Mode.JSON)

    instructor_prefix = _INSTRUCTOR_PROVIDER_PREFIX.get(provider)
    if instructor_prefix is None:
        raise ValueError(f"No Instructor provider mapping for {provider!r}")
    return instructor.from_provider(f"{instructor_prefix}/{model}")


# --- Mock mode ---


def _mock_extraction(image_bytes: bytes) -> ExtractedReceipt:
    """Deterministic canned extraction for smoke tests and CI (LLM_PROVIDER=mock).
    Returns a fixed ExtractedReceipt without ever calling Instructor or
    Anthropic. Byte length of the input drives one field so tests can verify
    "the mock actually saw the input" if they want to."""
    return ExtractedReceipt(
        vendor_name="Mock Vendor Ltd.",
        currency="USD",
        subtotal=10.00,
        tax_total=0.83,
        total=10.83,
        line_items=[
            LineItem(description=f"Mock line (input {len(image_bytes)} bytes)", total=10.00),
        ],
        notes=(
            "This is a mock extraction. Set LLM_PROVIDER to 'openai', 'anthropic', "
            "or 'gemini' and configure the matching API key for a real run."
        ),
    )


# --- CLI entry point (uv run python -m agent) ---


def main() -> int:
    """Command-line entry: takes a receipt image path, prints the extracted
    JSON to stdout, plus (if not mock) an estimated cost to stderr.
    """
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog="receipt-extractor",
        description="Extract structured JSON from a receipt/invoice image. "
        "Set LLM_PROVIDER to 'openai' (default), 'anthropic', or 'gemini' plus "
        "the matching API key for a real call, or LLM_PROVIDER=mock for a "
        "canned response. Use --provider/--model to override per-invocation "
        "without changing env vars.",
    )
    # `image` is optional (nargs="?") so `python -m agent --ui` can launch
    # standalone without a positional argument; when not using --ui, absence
    # of `image` produces a clear parser.error() message below instead.
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a receipt image (JPEG/PNG/WebP/GIF) or PDF (openai/anthropic only). "
        "Omit when using --ui.",
    )
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI instead of CLI.")
    parser.add_argument(
        "--provider",
        choices=(*SUPPORTED_PROVIDERS, "mock"),
        default=None,
        help="Override LLM_PROVIDER for this invocation (openai/anthropic/gemini/mock).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the resolved model for this invocation, e.g. gpt-5.4-mini-2026-03-17.",
    )
    args = parser.parse_args()

    if args.ui:
        # Same dual-mode import pattern as the schemas import above --
        # `python -m agent --ui` from inside the agent's own directory has
        # no parent-package context.
        try:
            from .ui import build_ui
        except ImportError:
            from ui import build_ui

        build_ui().launch()
        return 0

    if args.image is None:
        parser.error("image path is required (or pass --ui to launch the web interface)")
        return 2  # unreachable; parser.error() exits

    if not args.image.exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2

    image_bytes = args.image.read_bytes()
    media_type = _guess_media_type(args.image)

    start = time.perf_counter()
    try:
        result = extract_receipt(
            image_bytes, media_type=media_type, provider=args.provider, model=args.model
        )
    except ReceiptExtractionError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.partial is not None:
            print(f"raw model output:\n{exc.partial.raw_text}", file=sys.stderr)
        return 1

    elapsed_ms = (time.perf_counter() - start) * 1000
    print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    print(f"\n(extracted in {elapsed_ms:.0f}ms)", file=sys.stderr)
    return 0


def _guess_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".pdf": "application/pdf",
    }.get(ext, "image/jpeg")


if __name__ == "__main__":
    raise SystemExit(main())
