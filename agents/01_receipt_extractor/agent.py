"""Receipt/invoice extractor agent — agent #01 of real-world-agents.

Technique demonstrated: **structured extraction with Instructor + Claude vision**.
Pydantic schema (see schemas.py) is passed as `response_model` to Instructor,
which validates and retries the LLM's output until it conforms. The model
sees the receipt image directly (Claude Sonnet-5 vision) plus the prompt at
prompts/extract.txt, and fills in the schema.

Why this technique for this use case: receipts are structured documents that
happen to arrive as images. Instructor collapses "prompt the model + parse JSON
+ validate against Pydantic + retry on schema failure" into one call. Without
Instructor you'd hand-roll all four steps. This is exactly what the pattern is
designed for.

Real-error-handling per SPEC R5: three concrete failure modes are handled
explicitly (see extract_receipt below):
  1. Malformed/unreadable image -> ReceiptExtractionError with friendly message
  2. Rate limit / transient API failure -> exponential backoff via Instructor's
     max_retries, then surface the error rather than looping forever
  3. Partial extraction (validation fails after all retries) -> raise
     ReceiptExtractionError with the raw model output attached so the caller
     can show it as a warning banner

LLM_PROVIDER env-var semantics (documented per F1.2 review S8, differs from
common/llm.py's factory on purpose):
  - LLM_PROVIDER="mock" -> use _mock_extraction() (no network, no key)
  - anything else -> real Anthropic call via Instructor
This agent is Anthropic-only by design (Claude vision + Instructor). Setting
LLM_PROVIDER=openai does NOT route to OpenAI here -- it's treated as "not
mock" and falls through to Anthropic. Documented as a per-agent semantic;
common/llm.py's cross-provider routing applies only to agents that use
common/llm.py, which this one doesn't (Instructor sits directly on Anthropic).

Currently ships image-only (JPEG/PNG/WebP/GIF). PDF support promised by
SPEC.md §6.2 is tracked in tasks/todo.md as a prerequisite for F1.5 --
Anthropic's `document` content block will handle it natively.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path

# F1.3 review fix S1: dual-mode import. `.schemas` (relative) resolves when
# this module is loaded AS a submodule of the 01_receipt_extractor package
# (how tests/test_smoke.py imports it, via importlib since the dir name
# starts with a digit). The bare `schemas` (absolute) resolves when this
# file is run directly via `python -m agent` from inside the agent's own
# directory (the documented CLI invocation in README.md) -- in that mode
# Python has no parent-package context at all, so relative imports raise
# ImportError. Both invocation styles must work; this try/except is the
# standard pattern for a script that's also importable as a package member.
try:
    from .schemas import (  # F1.2 review S3: LineItem now imported at top instead of inside _mock_extraction
        ExtractedReceipt,
        LineItem,
    )
except ImportError:
    from schemas import ExtractedReceipt, LineItem

# --- Model + prompt ---

# SPEC R7: pinned dated snapshot. Re-verify before shipping F1 per §16 F1.1
# acceptance (real-provider smoke tests are a manual maintainer check).
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_RETRIES = 3  # covers rate-limit blips + one schema-validation retry

_PROMPT_PATH = Path(__file__).parent / "prompts" / "extract.txt"


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
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    _client=None,  # test-injection escape hatch; production callers should not pass this
) -> ExtractedReceipt:
    """Extract structured receipt data from an image.

    Args:
        image_bytes: raw bytes of a JPEG/PNG receipt image.
        media_type: MIME type of the image (e.g. "image/jpeg", "image/png").
        model: pinned Anthropic model ID. See DEFAULT_MODEL.
        max_retries: how many Instructor retries on schema-validation failures
            or transient API errors. Default 3 is a sane blend of "handle
            rate-limit blips" and "don't loop forever on a genuinely
            unextractable image."
        _client: injected Instructor client for tests. Production callers
            should leave this None; the client is built from LLM_PROVIDER.

    Returns:
        A validated ExtractedReceipt.

    Raises:
        ReceiptExtractionError: on any of the three R5-required failure modes.
    """
    if os.environ.get("LLM_PROVIDER", "").lower() == "mock":
        # Bypass Instructor entirely under mock -- see _mock_extraction docstring
        # for why (we don't want tests to depend on Instructor's internals).
        return _mock_extraction(image_bytes)

    client = _client if _client is not None else _get_real_client()
    prompt = _load_prompt()
    b64_data = base64.b64encode(image_bytes).decode("ascii")

    # Instructor's create() with response_model handles retries on schema
    # validation. Anthropic API errors (rate limit / auth / 400) surface as
    # exceptions we catch and translate below.
    try:
        result = client.messages.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            max_retries=max_retries,
            response_model=ExtractedReceipt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
    except Exception as exc:
        raise _translate_api_error(exc) from exc

    return result


# --- Error translation (R5: real error handling) ---


def _translate_api_error(exc: Exception) -> ReceiptExtractionError:
    """Turn an Anthropic/Instructor exception into a user-facing
    ReceiptExtractionError. Handles the three R5-required cases; anything
    unrecognized bubbles up as a generic ExtractionError with the original
    exception text preserved for debugging.

    Priority order (F1.2 review S9: check exception CLASS before message
    strings to prevent misclassification when a validation error happens to
    contain words like "overloaded" or "invalid image" in its body):
      1. Validation error (by class name) -> case 3, attach raw output
      2. HTTP status code 400 or bad-image message -> case 1
      3. HTTP status code 429 or rate-limit/overloaded message -> case 2
      4. Legacy validation-shaped message (fallback) -> case 3
      5. Fallback: preserve original exception details for debugging
    """
    exc_class_name = type(exc).__name__.lower()
    message_lower = str(exc).lower()
    status = getattr(exc, "status_code", None)

    # Case 3 (checked FIRST per S9): schema validation failure. Instructor's
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

    # Fallback: unknown error class. Preserve original message for debugging.
    return ReceiptExtractionError(
        f"Extraction failed: {type(exc).__name__}: {exc}. "
        "This is an unexpected error -- check the agent logs."
    )


def _build_validation_error(exc: Exception) -> ReceiptExtractionError:
    """Extracted from _translate_api_error (F1.2 review S9) so both the
    class-name path and the message-fallback path build the same partial-
    attempt structure. Instructor exceptions carry `raw_output` (raw text
    the model returned) and `errors()` (list of validation errors)."""
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


def _get_real_client():
    """Build an Instructor + Anthropic client. Imported lazily so tests
    running under LLM_PROVIDER=mock never need `instructor` or `anthropic`
    installed."""
    import instructor
    from anthropic import Anthropic

    return instructor.from_anthropic(Anthropic())


# --- Mock mode ---


def _mock_extraction(image_bytes: bytes) -> ExtractedReceipt:
    """Deterministic canned extraction for smoke tests and CI (LLM_PROVIDER=mock).
    Returns a fixed ExtractedReceipt without ever calling Instructor or
    Anthropic. Byte length of the input drives one field so tests can verify
    "the mock actually saw the input" if they want to."""
    # F1.2 review S3: LineItem now imported at module top; local import removed.
    return ExtractedReceipt(
        vendor_name="Mock Vendor Ltd.",
        currency="USD",
        subtotal=10.00,
        tax_total=0.83,
        total=10.83,
        line_items=[
            LineItem(description=f"Mock line (input {len(image_bytes)} bytes)", total=10.00),
        ],
        notes="This is a mock extraction. Set LLM_PROVIDER=anthropic and configure ANTHROPIC_API_KEY for a real run.",
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
        "Set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY for a real call, "
        "or LLM_PROVIDER=mock for a canned response.",
    )
    # F1.2 review S1: `image` was a required positional, so `python -m agent
    # --ui` failed with "the following arguments are required: image."
    # Made optional (nargs="?") so --ui works standalone; when not using
    # --ui, absence of `image` produces a clear parser.error() message.
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a receipt image (JPEG/PNG/WebP/GIF). Omit when using --ui.",
    )
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI instead of CLI.")
    args = parser.parse_args()

    if args.ui:
        # F1.3 review fix S1: same dual-mode import pattern as the schemas
        # import above -- `python -m agent --ui` from inside the agent's
        # own directory has no parent-package context.
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
        result = extract_receipt(image_bytes, media_type=media_type)
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
    }.get(ext, "image/jpeg")


if __name__ == "__main__":
    raise SystemExit(main())
