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
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .schemas import ExtractedReceipt

logger = logging.getLogger(__name__)

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
    """
    message_lower = str(exc).lower()
    status = getattr(exc, "status_code", None)

    # Case 1: malformed / unreadable image (Anthropic returns 400 with
    # "invalid image" or similar in the body).
    if status == 400 or "invalid image" in message_lower or "could not process" in message_lower:
        return ReceiptExtractionError(
            "Couldn't read this image. Is it a receipt or invoice? "
            "Try a clearer photo or a different file format (JPEG/PNG)."
        )

    # Case 2: rate limit / transient API failure — Instructor's max_retries
    # already tried; if we're here, retries are exhausted.
    if status == 429 or "rate limit" in message_lower or "overloaded" in message_lower:
        return ReceiptExtractionError(
            "The service is temporarily rate-limited or overloaded. "
            "Wait a minute and try again."
        )

    # Case 3: schema validation failure after all retries. Instructor's
    # ValidationError message typically includes the raw model output;
    # attach it as `partial` so the UI can surface it in a warning banner.
    if "validationerror" in type(exc).__name__.lower() or "validation" in message_lower:
        raw = getattr(exc, "raw_output", None) or str(exc)
        errors = getattr(exc, "errors", lambda: [str(exc)])()
        error_strs = [str(e) for e in errors] if isinstance(errors, list) else [str(errors)]
        return ReceiptExtractionError(
            "Extracted data didn't match the expected receipt schema. "
            "The model's raw output is attached — you can inspect it below.",
            partial=ExtractionAttempt(raw_text=str(raw), validation_errors=error_strs),
        )

    # Fallback: unknown error class. Preserve original message for debugging.
    return ReceiptExtractionError(
        f"Extraction failed: {type(exc).__name__}: {exc}. "
        "This is an unexpected error — check the agent logs."
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
    from .schemas import LineItem

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
    parser.add_argument("image", type=Path, help="Path to a receipt image (JPEG/PNG).")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI instead of CLI.")
    args = parser.parse_args()

    if args.ui:
        from .ui import build_ui

        build_ui().launch()
        return 0

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
