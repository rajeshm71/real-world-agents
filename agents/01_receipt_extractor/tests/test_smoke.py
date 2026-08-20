"""Smoke tests for the receipt extractor agent.

All tests run under LLM_PROVIDER=mock (R8 in CONTRIBUTING.md's hard rules)
-- CI never touches a real API key. Real-provider tests are a manual
maintainer check before shipping.

Covers:
- Mock extraction returns a valid ExtractedReceipt (proves the end-to-end
  code path from bytes -> Pydantic-validated object works)
- Schema validation catches bad input at the Pydantic layer
- Error translation (R5) correctly maps API errors -> ReceiptExtractionError
  with the right message + partial-attempt data for each of the three cases
- The mock extractor sees the actual input bytes (guards against a future
  refactor that accidentally makes mock output constant regardless of input)
"""

from __future__ import annotations

# Import as a package -- the agent is expected to be installed via
# `pip install -e agents/01_receipt_extractor/` OR `uv sync` at the
# workspace root. Test discovery via pytest treats `agents/01_receipt_extractor/`
# as a package because it has __init__.py.
# Use a relative name that works regardless of workspace install status:
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AGENT_DIR.parent))  # so `01_receipt_extractor` is importable

# Import via importlib since `01_receipt_extractor` starts with a digit
# (invalid Python identifier) and can't be imported with normal syntax.
import importlib

_agent_pkg = importlib.import_module("01_receipt_extractor.agent")
_schemas_pkg = importlib.import_module("01_receipt_extractor.schemas")

extract_receipt = _agent_pkg.extract_receipt
ReceiptExtractionError = _agent_pkg.ReceiptExtractionError
ExtractionAttempt = _agent_pkg.ExtractionAttempt
_translate_api_error = _agent_pkg._translate_api_error
resolve_provider = _agent_pkg.resolve_provider
_build_content = _agent_pkg._build_content
SUPPORTED_PROVIDERS = _agent_pkg.SUPPORTED_PROVIDERS
_guess_media_type = _agent_pkg._guess_media_type
ExtractedReceipt = _schemas_pkg.ExtractedReceipt
LineItem = _schemas_pkg.LineItem


# ---------- End-to-end mock extraction ----------


def test_mock_extraction_returns_valid_receipt(monkeypatch):
    """Under LLM_PROVIDER=mock, extract_receipt must return a valid
    ExtractedReceipt WITHOUT ever calling Instructor or Anthropic. This
    proves the whole code path (bytes in -> Pydantic-validated model out)
    works before we ever plug in a real API key."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    fake_bytes = b"\x89PNG\r\n\x1a\n" + b"x" * 100  # PNG magic + junk

    result = extract_receipt(fake_bytes, media_type="image/png")

    assert isinstance(result, ExtractedReceipt)
    assert result.total > 0
    assert result.currency == "USD"
    assert len(result.line_items) >= 1
    assert result.line_items[0].total > 0


def test_mock_extraction_sees_the_input_bytes(monkeypatch):
    """Regression guard: the mock includes the input byte count in its line
    description. If a future refactor accidentally makes the mock ignore its
    input entirely, this test catches it -- otherwise the smoke test could
    pass on a broken pipeline that never touches the input."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    small = extract_receipt(b"x" * 10, media_type="image/png")
    large = extract_receipt(b"x" * 1000, media_type="image/png")

    # The mock encodes byte count into the line description.
    assert "10 bytes" in small.line_items[0].description
    assert "1000 bytes" in large.line_items[0].description


def test_mock_extraction_output_is_json_serializable(monkeypatch):
    """The CLI + UI both dump the result to JSON. Any field type that isn't
    JSON-serializable (via Pydantic's model_dump(mode='json')) would break
    them. Test guards against that."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    result = extract_receipt(b"png-bytes", media_type="image/png")
    import json
    dumped = json.dumps(result.model_dump(mode="json"), default=str)
    assert '"total"' in dumped
    assert '"line_items"' in dumped


# ---------- Schema validation ----------


def test_schema_rejects_missing_required_total():
    """`total` is required per §6.2 -- Pydantic must reject an
    ExtractedReceipt without it. If someone quietly removes the requirement,
    this test catches it (the API contract with downstream tools would silently
    break)."""
    with pytest.raises(ValidationError):  # pydantic.ValidationError
        ExtractedReceipt(vendor_name="X")  # missing `total`


def test_schema_accepts_minimal_valid_receipt():
    """The minimum valid receipt: just a total and empty line_items."""
    r = ExtractedReceipt(total=42.50)
    assert r.total == 42.50
    assert r.line_items == []
    assert r.vendor_name is None


def test_line_item_requires_description_and_total():
    """Both description and total are required on LineItem. Missing either
    should be a validation error."""
    with pytest.raises(ValidationError):
        LineItem(total=10.0)  # missing description
    with pytest.raises(ValidationError):
        LineItem(description="X")  # missing total


# ---------- Error translation (R5: three cases handled explicitly) ----------


class _FakeAPIError(Exception):
    """Test double for Anthropic APIError. Includes status_code attribute
    like the real thing, so _translate_api_error's status checks fire."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def test_r5_case1_bad_image_translates_to_friendly_error():
    """R5 case 1: malformed/unreadable image. Anthropic returns HTTP 400.
    Should surface a user-friendly message, not the raw stack trace."""
    err = _translate_api_error(_FakeAPIError("Invalid image data", status_code=400))
    assert isinstance(err, ReceiptExtractionError)
    assert "couldn't read this image" in err.message.lower()
    # No partial data for a bad-image case -- nothing was extracted.
    assert err.partial is None


def test_r5_case2_rate_limit_translates_to_friendly_error():
    """R5 case 2: rate limit / overloaded. Should tell the user to wait,
    not surface the raw 429."""
    err = _translate_api_error(_FakeAPIError("rate limit exceeded", status_code=429))
    assert isinstance(err, ReceiptExtractionError)
    assert "rate-limited" in err.message.lower() or "overloaded" in err.message.lower()


def test_r5_case2_overloaded_message_also_matches():
    """Anthropic sometimes returns 'overloaded_error' in the body rather than
    a clean 429. Test that the substring match catches this too."""
    err = _translate_api_error(_FakeAPIError("service overloaded, please retry"))
    assert isinstance(err, ReceiptExtractionError)
    assert "overloaded" in err.message.lower() or "rate-limited" in err.message.lower()


def test_r5_case3_validation_failure_preserves_raw_output():
    """R5 case 3: after all retries, the model still returned something that
    doesn't match ExtractedReceipt. The UI needs the raw output to show as a
    warning banner -- if we drop it, the user has nothing to inspect."""

    class _FakeValidationError(Exception):
        raw_output = "{'total': 'not a number'}"

        def errors(self):
            return [{"msg": "total must be a float", "loc": ("total",)}]

    err = _translate_api_error(_FakeValidationError("validation error"))
    assert isinstance(err, ReceiptExtractionError)
    assert err.partial is not None
    assert "not a number" in err.partial.raw_text
    assert len(err.partial.validation_errors) >= 1


def test_unknown_error_class_falls_back_gracefully():
    """Some future SDK exception we haven't seen before must not crash the
    translator -- fall back to a generic ExtractionError with the original
    message preserved so the maintainer can debug from the logs."""
    err = _translate_api_error(RuntimeError("something we've never seen"))
    assert isinstance(err, ReceiptExtractionError)
    assert "something we've never seen" in err.message
    assert "RuntimeError" in err.message


# ---------- Prompt file ----------


# ---------- Provider resolution (multi-provider support, user request) ----------


def test_resolve_provider_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider() == "openai"


def test_resolve_provider_respects_env_var(monkeypatch):
    for provider in SUPPORTED_PROVIDERS:
        monkeypatch.setenv("LLM_PROVIDER", provider)
        assert resolve_provider() == provider


def test_resolve_provider_allows_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    assert resolve_provider() == "mock"


def test_resolve_provider_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


def test_extract_receipt_provider_kwarg_overrides_env(monkeypatch):
    """Passing provider="mock" explicitly must bypass whatever LLM_PROVIDER
    is set to -- this is what lets the CLI's --provider flag work per-call
    without mutating the environment."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    result = extract_receipt(b"x" * 50, provider="mock")
    assert isinstance(result, ExtractedReceipt)


# ---------- Image content-block construction (per-provider) ----------


def test_build_image_content_anthropic_shape():
    content = _build_content("anthropic", "image/png", "ZmFrZQ==", "extract this")
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[0]["source"]["data"] == "ZmFrZQ=="
    assert content[1] == {"type": "text", "text": "extract this"}


def test_build_image_content_openai_shape():
    content = _build_content("openai", "image/jpeg", "ZmFrZQ==", "extract this")
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,ZmFrZQ=="
    assert content[1] == {"type": "text", "text": "extract this"}


def test_build_image_content_gemini_uses_openai_shaped_fallback():
    """Gemini path is explicitly unverified (see agent.py module docstring)
    -- pins CURRENT behavior (same shape as OpenAI) so a future change is
    a deliberate decision, not an accidental drift."""
    content = _build_content("gemini", "image/png", "ZmFrZQ==", "extract this")
    assert content[0]["type"] == "image_url"


def test_build_image_content_every_supported_provider_produces_two_blocks():
    for provider in SUPPORTED_PROVIDERS:
        content = _build_content(provider, "image/png", "ZmFrZQ==", "prompt text")
        assert len(content) == 2
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "prompt text"


# ---------- PDF content construction (openai + anthropic only) ----------
# NOTE: these are structural tests only (isinstance / mock-mode), per the
# module docstring on `_build_content` -- the PDF path has NOT been verified
# against a real OpenAI or Anthropic API call in this sandbox.


def test_guess_media_type_recognizes_pdf():
    assert _guess_media_type(Path("invoice.pdf")) == "application/pdf"
    assert _guess_media_type(Path("invoice.PDF")) == "application/pdf"


def test_build_content_pdf_returns_instructor_pdf_object():
    from instructor.processing.multimodal import PDF

    content = _build_content("anthropic", "application/pdf", "ZmFrZQ==", "extract this")
    assert content[0] == "extract this"
    assert isinstance(content[1], PDF)


def test_build_content_pdf_shape_identical_for_openai_and_anthropic():
    """The PDF branch doesn't fork on provider -- Instructor's PDF class
    handles cross-provider formatting internally, unlike the hand-built
    image branches above."""
    from instructor.processing.multimodal import PDF

    openai_content = _build_content("openai", "application/pdf", "ZmFrZQ==", "prompt")
    anthropic_content = _build_content("anthropic", "application/pdf", "ZmFrZQ==", "prompt")
    assert isinstance(openai_content[1], PDF)
    assert isinstance(anthropic_content[1], PDF)


def test_build_content_rejects_pdf_for_gemini_directly():
    """_build_content() enforces the gemini+PDF rejection itself, not just
    extract_receipt() -- calling it directly with provider="gemini" must
    still raise rather than silently returning a PDF content block."""
    with pytest.raises(ReceiptExtractionError, match="PDF input is not yet supported"):
        _build_content("gemini", "application/pdf", "ZmFrZQ==", "extract this")


def test_extract_receipt_pdf_mock_mode_round_trips(monkeypatch):
    """Mock mode bypasses provider/content branching entirely, so this only
    proves the media_type plumbing doesn't break the existing mock path --
    not a test of the real PDF content-block construction."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    result = extract_receipt(b"%PDF-1.4 fake pdf bytes", media_type="application/pdf")
    assert isinstance(result, ExtractedReceipt)


def test_extract_receipt_rejects_pdf_for_gemini(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(ReceiptExtractionError, match="PDF input is not yet supported"):
        extract_receipt(b"%PDF-1.4 fake pdf bytes", media_type="application/pdf", provider="gemini")


def test_prompt_file_exists_and_is_nonempty():
    """The prompt is load-bearing (it's what tells Claude how to extract).
    An empty or missing prompt file would silently degrade extraction quality
    with no test signal — check both."""
    prompt_path = _AGENT_DIR / "prompts" / "extract.txt"
    assert prompt_path.exists(), f"Prompt missing at {prompt_path}"
    content = prompt_path.read_text(encoding="utf-8")
    assert len(content) > 100, "Prompt is suspiciously short — is it truncated?"
    # Load-bearing instructions the prompt MUST include (guards against a
    # future edit that accidentally drops them).
    lower = content.lower()
    assert "do not" in lower or "not" in lower  # something about not fabricating
    assert "total" in lower  # the load-bearing field name
