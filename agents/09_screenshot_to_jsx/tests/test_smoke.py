"""Smoke tests for the screenshot -> JSX agent.

All tests run under LLM_PROVIDER=mock (R8 in CONTRIBUTING.md) -- CI
never touches a real API key. This is Phase A (v0.1), so no Playwright
tests, no render/compare tests, no iteration tests -- those show up
in Phase B/C's session.

Structure:
1. Mock-mode round-trip + input passthrough guard.
2. Schema validators (component_name, jsx_code cross-check, imports).
3. `_build_content` per-provider structural (Anthropic image block,
   OpenAI image_url, Gemini best-effort).
4. R5 error branches: zero-byte input, unknown media_type, unknown
   styling, 6-branch _translate_api_error priority, validation-error
   partial attachment.
5. Provider resolution (default, override, unknown rejection).
6. `_guess_media_type` on extensions.
7. Constants + examples sanity.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_AGENT_DIR.parent))

_agent = importlib.import_module("09_screenshot_to_jsx.agent")
_schemas = importlib.import_module("09_screenshot_to_jsx.schemas")

reconstruct = _agent.reconstruct
resolve_provider = _agent.resolve_provider
ScreenshotToJsxError = _agent.ScreenshotToJsxError
ReconstructionAttempt = _agent.ReconstructionAttempt
_translate_api_error = _agent._translate_api_error
_build_content = _agent._build_content
_guess_media_type = _agent._guess_media_type
_mock_reconstruction = _agent._mock_reconstruction
SUPPORTED_PROVIDERS = _agent.SUPPORTED_PROVIDERS
DEFAULT_STYLING = _agent.DEFAULT_STYLING
MAX_SCREENSHOT_BYTES = _agent.MAX_SCREENSHOT_BYTES
ReconstructedComponent = _schemas.ReconstructedComponent

_EXAMPLES_DIR = _AGENT_DIR / "examples"


# ---------- 1. Mock path ----------


def test_mock_returns_valid_component(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    result = reconstruct(b"\x89PNG" + b"x" * 500, media_type="image/png")
    assert isinstance(result, ReconstructedComponent)
    assert result.component_name == "MockLandingPage"
    assert "function MockLandingPage" in result.jsx_code


def test_mock_echoes_input_byte_count_into_notes(monkeypatch):
    """Anti-refactor guard: if a future refactor makes the mock ignore
    its input, this surfaces. Same convention as agents #01, #08."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    small = reconstruct(b"x" * 10, media_type="image/png")
    large = reconstruct(b"x" * 1000, media_type="image/png")
    assert "10 bytes" in small.notes
    assert "1000 bytes" in large.notes


def test_mock_styling_override(monkeypatch):
    """`styling=inline_styles` must produce different JSX (no `className`,
    yes `style={{...}}`) than the default tailwind path."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    tw = reconstruct(b"x" * 100, media_type="image/png", styling="tailwind")
    inline = reconstruct(b"x" * 100, media_type="image/png", styling="inline_styles")
    assert "className=" in tw.jsx_code
    assert "className=" not in inline.jsx_code
    assert "style={{" in inline.jsx_code
    assert tw.styling_approach == "tailwind"
    assert inline.styling_approach == "inline_styles"


def test_mock_provider_kwarg_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    result = reconstruct(b"x" * 100, media_type="image/png", provider="mock")
    assert result.component_name == "MockLandingPage"


def test_mock_output_is_json_serializable(monkeypatch):
    """UI + CLI both dump result to JSON; guards against a schema
    field type that isn't JSON-clean."""
    import json
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    result = reconstruct(b"x" * 100, media_type="image/png")
    dumped = json.dumps(result.model_dump(mode="json"))
    assert '"component_name"' in dumped
    assert '"jsx_code"' in dumped


# ---------- 2. Schema validators ----------


def test_component_name_must_be_pascal_case():
    """Lowercase / kebab / digits-first component names would break
    a caller's `import <name>` statement. Reject at parse time."""
    valid_jsx = "function BadCase() { return null; }"
    for bad_name in ("landingpage", "landing_page", "landing-page", "9Page", ""):
        with pytest.raises(ValidationError):
            ReconstructedComponent(
                component_name=bad_name,
                jsx_code=valid_jsx.replace("BadCase", bad_name or "X"),
                styling_approach="tailwind",
            )


def test_component_name_accepts_valid_pascal_case():
    r = ReconstructedComponent(
        component_name="AdminDashboard",
        jsx_code="function AdminDashboard() { return null; }",
        styling_approach="tailwind",
    )
    assert r.component_name == "AdminDashboard"


def test_jsx_code_must_define_declared_component_name():
    """Cross-field check: the JSX must actually contain a function or
    const with the claimed component_name. Otherwise the caller's
    import fails silently."""
    with pytest.raises(ValidationError, match="does not define"):
        ReconstructedComponent(
            component_name="AdminDashboard",
            jsx_code="function Homepage() { return null; }",  # wrong name
            styling_approach="tailwind",
        )


def test_jsx_code_accepts_const_declaration_form():
    """Either `function Name(){}` OR `const Name =` should satisfy
    the cross-field check -- both are valid React component forms."""
    r = ReconstructedComponent(
        component_name="MyComp",
        jsx_code="const MyComp = () => <div/>;",
        styling_approach="inline_styles",
    )
    assert r.component_name == "MyComp"


def test_jsx_code_validator_false_passes_on_comment_mention():
    """KNOWN LIMITATION -- pinned here explicitly rather than left as
    an untested surprise. The regex validator matches `function <name>`
    or `const <name> =` anywhere in the string, including inside JS
    comments. A model whose real JSX defines `Homepage` but has a
    comment `// renamed from LandingPage` and claims
    component_name='LandingPage' passes validation. Real defense
    would need a JSX parser (out of scope for Phase A; documented in
    README's 'Where this fails'). If v1.1 adds a JSX parser, this
    test should flip to `pytest.raises(ValidationError)`."""
    misleading = ReconstructedComponent(
        component_name="LandingPage",
        jsx_code=(
            "// function LandingPage was renamed to Homepage\n"
            "function Homepage() { return null; }\n"
        ),
        styling_approach="tailwind",
    )
    # Currently PASSES; documenting the current behavior as intentional
    # for now, not desired long-term.
    assert misleading.component_name == "LandingPage"


def test_imports_reject_empty_strings():
    """An empty string in the imports list would produce a broken
    `import '' from ''` in the caller's code. Reject."""
    with pytest.raises(ValidationError, match="empty"):
        ReconstructedComponent(
            component_name="X",
            jsx_code="function X() {}",
            styling_approach="tailwind",
            imports=["react-icons", "", "clsx"],
        )


def test_styling_approach_is_closed_literal():
    """styling_approach must be one of the two documented values;
    a typo like 'tailwind_v3' or 'inline' should fail at parse."""
    with pytest.raises(ValidationError):
        ReconstructedComponent(
            component_name="X",
            jsx_code="function X() {}",
            styling_approach="css_modules_but_not_really",  # type: ignore[arg-type]
        )


# ---------- 3. _build_content per-provider structural ----------


def test_build_content_anthropic_shape():
    content = _build_content("anthropic", "image/png", "ZmFrZQ==", "prompt")
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[0]["source"]["data"] == "ZmFrZQ=="
    assert content[1] == {"type": "text", "text": "prompt"}


def test_build_content_openai_shape():
    content = _build_content("openai", "image/jpeg", "ZmFrZQ==", "prompt")
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,ZmFrZQ=="
    assert content[1] == {"type": "text", "text": "prompt"}


def test_build_content_gemini_uses_openai_shape():
    """Gemini path is explicitly best-effort (uses OpenAI-shaped blocks
    via Instructor's provider-agnostic normalization). Pins CURRENT
    behavior so a future change is deliberate, not accidental."""
    content = _build_content("gemini", "image/png", "ZmFrZQ==", "prompt")
    assert content[0]["type"] == "image_url"


def test_build_content_every_provider_produces_two_blocks():
    for provider in SUPPORTED_PROVIDERS:
        content = _build_content(provider, "image/png", "ZmFrZQ==", "prompt")
        assert len(content) == 2
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "prompt"


# ---------- 4. R5 error branches ----------


def test_reconstruct_rejects_zero_byte_screenshot(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ScreenshotToJsxError, match="empty"):
        reconstruct(b"", media_type="image/png")


def test_reconstruct_rejects_oversized_screenshot(monkeypatch):
    """Vision providers cap image input at ~20 MB; reject BEFORE
    base64-encoding so a caller passing a 50 MB screenshot gets a
    clear domain error rather than a cryptic 'content too large'
    surfacing from inside Instructor."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    huge = b"x" * (MAX_SCREENSHOT_BYTES + 1)
    with pytest.raises(ScreenshotToJsxError, match="over the"):
        reconstruct(huge, media_type="image/png")


def test_reconstruct_rejects_unknown_media_type(monkeypatch):
    """Unified error boundary: unknown media_type raises
    ScreenshotToJsxError (not ValueError) so a caller with a single
    `except ScreenshotToJsxError` catches every input-validation case."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ScreenshotToJsxError, match="Unknown media_type"):
        reconstruct(b"x" * 100, media_type="image/bmp")


def test_reconstruct_rejects_unknown_styling(monkeypatch):
    """Same unified error boundary as unknown media_type."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ScreenshotToJsxError, match="Unknown styling"):
        reconstruct(b"x" * 100, media_type="image/png", styling="css_modules")  # type: ignore[arg-type]


def test_reconstruct_input_validation_all_raise_the_same_type(monkeypatch):
    """Callers wrapping reconstruct() in `except ScreenshotToJsxError`
    should catch ALL boundary-validation failures with one clause. This
    test locks that contract; a future refactor that raises anything
    other than ScreenshotToJsxError for these four cases breaks it."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    cases = [
        (b"", "image/png", "tailwind"),
        (b"x" * (MAX_SCREENSHOT_BYTES + 1), "image/png", "tailwind"),
        (b"x" * 100, "image/bmp", "tailwind"),
        (b"x" * 100, "image/png", "css_modules"),
    ]
    for screenshot_bytes, media_type, styling in cases:
        with pytest.raises(ScreenshotToJsxError):
            reconstruct(
                screenshot_bytes, media_type=media_type, styling=styling  # type: ignore[arg-type]
            )


def test_translate_api_error_class_name_rate_limit():
    class RateLimitError(Exception):
        pass

    out = _translate_api_error(RateLimitError("whatever"))
    assert isinstance(out, ScreenshotToJsxError)
    assert "rate-limited" in out.message.lower() or "overloaded" in out.message.lower()


def test_translate_api_error_class_name_auth():
    class AuthenticationError(Exception):
        pass

    out = _translate_api_error(AuthenticationError("whatever"))
    assert "authentication" in out.message.lower() or "api key" in out.message.lower()


def test_translate_api_error_status_429():
    exc = RuntimeError("boom")
    exc.status_code = 429  # type: ignore[attr-defined]
    out = _translate_api_error(exc)
    assert "rate-limited" in out.message.lower() or "overloaded" in out.message.lower()


def test_translate_api_error_status_401():
    exc = RuntimeError("boom")
    exc.status_code = 401  # type: ignore[attr-defined]
    out = _translate_api_error(exc)
    assert "authentication" in out.message.lower() or "api key" in out.message.lower()


def test_translate_api_error_message_fallback_rate_limit():
    out = _translate_api_error(RuntimeError("You hit the rate limit"))
    assert "rate-limited" in out.message.lower() or "overloaded" in out.message.lower()


def test_translate_api_error_generic_fallback_preserves_original():
    out = _translate_api_error(RuntimeError("wildly unexpected"))
    assert "RuntimeError" in out.message
    assert "wildly unexpected" in out.message


def test_translate_api_error_validation_error_attaches_partial():
    """Instructor's ValidationError carries raw_output + errors().
    Must be attached as `partial` so a caller can inspect what the
    model actually produced when validation retries were exhausted."""

    class ValidationError(Exception):
        raw_output = "some malformed JSX-ish output"

        def errors(self):
            return [{"msg": "field missing", "loc": ("component_name",)}]

    out = _translate_api_error(ValidationError("validation failed"))
    assert isinstance(out, ScreenshotToJsxError)
    assert out.partial is not None
    assert "malformed JSX" in out.partial.raw_text


# ---------- 5. Provider resolution ----------


def test_resolve_provider_defaults_to_anthropic(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider() == "anthropic"


def test_resolve_provider_respects_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert resolve_provider() == "openai"


def test_resolve_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


# ---------- 6. _guess_media_type ----------


def test_guess_media_type_recognizes_png():
    assert _guess_media_type(Path("screenshot.png")) == "image/png"


def test_guess_media_type_recognizes_jpg_and_jpeg():
    assert _guess_media_type(Path("a.jpg")) == "image/jpeg"
    assert _guess_media_type(Path("a.jpeg")) == "image/jpeg"


def test_guess_media_type_defaults_to_png_on_unknown():
    """Unknown extensions default to png so a caller who passes
    `.tiff` doesn't get a cryptic 'unknown media_type' before the
    validation error fires."""
    assert _guess_media_type(Path("mystery.xyz")) == "image/png"


# ---------- 7. Constants + examples sanity ----------


def test_supported_providers_contains_the_three_expected():
    assert set(SUPPORTED_PROVIDERS) == {"openai", "anthropic", "gemini"}


def test_default_styling_is_tailwind():
    """v0.1 default -- if a future release changes this, tests catch it
    as a reminder to update the README and prompt examples too."""
    assert DEFAULT_STYLING == "tailwind"


def test_examples_dir_has_at_least_one_screenshot():
    """The Gradio UI's sample-picker widget shows nothing if this
    dir is empty. Regression guard against a future clean-up
    accidentally removing the shipped mocks."""
    if not _EXAMPLES_DIR.exists():
        pytest.skip("examples dir absent -- may be a fresh workspace state")
    pngs = [p for p in _EXAMPLES_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    assert len(pngs) >= 1, "expected at least one sample screenshot in examples/"


def test_examples_readme_documents_source():
    """CC BY-style rule: if we ship example images, attribution/source
    must be documented. Our images are self-generated PIL mockups
    (no third-party rights involved) but the file still needs to say so."""
    readme = _EXAMPLES_DIR / "README.md"
    if not readme.exists():
        pytest.skip("examples/README.md absent")
    content = readme.read_text(encoding="utf-8")
    # Must mention either 'self-owned', 'self-generated', or the sourcing
    # explanation so a reader knows what they're getting.
    assert (
        "self-owned" in content.lower()
        or "self-generated" in content.lower()
        or "no attribution" in content.lower()
        or "no third-party" in content.lower()
    )
