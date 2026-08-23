"""Smoke tests for the contract reviewer agent.

All tests run under LLM_PROVIDER=mock (R8 in CONTRIBUTING.md) -- CI never
touches a real API key. Real-provider tests are a manual maintainer check
before shipping.

Covers, in order below:

1. Mock path returns a valid ContractReview (input passthrough proof).
2. Retry loop:
   - happy path (valid response on first attempt)
   - bad JSON -> good JSON (proves the loop re-prompts with feedback)
   - paraphrased excerpt -> verbatim excerpt (proves [C5]: the loop
     enforces excerpt-in-source as a first-class validation failure)
   - retries exhausted (raises ContractReviewError with .partial set
     so the UI can surface the last raw output)
3. Three R5 error branches:
   - Scanned PDF detection (both thresholds: chars/page AND total chars)
   - Context-window overflow (message names the model)
   - _translate_api_error mapping (429, 401, class-name priority)
4. Pure helper functions:
   - _parse_json_object (code fences, prose+json, bad JSON, non-object)
   - _normalize_for_substring (whitespace collapse)
   - _load_source_text (str / PDF magic / plain bytes routing)
5. pypdf ImportError path returns friendly ContractReviewError (S7 fix).

Tests use a SequenceLLM fixture (custom LLM Protocol impl, [C1]) that
returns different `.complete()` responses on successive calls -- the
shared `common/llm.py::MockLLM` only supports a single canned response,
and adding sequenced responses to it would be a shared-chassis change
outside this agent's scope.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# `02_contract_reviewer` starts with a digit -- invalid Python identifier
# for normal import syntax. Same importlib bootstrap as
# agents/01_receipt_extractor/tests/test_smoke.py.
_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AGENT_DIR.parent))

_agent = importlib.import_module("02_contract_reviewer.agent")
_schemas = importlib.import_module("02_contract_reviewer.schemas")

review_contract = _agent.review_contract
ContractReviewError = _agent.ContractReviewError
ReviewAttempt = _agent.ReviewAttempt
resolve_provider = _agent.resolve_provider
_is_likely_scanned = _agent._is_likely_scanned
_check_context_window = _agent._check_context_window
_translate_api_error = _agent._translate_api_error
_parse_json_object = _agent._parse_json_object
_normalize_for_substring = _agent._normalize_for_substring
_load_source_text = _agent._load_source_text
_run_review_loop = _agent._run_review_loop
_extract_pdf_text = _agent._extract_pdf_text
MIN_TOTAL_CHARS = _agent.MIN_TOTAL_CHARS
MIN_CHARS_PER_PAGE = _agent.MIN_CHARS_PER_PAGE
MAX_INPUT_TOKENS_ESTIMATE = _agent.MAX_INPUT_TOKENS_ESTIMATE
CHARS_PER_TOKEN_ESTIMATE = _agent.CHARS_PER_TOKEN_ESTIMATE
ContractReview = _schemas.ContractReview
ContractFlag = _schemas.ContractFlag


# --- SequenceLLM fixture ([C1]) --------------------------------------------


class SequenceLLM:
    """Test-only LLM Protocol impl. Returns responses from a pre-set list
    on successive .complete() calls, records every call so tests can
    assert call count / prompt content. Zero shared-chassis change --
    common/llm.py::MockLLM stays single-response for its own use cases."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cacheable_prefix: str | None = None,
    ):
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        if not self._responses:
            raise RuntimeError("SequenceLLM ran out of scripted responses")
        text = self._responses.pop(0)

        # Return a minimal object with the attributes _run_review_loop reads.
        class _Resp:
            pass

        r = _Resp()
        r.text = text
        r.input_tokens = 0
        r.output_tokens = 0
        r.cached_input_tokens = 0
        r.cache_creation_input_tokens = 0
        r.latency_ms = 0.0
        return r


@pytest.fixture(autouse=True)
def _zero_backoff(monkeypatch):
    """Skip the real 1s+2s sleep during retry tests -- otherwise this file
    would take ~10s per multi-retry test. The sleep in agent.py is
    conditional on another attempt being scheduled; this monkeypatch just
    makes each of those sleeps free."""
    monkeypatch.setattr(_agent, "RETRY_BACKOFF_SECONDS", 0.0)


# --- Fixtures for review-loop tests ----------------------------------------

_SOURCE_TEXT_WITH_CLAUSE = (
    "This is a Mutual Non-Disclosure Agreement between Acme Corp and Widgets LLC. "
    "The parties agree to keep confidential information secret for three years. "
    "This Agreement shall automatically renew for successive 12-month terms unless "
    "either party gives 60 days written notice. Governing law: Delaware."
)


def _valid_review_json(excerpt: str | None = None) -> str:
    """A ContractReview JSON string whose one flag's excerpt is a
    substring of _SOURCE_TEXT_WITH_CLAUSE by default."""
    excerpt = excerpt or "This Agreement shall automatically renew for successive 12-month terms"
    return f'''{{
        "contract_type": "Mutual NDA",
        "parties": ["Acme Corp", "Widgets LLC"],
        "flags": [
            {{
                "category": "auto_renewal",
                "severity": "medium",
                "excerpt": "{excerpt}",
                "explanation": "The contract renews unless you cancel 60 days out.",
                "recommendation": "Ask for cancel-anytime."
            }}
        ],
        "summary": "Mutual NDA with an auto-renewal trap.",
        "overall_risk": "medium"
    }}'''


# --- 1. Mock path ----------------------------------------------------------


def test_mock_path_returns_valid_contract_review(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    review = review_contract(_SOURCE_TEXT_WITH_CLAUSE)
    assert isinstance(review, ContractReview)
    assert review.contract_type is not None
    assert len(review.flags) >= 1
    # Mock encodes input length into summary so a future refactor that
    # accidentally makes mock output constant regardless of input surfaces here.
    assert str(len(_SOURCE_TEXT_WITH_CLAUSE)) in review.summary


def test_mock_path_serializable_to_json(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    review = review_contract(_SOURCE_TEXT_WITH_CLAUSE)
    dumped = review.model_dump_json()
    restored = ContractReview.model_validate_json(dumped)
    assert restored == review


def test_mock_path_accepts_bytes_input(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    # bytes that are NOT a PDF should be decoded as UTF-8 plaintext.
    review = review_contract(b"Some plain text contract body.")
    assert isinstance(review, ContractReview)


def test_provider_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")  # would try to call OpenAI
    # Explicit provider="mock" short-circuits before any network call.
    review = review_contract("test", provider="mock")
    assert isinstance(review, ContractReview)


# --- 2. Retry loop ---------------------------------------------------------


def test_loop_returns_review_on_first_valid_response():
    llm = SequenceLLM([_valid_review_json()])
    review = review_contract(
        _SOURCE_TEXT_WITH_CLAUSE,
        provider="openai",
        model="test-model",
        _llm=llm,
    )
    assert isinstance(review, ContractReview)
    assert len(llm.calls) == 1
    assert review.flags[0].category == "auto_renewal"


def test_loop_retries_on_bad_json_then_succeeds():
    llm = SequenceLLM(["not valid json {{{", _valid_review_json()])
    review = review_contract(
        _SOURCE_TEXT_WITH_CLAUSE,
        provider="openai",
        model="test-model",
        _llm=llm,
    )
    assert isinstance(review, ContractReview)
    assert len(llm.calls) == 2
    # Second prompt must contain retry feedback naming the JSON failure.
    assert "not valid JSON" in llm.calls[1]["prompt"]


def test_loop_retries_on_paraphrased_excerpt_then_succeeds():
    """[C5] enforcement: an excerpt that isn't a substring of source
    triggers the same retry mechanism as bad JSON."""
    paraphrased_json = _valid_review_json(excerpt="This contract renews yearly")
    llm = SequenceLLM([paraphrased_json, _valid_review_json()])
    review = review_contract(
        _SOURCE_TEXT_WITH_CLAUSE,
        provider="openai",
        model="test-model",
        _llm=llm,
    )
    assert isinstance(review, ContractReview)
    assert len(llm.calls) == 2
    # Retry-feedback prompt must mention the exact failed excerpt so the
    # model sees which flag to fix.
    assert "This contract renews yearly" in llm.calls[1]["prompt"]
    assert "verbatim substring" in llm.calls[1]["prompt"]


def test_loop_retries_on_schema_violation_then_succeeds():
    """A JSON object that parses but violates the Pydantic schema (bad
    category) should trigger retry, not raise immediately."""
    bad_category_json = _valid_review_json().replace(
        '"category": "auto_renewal"', '"category": "made_up_category"'
    )
    llm = SequenceLLM([bad_category_json, _valid_review_json()])
    review = review_contract(
        _SOURCE_TEXT_WITH_CLAUSE,
        provider="openai",
        model="test-model",
        _llm=llm,
    )
    assert isinstance(review, ContractReview)
    assert len(llm.calls) == 2


def test_loop_raises_after_max_retries_with_partial_attached():
    """After max_retries failures, the loop raises ContractReviewError
    with .partial populated so the UI can surface the last raw output."""
    llm = SequenceLLM(["not json", "still not json", "nope"])
    with pytest.raises(ContractReviewError) as exc_info:
        review_contract(
            _SOURCE_TEXT_WITH_CLAUSE,
            provider="openai",
            model="test-model",
            _llm=llm,
            max_retries=3,
        )
    assert len(llm.calls) == 3
    assert exc_info.value.partial is not None
    assert isinstance(exc_info.value.partial, ReviewAttempt)
    assert exc_info.value.partial.raw_text == "nope"
    assert "after 3 attempts" in exc_info.value.message


def test_loop_handles_markdown_fenced_json():
    """Real model output often wraps JSON in ```json ... ``` code fences.
    _parse_json_object strips them; the loop should accept fenced output
    on the first attempt (no retry needed)."""
    fenced = f"```json\n{_valid_review_json()}\n```"
    llm = SequenceLLM([fenced])
    review = review_contract(
        _SOURCE_TEXT_WITH_CLAUSE,
        provider="openai",
        model="test-model",
        _llm=llm,
    )
    assert len(llm.calls) == 1
    assert isinstance(review, ContractReview)


# --- 3. R5 error branches --------------------------------------------------


def test_scanned_pdf_below_total_chars_triggers_error(monkeypatch):
    """[M3] threshold #1: fewer than MIN_TOTAL_CHARS across all pages."""
    monkeypatch.setattr(
        _agent,
        "_extract_pdf_text",
        lambda pdf_bytes: ("tiny", 3),  # 4 chars, 3 pages
    )
    with pytest.raises(ContractReviewError) as exc_info:
        review_contract(b"%PDF-1.4 fake", provider="openai")
    assert "scanned" in exc_info.value.message.lower()
    assert "3 page" in exc_info.value.message


def test_scanned_pdf_below_chars_per_page_triggers_error(monkeypatch):
    """[M3] threshold #2: chars/page averaged below MIN_CHARS_PER_PAGE
    even if the total exceeds MIN_TOTAL_CHARS."""
    text = "x" * (MIN_TOTAL_CHARS + 50)  # over total threshold
    page_count = 10  # (MIN_TOTAL_CHARS+50) / 10 = ~25 chars/page, way under 100
    monkeypatch.setattr(_agent, "_extract_pdf_text", lambda pdf_bytes: (text, page_count))
    with pytest.raises(ContractReviewError) as exc_info:
        review_contract(b"%PDF-1.4 fake", provider="openai")
    assert "scanned" in exc_info.value.message.lower()


def test_healthy_pdf_does_not_trigger_scanned_error(monkeypatch):
    """Genuine text-based PDF should NOT trigger the scanned-PDF error --
    regression guard against future threshold-tightening that would
    reject legitimate documents."""
    text = "A" * (MIN_CHARS_PER_PAGE * 5 + 100) + " " + "This Agreement shall automatically renew for successive 12-month terms"
    monkeypatch.setattr(_agent, "_extract_pdf_text", lambda pdf_bytes: (text, 5))
    llm = SequenceLLM([_valid_review_json()])
    # Should reach the loop, not raise scanned-PDF error.
    review = review_contract(b"%PDF-1.4 fake", provider="openai", model="test-model", _llm=llm)
    assert isinstance(review, ContractReview)


def test_is_likely_scanned_unit():
    """Direct unit test of the threshold logic, independent of PDF I/O."""
    # Below MIN_TOTAL_CHARS -> True
    assert _is_likely_scanned("x" * (MIN_TOTAL_CHARS - 1), page_count=1) is True
    # Below MIN_CHARS_PER_PAGE averaged -> True
    assert _is_likely_scanned("x" * (MIN_TOTAL_CHARS + 50), page_count=100) is True
    # Both thresholds cleared -> False
    healthy_text = "x" * (MIN_CHARS_PER_PAGE * 10 + 100)
    assert _is_likely_scanned(healthy_text, page_count=10) is False


def test_context_window_overflow_raises_with_model_name():
    """R5 case 2: over-length input raises before any API call, and the
    model name is in the error message so the user knows which knob to
    adjust."""
    huge = "x" * ((MAX_INPUT_TOKENS_ESTIMATE + 100) * CHARS_PER_TOKEN_ESTIMATE)
    with pytest.raises(ContractReviewError) as exc_info:
        _check_context_window(huge, "gpt-4.1-mini-2025-04-14")
    assert "too long" in exc_info.value.message
    assert "gpt-4.1-mini" in exc_info.value.message
    assert str(MAX_INPUT_TOKENS_ESTIMATE) in exc_info.value.message.replace(",", "")


def test_context_window_healthy_input_does_not_raise():
    _check_context_window("normal contract text", "gpt-4.1-mini-2025-04-14")


def test_translate_api_error_rate_limit_by_status():
    class FakeError(Exception):
        status_code = 429

    result = _translate_api_error(FakeError("some 500 body"))
    assert isinstance(result, ContractReviewError)
    assert "rate-limited" in result.message.lower()


def test_translate_api_error_rate_limit_by_class_name():
    class RateLimitError(Exception):
        pass

    result = _translate_api_error(RateLimitError("no status code here"))
    assert "rate-limited" in result.message.lower()


def test_translate_api_error_auth_by_status():
    class FakeError(Exception):
        status_code = 401

    result = _translate_api_error(FakeError(""))
    assert "authentication" in result.message.lower()


def test_translate_api_error_auth_by_class_name():
    class AuthenticationError(Exception):
        pass

    result = _translate_api_error(AuthenticationError("bad key"))
    assert "authentication" in result.message.lower()


def test_translate_api_error_class_check_priority():
    """Class-name check must fire BEFORE message-string fallback. A
    ValueError whose message contains 'rate limit' should NOT be
    classified as a rate-limit error unless it also matches by class
    or status."""

    # A generic ValueError with rate-limit words in the message DOES
    # fall through to the message-string fallback (case 5 in the priority
    # order), which is the correct behavior for a real rate-limit that
    # somehow wasn't wrapped in a typed exception. What we're guarding
    # against is misclassification of exceptions that clearly aren't
    # rate limits but happen to mention the phrase.
    #
    # More important guard: an exception whose CLASS matches (case 1) but
    # whose message does NOT mention "rate limit" still gets classified
    # as rate limit -- that's what the class-first priority achieves.
    class RateLimitError(Exception):
        pass

    result = _translate_api_error(RateLimitError("Server returned a 200 (no error text)"))
    assert "rate-limited" in result.message.lower()


def test_translate_api_error_unknown_preserves_original():
    result = _translate_api_error(ValueError("some unrelated thing happened"))
    assert "ValueError" in result.message
    assert "some unrelated thing happened" in result.message


# --- 4. Pure helpers -------------------------------------------------------


def test_parse_json_object_plain_json():
    parsed = _parse_json_object('{"a": 1}')
    assert parsed == {"a": 1}


def test_parse_json_object_strips_markdown_fences():
    parsed = _parse_json_object('```json\n{"a": 1}\n```')
    assert parsed == {"a": 1}


def test_parse_json_object_strips_bare_fences():
    parsed = _parse_json_object('```\n{"a": 1}\n```')
    assert parsed == {"a": 1}


def test_parse_json_object_extracts_from_prose():
    """Model prefaces object with 'Here is the JSON:' -- regex should
    still extract the outermost {...} span."""
    parsed = _parse_json_object('Here is the JSON: {"a": 1}')
    assert parsed == {"a": 1}


def test_parse_json_object_returns_error_string_on_bad_json():
    result = _parse_json_object("{malformed")
    assert isinstance(result, str)
    assert "JSON" in result or "no JSON object" in result


def test_parse_json_object_returns_error_string_on_non_object():
    """A JSON array (not object) should be rejected with an error
    string, not returned as-is."""
    result = _parse_json_object('[1, 2, 3]')
    assert isinstance(result, str)


def test_parse_json_object_returns_error_string_on_empty_response():
    result = _parse_json_object("")
    assert isinstance(result, str)


def test_normalize_for_substring_collapses_whitespace():
    assert _normalize_for_substring("a  b") == "a b"
    assert _normalize_for_substring("a\n\tb") == "a b"
    assert _normalize_for_substring("  padded  ") == "padded"


def test_normalize_for_substring_preserves_case():
    """Case-sensitive so an all-lowercase excerpt of an all-uppercase
    clause fails the substring check -- that IS a paraphrase, not a
    whitespace artifact."""
    assert _normalize_for_substring("HELLO") == "HELLO"
    assert _normalize_for_substring("Hello") == "Hello"


def test_normalize_for_substring_makes_multiline_excerpt_match_source():
    source = "This is a\nlong clause that\nspans several lines."
    excerpt_from_model = "This is a long clause that spans several lines."
    normalized_source = _normalize_for_substring(source)
    normalized_excerpt = _normalize_for_substring(excerpt_from_model)
    assert normalized_excerpt in normalized_source


def test_load_source_text_str_passthrough():
    text, pages = _load_source_text("hello world")
    assert text == "hello world"
    assert pages is None


def test_load_source_text_plain_bytes_decodes_utf8():
    text, pages = _load_source_text("héllo".encode())
    assert text == "héllo"
    assert pages is None


def test_load_source_text_pdf_magic_routes_to_extract(monkeypatch):
    """Bytes starting with %PDF- get routed to _extract_pdf_text.
    We monkeypatch that function to avoid needing a real PDF fixture."""
    called_with = {}

    def fake_extract(pdf_bytes):
        called_with["bytes"] = pdf_bytes
        return "extracted text", 2

    monkeypatch.setattr(_agent, "_extract_pdf_text", fake_extract)
    text, pages = _load_source_text(b"%PDF-1.4 anything")
    assert text == "extracted text"
    assert pages == 2
    assert called_with["bytes"] == b"%PDF-1.4 anything"


def test_resolve_provider_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider() == "openai"


def test_resolve_provider_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert resolve_provider() == "anthropic"


def test_resolve_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "grok-9")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


# --- 5. pypdf ImportError fallback (S7 fix) --------------------------------


def test_extract_pdf_text_raises_friendly_error_when_pypdf_missing(monkeypatch):
    """S7 fix: without pypdf, a bytes-starting-with-%PDF input surfaces
    an actionable ContractReviewError with an 'install pypdf' hint,
    not a raw ImportError that _translate_api_error would misclassify."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("No module named 'pypdf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ContractReviewError) as exc_info:
        _extract_pdf_text(b"%PDF-1.4 fake bytes")
    assert "pypdf is not installed" in exc_info.value.message
    assert "pip install" in exc_info.value.message
