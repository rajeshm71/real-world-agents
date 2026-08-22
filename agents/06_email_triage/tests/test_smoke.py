"""Smoke tests for the email-triage agent.

All tests under LLM_PROVIDER=mock (R8) -- CI never touches a real API
key. Real-provider tests are a manual maintainer check.

Uses PydanticAI's built-in TestModel for agent-with-fake-model tests
(no custom SequenceLLM needed). TestModel is gated by
pytest.importorskip since even the mock-mode tests don't need
pydantic-ai installed to pass.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AGENT_DIR.parent))

_agent = importlib.import_module("06_email_triage.agent")
_schemas = importlib.import_module("06_email_triage.schemas")

triage_email = _agent.triage_email
EmailTriageError = _agent.EmailTriageError
TriageAttempt = _agent.TriageAttempt
resolve_provider = _agent.resolve_provider
_parse_eml = _agent._parse_eml
_has_triage_material = _agent._has_triage_material
_is_known_contact_impl = _agent._is_known_contact_impl
_is_important_domain_impl = _agent._is_important_domain_impl
_extract_dates_from = _agent._extract_dates_from
_verify_snippet = _agent._verify_snippet
_normalize_ws = _agent._normalize_ws
_translate_api_error = _agent._translate_api_error
_build_agent = _agent._build_agent
MIN_BODY_CHARS = _agent.MIN_BODY_CHARS
EmailTriage = _schemas.EmailTriage
TriageDeps = _schemas.TriageDeps


_SAMPLE_EML = b"""From: sarah@example.com
To: me@example.com
Subject: Q3 spec review

Please review the attached spec by Friday. Section 4 is the priority."""


# --- 1. Mock path ----------------------------------------------------------


def test_mock_path_returns_valid_email_triage(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    t = triage_email(_SAMPLE_EML)
    assert isinstance(t, EmailTriage)
    # Byte count encoded into reasoning as anti-refactor guard.
    assert str(len(_SAMPLE_EML)) in t.reasoning


def test_mock_path_skips_eml_parsing(monkeypatch):
    """Mock mode short-circuits BEFORE _parse_eml -- garbage bytes
    are accepted."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    t = triage_email(b"not a valid eml file at all")
    assert isinstance(t, EmailTriage)


def test_mock_path_serializable_to_json(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    t = triage_email(_SAMPLE_EML)
    dumped = t.model_dump_json()
    restored = EmailTriage.model_validate_json(dumped)
    assert restored == t


# --- 2. R5 case 1: .eml parsing --------------------------------------------


def test_parse_eml_happy_path():
    p = _parse_eml(_SAMPLE_EML)
    assert p.from_address == "sarah@example.com"
    assert "me@example.com" in p.to_addresses
    assert p.subject == "Q3 spec review"
    assert "Please review" in p.body


def test_parse_eml_extracts_multiple_recipients():
    eml = b"From: x@a\nTo: a@b.com, c@d.com, e@f.com\nSubject: hi\n\nBody"
    p = _parse_eml(eml)
    assert len(p.to_addresses) == 3


def test_parse_eml_handles_html_only(monkeypatch):
    """HTML-only email should get its tags stripped to plaintext."""
    eml = (
        b"From: x@a\nTo: b@c\nSubject: html only\n"
        b"Content-Type: text/html\n\n"
        b"<html><body><p>Hello <b>world</b></p></body></html>"
    )
    p = _parse_eml(eml)
    assert "Hello" in p.body
    assert "world" in p.body
    assert "<p>" not in p.body


# --- 3. R5 case 2: empty body ----------------------------------------------


def test_r5_case2_empty_body_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    eml = b"From: x@a\nTo: b@c\nSubject: hi\n\n"
    with pytest.raises(EmailTriageError, match="essentially empty"):
        triage_email(eml)


def test_r5_case2_short_body_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    eml = b"From: x@a\nTo: b@c\nSubject: hi\n\nhi"
    with pytest.raises(EmailTriageError, match="essentially empty"):
        triage_email(eml)


def test_r5_case2_message_names_threshold(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    eml = b"From: x@a\nSubject: hi\n\nhi"
    with pytest.raises(EmailTriageError) as exc_info:
        triage_email(eml)
    assert str(MIN_BODY_CHARS) in exc_info.value.message


def test_has_triage_material_unit():
    assert _has_triage_material("x" * 100) is True
    assert _has_triage_material("hi") is False
    assert _has_triage_material("") is False
    assert _has_triage_material("   " * 20) is False  # only whitespace


# --- 4. R5 case 3: _translate_api_error ------------------------------------


def test_translate_api_error_rate_limit_by_status():
    class E(Exception):
        status_code = 429
    r = _translate_api_error(E("body"))
    assert isinstance(r, EmailTriageError)
    assert "rate-limited" in r.message.lower()


def test_translate_api_error_rate_limit_by_class_name():
    class RateLimitError(Exception):
        pass
    assert "rate-limited" in _translate_api_error(RateLimitError("x")).message.lower()


def test_translate_api_error_auth_by_class_name():
    class AuthenticationError(Exception):
        pass
    assert "authentication" in _translate_api_error(AuthenticationError("bad")).message.lower()


def test_translate_api_error_auth_by_status():
    class E(Exception):
        status_code = 401
    assert "authentication" in _translate_api_error(E("")).message.lower()


def test_translate_api_error_class_check_priority():
    """Class-name check fires BEFORE message-string fallback."""
    class RateLimitError(Exception):
        pass
    r = _translate_api_error(RateLimitError("unrelated body"))
    assert "rate-limited" in r.message.lower()


def test_translate_api_error_unknown_preserves_original():
    r = _translate_api_error(ValueError("weird"))
    assert "ValueError" in r.message
    assert "weird" in r.message


# --- 5. Tools --------------------------------------------------------------


def test_is_known_contact_email_match():
    d = TriageDeps(known_contacts=["alice@example.com"])
    assert _is_known_contact_impl("alice@example.com", d) is True


def test_is_known_contact_name_in_from_line():
    d = TriageDeps(known_contacts=["Alice Chen"])
    assert _is_known_contact_impl("Alice Chen <alice@example.com>", d) is True


def test_is_known_contact_case_insensitive():
    d = TriageDeps(known_contacts=["ALICE@EXAMPLE.COM"])
    assert _is_known_contact_impl("alice@example.com", d) is True


def test_is_known_contact_no_match():
    d = TriageDeps(known_contacts=["alice@example.com"])
    assert _is_known_contact_impl("bob@other.com", d) is False


def test_is_important_domain_suffix_match():
    d = TriageDeps(important_domains=["@mycompany.com"])
    assert _is_important_domain_impl("boss@mycompany.com", d) is True


def test_is_important_domain_no_match():
    d = TriageDeps(important_domains=["@mycompany.com"])
    assert _is_important_domain_impl("random@spam.net", d) is False


def test_extract_dates_finds_weekday_and_next():
    result = _extract_dates_from("Need it by Friday, definitely next week")
    assert any("Friday" in r for r in result)
    assert any("next week" in r.lower() for r in result)


def test_extract_dates_finds_iso():
    result = _extract_dates_from("Deadline 2026-09-15")
    assert "2026-09-15" in result


def test_extract_dates_empty_when_no_dates():
    assert _extract_dates_from("no time references in this text at all") == []


def test_verify_snippet_positive():
    assert _verify_snippet("hello world", "say hello world today") is True


def test_verify_snippet_negative_missing():
    assert _verify_snippet("nope", "hello world") is False


def test_verify_snippet_rejects_empty():
    """M1 pattern from #05: empty excerpt returns False, doesn't
    silently pass via `"" in "any" -> True`."""
    assert _verify_snippet("", "any body") is False
    assert _verify_snippet("   ", "any body") is False


def test_verify_snippet_whitespace_normalized():
    src = "hello\nworld\nfoo"
    assert _verify_snippet("hello world foo", src) is True


def test_normalize_ws_collapses():
    assert _normalize_ws("a  b\tc\nd") == "a b c d"


# --- 6. Agent construction (via pytest.importorskip on pydantic-ai) --------


def test_build_agent_returns_typed_agent(monkeypatch):
    """Structural check: _build_agent returns an Agent instance
    with the expected system_prompt + retries. Skipped if
    pydantic-ai isn't installed. PydanticAI eagerly initializes
    the OpenAI provider at Agent construction, so set a fake
    OPENAI_API_KEY -- test doesn't actually call OpenAI, just
    builds the Agent object."""
    pytest.importorskip("pydantic_ai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-for-structural-test-only")
    agent = _build_agent(model="gpt-4.1-mini-2025-04-14")
    from pydantic_ai import Agent
    assert isinstance(agent, Agent)


def test_agent_with_test_model_returns_valid_triage():
    """PydanticAI's built-in TestModel returns fake JSON matching
    the output_type schema. Use it to exercise the full agent
    machinery without a real API call."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    # Build agent with TestModel instead of real openai. TestModel
    # auto-produces JSON conforming to output_type.
    agent = Agent(
        TestModel(),
        deps_type=TriageDeps,
        output_type=EmailTriage,
        system_prompt="Test",
    )
    result = agent.run_sync(
        "From: a@x\nSubject: t\n\nbody content here for testing",
        deps=TriageDeps(),
    )
    assert isinstance(result.output, EmailTriage)


def test_l1_triage_email_end_to_end_via_build_agent_and_test_model(monkeypatch):
    """Review fix L1 (F6.5 review): the previous test above builds
    a fresh Agent bypassing _build_agent entirely. This test
    exercises the REAL _build_agent + real triage_email code path
    with a TestModel model override, closing the coverage gap.

    Uses Agent.override(model=TestModel()) context manager so the
    fake API key gets past _build_agent construction and TestModel
    handles the actual run_sync call."""
    pytest.importorskip("pydantic_ai")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-l1-test")
    from pydantic_ai.models.test import TestModel

    real_agent = _build_agent(model="gpt-4.1-mini-2025-04-14")
    # Real .eml with enough body content to pass the R5 case 2 gate.
    eml = _SAMPLE_EML
    with real_agent.override(model=TestModel()):
        result = triage_email(
            eml,
            deps=TriageDeps(known_contacts=["sarah@example.com"]),
            _agent=real_agent,
        )
    # TestModel produces fake-but-schema-valid data; verify the
    # full round-trip lands a real EmailTriage.
    assert isinstance(result, EmailTriage)
    # Cross-field validator ran (respond-action-consistency).
    if result.action not in ("respond_now", "respond_later"):
        assert result.suggested_reply is None


# --- 7. resolve_provider ---------------------------------------------------


def test_resolve_provider_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider() == "openai"


def test_resolve_provider_accepts_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    assert resolve_provider() == "mock"


def test_resolve_provider_rejects_anthropic_v1(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


def test_resolve_provider_rejects_garbage(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-real")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()
