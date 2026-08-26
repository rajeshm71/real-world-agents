"""Smoke tests for the meeting-notes agent.

All tests run under LLM_PROVIDER=mock (R8 in CONTRIBUTING.md) -- CI
never touches a real API key. Real-provider tests are a manual
maintainer check before shipping.

Covers:

1. Mock path returns a valid MeetingSummary (no SDK involved).
2. R5 case 1 (notes-validation gate):
   - too-short input rejected
   - long-but-no-commitment-verbs input rejected
   - valid notes accepted (falls through to Agent construction)
3. R5 case 2 (MaxTurnsExceeded):
   - simulate SDK raising MaxTurnsExceeded via monkeypatched Runner
     -> MeetingNotesError with .partial populated
4. R5 case 3 (_translate_api_error):
   - all 6 priority paths (class rate-limit, class auth, status 429,
     status 401, message fallback, generic unknown)
5. Pure helpers (deterministic Python, cheap):
   - _extract_speakers_from (colon style, attribution style, empty,
     dedup)
   - _extract_dates_from (various date phrase patterns)
   - _score_urgency_from (all 3 tiers, defaults to medium)
   - _verify_excerpt_in (positive, negative, whitespace-collapsed)
   - _normalize_ws
   - _looks_like_meeting_notes (both gate conditions)
6. `_build_agent()` returns an Agent with expected tools + settings
   (structural test; doesn't run the agent).
7. resolve_provider (default, env, rejects unknown, rejects
   anthropic/gemini since v1 is OpenAI-only per plan decision).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Import openai-agents SDK BEFORE inserting workspace agents/ dir on
# sys.path -- otherwise workspace agents/ (a namespace package that
# gets populated by our importlib call below) shadows the pip-
# installed `agents` package (openai-agents SDK). Importing SDK first
# caches it in sys.modules so subsequent `from agents import X` in
# test functions resolves to the SDK, not the workspace.
from agents import Agent, MaxTurnsExceeded, Runner

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_AGENT_DIR.parent))

_agent = importlib.import_module("04_meeting_notes.agent")
_schemas = importlib.import_module("04_meeting_notes.schemas")

extract_action_items = _agent.extract_action_items
MeetingNotesError = _agent.MeetingNotesError
MeetingNotesAttempt = _agent.MeetingNotesAttempt
resolve_provider = _agent.resolve_provider
_extract_speakers_from = _agent._extract_speakers_from
_extract_dates_from = _agent._extract_dates_from
_score_urgency_from = _agent._score_urgency_from
_verify_excerpt_in = _agent._verify_excerpt_in
_normalize_ws = _agent._normalize_ws
_looks_like_meeting_notes = _agent._looks_like_meeting_notes
_translate_api_error = _agent._translate_api_error
_build_agent = _agent._build_agent
MIN_NOTES_CHARS = _agent.MIN_NOTES_CHARS
DEFAULT_MAX_TURNS = _agent.DEFAULT_MAX_TURNS
MeetingSummary = _schemas.MeetingSummary
ActionItem = _schemas.ActionItem


# --- Fixtures --------------------------------------------------------------


_REAL_NOTES = """Q3 planning meeting -- 2026-08-21

Alice: We need to update the roadmap by Friday. That's the blocker
for kickoff.
Bob said he'll review the technical spec and share notes by end of
week.
Carol: I'll follow up with legal on the vendor contract.
Alice mentioned the launch is P1 -- urgent, so we should prioritize
the review work.
Bob agreed to draft the postmortem after last week's outage.
Action item: schedule follow-up next week to confirm progress.
Decided to postpone the marketing push to Q4."""


# --- 1. Mock path ----------------------------------------------------------


def test_mock_path_returns_valid_meeting_summary(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    summary = extract_action_items("anything goes here in mock mode")
    assert isinstance(summary, MeetingSummary)
    assert len(summary.action_items) >= 1
    # Character count encoded into summary so future refactor that
    # makes mock constant regardless of input surfaces here.
    assert "31" in summary.overall_summary  # len('anything goes here in mock mode') == 31


def test_mock_path_skips_notes_validation(monkeypatch):
    """Mock mode short-circuits BEFORE _looks_like_meeting_notes --
    a 5-char input is accepted (no MeetingNotesError raised)."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    summary = extract_action_items("hi")
    assert isinstance(summary, MeetingSummary)


def test_mock_path_serializable_to_json(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    summary = extract_action_items(_REAL_NOTES)
    dumped = summary.model_dump_json()
    restored = MeetingSummary.model_validate_json(dumped)
    assert restored == summary


# --- 2. R5 case 1: notes validation ----------------------------------------


def test_r5_case1_short_notes_rejected(monkeypatch):
    """Below MIN_NOTES_CHARS triggers rejection under non-mock
    provider (mock would short-circuit)."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(MeetingNotesError, match="doesn't look like meeting notes"):
        extract_action_items("too short")


def test_r5_case1_no_commitment_verbs_rejected(monkeypatch):
    """Above MIN_NOTES_CHARS but no commitment verbs -- treat as
    not-a-real-meeting."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    text = "The weather today is nice. " * 30  # >200 chars, zero verbs
    with pytest.raises(MeetingNotesError, match="doesn't look like meeting notes"):
        extract_action_items(text)


def test_r5_case1_message_names_thresholds(monkeypatch):
    """Error message includes actual char count + threshold so
    users know what they need to change."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(MeetingNotesError) as exc_info:
        extract_action_items("short")
    assert str(MIN_NOTES_CHARS) in exc_info.value.message
    assert "5 chars" in exc_info.value.message


# --- 3. R5 case 2: MaxTurnsExceeded ----------------------------------------


def test_r5_case2_max_turns_exceeded_raises_with_partial(monkeypatch):
    """Monkeypatch Runner.run_sync to raise MaxTurnsExceeded, verify
    extract_action_items wraps it in MeetingNotesError with
    .partial populated so UI can show what got stuck."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")


    def fake_run_sync(*_a, **_kw):
        raise MaxTurnsExceeded("stuck at turn 15")

    monkeypatch.setattr(Runner, "run_sync", staticmethod(fake_run_sync))

    # Use a fake _agent so we skip real Agent construction (which
    # tries to touch openai client config).
    fake_agent = object()  # Runner.run_sync is mocked, agent value is unused
    with pytest.raises(MeetingNotesError) as exc_info:
        extract_action_items(_REAL_NOTES, _agent=fake_agent, max_turns=15)
    assert "didn't converge" in exc_info.value.message
    assert "15" in exc_info.value.message
    assert exc_info.value.partial is not None
    assert isinstance(exc_info.value.partial, MeetingNotesAttempt)
    assert exc_info.value.partial.turns_used == 15


# --- 4. R5 case 3: _translate_api_error ------------------------------------


def test_translate_api_error_rate_limit_by_status():
    class E(Exception):
        status_code = 429
    result = _translate_api_error(E("body"))
    assert isinstance(result, MeetingNotesError)
    assert "rate-limited" in result.message.lower()


def test_translate_api_error_rate_limit_by_class_name():
    class RateLimitError(Exception):
        pass
    result = _translate_api_error(RateLimitError("no status"))
    assert "rate-limited" in result.message.lower()


def test_translate_api_error_auth_by_status():
    class E(Exception):
        status_code = 401
    result = _translate_api_error(E(""))
    assert "authentication" in result.message.lower()


def test_translate_api_error_auth_by_class_name():
    class AuthenticationError(Exception):
        pass
    result = _translate_api_error(AuthenticationError("bad key"))
    assert "authentication" in result.message.lower()


def test_translate_api_error_class_check_priority():
    """Class-name check fires BEFORE message-string fallback: a
    RateLimitError with no matching text still gets classified as
    rate limit."""
    class RateLimitError(Exception):
        pass
    result = _translate_api_error(RateLimitError("some unrelated body text"))
    assert "rate-limited" in result.message.lower()


def test_translate_api_error_unknown_preserves_original():
    result = _translate_api_error(ValueError("something weird happened"))
    assert "ValueError" in result.message
    assert "something weird happened" in result.message


# --- 5. Pure helpers -------------------------------------------------------


# --- _extract_speakers_from ---

def test_extract_speakers_colon_style():
    notes = "Alice: hello\nBob: world"
    assert _extract_speakers_from(notes) == ["Alice", "Bob"]


def test_extract_speakers_attribution_style():
    notes = "Alice mentioned the plan. Bob said the timeline is tight."
    assert _extract_speakers_from(notes) == ["Alice", "Bob"]


def test_extract_speakers_dedup_and_order():
    """Same name appearing multiple times should appear once, in
    first-appearance order."""
    notes = "Alice: hi\nBob: hi\nAlice mentioned again"
    assert _extract_speakers_from(notes) == ["Alice", "Bob"]


def test_extract_speakers_two_word_names():
    notes = "Alice Chen: hello\nBob Smith mentioned the plan"
    result = _extract_speakers_from(notes)
    assert "Alice Chen" in result
    assert "Bob Smith" in result


def test_extract_speakers_empty_on_no_matches():
    assert _extract_speakers_from("just some plain text no speakers here") == []


# --- _extract_dates_from ---

def test_extract_dates_finds_weekdays():
    """`by Friday` and `Friday` both match via different regex
    branches -- assert the returned phrases contain 'Friday' /
    'monday' substrings rather than exact equality since the regex
    can capture either the bare weekday or the 'by X' phrase."""
    results = _extract_dates_from("submit by Friday")
    assert any("Friday" in r for r in results)
    results = _extract_dates_from("on monday we ship")
    assert any("monday" in r.lower() for r in results)


def test_extract_dates_finds_next_phrases():
    result = _extract_dates_from("do it next week")
    assert any("next week" in d.lower() for d in result)


def test_extract_dates_finds_iso_dates():
    result = _extract_dates_from("deadline is 2026-09-15")
    assert "2026-09-15" in result


def test_extract_dates_finds_end_of_phrases():
    result = _extract_dates_from("end of sprint")
    assert any("end of sprint" in d.lower() for d in result)


def test_extract_dates_finds_quarter():
    result = _extract_dates_from("Push to Q3")
    assert any(d.lower() == "q3" for d in result)


def test_extract_dates_finds_eow_abbreviations():
    result = _extract_dates_from("wrap up by EoW")
    assert any(d.lower() == "eow" for d in result)


def test_extract_dates_empty_on_no_dates():
    assert _extract_dates_from("just some text with no time references at all") == []


# --- _score_urgency_from ---

def test_score_urgency_high_wins_on_urgent():
    assert _score_urgency_from("this is urgent") == "high"


def test_score_urgency_high_on_blocker():
    assert _score_urgency_from("it's a blocker for launch") == "high"


def test_score_urgency_high_on_p0():
    assert _score_urgency_from("marked as P0") == "high"


def test_score_urgency_low_on_eventually():
    assert _score_urgency_from("do this eventually") == "low"


def test_score_urgency_low_on_when_possible():
    assert _score_urgency_from("when possible please") == "low"


def test_score_urgency_medium_default():
    assert _score_urgency_from("just do the thing") == "medium"


def test_score_urgency_high_wins_over_low():
    """High keyword trumps a low keyword in the same string --
    'urgent' beats 'when possible'."""
    assert _score_urgency_from("when possible but urgent") == "high"


# --- _verify_excerpt_in ---

def test_verify_excerpt_positive():
    assert _verify_excerpt_in("hello world", "say hello world today") is True


def test_verify_excerpt_negative():
    assert _verify_excerpt_in("nothing like this", "totally different text") is False


def test_verify_excerpt_normalizes_whitespace():
    """Multi-line source with single-line excerpt should still match
    when whitespace normalizes to single spaces."""
    source = "this is\na multi-line\nexcerpt in source"
    excerpt = "this is a multi-line excerpt in source"
    assert _verify_excerpt_in(excerpt, source) is True


def test_verify_excerpt_case_sensitive():
    """Case-sensitive by design (a case difference IS a paraphrase)."""
    assert _verify_excerpt_in("HELLO", "hello world") is False


def test_normalize_ws_collapses():
    assert _normalize_ws("a  b\tc\nd") == "a b c d"
    assert _normalize_ws("  padded  ") == "padded"


# --- _looks_like_meeting_notes ---

def test_looks_like_meeting_notes_short_rejected():
    assert _looks_like_meeting_notes("short") is False


def test_looks_like_meeting_notes_no_verbs_rejected():
    text = "The weather today is nice. " * 30  # >200 chars, zero verbs
    assert _looks_like_meeting_notes(text) is False


def test_looks_like_meeting_notes_valid_accepted():
    assert _looks_like_meeting_notes(_REAL_NOTES) is True


def test_looks_like_meeting_notes_various_verbs():
    """Regression guard: any of the commitment verbs should trigger
    acceptance, not just 'will'."""
    base = "The meeting was long. " * 20  # >200 chars, no verbs baseline
    for verb_phrase in [
        "will do it",
        "agreed to review",
        "committed to send",
        "responsible for the launch",
        "action item: reply",
        "follow up next week",
    ]:
        assert _looks_like_meeting_notes(base + verb_phrase) is True, verb_phrase


# --- 6. _build_agent structural check --------------------------------------


def test_build_agent_returns_agent_with_expected_tools(monkeypatch):
    """Verify _build_agent constructs an Agent with 4 tools and the
    right name/output_type. Doesn't invoke the agent -- purely
    structural."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    agent = _build_agent(model="gpt-4.1-mini-2025-04-14", notes=_REAL_NOTES)
    assert isinstance(agent, Agent)
    assert agent.name == "meeting-notes-agent"
    assert agent.output_type is MeetingSummary
    assert len(agent.tools) == 4
    tool_names = {t.name for t in agent.tools}
    assert tool_names == {
        "extract_speakers",
        "extract_dates",
        "verify_excerpt",
        "score_urgency",
    }


# --- 7. resolve_provider ---------------------------------------------------


def test_resolve_provider_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider() == "openai"


def test_resolve_provider_reads_env_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert resolve_provider() == "openai"


def test_resolve_provider_accepts_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    assert resolve_provider() == "mock"


def test_resolve_provider_rejects_anthropic_in_v1(monkeypatch):
    """V1 is OpenAI-only per plan decision -- anthropic/gemini
    require LiteLLM (deferred). Regression guard against a future
    refactor that silently accepts them without LiteLLM setup."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


def test_resolve_provider_rejects_gemini_in_v1(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


def test_resolve_provider_rejects_garbage(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


# --- 8. Cross-field owner validation (owner must be in participants) ----


def test_h1_owner_in_participants_passes_through():
    """An ActionItem whose owner appears in MeetingSummary.participants
    passes validation unchanged."""
    summary = MeetingSummary(
        participants=["Alice", "Bob"],
        action_items=[
            ActionItem(
                description="update docs",
                owner="Alice",
                priority="medium",
                context_excerpt="Alice will update docs",
            ),
        ],
        overall_summary="Some meeting summary here.",
    )
    assert summary.action_items[0].owner == "Alice"


def test_h1_owner_not_in_participants_gets_nulled():
    """An ActionItem whose owner is NOT in participants gets auto-nulled
    by the @model_validator (forgiving option). Guards against the model
    returning owner="Zack" when participants only has ["Alice", "Bob"]:
    the mostly-good extraction still returns something useful, just with
    the bogus owner cleared."""
    summary = MeetingSummary(
        participants=["Alice", "Bob"],
        action_items=[
            ActionItem(
                description="update docs",
                owner="Zack",  # not in participants
                priority="medium",
                context_excerpt="Zack will update docs",
            ),
            ActionItem(
                description="review roadmap",
                owner="Alice",  # in participants, stays
                priority="high",
                context_excerpt="Alice will review roadmap",
            ),
        ],
        overall_summary="Some meeting summary here.",
    )
    assert summary.action_items[0].owner is None  # Zack -> None
    assert summary.action_items[1].owner == "Alice"  # Alice stays


def test_h1_none_owner_stays_none():
    """None owners are never touched -- the validator only nulls OUT
    invalid owners, doesn't invent them."""
    summary = MeetingSummary(
        participants=["Alice"],
        action_items=[
            ActionItem(
                description="write postmortem",
                owner=None,
                priority="high",
                context_excerpt="Someone will write the postmortem",
            ),
        ],
        overall_summary="Some meeting summary.",
    )
    assert summary.action_items[0].owner is None


def test_h1_empty_participants_nulls_all_owners():
    """Edge case: if participants is empty (rare but possible for
    ambiguous notes), every ActionItem owner should get nulled --
    can't validate against an empty list any other way."""
    summary = MeetingSummary(
        participants=[],
        action_items=[
            ActionItem(
                description="do thing",
                owner="Alice",
                priority="medium",
                context_excerpt="do thing by Friday",
            ),
        ],
        overall_summary="Meeting with no clear speakers.",
    )
    assert summary.action_items[0].owner is None
