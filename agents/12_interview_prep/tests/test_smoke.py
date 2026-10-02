"""Smoke tests for the interview-prep agent.

All tests run under LLM_PROVIDER=mock (R8 in CONTRIBUTING.md) -- CI
never touches a real API key. Real-provider tests are a manual
maintainer check before shipping.

Covers:

1. Mock path returns a valid InterviewSession (no SDK involved),
   exercising the SAME _build_session_report aggregation the real
   path uses.
2. R5 case 1 (JD-validation gate): too-short JD rejected; bad
   max_topics/max_followups rejected.
3. R5 case 2 (two-tier safety caps):
   - follow-up cap forces a topic change WITHOUT ending the session
   - topic cap ends the session with ended_reason="cap_reached"
4. R5 case 3 (_translate_api_error): all 6 priority paths.
5. _build_session_report pure-function tests (shift-by-one scoring
   pairing, topic grouping, improvement-area extraction).
6. _build_agent structural test (no tools, correct output_type).
7. resolve_provider (default, env, rejects unknown).
8. Ollama-specific: local-endpoint wiring + connection-refused hint.
9. Turn-1 non-ask_new_topic forcing, EOF/KeyboardInterrupt handling,
   and the _status_line feedback reminder -- three real implementation
   gaps found via live local-model testing (see agent.py for the
   root-cause writeups).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Import openai-agents SDK BEFORE inserting workspace agents/ dir on
# sys.path -- otherwise workspace agents/ (a namespace package that
# gets populated by our importlib call below) shadows the pip-
# installed `agents` package (openai-agents SDK). Importing SDK first
# caches it in sys.modules so subsequent `from agents import X` in
# test functions resolves to the SDK, not the workspace. Same L-13
# lesson #04's test file documents.
from agents import Agent, Runner

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_AGENT_DIR.parent))

_agent = importlib.import_module("12_interview_prep.agent")
_schemas = importlib.import_module("12_interview_prep.schemas")

run_interview = _agent.run_interview
InterviewPrepError = _agent.InterviewPrepError
InterviewPrepAttempt = _agent.InterviewPrepAttempt
resolve_provider = _agent.resolve_provider
_build_agent = _agent._build_agent
_build_session_report = _agent._build_session_report
_translate_api_error = _agent._translate_api_error
_status_line = _agent._status_line
MIN_JD_CHARS = _agent.MIN_JD_CHARS
InterviewTurn = _schemas.InterviewTurn
InterviewSession = _schemas.InterviewSession
AnswerFeedback = _schemas.AnswerFeedback
TurnRecord = _schemas.TurnRecord


_REAL_JD = """Senior Backend Engineer -- Payments Platform

Own services that process transactions at scale. Design and evolve
REST APIs. Participate in on-call, drive postmortems. Deep Postgres
experience required. Mentor junior engineers."""


class _FakeResult:
    """Minimal stand-in for openai-agents' RunResult -- exposes only
    the two methods run_interview's loop actually calls."""

    def __init__(self, turn: InterviewTurn):
        self._turn = turn

    def final_output_as(self, cls, raise_if_incorrect_type=True):
        assert cls is InterviewTurn
        return self._turn

    def to_input_list(self):
        return []


def _turn(action, topic="Some topic", requirement="some requirement", question="Q?", score=None, gaps=None):
    feedback = None
    if score is not None:
        feedback = AnswerFeedback(
            score=score, strengths=[], gaps=gaps or [], improvement_tip="tip"
        )
    return InterviewTurn(
        action=action,
        topic=topic,
        requirement_from_jd=requirement,
        question=None if action == "conclude" else question,
        feedback_on_previous_answer=feedback,
        role_title="Mock Role" if score is None and action != "conclude" else None,
    )


# --- 1. Mock path ------------------------------------------------------------


def test_mock_path_returns_valid_session(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    session = run_interview(_REAL_JD)
    assert isinstance(session, InterviewSession)
    assert session.ended_reason == "concluded"
    assert len(session.topics) == 2
    assert len(session.transcript) == 4
    assert session.role_title == "Mock Backend Engineer"


def test_mock_path_encodes_input_length(monkeypatch):
    """Character count encoded into a gap string so a future refactor
    that makes mock output constant regardless of input surfaces
    here (same convention as #04's _mock_result)."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    session = run_interview("x" * 500)
    assert any("500" in area for area in session.top_improvement_areas)


def test_mock_path_serializable_to_json(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    session = run_interview(_REAL_JD)
    dumped = session.model_dump_json()
    restored = InterviewSession.model_validate_json(dumped)
    assert restored == session


def test_mock_path_skips_jd_validation(monkeypatch):
    """Mock mode short-circuits before _looks_like_a_jd -- a 2-char
    input is accepted (no InterviewPrepError raised)."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    session = run_interview("hi")
    assert isinstance(session, InterviewSession)


# --- 2. R5 case 1: input validation ------------------------------------------


def test_r5_case1_short_jd_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(InterviewPrepError, match="too short"):
        run_interview("too short")


def test_r5_case1_message_names_thresholds(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(InterviewPrepError) as exc_info:
        run_interview("short")
    assert str(MIN_JD_CHARS) in exc_info.value.message
    assert "5 chars" in exc_info.value.message


def test_r5_case1_rejects_zero_max_topics(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(InterviewPrepError, match="max_topics"):
        run_interview(_REAL_JD, max_topics=0)


def test_r5_case1_rejects_negative_max_followups(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(InterviewPrepError, match="max_followups_per_topic"):
        run_interview(_REAL_JD, max_followups_per_topic=-1)


# --- 3. R5 case 2: two-tier safety caps --------------------------------------


def test_r5_case2a_followup_cap_forces_topic_change_not_session_end(monkeypatch):
    """max_followups_per_topic=1: after 1 confirmed follow-up, a
    second ask_followup attempt must be FORCED into a topic change --
    the session must NOT end early from this alone."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    turns = [
        _turn("ask_new_topic", topic="Topic A"),  # turn 1: establishes topic 0
        _turn("ask_followup", topic="Topic A", score=40, gaps=["thin"]),  # allowed (0 < 1)
        _turn("ask_followup", topic="Topic A", score=45, gaps=["still thin"]),  # forced -> new topic
        _turn("conclude", topic="Topic B", score=90),
    ]
    fake_results = [_FakeResult(t) for t in turns]
    monkeypatch.setattr(Runner, "run_sync", staticmethod(Mock(side_effect=fake_results)))

    answers = iter(["A1", "A2", "A3"])
    session = run_interview(
        _REAL_JD,
        max_topics=5,
        max_followups_per_topic=1,
        _agent=object(),
        answer_source=lambda turn: next(answers),
    )
    assert session.ended_reason == "concluded"
    assert len(session.topics) == 2  # forced switch really happened


def test_r5_case2b_topic_cap_ends_session_with_cap_reached(monkeypatch):
    """max_topics=1: turn 1 establishes the only allowed topic; a
    SECOND ask_new_topic attempt must be forced to conclude, and
    ended_reason must reflect that it wasn't a natural conclude."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    turns = [
        _turn("ask_new_topic", topic="Topic A"),  # turn 1: establishes topic 0 -- must NOT be capped
        _turn("ask_new_topic", topic="Topic B", score=80),  # forced -> conclude (cap already at max)
    ]
    fake_results = [_FakeResult(t) for t in turns]
    monkeypatch.setattr(Runner, "run_sync", staticmethod(Mock(side_effect=fake_results)))

    answers = iter(["A1"])
    session = run_interview(
        _REAL_JD,
        max_topics=1,
        _agent=object(),
        answer_source=lambda turn: next(answers),
    )
    assert session.ended_reason == "cap_reached"
    assert len(session.topics) == 1  # only the first topic was ever established


def test_first_turn_ask_new_topic_never_capped_even_with_max_topics_one(monkeypatch):
    """Regression guard for the off-by-one this test suite caught
    during implementation: turn 1 establishing topic 0 must succeed
    even when max_topics=1, not be immediately force-concluded before
    any topic exists."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    turns = [_turn("ask_new_topic", topic="Only topic"), _turn("conclude", score=70)]
    fake_results = [_FakeResult(t) for t in turns]
    monkeypatch.setattr(Runner, "run_sync", staticmethod(Mock(side_effect=fake_results)))

    answers = iter(["A1"])
    session = run_interview(
        _REAL_JD, max_topics=1, _agent=object(), answer_source=lambda turn: next(answers)
    )
    assert session.ended_reason == "concluded"
    assert len(session.topics) == 1


@pytest.mark.parametrize("bad_first_action", ["ask_followup", "conclude"])
def test_turn1_non_ask_new_topic_action_is_forced(monkeypatch, bad_first_action):
    """A schema-valid but prompt-noncompliant first turn
    (action='ask_followup' or 'conclude' instead of the required
    'ask_new_topic') would otherwise leave topic_index at its -1
    sentinel, crashing TurnRecord's ge=0 validation with an uncaught
    pydantic.ValidationError. Turn 1 must always be treated as
    establishing topic 0, regardless of what the model actually
    returned."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    turns = [
        _turn(bad_first_action, topic="Whatever"),
        _turn("conclude", score=70),
    ]
    fake_results = [_FakeResult(t) for t in turns]
    monkeypatch.setattr(Runner, "run_sync", staticmethod(Mock(side_effect=fake_results)))

    answers = iter(["A1"])
    session = run_interview(
        _REAL_JD, _agent=object(), answer_source=lambda turn: next(answers)
    )
    # No crash, and topic_index correctly landed on 0 (not -1) for turn 1.
    assert session.transcript[0].topic_index == 0
    assert len(session.topics) == 1


def test_answer_source_eof_raises_interview_prep_error_with_partial(monkeypatch):
    """answer_source raising EOFError/KeyboardInterrupt (e.g. an
    interactive session ended via Ctrl-D) must become a friendly
    InterviewPrepError with the partial transcript attached, like every
    other failure mode in run_interview, not propagate unwrapped."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    turns = [_turn("ask_new_topic", topic="Topic A")]
    fake_results = [_FakeResult(t) for t in turns]
    monkeypatch.setattr(Runner, "run_sync", staticmethod(Mock(side_effect=fake_results)))

    def _eof_answer_source(turn):
        raise EOFError("stdin closed")

    with pytest.raises(InterviewPrepError) as exc_info:
        run_interview(_REAL_JD, _agent=object(), answer_source=_eof_answer_source)
    assert exc_info.value.partial is not None
    assert exc_info.value.partial.topic_index == 0


def test_status_line_feedback_reminder_only_when_requested():
    """Two real implementation gaps surfaced via live local-model
    testing (not model-capability limitations), same root cause both
    times: this model's adherence to an instruction stated only once in
    the system prompt degrades once conversation history accumulates.

    1. feedback_on_previous_answer was reliably null on every non-first
       turn against gemma4:e4b/qwen2.5vl:7b/qwen2.5:7b. Fixed by
       restating the instruction in the per-turn user message.
    2. score clustered in a 2-4/100 band regardless of answer quality
       (an isolated single-call test scored an "excellent" answer 90;
       the same answer through the real multi-turn flow scored 4). A/B
       tested directly: adding explicit 0-100 anchors to the SAME
       per-turn reminder took the score from 4 to 82 for an identical
       answer. Fixed the same way.

    This test locks both reminders in place so a future refactor can't
    silently drop either from the turn-1 or post-answer call sites in
    agent.py/ui.py."""
    without_reminder = _status_line(
        topic_index=0, followups_this_topic=0, max_topics=5, max_followups_per_topic=2,
    )
    assert "feedback_on_previous_answer" not in without_reminder
    assert "0-100" not in without_reminder

    with_reminder = _status_line(
        topic_index=0, followups_this_topic=0, max_topics=5, max_followups_per_topic=2,
        include_feedback_reminder=True,
    )
    assert "feedback_on_previous_answer" in with_reminder
    assert "MUST" in with_reminder
    assert "0-100" in with_reminder
    assert "70+" in with_reminder


# --- 4. R5 case 3: _translate_api_error --------------------------------------


def test_translate_api_error_rate_limit_by_status():
    class E(Exception):
        status_code = 429
    result = _translate_api_error(E("body"))
    assert isinstance(result, InterviewPrepError)
    assert "rate-limited" in result.message.lower()


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


def test_translate_api_error_message_fallback_rate_limit():
    result = _translate_api_error(RuntimeError("You hit the rate limit for now"))
    assert "rate-limited" in result.message.lower()


def test_translate_api_error_message_fallback_auth():
    result = _translate_api_error(RuntimeError("invalid api key provided"))
    assert "authentication" in result.message.lower()


def test_translate_api_error_connection_refused_returns_ollama_hint():
    exc = ConnectionError("Connection refused to http://localhost:11434/v1/chat")
    err = _translate_api_error(exc)
    assert "ollama serve" in str(err.message).lower()


def test_translate_api_error_generic_fallback_preserves_original():
    result = _translate_api_error(RuntimeError("some genuinely unexpected thing"))
    assert "some genuinely unexpected thing" in result.message


# --- 5. _build_session_report pure-function tests ----------------------------


def test_build_session_report_scoring_is_shift_by_one():
    """Feedback on transcript[i]'s answer arrives in
    transcript[i+1].turn.feedback_on_previous_answer -- verify the
    score attributed to topic 0 comes from the SECOND record, not
    the first (which has no feedback yet, being turn 1)."""
    t0 = _turn("ask_new_topic", topic="T0")
    t1 = _turn("conclude", topic="T0", score=42, gaps=["gap-for-topic-0"])
    transcript = [
        TurnRecord(topic_index=0, turn=t0, candidate_answer="ans"),
        TurnRecord(topic_index=0, turn=t1, candidate_answer=None),
    ]
    session = _build_session_report(transcript, "concluded")
    assert len(session.topics) == 1
    assert session.topics[0].best_score == 42
    assert "gap-for-topic-0" in session.top_improvement_areas


def test_build_session_report_groups_by_topic_index_not_label():
    """Two records sharing topic_index=0 but with DIFFERENT `topic`
    label strings must still be grouped together (topic_index is
    load-bearing, the label is display-only)."""
    t0 = _turn("ask_new_topic", topic="Label A")
    t1 = _turn("conclude", topic="Totally different label", score=60)
    transcript = [
        TurnRecord(topic_index=0, turn=t0, candidate_answer="ans"),
        TurnRecord(topic_index=0, turn=t1, candidate_answer=None),
    ]
    session = _build_session_report(transcript, "concluded")
    assert len(session.topics) == 1
    assert session.topics[0].questions_asked == 2


def test_build_session_report_topic_label_comes_from_establishing_turn():
    """A real bug surfaced via live testing with a customer-success JD:
    when run_interview's two-tier cap force-
    converts a genuine ask_new_topic into "conclude" (topic budget
    already exhausted), that final record still carries the model's
    freshly-generated topic/requirement_from_jd describing the NEW
    topic it wanted to move to -- reproduced live as a group that was
    actually about "identifying at-risk accounts" getting reported as
    "Running onboarding for new accounts" because the LAST record in
    the group (the forced-conclude turn) had that unrelated label.
    The label must come from the FIRST record in the group (the one
    that genuinely established topic_index via an unforced
    ask_new_topic), not the last."""
    established = _turn(
        "ask_new_topic", topic="Identifying at-risk accounts",
        requirement="proactively identify at-risk accounts using usage data",
    )
    followup = _turn(
        "ask_followup", topic="Identifying at-risk accounts",
        requirement="proactively identify at-risk accounts using usage data",
        score=30, gaps=["vague"],
    )
    # Simulates a genuine ask_new_topic that got force-converted to
    # "conclude" by the topic-cap backstop: the model's own topic/
    # requirement fields describe a topic that was NEVER discussed
    # under this topic_index.
    mislabeled_forced_conclude = _turn(
        "conclude", topic="Running onboarding for new accounts",
        requirement="Run onboarding for new accounts: kickoff call, success plan",
        score=15, gaps=["off-topic"],
    )
    transcript = [
        TurnRecord(topic_index=0, turn=established, candidate_answer="ans1"),
        TurnRecord(topic_index=0, turn=followup, candidate_answer="ans2"),
        TurnRecord(topic_index=0, turn=mislabeled_forced_conclude, candidate_answer=None),
    ]
    session = _build_session_report(transcript, "cap_reached")
    assert len(session.topics) == 1
    assert session.topics[0].topic_label == "Identifying at-risk accounts"
    assert session.topics[0].requirement_from_jd == (
        "proactively identify at-risk accounts using usage data"
    )


def test_build_session_report_overall_score_is_mean_of_topic_bests():
    t0 = _turn("ask_new_topic", topic="T0")
    t1 = _turn("ask_new_topic", topic="T1", score=100)
    t2 = _turn("conclude", topic="T1", score=50)
    transcript = [
        TurnRecord(topic_index=0, turn=t0, candidate_answer="a0"),
        TurnRecord(topic_index=1, turn=t1, candidate_answer="a1"),
        TurnRecord(topic_index=1, turn=t2, candidate_answer=None),
    ]
    session = _build_session_report(transcript, "concluded")
    assert len(session.topics) == 2
    assert session.overall_readiness_score == 75  # mean(100, 50)


def test_build_session_report_role_title_taken_from_first_nonnull():
    t0 = InterviewTurn(
        action="ask_new_topic", topic="T0", requirement_from_jd="r",
        question="Q", role_title="Extracted Title",
    )
    t1 = _turn("conclude", score=80)
    transcript = [
        TurnRecord(topic_index=0, turn=t0, candidate_answer="a"),
        TurnRecord(topic_index=0, turn=t1, candidate_answer=None),
    ]
    session = _build_session_report(transcript, "concluded")
    assert session.role_title == "Extracted Title"


def test_build_session_report_empty_transcript_is_safe():
    session = _build_session_report([], "concluded")
    assert session.topics == []
    assert session.overall_readiness_score == 0
    assert session.top_improvement_areas == []


# --- 6. _build_agent structural check ----------------------------------------


def test_build_agent_returns_agent_with_no_tools(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    agent = _build_agent(model="gpt-4.1-mini-2025-04-14")
    assert isinstance(agent, Agent)
    assert agent.name == "interview-prep-agent"
    assert agent.output_type is InterviewTurn
    assert agent.tools == []


# --- 7. resolve_provider ------------------------------------------------------


def test_resolve_provider_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider() == "openai"


def test_resolve_provider_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert resolve_provider() == "ollama"


def test_resolve_provider_allows_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    assert resolve_provider() == "mock"


def test_resolve_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


# --- 8. Ollama-specific -------------------------------------------------------


def test_supported_providers_includes_ollama():
    assert _agent.SUPPORTED_PROVIDERS == ("openai", "ollama")


def test_build_agent_ollama_uses_local_endpoint(monkeypatch):
    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeClient)

    agent = _build_agent(model="gemma4:e4b", provider="ollama")
    assert captured["base_url"].endswith("/v1")
    assert "11434" in captured["base_url"]
    assert captured["api_key"] == "ollama"
    assert agent is not None
