"""Interview-prep agent -- agent #12 of real-world-agents.

Technique demonstrated: **adaptive multi-turn dialogue with real-time
answer assessment**, via the OpenAI Agents SDK. Unlike agents #04 and
#08 (also OpenAI Agents SDK), the whole interaction is NOT one
`Runner.run_sync` call -- it's our own Python loop, one
`Runner.run_sync` call per turn, because a real human answer has to
land between turns:

    Turn 1: agent asks Q1 (grounded in a JD requirement)
    -> candidate answers A1
    Turn 2: agent assesses A1, decides ask_followup / ask_new_topic /
            conclude, asks Q2 if not concluding
    -> candidate answers A2
    Turn 3: agent assesses A2, ...
    ...

Conversation state carries across these separate calls via the SDK's
`RunResult.to_input_list()` (confirmed against the SDK's own reference
REPL implementation, `agents/repl.py`'s `run_demo_loop` -- the exact
same "get input_items, append the next user message, call Runner
again" shape). Confirmed empirically that a *structured*
`output_type` (not just plain text, which is all the reference REPL
uses) round-trips correctly through this continuation.

Why this technique for this use case: a real interview doesn't have
one fixed question list -- how deep to drill on a topic and when to
move on depends on how well the candidate actually answered. Turning
that into a technique means: no tools (the JD text is already in
context, nothing external to ground against), no giant `max_turns`
ReAct loop (a human has to answer between every step), and a
deliberately bounded, code-owned pair of safety counters so an
LLM that ignores its own budget can't turn a mock interview into an
infinite loop.

Real error handling (R5 in CONTRIBUTING.md's hard rules):
  1. JD text too short to generate meaningful questions from ->
     InterviewPrepError raised before any LLM call.
  2. Two code-owned counters (topic_index, followups_this_topic) bound
     the interview even if the model ignores its own instructions.
     Only exhausting the total-topics budget ends the whole session;
     exhausting the per-topic follow-up budget only forces a topic
     change. See `run_interview`'s loop body.
  3. Rate limit / auth / API failure -> _translate_api_error, same
     6-branch shape as agents #02-#04/#08.

Provider strategy: OpenAI by default, LLM_PROVIDER=ollama supported
(routes through Ollama's OpenAI-compat surface, same pattern as #04).
Anthropic/Gemini via LiteLLM documented as a future one-line swap, not
implemented in v1 -- same stance as #04/#08.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Dual-mode import (same rationale as #01-#04/#08).
try:
    from .schemas import AnswerFeedback, InterviewSession, InterviewTurn, TopicSummary, TurnRecord
except ImportError:
    from schemas import AnswerFeedback, InterviewSession, InterviewTurn, TopicSummary, TurnRecord

from common.llm import resolve_model

# --- Provider + constants ---------------------------------------------------

SUPPORTED_PROVIDERS = ("openai", "ollama")

_DEFAULT_MODEL_BY_PROVIDER = {
    # No "openai" entry: falls through to resolve_model("openai") in
    # run_interview's lookup chain (below), which is resolved lazily
    # at call time (respects OPENAI_DEFAULT_MODEL if a caller sets it
    # after import) rather than baked in at module-import time.
    # Resolves to gpt-4.1-mini-2025-04-14 by default -- already priced
    # in common/pricing.py, unlike #04/#08's undated "gpt-4o-mini".
    "ollama": "gemma4:e4b",
}

MIN_JD_CHARS = 200  # below this, treat as not-a-real-JD (R5 case 1)
DEFAULT_MAX_TOPICS = 5
DEFAULT_MAX_FOLLOWUPS_PER_TOPIC = 2

# Named rather than inlined so their meaning is discoverable from
# _build_session_report's body without re-deriving it.
_LOWEST_SCORING_TOPICS_FOR_IMPROVEMENT_AREAS = 2
_MAX_TOP_IMPROVEMENT_AREAS = 5

_PROMPT_PATH = Path(__file__).parent / "prompts" / "interviewer.txt"


def resolve_provider() -> str:
    """LLM_PROVIDER env var, default "openai". "mock" is always
    accepted regardless of SUPPORTED_PROVIDERS (handled by the
    caller, not here) -- same contract as #04."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}. "
            "Multi-provider support via LiteLLM is a documented follow-up "
            "-- see the README."
        )
    return provider


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# --- Error type --------------------------------------------------------------


@dataclass
class InterviewPrepAttempt:
    """Partial state from a failed session (API error mid-interview).
    Attached to InterviewPrepError so the CLI/UI can show how far the
    candidate got before the failure."""

    transcript_so_far: list[TurnRecord] = field(default_factory=list)
    topic_index: int = 0


class InterviewPrepError(Exception):
    """Raised on any user-facing session failure. `message` is
    user-friendly; `partial` carries what we got before giving up."""

    def __init__(self, message: str, partial: InterviewPrepAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- R5 case 1: JD validation -------------------------------------------------


def _looks_like_a_jd(jd_text: str) -> bool:
    """R5 case 1 gate. Cheap length check only -- semantic
    classification ("does this actually describe a job?") is left to
    the LLM. A keyword-regex gate was tried for an almost identical
    purpose in agent #04's history and reverted for rejecting real,
    informally-phrased input -- #12 must not reintroduce that
    mistake."""
    return len(jd_text) >= MIN_JD_CHARS


# --- Status line (proactive cap awareness, every turn) -----------------------


def _status_line(
    *,
    topic_index: int,
    followups_this_topic: int,
    max_topics: int,
    max_followups_per_topic: int,
    include_feedback_reminder: bool = False,
) -> str:
    """One line appended to every turn's input so the model always
    knows the current budget -- this is the interview's PROACTIVE
    half of the two-tier safety-cap mechanism (see run_interview's
    docstring for the deterministic backstop half).

    Two real implementation gaps surfaced via live local-model testing,
    not model-capability limitations -- same root cause both times:
    this model's adherence to an instruction stated only ONCE in the
    system prompt measurably degrades once conversation history has
    accumulated; restating the instruction in the LATEST turn's own
    text reliably restores it:

    1. `feedback_on_previous_answer` was reliably null on every non-
       first turn against every local model tried (gemma4:e4b,
       qwen2.5vl:7b, qwen2.5:7b). Isolated single-call test (no
       conversation history) proved the model CAN populate it
       correctly; only the multi-turn version failed. Fixed by the
       first reminder paragraph below.
    2. Separately, `score` clustered tightly in a 2-4 (out of 100)
       range regardless of answer quality -- an isolated single-call
       test scored a genuinely excellent answer 90/100, but the same
       answer through the real multi-turn flow scored 4/100 with
       qualitative text that said "excellent." A/B tested directly:
       identical conversation history, identical answer -- adding
       explicit 0-100 anchor points to the SAME per-turn reminder
       (not just the system prompt) took the score from 4 to 82.
       Fixed by the second reminder paragraph below.

    `include_feedback_reminder=True` on every call site where a
    candidate answer was just submitted (there IS a previous answer to
    assess and score); left False on the turn-1 call site (there is
    NOT)."""
    lines = [
        (
            f"[Session status: on topic {topic_index + 1} of {max_topics} max; "
            f"{followups_this_topic} of {max_followups_per_topic} follow-up(s) "
            "used on the current topic. If you choose ask_followup and the "
            "follow-up budget for this topic is already used up, the tool will "
            "move to a new topic regardless of what you choose. If you choose "
            "ask_new_topic and the topic budget is already used up, the tool "
            "will conclude the interview regardless of what you choose. Plan "
            "accordingly.]"
        )
    ]
    if include_feedback_reminder:
        lines.append(
            "[Reminder: you MUST populate feedback_on_previous_answer with "
            "a real assessment object (score, strengths, gaps, "
            "improvement_tip) for the candidate's answer above. It is only "
            "allowed to be null on turn 1 -- this is not turn 1, so it must "
            "not be null.]"
        )
        lines.append(
            "[Score-scale reminder: score must use the FULL 0-100 range, "
            "not a small number. 0-20 = no real answer / off-topic; "
            "21-40 = vague or generic, no specifics; 41-60 = adequate but "
            "shallow; 61-80 = solid, with a specific example; "
            "81-100 = exceptional, specific and shows real judgment. A "
            "genuinely detailed, specific answer with a concrete example "
            "MUST score 70+, not single digits.]"
        )
    return "\n".join(lines)


def _default_answer_source(turn: InterviewTurn) -> str:
    """Default answer_source: prints the question (plus feedback on
    the previous answer, if any) and reads the candidate's typed
    answer from stdin. Tests override this parameter entirely with a
    canned-answer callable; this default is only exercised by the
    interactive CLI."""
    if turn.feedback_on_previous_answer is not None:
        fb = turn.feedback_on_previous_answer
        print(f"\n  [score: {fb.score}/100]")
        if fb.strengths:
            print(f"  strengths: {', '.join(fb.strengths)}")
        if fb.gaps:
            print(f"  gaps: {', '.join(fb.gaps)}")
        print(f"  tip: {fb.improvement_tip}")
    print(f"\n{turn.topic}")
    print(turn.question)
    return input("\nYour answer: ")


# --- Public API ----------------------------------------------------------


def run_interview(
    jd_text: str,
    background_text: str | None = None,
    *,
    max_topics: int = DEFAULT_MAX_TOPICS,
    max_followups_per_topic: int = DEFAULT_MAX_FOLLOWUPS_PER_TOPIC,
    provider: str | None = None,
    model: str | None = None,
    answer_source: Callable[[InterviewTurn], str] | None = None,
    _agent=None,  # test-injection escape hatch (bypass _build_agent)
) -> InterviewSession:
    """Run a full adaptive mock-interview session.

    Args:
        jd_text: the job description, plain text.
        background_text: optional candidate background (resume
            snippet or short bio), plain text.
        max_topics: hard ceiling on distinct topics covered. Only
            exhausting this ends the whole session
            (ended_reason="cap_reached").
        max_followups_per_topic: hard ceiling on follow-up questions
            per topic. Exhausting this only forces a topic change; it
            never ends the session by itself.
        provider: "openai" (default) / "ollama" / "mock". Defaults to
            the LLM_PROVIDER env var if not passed.
        model: model ID override for the resolved provider.
        answer_source: callable that takes the just-produced
            InterviewTurn and returns the candidate's answer as a
            string. Defaults to an interactive stdin prompt; tests
            pass a canned-answer callable instead.
        _agent: injected pre-built Agent for tests. Production callers
            leave this None.

    Returns:
        A validated InterviewSession (topics covered, full transcript,
        overall readiness score, top improvement areas).

    Raises:
        InterviewPrepError: on any of the 3 R5 failure modes.
    """
    resolved_provider = provider or resolve_provider()

    if resolved_provider == "mock":
        return _mock_result(jd_text, background_text)

    # R5 case 1: JD too short to generate meaningful questions from.
    # Fails fast before any Agent construction so bogus input costs
    # nothing.
    if not _looks_like_a_jd(jd_text):
        raise InterviewPrepError(
            f"Job description is too short (got {len(jd_text)} chars, need "
            f"at least {MIN_JD_CHARS}). Paste the real job description."
        )
    if max_topics < 1:
        raise InterviewPrepError(f"max_topics must be >= 1, got {max_topics}.")
    if max_followups_per_topic < 0:
        raise InterviewPrepError(
            f"max_followups_per_topic must be >= 0, got {max_followups_per_topic}."
        )

    resolved_model = model or _DEFAULT_MODEL_BY_PROVIDER.get(
        resolved_provider
    ) or resolve_model(resolved_provider)

    # Lazy import: openai-agents pulls in openai + other deps. Mock
    # mode + JD-validation path should not require it installed.
    try:
        from agents import Runner
    except ImportError as exc:
        raise InterviewPrepError(
            "openai-agents is not installed. Run `uv sync` at the workspace "
            "root, or `pip install openai-agents>=0.20,<0.21`."
        ) from exc

    agent = _agent if _agent is not None else _build_agent(
        model=resolved_model, provider=resolved_provider
    )
    answer_fn = answer_source or _default_answer_source

    # -1 is a sentinel meaning "no topic established yet" -- turn 1's
    # ask_new_topic (which ESTABLISHES the first topic, index 0) must
    # never itself be capped. Only a LATER ask_new_topic that would
    # push topic_index past max_topics-1 gets forced to conclude. See
    # the cap-check below; -1 is never used for display (clamped to 0
    # in _status_line calls).
    topic_index = -1
    followups_this_topic = 0
    transcript: list[TurnRecord] = []

    jd_block = f"Job description:\n\n{jd_text.strip()}"
    if background_text:
        jd_block += f"\n\nCandidate background:\n\n{background_text.strip()}"
    status = _status_line(
        topic_index=max(topic_index, 0),
        followups_this_topic=followups_this_topic,
        max_topics=max_topics,
        max_followups_per_topic=max_followups_per_topic,
    )
    current_input: str | list = f"{jd_block}\n\n{status}"

    ended_reason = "concluded"

    while True:
        try:
            # final_output_as(...) belongs inside this try/except: a
            # post-hoc schema-validation failure from the SDK must go
            # through the same R5 case 3 translator as every other
            # exception from a single "get the model's turn" step.
            result = Runner.run_sync(agent, input=current_input)
            turn = result.final_output_as(InterviewTurn, raise_if_incorrect_type=True)
        except Exception as exc:  # R5 case 3: SDK / API / rate-limit failure
            err = _translate_api_error(exc)
            # max(topic_index, 0): if the SDK call itself fails on turn
            # 1, topic_index is still its -1 sentinel (the turn-1
            # forcing logic below never got a chance to run). Clamp so
            # this diagnostic-only field never shows the internal
            # sentinel value to a caller inspecting .partial.
            err.partial = InterviewPrepAttempt(
                transcript_so_far=transcript, topic_index=max(topic_index, 0)
            )
            raise err from exc

        # --- two-tier safety-cap backstop (deterministic; the model is
        # already told the live budget every turn via _status_line, so
        # this should rarely trigger in practice) ---
        effective_action = turn.action
        # Turn 1's action MUST be treated as ask_new_topic regardless of
        # what the model actually returned. The prompt instructs this,
        # but a schema-valid "ask_followup" or "conclude" on turn 1 would
        # otherwise leave topic_index at its -1 sentinel, and
        # TurnRecord.topic_index (ge=0) would raise an uncaught
        # pydantic.ValidationError. See
        # test_turn1_non_ask_new_topic_action_is_forced.
        if topic_index == -1:
            effective_action = "ask_new_topic"
        elif effective_action == "ask_followup" and followups_this_topic >= max_followups_per_topic:
            effective_action = "ask_new_topic"  # forced: per-topic follow-up budget exhausted
        if effective_action == "ask_new_topic" and topic_index >= max_topics - 1:
            effective_action = "conclude"  # forced: total-topic budget exhausted

        if effective_action == "ask_new_topic":
            topic_index += 1
            followups_this_topic = 0
        elif effective_action == "ask_followup":
            followups_this_topic += 1

        if effective_action == "conclude":
            transcript.append(
                TurnRecord(topic_index=topic_index, turn=turn, candidate_answer=None)
            )
            if turn.action != "conclude":
                ended_reason = "cap_reached"
            break

        # answer_fn needs its own exception handling: an interactive
        # session ended by Ctrl-D/Ctrl-C (EOFError/KeyboardInterrupt from
        # the default stdin-reading answer_source) should raise a
        # friendly, partial-state-attached InterviewPrepError like every
        # other failure mode in this function, not a raw traceback.
        try:
            answer = answer_fn(turn)
        except (EOFError, KeyboardInterrupt) as exc:
            # topic_index is guaranteed >= 0 here, never the -1 sentinel --
            # the turn-1 forcing logic above (topic_index == -1 ->
            # ask_new_topic) always runs and increments it before
            # answer_fn is ever called, even on turn 1.
            raise InterviewPrepError(
                "Session ended before the candidate answered this question.",
                partial=InterviewPrepAttempt(
                    transcript_so_far=transcript, topic_index=topic_index
                ),
            ) from exc
        transcript.append(
            TurnRecord(topic_index=topic_index, turn=turn, candidate_answer=answer)
        )

        # include_feedback_reminder=True: a real candidate answer was
        # just submitted above, so feedback_on_previous_answer is now
        # required (see _status_line's docstring for why this can't be
        # left to the system prompt alone on some local models).
        status = _status_line(
            topic_index=topic_index,
            followups_this_topic=followups_this_topic,
            max_topics=max_topics,
            max_followups_per_topic=max_followups_per_topic,
            include_feedback_reminder=True,
        )
        current_input = result.to_input_list() + [
            {"role": "user", "content": f"{answer}\n\n{status}"}
        ]

    return _build_session_report(transcript, ended_reason)


def _build_session_report(transcript: list[TurnRecord], ended_reason: str) -> InterviewSession:
    """Pure-Python aggregation over a completed transcript -- no LLM
    call. Used by both the real path and `_mock_result`, so mock mode
    also exercises this logic instead of duplicating it.

    Scoring subtlety: `transcript[i+1].turn.feedback_on_previous_answer`
    assesses `transcript[i]`'s answer, NOT transcript[i+1]'s own
    question -- feedback is always about the PRECEDING turn's answer.
    So scores/gaps for `transcript[i].topic_index` are read off
    `transcript[i+1]`, a shift-by-one pairing. The final ("conclude")
    record has no question of its own but DOES carry feedback scoring
    the second-to-last record's answer, so every real Q&A turn ends up
    scored exactly once.
    """
    role_title: str | None = None
    for record in transcript:
        if record.turn.role_title:
            role_title = record.turn.role_title
            break

    by_topic: dict[int, list[TurnRecord]] = {}
    for record in transcript:
        by_topic.setdefault(record.topic_index, []).append(record)

    scores_by_topic: dict[int, list[int]] = {}
    gaps_by_topic: dict[int, list[str]] = {}
    for i in range(len(transcript) - 1):
        asked_topic = transcript[i].topic_index
        feedback = transcript[i + 1].turn.feedback_on_previous_answer
        if feedback is not None:
            scores_by_topic.setdefault(asked_topic, []).append(feedback.score)
            gaps_by_topic.setdefault(asked_topic, []).extend(feedback.gaps)

    topics: list[TopicSummary] = []
    for topic_index in sorted(by_topic):
        records = by_topic[topic_index]
        # Use the FIRST record's topic/requirement labels, not the LAST.
        # When run_interview's two-tier cap force-converts a
        # genuine ask_new_topic into "conclude" (topic budget already
        # exhausted), that final record still carries the model's
        # freshly-generated topic/requirement_from_jd fields describing
        # the NEW topic it wanted to move to -- one that was never
        # actually discussed under this topic_index. Taking the label
        # from records[-1] in that case mislabels the ENTIRE group with
        # unrelated content (reproduced: a group that was actually about
        # "identifying at-risk accounts" got reported as "Running
        # onboarding for new accounts" because the forced-conclude
        # turn's own topic field said "onboarding"). The first record in
        # a group is always the one that genuinely established
        # topic_index via an unforced ask_new_topic, so its label is
        # trustworthy; later records in the same group only ever
        # narrow/follow up on that established topic (or, in the buggy
        # case, get force-kept in the group despite describing
        # something else) -- never legitimately rename it.
        established = records[0]
        scores = scores_by_topic.get(topic_index, [])
        topics.append(
            TopicSummary(
                topic_index=topic_index,
                topic_label=established.turn.topic,
                requirement_from_jd=established.turn.requirement_from_jd,
                questions_asked=len(records),
                best_score=max(scores) if scores else 0,
            )
        )

    overall = round(sum(t.best_score for t in topics) / len(topics)) if topics else 0

    lowest = sorted(topics, key=lambda t: t.best_score)[:_LOWEST_SCORING_TOPICS_FOR_IMPROVEMENT_AREAS]
    improvement_areas: list[str] = []
    for t in lowest:
        improvement_areas.extend(gaps_by_topic.get(t.topic_index, []))

    return InterviewSession(
        role_title=role_title,
        ended_reason=ended_reason,
        topics=topics,
        transcript=transcript,
        overall_readiness_score=overall,
        top_improvement_areas=improvement_areas[:_MAX_TOP_IMPROVEMENT_AREAS],
    )


# --- Agent factory ---------------------------------------------------------


def _build_agent(*, model: str, provider: str = "openai"):
    """Build the interview-prep Agent. No tools -- the JD/background
    text is already in the prompt context, so there's nothing external
    to ground against. `output_type=InterviewTurn` is the whole
    technique: one structured decision per human turn.

    Lazy import: openai-agents is heavy; only pulled in on real-
    provider runs (mock path skips this factory entirely)."""
    from agents import Agent, ModelSettings

    if provider == "ollama":
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
        from openai import AsyncOpenAI

        from common.llm import ollama_base_url
        model_arg = OpenAIChatCompletionsModel(
            model=model,
            openai_client=AsyncOpenAI(base_url=ollama_base_url(), api_key="ollama"),
        )
        # Same mitigation #04 already proved necessary: Ollama's
        # num_predict defaults small, and temperature=0 improves JSON
        # reliability on smaller local models under a strict schema.
        agent_settings = ModelSettings(max_tokens=16384, temperature=0.0)
    else:
        model_arg = model
        agent_settings = None

    agent_kwargs: dict = {
        "name": "interview-prep-agent",
        "instructions": _load_system_prompt(),
        "model": model_arg,
        "tools": [],
        "output_type": InterviewTurn,
    }
    if agent_settings is not None:
        agent_kwargs["model_settings"] = agent_settings
    return Agent(**agent_kwargs)


# --- Error translation (R5 case 3) ------------------------------------------


def _translate_api_error(exc: Exception) -> InterviewPrepError:
    """Turn an openai-agents SDK or OpenAI SDK exception into a
    user-facing InterviewPrepError. Same 6-branch priority order as
    agent #04 (class-name first, status-code second, message-
    fallback, generic). No auto-retry for transient errors."""
    exc_class_name = type(exc).__name__.lower()
    message_lower = str(exc).lower()
    status = getattr(exc, "status_code", None)

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
    from common.llm import OLLAMA_CONNECTION_HINT, is_ollama_connection_error
    if is_ollama_connection_error(exc):
        return InterviewPrepError(OLLAMA_CONNECTION_HINT)

    return InterviewPrepError(
        f"Interview session failed: {type(exc).__name__}: {exc}. "
        "This is an unexpected error -- check the agent logs."
    )


def _rate_limit_error() -> InterviewPrepError:
    return InterviewPrepError(
        "The service is temporarily rate-limited or overloaded. "
        "Wait a minute and try again."
    )


def _auth_error() -> InterviewPrepError:
    return InterviewPrepError(
        "API authentication failed. Check that OPENAI_API_KEY is set in "
        ".env (or your shell environment)."
    )


# --- Mock mode ---------------------------------------------------------------


def _mock_result(jd_text: str, background_text: str | None) -> InterviewSession:
    """Deterministic canned InterviewSession for smoke tests and CI.
    Does NOT run the SDK, does NOT touch the network, does NOT touch
    the JD-validation R5 gate (mock is for exercising the downstream
    Pydantic + report-building pipeline). Builds a fixed 4-turn
    transcript covering 2 topics (one of which gets a follow-up) by
    hand, then feeds it through the SAME `_build_session_report(...)`
    the real path uses -- so mock mode also exercises the aggregation
    logic instead of duplicating it.

    Character count of jd_text is encoded into one gap string so a
    future refactor that makes mock output constant regardless of
    input surfaces at test time (same convention as #04's
    _mock_result)."""
    turn1 = InterviewTurn(
        action="ask_new_topic",
        topic="API design",
        requirement_from_jd="experience designing REST APIs",
        question="Tell me about a REST API you designed from scratch.",
        feedback_on_previous_answer=None,
        role_title="Mock Backend Engineer",
    )
    turn2 = InterviewTurn(
        action="ask_followup",
        topic="API design",
        requirement_from_jd="experience designing REST APIs",
        question="What specifically made that API's versioning strategy work?",
        feedback_on_previous_answer=AnswerFeedback(
            score=55,
            strengths=["Mentioned real endpoints"],
            gaps=["No mention of versioning or backward compatibility"],
            improvement_tip="Discuss how you handled breaking changes.",
        ),
    )
    turn3 = InterviewTurn(
        action="ask_new_topic",
        topic="Incident response",
        requirement_from_jd="on-call experience with production incidents",
        question="Walk me through the last production incident you were on-call for.",
        feedback_on_previous_answer=AnswerFeedback(
            score=80,
            strengths=["Clear versioning strategy", "Concrete example"],
            gaps=[],
            improvement_tip="Mention deprecation timelines next time.",
        ),
    )
    turn4 = InterviewTurn(
        action="conclude",
        topic="Incident response",
        requirement_from_jd="on-call experience with production incidents",
        question=None,
        feedback_on_previous_answer=AnswerFeedback(
            score=70,
            strengths=["Walked through the timeline"],
            gaps=[
                f"Didn't mention the postmortem process (mock input: {len(jd_text)} chars)"
            ],
            improvement_tip="Always close with what changed afterward (postmortem, runbook update).",
        ),
    )

    transcript = [
        TurnRecord(
            topic_index=0, turn=turn1,
            candidate_answer="I built a payments API at my last job.",
        ),
        TurnRecord(
            topic_index=0, turn=turn2,
            candidate_answer="We used /v1/ and /v2/ URL prefixes.",
        ),
        TurnRecord(
            topic_index=1, turn=turn3,
            candidate_answer="A database failover took down checkout for 20 minutes.",
        ),
        TurnRecord(topic_index=1, turn=turn4, candidate_answer=None),
    ]
    return _build_session_report(transcript, ended_reason="concluded")


# --- CLI entry point (uv run python -m agent) -------------------------------


def main() -> int:
    """CLI: takes a path to a job-description text file, runs an
    interactive mock interview, writes the session report to
    last_run.json, prints a compact summary table."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="interview-prep",
        description="Adaptive mock-interview practice from a job description.",
    )
    parser.add_argument("jd_path", nargs="?", help="Path to a job-description text file.")
    parser.add_argument("--background", help="Path to an optional candidate-background text file.")
    parser.add_argument(
        "--max-topics", type=int, default=DEFAULT_MAX_TOPICS,
        help=f"Max distinct topics covered (default: {DEFAULT_MAX_TOPICS}).",
    )
    parser.add_argument(
        "--max-followups", type=int, default=DEFAULT_MAX_FOLLOWUPS_PER_TOPIC,
        help=f"Max follow-ups per topic (default: {DEFAULT_MAX_FOLLOWUPS_PER_TOPIC}).",
    )
    parser.add_argument("--model", help="Override the resolved model for this invocation.")
    parser.add_argument("--provider", choices=(*SUPPORTED_PROVIDERS, "mock"), default=None)
    parser.add_argument("--ui", action="store_true", help="Launch the Gradio UI instead of a CLI session.")
    args = parser.parse_args()

    if args.ui:
        try:
            from .ui import build_ui  # type: ignore[import-not-found]
        except ImportError:
            from ui import build_ui  # type: ignore[import-not-found]
        build_ui().launch()
        return 0

    if not args.jd_path:
        parser.error("jd_path is required unless --ui is passed")

    jd_path = Path(args.jd_path)
    if not jd_path.exists():
        print(f"error: file not found: {jd_path}", file=sys.stderr)
        return 2
    jd_text = jd_path.read_text(encoding="utf-8")

    background_text = None
    if args.background:
        background_path = Path(args.background)
        if not background_path.exists():
            print(f"error: file not found: {background_path}", file=sys.stderr)
            return 2
        background_text = background_path.read_text(encoding="utf-8")

    try:
        session = run_interview(
            jd_text,
            background_text,
            max_topics=args.max_topics,
            max_followups_per_topic=args.max_followups,
            provider=args.provider,
            model=args.model,
        )
    except InterviewPrepError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.partial is not None and exc.partial.transcript_so_far:
            print(
                f"---- {len(exc.partial.transcript_so_far)} turn(s) completed before failure ----",
                file=sys.stderr,
            )
        return 1

    out_path = Path(__file__).parent / "last_run.json"
    out_path.write_text(session.model_dump_json(indent=2), encoding="utf-8")

    if session.ended_reason == "cap_reached":
        print("\n(Note: ended early -- the topic budget was reached.)")

    print(f"\n=== Session report: {session.role_title or 'role not identified'} ===")
    print(f"Overall readiness score: {session.overall_readiness_score}/100\n")
    for t in session.topics:
        print(f"  [{t.best_score:3d}/100] {t.topic_label} ({t.questions_asked} question(s))")
    if session.top_improvement_areas:
        print("\nTop improvement areas:")
        for area in session.top_improvement_areas:
            print(f"  - {area}")
    print(f"\nFull session written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
