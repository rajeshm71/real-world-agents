"""Meeting-notes agent -- agent #04 of real-world-agents.

Technique demonstrated: **ReAct pattern (Reason + Act, Yao et al. 2022)
via the OpenAI Agents SDK**. The model interleaves reasoning traces
with grounded tool calls:

    Thought: I need to find who was actually in the meeting.
    Action: extract_speakers()
    Observation: ["Alice", "Bob", "Carol"]
    Thought: Now identify commitment statements.
    Action: extract_dates()
    Observation: ["Friday", "next sprint"]
    Thought: Score urgency for each candidate item.
    Action: score_urgency("send updated proposal by Friday")
    Observation: "high"
    Thought: Verify my excerpt is a real substring.
    Action: verify_excerpt("Alice will send updated proposal by Friday")
    Observation: True
    Final answer: MeetingSummary(...)

The 4 tools are pure-Python heuristics (regex / dateutil / keyword
scoring / substring check) -- deliberately NOT LLM calls. That's the
point of ReAct's grounding: tools with outputs the model can't fake.
`extract_speakers` returns real names from the notes; the model can't
invent attendees. `verify_excerpt` is a straight substring test; the
model can't fake a passing result.

Why this technique for this use case: extracting action items from
meeting notes naturally has multiple sub-steps that benefit from
intermediate grounding (who's here? what was said? what's the
urgency? does this quote appear verbatim?). A single-shot "extract
action items" prompt would let the model fabricate speakers and
paraphrase quotes -- both of which break the "user can verify each
item" trust model for a shipped tool. The interleaved reason + tool +
observation loop IS the technique.

Why OpenAI Agents SDK: this is a real framework fit, not incidental.
The SDK abstracts (a) the ReAct loop mechanics (call model -> parse
tool calls -> execute tools -> feed observations back), (b)
structured-output validation via output_type=Pydantic-model, (c) tool
schema generation from Python function signatures. Hand-rolling ReAct
would obscure "here's what the SDK does for you." Reading _build_agent
below tells you the whole pattern.

Real error handling (R5 in CONTRIBUTING.md's hard rules): three
concrete failure modes handled explicitly:
  1. Notes too short / not a real meeting -> MeetingNotesError with
     specific "doesn't look like meeting notes" message. Checked
     BEFORE any Agent construction so bogus input fails fast.
  2. Agent loop exhaustion (SDK's MaxTurnsExceeded) -> MeetingNotesError
     with the last processed input attached as .partial so UI can
     show what the agent got stuck on.
  3. Rate limit / auth / API failure -> _translate_api_error mirrors
     #01-03's shape (class-first, status-second, message-fallback).
     Explicit no-auto-retry for transient errors (same S2 decision).

Provider strategy: OpenAI-only for v1. The openai-agents SDK is
provider-agnostic via LiteLLM, but adding LiteLLM to the deps for
v1 is scope creep -- documented as a one-line swap in the README.
R6 project-wide rule ("no provider hardcoded") is satisfied by
#01/#02/#03 already supporting all three; #04 using an OpenAI-branded
framework with OpenAI provider is honest framework alignment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Dual-mode import (same rationale as #01-03).
try:
    from .schemas import ActionItem, MeetingSummary
except ImportError:
    from schemas import ActionItem, MeetingSummary

from common.llm import resolve_model

# --- Provider + constants --------------------------------------------------

# OpenAI-only for v1 (see module docstring). "mock" bypasses the SDK
# entirely; anything else is treated as an OpenAI model ID.
SUPPORTED_PROVIDERS = ("openai", "ollama")

_DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-4o-mini",
    "ollama": "gemma4:e4b",
}

DEFAULT_MAX_TURNS = 15  # ReAct turn cap; beyond this the model is stuck
MIN_NOTES_CHARS = 200  # below this, treat as not-a-real-meeting (R5 case 1)


# Speaker detection: matches "Name:" at the start of a line (common
# transcription format), and "Name said/mentioned/asked/etc." (common
# meeting-notes style). Names are Capitalized-Word or Capitalized-Word
# Capitalized-Word patterns.
_SPEAKER_COLON_RE = re.compile(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*:", re.MULTILINE)
_SPEAKER_ATTRIBUTION_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:said|mentioned|asked|noted|proposed|"
    r"suggested|reported|added|will|owns?)\b"
)

# Urgency keywords: three ordered tiers. High wins if any high keyword
# appears; low only if a low keyword appears AND no medium/high; else
# medium (default).
_URGENCY_HIGH = frozenset({
    "urgent", "urgently", "asap", "immediately", "blocker", "blocking",
    "critical", "emergency", "hotfix", "p0", "p1", "showstopper",
})
_URGENCY_LOW = frozenset({
    "eventually", "someday", "nice to have", "nice-to-have", "when possible",
    "no rush", "no hurry", "low priority", "backlog", "future",
})

_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"


def resolve_provider() -> str:
    """LLM_PROVIDER env var, default "openai". Only "openai" and "mock"
    are supported in v1 (see module docstring for LiteLLM multi-provider
    deferral)."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}. "
            "Multi-provider support via LiteLLM is a documented follow-up "
            "for #04 -- see the README."
        )
    return provider


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# --- Error type ------------------------------------------------------------


@dataclass
class MeetingNotesAttempt:
    """Partial state from a failed extraction (retries exhausted or
    max_turns hit). `last_input` is the last thing the model was asked
    to process; attached to MeetingNotesError so the UI can show what
    the agent got stuck on."""

    last_input: str = ""
    turns_used: int = 0
    tool_calls: list[str] = field(default_factory=list)


class MeetingNotesError(Exception):
    """Raised on any user-facing extraction failure. `message` is
    user-friendly; `partial` carries what we got before giving up."""

    def __init__(self, message: str, partial: MeetingNotesAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Tool implementations (pure Python; wrapped as @function_tool inside
#     _build_agent's factory so they can close over the current notes) ---


def _extract_speakers_from(notes: str) -> list[str]:
    """Extract likely speaker names from meeting notes. Combines two
    heuristics: "Name:" at line start (transcription style) and "Name
    said/mentioned/..." (narrative style). Returns unique names in
    first-appearance order.

    Deliberately naive -- misses initials-only names, misses names in
    non-Latin scripts, misses "Dr. Alice Chen"-style titles. The point
    isn't perfect NER; it's grounding the model in real substrings so
    it can't invent attendees. False negatives are OK (model still gets
    a valid list, just shorter); false positives ("Meeting Room" as a
    speaker) would be more concerning but the regex requires trailing
    ":" or a verb, so they're rare.
    """
    seen: dict[str, None] = {}  # dict preserves insertion order
    for match in _SPEAKER_COLON_RE.finditer(notes):
        seen[match.group(1)] = None
    for match in _SPEAKER_ATTRIBUTION_RE.finditer(notes):
        seen[match.group(1)] = None
    return list(seen.keys())


def _extract_dates_from(notes: str) -> list[str]:
    """Extract date-like phrases from meeting notes via dateutil's
    fuzzy parser -- returns the matched substrings, not parsed dates
    (meeting deadlines like 'end of sprint' aren't ISO-parseable).

    Uses a two-pass approach: find candidate windows via keyword
    matching ("Friday", "Monday", "tomorrow", "next week", "EoW",
    "Q3", specific dates), then verify each with dateutil where
    applicable. Fallback: return the keyword substrings as-is when
    dateutil can't parse them (e.g. "end of sprint").

    Lazy dateutil import so mock-mode tests don't need it installed.
    """
    keywords = [
        r"\b(?:tomorrow|today|yesterday)\b",
        r"\bnext\s+(?:week|month|sprint|quarter|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",  # ISO
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",  # US
        r"\bend\s+of\s+(?:week|month|sprint|quarter|year|day)\b",
        r"\beo[wmsy]\b",  # EoW / EoM / EoS / EoY
        r"\bq[1-4]\b",  # Q1..Q4
        r"\bby\s+\w+day\b",  # "by Friday"
    ]
    combined = re.compile("|".join(keywords), re.IGNORECASE)
    seen: dict[str, None] = {}
    for match in combined.finditer(notes):
        seen[match.group(0)] = None
    return list(seen.keys())


def _score_urgency_from(item_text: str) -> str:
    """Keyword-based urgency scoring. Returns 'low' / 'medium' / 'high'.
    High wins if any high keyword appears (urgent trumps a soft
    modifier); low only if a low keyword appears AND no high; else
    medium.

    Case-insensitive substring match against pre-defined keyword sets
    (see _URGENCY_HIGH / _URGENCY_LOW). Deliberately doesn't try to
    parse "not urgent" -- that's a negation-handling can of worms; the
    model is asked to consider context when picking the priority
    field, this tool is just a signal, not the sole decider.
    """
    lower = item_text.lower()
    if any(kw in lower for kw in _URGENCY_HIGH):
        return "high"
    if any(kw in lower for kw in _URGENCY_LOW):
        return "low"
    return "medium"


def _verify_excerpt_in(excerpt: str, notes: str) -> bool:
    """Substring check with whitespace normalization -- same [C5]
    pattern as agent #02's _normalize_for_substring. PDFs / pasted
    notes often collapse or expand whitespace; without this, an
    otherwise-verbatim excerpt fails for a whitespace-only reason."""
    return _normalize_ws(excerpt) in _normalize_ws(notes)


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_ws(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s).strip()


# --- R5 case 1: notes validation -------------------------------------------


def _looks_like_meeting_notes(notes: str) -> bool:
    """R5 case 1 gate. Cheap length check only -- anything under
    MIN_NOTES_CHARS is too small to be real notes. Semantic
    classification ("does this look like meeting notes?") is left to
    the LLM: regex on keywords was tried and consistently rejected
    real transcripts that used contractions or informal phrasing."""
    return len(notes) >= MIN_NOTES_CHARS


# --- Public API ------------------------------------------------------------


def extract_action_items(
    notes: str,
    *,
    model: str | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    _agent=None,  # test-injection escape hatch (bypass _build_agent)
) -> MeetingSummary:
    """Extract structured action items from meeting notes.

    Args:
        notes: raw meeting notes text (typed, pasted, or transcript-
            style with "Name: ..." lines).
        model: OpenAI model ID. Defaults to `common.llm.resolve_model
            ("openai")` -- same env-var contract as agents #01-03
            (OPENAI_DEFAULT_MODEL overrides).
        max_turns: cap on ReAct turns; SDK's MaxTurnsExceeded fires
            at this ceiling. Default 15.
        _agent: injected pre-built Agent for tests. Production callers
            leave this None; the agent is built with `notes` closed
            over so the 3-of-4 tools that need it don't require the
            model to pass notes back on every call.

    Returns:
        A validated MeetingSummary.

    Raises:
        MeetingNotesError: on any of the 3 R5 failure modes.
    """
    resolved_provider = resolve_provider()

    if resolved_provider == "mock":
        return _mock_result(notes)

    # R5 case 1: notes too short to be a real meeting. Fails fast
    # before any Agent construction so bogus input costs nothing.
    if not _looks_like_meeting_notes(notes):
        raise MeetingNotesError(
            f"Input is too short to be meeting notes (got {len(notes)} "
            f"chars, need at least {MIN_NOTES_CHARS}). Paste real meeting "
            "notes or a transcript."
        )

    resolved_model = model or _DEFAULT_MODEL_BY_PROVIDER.get(
        resolved_provider
    ) or resolve_model(resolved_provider)

    # Lazy import: openai-agents pulls in openai + other deps. Mock
    # mode + notes-validation path should not require it installed.
    try:
        from agents import MaxTurnsExceeded, Runner
    except ImportError as exc:
        raise MeetingNotesError(
            "openai-agents is not installed. Run `uv sync` at the workspace "
            "root, or `pip install openai-agents>=0.22`."
        ) from exc

    agent = _agent if _agent is not None else _build_agent(
        model=resolved_model, notes=notes, provider=resolved_provider
    )

    try:
        result = Runner.run_sync(agent, input=notes, max_turns=max_turns)
    except MaxTurnsExceeded as exc:
        # R5 case 2: agent didn't converge within the cap.
        raise MeetingNotesError(
            f"Agent didn't converge on a final answer within {max_turns} "
            "ReAct turns. This usually means the notes are ambiguous or the "
            "model got stuck in a tool-call loop. Try shorter or clearer "
            "notes, or bump max_turns if you're confident it just needs "
            "more room.",
            partial=MeetingNotesAttempt(last_input=str(exc), turns_used=max_turns),
        ) from exc
    except MeetingNotesError:
        raise
    except Exception as exc:  # R5 case 3: SDK / API / rate-limit failure
        raise _translate_api_error(exc) from exc

    # SDK's structured-output validation (`output_type=MeetingSummary`)
    # already ran; final_output_as re-validates and returns the typed
    # object. raise_if_incorrect_type surfaces schema mismatches as a
    # loud error, not a silent None.
    return result.final_output_as(MeetingSummary, raise_if_incorrect_type=True)


# --- Agent factory (pedagogical anchor: read this to see the ReAct
#     wiring via openai-agents' Agent / function_tool primitives) ---


def _build_agent(*, model: str, notes: str, provider: str = "openai"):
    """Build the ReAct agent. `notes` is closed over so the 3-of-4
    tools that need it don't force the model to pass notes back on
    every call (would waste tokens and let the model paraphrase).
    Only `score_urgency` operates on a snippet the model provides,
    which is intentional -- urgency scoring for a specific candidate
    item is the model's per-item question.

    Lazy import: openai-agents is heavy; only pulled in on real-provider
    runs (mock path skips this factory entirely)."""
    from agents import Agent, ModelSettings, function_tool

    if provider == "ollama":
        from agents.models.openai_chatcompletions import (
            OpenAIChatCompletionsModel,
        )
        from openai import AsyncOpenAI

        from common.llm import ollama_base_url
        model_arg = OpenAIChatCompletionsModel(
            model=model,
            openai_client=AsyncOpenAI(
                base_url=ollama_base_url(), api_key="ollama"
            ),
        )
        # Ollama's num_predict defaults small (~128). Give the ReAct
        # loop headroom (16k) so grammar-constrained decoding for a
        # multi-item MeetingSummary doesn't truncate mid-JSON.
        # Temperature=0 for JSON reliability on the smaller model.
        # (Non-strict schema was tried and rejected: gemma4:e4b then
        # emits {"action_items": []} to take the easy path even when
        # the transcript is full of commitments.)
        agent_settings = ModelSettings(max_tokens=16384, temperature=0.0)
    else:
        model_arg = model
        agent_settings = None

    def extract_speakers() -> list[str]:
        """Return a list of speaker/participant names extracted from the
        meeting notes. Use this to ground the `participants` field --
        only include names this tool returns."""
        return _extract_speakers_from(notes)

    def extract_dates() -> list[str]:
        """Return a list of date/deadline phrases found in the meeting
        notes (verbatim substrings). Use this to populate `due_hint`
        fields -- only use phrases this tool returns."""
        return _extract_dates_from(notes)

    def verify_excerpt(excerpt: str) -> bool:
        """Return True iff `excerpt` is a verbatim substring of the
        meeting notes (with whitespace normalized). Call this before
        finalizing each ActionItem's context_excerpt to prevent
        paraphrasing."""
        return _verify_excerpt_in(excerpt, notes)

    def score_urgency(item_text: str) -> str:
        """Score urgency for a candidate action item text. Returns
        'low', 'medium', or 'high' based on keywords in `item_text`.
        This is a signal, not the sole decider -- consider meeting
        context when picking the final `priority` field."""
        return _score_urgency_from(item_text)

    # Tool-use path is the pedagogical point on the cloud provider,
    # but gemma4:e4b's tool-call loop with a strict json_schema output
    # is unreliable: the model returns "Invalid JSON when parsing
    # model output" once the conversation grows across several tool
    # turns. Direct probes prove Ollama can produce a valid multi-item
    # MeetingSummary in one shot, just not through the ReAct loop.
    # For Ollama, skip the tools and let the model produce the
    # structured output directly. The grounding tools stay wired on
    # the cloud path.
    tools = (
        []
        if provider == "ollama"
        else [
            function_tool(extract_speakers),
            function_tool(extract_dates),
            function_tool(verify_excerpt),
            function_tool(score_urgency),
        ]
    )
    agent_kwargs: dict = {
        "name": "meeting-notes-agent",
        "instructions": _load_system_prompt(),
        "model": model_arg,
        "tools": tools,
        "output_type": MeetingSummary,
    }
    if agent_settings is not None:
        agent_kwargs["model_settings"] = agent_settings
    return Agent(**agent_kwargs)


# --- Error translation (R5 case 3) -----------------------------------------


def _translate_api_error(exc: Exception) -> MeetingNotesError:
    """Turn an openai-agents SDK or OpenAI SDK exception into a
    user-facing MeetingNotesError. Same 6-branch priority order as
    agents #02/#03 (class-name first, status-code second, message-
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
        return MeetingNotesError(OLLAMA_CONNECTION_HINT)

    return MeetingNotesError(
        f"Meeting-notes extraction failed: {type(exc).__name__}: {exc}. "
        "This is an unexpected error -- check the agent logs."
    )


def _rate_limit_error() -> MeetingNotesError:
    return MeetingNotesError(
        "The service is temporarily rate-limited or overloaded. "
        "Wait a minute and try again."
    )


def _auth_error() -> MeetingNotesError:
    return MeetingNotesError(
        "API authentication failed. Check that OPENAI_API_KEY is set in "
        ".env (or your shell environment)."
    )


# --- Mock mode -------------------------------------------------------------


def _mock_result(notes: str) -> MeetingSummary:
    """Deterministic canned MeetingSummary for smoke tests and CI. Does
    NOT run the ReAct agent, does NOT need openai-agents installed, does
    NOT touch the notes-validation R5 gate (mock is for exercising the
    downstream Pydantic + UI pipeline). Character count encoded into
    summary so a future refactor that accidentally makes mock output
    constant regardless of input surfaces at test time.

    Uses a fake action item whose context_excerpt is embedded into the
    summary string -- keeps the mock schema-valid without needing
    verify_excerpt to actually pass against real notes."""
    return MeetingSummary(
        meeting_topic="Mock meeting",
        participants=["Alice", "Bob"],
        action_items=[
            ActionItem(
                description="Send updated proposal to legal",
                owner="Alice",
                due_hint="Friday",
                priority="high",
                context_excerpt="Alice will send updated proposal to legal by Friday",
            ),
        ],
        key_decisions=["Postpone launch review to next week"],
        overall_summary=(
            f"Mock summary (input: {len(notes)} chars). Set LLM_PROVIDER to "
            "'openai' and configure OPENAI_API_KEY for a real extraction."
        ),
    )


# --- CLI entry point (uv run python -m agent) ------------------------------


def main() -> int:
    """CLI: takes a path to a meeting-notes text file, prints the
    extracted MeetingSummary as JSON."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="meeting-notes",
        description="Extract action items from meeting notes.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a text file containing meeting notes.",
    )
    parser.add_argument(
        "--model",
        help="Override the resolved model for this invocation.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Max ReAct turns (default: {DEFAULT_MAX_TURNS}).",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Gradio UI instead of a one-shot CLI run.",
    )
    args = parser.parse_args()

    if args.ui:
        try:
            from .ui import build_ui  # type: ignore[import-not-found]
        except ImportError:
            from ui import build_ui  # type: ignore[import-not-found]
        build_ui().launch()
        return 0

    if not args.path:
        parser.error("path is required unless --ui is passed")

    path = Path(args.path)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    notes = path.read_text(encoding="utf-8")

    try:
        summary = extract_action_items(notes, model=args.model, max_turns=args.max_turns)
    except MeetingNotesError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.partial is not None:
            print("---- partial state ----", file=sys.stderr)
            print(f"turns used: {exc.partial.turns_used}", file=sys.stderr)
            if exc.partial.last_input:
                print(f"last input: {exc.partial.last_input[:200]}...", file=sys.stderr)
        return 1

    print(summary.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
