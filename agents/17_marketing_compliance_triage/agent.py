"""Marketing copy compliance triage agent -- agent #17 of real-world-agents.

Technique demonstrated: **a genuine N-way conditional fan-out with
LangGraph, using a reducer to merge parallel branches.** A classify
node's router function returns a LIST of node names, causing LangGraph
to run only the applicable specialist reviewer nodes -- in parallel,
within one superstep -- before a shared `aggregate` node runs exactly
once with their results merged:

    START -> classify -> [conditional fan-out: 0..4 of the 4 reviewers] -> aggregate -> END
                            |-- review_health          (if health_wellness active)
                            |-- review_financial        (if financial_earnings active)
                            |-- review_privacy          (if childrens_privacy active)
                            |-- review_environmental    (if environmental active)
                            |-- (none active -> routes straight to "aggregate")

This is NOT agent #03's retry-loop shape. #03's `_route_after_execute`
is a *binary* router (retry vs. proceed vs. end) and its `GraphState`
deliberately avoids reducers -- its one list field (`attempts_history`)
is rebuilt wholesale on every turn, never written concurrently by more
than one node. This agent is the inverse case: `lens_findings` IS a
reducer field (`Annotated[list[LensFinding], operator.add]`) *because*
it genuinely gets written by 0-4 parallel branches in the same
superstep, which is exactly the case `operator.add` reducers exist
for.

Verified empirically against this repo's installed `langgraph==1.2.11`
before writing this file (not assumed from memory):
  1. `add_conditional_edges(source, router_fn, path_map)` accepts a
     router that returns `Sequence[str]` for fan-out to multiple named
     nodes; `path_map` should list every possible individual
     destination once (a flat list), even though the router returns
     only a subsequence of them per call.
  2. The `operator.add` reducer works correctly with a list of real
     Pydantic `BaseModel` instances, not just dicts/primitives --
     `LensFinding` objects are appended directly, no dict-downgrade
     needed.
  3. A downstream node with static edges from all possible parallel
     predecessors (`aggregate`) runs exactly once, after only the
     branches actually scheduled in that superstep complete -- not all
     four, when only two were routed to.

Why this technique for this use case: marketing copy triggers
different regulatory concerns depending on its content -- health
claims trigger FTC/FDA substantiation rules, financial claims trigger
SEC/CFPB disclosure rules, data-collection-from-kids triggers COPPA,
environmental claims trigger FTC Green Guides -- but not every piece
of copy touches every category. Running all 4 specialist reviewers on
every submission wastes cost on irrelevant checks; a single monolithic
prompt trying to cover all 4 lenses at once gives none of them the
focused attention a real compliance reviewer would give. Classify-then
-conditionally-dispatch mirrors how a real compliance team actually
triages: skim first, then route to the right subject-matter reviewer.

Real error handling (R5 in CONTRIBUTING.md's hard rules): three
concrete failure modes handled explicitly (see triage_marketing_copy
below):
  1. Copy text too short to be real marketing copy, or too long for
     single-pass analysis -> TriageError with a specific message,
     raised before any graph/LLM work.
  2. A node's structured-output retry loop exhausts its attempts
     (model can't produce valid JSON matching the target schema) ->
     TriageError with the last raw output + validation errors attached
     as .partial.
  3. Rate limit / auth / API failure -> translated to TriageError with
     a clear message, including an Ollama-connection-refused hint.
     This agent does NOT auto-retry transient API errors (same
     decision as agents #02/#03) -- the user pays per real API call
     and is better-placed to decide whether to wait and re-run.

Provider + model are fully user-configurable: every real LLM call goes
through common.llm.get_llm() / resolve_model(), same env-var contract
as every other agent in the catalog.
"""

from __future__ import annotations

import json
import operator
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, TypedDict, TypeVar

from pydantic import BaseModel

# Dual-mode import (same rationale as agents #01/#02/#03): `.schemas`
# resolves when this file is loaded as a submodule of the
# 17_marketing_compliance_triage package (how tests import it, via
# importlib since the dir name starts with a digit). The bare `schemas`
# (absolute) resolves when this file is run directly via `python -m
# agent` from inside the agent's own directory.
try:
    from .schemas import ComplianceReport, Lens, LensClassification, LensFinding
except ImportError:
    from schemas import ComplianceReport, Lens, LensClassification, LensFinding

from common.llm import (
    LLM,
    OLLAMA_CONNECTION_HINT,
    get_llm,
    is_ollama_connection_error,
    resolve_model,
)

# --- Provider ---------------------------------------------------------------

SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini", "ollama")

# --- Input-size guards (R5 case 1) ------------------------------------------
#
# MIN_COPY_CHARS: below this, there isn't enough text to be real
# marketing copy -- classify would just be guessing. 20 chars is a
# generous floor (a short tagline like "Clean energy, cleaner planet."
# clears it comfortably; a single word or two does not).
MIN_COPY_CHARS = 20

# Context-window ceiling, mirroring agent #02's _check_context_window /
# agent #03's _check_prompt_size: a rough chars-per-token heuristic, no
# native tokenizer (would tie this agent to one provider's SDK). Set
# generously high since marketing copy is almost never this long in
# practice -- this guard exists for the pathological "someone pasted an
# entire brand style guide" case.
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_COPY_TOKENS_ESTIMATE = 50_000

# --- Retry policy -------------------------------------------------------

DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_TOKENS_OUT = 2048
RETRY_BACKOFF_SECONDS = 1.0

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# One source of truth for the lens <-> node-name mapping, driving both
# graph construction (_build_graph) and routing (_route_after_classify)
# so there's no risk of the two drifting out of sync.
_LENS_TO_NODE: dict[Lens, str] = {
    "health_wellness": "review_health",
    "financial_earnings": "review_financial",
    "childrens_privacy": "review_privacy",
    "environmental": "review_environmental",
}
_LENS_PROMPTS: dict[Lens, str] = {
    "health_wellness": "review_health.txt",
    "financial_earnings": "review_financial.txt",
    "childrens_privacy": "review_privacy.txt",
    "environmental": "review_environmental.txt",
}

_SEVERITY_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def resolve_provider() -> str:
    """LLM_PROVIDER env var, defaulting to "openai". No provider is
    hardcoded."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}."
        )
    return provider


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


# --- Error type --------------------------------------------------------


@dataclass
class TriageAttempt:
    """What we got back when a structured-output retry loop gave up.
    Attached to TriageError so the UI can surface the raw model output
    in a warning banner rather than dropping it silently."""

    raw_text: str
    validation_errors: list[str] = field(default_factory=list)


class TriageError(Exception):
    """Raised on any user-facing triage failure (bad input, retry
    exhaustion, API failure). `message` is user-friendly; `partial`
    carries the last raw output when relevant."""

    def __init__(self, message: str, partial: TriageAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Graph state -------------------------------------------------------


class GraphState(TypedDict, total=False):
    """LangGraph state carried between nodes.

    `lens_findings` IS a reducer field (`Annotated[..., operator.add]`)
    -- unlike every list field in agent #03's GraphState -- because it
    is genuinely written concurrently by 0-4 parallel reviewer nodes in
    one superstep. Each active reviewer returns `{"lens_findings":
    [LensFinding(...)]}` independently; LangGraph merges all of them
    via operator.add before `aggregate` runs. Verified this works with
    real Pydantic BaseModel instances (not just dicts/primitives) --
    see module docstring.
    """

    copy_text: str
    context: str | None
    active_lenses: list[Lens]
    classify_reasoning: str
    lens_findings: Annotated[list[LensFinding], operator.add]
    report: ComplianceReport | None


# --- Public API ----------------------------------------------------------


def triage_marketing_copy(
    copy_text: str,
    context: str | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    _llm: LLM | None = None,  # test-injection escape hatch
) -> ComplianceReport:
    """Triage `copy_text` for regulatory compliance concerns.

    Args:
        copy_text: the draft marketing/ad copy to check.
        context: optional free-text product/industry context (helps
            the classifier, e.g. "supplement brand" or "fintech app").
        provider: "openai" (default) / "anthropic" / "gemini" / "mock".
            Defaults to the LLM_PROVIDER env var.
        model: model ID; defaults to the resolved provider's
            DEFAULT_MODELS entry.
        max_retries: how many times each node's structured-output loop
            re-prompts on a validation failure before raising. Default
            3.
        _llm: injected LLM for tests; production callers leave this
            None.

    Returns:
        A validated ComplianceReport.

    Raises:
        TriageError: on any of the three R5 failure modes.
    """
    resolved_provider = (provider or resolve_provider()).lower()

    if resolved_provider == "mock":
        return _mock_report(copy_text)

    # R5 case 1a: copy too short to be real marketing copy.
    if len(copy_text.strip()) < MIN_COPY_CHARS:
        raise TriageError(
            f"Copy text is too short ({len(copy_text.strip())} chars) to "
            f"triage meaningfully -- need at least {MIN_COPY_CHARS} chars. "
            "Paste the actual draft ad/marketing copy, not a fragment."
        )

    resolved_model = model or resolve_model(resolved_provider)

    # R5 case 1b: copy too long for single-pass analysis.
    _check_prompt_size(copy_text, resolved_model)

    llm = _llm if _llm is not None else get_llm(resolved_provider)
    graph = _build_graph(llm=llm, model=resolved_model, max_retries=max_retries)

    initial_state: GraphState = {
        "copy_text": copy_text,
        "context": context,
        "active_lenses": [],
        "classify_reasoning": "",
        "lens_findings": [],
        "report": None,
    }

    try:
        final_state = graph.invoke(initial_state)
    except TriageError:
        raise
    except Exception as exc:  # R5 case 3: rate limit / API failure
        raise _translate_api_error(exc) from exc

    report = final_state.get("report")
    if report is None:
        # Should be unreachable -- _aggregate always sets "report" on
        # every path. Defensive guard so a future bug surfaces as a
        # clear TriageError rather than an AttributeError deep in a
        # caller that assumed a non-None ComplianceReport.
        raise TriageError(
            "Internal error: the graph finished without producing a report."
        )
    return report


# --- Input-size guard ----------------------------------------------------


def _check_prompt_size(copy_text: str, model: str) -> None:
    """R5 case 1b. Raises TriageError with the model named if
    copy_text alone (before adding the prompt template + reserving
    output tokens) already exceeds MAX_COPY_TOKENS_ESTIMATE. Mirrors
    agent #02's _check_context_window / agent #03's _check_prompt_size
    shape."""
    estimated = len(copy_text) // CHARS_PER_TOKEN_ESTIMATE
    if estimated > MAX_COPY_TOKENS_ESTIMATE:
        raise TriageError(
            f"This copy is too long for single-pass triage (~{estimated:,} "
            f"tokens estimated, over the {MAX_COPY_TOKENS_ESTIMATE:,}-token "
            f"ceiling this agent applies before calling {model}). Split it "
            "into shorter pieces, or bump MAX_COPY_TOKENS_ESTIMATE in "
            "agent.py if you're using a larger-context model."
        )


# --- The LangGraph fan-out graph (pedagogical anchor) -----------------------

_BaseModelT = TypeVar("_BaseModelT", bound=BaseModel)


def _route_after_classify(state: GraphState) -> list[str]:
    """Conditional-edge router: returns a LIST of node names (0 to 4 of
    the 4 reviewer nodes), or ["aggregate"] directly when no lens is
    active. Pure function of state + the module-level _LENS_TO_NODE
    mapping -- no closed-over resources needed, so it's a module-level
    function (unlike the other nodes, which must close over llm/model)
    and directly unit-testable without building a graph."""
    active = state.get("active_lenses", [])
    if not active:
        return ["aggregate"]
    # dict.fromkeys dedupes while preserving order -- defends against a
    # model hallucinating the same lens twice (unverified whether
    # LangGraph's conditional-edge dispatch handles a duplicate
    # destination name gracefully, so don't rely on it).
    deduped = dict.fromkeys(active)
    return [_LENS_TO_NODE[lens] for lens in deduped]


def _build_graph(*, llm: LLM, model: str, max_retries: int):
    """Build and compile the classify -> conditional fan-out ->
    aggregate state graph.

    Nodes (6 real + START + END = 8 total):
      classify              -- LLM call; decides which of the 4 fixed
                                lenses apply
      review_health         -- LLM call; health/wellness claims lens
      review_financial       -- LLM call; financial/earnings claims lens
      review_privacy        -- LLM call; children's privacy lens
      review_environmental  -- LLM call; environmental claims lens
      aggregate              -- LLM call (or pure Python if zero lenses
                                active); merges all findings into one
                                ComplianceReport

    Conditional edge from classify: `_route_after_classify` returns a
    LIST of the active reviewer node names (0 to 4 of them), or
    `["aggregate"]` directly when no lens applies. LangGraph runs all
    returned nodes concurrently within one superstep; `aggregate` has
    static edges from all 4 reviewer nodes plus is a valid direct
    target, so it runs exactly once regardless of how many reviewers
    were actually scheduled.

    LLM + model are closed over via this factory, so nodes stay pure
    `state -> partial_state` functions -- tests inject a SequenceLLM by
    passing a different `llm` to _build_graph (indirectly, via
    triage_marketing_copy's `_llm` parameter).
    """
    # Lazy import: langgraph is a heavy dependency we don't want in
    # every mock-mode test run.
    from langgraph.graph import END, START, StateGraph

    def _classify(state: GraphState) -> dict:
        prompt_template = _load_prompt("classify.txt")
        prompt = _fill_prompt(prompt_template, state)
        result = _run_structured_loop(
            llm=llm,
            model=model,
            prompt=prompt,
            target_model=LensClassification,
            max_retries=max_retries,
        )
        return {
            "active_lenses": result.active_lenses,
            "classify_reasoning": result.reasoning,
        }

    def _make_reviewer_node(lens: Lens, prompt_name: str) -> Callable[[GraphState], dict]:
        def _reviewer(state: GraphState) -> dict:
            prompt_template = _load_prompt(prompt_name)
            prompt = _fill_prompt(prompt_template, state)

            def _check_lens(finding: LensFinding) -> list[str]:
                if finding.lens != lens:
                    return [
                        f"'lens' must be {lens!r} for this reviewer, got {finding.lens!r}."
                    ]
                return []

            def _check_phrases(finding: LensFinding) -> list[str]:
                errors = _check_lens(finding)
                normalized_source = _normalize_for_substring(state["copy_text"])
                for i, phrase in enumerate(finding.flagged_phrases):
                    if _normalize_for_substring(phrase) not in normalized_source:
                        errors.append(
                            f"flagged_phrases[{i}] {phrase!r} is not a "
                            "verbatim substring of the copy text. Quote the "
                            "exact wording from the copy."
                        )
                return errors

            result = _run_structured_loop(
                llm=llm,
                model=model,
                prompt=prompt,
                target_model=LensFinding,
                max_retries=max_retries,
                extra_validate=_check_phrases,
            )
            return {"lens_findings": [result]}

        return _reviewer

    def _aggregate(state: GraphState) -> dict:
        active_lenses = state.get("active_lenses", [])
        if not active_lenses:
            # No LLM call at all -- a clean, deterministic "no concerns"
            # report. This is the direct classify -> aggregate path.
            return {
                "report": ComplianceReport(
                    active_lenses=[],
                    findings=[],
                    overall_risk_level="low",
                    summary=(
                        "No regulatory concerns identified across the 4 "
                        "lenses this agent checks (health/wellness, "
                        "financial/earnings, children's privacy, "
                        "environmental/sustainability). This copy did not "
                        "trigger any of them."
                    ),
                )
            }

        findings = state.get("lens_findings", [])
        prompt_template = _load_prompt("aggregate.txt")
        filled = (
            prompt_template.replace("{copy_text}", state["copy_text"])
            .replace("{context}", state.get("context") or "(none provided)")
            .replace("{active_lenses}", json.dumps(active_lenses))
            .replace(
                "{findings_json}",
                json.dumps([f.model_dump() for f in findings], default=str),
            )
        )

        worst_severity = max(
            (_SEVERITY_ORDER[f.severity] for f in findings), default=0
        )

        def _check_risk_floor(report: ComplianceReport) -> list[str]:
            if _SEVERITY_ORDER[report.overall_risk_level] < worst_severity:
                return [
                    (
                        f"overall_risk_level ({report.overall_risk_level!r}) is "
                        "below the highest individual finding's severity -- "
                        "never return an overall level lower than the worst "
                        "finding. Raise it to match or exceed that finding."
                    )
                ]
            return []

        result = _run_structured_loop(
            llm=llm,
            model=model,
            prompt=filled,
            target_model=ComplianceReport,
            max_retries=max_retries,
            extra_validate=_check_risk_floor,
        )
        # The model doesn't see active_lenses/findings as fields it
        # must echo correctly char-for-char -- set them directly from
        # known-good state rather than trusting the model's copy.
        result.active_lenses = active_lenses
        result.findings = findings
        return {"report": result}

    g: StateGraph = StateGraph(GraphState)
    g.add_node("classify", _classify)
    for lens, node_name in _LENS_TO_NODE.items():
        g.add_node(node_name, _make_reviewer_node(lens, _LENS_PROMPTS[lens]))
    g.add_node("aggregate", _aggregate)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        _route_after_classify,
        [*_LENS_TO_NODE.values(), "aggregate"],
    )
    for node_name in _LENS_TO_NODE.values():
        g.add_edge(node_name, "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()


def _fill_prompt(template: str, state: GraphState) -> str:
    return template.replace("{copy_text}", state["copy_text"]).replace(
        "{context}", state.get("context") or "(none provided)"
    )


# --- The hand-rolled JSON-validate-retry loop -------------------------------
#
# Generalizes agent #02's _run_review_loop into one helper reused at
# all 6 call sites in this agent (1 classify + 4 reviewers + 1
# aggregate) instead of copy-pasting a retry loop 6 times. The pieces
# below (_parse_json_object, _normalize_for_substring, _sleep_backoff)
# are reused verbatim in shape from agent #02's agent.py.


def _run_structured_loop(
    *,
    llm: LLM,
    model: str,
    prompt: str,
    target_model: type[_BaseModelT],
    max_retries: int,
    extra_validate: Callable[[_BaseModelT], list[str]] | None = None,
) -> _BaseModelT:
    """Prompt -> parse JSON -> Pydantic-validate against target_model
    -> optional extra_validate() for cross-field invariants Pydantic
    alone can't enforce -> retry with feedback on any failure, up to
    max_retries.

    On exhaustion, raises TriageError with the last raw output
    attached as `partial` so the UI can surface it.
    """
    last_raw = ""
    last_errors: list[str] = []
    retry_feedback: str | None = None

    for attempt in range(1, max_retries + 1):
        attempt_prompt = prompt
        if retry_feedback:
            attempt_prompt = f"{retry_feedback}\n\n{prompt}"

        response = llm.complete(
            prompt=attempt_prompt,
            model=model,
            temperature=0.0,
            max_tokens=DEFAULT_MAX_TOKENS_OUT,
        )
        last_raw = response.text

        parsed = _parse_json_object(last_raw)
        if isinstance(parsed, str):  # error message
            last_errors = [parsed]
            retry_feedback = f"Your previous response was not valid JSON: {parsed}"
            if attempt < max_retries:
                _sleep_backoff(attempt)
            continue

        try:
            instance = target_model.model_validate(parsed)
        except Exception as exc:
            last_errors = [f"schema validation failed: {exc}"]
            retry_feedback = (
                "Your previous response was valid JSON but failed these "
                f"checks:\n- {last_errors[0]}\nFix ONLY these issues; keep "
                "everything else the same."
            )
            if attempt < max_retries:
                _sleep_backoff(attempt)
            continue

        errors = extra_validate(instance) if extra_validate else []
        if not errors:
            return instance

        last_errors = errors
        retry_feedback = (
            "Your previous response was valid JSON but failed these checks:\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\nFix ONLY these issues; keep everything else the same."
        )
        if attempt < max_retries:
            _sleep_backoff(attempt)

    raise TriageError(
        f"Model output failed validation after {max_retries} attempts. "
        "The last raw response is attached below.",
        partial=TriageAttempt(raw_text=last_raw, validation_errors=last_errors),
    )


def _sleep_backoff(attempt: int) -> None:
    """Linear backoff between retries. Kept tiny so tests aren't slow --
    this loop only exists for validation retries, not transient-API-
    error retries (the SDK handles those)."""
    time.sleep(RETRY_BACKOFF_SECONDS * attempt)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_object(raw_text: str) -> dict | str:
    """Extract and parse the first top-level JSON object from
    `raw_text`. Returns the parsed dict on success, or an error string
    on failure. Copied verbatim in shape from agent #02's
    _parse_json_object: handles markdown code fences and leading/
    trailing prose."""
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            stripped = (
                "\n".join(lines[1:-1])
                if lines[-1].strip().startswith("```")
                else "\n".join(lines[1:])
            )
    match = _JSON_OBJECT_RE.search(stripped)
    if not match:
        return "no JSON object found in response"
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return f"JSON parse error: {exc.msg} at line {exc.lineno} col {exc.colno}"
    if not isinstance(parsed, dict):
        return f"expected a JSON object, got {type(parsed).__name__}"
    return parsed


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_substring(s: str) -> str:
    """Collapse runs of whitespace to single spaces. Copied verbatim
    in shape from agent #02's _normalize_for_substring."""
    return _WHITESPACE_RE.sub(" ", s).strip()


# --- Error translation (R5 case 3) ------------------------------------------


def _translate_api_error(exc: Exception) -> TriageError:
    """Turn an OpenAI/Anthropic/Gemini/Ollama exception into a
    user-facing TriageError. Priority order mirrors agents #02/#03's
    _translate_api_error (class-name -> status-code -> message-string
    fallback -> ollama-connection check -> generic), plus the
    ollama-connection branch that #02/#03 predate but #04/#05 already
    use as the current convention."""
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
    if is_ollama_connection_error(exc):
        return TriageError(OLLAMA_CONNECTION_HINT)

    return TriageError(
        f"Marketing compliance triage failed: {type(exc).__name__}: {exc}. "
        "This is an unexpected error -- check the agent logs."
    )


def _rate_limit_error() -> TriageError:
    return TriageError(
        "The service is temporarily rate-limited or overloaded. "
        "Wait a minute and try again."
    )


def _auth_error() -> TriageError:
    return TriageError(
        "API authentication failed. Check that your LLM_PROVIDER matches "
        "the API key you've set in .env (OPENAI_API_KEY / "
        "ANTHROPIC_API_KEY / GEMINI_API_KEY)."
    )


# --- Mock mode ---------------------------------------------------------


def _mock_report(copy_text: str) -> ComplianceReport:
    """Deterministic canned report for smoke tests and CI
    (LLM_PROVIDER=mock). Does not build the graph or call any LLM."""
    return ComplianceReport(
        active_lenses=["health_wellness"],
        findings=[
            LensFinding(
                lens="health_wellness",
                flagged_phrases=["clinically proven"],
                concern=(
                    "Mock finding (would check real FTC/FDA substantiation "
                    f"concerns in real mode). Copy length: {len(copy_text)} chars."
                ),
                suggested_fix=(
                    "Set LLM_PROVIDER to 'openai', 'anthropic', 'gemini', or "
                    "'ollama' and configure the matching API key for a real "
                    "triage."
                ),
                severity="medium",
            )
        ],
        overall_risk_level="medium",
        summary=(
            "Mock report -- would triage real regulatory lenses against "
            "the submitted copy in real mode."
        ),
    )


# --- CLI entry point (uv run python -m agent) -------------------------------


def main() -> int:
    """CLI: takes marketing copy (as a file path or positional text) +
    optional context, prints the compliance report as JSON."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="marketing-compliance-triage",
        description=(
            "Triage draft marketing copy for applicable regulatory "
            "compliance concerns."
        ),
    )
    parser.add_argument(
        "copy_path",
        nargs="?",
        help="Path to a text file containing the draft marketing copy.",
    )
    parser.add_argument(
        "--context",
        help="Optional free-text product/industry context.",
    )
    parser.add_argument(
        "--provider",
        choices=[*SUPPORTED_PROVIDERS, "mock"],
        help="Override LLM_PROVIDER for this invocation.",
    )
    parser.add_argument(
        "--model",
        help="Override the resolved model for this invocation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max structured-output retries per node (default: {DEFAULT_MAX_RETRIES}).",
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

    if not args.copy_path:
        parser.error("copy_path is required unless --ui is passed")

    copy_text = Path(args.copy_path).read_text(encoding="utf-8")

    try:
        report = triage_marketing_copy(
            copy_text,
            args.context,
            provider=args.provider,
            model=args.model,
            max_retries=args.max_retries,
        )
    except TriageError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.partial is not None:
            print("---- last raw output ----", file=sys.stderr)
            print(exc.partial.raw_text, file=sys.stderr)
            for err in exc.partial.validation_errors:
                print(f"  - {err}", file=sys.stderr)
        return 1

    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
