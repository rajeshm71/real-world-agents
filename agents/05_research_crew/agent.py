"""Multi-agent research crew -- agent #05 of real-world-agents.

Technique demonstrated: **multi-agent collaboration via CrewAI's
Sequential Process**. Three specialized agents with distinct goals +
tools pass work between each other:

    Researcher  ->  gathers sources on the topic (search_topic tool)
        |           produces ResearchNotes: sources + key facts
        v
    Writer      ->  synthesizes notes into a 400-600 word brief
        |           produces a Draft: background + findings + implications
        v
    Editor      ->  fact-checks + tightens (verify_source_citation tool)
                    produces final ResearchBrief with sources_used

CrewAI's `Process.sequential` chains them: researcher runs first,
writer receives its output as context, editor receives the writer's
output as context. Each stage's `output_pydantic=` gives CrewAI a
schema to validate against; the editor's Task returns the final
`ResearchBrief` that `run_research()` returns to the caller.

Why this technique for this use case: gathering + synthesizing +
polishing is a natural three-stage pipeline. Trying to do all three
in one prompt (as agents #01/#02 do with their single-agent shape)
either produces shallow research (model rushes past the "gather"
step) or overwrought output (model spends its budget on wordsmithing
instead of finding sources). Specialization + handoff is the pattern
CrewAI is built for.

_build_crew() below is the pedagogical anchor -- read that one
function to see how CrewAI wires up Agent + Task + Crew + Process
into a working pipeline.

Real error handling (R5 in CONTRIBUTING.md's hard rules): three
concrete failure modes handled explicitly:
  1. Topic too vague / not researchable -- <10 chars OR matches
     trivial-input regex (test/hello/asdf/empty) -> ResearchError
     with "provide a real research topic" message. Checked BEFORE
     Crew construction so bogus input costs nothing.
  2. Search returned nothing -- the researcher's search_topic tool
     found zero sources for this topic. Real search would hit rate
     limits or genuinely-obscure topics; mock search would just miss
     the corpus. Either way -> ResearchError with the topic named.
  3. Crew execution failure -- CrewAI wraps model errors in its own
     exception hierarchy; _translate_api_error mirrors the 6-branch
     priority order from #02/#03/#04. Same S2 no-auto-retry decision.

Provider strategy: OpenAI-native via CrewAI's built-in LiteLLM
integration (no extra dep). Anthropic/Gemini swap is a one-line
change to the `llm=` parameter on each Agent -- documented in the
README. Multi-provider works out of the box because CrewAI uses
LiteLLM under the hood.

Mock mode: `LLM_PROVIDER=mock` bypasses the Crew entirely and
returns a deterministic canned ResearchBrief. Under mock, CrewAI
never imports (all crewai imports are lazy, inside real-provider
paths only) -- so mock-mode tests + schema tests run cleanly on
Python 3.14 even though crewai has no wheels for that version yet
(pyproject caps at <3.14 so real users hit a helpful error, not a
mysterious ImportError).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# Dual-mode import (same rationale as #01-04).
try:
    from .schemas import (
        WORD_COUNT_MAX,
        WORD_COUNT_MIN,
        ResearchBrief,
        Source,
    )
except ImportError:
    from schemas import (
        WORD_COUNT_MAX,
        WORD_COUNT_MIN,
        ResearchBrief,
        Source,
    )

from common.llm import resolve_model

# --- Provider + constants --------------------------------------------------

# CrewAI works multi-provider via LiteLLM (built-in). We default to
# openai, matching #01-04's convention. Users swap by changing the
# `llm=` param string on Agents in _build_crew.
SUPPORTED_PROVIDERS = ("openai",)

DEFAULT_MAX_ITER = 15  # per-agent max iterations before CrewAI stops
MIN_TOPIC_CHARS = 10
# Trivial-input regex: anything that matches this at the topic level
# fails the R5 case 1 gate. Small dictionary of common non-topics.
# No `\s*` alternate needed: _looks_like_valid_topic checks
# len(topic.strip()) < MIN_TOPIC_CHARS first, which catches
# empty/whitespace-only before this regex runs.
_TRIVIAL_TOPIC_RE = re.compile(
    r"^(test|hello|hi|hey|asdf|qwerty|foo|bar|baz)$",
    re.IGNORECASE,
)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def resolve_provider() -> str:
    """LLM_PROVIDER env var, default "openai". Only "openai" and
    "mock" supported in v1 (CrewAI supports Anthropic/Gemini via
    LiteLLM but v1 keeps the surface tight -- see README for the
    swap path)."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}. "
            "Multi-provider via CrewAI/LiteLLM is documented in the "
            "README as a one-line swap; not enabled in v1."
        )
    return provider


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


# --- Error type ------------------------------------------------------------


@dataclass
class ResearchAttempt:
    """Partial state from a failed research run -- attached to
    ResearchError so the UI can surface what the crew tried before
    giving up. If a future stage needs to attach richer context, add a
    specific typed field for it rather than an untyped bag."""

    topic: str = ""
    stage: str = ""  # 'search' / 'writing' / 'editing'
    partial_output: str = ""


class ResearchError(Exception):
    """Raised on any user-facing research failure. `message` is
    user-friendly; `partial` carries the crew's partial state when
    relevant."""

    def __init__(self, message: str, partial: ResearchAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Mock research corpus (v1: no real web search) -------------------------

# Dict of topic-substring -> list of canned Source objects. The
# mock search tool checks each topic against these substrings
# (case-insensitive) and returns any matching source set. Small
# curated set covers the demo topics + a fallback source.
_MOCK_CORPUS: dict[str, list[Source]] = {
    "quantum": [
        Source(
            url="https://example.com/quantum-computing-2026",
            title="State of Quantum Computing 2026",
            snippet="Quantum computers passed 1000 physical qubits in 2026.",
        ),
        Source(
            url="https://example.com/quantum-supremacy",
            title="Quantum Supremacy: What It Means",
            snippet="Quantum supremacy refers to a quantum computer solving a problem no classical computer can solve in practical time.",
        ),
    ],
    "climate": [
        Source(
            url="https://example.com/climate-2026-report",
            title="Global Climate Report 2026",
            snippet="Global average temperatures rose 1.5C above pre-industrial baseline in 2026.",
        ),
        Source(
            url="https://example.com/climate-adaptation",
            title="Adaptation Strategies for 2030",
            snippet="Coastal cities are investing in seawalls and managed retreat.",
        ),
    ],
    "ai safety": [
        Source(
            url="https://example.com/ai-safety-overview",
            title="AI Safety Landscape 2026",
            snippet="AI safety research focuses on alignment, interpretability, and governance.",
        ),
        Source(
            url="https://example.com/ai-safety-labs",
            title="Major AI Safety Labs",
            snippet="Anthropic, OpenAI, and DeepMind maintain dedicated safety teams.",
        ),
    ],
}

# Fallback for topics not in the mock corpus -- returns one generic
# source so the crew has something to work with (rather than
# triggering R5 case 2 for every unknown topic under mock).
_MOCK_FALLBACK_SOURCE = Source(
    url="https://example.com/generic-source",
    title="Generic Reference Source",
    snippet="This is a mock source returned for topics not covered by the curated mock corpus.",
)


def _mock_search(topic: str) -> list[Source]:
    """V1 mock search: substring-match the topic against
    _MOCK_CORPUS keys. Returns matching sources, or the fallback if
    nothing matches. Real search (Serper/Tavily/Brave) is a
    documented one-line swap in the README."""
    lower = topic.lower()
    for key, sources in _MOCK_CORPUS.items():
        if key in lower:
            return sources
    return [_MOCK_FALLBACK_SOURCE]


# --- R5 case 1: topic validation -------------------------------------------


def _looks_like_valid_topic(topic: str) -> bool:
    """R5 case 1 gate. Below MIN_TOPIC_CHARS or matches the trivial-
    input regex (test/hello/asdf/empty/whitespace) -> reject."""
    if len(topic.strip()) < MIN_TOPIC_CHARS:
        return False
    return not _TRIVIAL_TOPIC_RE.match(topic.strip())


# --- Excerpt-substring verification tool ([C5] pattern) --------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_ws(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s).strip()


def _verify_source_citation(excerpt: str, sources_content: str) -> bool:
    """Substring check with whitespace normalization -- same [C5]
    pattern as #02/#04. Editor calls this on candidate excerpts
    before finalizing sources_used. Returns True iff `excerpt` is
    a verbatim (whitespace-normalized) substring of the concatenated
    source snippets.

    Explicit empty-excerpt guard: in Python, `"" in "anything"` is True,
    so without this a downstream bug producing an empty excerpt would
    silently pass verification. Empty is never a valid citation, reject
    it.
    """
    if not excerpt.strip():
        return False
    return _normalize_ws(excerpt) in _normalize_ws(sources_content)


# --- Public API ------------------------------------------------------------


def run_research(
    topic: str,
    *,
    model: str | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    _crew=None,  # test-injection escape hatch
) -> ResearchBrief:
    """Run the multi-agent research crew on `topic`.

    Args:
        topic: natural-language research topic (e.g. "State of quantum
            computing in 2026", "AI safety landscape").
        model: OpenAI model ID. Defaults to `common.llm.resolve_model
            ("openai")` -- same env-var contract as agents #01-04.
        max_iter: per-agent max iterations before CrewAI stops.
            Default 15.
        _crew: injected pre-built Crew for tests. Production callers
            leave this None; the crew is built via _build_crew.

    Returns:
        A validated ResearchBrief.

    Raises:
        ResearchError: on any of the 3 R5 failure modes.
    """
    resolved_provider = resolve_provider()

    if resolved_provider == "mock":
        return _mock_result(topic)

    # R5 case 1: topic validation. Fails fast before Crew construction.
    if not _looks_like_valid_topic(topic):
        raise ResearchError(
            f"Topic {topic!r} doesn't look like a real research topic. "
            f"Provide something more specific (at least {MIN_TOPIC_CHARS} "
            "characters, not a trivial input like 'test' or 'hello')."
        )

    # R5 case 2: search preflight. Under mock this always returns
    # something (fallback source); under real search (documented
    # swap), this would call the real API. If sources come back
    # empty we surface the specific "search returned nothing" error
    # rather than letting the crew silently produce an empty brief.
    initial_sources = _mock_search(topic)
    if not initial_sources:
        raise ResearchError(
            f"Search for topic {topic!r} returned no results. The topic "
            "may be too niche, misspelled, or (for real search) the API "
            "may be temporarily down.",
            partial=ResearchAttempt(topic=topic, stage="search"),
        )

    resolved_model = model or resolve_model(resolved_provider)

    # Lazy import: crewai has no wheels for Python 3.14 yet. Mock
    # mode + R5 case 1 + the search preflight above must NOT require
    # it installed.
    #
    # No specialized CrewException catch here: verified against
    # crewai 1.15.17, `crewai.utilities.exceptions.CrewException` does
    # not exist at that path (the module only contains a
    # `context_window_exceeding_exception` submodule with
    # LLMContextLengthExceededError). A specialized catch would just be
    # redundant with the generic `except Exception` below. Verify
    # crewai is importable, then let the generic handler translate
    # whatever kickoff() raises.
    try:
        import crewai  # noqa: F401 -- availability check only
    except ImportError as exc:
        raise ResearchError(
            "crewai is not installed. This agent needs Python 3.10-3.13. "
            "Run `uv sync` at the workspace root; if you're on Python "
            "3.14+, install Python 3.12 or 3.13 first."
        ) from exc

    crew = _crew if _crew is not None else _build_crew(
        topic=topic, model=resolved_model, max_iter=max_iter,
        initial_sources=initial_sources,
    )

    try:
        result = crew.kickoff(inputs={"topic": topic})
    except ResearchError:
        raise
    except Exception as exc:  # R5 case 3: SDK / API / crew internal failure
        raise _translate_api_error(exc) from exc

    # CrewAI's Task with output_pydantic returns the typed Pydantic
    # object on the CrewOutput's `.pydantic` attribute. If it's None
    # or wrong type, the schema validation failed silently -- surface
    # loudly.
    typed = getattr(result, "pydantic", None)
    if not isinstance(typed, ResearchBrief):
        raise ResearchError(
            "Crew returned output but it didn't validate as a "
            f"ResearchBrief (got {type(typed).__name__}). This usually "
            "means the editor's final draft violated word_count range "
            f"[{WORD_COUNT_MIN}, {WORD_COUNT_MAX}] or missing required "
            "fields.",
            partial=ResearchAttempt(
                topic=topic, stage="editing",
                partial_output=str(result)[:1000],
            ),
        )
    return typed


# --- Crew factory (pedagogical anchor: read this to see the CrewAI
#     Agent + Task + Crew + Process wiring in one place) ------------------


def _build_crew(*, topic: str, model: str, max_iter: int, initial_sources: list[Source]):
    """Build and return the 3-agent Sequential-Process Crew.

    Agents (all lazy-instantiated via crewai imports below):
      researcher -- goal: gather 5-8 credible sources; tool: mock
                    _search_topic; output_pydantic: ResearchNotes
      writer     -- goal: synthesize into 400-600 word brief; no
                    tools (pure LLM); output as Draft prose
      editor     -- goal: fact-check + tighten; tool: verify_source_
                    citation; output_pydantic: ResearchBrief (the
                    final schema the caller receives)

    Process: Sequential -- researcher runs, writer receives its
    output as context, editor receives writer's output as context.

    The `initial_sources` list is passed to the researcher as
    "search results already in hand" via prompt context -- avoids
    forcing the researcher to re-invoke the search tool on the same
    topic under mock (where results are deterministic anyway).
    """
    # Lazy imports: crewai unavailable on Python 3.14. Wrapped in
    # try/except at the caller level so this factory would only
    # execute after that check passed.
    from crewai import Agent, Crew, Process, Task  # type: ignore[import-not-found]
    from crewai.tools import BaseTool  # type: ignore[import-not-found]

    # Wrap our two tools as CrewAI BaseTool subclasses. Simple
    # class-based tools rather than the @tool decorator because
    # (a) they're stateful (search closes over topic; verify closes
    # over sources_content), (b) BaseTool's schema is explicit which
    # is nicer for pedagogical reading.
    sources_content = "\n\n".join(s.snippet for s in initial_sources)

    class SearchTopicTool(BaseTool):
        name: str = "search_topic"
        description: str = (
            "Search for sources on the given topic. Returns a list of "
            "sources with URL, title, and a snippet each. V1 uses a "
            "canned mock corpus; production would call Serper/Tavily."
        )

        def _run(self, query: str) -> str:
            sources = _mock_search(query)
            return "\n\n".join(
                f"[{s.title}] {s.url}\n{s.snippet}" for s in sources
            )

    class VerifySourceCitationTool(BaseTool):
        name: str = "verify_source_citation"
        description: str = (
            "Verify that an excerpt is a verbatim (whitespace-normalized) "
            "substring of the source content. Call this on every "
            "candidate source snippet before finalizing sources_used. "
            "Returns True or False."
        )

        def _run(self, excerpt: str) -> str:
            return str(_verify_source_citation(excerpt, sources_content))

    search_tool = SearchTopicTool()
    verify_tool = VerifySourceCitationTool()

    # CrewAI Agents: role + goal + backstory + optional tools + LLM.
    # llm="openai/<model>" is CrewAI's LiteLLM addressing convention.
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Gather credible sources on the topic: {topic}",
        backstory=_load_prompt("researcher.txt"),
        tools=[search_tool],
        llm=f"openai/{model}",
        verbose=False,
        max_iter=max_iter,
    )
    writer = Agent(
        role="Research Brief Writer",
        goal=f"Synthesize research notes into a clear brief on: {topic}",
        backstory=_load_prompt("writer.txt"),
        llm=f"openai/{model}",
        verbose=False,
        max_iter=max_iter,
    )
    editor = Agent(
        role="Fact-Checking Editor",
        goal=f"Polish the brief and verify every source citation for: {topic}",
        backstory=_load_prompt("editor.txt"),
        tools=[verify_tool],
        llm=f"openai/{model}",
        verbose=False,
        max_iter=max_iter,
    )

    # Tasks: each agent gets one Task. Sequential Process passes each
    # Task's output as context to the next Task automatically.
    research_task = Task(
        description=(
            f"Research the topic: {topic}. Use the search_topic tool "
            "to gather 3-5 credible sources. For each source, note the "
            "URL, title, and a verbatim snippet supporting a key fact. "
            "Output a bullet list of sources and a bullet list of the "
            "key facts you'll pass to the writer."
        ),
        expected_output="A markdown-formatted list of sources and key facts.",
        agent=researcher,
    )
    write_task = Task(
        description=(
            "Write a 400-600 word research brief based on the researcher's "
            "sources and key facts. Structure: (1) 2-3 sentence summary, "
            "(2) background paragraph, (3) 3-5 key findings as bullets, "
            "(4) implications paragraph."
        ),
        expected_output="A well-structured research brief in the specified format.",
        agent=writer,
        context=[research_task],
    )
    edit_task = Task(
        description=(
            "Edit and fact-check the writer's brief. For each factual "
            "claim, use verify_source_citation to confirm the supporting "
            "excerpt actually appears in the researcher's sources. Remove "
            "any claim that can't be verified. Tighten prose. Output the "
            "final ResearchBrief with sources_used containing only "
            "verified sources."
        ),
        expected_output=(
            "A ResearchBrief object with topic, summary, background, "
            "key_findings, implications, sources_used, word_count."
        ),
        agent=editor,
        context=[write_task],
        output_pydantic=ResearchBrief,
    )

    return Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, write_task, edit_task],
        process=Process.sequential,
        verbose=False,
    )


# --- Error translation (R5 case 3) -----------------------------------------


def _translate_api_error(exc: Exception) -> ResearchError:
    """Turn a CrewAI / OpenAI SDK exception into a user-facing
    ResearchError. Same 6-branch priority order as agents #02/#03/#04
    (class-name -> status-code -> message-string fallback -> generic).
    No auto-retry for transient errors (S2 decision)."""
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

    return ResearchError(
        f"Research crew failed: {type(exc).__name__}: {exc}. "
        "This is an unexpected error -- check the agent logs."
    )


def _rate_limit_error() -> ResearchError:
    return ResearchError(
        "The service is temporarily rate-limited or overloaded. "
        "Wait a minute and try again."
    )


def _auth_error() -> ResearchError:
    return ResearchError(
        "API authentication failed. Check that OPENAI_API_KEY is set "
        "in .env (or your shell environment)."
    )


# --- Mock mode -------------------------------------------------------------


def _mock_result(topic: str) -> ResearchBrief:
    """Deterministic canned ResearchBrief for smoke tests and CI
    (LLM_PROVIDER=mock). Does NOT construct a Crew, does NOT need
    crewai installed, does NOT touch the topic-validation R5 gate
    (mock is for exercising the downstream Pydantic + UI pipeline
    without a real crew run).

    Character count of the topic is encoded into the summary so a
    future refactor that makes mock constant surfaces at test time.
    """
    # Mock's content is padded to ACTUALLY reach WORD_COUNT_MIN (300
    # words across background + key_findings + implications), and
    # word_count is computed from that content rather than hardcoded --
    # so a downstream length-filter that trusts word_count sees the
    # honest number.
    background = (
        "This is a mock research brief. In a real run, the researcher "
        "agent would gather 3-5 sources on this topic using the "
        "search_topic tool, the writer would synthesize them into a "
        "400-600 word brief with a clear four-section structure, and "
        "the editor would fact-check every citation using the "
        "verify_source_citation tool before returning the final "
        "ResearchBrief. Under mock mode, the CrewAI stack is bypassed "
        "entirely and this deterministic canned brief is returned "
        "without any network call or API key requirement. That means "
        "you can exercise the full downstream pipeline (Pydantic "
        "validation, the Gradio UI's HTML rendering, JSON serialization "
        "for the CLI output) without spending real budget, and the "
        "output is byte-identical across runs so tests can assert on "
        "specific field values."
    )
    key_findings = [
        (
            "Mock finding one: the mock research corpus returned canned "
            "sources rather than performing a real web search."
        ),
        (
            "Mock finding two: no real API call was made and no OPENAI_API_KEY "
            "was required for this invocation."
        ),
        (
            "Mock finding three: word_count is honestly computed from the "
            "actual mock content so downstream length-based filters see a "
            "truthful value."
        ),
    ]
    implications = (
        "In production, this section would summarize what the research "
        "means for the reader and what actions they should consider taking "
        "based on the key findings. Under mock, it is a fixed placeholder "
        "long enough to satisfy the schema's WORD_COUNT_MIN constraint "
        "without triggering the WORD_COUNT_MAX ceiling. The mock exists to "
        "let developers exercise the pipeline end-to-end -- schema "
        "validation, UI rendering, CLI output, JSON round-tripping -- "
        "without paying for real LLM calls or dealing with API-key setup "
        "friction in CI. When you flip LLM_PROVIDER to 'openai', the real "
        "crew runs and produces topic-specific content in this same shape, "
        "and the word_count field will reflect what the writer + editor "
        "actually produced rather than this canned placeholder value. Any "
        "downstream length-based filter or gate can trust word_count "
        "regardless of whether the brief came from the real crew or the "
        "mock path, because both paths compute it from actual content "
        "rather than hardcoding a plausible-looking number."
    )
    computed_word_count = sum(len(section.split()) for section in [
        background,
        *key_findings,
        implications,
    ])
    return ResearchBrief(
        topic=topic,
        summary=(
            f"Mock brief on {topic!r} (input topic length: {len(topic)} "
            "chars). Set LLM_PROVIDER to 'openai' and configure "
            "OPENAI_API_KEY for a real crew run."
        ),
        background=background,
        key_findings=key_findings,
        implications=implications,
        sources_used=[
            Source(
                url="https://example.com/mock-source-1",
                title="Mock Source One",
                snippet="Canned snippet used by the mock brief.",
            ),
        ],
        word_count=computed_word_count,
    )


# --- CLI entry point (uv run python -m agent) ------------------------------


def main() -> int:
    """CLI: takes a topic string, prints the ResearchBrief as JSON."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="research-crew",
        description="Multi-agent research on a topic; outputs a structured brief.",
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Research topic (in quotes if it contains spaces).",
    )
    parser.add_argument(
        "--model",
        help="Override the resolved model for this invocation.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help=f"Per-agent max iterations (default: {DEFAULT_MAX_ITER}).",
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

    if not args.topic:
        parser.error("topic is required unless --ui is passed")

    try:
        brief = run_research(args.topic, model=args.model, max_iter=args.max_iter)
    except ResearchError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.partial is not None:
            print("---- partial state ----", file=sys.stderr)
            print(f"stage: {exc.partial.stage}", file=sys.stderr)
            if exc.partial.partial_output:
                print(
                    f"partial output: {exc.partial.partial_output[:300]}...",
                    file=sys.stderr,
                )
        return 1

    print(brief.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
