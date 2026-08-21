# Research crew → structured brief

Enter a research topic; three specialized agents (**researcher**, **writer**, **editor**) collaborate to produce a fact-checked brief with verified source citations. A deployable OSS alternative to "spend 2 hours in 15 browser tabs synthesizing sources yourself" for anyone who needs a well-researched brief on a topic quickly.

## Technique demonstrated

**Multi-agent collaboration via a [CrewAI](https://docs.crewai.com/) Sequential Process.** Three specialized agents with distinct roles + tools + goals pass work between each other:

```
Researcher  →  gathers 3-5 sources on the topic (search_topic tool)
    │          hands off Sources + Key-facts to writer
    ▼
Writer      →  synthesizes 400-600 word brief (no tools, pure LLM)
    │          hands off Draft to editor
    ▼
Editor      →  fact-checks + tightens (verify_source_citation tool)
                returns final ResearchBrief with sources_used
```

CrewAI's `Process.sequential` chains them: researcher runs first, writer receives its output as `context`, editor receives writer's output as `context`. Each agent's `Task` has an `output_pydantic=` schema CrewAI validates against. The editor's final Task returns the `ResearchBrief` the caller receives.

Distinct technique from #01-#04: **specialized-agent handoff** (three agents, three roles, work flows between them). #01 does single-agent structured extraction; #02 hand-rolls a linear retry loop; #03 wires a LangGraph state machine; #04 runs a single ReAct agent with tools. #05 is the first true multi-agent pattern in the catalog.

## Why this technique for this use case

Gathering + synthesizing + polishing is a natural three-stage pipeline. Trying to do all three in one prompt (as #01/#02 do with their single-agent shape) either produces shallow research (model rushes past the "gather" step to get to writing) or overwrought output (model spends its budget on wordsmithing instead of finding sources). Specialization + handoff is the pattern CrewAI is built for: give each agent a narrow role, a specific goal, and only the tools it needs, and let the framework wire the sequential handoff.

Cross-field verification stays honest via the editor's `verify_source_citation` tool — a pure-Python substring check (same [C5] pattern as #02 and #04). This prevents the writer from citing sources it paraphrased or made up; every source in `sources_used` on the returned `ResearchBrief` has been verified as a verbatim substring of the researcher's original snippets.

Where this technique is NOT the right fit: (a) simple single-shot text extraction (use #01's Instructor pattern), (b) linear retry-on-error flows (use #02's hand-rolled loop or #03's state machine), (c) real-time / low-latency queries — three sequential LLM calls make this measurably slower than single-agent approaches.

**Cost note:** three agent executions per invocation, but each agent runs its own internal tool-use loop — real LLM-call count is more like **5-10 total per brief** (researcher plans + calls `search_topic` 1-3× + synthesizes handoff; editor loops with `verify_source_citation` for each claim). Rough ballpark at gpt-4.1-mini pricing: **~$0.01-0.05 per brief** depending on topic complexity + brief length. The `max_iter=15` per-agent cap bounds worst-case cost.

## What it does

Input: a natural-language research topic (`"State of quantum computing in 2026"`, `"AI safety landscape"`). Output: a validated `ResearchBrief` with `topic`, `summary` (2-3 sentence TL;DR), `background` (1-2 paragraphs), `key_findings` (3-5 bullets), `implications` (1 paragraph), `sources_used` (list of verified sources with URL + title + snippet), and `word_count` (validator enforces 300-700).

**V1 uses a mock research corpus** (curated dict of topic-keyword → canned sources) so the demo + tests work out of the box without any external API key. Real web search (Serper, Tavily, Brave, Firecrawl) is a documented one-line swap — replace `_mock_search` with a call to `crewai_tools.SerperDevTool()` and add `SERPER_API_KEY` to `.env`.

## How to run locally

**Requires Python 3.10-3.13.** CrewAI has no wheels for Python 3.14+ yet (verified 2026-08-21) — `pyproject.toml` pins `requires-python = ">=3.10,<3.14"` so `uv sync` gives a clear signal instead of a cryptic wheel-not-found error.

Four commands from a fresh clone:

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # set OPENAI_API_KEY
cd agents/05_research_crew
```

CLI:

```bash
uv run python -m agent "State of quantum computing in 2026"
```

Override model or per-agent iteration cap:

```bash
uv run python -m agent "AI safety landscape" --model gpt-4.1-2025-04-14 --max-iter 20
```

Gradio UI:

```bash
uv run python -m agent --ui
```

Mock mode (no API key — for testing the pipeline end-to-end):

```bash
LLM_PROVIDER=mock uv run python -m agent "any topic here"
```

**Provider note:** V1 is OpenAI-only via CrewAI's built-in LiteLLM integration. Anthropic/Gemini swap is a one-line change to the `llm=` parameter on each Agent in `_build_crew` (LiteLLM addressing: `"anthropic/claude-sonnet-5"`, `"gemini/gemini-2.5-flash"`). Not enabled in v1 to keep the surface tight; documented for readers who want production research quality.

## Code walkthrough

Under 500 LOC excluding UI. Read these in order to understand the pattern:

1. **`schemas.py`** — `Source`, `ResearchNotes` (intermediate handoff), and `ResearchBrief` Pydantic models. `ResearchBrief.@model_validator` enforces word_count 300-700 + non-empty snippets on `sources_used` (cross-field [C5] pattern from #02/#04 applied preemptively). `sources_used.min_length=1` guarantees the editor kept at least one verified source.
2. **`prompts/researcher.txt`** — senior-analyst persona; rules: credibility over quantity, verbatim snippets only, use search_topic tool, structured Sources + Key-facts handoff (no self-analysis at this stage).
3. **`prompts/writer.txt`** — brief-writer persona; strict 4-section structure (Summary/Background/Key findings/Implications), 400-600 word target, synthesize-don't-copy, plain English, no editorializing.
4. **`prompts/editor.txt`** — fact-checker persona; use `verify_source_citation` on every claim, drop unverifiable claims, populate final ResearchBrief with word_count 300-700, tighten-don't-rewrite.
5. **`agent.py::resolve_provider()`** — reads `LLM_PROVIDER` (default `"openai"`). Rejects anthropic/gemini in v1 with a message pointing at the CrewAI/LiteLLM swap path documented above.
6. **`agent.py::run_research()`** — the public API. Under `provider="mock"` short-circuits to `_mock_result` (no CrewAI import, no key). Otherwise: R5 case 1 topic validation → R5 case 2 search preflight → `_build_crew` → `crew.kickoff(inputs={"topic": ...})` → extract `ResearchBrief` from `result.pydantic`. Catches `CrewException` explicitly (R5 case 3), generic exceptions via `_translate_api_error`.
7. **`agent.py::_build_crew()`** — **THE pedagogical anchor.** Constructs 3 CrewAI `Agent` objects (researcher/writer/editor), 3 `Task` objects with `context=` chaining, and wraps them in a `Crew(agents=[...], tasks=[...], process=Process.sequential)`. Two `BaseTool` subclasses (SearchTopicTool, VerifySourceCitationTool) close over `topic` / `sources_content` so the model doesn't have to pass context back on every tool call. Reader sees Agent + Task + Crew + Process + BaseTool wiring in one place.
8. **`agent.py::_mock_search`** — v1 mock: substring-match the topic against a small curated corpus (quantum / climate / ai-safety) with a graceful fallback. Real search (Serper/Tavily) is a documented one-line swap.
9. **`agent.py::_verify_source_citation()`** — pure-Python substring check with whitespace normalization. Editor's tool wraps this. Same [C5] pattern as #02/#04 — prevents paraphrased citations.
10. **`agent.py::_looks_like_valid_topic()`** — R5 case 1 gate. Below `MIN_TOPIC_CHARS=10` OR matches trivial-input regex (test/hello/asdf) → refuse before Crew construction, so bogus input costs nothing.
11. **`agent.py::_translate_api_error()`** — R5 case 3. Six-branch priority order (class-name → status-code → message-string fallback → generic), same shape as agents #02/#03/#04. No auto-retry for transient errors (S2 decision).
12. **`agent.py::_mock_result()`** — deterministic canned ResearchBrief bypassing the Crew entirely; topic char count encoded into summary as an anti-refactor guard.
13. **`ui.py::build_ui()`** — Gradio Blocks: topic textbox + sample-topics dropdown + Run button. Right column: brief HTML with all 4 sections + sources HTML with linked titles + verified snippets + full JSON.
14. **`tests/test_smoke.py`** — 31 tests, all under `LLM_PROVIDER=mock` (29 run + 2 skip on Python 3.14 via `pytest.importorskip("crewai")`). Covers mock path, all 3 R5 branches, `_translate_api_error` 6 priority paths, pure helpers, and `_build_crew` structural check (runs when crewai is installed).

## When to use / When NOT to use

**Use when:**
- You need a structured brief on a topic and want the model's picks to be verifiable (every source excerpt is a verbatim substring of what the researcher actually found)
- You're teaching or learning multi-agent orchestration and want a readable CrewAI reference
- You want the sources listed separately from the brief text so a reader can jump to the primary source
- You're OK with 3 LLM calls per invocation for higher-quality output vs single-shot

**Do NOT use when:**
- You need real-time / low-latency queries — three sequential LLM calls take noticeably longer than single-agent approaches
- Your topic is highly specialized and needs domain-specific search backends (medical PubMed, legal Westlaw) — this agent's mock corpus + generic web search shape doesn't fit
- You need conversational memory ("dig deeper on finding #2") — each `run_research` call is stateless; no session persistence in v1
- You need multi-language research — prompts + mock corpus are English-only in v1

## Where this fails

- **Mock corpus is tiny** — v1 ships a 3-keyword canned corpus (quantum / climate / ai-safety). Topics not matching any keyword get the single fallback source. Real research value comes from swapping in Serper/Tavily/Brave (documented above). Symptom: every brief on an unusual topic cites the same generic source.
- **Sequential Process is slower than parallel** — researcher → writer → editor runs in series, ~3× the latency of a single-agent brief. For genuine parallel research (fanning out on subtopics, then merging), CrewAI supports Hierarchical Process — deferred to a future release when the workflow actually benefits from branching.
- **Editor can hallucinate verifications** — the system prompt tells the editor to call `verify_source_citation` on every claim, but a rushed model might skip the check for some. The Pydantic validator catches whitespace-only snippets but not "verified but wrong" ones. Mitigation: `sources_used` cards in the UI let a reader spot-check the snippet against their own memory of the source.
- **`_looks_like_valid_topic` heuristic is coarse** — a legitimate short topic like `"NIST AI RMF"` (13 chars, no trivial-input match) passes; a longer non-topic like `"asdf asdf asdf asdf"` also passes (the regex only catches exact matches). Real topic-quality validation would need a semantic check; this is a fast fail-fast filter, not a QA gate.
- **Word count is enforced strictly** — if the editor's brief comes in at 299 or 701 words, Pydantic validation rejects it and CrewAI's re-run kicks in (or the caller sees a raw validation error). Occasionally the writer produces material outside the range and the editor doesn't tighten enough. Symptom: retry loop chews through iterations trying to hit the bounds.
- **Sources_used may be empty on hard topics** — if every excerpt the writer picked fails `verify_source_citation`, the editor's system prompt tells it to fall back to the researcher's original sources, but a model that follows instructions loosely might return `sources_used=[]` → Pydantic rejects it → wasted iteration. Fix: the schema validator's `min_length=1` guarantees the error surfaces at extraction time, not silently downstream.
- **`verify_source_citation` snapshots sources at crew-build time** — the tool's `sources_content` closure is built once from the preflight search's `initial_sources`. If the researcher's `search_topic` calls during the crew's execution return DIFFERENT sources (real Serper/Tavily would, mock corpus rarely does), the editor's verify tool won't see those newer sources and would reject valid excerpts from them. V1 mock corpus is deterministic enough that this doesn't manifest; a v1.1 fix would need a stateful sources-tracker that `search_topic` appends to and `verify_source_citation` reads from.
- **Rate limit / API failure** — no auto-retry with backoff (explicit design decision, `_rate_limit_error`). One failed attempt → translated to "temporarily rate-limited" → user re-runs manually. Trade: no silent budget-burning on 3-agent runs (each retry would cost 3 LLM calls); downside: user has to click "run" again after a wait.
