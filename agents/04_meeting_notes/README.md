# Meeting notes → structured action items

Paste meeting notes (typed, pasted, or transcript-style with `Name: ...` lines); get back a structured list of action items with owner, deadline, priority, and a verbatim excerpt showing where the item came from. A deployable OSS alternative to "someone else writes up the follow-ups" for anyone running meetings where action items get lost between "who agreed to what" and "who actually did it."

## Technique demonstrated

**ReAct pattern (Reason + Act, [Yao et al. 2022](https://arxiv.org/abs/2210.03629)) via the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/).** The model interleaves reasoning traces with grounded tool calls:

```
Thought: I need to find who was actually in the meeting.
Action: extract_speakers()
Observation: ["Alice", "Bob", "Carol"]
Thought: Now what dates were mentioned?
Action: extract_dates()
Observation: ["by Friday", "next sprint"]
Thought: Score urgency for "send updated proposal by Friday".
Action: score_urgency("send updated proposal by Friday")
Observation: "medium"
Thought: Verify my chosen excerpt is a real substring.
Action: verify_excerpt("Alice will send updated proposal by Friday")
Observation: True
Final: MeetingSummary(...)
```

The 4 tools are pure-Python heuristics (regex / keyword scoring / substring check), deliberately NOT LLM calls. That's the point of ReAct's grounding: **tools with outputs the model can't fake**. `extract_speakers` returns real names from the notes; the model can't invent attendees. `verify_excerpt` is a straight substring test; the model can't fake a passing result. Distinct technique from #01 (structured extraction, one call), #02 (hand-rolled retry loop), #03 (LangGraph state machine).

## Why this technique for this use case

Extracting action items has multiple sub-steps that benefit from intermediate grounding: **who's here** (participants), **what was said** (candidate items), **what's the urgency** (priority scoring), **does this quote appear verbatim** (excerpt verification). A single-shot "extract action items" prompt would let the model fabricate speakers and paraphrase quotes, both of which break the "user can verify each item" trust model a shipped tool needs. The interleaved reason + tool + observation loop IS the technique.

The OpenAI Agents SDK fits because it abstracts: (a) the ReAct loop mechanics (call model → parse tool calls → execute tools → feed observations back), (b) structured-output validation via `output_type=MeetingSummary`, (c) tool schema generation from Python function signatures. Hand-rolling ReAct would obscure "here's what the SDK does for you"; reading `_build_agent` below tells you the whole pattern.

Where this technique is NOT the right fit: (a) tasks where a single LLM call would succeed (short text extraction: use #01's Instructor pattern), (b) workflows without real branching or tool use (use #02's linear retry loop pattern), (c) stateful workflows with explicit branching (use #03's LangGraph pattern).

**Cost note:** ReAct loops make multiple LLM calls per invocation (typically 3-8 depending on notes complexity). Rough ballpark at gpt-4.1-mini pricing: **~$0.005-0.02 per meeting**. The `max_turns=15` cap bounds worst-case cost per call.

## What it does

Input: raw meeting notes text. Output: a validated `MeetingSummary` Pydantic object with `meeting_topic` (best-effort), `participants` (grounded via `extract_speakers`), `action_items` (each with `description`, optional `owner`, optional `due_hint`, `priority`, and verbatim `context_excerpt`), `key_decisions` (agreed outcomes that aren't action items), and `overall_summary`.

Empty `action_items` is a valid outcome: some meetings are pure discussion or info-share with no commitments. The agent doesn't manufacture items to fill space.

## How to run locally

Four commands from a fresh clone (`python -m agent` must run from inside the agent's own directory: `agent` is a submodule of the digit-prefixed `04_meeting_notes` package):

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # set OPENAI_API_KEY
cd agents/04_meeting_notes
```

CLI:

```bash
uv run python -m agent path/to/meeting_notes.txt
```

Override model or turn cap:

```bash
uv run python -m agent path/to/notes.txt --model gpt-4.1-2025-04-14 --max-turns 20
```

Gradio UI:

```bash
uv run python -m agent --ui
```

Mock mode (no API key, canned response, for testing the pipeline end-to-end):

```bash
LLM_PROVIDER=mock uv run python -m agent path/to/notes.txt
```

**Provider note:** #04 is OpenAI-only in v1. The `openai-agents` SDK is provider-agnostic via [LiteLLM](https://docs.litellm.ai/), so swapping in Claude or Gemini is a one-line change; LiteLLM is deferred from v1 to keep the dep count tight. See the SDK's provider docs when you want to swap.

## Code walkthrough

Under 500 LOC excluding UI. Read these in order to understand the pattern:

1. **`schemas.py`**: `ActionItem` + `MeetingSummary` Pydantic models. `priority` is a closed `Literal` (3 values) so the model can't invent labels. `context_excerpt` is required to be a verbatim substring of the notes, enforced at agent.py time via the `verify_excerpt` tool (Pydantic can't see the source notes from inside a field validator). Same [C5] pattern as agent #02.
2. **`prompts/system.txt`**: the agent's `instructions` (system prompt). Explicit 6-step ReAct reasoning strategy: call `extract_speakers` first, then `extract_dates`, then per-item drafting with `score_urgency` + `verify_excerpt`. Rules: never invent participants, every excerpt must pass `verify_excerpt`, empty action_items is valid, don't include ReAct scratchpad in final output.
3. **`agent.py::resolve_provider()`**: reads `LLM_PROVIDER` (default `"openai"`). Rejects `"anthropic"` / `"gemini"` in v1 with a message pointing at the LiteLLM swap path documented above.
4. **`agent.py::extract_action_items()`**: the public API. Under `provider="mock"` short-circuits to `_mock_result` (no SDK, no key). Otherwise: R5 case 1 notes-validation → `_build_agent` → `Runner.run_sync(agent, notes, max_turns=15)` → `result.final_output_as(MeetingSummary)`. Catches `MaxTurnsExceeded` explicitly (R5 case 2) and generic exceptions via `_translate_api_error` (R5 case 3).
5. **`agent.py::_build_agent()`**: **THE pedagogical anchor.** Factory that builds an `Agent(name, instructions, model, tools, output_type)` with 4 `function_tool`-wrapped closures. LLM + notes are closed over via the factory so 3-of-4 tools (extract_speakers, extract_dates, verify_excerpt) don't force the model to pass notes back on every call. Reader sees Agent + function_tool + output_type wiring in one place.
6. **`agent.py::_extract_speakers_from` / `_extract_dates_from` / `_score_urgency_from` / `_verify_excerpt_in`**: the four pure-Python tool implementations. Regex heuristics for speakers/dates, keyword scoring for urgency, whitespace-normalized substring check for excerpt verification. Deliberately not LLM calls (see "Technique demonstrated" for why).
7. **`agent.py::_looks_like_meeting_notes()`**: R5 case 1 gate. Cheap length check only (below `MIN_NOTES_CHARS=200` → refuse). Semantic "is this actually meeting notes?" classification is left to the LLM: a keyword regex was tried and consistently rejected real transcripts that used contractions or informal phrasing, so it was removed.
8. **`agent.py::_translate_api_error()`**: R5 case 3. Six-branch priority order (class-name → status-code → message-string fallback → generic), same shape as agents #02/#03. This agent does NOT auto-retry rate-limit errors, same explicit S2-option-b decision as sibling agents: the user pays per real API call and is better-placed to decide whether to wait and re-run than to have the loop silently spend more of their budget.
9. **`agent.py::_mock_result()`**: deterministic canned MeetingSummary; character count encoded into summary so a future refactor that makes mock output constant regardless of input surfaces at test time.
10. **`ui.py::build_ui()`**: Gradio Blocks: notes textarea + sample-notes dropdown, priority-tinted action-item HTML cards (red/amber/green), participants + key-decisions block, full JSON. Kept intentionally small so the load-bearing code stays in `agent.py`.
11. **`tests/test_smoke.py`**: 48 tests, all under `LLM_PROVIDER=mock`. Covers mock path, R5 branches (notes-validation, MaxTurnsExceeded, all 6 `_translate_api_error` priority paths), pure helper functions across their edge cases, and `_build_agent` structural check. Bootstraps openai-agents SDK BEFORE inserting the workspace `agents/` dir on sys.path to avoid a namespace-package shadowing bug (documented as lesson L-13 for future SDK-based agents #08 / #12).

## When to use / When NOT to use

**Use when:**
- You run meetings and want structured follow-ups without asking someone to write them up manually
- You want the action items in a machine-readable format (JSON) to pipe into Linear / Jira / a spreadsheet
- You want the model's picks to be verifiable: each item comes with a verbatim excerpt so a reader can locate the source in the notes
- You're building your own ReAct agent and want a readable reference for how the OpenAI Agents SDK wires up tool-calling + structured output

**Do NOT use when:**
- Your input is audio, not text: pipe the audio through Whisper (or agent #14 when it ships) first, then feed the transcript here
- You need conversational memory ("what about the action items from last week?"): each call is stateless; no history persists between requests in v1
- You need calendar / issue-tracker integration (auto-file to Linear, add to Google Calendar): this agent produces JSON; the integration is deliberately out of scope
- You need multi-language support: the prompts and heuristics are English-only in v1

## Where this fails

- **Non-standard name formats**: `_extract_speakers_from` matches "Name:" and "Name said/mentioned/..." patterns. Initials-only names ("A.C."), non-Latin scripts ("陈"), or titles ("Dr. Smith") get missed. The model can then either omit those participants or add them from its own reading of the notes (and might, per the system prompt's rules). Symptom: participants list is incomplete for a meeting with unusual naming.
- **Ambiguous ownership**: "we should really update the roadmap" doesn't say who owns it. The system prompt tells the model to set `owner=None` in this case, but a model might guess wrong and assign it to whoever spoke last. Symptom: action items with a plausible-but-wrong owner. Mitigation: read `context_excerpt` for each item; if the excerpt doesn't clearly assign ownership, treat the owner field as a suggestion.
- **Extremely short meetings**: the R5 case 1 gate (`< 200 chars`) rejects fragmentary input. A real one-sentence follow-up ("Sync with Legal by Friday.") gets refused because it's under threshold. Workaround: pad with context, or lower `MIN_NOTES_CHARS` in `agent.py` for your use case.
- **Model gets stuck in a tool-call loop**: at `max_turns=15`, `MaxTurnsExceeded` fires and the agent gives up honestly with the partial state attached. Rare in practice (most meetings converge in 3-6 turns) but real for pathologically ambiguous input. Mitigation: bump `--max-turns` if you know the notes are complex, or shorten the notes so each item is clearer.
- **`extract_dates` misses non-English dates**: the keyword regex matches English weekdays / months / phrases only. A meeting with Spanish dates ("viernes") would return an empty list, and the model would have to set `due_hint=None` on items that HAD explicit deadlines. Symptom: `due_hint` empty on items where the notes clearly named a date.
- **Paraphrased excerpts**: the model gets `verify_excerpt` as a tool and the system prompt tells it to call before finalizing, but rare cases still slip through (model uses excerpt without checking). Would fail the SDK's structured-output validation only if the substring check is baked into schema (it isn't; Pydantic can't see the notes). Symptom: a MeetingSummary with an excerpt that isn't in the notes. Mitigation: manually verify against `context_excerpt` for anything you'll act on.
- **Curly quotes vs straight quotes**: notes copied from Word or Google Docs often use curly quotes (`""`), but models tend to output straight quotes (`""`). `verify_excerpt` normalizes whitespace but does NOT normalize quote characters, so an otherwise-verbatim excerpt fails the substring check for a quote-only reason and the model wastes a tool turn re-picking it. Same edge case #02 has; deferred as a v1.1 improvement.
- **"not urgent" scored as high**: `score_urgency` matches substrings (`"urgent"` appears in `"not urgent"`), so a genuinely-non-urgent item can get scored high by the tool. The system prompt tells the model to consider meeting context and override, but a model that trusts the tool output blindly will get this wrong. Symptom: `priority="high"` on an item the notes actually deprioritized. Mitigation: read the `context_excerpt` before trusting priority.
- **Missed "by X" date phrases**: `extract_dates` catches "by Friday" but not "by tomorrow" or "by Aug 15". Some date shapes (ISO, weekday alone, EoW/EoM/EoQ, `Q1..Q4`) ARE caught by the other regex branches; the "by X" prefix version is narrower. Symptom: `due_hint=None` on items where the notes clearly named a deadline. Workaround: expand the regex list in `agent.py::_extract_dates_from` for your date vocabulary.
- **Rate limit / API failure**: no auto-retry with backoff (explicit design decision, `agent.py::_rate_limit_error`). One failed attempt → translated to "temporarily rate-limited" → user re-runs manually. Trade: no silent budget-burning; downside: the user has to click "extract" again after a wait.
