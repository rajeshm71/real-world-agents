# Type-safe email triage

Upload a `.eml`; get back a structured triage decision (category, priority, action, and a draft reply when applicable) grounded in your own known-contact list and important-domain list. A deployable OSS alternative to "spend 30 minutes at the top of every day sorting the inbox" for anyone handling high email volume.

## Technique demonstrated

**Type-safe single-agent + dependency injection via [PydanticAI](https://ai.pydantic.dev/).** The core PydanticAI construction:

```python
agent = Agent(
    "openai:gpt-4.1-mini-2025-04-14",
    deps_type=TriageDeps,        # <-- typed injected context
    output_type=EmailTriage,     # <-- typed Pydantic output
    system_prompt=...,
)

@agent.tool
def is_known_contact(ctx: RunContext[TriageDeps], email_address: str) -> bool:
    return email_address in ctx.deps.known_contacts
```

Every tool receives `RunContext[TriageDeps]`: the deps are TYPED. If a tool signature drifts from the Agent's `deps_type=`, mypy/pyright catches it before runtime. If `output_type=` changes, every downstream caller sees the new shape.

Distinct from #01-#05: this is the first agent in the catalog where **the deps carried through every tool call are Pydantic-typed and Agent-generic**. The 4 tools use those deps to ground decisions in real user state (`known_contacts`, `important_domains`) rather than guesses.

## Why this technique for this use case

Email triage needs: (a) a strict output schema (`Category`/`Priority`/`Action` `Literal`s prevent invented labels that break downstream routing), (b) tools that need typed context (is this sender known to the user? is this domain important to them?), (c) a single-turn workflow (one email in, one triage out, no branching or multi-step state). PydanticAI is designed for exactly this shape.

Cross-field verification stays honest via the `verify_snippet` tool (pure-Python substring check with whitespace normalization): same [C5] pattern threading through #02/#04/#05. The `@model_validator` on `EmailTriage` also auto-nulls `suggested_reply` when the action isn't a respond action (H1-option-b pattern from #04).

Where this technique is NOT the right fit: (a) multi-turn conversations (PydanticAI supports message history, but the "one email → one triage" shape doesn't need it), (b) workflows with real branching or hand-offs (use #05's CrewAI shape), (c) real-time streaming where each token matters (add `agent.run_stream()`, out of v1 scope).

**Cost note:** typically 1-3 LLM calls per triage (one initial call + optional tool call for grounding + optional retry if validation fails). Rough ballpark at gpt-4.1-mini pricing: **~$0.001-0.003 per email**. Cheap enough to run over a whole inbox.

## What it does

Input: raw `.eml` bytes (RFC 2822 email) + optional `TriageDeps(known_contacts, important_domains)`. Output: a validated `EmailTriage` with `category` (7 values), `priority` (4), `action` (5), `reasoning` (2-3 sentence justification), `key_snippet` (verbatim excerpt from the body, [C5]-validated), and optional `suggested_reply` (only for respond actions).

The `TriageDeps` list-shape is intentional: your known contacts are a list of addresses/names you actually know; your important domains are a list of `@company.com`-style patterns that should bump priority. Both default to empty; the agent still runs, just without the extra grounding signal.

## How to run locally

Four commands from a fresh clone:

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # set OPENAI_API_KEY
cd agents/06_email_triage
```

CLI (with optional deps flags, both repeatable):

```bash
uv run python -m agent examples/work_request.eml \
    --known-contact "sarah.chen@acme-corp.example.com" \
    --important-domain "@acme-corp.example.com"
```

Gradio UI:

```bash
uv run python -m agent --ui
```

Mock mode (no API key, canned response):

```bash
LLM_PROVIDER=mock uv run python -m agent examples/work_request.eml
```

**Provider note:** V1 is OpenAI-only. Anthropic/Gemini swap is a one-line change to `agent.py::_build_agent`'s model string (`"openai:{model}"` → `"anthropic:claude-sonnet-5"` or `"gemini:gemini-2.5-flash"`). PydanticAI supports these natively; no extra dep needed.

## Code walkthrough

Under 500 LOC excluding UI. Read these in order:

1. **`schemas.py`**: `Category`/`Priority`/`Action` closed `Literal`s + `EmailTriage` + `TriageDeps` dataclass. The `@model_validator` auto-nulls `suggested_reply` when action isn't respond-shaped (H1-option-b pattern).
2. **`prompts/system.txt`**: the agent's system prompt: role + reasoning strategy + tool usage rules. Explicit "call `verify_snippet` before finalizing key_snippet" rule so [C5] verification runs.
3. **`agent.py::resolve_provider()`**: reads `LLM_PROVIDER`; rejects anthropic/gemini in v1 with a pointer to the swap-path in `_build_agent`.
4. **`agent.py::triage_email()`**: the public API. Mock short-circuits to `_mock_result`. Real path: `_parse_eml` (R5 case 1) → body-content gate (R5 case 2) → `_build_agent` → `agent.run_sync(prompt, deps=...)` → `result.output`. Generic exception path via `_translate_api_error` (R5 case 3).
5. **`agent.py::_build_agent()`**: **THE pedagogical anchor.** Constructs `Agent(model_string, deps_type=TriageDeps, output_type=EmailTriage, ...)` and registers 4 `@agent.tool` closures. Each tool takes `ctx: RunContext[TriageDeps]` as first arg: that's the typed dependency injection. Reader sees Agent + tool + typed deps + typed output wiring in one place.
6. **`agent.py::_parse_eml()`**: R5 case 1. Uses stdlib `email` module; handles single-part `text/plain` and multipart `text/plain`+`text/html` (prefers plain; strips HTML tags crudely if plain isn't available).
7. **`agent.py::_has_triage_material()`**: R5 case 2 gate. Body <20 chars → refuse.
8. **`agent.py::_translate_api_error()`**: R5 case 3. Six-branch priority (class-name → status-code → message-string → generic), same shape as #02-#05. No auto-retry (S2 decision).
9. **`agent.py::_verify_snippet()`**: [C5] substring check with whitespace normalization + empty-excerpt guard (M1 pattern from #05).
10. **`agent.py::_is_known_contact_impl` / `_is_important_domain_impl` / `_extract_dates_from`**: pure-Python tool implementations. Regex + membership checks, no LLM calls. The `@agent.tool` closures in `_build_agent` wrap these with `RunContext[TriageDeps]` access.
11. **`ui.py::build_ui()`**: Gradio Blocks: .eml upload + known-contacts/important-domains textboxes + Triage button. Right side: category/priority/action pills + reasoning + snippet + suggested reply.
12. **`tests/test_smoke.py`**: 36 tests, all under `LLM_PROVIDER=mock`. Covers mock path, .eml parsing (happy/HTML-only/malformed), R5 branches, `_translate_api_error` 6 priority paths, tool implementations, PydanticAI's built-in `TestModel` for real-agent-with-fake-model tests (no custom SequenceLLM needed).

## When to use / When NOT to use

**Use when:**
- You process high email volume and want typed, machine-readable triage decisions
- You want your own contact list + important domains to inform the agent's decisions (dependency injection is why this differs from a naive prompt)
- You need the sender-verification signals surfaced separately (a triage that says "this looks urgent but sender isn't known" is more useful than a monolithic "medium priority")
- You're learning PydanticAI and want a readable typed-agent + injected-deps reference

**Do NOT use when:**
- You need to process bulk email in parallel: this agent is one-at-a-time; batching + rate-limit handling is #11 (resume screener)'s shape
- You need to actually SEND replies (Gmail API, IMAP push): out of scope; this agent outputs draft text, user acts on it
- You need OCR / attachment inspection: this agent triages on body + headers only

## Where this fails

- **HTML-heavy marketing emails**: my HTML stripper is a crude regex (`re.sub(r"<[^>]+>", " ", ...)`), not a real HTML parser. Emails with heavy `<style>` blocks, tables, or tracking pixels can leave garbage in the body. Symptom: reasoning cites "content" that's actually CSS. Fix: swap in `html2text` or `beautifulsoup4`, deferred to keep the dep count tight.
- **Non-English emails**: the system prompt is English-only; the taxonomy is anglocentric ("newsletter" vs "spam" distinctions vary by locale). Symptom: model over-defaults to "other" for non-English.
- **Attachments**: `_parse_eml` skips non-text parts entirely. An email whose meaning is in an attached PDF (contract, spec) will get triaged on just the cover message. Real fix: pipe attachments through #01 or #02 first.
- **Threaded reply context**: this agent triages one .eml at a time. If a reply's meaning depends on the earlier thread, the model doesn't see it. Symptom: "Sure, let's do it" reads as spam-adjacent because there's no context.
- **Sender-spoofing / display-name tricks**: `is_known_contact` matches on substring; an email from `alice@evil.com` with display name "Alice" would falsely pass. Real anti-spoof needs DKIM/SPF/DMARC verification, out of scope.
- **`_has_triage_material` threshold is coarse**: 20 chars catches "hi" but not a genuine one-liner like "Approved, thanks!" (17 chars) which is a legit business email. Lower `MIN_BODY_CHARS` if that hits your use case.
- **PydanticAI eagerly initializes providers**: importing `Agent("openai:...")` requires `OPENAI_API_KEY` at construction time, even for structural tests. The `_build_agent` structural test sets a fake key; real callers get a clear "set OPENAI_API_KEY" error.
- **Rate limit / API failure**: no auto-retry with backoff (explicit S2 decision). One failed attempt → translated to "temporarily rate-limited" → user re-runs manually.
