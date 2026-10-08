# Marketing copy compliance triage → only the regulatory checks that actually apply

Paste draft marketing/ad copy, get back which of 4 regulatory lenses apply and specific findings for each -- a first-pass compliance check a marketing team can run before sending copy to legal review.

## Technique demonstrated

**A genuine N-way conditional fan-out with [LangGraph](https://langchain-ai.github.io/langgraph/), using a reducer to merge parallel branches.** A `classify` node's router function returns a *list* of node names, causing LangGraph to run only the applicable specialist reviewer nodes -- in parallel, within one superstep -- before a shared `aggregate` node runs exactly once with their results merged via an `Annotated[list[LensFinding], operator.add]` reducer.

This is new to the catalog. Agent #03 (the only other LangGraph agent, and the only other user of `add_conditional_edges`) is a *binary* router (retry vs. proceed vs. end) and its `GraphState` deliberately avoids reducers -- its one list field is rebuilt wholesale every turn, never written concurrently by more than one node. This agent is the inverse case: `lens_findings` IS a reducer field *because* it's genuinely written by 0-4 parallel branches in the same superstep, which is exactly what `operator.add` reducers exist for. Contrast with agent #05 (CrewAI), whose multi-agent collaboration is a fixed Sequential Process (researcher → writer → editor, always all three, always in that order) -- this agent's fan-out is conditional and variable-width (0 to 4 branches, decided per input), not a fixed pipeline.

## Why this technique for this use case

Marketing copy triggers different regulatory concerns depending on its content: health claims trigger FTC/FDA substantiation rules, financial claims trigger SEC/CFPB disclosure rules, data collection from children triggers COPPA, environmental claims trigger FTC Green Guides -- but not every piece of copy touches every category. A single monolithic prompt trying to cover all 4 lenses at once gives none of them the focused attention a real compliance reviewer would give; running all 4 specialist reviewers on every submission regardless of content wastes cost on checks that don't apply. Classify-then-conditionally-dispatch mirrors how a real compliance team actually triages: skim first, then route to the right subject-matter reviewer -- and only pay for the reviewers actually needed.

Where this technique is NOT the right fit: (a) a fixed, small set of checks that should always ALL run regardless of input (a fixed pipeline, like agent #05's crew, is simpler and clearer there), (b) checks that need to see each other's output before finishing (this agent's reviewers run independently and in parallel -- if lens B's finding should change depending on lens A's finding, this fan-out shape is wrong; a sequential chain is).

## What it does

Input: draft marketing/ad copy, plus optional free-text product/industry context. A `classify` node decides which of 4 fixed lenses apply -- Health/Wellness Claims, Financial/Earnings Claims, Children's Privacy, Environmental/Sustainability Claims -- then only the applicable lenses' specialist reviewers run, each producing structured findings (flagged phrases, concern, suggested fix, severity). A final `aggregate` step merges everything into one `ComplianceReport` with an overall risk level. Zero active lenses is a valid, common outcome -- a plain brand-awareness ad with no claims produces a clean "no concerns" report with no further LLM calls at all.

## How to run locally

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # then edit .env: set LLM_PROVIDER + the matching API key
cd agents/17_marketing_compliance_triage
```

`.env` defaults to `LLM_PROVIDER=openai`. Switch providers with `LLM_PROVIDER=anthropic`, `LLM_PROVIDER=gemini`, or `LLM_PROVIDER=ollama` and the matching key/local server; no code changes needed.

CLI:

```bash
uv run python -m agent examples/sample_copy_multi_lens.txt
```

Override provider/model/retries, add product context:

```bash
uv run python -m agent examples/sample_copy_multi_lens.txt --context "supplement brand" --provider anthropic --model claude-sonnet-5
```

Gradio UI:

```bash
uv run python -m agent --ui
```

Mock mode (no API key, canned response, for testing the pipeline end-to-end):

```bash
uv run python -m agent examples/sample_copy_multi_lens.txt --provider mock
```

## Code walkthrough

Under 500 LOC excluding UI. Read these in order to understand the pattern:

1. **`schemas.py`**: `LensClassification` (classify's output, drives the fan-out directly), `LensFinding` (one reviewer's output for one lens), `ComplianceReport` (the final merged output).
2. **`prompts/classify.txt`**: names all 4 lenses with their regulatory grounding explicitly; instructs that an empty `active_lenses` list is a correct, common answer -- not a failure to force-fit a lens.
3. **`prompts/review_health.txt` / `review_financial.txt` / `review_privacy.txt` / `review_environmental.txt`**: one lens each, told explicitly to ignore every other regulatory concern even if noticed.
4. **`agent.py::_build_graph()`**: **THE pedagogical anchor.** 6 real nodes (classify, 4 reviewers, aggregate) + START + END. `_route_after_classify` is a module-level pure function (directly unit-testable) returning a list of 0-4 reviewer node names; `GraphState.lens_findings` is the one `Annotated[..., operator.add]` reducer field in the whole catalog.
5. **`agent.py::_run_structured_loop()`**: one shared hand-rolled JSON-validate-retry helper reused at all 6 LLM call sites, generalizing agent #02's `_run_review_loop` instead of copy-pasting a retry loop 6 times. Each reviewer's `extra_validate` checks every flagged phrase is a verbatim substring of the copy text (agent #02's excerpt-in-source check, adapted from a single excerpt field to a phrase list); `aggregate`'s checks `overall_risk_level` never falls below the worst individual finding's severity.
6. **`agent.py::triage_marketing_copy()`**: the public API. R5 case 1 (bad input): `MIN_COPY_CHARS` / `MAX_COPY_TOKENS_ESTIMATE` gates, raised as `TriageError` before any graph/LLM work. R5 case 2 (retry exhaustion): inside `_run_structured_loop`, raises `TriageError` with `.partial` attached. R5 case 3 (API failure): `_translate_api_error()`, a 6-branch translator (class-name → status-code → message-string → **Ollama-connection-refused check** → generic) -- the Ollama branch is new relative to agents #02/#03, matching the current convention agents #04/#05 already use.
7. **`ui.py::build_ui()`**: Gradio Blocks, severity-colored findings (mirrors agent #03's colored-attempt-history rendering). UI glue only -- the load-bearing code is in `agent.py`.
8. **`tests/test_smoke.py`**: 32 tests under `LLM_PROVIDER=mock`. The 2-active-lenses test is the most important one: it proves the `operator.add` reducer actually merges both parallel reviewer branches' findings without one clobbering the other. It uses a `KeyedLLM` fixture (prompt-content-keyed, not call-order-keyed) rather than the `SequenceLLM` FIFO fixture every other agent uses -- this is the first agent in the catalog where two LLM calls can genuinely race, so a strict-order fixture would be flaky.

## When to use / When NOT to use

**Use when:**
- You want a fast first-pass check on draft marketing copy before it goes to a human legal/compliance reviewer
- Your copy might touch 0, 1, or several of: health/wellness claims, financial/earnings claims, children's-audience data collection, environmental claims
- You want to see WHY a lens was flagged (verbatim quoted phrases), not just a risk score
- You're building your own conditional multi-specialist agent and want a readable reference for LangGraph's list-returning router + reducer-merge pattern

**Do NOT use when:**
- You need a legally binding compliance sign-off: this is a triage tool, not a lawyer
- Your regulatory concern isn't one of the 4 fixed lenses (e.g. alcohol/tobacco marketing restrictions, data-breach notification requirements): v1 doesn't cover those
- You need the reviewers to see each other's findings before finishing (e.g. a financial claim that becomes MORE concerning in light of a privacy finding): this agent's reviewers run independently in parallel, with no cross-branch visibility until aggregate

## Where this fails

- **The 4 fixed lenses don't cover every regulatory domain.** No alcohol/tobacco-marketing lens, no data-security/breach-notification lens, no advertising-to-seniors lens. A submission with real regulatory exposure outside these 4 categories gets a clean report that says nothing about the risk that actually matters.
- **`classify`'s lens selection is an LLM judgment call, not a deterministic rule -- and it can under-trigger.** This is the single most dangerous failure mode: a borderline health claim that the classifier doesn't recognize as a health claim never reaches `review_health` at all, and the final report looks clean and authoritative. There's no second check behind the classifier in v1.
- **Over-triggering is the safer failure direction but still real.** The classifier can activate a lens that doesn't genuinely apply (e.g. flagging `environmental` for an incidental "green packaging" mention with no actual sustainability claim), wasting a reviewer call and sometimes generating a low-value finding.
- **Findings are only as good as the verbatim-substring check.** A reviewer's flagged phrase is checked against the source copy, but the CONCERN and SUGGESTED FIX text are free-form LLM output -- regulatory nuance (e.g. the exact FTC Green Guides qualification language a specific claim needs) is not verified against any authoritative source, just the model's training.
- **This is a first-pass triage tool, not a substitute for actual legal/compliance review.** A true positive here still needs a human reviewer to confirm before action is taken; a false negative (the under-triggering failure mode above) is the one that matters most, since it produces silence rather than a visible error.
