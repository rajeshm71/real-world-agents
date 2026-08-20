# Contributing to real-world-agents

Thanks for your interest. This guide covers both "how do I use this" (for
people who just want to run a pattern) and "how do I contribute" (for people
who want to submit a PR).

## How to use this repo (non-contributors)

1. `git clone https://github.com/rajeshm71/real-world-agents && cd real-world-agents`
2. Pick an agent from the shipped list in the [main README](README.md)
3. `cd agents/<NN_name>/` and read its README first
4. Follow the "How to run locally" section
5. To copy the agent's pattern into your own project: see its "Code
   walkthrough" section, copy the files it points at

No API key is required to explore the code or run the test suite — every
test runs under `LLM_PROVIDER=mock`. A key is only needed if you want to make
a real (non-mock) call against OpenAI, Anthropic, or Gemini.

## Ways to contribute

- **Propose a new agent** demonstrating a technique not yet in the catalog
  (any framework, subject to the soft ≤3-agents-per-framework ceiling — see
  "Hard rules" below, R3)
- **Improve an existing agent** — better error handling, more eval cases,
  honest failure-mode analysis, better README pedagogy
- **Add cases to an agent's eval set** — helps make the accuracy claim more
  credible
- **Improve `common/`** — better error messages, better docs, fewer
  surprises
- **Report bugs**

Do NOT open a PR without an issue first for: new top-level deps in
`common/`, new LLM providers (chassis contract change), UI/dashboard
reworks, or changes to the per-agent README template (see "Agent README
template" below). Open an issue and get a maintainer nod before investing
time in any of these.

## Hard rules (R1-R10)

Ordered by priority. R1 is THE primary bar — everything else is subordinate
to it.

- **R1. Real-world use case (PRIMARY BAR).** Every shipped agent solves a
  real task a person or business would use in production — it could be
  copied into a real project and put behind real users. If an agent fails
  this test, it does not ship, regardless of how interesting the technique
  is.
- **R2. Simplicity + readability.** Each agent's own code stays under ~500
  LOC, excluding UI. `common/` stays minimal. If a new agent tempts you to
  add a shared abstraction, ask "would a reader understand THIS agent
  without reading the abstraction?" — if no, keep the code local to the
  agent.
- **R3. Technique variety (with soft framework ceiling).** Every agent
  demonstrates a distinctly different TECHNIQUE (structured extraction,
  ReAct, tool use with retry, RAG with citations, multi-agent collaboration,
  meta-optimization, a from-scratch agent loop, etc.). Framework is an
  implementation choice with a soft ceiling of ≤3 agents per framework — a
  4th agent on an already-used framework must demonstrate a technique the
  earlier ones don't.
- **R4. Per-agent README template enforcement.** All 8 required sections
  (see "Agent README template" below), in order, on every shipped agent's
  README. Enforced by `scripts/lint_agents.py` in CI.
- **R5. Real error handling.** The three obvious error cases for each
  agent's inputs are handled explicitly (rate limits, malformed input,
  tool/API failure) — not happy-path only. This is what makes an agent
  "deployable."
- **R6. Dependency budget.** `common/` runtime deps stay at 4 (openai,
  anthropic, google-genai, pydantic) — no single provider is hardcoded. Dev
  deps (pytest, ruff, uv itself) aren't counted. Per-agent runtime budget:
  ≤10. Never add a monolithic framework (LangChain-style) to `common/` — an
  individual agent may use one, that's fine there.
- **R7. Model version pinning.** Every LLM call uses a pinned dated
  snapshot. Every agent's README lists exact model IDs and their
  verification date.
- **R8. CI is mock-only.** No real API keys in CI secrets. Every test suite
  runs under `LLM_PROVIDER=mock`.
- **R9. Prose style + git hygiene.** No em dashes or en dashes between
  words; hyphens only for real technical terms. Conventional commits, no AI
  co-author trailers, squash merge only.
- **R10. No invented scope + honest failure modes.** A new agent needs a new
  issue, a maintainer nod, and an articulated differentiation angle from any
  close OSS competitor. Every agent's README has a "Where this fails"
  section with specific examples, not generic warnings.

## Agent README template

Every agent's README must have these sections, in order (enforced by
`scripts/lint_agents.py`, run in CI). Sections 1-8 are required; 9-11 are
optional badges that show up in the main README's agent grid.

1. **Title + one-liner** — clear tool description in plain English
2. **Technique demonstrated** — the educational anchor: what pattern this teaches
3. **Why this technique for this use case** — 2-4 sentences on why this framework/pattern fits here, and when it doesn't
4. **What it does** — 2-3 sentences, plain English
5. **How to run locally** — the 2-3 commands
6. **Code walkthrough** — 5-10 bullets pointing at the key files/lines a reader should read to understand the technique. At least one bullet must point at where the three R5 error cases are handled — this makes the "deployable" claim verifiable by reading the code
7. **When to use / when NOT to use** — bullet lists for each
8. **Where this fails** — honest failure modes with specific examples
9. **Optional: Live demo** — link to a HuggingFace Space if one exists
10. **Optional: Measured cost + latency** — from real runs, not estimates
11. **Optional: Accuracy** — from `eval/cases.jsonl` if applicable

## Proposing a new agent — step-by-step

1. Open an issue using the "New agent" template. It asks: what task, what
   technique/framework it demonstrates, why that framework is the right fit
   for this task, who uses this in real life, and what the closest existing
   OSS competitor is and how this differs.
2. Wait for a maintainer nod before writing code. This gate exists
   specifically to enforce R1 (real-world use case) and R3 (technique
   variety) — a proposed agent that doesn't clear the real-world bar, or
   duplicates a technique already in the catalog, will be turned back
   regardless of which framework it uses.
3. Fork the repo, branch `agent/<NN>_<short-name>` off `main`.
4. `python scripts/new_agent.py <short-name>` scaffolds the directory tree.
5. Implement your agent under `agents/NN_<name>/` (see `scripts/new_agent.py`'s
   scaffold for the expected file layout: `agent.py`, `schemas.py`,
   `prompts/`, optional `ui.py`, `eval/`, `tests/test_smoke.py`,
   `examples/`). Your agent's README must pass `python scripts/lint_agents.py`.
6. Add at least one smoke test that runs under `LLM_PROVIDER=mock` — CI
   never touches a real API key, so your agent must be exercisable without
   one.
7. Local checks before opening a PR:
   ```bash
   ruff check .
   python scripts/lint_agents.py
   python scripts/lint_readme.py
   uv run pytest
   ```
8. Open a PR against `main` using the PR template. Add your agent to the
   main README's shipped-agents table (and remove it from the roadmap table
   if it was listed there).
9. CI must be green. A maintainer reviews for: does the technique genuinely
   fit the use case, does the README teach the technique (not just describe
   the code), is the failure-mode section honest, does it follow R1-R10
   above.
10. Squash merge is the default merge strategy.

## Fixing a bug

1. Check open issues first — comment to claim it, or open a new issue if
   none exists.
2. Branch off `main` as `fix/<short-description>`.
3. Keep the diff as small as possible. Don't refactor unrelated code in the
   same PR.
4. Add or update a test that would have caught the bug, when practical.
5. Open a PR referencing the issue (`Fixes #123`).

## Pull request expectations

- One logical change per PR. A new agent and a bug fix are two PRs, not one.
- PR description states: what changed, why, and how it was tested (which
  commands you ran locally).
- CI (lint + smoke tests + unit tests) must pass before review. A red PR
  won't be reviewed.
- Expect review within roughly a week for a well-scoped PR — this is a side
  project maintained by one person; larger or more ambiguous PRs may take
  longer or get requested changes.
- No `Co-Authored-By: Claude` or other AI-tool attribution trailers on
  commits. PRs written substantially by an AI coding tool are welcome, but
  the commit author should be the human submitting the PR.
- Squash merge is the default; don't worry about crafting a pristine commit
  history inside your branch.

## Code of conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md). Be
respectful, no harassment. Report issues to rajeshmane711@gmail.com.
