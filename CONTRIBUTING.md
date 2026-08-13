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
  [SPEC.md](SPEC.md) R3)
- **Improve an existing agent** — better error handling, more eval cases,
  honest failure-mode analysis, better README pedagogy
- **Add cases to an agent's eval set** — helps make the accuracy claim more
  credible
- **Improve `common/`** — better error messages, better docs, fewer
  surprises
- **Report bugs**

Do NOT open a PR without an issue first for: new top-level deps in
`common/`, new LLM providers (chassis contract change), UI/dashboard
reworks, or changes to the per-agent README template ([SPEC.md](SPEC.md)
§8.1). Open an issue and get a maintainer nod before investing time in any
of these.

## Proposing a new agent — step-by-step

1. Open an issue using the "New agent" template. It asks: what task, what
   technique/framework it demonstrates, why that framework is the right fit
   for this task, who uses this in real life, and what the closest existing
   OSS competitor is and how this differs (see [SPEC.md](SPEC.md) §7's
   "excluded" list for the level of scrutiny expected here).
2. Wait for a maintainer nod before writing code. This gate exists
   specifically to enforce R1 (real-world use case) and R3 (technique
   variety) — a proposed agent that doesn't clear the real-world bar, or
   duplicates a technique already in the catalog, will be turned back
   regardless of which framework it uses.
3. Fork the repo, branch `agent/<NN>_<short-name>` off `main`.
4. `python scripts/new_agent.py <short-name>` scaffolds the directory tree.
5. Implement following [SPEC.md](SPEC.md) §8's structure. Your agent's
   README must pass `python scripts/lint_agents.py`.
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
   ([SPEC.md](SPEC.md) §17).
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
