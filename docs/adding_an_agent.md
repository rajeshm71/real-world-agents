# Adding a new agent

This is the implementation-level companion to
[CONTRIBUTING.md](../CONTRIBUTING.md)'s step-by-step overview: concrete file
templates and a checklist for building an agent that passes every gate this
repo enforces.

Before writing code: open an issue with the "New agent" template and get a
maintainer nod. See [CONTRIBUTING.md](../CONTRIBUTING.md) for why: mainly to
avoid you investing time in an agent that fails the R1 (real-world use case)
or R3 (technique variety) bar, both explained in CONTRIBUTING.md's "Hard
rules" section.

## 1. Scaffold the directory

```bash
python scripts/new_agent.py <short-name>
```

This creates `agents/<NN>_<short-name>/` with the structure below, stubbed
with TODO comments. `<NN>` is the next unused two-digit number.

```
agents/NN_<short-name>/
├── README.md              # fill in the 8 required sections (see §4 below)
├── pyproject.toml         # your own runtime deps (budget: ≤10)
├── agent.py               # main entry point, runnable via `python -m agent`
├── schemas.py             # Pydantic models, if your technique uses them
├── prompts/                # prompt templates as .txt files
├── ui.py                  # Gradio Blocks or minimal HTMX, if you have a UI
├── eval/                  # optional: cases.jsonl + run_eval.py
├── tests/
│   └── test_smoke.py      # MUST run under LLM_PROVIDER=mock
└── examples/              # sample inputs for the demo
```

## 2. `agent.py` skeleton

The one contract every agent's `agent.py` must satisfy: runnable via
`python -m agent` from inside the agent's own directory, and testable without
any real API key.

```python
"""<Agent name>: agent #NN of real-world-agents.

Technique demonstrated: <one line>.
Why this technique for this use case: <why the framework fits>.
"""

from __future__ import annotations

import os

# Dual-mode import pattern (see agents/01_receipt_extractor/agent.py for the
# full rationale): relative resolves when loaded as a package submodule by
# tests; absolute resolves when run directly via `python -m agent`.
try:
    from .schemas import YourResultModel
except ImportError:
    from schemas import YourResultModel


def resolve_provider() -> str:
    """LLM_PROVIDER env var, default "openai". See common/llm.py's
    resolve_model() for the matching model-resolution half of this contract.
    """
    return os.environ.get("LLM_PROVIDER", "openai").lower()


def run_agent(...) -> YourResultModel:
    if resolve_provider() == "mock":
        return _mock_result(...)
    # real provider call here
    ...


def _mock_result(...) -> YourResultModel:
    """Deterministic canned result. No network call, no API key. This is
    what every test and CI run exercises."""
    ...


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="<short-name>")
    # ... your CLI args ...
    args = parser.parse_args()
    # ... call run_agent(), print result ...
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## 3. Real error handling (R5)

Every agent must handle the obvious failure modes for its own inputs
explicitly, not happy-path only. At minimum: malformed input, a rate-limit
or transient API failure, and whatever "partial success" looks like for your
technique. See `agents/01_receipt_extractor/agent.py`'s
`_translate_api_error()` for a worked example (checks exception class before
message strings, attaches partial output for the UI to show).

## 4. The README template

Required sections, in order, checked by `python scripts/lint_agents.py`:

1. Title + one-liner (H1 + a paragraph immediately below it)
2. `## Technique demonstrated`
3. `## Why this technique for this use case`
4. `## What it does`
5. `## How to run locally`
6. `## Code walkthrough`: at least one bullet MUST point at where your R5
   error handling lives
7. `## When to use / When NOT to use`
8. `## Where this fails`: specific, honest examples

Optional (earn a badge in the main README when present, not required to
ship): `## Live demo`, `## Measured cost + latency`, `## Accuracy`.

Run the linter locally before opening a PR:

```bash
python scripts/lint_agents.py
```

## 5. Tests

At minimum, `tests/test_smoke.py` with:

- One test that your agent's mock path returns a valid, schema-conformant
  result
- Tests for your R5 error-handling branches
- Tests for anything that's pure logic and doesn't need a real API call
  (e.g. `agents/01_receipt_extractor/tests/test_smoke.py`'s provider/model
  resolution tests)

## 6. Register it in the main README

Add a row to the "Shipped agents" table (and remove the corresponding row
from "Roadmap" if it was listed there). Run:

```bash
python scripts/lint_readme.py
```

This checks the main README's shipped-agents table actually matches the
`agents/` directory: catches a forgotten README update or a stale entry for
an agent that got renamed/removed.

## 7. Full local check before opening a PR

```bash
ruff check .
python scripts/lint_agents.py
python scripts/lint_readme.py
uv run pytest
```

All four must be clean/green. CI runs the same checks.
