## What changed

<!-- One or two sentences. What did you add/fix/change? -->

## Why

<!-- What real problem does this solve? For a new agent, what real-world use case does it serve? -->

## How this was tested

<!-- Which commands did you run locally? -->

```bash
ruff check .
python scripts/lint_agents.py
python scripts/lint_readme.py
uv run pytest
```

## Checklist (new agent PRs)

- [ ] Solves a real-world use case, not a toy demo (see CONTRIBUTING.md R1)
- [ ] Demonstrates a technique not already covered by another agent (R3)
- [ ] Agent's own code stays under ~500 LOC excluding UI (R2)
- [ ] README passes `python scripts/lint_agents.py` (all 8 required sections, see CONTRIBUTING.md)
- [ ] "Code walkthrough" section points at where the 3 real-error-handling cases live (R5)
- [ ] "Where this fails" has specific, honest examples — not generic warnings (R10)
- [ ] At least one smoke test runs under `LLM_PROVIDER=mock`
- [ ] Added to the main README's shipped-agents table (`python scripts/lint_readme.py` passes)
- [ ] Linked issue: Fixes #

## Checklist (bug fix PRs)

- [ ] One logical change, no unrelated refactors
- [ ] Added or updated a test that would have caught the bug
- [ ] Linked issue: Fixes #
