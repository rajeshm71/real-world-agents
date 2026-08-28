# Sample repo

A tiny fake Python project used as the default target for the codebase
Q&A demo. Three source files and a top-level README, all self-authored
for this repository -- no third-party content, MIT alongside the rest
of the project.

The layout intentionally mirrors what a small hobby project looks like
so realistic questions ("where is `greet` defined?", "what does
`cli.py` do?") exercise the agent's tools end-to-end.

To swap in a different repo, point `--repo` at any directory:

```bash
uv run python -m agent "where is X defined?" --repo ~/code/my-project
```
