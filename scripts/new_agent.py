"""Scaffolds a new agent's directory tree, per SPEC.md §8.

Run:
    python scripts/new_agent.py <short-name>

Creates `agents/<NN>_<short-name>/` where `<NN>` is the next unused
two-digit number (one past the highest existing agent number), with stub
files matching the structure every agent is expected to have. Every stub
file has a TODO comment pointing at what to fill in and, where relevant,
a link to docs/adding_an_agent.md.

Does NOT: create a git branch, open an issue, or write any real content --
see CONTRIBUTING.md for the full workflow this script is step 4 of.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_DIR = _REPO_ROOT / "agents"

_NUMBER_PREFIX_RE = re.compile(r"^(\d+)_")


def next_agent_number(agents_dir: Path | None = None) -> int:
    """One past the highest NN prefix currently in `agents_dir`. Starts at 1
    if that directory is empty or doesn't exist yet. Accepts an explicit
    `agents_dir` (rather than always reading the module-level default) so
    tests can point it at a tmp_path without touching the real repo.

    `agents_dir` defaults to None (resolved to the CURRENT value of the
    module-level `_AGENTS_DIR` inside the function body) rather than
    defaulting directly to `_AGENTS_DIR` -- Python binds a parameter default
    once at function-definition time, so a default of `_AGENTS_DIR` would
    freeze in the value from module load and silently ignore any later
    `monkeypatch.setattr(new_agent, "_AGENTS_DIR", ...)` in tests.
    """
    if agents_dir is None:
        agents_dir = _AGENTS_DIR
    if not agents_dir.exists():
        return 1
    numbers = []
    for d in agents_dir.iterdir():
        if not d.is_dir():
            continue
        match = _NUMBER_PREFIX_RE.match(d.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def slugify(short_name: str) -> str:
    """Normalize a human-typed name into a directory-safe slug: lowercase,
    spaces/hyphens -> underscores, strip anything else non-alphanumeric."""
    slug = short_name.strip().lower()
    slug = re.sub(r"[\s-]+", "_", slug)
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        raise ValueError(f"'{short_name}' has no valid characters for a directory name")
    return slug


def scaffold(short_name: str, agents_dir: Path | None = None) -> Path:
    """Create the agent directory tree. Returns the created directory path.
    Raises FileExistsError if the target directory already exists (never
    silently overwrite a contributor's in-progress work). Same None-default
    rationale as next_agent_number() above -- avoids the late-binding trap.
    """
    if agents_dir is None:
        agents_dir = _AGENTS_DIR
    number = next_agent_number(agents_dir)
    slug = slugify(short_name)
    dir_name = f"{number:02d}_{slug}"
    agent_dir = agents_dir / dir_name

    if agent_dir.exists():
        raise FileExistsError(f"{agent_dir} already exists")

    (agent_dir / "prompts").mkdir(parents=True)
    (agent_dir / "tests").mkdir(parents=True)
    (agent_dir / "examples").mkdir(parents=True)
    (agent_dir / "__init__.py").write_text("", encoding="utf-8")
    (agent_dir / "tests" / "__init__.py").write_text("", encoding="utf-8")

    (agent_dir / "README.md").write_text(
        """# TODO: <Agent title>

TODO: one-liner description here.

## Technique demonstrated

TODO: what pattern does this teach? See SPEC.md R3 -- must be a technique
not already covered by another agent.

## Why this technique for this use case

TODO: 2-4 sentences on why this framework/pattern is the right fit, and
when it isn't.

## What it does

TODO: 2-3 sentences, plain English input/output description.

## How to run locally

TODO: the commands to run this agent (CLI, and UI if it has one).

## Code walkthrough

TODO: bullets pointing at key files/lines. At least one MUST point at
where your R5 error handling lives (rate limits, malformed input, tool/API
failure).

## When to use / When NOT to use

TODO.

## Where this fails

TODO: specific, honest failure modes with real examples -- not generic
warnings. See docs/adding_an_agent.md and SPEC.md R10.
""",
        encoding="utf-8",
    )

    (agent_dir / "pyproject.toml").write_text(
        f"""[project]
name = "{slug.replace('_', '-')}"
version = "0.1.0"
description = "TODO: one-liner. Agent #{number:02d} of real-world-agents."
requires-python = ">=3.11"
license = {{ text = "MIT" }}
dependencies = [
    "real-world-agents",  # for common.llm / common.pricing
    # TODO: add your agent's own runtime deps here (budget: <=10)
]

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.uv.sources]
real-world-agents = {{ workspace = true }}

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["."]
""",
        encoding="utf-8",
    )

    (agent_dir / "agent.py").write_text(
        f'''"""TODO: agent title -- agent #{number:02d} of real-world-agents.

Technique demonstrated: TODO.
Why this technique for this use case: TODO.

See docs/adding_an_agent.md for the full checklist this file should satisfy.
"""

from __future__ import annotations

import os

# Dual-mode import pattern: relative resolves when loaded as a package
# submodule (by tests); absolute resolves when run directly via
# `python -m agent`. See agents/01_receipt_extractor/agent.py for the full
# rationale if this looks unfamiliar.
try:
    from .schemas import Result  # TODO: replace with your actual schema import
except ImportError:
    from schemas import Result


def resolve_provider() -> str:
    """LLM_PROVIDER env var, default "openai"."""
    return os.environ.get("LLM_PROVIDER", "openai").lower()


# F1.6 review fix S2: `def run(...) -> Result:` (bare Ellipsis as a
# parameter list) is not valid Python syntax -- a contributor running this
# scaffold immediately after `new_agent.py` (before editing anything) hit a
# SyntaxError instead of a runnable "fill in the TODOs" stub. Real
# parameter name + explicit `# TODO` comment instead, so the file parses
# as-is and the placeholder-ness is communicated without breaking syntax.
def run(input_data: str) -> Result:  # TODO: replace input_data with your real signature
    if resolve_provider() == "mock":
        return _mock_result(input_data)
    # TODO: real provider call
    raise NotImplementedError


def _mock_result(input_data: str) -> Result:  # TODO: replace input_data with your real signature
    """Deterministic canned result -- no network call, no API key. This is
    what CI and the test suite exercise."""
    raise NotImplementedError


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="{slug}")
    # TODO: add your CLI args
    parser.parse_args()
    # TODO: call run(), print result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
        encoding="utf-8",
    )

    (agent_dir / "schemas.py").write_text(
        '''"""TODO: Pydantic models for this agent's structured input/output, if any."""

from __future__ import annotations

from pydantic import BaseModel


class Result(BaseModel):
    """TODO: replace with your actual result schema."""

    pass
''',
        encoding="utf-8",
    )

    (agent_dir / "prompts" / "prompt.txt").write_text(
        "TODO: prompt text for this agent's LLM call(s).\n", encoding="utf-8"
    )

    (agent_dir / "tests" / "test_smoke.py").write_text(
        f'''"""Smoke tests for TODO-agent-name. All run under LLM_PROVIDER=mock
per SPEC R8 -- CI never touches a real API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AGENT_DIR.parent))

import importlib

_agent_pkg = importlib.import_module("{dir_name}.agent")


def test_mock_path_returns_a_result(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    # TODO: call _agent_pkg's run()/mock path and assert on the result.
    raise NotImplementedError("fill in the mock-path smoke test")
''',
        encoding="utf-8",
    )

    return agent_dir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/new_agent.py <short-name>", file=sys.stderr)
        return 2

    try:
        agent_dir = scaffold(sys.argv[1])
    except (ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rel = agent_dir.relative_to(_REPO_ROOT)
    print(f"Scaffolded {rel}/")
    print("Next: fill in the TODOs, then see docs/adding_an_agent.md for the full checklist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
