"""Tests for scripts/new_agent.py -- the agent-scaffolding tool."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import new_agent
import pytest

# ---------- slugify ----------


def test_slugify_lowercases_and_replaces_spaces():
    assert new_agent.slugify("My New Agent") == "my_new_agent"


def test_slugify_replaces_hyphens():
    assert new_agent.slugify("contract-reviewer") == "contract_reviewer"


def test_slugify_strips_invalid_characters():
    assert new_agent.slugify("Agent!! #2 (beta)") == "agent_2_beta"


def test_slugify_collapses_repeated_underscores():
    assert new_agent.slugify("a   b---c") == "a_b_c"


def test_slugify_rejects_empty_result():
    with pytest.raises(ValueError):
        new_agent.slugify("!!!")


# ---------- next_agent_number ----------


def test_next_agent_number_starts_at_one_when_empty(tmp_path, monkeypatch):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    monkeypatch.setattr(new_agent, "_AGENTS_DIR", agents_dir)
    assert new_agent.next_agent_number() == 1


def test_next_agent_number_one_past_highest(tmp_path, monkeypatch):
    agents_dir = tmp_path / "agents"
    (agents_dir / "01_foo").mkdir(parents=True)
    (agents_dir / "03_bar").mkdir(parents=True)  # gap at 02 -- must not reuse it
    monkeypatch.setattr(new_agent, "_AGENTS_DIR", agents_dir)
    assert new_agent.next_agent_number() == 4


def test_next_agent_number_ignores_non_numeric_dirs(tmp_path, monkeypatch):
    agents_dir = tmp_path / "agents"
    (agents_dir / "01_foo").mkdir(parents=True)
    (agents_dir / "not_numbered").mkdir(parents=True)
    monkeypatch.setattr(new_agent, "_AGENTS_DIR", agents_dir)
    assert new_agent.next_agent_number() == 2


def test_next_agent_number_when_agents_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(new_agent, "_AGENTS_DIR", tmp_path / "does_not_exist")
    assert new_agent.next_agent_number() == 1


# ---------- scaffold ----------


def test_scaffold_creates_expected_structure(tmp_path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    result = new_agent.scaffold("test agent", agents_dir=agents_dir)

    assert result.name == "01_test_agent"
    assert (result / "README.md").exists()
    assert (result / "pyproject.toml").exists()
    assert (result / "agent.py").exists()
    assert (result / "schemas.py").exists()
    assert (result / "prompts" / "prompt.txt").exists()
    assert (result / "tests" / "test_smoke.py").exists()
    # Regression guard: tests/__init__.py must NOT be scaffolded --
    # it causes pytest's importlib mode to collide every agent's
    # test_smoke.py under the same shared "tests.test_smoke" module
    # name. See conftest.py at repo root for the full context.
    assert not (result / "tests" / "__init__.py").exists()
    assert (result / "__init__.py").exists()
    assert (result / "examples").is_dir()


def test_scaffold_numbers_sequentially(tmp_path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    first = new_agent.scaffold("first", agents_dir=agents_dir)
    second = new_agent.scaffold("second", agents_dir=agents_dir)
    assert first.name == "01_first"
    assert second.name == "02_second"


def test_scaffold_refuses_to_overwrite_existing_dir(tmp_path, monkeypatch):
    """Auto-numbering (next_agent_number always returns max+1) means a
    genuine collision can't happen through scaffold()'s normal auto-
    increment path -- each call picks a fresh number. To exercise the
    FileExistsError guard (defensive against races / manual pre-creation),
    force next_agent_number to return the SAME number twice."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    monkeypatch.setattr(new_agent, "next_agent_number", lambda agents_dir=None: 1)

    new_agent.scaffold("dup", agents_dir=agents_dir)
    with pytest.raises(FileExistsError):
        new_agent.scaffold("dup", agents_dir=agents_dir)


def test_scaffold_readme_has_all_required_section_headers(tmp_path):
    """The scaffolded README stub should already have every §8.1 required
    H2 present (even if content is TODO) -- so a contributor who runs
    lint_agents.py immediately after scaffolding sees "fill in the TODOs"
    failures, not "you forgot a whole section" failures."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    result = new_agent.scaffold("stub check", agents_dir=agents_dir)
    content = (result / "README.md").read_text(encoding="utf-8")
    for heading in [
        "## Technique demonstrated",
        "## Why this technique for this use case",
        "## What it does",
        "## How to run locally",
        "## Code walkthrough",
        "## When to use / When NOT to use",
        "## Where this fails",
    ]:
        assert heading in content, f"scaffolded README missing {heading!r}"


def test_scaffold_agent_py_is_syntactically_valid_python(tmp_path):
    """A bare Ellipsis parameter list (`def run(...) -> Result:`) is not
    valid Python syntax -- a contributor running the scaffold and
    immediately trying `python -m agent` before editing anything would hit
    a SyntaxError the tool itself introduced. Parse every generated .py file
    with ast.parse() to guarantee this can't regress silently."""
    import ast

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    result = new_agent.scaffold("syntax check", agents_dir=agents_dir)

    for py_file in result.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"{py_file.relative_to(result)} is not valid Python: {exc}")
