"""Tests for scripts/lint_readme.py -- the main-README-vs-agents/-dir
consistency checker.

Covers:
- OK when README links exactly match the real agents/ directory
- FAIL when an actual agent dir isn't linked in the README (missing entry)
- FAIL when the README links a directory that doesn't exist (stale entry)
- Link-extraction regex correctly parses markdown table link syntax
- Real-repo integration: the actual README must pass against the actual
  agents/ directory
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import lint_readme


def _make_agent_dir(agents_root: Path, name: str) -> None:
    d = agents_root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "README.md").write_text("# stub\n\nstub agent for testing.\n", encoding="utf-8")


# ---------- find_referenced_agent_dirs ----------


def test_extracts_single_link():
    content = "| 01 | [Receipt Extractor](agents/01_receipt_extractor/) | ... |"
    assert lint_readme.find_referenced_agent_dirs(content) == {"01_receipt_extractor"}


def test_extracts_link_without_trailing_slash():
    content = "[Foo](agents/02_foo)"
    assert lint_readme.find_referenced_agent_dirs(content) == {"02_foo"}


def test_extracts_multiple_links():
    content = """
    | 01 | [A](agents/01_a/) | x |
    | 02 | [B](agents/02_b/) | y |
    """
    assert lint_readme.find_referenced_agent_dirs(content) == {"01_a", "02_b"}


def test_ignores_non_agent_links():
    content = "[SPEC](SPEC.md) and [docs](docs/adding_an_agent.md) but no agents link here"
    assert lint_readme.find_referenced_agent_dirs(content) == set()


def test_duplicate_links_to_same_agent_collapse_to_one():
    """An agent might be linked twice (once in the shipped table, once in
    the 'how to read this repo' prose) -- that's fine, not a duplicate-
    entry problem, just the same dir referenced from two places."""
    content = "[A](agents/01_a/) ... later ... [A again](agents/01_a/)"
    assert lint_readme.find_referenced_agent_dirs(content) == {"01_a"}


# ---------- check_readme: OK path ----------


def test_ok_when_readme_matches_agents_dir(tmp_path):
    agents_root = tmp_path / "agents"
    _make_agent_dir(agents_root, "01_receipt_extractor")

    readme = tmp_path / "README.md"
    readme.write_text(
        "| 01 | [Receipt Extractor](agents/01_receipt_extractor/) | ... |",
        encoding="utf-8",
    )

    problems = lint_readme.check_readme(readme, agents_root)
    assert problems == []


def test_ok_with_multiple_agents_all_linked(tmp_path):
    agents_root = tmp_path / "agents"
    _make_agent_dir(agents_root, "01_a")
    _make_agent_dir(agents_root, "02_b")

    readme = tmp_path / "README.md"
    readme.write_text(
        "[A](agents/01_a/)\n[B](agents/02_b/)\n",
        encoding="utf-8",
    )

    assert lint_readme.check_readme(readme, agents_root) == []


# ---------- check_readme: FAIL paths ----------


def test_fail_when_agent_dir_not_linked(tmp_path):
    agents_root = tmp_path / "agents"
    _make_agent_dir(agents_root, "01_receipt_extractor")

    readme = tmp_path / "README.md"
    readme.write_text("Nothing links to any agent here.", encoding="utf-8")

    problems = lint_readme.check_readme(readme, agents_root)
    assert len(problems) == 1
    assert "01_receipt_extractor" in problems[0]
    assert "not linked" in problems[0]


def test_fail_when_readme_links_nonexistent_agent(tmp_path):
    agents_root = tmp_path / "agents"
    agents_root.mkdir()

    readme = tmp_path / "README.md"
    readme.write_text("[Ghost](agents/99_ghost/)", encoding="utf-8")

    problems = lint_readme.check_readme(readme, agents_root)
    assert len(problems) == 1
    assert "99_ghost" in problems[0]
    assert "doesn't exist" in problems[0]


def test_fail_reports_both_missing_and_stale_together(tmp_path):
    agents_root = tmp_path / "agents"
    _make_agent_dir(agents_root, "01_real")

    readme = tmp_path / "README.md"
    readme.write_text("[Ghost](agents/99_ghost/)", encoding="utf-8")

    problems = lint_readme.check_readme(readme, agents_root)
    assert len(problems) == 2
    joined = " ".join(problems)
    assert "01_real" in joined
    assert "99_ghost" in joined


def test_fail_when_readme_missing_entirely(tmp_path):
    agents_root = tmp_path / "agents"
    _make_agent_dir(agents_root, "01_real")

    readme = tmp_path / "does_not_exist.md"
    problems = lint_readme.check_readme(readme, agents_root)
    assert len(problems) == 1
    assert "not found" in problems[0]


def test_agent_dir_without_readme_is_not_counted_as_shipped(tmp_path):
    """A scaffolded-but-unfinished agent dir (no README.md yet) must not be
    treated as 'shipped' -- it shouldn't force a README entry."""
    agents_root = tmp_path / "agents"
    (agents_root / "03_in_progress").mkdir(parents=True)
    # No README.md written for this one.

    readme = tmp_path / "README.md"
    readme.write_text("nothing here", encoding="utf-8")

    problems = lint_readme.check_readme(readme, agents_root)
    assert problems == []


def test_check_readme_does_not_crash_when_agents_dir_missing(tmp_path):
    """F1.6 review fix S1 regression guard: check_readme() must not raise
    FileNotFoundError when agents_dir doesn't exist. Previously the
    `if agents_dir.exists()` guard was placed INSIDE the set comprehension
    (filtering items after `.iterdir()` already ran), so it could never
    actually prevent `.iterdir()` from raising on a nonexistent directory --
    dead code that looked like a guard but wasn't one. A repo checked out
    without ever running scripts/new_agent.py (agents/ genuinely absent)
    must still get a clean, non-crashing report."""
    agents_root = tmp_path / "agents"  # deliberately never created

    readme = tmp_path / "README.md"
    readme.write_text("[Ghost](agents/99_ghost/)", encoding="utf-8")

    # Must not raise. Every referenced link should be reported as stale
    # since nothing in agents_dir exists at all.
    problems = lint_readme.check_readme(readme, agents_root)
    assert len(problems) == 1
    assert "99_ghost" in problems[0]
    assert "doesn't exist" in problems[0]


def test_check_readme_ok_when_both_readme_and_agents_dir_have_nothing(tmp_path):
    """Same missing-agents_dir scenario, but with a README that references
    no agents at all -- must report zero problems, not crash."""
    agents_root = tmp_path / "agents"  # deliberately never created

    readme = tmp_path / "README.md"
    readme.write_text("Nothing about agents here.", encoding="utf-8")

    assert lint_readme.check_readme(readme, agents_root) == []


# ---------- Real-repo integration ----------


def test_real_readme_matches_real_agents_dir():
    """The actual main README must pass against the actual agents/
    directory. If this fails, either the README wasn't updated after an
    agent was added/removed, or the linter has a bug -- either way we want
    to know before it reaches CI."""
    problems = lint_readme.check_readme(lint_readme._README_PATH, lint_readme._AGENTS_DIR)
    assert problems == [], f"README/agents mismatch: {problems}"
