"""Checks the main README's "Shipped agents" table matches the actual
`agents/` directory — catches a forgotten README update after adding/
renaming/removing an agent, or a stale row left behind.

The table is expected to link each row to its agent directory, e.g.:

    | 01 | [Receipt Extractor](agents/01_receipt_extractor/) | ... |

Any `agents/<dir>/` with a README.md that ISN'T referenced by a link in the
table is a FAIL (README is missing an entry). Any link in the table pointing
at a directory that doesn't exist is also a FAIL (stale/broken entry).

Run:
    python scripts/lint_readme.py

Exit codes:
    0 -- README matches the agents/ directory exactly
    1 -- mismatch found (details printed to stdout)
    2 -- README.md not found at the repo root
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_README_PATH = _REPO_ROOT / "README.md"
_AGENTS_DIR = _REPO_ROOT / "agents"

# Matches markdown links whose target starts with "agents/", e.g.
# "[Receipt Extractor](agents/01_receipt_extractor/)" or without trailing
# slash. Captures the directory name (the path segment right after
# "agents/").
_AGENT_LINK_RE = re.compile(r"\]\(agents/([^/)\s]+)/?\)")


def find_actual_agent_dirs() -> set[str]:
    """Directory names under agents/ that have a README.md (i.e. are
    actually shipped agents, not scaffolding-in-progress or stray dirs)."""
    if not _AGENTS_DIR.exists():
        return set()
    return {
        d.name
        for d in _AGENTS_DIR.iterdir()
        if d.is_dir() and (d / "README.md").exists()
    }


def find_referenced_agent_dirs(readme_content: str) -> set[str]:
    """Every distinct agents/<dir>/ target linked anywhere in the README."""
    return set(_AGENT_LINK_RE.findall(readme_content))


def check_readme(readme_path: Path, agents_dir: Path) -> list[str]:
    """Returns a list of human-readable problems; empty means it passes."""
    problems: list[str] = []

    if not readme_path.exists():
        return [f"README not found at {readme_path}"]

    content = readme_path.read_text(encoding="utf-8")
    referenced = find_referenced_agent_dirs(content)

    # F1.6 review fix S1: the `if agents_dir.exists()` used to sit INSIDE the
    # comprehension, filtering items already yielded by `.iterdir()` -- but
    # `.iterdir()` itself raises FileNotFoundError before that filter ever
    # runs if agents_dir doesn't exist, so the guard was dead code that could
    # never prevent the crash it looked like it was guarding against. Moved
    # to an early check, matching find_actual_agent_dirs()'s already-correct
    # pattern above.
    actual: set[str] = set()
    if agents_dir.exists():
        actual = {
            d.name for d in agents_dir.iterdir() if d.is_dir() and (d / "README.md").exists()
        }

    missing_from_readme = sorted(actual - referenced)
    stale_in_readme = sorted(referenced - actual)

    for name in missing_from_readme:
        problems.append(
            f"agents/{name}/ exists (has a README.md) but is not linked anywhere "
            f"in the main README -- add it to the shipped-agents table"
        )
    for name in stale_in_readme:
        problems.append(
            f"main README links to agents/{name}/ but that directory doesn't exist "
            f"(or has no README.md) -- stale entry, remove or fix the link"
        )

    return problems


def main() -> int:
    if not _README_PATH.exists():
        print(f"error: README not found at {_README_PATH}", file=sys.stderr)
        return 2

    problems = check_readme(_README_PATH, _AGENTS_DIR)
    if not problems:
        actual_count = len(find_actual_agent_dirs())
        print(f"OK   README.md matches agents/ ({actual_count} shipped agent(s) linked)")
        return 0

    print("FAIL README.md")
    for problem in problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
