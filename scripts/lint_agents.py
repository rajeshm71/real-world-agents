"""Enforces SPEC.md §8.1's 11-section README template on every shipped agent.

Sections 1-8 are REQUIRED for shipping (R4). Sections 9-11 are OPTIONAL
badges that show up in the main README's agent grid when present.

Section 1 is title + one-liner (H1 + paragraph); sections 2-11 are H2
headers matching required keyword substrings (case-insensitive), in order,
no duplicates.

Run:
    python scripts/lint_agents.py

Exit codes:
    0 -- every agent README passes
    1 -- at least one FAIL (details printed to stdout)
    2 -- no agent READMEs found (probably a wrong working directory)

Parallel to sibling rag-recipes' `scripts/lint_notebooks.py` -- same pattern
of "parse the artifact, check for required sections, print per-file
OK/FAIL, exit non-zero on failure."
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_DIR = _REPO_ROOT / "agents"

# H1 (single #) is the title; H2 (##) marks each subsequent section.
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

# F1.3 review fix S2: fenced code blocks (```...```) can contain lines
# starting with # (shell comments, e.g. "# then edit .env...") that the
# H1/H2 regexes would otherwise misinterpret as markdown headers, producing
# confusing "duplicate section" / "out of order" errors on a perfectly
# valid README. Strip fenced blocks before header extraction. DOTALL so
# `.` matches newlines inside the fence; non-greedy so multiple separate
# fences in one file are each matched individually rather than the whole
# span from the first opening ``` to the last closing ``` collapsing into
# one blob.
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _strip_code_fences(content: str) -> str:
    """Replace fenced code blocks with nothing (not even a placeholder line)
    before header extraction, so `#`/`##` inside a shell comment never
    matches as a markdown header."""
    return _CODE_FENCE_RE.sub("", content)

# (section number, required substring in the H2 title, case-insensitive)
# Section 1 is title + one-liner (H1 + paragraph), checked separately.
# Sections 2-8 are the REQUIRED H2 sections. Sections 9-11 are OPTIONAL.
_REQUIRED_H2_KEYWORDS: list[tuple[int, str]] = [
    (2, "technique demonstrated"),
    (3, "why this technique"),
    (4, "what it does"),
    (5, "how to run"),
    (6, "code walkthrough"),
    (7, "when to use"),
    (8, "where this fails"),
]

_OPTIONAL_H2_KEYWORDS: list[tuple[int, str]] = [
    (9, "live demo"),
    (10, "measured cost"),
    (11, "accuracy"),
]


@dataclass
class LintResult:
    path: Path
    problems: list[str]
    optional_present: list[int]  # section numbers 9-11 that ARE present (earn badges)

    @property
    def ok(self) -> bool:
        return not self.problems


def find_agent_readmes() -> list[Path]:
    """Return every `agents/*/README.md` under the repo, sorted."""
    if not _AGENTS_DIR.exists():
        return []
    return sorted(
        agent_dir / "README.md"
        for agent_dir in _AGENTS_DIR.iterdir()
        if agent_dir.is_dir() and (agent_dir / "README.md").exists()
    )


def _extract_h1(content: str) -> tuple[str | None, int]:
    """Return (title text, line offset of the H1 in `content`), or (None, -1)."""
    match = _H1_RE.search(content)
    if match is None:
        return None, -1
    return match.group(1), match.start()


def _paragraph_after(content: str, offset: int) -> str | None:
    """Return the first non-empty paragraph AFTER a given offset in `content`.
    Used to check that section 1 (title + one-liner) actually has the
    one-liner following the H1."""
    tail = content[offset:]
    # Skip the H1 line itself
    tail_lines = tail.splitlines()[1:]
    para: list[str] = []
    for line in tail_lines:
        stripped = line.strip()
        if not stripped:
            if para:
                break
            continue
        # A subsequent header (H1/H2/H3...) ends the one-liner search
        # before we found one.
        if stripped.startswith("#"):
            break
        para.append(stripped)
    return " ".join(para) if para else None


def _extract_h2_sequence(content: str) -> list[str]:
    """Return the H2 titles in document order."""
    return [m.group(1) for m in _H2_RE.finditer(content)]


def lint_readme(path: Path) -> LintResult:
    """Lint one agent's README against SPEC §8.1. Returns LintResult with
    a (possibly empty) list of human-readable problems + a list of which
    optional sections (9-11) are present.
    """
    problems: list[str] = []
    raw_content = path.read_text(encoding="utf-8")
    # F1.3 review fix S2: strip fenced code blocks before any header
    # extraction happens (both H1 and H2), so shell comments inside
    # ```bash ... ``` blocks never get misread as markdown headers.
    content = _strip_code_fences(raw_content)

    # --- Section 1: title (H1) + one-liner (non-empty paragraph after H1) ---
    title, h1_offset = _extract_h1(content)
    if title is None:
        problems.append("section 1 missing: no H1 title (expected `# <title>` on its own line)")
    else:
        one_liner = _paragraph_after(content, h1_offset)
        if not one_liner:
            problems.append(
                "section 1 incomplete: H1 title present but no one-liner paragraph "
                "immediately below it (SPEC §8.1: 'clear tool description in plain English')"
            )

    # --- Sections 2-8 (REQUIRED) + optional 9-11 detection ---
    h2_titles = _extract_h2_sequence(content)
    h2_titles_lower = [t.lower() for t in h2_titles]

    # Check for duplicate section keywords -- a duplicate is an authoring bug
    # (a naive extraction would silently keep just the last one).
    for keyword_list in (_REQUIRED_H2_KEYWORDS, _OPTIONAL_H2_KEYWORDS):
        for _section_num, keyword in keyword_list:
            count = sum(1 for title in h2_titles_lower if keyword in title)
            if count > 1:
                problems.append(
                    f"duplicate section: keyword {keyword!r} appears in {count} H2 titles "
                    f"(each SPEC §8.1 section must appear at most once)"
                )

    # Find each required section's position; check presence and order.
    def _find_position(keyword: str) -> int:
        """Return index of the first H2 whose title contains `keyword`, or -1."""
        for i, title in enumerate(h2_titles_lower):
            if keyword in title:
                return i
        return -1

    positions: dict[int, int] = {}
    for section_num, keyword in _REQUIRED_H2_KEYWORDS:
        pos = _find_position(keyword)
        if pos == -1:
            problems.append(
                f"section {section_num} missing: no H2 title containing {keyword!r}"
            )
        else:
            positions[section_num] = pos

    # Order check: required section positions must be strictly increasing.
    prev_pos = -1
    prev_section = 0
    for section_num, _keyword in _REQUIRED_H2_KEYWORDS:
        pos = positions.get(section_num)
        if pos is None:
            continue  # already flagged as missing
        if pos <= prev_pos:
            problems.append(
                f"section {section_num} out of order: appears at H2 index {pos} "
                f"but section {prev_section} was at index {prev_pos} (expected strictly increasing)"
            )
        prev_pos = pos
        prev_section = section_num

    # Detect optional-section badges (no fail; just report).
    optional_present: list[int] = []
    for section_num, keyword in _OPTIONAL_H2_KEYWORDS:
        if _find_position(keyword) != -1:
            optional_present.append(section_num)

    return LintResult(path=path, problems=problems, optional_present=optional_present)


def main() -> int:
    readmes = find_agent_readmes()
    if not readmes:
        print(
            f"error: no agent READMEs found under {_AGENTS_DIR}/*/README.md",
            file=sys.stderr,
        )
        return 2

    any_failed = False
    for path in readmes:
        result = lint_readme(path)
        # Relative path for readability in CI logs.
        try:
            rel = path.relative_to(_REPO_ROOT)
        except ValueError:
            rel = path

        if result.ok:
            badge_summary = (
                f"  (optional badges: {', '.join(str(n) for n in result.optional_present)})"
                if result.optional_present
                else ""
            )
            print(f"OK   {rel}{badge_summary}")
        else:
            any_failed = True
            print(f"FAIL {rel}")
            for problem in result.problems:
                print(f"  - {problem}")

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
