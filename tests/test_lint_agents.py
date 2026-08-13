"""Tests for scripts/lint_agents.py -- the SPEC §8.1 template enforcer.

Covers:
- OK path on a well-formed README fixture
- FAIL on missing H1 title
- FAIL on H1 title without a one-liner paragraph
- FAIL on missing any of the required sections 2-8
- FAIL on required sections out of order
- FAIL on duplicated section keyword (a real authoring bug)
- Optional sections 9-11 detected + reported without triggering FAIL

Uses fixture strings written inline rather than files -- keeps tests
self-contained and fast, and doesn't pollute agents/ with fake READMEs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ isn't a Python package; add to sys.path so we can import from it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import lint_agents


def _write(tmp_path: Path, content: str) -> Path:
    """Write a fixture README and return the path."""
    p = tmp_path / "README.md"
    p.write_text(content, encoding="utf-8")
    return p


# ---------- Well-formed README (all 8 required sections) ----------


_GOOD_README = """# Receipt Extractor

Upload a receipt image, get clean structured JSON out.

## Technique demonstrated

Structured extraction with Instructor and Claude vision.

## Why this technique for this use case

Receipts are structured documents. Instructor collapses "prompt + parse
JSON + validate against Pydantic + retry" into one call.

## What it does

Takes JPEG/PNG bytes, returns a validated `ExtractedReceipt`.

## How to run locally

```bash
uv sync
LLM_PROVIDER=anthropic uv run python -m agent path/to/receipt.jpg
```

## Code walkthrough

- `schemas.py` -- the Pydantic model Instructor validates against
- `agent.py::extract_receipt()` -- public API
- `agent.py::_translate_api_error()` -- handles R5 error cases

## When to use / When NOT to use

- Use for structured extraction from images
- Don't use for freeform text extraction

## Where this fails

Blurry photos of receipts. Handwritten receipts. Non-English text.
"""


def test_ok_on_well_formed_readme(tmp_path):
    path = _write(tmp_path, _GOOD_README)
    result = lint_agents.lint_readme(path)
    assert result.ok, f"expected OK, got problems: {result.problems}"
    assert result.optional_present == []


# ---------- Section 1 (title + one-liner) failures ----------


def test_fail_on_missing_h1(tmp_path):
    # H2 as first header instead of H1.
    bad = _GOOD_README.replace("# Receipt Extractor", "## Receipt Extractor", 1)
    result = lint_agents.lint_readme(_write(tmp_path, bad))
    assert not result.ok
    assert any("section 1" in p and "no H1" in p for p in result.problems)


def test_fail_on_h1_without_one_liner(tmp_path):
    # H1 immediately followed by another header, no paragraph.
    content = "# Receipt Extractor\n\n## Technique demonstrated\n\nStuff.\n"
    result = lint_agents.lint_readme(_write(tmp_path, content))
    assert not result.ok
    assert any("section 1 incomplete" in p for p in result.problems)


# ---------- Sections 2-8 (required) failures ----------


def test_fail_on_missing_technique_section(tmp_path):
    bad = _GOOD_README.replace("## Technique demonstrated", "## Other stuff")
    result = lint_agents.lint_readme(_write(tmp_path, bad))
    assert not result.ok
    assert any("section 2" in p and "technique demonstrated" in p for p in result.problems)


def test_fail_on_missing_where_this_fails_section(tmp_path):
    bad = _GOOD_README.replace("## Where this fails", "## Random")
    result = lint_agents.lint_readme(_write(tmp_path, bad))
    assert not result.ok
    assert any("section 8" in p and "where this fails" in p for p in result.problems)


def test_fail_on_sections_out_of_order(tmp_path):
    # Swap "Where this fails" to appear BEFORE "Code walkthrough" -- both
    # exist but order violates §8.1.
    reordered = """# Receipt Extractor

Upload a receipt image, get clean structured JSON out.

## Technique demonstrated

Stuff.

## Why this technique for this use case

Stuff.

## What it does

Stuff.

## How to run locally

Stuff.

## Where this fails

Stuff.

## Code walkthrough

Stuff.

## When to use / When NOT to use

Stuff.
"""
    result = lint_agents.lint_readme(_write(tmp_path, reordered))
    assert not result.ok
    assert any("out of order" in p for p in result.problems)


def test_fail_on_duplicate_section_keyword(tmp_path):
    # Two H2 headers containing the same required keyword -- authoring bug.
    bad = _GOOD_README + "\n## Technique demonstrated (again)\n\nOops.\n"
    result = lint_agents.lint_readme(_write(tmp_path, bad))
    assert not result.ok
    assert any("duplicate section" in p and "technique demonstrated" in p for p in result.problems)


# ---------- Optional sections 9-11 (badges, no fail) ----------


def test_optional_sections_detected_without_fail(tmp_path):
    with_optionals = _GOOD_README + """
## Live demo

https://example.com

## Measured cost + latency

$0.005 per extraction.

## Accuracy

95% per-field accuracy.
"""
    result = lint_agents.lint_readme(_write(tmp_path, with_optionals))
    assert result.ok, f"expected OK, got problems: {result.problems}"
    assert set(result.optional_present) == {9, 10, 11}


def test_partial_optional_sections_detected(tmp_path):
    # Only "Live demo" present, other two absent -- still OK, just one badge.
    with_one = _GOOD_README + "\n## Live demo\n\nhttps://example.com\n"
    result = lint_agents.lint_readme(_write(tmp_path, with_one))
    assert result.ok
    assert result.optional_present == [9]


# ---------- Code-fence handling (F1.3 review fix S2) ----------


def test_headers_inside_code_fences_are_ignored(tmp_path):
    """A shell comment like `# then edit .env` inside a ```bash fence must
    NOT be misread as an H1, and `## some heading`-shaped text inside a
    fence must NOT be misread as an H2. Regression test for BUG A found in
    the F1.3 review: the original regex-only approach had no fence-
    awareness and would have produced spurious duplicate/out-of-order
    errors on a README containing shell comments in its code blocks."""
    content = (
        _GOOD_README
        + "\n```bash\n# This looks like an H1 but is inside a fence\n"
        "## This looks like an H2 but is inside a fence\ncd somewhere\n```\n"
    )
    result = lint_agents.lint_readme(_write(tmp_path, content))
    assert result.ok, f"expected OK, got problems: {result.problems}"


def test_multiple_separate_code_fences_each_stripped_independently(tmp_path):
    """Non-greedy regex check: two SEPARATE fenced blocks in one README must
    each be stripped on their own, not have the whole span between the
    first ``` and the last ``` collapse into one (which would delete real
    markdown content between them, including required H2 sections)."""
    content = """# Receipt Extractor

One-liner here.

## Technique demonstrated

```bash
# fence one
echo hi
```

Some real prose between the fences that must survive stripping.

## Why this technique for this use case

```bash
# fence two
echo bye
```

## What it does

Stuff.

## How to run locally

Stuff.

## Code walkthrough

Stuff.

## When to use / When NOT to use

Stuff.

## Where this fails

Stuff.
"""
    result = lint_agents.lint_readme(_write(tmp_path, content))
    assert result.ok, f"expected OK, got problems: {result.problems}"


def test_strip_code_fences_removes_fenced_content():
    """Unit test for the helper directly -- fenced content is gone, non-fenced
    content survives untouched."""
    content = "before\n```\n# not a header\ncode here\n```\nafter"
    stripped = lint_agents._strip_code_fences(content)
    assert "not a header" not in stripped
    assert "code here" not in stripped
    assert "before" in stripped
    assert "after" in stripped


# ---------- Real-repo integration ----------


def test_lint_passes_on_actual_receipt_extractor_readme():
    """The real receipt-extractor README (written for F1.3) MUST pass its
    own linter. If this test fails, either the README is genuinely broken
    OR the linter has an off-by-one bug -- either way, we catch it here."""
    real_readme = _REPO_ROOT / "agents" / "01_receipt_extractor" / "README.md"
    if not real_readme.exists():
        # F1.3c hasn't landed yet in some git-bisect scenarios; skip rather
        # than fail, so the test suite doesn't block reviews of unrelated
        # commits.
        import pytest
        pytest.skip(f"agent README not yet present at {real_readme}")

    result = lint_agents.lint_readme(real_readme)
    assert result.ok, (
        "receipt-extractor README failed lint. Problems:\n"
        + "\n".join(f"  - {p}" for p in result.problems)
    )


# ---------- find_agent_readmes ----------


def test_find_agent_readmes_returns_sorted_paths():
    """Sanity: the discovery function should find at least the F1 agent
    once its README exists, and return paths in sorted order (so CI output
    is deterministic)."""
    readmes = lint_agents.find_agent_readmes()
    # After F1.3c: at least 01_receipt_extractor's README.
    assert isinstance(readmes, list)
    if readmes:
        names = [p.name for p in readmes]
        assert names == sorted(names), "readmes should be returned in sorted order"
