"""Accuracy eval harness for the receipt extractor. Scores extract_receipt()
against a hand-labeled set of real receipts, prints and writes a per-field
accuracy table.

Run (from inside agents/01_receipt_extractor/, real API key required):
    LLM_PROVIDER=openai uv run python eval/run_eval.py

See eval/README.md for the cases.jsonl schema and how to add real cases.
This file's pure scoring logic (compare_field, score_case,
aggregate_accuracy, format_results_table, load_cases) has zero dependency
on the agent module or any provider SDK -- it's tested directly with
synthetic fixtures, no API key or real images needed. Only main() touches
extract_receipt(), and only when actually run as a script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_AGENT_DIR = _EVAL_DIR.parent
_CASES_PATH = _EVAL_DIR / "cases.jsonl"
_IMAGES_DIR = _EVAL_DIR / "images"
_RESULTS_PATH = _EVAL_DIR / "results.md"

# Money fields get a tolerance comparison (float rounding); everything else
# in the schema is exact-match. line_items is scored by count only -- see
# eval/README.md "Scope note" for why per-item scoring isn't implemented.
_MONEY_FIELDS = {"total", "subtotal", "tax_total"}
_MONEY_TOLERANCE = 0.01


def load_cases(cases_path: Path) -> list[dict]:
    """Parse cases.jsonl into a list of case dicts. Empty file (or file with
    only blank lines) returns an empty list -- not an error, since the
    template ships with zero real cases until a user adds them."""
    if not cases_path.exists():
        return []
    cases = []
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def compare_field(field: str, expected_value, actual_value) -> bool:
    """True if `actual_value` matches `expected_value` for `field`, using
    the right comparison for that field's type. Both None -> match (an
    expected null correctly extracted as null).
    """
    if expected_value is None and actual_value is None:
        return True
    if expected_value is None or actual_value is None:
        return False

    if field == "line_items":
        # Count-only comparison -- see eval/README.md "Scope note."
        return len(actual_value) == len(expected_value)

    if field in _MONEY_FIELDS:
        try:
            return abs(float(expected_value) - float(actual_value)) < _MONEY_TOLERANCE
        except (TypeError, ValueError):
            return False

    return expected_value == actual_value


def score_case(expected: dict, actual: dict) -> dict[str, bool]:
    """Compare only the fields present in `expected` (per eval/README.md:
    fields the labeler omitted are not scored, rather than penalized as a
    guessed None vs. non-None mismatch)."""
    return {field: compare_field(field, value, actual.get(field)) for field, value in expected.items()}


def aggregate_accuracy(per_case_results: list[dict[str, bool]]) -> dict[str, tuple[float, int]]:
    """For each field scored in ANY case, return (accuracy, n_cases_scored).
    A field only present in some cases' `expected` is averaged over just
    those cases, not penalized for cases that didn't label it."""
    totals: dict[str, list[bool]] = {}
    for case_result in per_case_results:
        for field, matched in case_result.items():
            totals.setdefault(field, []).append(matched)
    return {field: (sum(matches) / len(matches), len(matches)) for field, matches in totals.items()}


def format_results_table(accuracy: dict[str, tuple[float, int]], n_cases: int) -> str:
    """Markdown table: one row per field, sorted alphabetically for a
    deterministic, diffable output."""
    lines = [
        f"**Eval set: {n_cases} case(s).**",
        "",
        "| Field | Accuracy | Cases scored |",
        "|---|---|---|",
    ]
    for field in sorted(accuracy):
        acc, n_scored = accuracy[field]
        lines.append(f"| `{field}` | {acc:.0%} | {n_scored} |")
    return "\n".join(lines)


def _load_agent_module():
    """Import the agent's extract_receipt + media-type guesser. Deferred to
    call time (not module import time) so this file can be imported and its
    pure scoring functions tested without ever needing `instructor` or any
    provider SDK installed.
    """
    sys.path.insert(0, str(_AGENT_DIR))
    import agent as agent_module

    return agent_module


def main() -> int:
    cases = load_cases(_CASES_PATH)
    if not cases:
        print(
            f"No cases found in {_CASES_PATH}. See eval/README.md to add real "
            "receipts before running the eval.",
            file=sys.stderr,
        )
        return 1

    agent_module = _load_agent_module()

    per_case_results = []
    for case in cases:
        case_id = case.get("case_id", "<unnamed>")
        image_path = _IMAGES_DIR / case["image_path"]
        if not image_path.exists():
            print(f"warning: {case_id}: image not found at {image_path}, skipping", file=sys.stderr)
            continue

        image_bytes = image_path.read_bytes()
        media_type = agent_module._guess_media_type(image_path)
        try:
            result = agent_module.extract_receipt(image_bytes, media_type=media_type)
        except agent_module.ReceiptExtractionError as exc:
            print(f"warning: {case_id}: extraction failed ({exc.message}), skipping", file=sys.stderr)
            continue

        actual = result.model_dump(mode="json")
        per_case_results.append(score_case(case["expected"], actual))
        print(f"scored: {case_id}")

    if not per_case_results:
        print("No cases were successfully scored.", file=sys.stderr)
        return 1

    accuracy = aggregate_accuracy(per_case_results)
    table = format_results_table(accuracy, len(per_case_results))
    print("\n" + table)
    _RESULTS_PATH.write_text(table + "\n", encoding="utf-8")
    print(f"\nWrote {_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
