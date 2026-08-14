"""Tests for eval/run_eval.py's pure scoring logic. Zero dependency on
extract_receipt(), any provider SDK, or real images -- these test
load_cases/compare_field/score_case/aggregate_accuracy/format_results_table
directly with synthetic fixtures. `main()` (which DOES call the real agent)
is intentionally not covered here; it needs a real API key + real images,
which is exactly what this eval harness exists to be run against later
(see eval/README.md).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AGENT_DIR.parent))

run_eval = importlib.import_module("01_receipt_extractor.eval.run_eval")

load_cases = run_eval.load_cases
compare_field = run_eval.compare_field
score_case = run_eval.score_case
aggregate_accuracy = run_eval.aggregate_accuracy
format_results_table = run_eval.format_results_table


# ---------- load_cases ----------


def test_load_cases_empty_file_returns_empty_list(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("", encoding="utf-8")
    assert load_cases(cases_path) == []


def test_load_cases_missing_file_returns_empty_list(tmp_path):
    """The shipped cases.jsonl template has zero real cases -- must not
    crash, matches the honest "harness built, not run" F1.5 status."""
    assert load_cases(tmp_path / "does_not_exist.jsonl") == []


def test_load_cases_skips_blank_lines(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        '{"case_id": "a", "image_path": "a.jpg", "expected": {}}\n\n'
        '{"case_id": "b", "image_path": "b.jpg", "expected": {}}\n',
        encoding="utf-8",
    )
    cases = load_cases(cases_path)
    assert len(cases) == 2
    assert cases[0]["case_id"] == "a"
    assert cases[1]["case_id"] == "b"


def test_load_cases_parses_expected_fields(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        '{"case_id": "x", "image_path": "x.jpg", "expected": {"total": 12.5, "currency": "USD"}}\n',
        encoding="utf-8",
    )
    cases = load_cases(cases_path)
    assert cases[0]["expected"] == {"total": 12.5, "currency": "USD"}


# ---------- compare_field ----------


def test_compare_field_both_none_matches():
    assert compare_field("vendor_name", None, None) is True


def test_compare_field_one_none_does_not_match():
    assert compare_field("vendor_name", "Acme", None) is False
    assert compare_field("vendor_name", None, "Acme") is False


def test_compare_field_exact_string_match():
    assert compare_field("vendor_name", "Blue Bottle Coffee", "Blue Bottle Coffee") is True
    assert compare_field("vendor_name", "Blue Bottle Coffee", "blue bottle coffee") is False


def test_compare_field_money_within_tolerance_matches():
    assert compare_field("total", 12.50, 12.50) is True
    assert compare_field("total", 12.50, 12.505) is True  # within 0.01 tolerance


def test_compare_field_money_outside_tolerance_does_not_match():
    assert compare_field("total", 12.50, 12.60) is False


def test_compare_field_money_handles_int_vs_float():
    """Ground truth might be typed as int (e.g. `"total": 10`) while the
    model always returns float -- must still compare correctly."""
    assert compare_field("total", 10, 10.0) is True


def test_compare_field_line_items_counts_only():
    expected = [{"description": "coffee", "total": 5.0}, {"description": "muffin", "total": 3.0}]
    actual_matching = [{"description": "coffee", "total": 5.0}, {"description": "bagel", "total": 3.0}]
    actual_wrong_count = [{"description": "coffee", "total": 5.0}]

    assert compare_field("line_items", expected, actual_matching) is True
    assert compare_field("line_items", expected, actual_wrong_count) is False


def test_compare_field_date_exact_match():
    assert compare_field("invoice_date", "2024-01-15", "2024-01-15") is True
    assert compare_field("invoice_date", "2024-01-15", "2024-01-16") is False


# ---------- score_case ----------


def test_score_case_only_scores_fields_in_expected():
    """A field the labeler didn't include in `expected` (uncertain, per
    eval/README.md's guidance) must not be scored at all -- neither counted
    as a pass nor a fail."""
    expected = {"vendor_name": "Acme"}
    actual = {"vendor_name": "Acme", "currency": "EUR", "total": 99.99}
    result = score_case(expected, actual)
    assert result == {"vendor_name": True}
    assert "currency" not in result
    assert "total" not in result


def test_score_case_mixed_pass_fail():
    expected = {"vendor_name": "Acme", "total": 10.0, "currency": "USD"}
    actual = {"vendor_name": "Acme", "total": 15.0, "currency": "USD"}
    result = score_case(expected, actual)
    assert result == {"vendor_name": True, "total": False, "currency": True}


# ---------- aggregate_accuracy ----------


def test_aggregate_accuracy_single_field_all_cases():
    per_case = [{"vendor_name": True}, {"vendor_name": True}, {"vendor_name": False}]
    result = aggregate_accuracy(per_case)
    acc, n = result["vendor_name"]
    assert acc == pytest.approx(2 / 3)
    assert n == 3


def test_aggregate_accuracy_field_scored_in_only_some_cases():
    """A field labeled in only 2 of 3 cases must be averaged over those 2,
    not diluted by the case that didn't score it (per score_case's
    only-score-what's-labeled contract)."""
    per_case = [
        {"vendor_name": True, "due_date": True},
        {"vendor_name": True},  # due_date not labeled in this case
        {"vendor_name": False},
    ]
    result = aggregate_accuracy(per_case)
    vendor_acc, vendor_n = result["vendor_name"]
    due_date_acc, due_date_n = result["due_date"]
    assert vendor_n == 3
    assert vendor_acc == pytest.approx(2 / 3)
    assert due_date_n == 1
    assert due_date_acc == 1.0


def test_aggregate_accuracy_empty_input():
    assert aggregate_accuracy([]) == {}


# ---------- format_results_table ----------


def test_format_results_table_includes_case_count():
    table = format_results_table({"vendor_name": (1.0, 5)}, n_cases=5)
    assert "5 case(s)" in table


def test_format_results_table_sorted_alphabetically():
    accuracy = {"vendor_name": (1.0, 2), "currency": (0.5, 2), "total": (1.0, 2)}
    table = format_results_table(accuracy, n_cases=2)
    # currency < total < vendor_name alphabetically
    currency_pos = table.index("`currency`")
    total_pos = table.index("`total`")
    vendor_pos = table.index("`vendor_name`")
    assert currency_pos < total_pos < vendor_pos


def test_format_results_table_shows_percentage():
    table = format_results_table({"vendor_name": (0.75, 4)}, n_cases=4)
    assert "75%" in table
