"""Schema unit tests for the CSV chat agent.

F3.1 scope: schemas only. Behavioral tests for agent.py's state graph,
R5 error branches, and the retry loop land in test_smoke.py at F3.4.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# `03_csv_chat` starts with a digit -- invalid Python identifier for
# normal import syntax. Same importlib bootstrap as agents #01 / #02.
_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AGENT_DIR.parent))

_schemas = importlib.import_module("03_csv_chat.schemas")
SqlAttempt = _schemas.SqlAttempt
CsvAnswer = _schemas.CsvAnswer


# --- SqlAttempt ------------------------------------------------------------


def test_sql_attempt_successful_construction():
    attempt = SqlAttempt(sql="SELECT COUNT(*) FROM t", row_count=42)
    assert attempt.sql == "SELECT COUNT(*) FROM t"
    assert attempt.error is None
    assert attempt.row_count == 42


def test_sql_attempt_failed_construction():
    attempt = SqlAttempt(sql="SELCT * FROM t", error="near \"SELCT\": syntax error")
    assert attempt.error == "near \"SELCT\": syntax error"
    assert attempt.row_count is None


def test_sql_attempt_rejects_empty_sql():
    """Even an errored attempt needs a SQL string -- the UI shows WHICH
    SQL failed, not just 'something failed.'"""
    with pytest.raises(ValidationError):
        SqlAttempt(sql="", error="anything")


def test_sql_attempt_rejects_negative_row_count():
    with pytest.raises(ValidationError):
        SqlAttempt(sql="SELECT 1", row_count=-1)


def test_sql_attempt_zero_row_count_is_valid():
    """An empty result set is a valid successful outcome (e.g. 'how
    many rows match X?' where the answer is genuinely 0). Regression
    guard against a future field-level constraint that would treat 0
    as an error."""
    attempt = SqlAttempt(sql="SELECT * FROM t WHERE 1=0", row_count=0)
    assert attempt.error is None
    assert attempt.row_count == 0


# --- CsvAnswer -------------------------------------------------------------


def _make_valid_answer(**overrides) -> CsvAnswer:
    defaults = {
        "question": "How many penguins?",
        "answer": "There are 344 penguins across 3 species.",
        "sql_used": "SELECT COUNT(*) FROM penguins",
        "row_count": 1,
        "attempts": [SqlAttempt(sql="SELECT COUNT(*) FROM penguins", row_count=1)],
        "result_sample": [{"count": 344}],
    }
    defaults.update(overrides)
    return CsvAnswer(**defaults)


def test_answer_minimal_valid_construction():
    answer = _make_valid_answer()
    assert answer.question == "How many penguins?"
    assert answer.row_count == 1
    assert len(answer.attempts) == 1


def test_answer_rejects_empty_question():
    with pytest.raises(ValidationError):
        _make_valid_answer(question="")


def test_answer_rejects_empty_answer():
    """The whole point of the agent is producing an answer -- an empty
    string is never a valid response, even if the SQL succeeded."""
    with pytest.raises(ValidationError):
        _make_valid_answer(answer="")


def test_answer_rejects_empty_sql_used():
    with pytest.raises(ValidationError):
        _make_valid_answer(sql_used="")


def test_answer_rejects_empty_attempts_history():
    """A successful CsvAnswer must record at least its winning attempt
    -- an empty history means the state graph forgot to log something,
    surface it as a schema failure not a silent UI blank."""
    with pytest.raises(ValidationError):
        _make_valid_answer(attempts=[])


def test_answer_rejects_negative_row_count():
    with pytest.raises(ValidationError):
        _make_valid_answer(row_count=-1)


def test_answer_zero_row_count_is_valid():
    """Same reasoning as SqlAttempt: empty result set is legitimate."""
    answer = _make_valid_answer(
        question="How many penguins weigh over 10kg?",
        answer="None -- the heaviest penguin in the dataset is 6.3kg.",
        row_count=0,
        result_sample=[],
    )
    assert answer.row_count == 0
    assert answer.result_sample == []


def test_answer_records_multi_attempt_history():
    """A retry-then-success run should carry both attempts in
    .attempts, with the winner last. UI walks this list to show 'we
    tried X, got error Y, then tried Z, succeeded.'"""
    answer = _make_valid_answer(
        attempts=[
            SqlAttempt(sql="SELECT * FROM penguin", error="no such table: penguin"),
            SqlAttempt(sql="SELECT * FROM penguins", row_count=344),
        ]
    )
    assert len(answer.attempts) == 2
    assert answer.attempts[0].error is not None
    assert answer.attempts[-1].error is None


def test_answer_serializable_to_and_from_json():
    """Round-trips cleanly -- the mock path, the graph's final step,
    and the UI/CLI all depend on this."""
    original = _make_valid_answer(
        attempts=[
            SqlAttempt(sql="SELCT 1", error="near \"SELCT\": syntax error"),
            SqlAttempt(sql="SELECT 1", row_count=1),
        ],
        result_sample=[{"1": 1}],
    )
    dumped = original.model_dump_json()
    restored = CsvAnswer.model_validate_json(dumped)
    assert restored == original


def test_answer_result_sample_defaults_to_empty_list():
    """Convenience default -- constructing an answer for a scalar
    result (e.g. COUNT(*)) shouldn't force an explicit empty list."""
    answer = CsvAnswer(
        question="q",
        answer="a",
        sql_used="SELECT 1",
        row_count=1,
        attempts=[SqlAttempt(sql="SELECT 1", row_count=1)],
    )
    assert answer.result_sample == []
