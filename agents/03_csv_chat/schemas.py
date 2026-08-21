"""Pydantic models for the CSV chat agent's structured output.

Two models: `SqlAttempt` (one SQL attempt + whether it succeeded) and
`CsvAnswer` (the full turn's output: the model's answer, the successful
SQL, the row-count, the sample rows shown, and the history of attempts
for UI display). `CsvAnswer.attempts` is a list because the retry loop
may have gone through several bad SQLs before landing a good one, and
the UI showing "we tried X, got error Y, then tried Z, succeeded" is
part of the user's trust model for a text-to-SQL tool.

Design notes:

- `attempts` is always non-empty for a successful `CsvAnswer` -- at
  minimum the winning attempt appears there. Enforced by `min_length=1`
  so a construction bug (empty history) surfaces at validation time,
  not as a mysteriously-missing "the model tried X" line in the UI.
- `result_sample` is a plain `list[dict]` (not a DataFrame) because the
  final output must be JSON-serializable for the UI + downstream tools.
  Capped at 10 rows in `agent.py`'s state graph -- the full row_count
  is still reported, users click "download CSV" for the full result.
- `sql_used` mirrors `attempts[-1].sql` for convenience -- the UI
  wants the winning SQL directly without walking the history list.
- No `error` field on `CsvAnswer`: on retry exhaustion, `agent.py`
  raises `ChatCsvError` with the failed `.partial` attached, rather
  than returning a "failed" CsvAnswer. Kept the success/failure split
  clean at the exception layer, not in the schema.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SqlAttempt(BaseModel):
    """One SQL attempt made by the state graph.

    `error` is None iff the SQL executed successfully. `row_count` is
    populated on success (may be 0 -- an empty result set is a valid
    outcome, not an error). Attempts with an error still get a `sql`
    field populated so the UI can show WHICH SQL failed, not just that
    something failed.
    """

    sql: str = Field(..., min_length=1)
    error: str | None = Field(
        default=None,
        description="None if this attempt succeeded. Otherwise the SQLite error message, verbatim, so the retry-feedback prompt can quote it directly.",
    )
    row_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of rows the successful SQL returned. None if this attempt errored (no result set to count).",
    )


class CsvAnswer(BaseModel):
    """The full output of one chat_with_csv() call.

    On success, includes the plain-English `answer`, the winning
    `sql_used`, the total `row_count`, up to 10 sample rows for
    display, and the full `attempts` history (including any failed
    attempts before the winning one)."""

    question: str = Field(..., min_length=1)
    answer: str = Field(
        ...,
        min_length=1,
        description="Plain-English answer to the user's question, generated from the SQL result rows. NOT the SQL itself and NOT a rewrite of the question -- an actual answer.",
    )
    sql_used: str = Field(
        ...,
        min_length=1,
        description="The winning SQL that produced the result set. Mirrors attempts[-1].sql for convenience.",
    )
    row_count: int = Field(
        ...,
        ge=0,
        description="Total row count from the winning SQL (may be 0 -- empty result is a valid outcome).",
    )
    attempts: list[SqlAttempt] = Field(
        ...,
        min_length=1,
        description="History of every SQL attempt this turn made, in order. Last one is always the winner (if a CsvAnswer was returned successfully). Multi-attempt histories are shown in the UI to explain retries to the user.",
    )
    result_sample: list[dict] = Field(
        default_factory=list,
        description="Up to 10 sample rows from the result set, as JSON-serializable dicts. Empty if row_count is 0 or the result set has no columns. Full result is NOT returned here to keep CsvAnswer payload small; the CLI/UI can re-run the SQL for a full download.",
    )
