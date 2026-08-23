"""Smoke tests for the CSV chat agent.

All tests run under LLM_PROVIDER=mock (R8 in CONTRIBUTING.md) -- CI
never touches a real API key. Real-provider tests are a manual
maintainer check before shipping.

Covers:

1. Mock path returns a valid CsvAnswer (no CSV load, no graph, no LLM).
2. CSV loading (`_load_csv_to_sqlite`):
   - happy path (real tmp CSV -> connection + schema_ddl)
   - file not found
   - empty file
   - headers-only file
   - too-many-columns cap
   - malformed CSV (parser error)
3. State graph via `chat_with_csv`:
   - happy path (SequenceLLM: 1 SQL + 1 answer = 2 calls, 1 attempt)
   - retry after SQL error (SequenceLLM: bad SQL -> good SQL -> answer
     = 3 calls, 2 attempts, .attempts_history has both)
   - retry exhaustion (3 consecutive bad SQLs -> ChatCsvError with
     .partial=ChatCsvAttempt(attempts=[3 failed])
4. R5 error branch: `_translate_api_error` maps 429/401/class-name.
5. Pure helpers:
   - `_extract_sql` (plain, ```sql-fenced```, bare fence, prose)
   - `_format_schema_ddl` (produces CREATE TABLE + rows preview)
   - `resolve_provider` (default, env, rejects unknown)

SequenceLLM fixture supplies scripted responses on successive
`.complete()` calls -- same shape as agent #02's, just reused because
the state graph makes multiple LLM calls per turn (SQL then answer)
which sequenced responses handle naturally.
"""

from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AGENT_DIR.parent))

_agent = importlib.import_module("03_csv_chat.agent")
_schemas = importlib.import_module("03_csv_chat.schemas")

chat_with_csv = _agent.chat_with_csv
ChatCsvError = _agent.ChatCsvError
ChatCsvAttempt = _agent.ChatCsvAttempt
resolve_provider = _agent.resolve_provider
_load_csv_to_sqlite = _agent._load_csv_to_sqlite
_format_schema_ddl = _agent._format_schema_ddl
_extract_sql = _agent._extract_sql
_translate_api_error = _agent._translate_api_error
MAX_COLUMNS_FOR_SCHEMA_PREVIEW = _agent.MAX_COLUMNS_FOR_SCHEMA_PREVIEW
CsvAnswer = _schemas.CsvAnswer
SqlAttempt = _schemas.SqlAttempt


# --- SequenceLLM fixture ---------------------------------------------------


class SequenceLLM:
    """Test-only LLM Protocol impl. Returns responses from a pre-set
    list on successive .complete() calls, records every call so tests
    can assert call count / prompt content."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cacheable_prefix: str | None = None,
    ):
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        if not self._responses:
            raise RuntimeError("SequenceLLM ran out of scripted responses")
        text = self._responses.pop(0)

        class _Resp:
            pass

        r = _Resp()
        r.text = text
        r.input_tokens = 0
        r.output_tokens = 0
        r.cached_input_tokens = 0
        r.cache_creation_input_tokens = 0
        r.latency_ms = 0.0
        return r


# --- Fixtures --------------------------------------------------------------


@pytest.fixture
def sample_csv(tmp_path):
    """A tiny real CSV: 3 penguins, 3 columns. Enough for the graph
    to introspect a schema and run a real SELECT against."""
    csv_path = tmp_path / "penguins.csv"
    csv_path.write_text(
        "species,island,body_mass_g\n"
        "Adelie,Torgersen,3750\n"
        "Adelie,Torgersen,3800\n"
        "Gentoo,Biscoe,5000\n",
        encoding="utf-8",
    )
    return csv_path


# --- 1. Mock path ----------------------------------------------------------


def test_mock_path_returns_valid_csv_answer(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    answer = chat_with_csv("fake/path.csv", "How many rows?")
    assert isinstance(answer, CsvAnswer)
    assert "path.csv" in answer.answer
    assert len(answer.attempts) >= 1


def test_mock_path_does_not_touch_csv_file():
    """Mock mode short-circuits BEFORE _load_csv_to_sqlite -- passing
    a non-existent path should not raise."""
    answer = chat_with_csv("/definitely/does/not/exist.csv", "q", provider="mock")
    assert isinstance(answer, CsvAnswer)


def test_mock_path_serializable_to_json():
    answer = chat_with_csv("x.csv", "q", provider="mock")
    dumped = answer.model_dump_json()
    restored = CsvAnswer.model_validate_json(dumped)
    assert restored == answer


def test_provider_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    answer = chat_with_csv("x.csv", "q", provider="mock")
    assert isinstance(answer, CsvAnswer)


# --- 2. CSV loading --------------------------------------------------------


def test_load_csv_happy_path(sample_csv):
    conn, schema_ddl = _load_csv_to_sqlite(sample_csv)
    assert isinstance(conn, sqlite3.Connection)
    assert "CREATE TABLE" in schema_ddl
    assert "species" in schema_ddl
    assert "First 3 rows:" in schema_ddl
    # Verify the table actually got populated.
    rows = conn.execute("SELECT COUNT(*) FROM data").fetchone()
    assert rows[0] == 3


def test_load_csv_file_not_found(tmp_path):
    with pytest.raises(ChatCsvError, match="not found"):
        _load_csv_to_sqlite(tmp_path / "nope.csv")


def test_load_csv_empty_file(tmp_path):
    """Truly empty file (no bytes) triggers pandas EmptyDataError."""
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ChatCsvError, match="empty"):
        _load_csv_to_sqlite(empty)


def test_load_csv_headers_only(tmp_path):
    """CSV with header row but no data rows -- pandas parses but df
    is empty; our own check catches it with an actionable message."""
    hdr = tmp_path / "hdr.csv"
    hdr.write_text("a,b,c\n", encoding="utf-8")
    with pytest.raises(ChatCsvError, match="no data rows"):
        _load_csv_to_sqlite(hdr)


def test_load_csv_too_many_columns(tmp_path):
    """CSV wider than MAX_COLUMNS_FOR_SCHEMA_PREVIEW gets a friendly
    'this agent handles up to N columns' error, not a mysterious
    prompt-too-long failure downstream."""
    too_wide = tmp_path / "wide.csv"
    cols = ",".join(f"c{i}" for i in range(MAX_COLUMNS_FOR_SCHEMA_PREVIEW + 5))
    vals = ",".join(["x"] * (MAX_COLUMNS_FOR_SCHEMA_PREVIEW + 5))
    too_wide.write_text(f"{cols}\n{vals}\n", encoding="utf-8")
    with pytest.raises(ChatCsvError, match=f"handles up to {MAX_COLUMNS_FOR_SCHEMA_PREVIEW}"):
        _load_csv_to_sqlite(too_wide)


def test_format_schema_ddl_includes_create_table_and_rows(sample_csv):
    """Guard against a future refactor that drops one half of the
    schema-preview string -- the prompt depends on both halves."""
    _conn, ddl = _load_csv_to_sqlite(sample_csv)
    assert "CREATE TABLE" in ddl
    assert "data" in ddl.lower()
    assert "Adelie" in ddl  # a real sample-row value
    assert "3750" in ddl


# --- 3. State graph via chat_with_csv --------------------------------------


def test_graph_happy_path_single_attempt(sample_csv):
    """First SQL is valid; graph does 1 SQL call + 1 answer call = 2
    total, 1 attempt in history, answer is the model's text verbatim."""
    llm = SequenceLLM([
        "SELECT species, COUNT(*) as cnt FROM data GROUP BY species",
        "There are 2 Adelie and 1 Gentoo penguin.",
    ])
    answer = chat_with_csv(
        sample_csv,
        "How many penguins per species?",
        provider="openai",
        model="test-model",
        _llm=llm,
    )
    assert isinstance(answer, CsvAnswer)
    assert answer.answer == "There are 2 Adelie and 1 Gentoo penguin."
    assert len(llm.calls) == 2  # 1 SQL + 1 answer
    assert len(answer.attempts) == 1
    assert answer.attempts[0].error is None
    assert answer.row_count == 2  # 2 species groups
    # result_sample should contain both group rows
    assert len(answer.result_sample) == 2


def test_graph_retries_on_sql_error_then_succeeds(sample_csv):
    """Bad SQL (typo in column name) -> execute fails -> retry edge ->
    good SQL -> answer. 3 LLM calls total (2 SQL + 1 answer), 2 attempts
    in history with the first flagged as errored."""
    llm = SequenceLLM([
        # Attempt 1: wrong column name (species is real; specis is a typo)
        "SELECT COUNT(*) FROM data WHERE specis = 'Adelie'",
        # Attempt 2: correct column
        "SELECT COUNT(*) as cnt FROM data WHERE species = 'Adelie'",
        # Answer node
        "There are 2 Adelie penguins.",
    ])
    answer = chat_with_csv(
        sample_csv,
        "How many Adelies?",
        provider="openai",
        model="test-model",
        _llm=llm,
    )
    assert isinstance(answer, CsvAnswer)
    assert len(llm.calls) == 3
    assert len(answer.attempts) == 2
    assert answer.attempts[0].error is not None
    assert "specis" in answer.attempts[0].error or "no such column" in answer.attempts[0].error.lower()
    assert answer.attempts[1].error is None
    # Retry-feedback prompt for attempt 2 should include the failed SQL + error.
    retry_prompt = llm.calls[1]["prompt"]
    assert "FAILED" in retry_prompt or "failed" in retry_prompt.lower()
    assert "specis" in retry_prompt  # includes the bad SQL


def test_graph_retry_exhaustion_raises_with_partial(sample_csv):
    """Three consecutive bad SQLs at max_attempts=3 -> graph reaches
    END with answer=None -> chat_with_csv raises ChatCsvError with
    .partial=ChatCsvAttempt(attempts=[3 failed])."""
    llm = SequenceLLM([
        "SELECT * FROM does_not_exist",
        "SELECT * FROM also_bad",
        "SELECT * FROM still_bad",
    ])
    with pytest.raises(ChatCsvError) as exc_info:
        chat_with_csv(
            sample_csv,
            "q",
            provider="openai",
            model="test-model",
            max_attempts=3,
            _llm=llm,
        )
    assert len(llm.calls) == 3  # no answer call happened
    assert "3 attempts" in exc_info.value.message
    assert exc_info.value.partial is not None
    assert isinstance(exc_info.value.partial, ChatCsvAttempt)
    assert len(exc_info.value.partial.attempts) == 3
    assert all(a.error is not None for a in exc_info.value.partial.attempts)


def test_graph_answer_node_gets_result_rows_in_prompt(sample_csv):
    """The format_answer node MUST see actual result rows -- if it
    doesn't, its answer can't cite real numbers. Regression guard
    against a refactor that accidentally passes an empty rows_preview."""
    llm = SequenceLLM([
        "SELECT species, COUNT(*) as cnt FROM data GROUP BY species",
        "There are 2 Adelie and 1 Gentoo.",
    ])
    chat_with_csv(sample_csv, "counts?", provider="openai", model="m", _llm=llm)
    answer_prompt = llm.calls[1]["prompt"]
    # Actual result data appears in the prompt (JSON-serialized rows).
    assert "Adelie" in answer_prompt
    assert "Gentoo" in answer_prompt


def test_graph_handles_markdown_fenced_sql(sample_csv):
    """Real models often wrap SQL in ```sql ... ``` fences. _extract_sql
    strips them; the graph should accept fenced output on first try
    (no retry needed)."""
    llm = SequenceLLM([
        "```sql\nSELECT COUNT(*) FROM data\n```",
        "There are 3 rows.",
    ])
    answer = chat_with_csv(sample_csv, "how many?", provider="openai", model="m", _llm=llm)
    assert len(llm.calls) == 2  # single-try success
    assert answer.attempts[0].error is None


def test_graph_zero_row_result_still_succeeds(sample_csv):
    """An empty result set (`WHERE species = 'DoesNotExist'`) is a
    VALID outcome, not an error -- CsvAnswer should carry row_count=0
    and the answer should still be produced."""
    llm = SequenceLLM([
        "SELECT * FROM data WHERE species = 'DoesNotExist'",
        "No penguins match that filter.",
    ])
    answer = chat_with_csv(sample_csv, "any", provider="openai", model="m", _llm=llm)
    assert answer.row_count == 0
    assert answer.result_sample == []
    assert answer.attempts[0].error is None


# --- 4. R5 error branch: _translate_api_error ------------------------------


def test_translate_api_error_rate_limit_by_status():
    class E(Exception):
        status_code = 429
    result = _translate_api_error(E("body"))
    assert isinstance(result, ChatCsvError)
    assert "rate-limited" in result.message.lower()


def test_translate_api_error_rate_limit_by_class_name():
    class RateLimitError(Exception):
        pass
    result = _translate_api_error(RateLimitError("no status"))
    assert "rate-limited" in result.message.lower()


def test_translate_api_error_auth_by_status():
    class E(Exception):
        status_code = 401
    result = _translate_api_error(E(""))
    assert "authentication" in result.message.lower()


def test_translate_api_error_auth_by_class_name():
    class AuthenticationError(Exception):
        pass
    result = _translate_api_error(AuthenticationError("bad key"))
    assert "authentication" in result.message.lower()


def test_translate_api_error_class_check_priority():
    """Class-name check fires BEFORE message-string fallback: a
    RateLimitError with no matching text still gets classified as
    rate limit."""
    class RateLimitError(Exception):
        pass
    result = _translate_api_error(RateLimitError("some unrelated body text"))
    assert "rate-limited" in result.message.lower()


def test_translate_api_error_unknown_preserves_original():
    result = _translate_api_error(ValueError("something weird happened"))
    assert "ValueError" in result.message
    assert "something weird happened" in result.message


# --- 5. Pure helpers -------------------------------------------------------


def test_extract_sql_plain():
    assert _extract_sql("SELECT * FROM data") == "SELECT * FROM data"


def test_extract_sql_strips_sql_fence():
    assert _extract_sql("```sql\nSELECT * FROM data\n```") == "SELECT * FROM data"


def test_extract_sql_strips_bare_fence():
    assert _extract_sql("```\nSELECT * FROM data\n```") == "SELECT * FROM data"


def test_extract_sql_strips_missing_closing_fence():
    """Model sometimes forgets to close the fence -- still extract."""
    assert _extract_sql("```sql\nSELECT * FROM data") == "SELECT * FROM data"


def test_extract_sql_preserves_multiline():
    sql = "SELECT species,\n  COUNT(*) as cnt\nFROM data\nGROUP BY species"
    assert _extract_sql(f"```sql\n{sql}\n```") == sql


def test_resolve_provider_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider() == "openai"


def test_resolve_provider_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert resolve_provider() == "anthropic"


def test_resolve_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


# --- 6. Review-fix regression guards (H1, M1, M3) --------------------------


def _spy_load_csv(real_load):
    """Wrap `_load_csv_to_sqlite` so tests can assert whether the
    returned connection was closed after chat_with_csv finishes."""
    captured = {}

    def wrapped(path):
        conn, schema = real_load(path)
        captured["conn"] = conn
        return conn, schema

    return wrapped, captured


def test_h1_conn_closed_on_success(sample_csv, monkeypatch):
    """SQLite connection MUST be closed after a successful chat_with_csv.
    Regression guard against a future refactor that removes the
    try/finally in chat_with_csv."""
    wrapped, captured = _spy_load_csv(_load_csv_to_sqlite)
    monkeypatch.setattr(_agent, "_load_csv_to_sqlite", wrapped)
    llm = SequenceLLM([
        "SELECT COUNT(*) FROM data",
        "There are 3 rows.",
    ])
    chat_with_csv(sample_csv, "count?", provider="openai", model="m", _llm=llm)
    # Post-close, any DB operation raises ProgrammingError.
    with pytest.raises(sqlite3.ProgrammingError):
        captured["conn"].execute("SELECT 1")


def test_h1_conn_closed_on_retry_exhaustion(sample_csv, monkeypatch):
    """Same guard, but through the retry-exhaustion path -- the
    try/finally has to fire when the retry cap is hit, not just on
    success."""
    wrapped, captured = _spy_load_csv(_load_csv_to_sqlite)
    monkeypatch.setattr(_agent, "_load_csv_to_sqlite", wrapped)
    llm = SequenceLLM([
        "SELECT * FROM nope",
        "SELECT * FROM nope2",
        "SELECT * FROM nope3",
    ])
    with pytest.raises(ChatCsvError):
        chat_with_csv(
            sample_csv, "q", provider="openai", model="m", max_attempts=3, _llm=llm
        )
    with pytest.raises(sqlite3.ProgrammingError):
        captured["conn"].execute("SELECT 1")


def test_h1_conn_closed_on_llm_exception(sample_csv, monkeypatch):
    """And through the R5 case 3 path -- an LLM SDK exception during
    the graph should still let the finally block close the conn."""

    class ExplodingLLM:
        def complete(self, **_kwargs):
            raise RuntimeError("simulated network failure")

    wrapped, captured = _spy_load_csv(_load_csv_to_sqlite)
    monkeypatch.setattr(_agent, "_load_csv_to_sqlite", wrapped)
    with pytest.raises(ChatCsvError):
        chat_with_csv(
            sample_csv, "q", provider="openai", model="m", _llm=ExplodingLLM()
        )
    with pytest.raises(sqlite3.ProgrammingError):
        captured["conn"].execute("SELECT 1")


def test_m1_prompt_size_check_raises_on_oversized_schema(sample_csv, monkeypatch):
    """A schema_ddl above the token ceiling should raise ChatCsvError
    with the model named, BEFORE any LLM call happens. Simulate by
    forcing the ceiling low so a normal-sized schema trips it -- easier
    than fabricating a 400KB CSV."""
    monkeypatch.setattr(_agent, "MAX_SCHEMA_TOKENS_ESTIMATE", 5)
    # llm should NEVER be called -- the size check raises first.
    llm = SequenceLLM([])  # empty; would RuntimeError on any .complete() call
    with pytest.raises(ChatCsvError) as exc_info:
        chat_with_csv(
            sample_csv,
            "count?",
            provider="openai",
            model="test-model-xyz",
            _llm=llm,
        )
    assert "too large" in exc_info.value.message.lower()
    assert "test-model-xyz" in exc_info.value.message
    assert len(llm.calls) == 0  # short-circuited before any LLM call


def test_m1_prompt_size_check_passes_normal_schema(sample_csv):
    """The default ceiling is 100k tokens; a 3-row penguins CSV
    produces a schema_ddl of well under 1k tokens. Regression guard
    against a future refactor that would set the ceiling too low
    and reject legitimate small CSVs."""
    _check_prompt_size = _agent._check_prompt_size
    _conn, ddl = _load_csv_to_sqlite(sample_csv)
    _check_prompt_size(ddl, "gpt-4.1-mini-2025-04-14")  # should not raise


class _AlwaysBadExecuteConn:
    """Fake connection whose .execute() always raises TypeError. Used
    to test the M3 broad-exception path -- sqlite3.Connection.execute
    is read-only in Python 3.14 so we can't monkeypatch it in place;
    instead swap the whole connection object out via a fake loader.

    Implements the minimal surface the graph touches: .execute() and
    .close()."""

    def __init__(self):
        self.closed = False

    def execute(self, *_a, **_kw):
        raise TypeError("simulated non-sqlite runtime failure")

    def close(self):
        self.closed = True


def test_m3_non_sqlite_exception_in_execute_treated_as_attempt(sample_csv, monkeypatch):
    """If _execute_sql raises a non-sqlite3 exception (TypeError,
    MemoryError, etc.), the graph should treat it as an ATTEMPT FAILURE
    (retry-worthy) not let it escape and get misclassified as an API
    error by _translate_api_error.

    Simulate via a fake connection with a always-raising .execute --
    the whole thing runs through the graph's retry loop and lands at
    retry exhaustion with ChatCsvError.partial populated, NOT a
    generic 'Chat failed: TypeError' from the outer translator."""
    real_load = _load_csv_to_sqlite

    def loader(path):
        _real_conn, schema = real_load(path)
        _real_conn.close()  # don't leak the real conn we ignored
        return _AlwaysBadExecuteConn(), schema

    monkeypatch.setattr(_agent, "_load_csv_to_sqlite", loader)
    llm = SequenceLLM(["SELECT 1", "SELECT 2", "SELECT 3"])
    with pytest.raises(ChatCsvError) as exc_info:
        chat_with_csv(
            sample_csv, "q", provider="openai", model="m", max_attempts=3, _llm=llm
        )
    # Should end at retry exhaustion (ChatCsvError with .partial),
    # NOT get translated as a generic API error.
    assert exc_info.value.partial is not None
    assert len(exc_info.value.partial.attempts) == 3
    # Every attempt's error should record the TypeError, not silently
    # convert to "chat failed" or classify as rate-limit/auth.
    for attempt in exc_info.value.partial.attempts:
        assert "TypeError" in (attempt.error or "")
