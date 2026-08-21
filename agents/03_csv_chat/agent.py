"""CSV chat agent -- agent #03 of real-world-agents.

Technique demonstrated: **text-to-SQL with a LangGraph state machine +
retry-on-error loop**. LangGraph is the right shape here because the
workflow has real branching (SQL succeeded -> format answer; SQL failed
-> retry with error context), not the linear pipeline of agents #01/#02.
The state graph makes those branches explicit:

    START -> generate_sql -> execute_sql -> ?
                                            |-- success --> format_answer -> END
                                            |-- error + attempts left --> generate_sql (retry)
                                            |-- error + retries exhausted --> END

`_build_graph` below is the pedagogical anchor -- read that one function
to see how LangGraph turns a workflow-with-branching into declarative
node + edge definitions. The retry loop isn't a `for` loop hidden in
`_run_review_loop` (agent #02's shape); it's an EDGE, visible as data
in the graph structure.

Why this technique for this use case: text-to-SQL naturally has two LLM
calls (generate SQL, then interpret result rows) with an error-driven
retry between them. Trying to squeeze that into one prompt loses the
retry-on-execution-error signal that makes the whole system work --
without it, the model has no chance to correct "no such column
'penquin_weight'" (typo the model itself might have made) into
"penguin_weight".

Real error handling (R5 in CONTRIBUTING.md's hard rules): three concrete
failure modes handled explicitly (see chat_with_csv below):
  1. CSV won't load (pandas.read_csv raises, or file is empty) ->
     ChatCsvError with a specific "check headers/delimiter" message
  2. SQL retries exhausted (model can't produce a working query after
     max_attempts) -> ChatCsvError with the last SQL + last error
     attached as .partial so UI can show the failed attempts
  3. Rate limit / auth / API failure -> translated to ChatCsvError with
     a clear message. This agent does NOT auto-retry transient API
     errors (same S2-option-b decision as agent #02) -- the user pays
     per real API call, and is better-placed to decide whether to wait
     and re-run than to have the graph silently spend more of their
     budget on a failing endpoint.

Two-node LLM design (generate_sql + format_answer, not a single-shot
"answer the question directly from CSV"): trades doubled cost/latency
for (a) letting the state graph retry ONLY the SQL step on execution
error (cheaper than re-generating the whole answer), (b) letting the
format_answer node see real result rows as context (much better answer
quality). Cost is honestly documented in the README.

Provider + model are fully user-configurable (SPEC R6): every real LLM
call goes through common.llm.get_llm() / resolve_model(). Same env-var
contract as agents #01/#02.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

# Dual-mode import (same rationale as agents #01/#02): `.schemas`
# resolves when this file is loaded as a submodule of 03_csv_chat by
# tests (via importlib since the dir name starts with a digit); bare
# `schemas` resolves when run directly via `python -m agent` from
# inside the agent's own directory.
try:
    from .schemas import CsvAnswer, SqlAttempt
except ImportError:
    from schemas import CsvAnswer, SqlAttempt

from common.llm import LLM, get_llm, resolve_model

# --- Provider + constants --------------------------------------------------

SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini")

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_TOKENS_SQL = 1024
DEFAULT_MAX_TOKENS_ANSWER = 512
DEFAULT_TABLE_NAME = "data"
DEFAULT_RESULT_SAMPLE_SIZE = 10  # how many rows CsvAnswer.result_sample carries

# Cap the pandas dtype-preview + first-3-rows preview we inject into the
# generate_sql prompt. A CSV with hundreds of columns would blow the
# prompt budget otherwise; 40 columns covers realistic small-business
# CSVs comfortably and gets a friendly error above that.
MAX_COLUMNS_FOR_SCHEMA_PREVIEW = 40

# Prompt-size guard (review fix M1, F3.5 review). Mirrors agent #02's
# _check_context_window: chars-per-token heuristic, no SDK dep, ceiling
# well under gpt-4.1-mini's 1M cap so there's headroom for the prompt
# template + generated SQL + retry-feedback text + output tokens.
# Above this, a model with a smaller context window would reject the
# request; better to fail-fast with an actionable message than surface
# a raw BadRequestError from the provider.
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_SCHEMA_TOKENS_ESTIMATE = 100_000  # schema alone; leaves ~900k for everything else

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def resolve_provider() -> str:
    """LLM_PROVIDER env var, defaulting to "openai" (SPEC R6: no
    provider is hardcoded)."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}."
        )
    return provider


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


# --- Error type ------------------------------------------------------------


@dataclass
class ChatCsvAttempt:
    """What we got back when retries were exhausted. Attached to
    ChatCsvError so the UI can surface the failed attempts in a warning
    banner rather than dropping them silently."""

    attempts: list[SqlAttempt] = field(default_factory=list)


class ChatCsvError(Exception):
    """Raised on any user-facing chat failure (bad CSV, retry
    exhaustion, API failure). `message` is user-friendly; `partial`
    carries the failed SQL attempts when relevant."""

    def __init__(self, message: str, partial: ChatCsvAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Graph state -----------------------------------------------------------


class GraphState(TypedDict, total=False):
    """LangGraph state carried between nodes. `total=False` so nodes
    can return partial state dicts and LangGraph's default reducer
    merges them; we never delete keys, only add or overwrite.

    The `attempts_history` list is appended-to by _execute_sql on
    every attempt (successful or failed). Node returns the full
    updated list rather than a diff -- simpler than wiring an
    Annotated[list, add] reducer for a low-frequency append."""

    question: str
    schema_ddl: str
    sql: str
    result_rows: list[dict] | None
    error: str | None
    answer: str | None
    attempts_history: list[SqlAttempt]
    attempt: int
    max_attempts: int


# --- Public API ------------------------------------------------------------


def chat_with_csv(
    csv_path: str | Path,
    question: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    _llm: LLM | None = None,  # test-injection escape hatch
) -> CsvAnswer:
    """Answer `question` about the CSV at `csv_path` in plain English.

    Args:
        csv_path: path to a CSV file.
        question: natural-language question about the data.
        provider: "openai" (default) / "anthropic" / "gemini" / "mock".
            Defaults to the LLM_PROVIDER env var.
        model: model ID; defaults to the resolved provider's
            DEFAULT_MODELS entry (override with the per-provider env
            var, e.g. OPENAI_DEFAULT_MODEL).
        max_attempts: how many times the graph's retry edge fires on
            SQL execution errors before giving up. Default 3.
        _llm: injected LLM for tests; production callers leave this
            None.

    Returns:
        A validated CsvAnswer with the plain-English answer, the
        winning SQL, the row count, up to 10 sample rows, and the
        history of every attempt (including any that failed).

    Raises:
        ChatCsvError: on any of the three R5 failure modes.
    """
    resolved_provider = (provider or resolve_provider()).lower()

    if resolved_provider == "mock":
        return _mock_answer(csv_path, question)

    resolved_model = model or resolve_model(resolved_provider)

    # R5 case 1: CSV won't load. Raised inline so the message is
    # specific to the load failure (pandas exception type) rather than
    # a generic "something failed."
    conn, schema_ddl = _load_csv_to_sqlite(csv_path)

    # Review fix H1 (F3.5 review): every path from here on MUST close
    # the SQLite connection, or it leaks per-call into process memory.
    # In-memory DB so the leak is bounded, but a long-running UI
    # session or repeated CLI calls would accumulate connections. The
    # try/finally guarantees close on success, on retry-exhaustion, and
    # on any exception from within the graph.
    try:
        # Review fix M1 (F3.5 review): prompt-size guard. #02 has an
        # analogous _check_context_window; #03 needed one too because
        # wide CSVs with long cell values can produce very large
        # schema_ddl previews that blow smaller models' context.
        _check_prompt_size(schema_ddl, resolved_model)

        llm = _llm if _llm is not None else get_llm(resolved_provider)
        graph = _build_graph(llm=llm, conn=conn, model=resolved_model)

        initial_state: GraphState = {
            "question": question,
            "schema_ddl": schema_ddl,
            "sql": "",
            "result_rows": None,
            "error": None,
            "answer": None,
            "attempts_history": [],
            "attempt": 0,
            "max_attempts": max_attempts,
        }

        try:
            final_state = graph.invoke(initial_state)
        except ChatCsvError:
            raise
        except Exception as exc:  # R5 case 3: rate limit / API failure
            raise _translate_api_error(exc) from exc

        # R5 case 2: retries exhausted. The graph ran to END with error
        # populated and answer None -- surface as ChatCsvError with the
        # attempts history attached so the UI can show what was tried.
        if final_state.get("answer") is None:
            history = final_state.get("attempts_history", [])
            last_error = final_state.get("error") or "unknown error"
            raise ChatCsvError(
                f"Model couldn't produce a working SQL query after "
                f"{max_attempts} attempts. Last error from SQLite: {last_error}",
                partial=ChatCsvAttempt(attempts=history),
            )

        winning_rows: list[dict] = final_state["result_rows"] or []
        return CsvAnswer(
            question=question,
            answer=final_state["answer"],
            sql_used=final_state["sql"],
            row_count=len(winning_rows),
            attempts=final_state["attempts_history"],
            result_sample=winning_rows[:DEFAULT_RESULT_SAMPLE_SIZE],
        )
    finally:
        conn.close()


# --- CSV loading (outside the graph; deterministic + LLM-free) -------------


def _load_csv_to_sqlite(csv_path: str | Path) -> tuple[sqlite3.Connection, str]:
    """Load `csv_path` into an in-memory SQLite database and return
    (connection, schema_ddl_string).

    schema_ddl_string is the CREATE TABLE + first 3 sample rows,
    formatted for the generate_sql prompt. The model sees column
    names + dtypes + real example values -- much better prompt shape
    than a bare CREATE TABLE with no data.

    Raises:
        ChatCsvError: on any pandas/CSV load failure (R5 case 1).
    """
    # Lazy imports: pandas is heavy, and a mock-mode test run should
    # not require it installed. If pandas is missing, raise a friendly
    # error rather than a raw ImportError.
    try:
        import pandas as pd
    except ImportError as exc:
        raise ChatCsvError(
            "pandas is not installed but this agent needs it to load CSV files. "
            "Install with `pip install pandas>=2.0` (or `uv sync` at the "
            "workspace root)."
        ) from exc

    path = Path(csv_path)
    if not path.exists():
        raise ChatCsvError(f"CSV file not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ChatCsvError(
            f"CSV file is empty: {path}. A CSV needs at least a header row."
        ) from exc
    except pd.errors.ParserError as exc:
        raise ChatCsvError(
            f"Couldn't parse {path} as CSV: {exc}. "
            "Check that the delimiter is standard (comma/tab) and headers exist."
        ) from exc

    if df.empty:
        raise ChatCsvError(
            f"CSV file has headers but no data rows: {path}. "
            "Nothing to query."
        )

    if len(df.columns) > MAX_COLUMNS_FOR_SCHEMA_PREVIEW:
        raise ChatCsvError(
            f"CSV has {len(df.columns)} columns; this agent handles up to "
            f"{MAX_COLUMNS_FOR_SCHEMA_PREVIEW}. Wide tables need a schema-"
            "aware preview strategy this agent doesn't implement in v1."
        )

    conn = sqlite3.connect(":memory:")
    df.to_sql(DEFAULT_TABLE_NAME, conn, if_exists="replace", index=False)
    schema_ddl = _format_schema_ddl(conn, df)
    return conn, schema_ddl


def _check_prompt_size(schema_ddl: str, model: str) -> None:
    """Review fix M1 (F3.5 review). Raise ChatCsvError with the model
    named if the schema preview alone (before adding the prompt
    template + question + retry-feedback + output tokens) already
    exceeds MAX_SCHEMA_TOKENS_ESTIMATE. Uses a rough chars-per-token
    heuristic, not a native tokenizer (would tie us to one provider's
    SDK). Mirrors #02's _check_context_window shape."""
    estimated = len(schema_ddl) // CHARS_PER_TOKEN_ESTIMATE
    if estimated > MAX_SCHEMA_TOKENS_ESTIMATE:
        raise ChatCsvError(
            f"CSV schema preview is too large for single-pass analysis "
            f"(~{estimated:,} tokens estimated for the schema alone, "
            f"over the {MAX_SCHEMA_TOKENS_ESTIMATE:,}-token ceiling "
            f"this agent applies before calling {model}). Reduce the "
            "CSV's column count or column-value width, or bump "
            "MAX_SCHEMA_TOKENS_ESTIMATE in agent.py if you're on a "
            "larger-context model."
        )


def _format_schema_ddl(conn: sqlite3.Connection, df: Any) -> str:
    """Build the schema-preview string the generate_sql prompt injects:
    CREATE TABLE (from SQLite's own sqlite_master, so dtypes match
    reality) + first 3 rows rendered as a simple TSV-ish preview."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (DEFAULT_TABLE_NAME,),
    ).fetchone()
    create_table = row[0] if row else f"-- (no schema found for {DEFAULT_TABLE_NAME})"

    preview_lines = ["\nFirst 3 rows:"]
    preview_lines.append("\t".join(str(c) for c in df.columns))
    for _, r in df.head(3).iterrows():
        preview_lines.append("\t".join(str(v) for v in r.tolist()))

    return f"{create_table}\n{chr(10).join(preview_lines)}"


# --- The LangGraph state machine (pedagogical anchor) ----------------------


def _build_graph(*, llm: LLM, conn: sqlite3.Connection, model: str):
    """Build and compile the text-to-SQL state graph.

    Nodes (3 real + START + END = 5 total, matching the plan):
      generate_sql   -- LLM call; produces SQL from schema + question
                        (+ error feedback if this isn't the first try)
      execute_sql    -- runs SQL against in-memory SQLite; on error,
                        logs the attempt and lets the router retry
      format_answer  -- LLM call; produces plain-English answer from
                        (question, sql, result_rows)

    Conditional edge from execute_sql:
      - success                    -> format_answer
      - error, attempts < max      -> generate_sql (retry)
      - error, attempts >= max     -> END (state.answer stays None,
                                     caller sees exhaustion + raises
                                     ChatCsvError with .partial)

    LLM + SQLite connection + model are closed over via this factory,
    so nodes stay pure `state -> partial_state` functions -- no thread-
    local state, no global dependency, and tests can inject a
    SequenceLLM by passing a different `llm` to _build_graph.
    """
    # Lazy import: langgraph is a heavy dependency we don't want in
    # every mock-mode test run.
    from langgraph.graph import END, START, StateGraph

    def _generate_sql(state: GraphState) -> dict:
        prompt = _load_prompt("generate_sql.txt")
        retry_feedback = ""
        if state.get("error"):
            retry_feedback = (
                f"\n\n---\nYour previous SQL FAILED with this SQLite error:\n"
                f"{state['error']}\n\nThe SQL you tried was:\n{state['sql']}\n\n"
                "Fix ONLY what the error message named; keep the rest of the "
                "query if possible."
            )
        filled = prompt.replace("{schema_ddl}", state["schema_ddl"]).replace(
            "{question}", state["question"]
        ) + retry_feedback
        response = llm.complete(
            prompt=filled,
            model=model,
            temperature=0.0,
            max_tokens=DEFAULT_MAX_TOKENS_SQL,
        )
        return {
            "sql": _extract_sql(response.text),
            "attempt": state.get("attempt", 0) + 1,
        }

    def _execute_sql(state: GraphState) -> dict:
        sql = state["sql"]
        history = list(state.get("attempts_history", []))
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            columns = [d[0] for d in cursor.description] if cursor.description else []
            result_rows = [dict(zip(columns, row, strict=False)) for row in rows]
            history.append(SqlAttempt(sql=sql, row_count=len(result_rows)))
            return {
                "result_rows": result_rows,
                "error": None,
                "attempts_history": history,
            }
        except Exception as exc:
            # Review fix M3 (F3.5 review): was `except sqlite3.Error`,
            # but non-sqlite3 exceptions (TypeError on a bad arg,
            # MemoryError on huge results, etc.) would escape the graph
            # and get misclassified by _translate_api_error as an API
            # error. Broader catch treats all runtime failures in the
            # SQL execution step as retry-worthy attempts, which is
            # honest: the model gets a chance to fix its SQL, and
            # non-retryable errors will re-fail identically and end at
            # retry exhaustion with the raw exception text as evidence.
            error_msg = f"{type(exc).__name__}: {exc}"
            history.append(SqlAttempt(sql=sql, error=error_msg))
            return {
                "result_rows": None,
                "error": error_msg,
                "attempts_history": history,
            }

    def _format_answer(state: GraphState) -> dict:
        prompt = _load_prompt("format_answer.txt")
        rows_preview = (state["result_rows"] or [])[:DEFAULT_RESULT_SAMPLE_SIZE]
        filled = (
            prompt.replace("{question}", state["question"])
            .replace("{sql}", state["sql"])
            .replace("{row_count}", str(len(state["result_rows"] or [])))
            .replace("{rows_preview}", json.dumps(rows_preview, default=str))
        )
        response = llm.complete(
            prompt=filled,
            model=model,
            temperature=0.3,
            max_tokens=DEFAULT_MAX_TOKENS_ANSWER,
        )
        return {"answer": response.text.strip()}

    def _route_after_execute(state: GraphState) -> str:
        """Conditional edge: return the name of the next node (or END).
        Routes on (a) whether execute_sql succeeded, (b) whether
        retries remain."""
        if state.get("error") is None:
            return "format_answer"
        if state.get("attempt", 0) >= state.get("max_attempts", DEFAULT_MAX_ATTEMPTS):
            return END
        return "generate_sql"

    g: StateGraph = StateGraph(GraphState)
    g.add_node("generate_sql", _generate_sql)
    g.add_node("execute_sql", _execute_sql)
    g.add_node("format_answer", _format_answer)
    g.add_edge(START, "generate_sql")
    g.add_edge("generate_sql", "execute_sql")
    g.add_conditional_edges(
        "execute_sql",
        _route_after_execute,
        {
            "format_answer": "format_answer",
            "generate_sql": "generate_sql",
            END: END,
        },
    )
    g.add_edge("format_answer", END)
    return g.compile()


# --- SQL extraction --------------------------------------------------------


def _extract_sql(raw_text: str) -> str:
    """Pull SQL out of a model response. Handles the three most common
    model-authored deviations from 'return only SQL':
      - markdown code fences (```sql ... ``` or bare ```)
      - leading prose ("Here is the SQL:\\n<sql>")
      - trailing prose ("<sql>\\n\\nThis query counts...")

    Not a full SQL parser -- just strips fences and trims. The
    retry-on-execution-error branch catches whatever slips through.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening fence line (```sql or ```)
        lines = lines[1:]
        # Drop closing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# --- Error translation (R5 case 3) -----------------------------------------


def _translate_api_error(exc: Exception) -> ChatCsvError:
    """Turn an OpenAI/Anthropic/Gemini SDK exception into a
    user-facing ChatCsvError. Priority order mirrors agent #02's
    _translate_api_error (class-name first, status-code second,
    message-string fallback, generic).

    This agent does NOT auto-retry transient API errors -- see the
    S2-option-b rationale in agent #02's _rate_limit_error() and this
    file's module docstring."""
    exc_class_name = type(exc).__name__.lower()
    message_lower = str(exc).lower()
    status = getattr(exc, "status_code", None)

    if "ratelimiterror" in exc_class_name:
        return _rate_limit_error()
    if "authenticationerror" in exc_class_name or "apikeyerror" in exc_class_name:
        return _auth_error()
    if status == 429:
        return _rate_limit_error()
    if status == 401:
        return _auth_error()
    if "rate limit" in message_lower or "overloaded" in message_lower:
        return _rate_limit_error()
    if "authentication" in message_lower or "api key" in message_lower:
        return _auth_error()

    return ChatCsvError(
        f"CSV chat failed: {type(exc).__name__}: {exc}. "
        "This is an unexpected error -- check the agent logs."
    )


def _rate_limit_error() -> ChatCsvError:
    return ChatCsvError(
        "The service is temporarily rate-limited or overloaded. "
        "Wait a minute and try again."
    )


def _auth_error() -> ChatCsvError:
    return ChatCsvError(
        "API authentication failed. Check that your LLM_PROVIDER matches "
        "the API key you've set in .env (OPENAI_API_KEY / "
        "ANTHROPIC_API_KEY / GEMINI_API_KEY)."
    )


# --- Mock mode -------------------------------------------------------------


def _mock_answer(csv_path: str | Path, question: str) -> CsvAnswer:
    """Deterministic canned answer for smoke tests and CI (LLM_PROVIDER
    =mock). Does NOT load the CSV -- mock mode is for exercising the
    downstream pipeline without either an API key OR a real CSV file.
    """
    return CsvAnswer(
        question=question,
        answer=(
            f"Mock answer (would query {Path(csv_path).name} in real mode). "
            "Set LLM_PROVIDER to 'openai', 'anthropic', or 'gemini' and "
            "configure the matching API key for a real query."
        ),
        sql_used="SELECT COUNT(*) FROM data",
        row_count=1,
        attempts=[
            SqlAttempt(sql="SELECT COUNT(*) FROM data", row_count=1),
        ],
        result_sample=[{"count": 1}],
    )


# --- CLI entry point (uv run python -m agent) ------------------------------


def main() -> int:
    """CLI: takes a CSV path + a question, prints the answer as JSON."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="csv-chat",
        description="Ask a CSV a question in plain English, get an answer.",
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="Path to a CSV file.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Natural-language question about the data.",
    )
    parser.add_argument(
        "--provider",
        choices=[*SUPPORTED_PROVIDERS, "mock"],
        help="Override LLM_PROVIDER for this invocation.",
    )
    parser.add_argument(
        "--model",
        help="Override the resolved model for this invocation.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Max SQL retry attempts (default: {DEFAULT_MAX_ATTEMPTS}).",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Gradio UI instead of a one-shot CLI run.",
    )
    args = parser.parse_args()

    if args.ui:
        # Lazy import: gradio is heavy. Matches agents #01/#02's
        # build_ui() pattern for consistency.
        try:
            from .ui import build_ui  # type: ignore[import-not-found]
        except ImportError:
            from ui import build_ui  # type: ignore[import-not-found]
        build_ui().launch()
        return 0

    if not args.csv_path or not args.question:
        parser.error("csv_path and question are both required unless --ui is passed")

    try:
        result = chat_with_csv(
            args.csv_path,
            args.question,
            provider=args.provider,
            model=args.model,
            max_attempts=args.max_attempts,
        )
    except ChatCsvError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.partial is not None:
            print("---- attempts ----", file=sys.stderr)
            for i, attempt in enumerate(exc.partial.attempts, start=1):
                print(f"[{i}] SQL: {attempt.sql}", file=sys.stderr)
                if attempt.error:
                    print(f"    ERROR: {attempt.error}", file=sys.stderr)
        return 1

    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
