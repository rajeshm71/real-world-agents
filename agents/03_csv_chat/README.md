# Chat with your CSV → plain-English answers backed by real SQL

Drop a CSV, ask a question in English, get a plain-English answer with the actual SQL and the underlying rows behind it. A deployable OSS alternative to "learn SQL" or "beg the data team" for the marketers, ops folks, founders, and analysts who need an answer from a spreadsheet without writing a query themselves.

## Technique demonstrated

**Text-to-SQL with a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine + retry-on-error loop.** LangGraph is the right shape here because the workflow has real branching, not the linear pipeline of agents #01 or #02: SQL succeeded → format an answer; SQL failed → retry with the SQLite error text as feedback; retries exhausted → surface the failed attempts to the user.

The state graph makes those branches explicit as data: nodes are pure `state -> partial_state` functions, and the retry loop is a **conditional edge** (`_route_after_execute` returns `"generate_sql"` on error, `"format_answer"` on success, `END` on exhaustion). Contrast with agent #02, where the retry loop is a hand-rolled `for` loop hidden inside `_run_review_loop`. Same "retry on failure with error feedback" pattern; two different levels of visibility.

## Why this technique for this use case

Text-to-SQL naturally decomposes into two LLM calls: **generate SQL** (schema + question → SQL string) and **format answer** (question + SQL + result rows → plain-English answer). Squeezing both into one prompt loses the retry-on-execution-error signal that makes the whole system work — without it, the model has no chance to correct "no such column 'penquin_weight'" (a typo the model itself might have made) into "penguin_weight" on the next try.

Splitting them also lets the state graph retry ONLY the SQL step on execution error (cheaper than re-generating the whole answer), and lets the format-answer node see actual result rows as context (much better answer quality). Trade: doubled cost + latency vs a one-shot approach. Documented honestly in the Cost note at the bottom of this section.

Where this technique is NOT the right fit: (a) datasets that don't fit an in-memory SQLite (millions of rows, or a real database with hundreds of tables — use a proper text-to-SQL tool with schema linking there), (b) questions that need multiple JOINs across files (v1 loads one CSV per session), (c) questions about the SHAPE of data rather than its contents (a bar chart of X — this agent returns tables, not visualisations).

**Cost note:** two LLM calls per question, one at temperature=0 (SQL generation, tight) and one at temperature=0.3 (answer formatting, natural). Rough ballpark at gpt-4.1-mini pricing: ~$0.001-0.005 per question depending on schema size + result-row count.

## What it does

Input: a CSV file + a natural-language question. Output: a validated `CsvAnswer` Pydantic object with the plain-English `answer`, the winning `sql_used`, the total `row_count`, up to 10 sample rows in `result_sample`, and the full `attempts` history (including any SQL attempts that failed before the winning one). Empty result set is a valid outcome (`row_count=0`, `result_sample=[]`, and an answer that says so honestly).

CSV is loaded into an in-memory SQLite database (via `pandas.read_csv` + `df.to_sql`) named `data`. Each call gets a fresh database; there's no conversational history in v1. The schema preview (CREATE TABLE + first 3 rows) is injected into the SQL-generation prompt so the model sees column names + dtypes + example values, not just a bare schema.

## How to run locally

Four commands from a fresh clone (`python -m agent` must run from inside the agent's own directory — `agent` is a submodule of the digit-prefixed `03_csv_chat` package, not importable as a top-level module from the repo root):

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # then edit .env: set LLM_PROVIDER + the matching API key
cd agents/03_csv_chat
```

`.env` defaults to `LLM_PROVIDER=openai` (using `gpt-4.1-mini`). Switch providers with `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=gemini` and the matching key — no code changes needed. See `.env.example` for every option, including per-provider model overrides.

CLI:

```bash
uv run python -m agent path/to/data.csv "your question here"
```

Override provider/model/retries for a single call:

```bash
uv run python -m agent data.csv "how many rows?" --provider anthropic --model claude-sonnet-5 --max-attempts 5
```

Gradio UI:

```bash
uv run python -m agent --ui
```

Mock mode (no API key, canned response — for testing the pipeline end-to-end):

```bash
LLM_PROVIDER=mock uv run python -m agent data.csv "any question"
# or, without touching the env var:
uv run python -m agent data.csv "any question" --provider mock
```

## Code walkthrough

Under 500 LOC excluding UI. Read these in order to understand the pattern:

1. **`schemas.py`** — `SqlAttempt` + `CsvAnswer` Pydantic models. `attempts` has `min_length=1` so a state-graph logging bug surfaces at validation, not as a blank UI. `result_sample` is `list[dict]` (JSON-serializable, not a DataFrame) for downstream tool interop.
2. **`prompts/generate_sql.txt`** — SQLite-dialect-specific instructions (LIKE not ILIKE, `||` for concat, booleans as 0/1, quote identifiers with spaces). Table always named `data`. Retry-feedback text is appended by `agent.py` at retry time; the base prompt is dialect + schema + rules only.
3. **`prompts/format_answer.txt`** — data-explainer role: gets question + SQL + row_count + rows_preview, produces a plain-English answer. Rules: cite actual numbers, don't repeat the SQL, be honest on 0-row results.
4. **`agent.py::resolve_provider()`** — reads `LLM_PROVIDER` (default `"openai"`); `common/llm.py::resolve_model()` reads the matching per-provider model override env var. Same env-var contract as agents #01/#02.
5. **`agent.py::chat_with_csv()`** — the public API. Under `provider="mock"` short-circuits to `_mock_answer` (no CSV load, no graph). Otherwise: R5 CSV-load check → build state graph → `graph.invoke(initial_state)` → translate result to `CsvAnswer` or raise `ChatCsvError` on retry exhaustion.
6. **`agent.py::_build_graph()`** — **THE pedagogical anchor.** LangGraph state machine with 3 real nodes + START + END = 5 nodes total. Nodes are closures over the LLM + SQLite connection + model, so they stay pure `state -> partial_state`. The retry loop is a REAL EDGE — `_route_after_execute` inspects state and returns the name of the next node, which LangGraph traverses. This is the visible-workflow version of agent #02's `_run_review_loop` `for` loop.
7. **`agent.py::_load_csv_to_sqlite()`** — R5 case 1. pandas.read_csv → sqlite3 in-memory. Catches 5 failure modes with actionable messages: file-not-found, empty file, pandas.ParserError, headers-with-no-data, too-many-columns (>40 cap for prompt-size sanity). Fails fast BEFORE the graph runs so the error path is dead simple.
8. **`agent.py::_translate_api_error()`** — R5 case 3. Six-branch priority order (class-name → status-code → message-string fallback → generic), same shape as agent #02. This agent does NOT auto-retry rate-limit errors: the user pays per real API call and is better-placed to decide whether to wait and re-run than to have the graph silently spend more of their budget on a failing endpoint.
9. **`agent.py::_extract_sql()`** — pulls SQL out of a model response, handling three common deviations from "return only SQL": markdown code fences (```sql ... ``` or bare ```), leading prose ("Here is the SQL:"), trailing prose. Not a full SQL parser — whatever slips through gets caught by the retry-on-execution-error branch.
10. **`ui.py::build_ui()`** — Gradio Blocks: CSV upload widget, question textbox, provider Radio, answer + SQL used + attempts-history (only shown when there was more than one attempt) + result-table Dataframe. Kept intentionally small so the load-bearing code stays in `agent.py`.
11. **`tests/test_smoke.py`** — 30 tests, all under `LLM_PROVIDER=mock`. Covers mock path, CSV loading edge cases, state graph (happy / retry / exhaustion / zero-row / fenced SQL), `_translate_api_error`, and pure helpers. `SequenceLLM` fixture scripts multi-response scenarios for the retry-loop tests without touching shared chassis.

## When to use / When NOT to use

**Use when:**
- You have a CSV (spreadsheet export, database dump, API response) and want to ask questions in English without writing SQL
- You want to see the SQL the model wrote (auditable) alongside the plain-English answer
- Your CSV fits comfortably in memory (thousands of rows, dozens of columns — the 40-column cap is a soft signal that a real database is a better tool past that)
- You're building your own text-to-SQL tool and want a readable reference for how a LangGraph state machine with retry-on-error is structured

**Do NOT use when:**
- Your data is in a real database with hundreds of tables — a proper text-to-SQL tool with schema linking (e.g. Vanna, DIN-SQL, or a hosted service) will do far better on schema you didn't hand-curate
- You need to JOIN across multiple CSVs — v1 loads one CSV per session; multi-file JOIN needs schema-preview changes not implemented yet
- You need chart/graph output — this agent returns rows, not visualisations
- You need conversation memory ("what about the ones I asked about last time?") — each call is stateless; no history persists between requests in v1
- Your questions require reasoning the model can't do in SQL (fuzzy matching human names, semantic similarity, external knowledge) — pick a different tool shape

## Where this fails

- **Ambiguous column names** — a CSV with columns `revenue` and `revenue_ytd` will trip the model when the user asks about "revenue"; it'll usually pick the wrong one and the answer will be technically correct but semantically wrong. No error to catch here — the SQL runs successfully. Mitigation: name your columns precisely, or accept the model's pick and check.
- **Wide tables (>40 columns)** — the schema-preview approach can't fit that many column names + dtypes + sample rows into the prompt without overwhelming the model's attention. The agent refuses via `_load_csv_to_sqlite`'s R5 branch with a clear message rather than silently truncating.
- **Non-standard delimiters** — pandas.read_csv defaults to comma; a tab-delimited or semicolon-delimited file may parse to a single mangled column. Error surfaces via pandas ParserError → R5 CSV-load branch with a "check delimiter" message. Workaround: convert to CSV first, or add a delimiter param to `_load_csv_to_sqlite` in a follow-up.
- **Date columns stored as strings** — SQLite has no native date type; the schema shows dates as TEXT, and the model needs to use `date()` / `strftime()` for date arithmetic. Sometimes the model forgets, produces a broken WHERE clause, and the retry-on-error loop catches it. Adds latency but doesn't fail.
- **Very large result sets** — if the SQL returns 100,000+ rows, the `format_answer` node still only sees the first 10 in `rows_preview`. Its answer is correct about aggregates (row_count is exact) but can't describe patterns in unshown rows. The full result IS in `result_sample[:10]` in the returned `CsvAnswer`, and `row_count` is the honest total.
- **The model producing plausibly-wrong SQL that runs anyway** — if the model interprets the question as "average height by species" but the user meant "median height", the SQL succeeds and the answer confidently gives the wrong statistic. No error to catch, no retry triggered. Only mitigation: read the `sql_used` field on the returned `CsvAnswer` before trusting the number.
- **Rate limit / API failure** — no auto-retry with backoff (explicit design decision, see `agent.py::_rate_limit_error`). One failed attempt → translated to a friendly "temporarily rate-limited" message → user re-runs manually.
