# Test generator -> generate + iterate a pytest suite for any Python module

Point the agent at a `.py` file. It reads the source, generates a pytest suite, runs the tests in a sandbox, reads the failures, rewrites, and repeats until everything passes (or hits the iteration cap). A deployable OSS demo for anyone who spends time hand-writing tests for small utility modules and wants to compare "what would an LLM produce for this" against their own coverage.

## Verification status

| Path | Status |
|---|---|
| Mock mode (`LLM_PROVIDER=mock`) | Fully covered by 46-test suite |
| Sandbox subprocess helpers | Fully covered by 12 real-subprocess tests (no LLM needed) |
| Real ReAct loop (`LLM_PROVIDER=openai` + key) | **Not yet verified against a live API call.** Structural correctness proven via #04's shipping code (same SDK surfaces), but no end-to-end run has been billed yet. Same open-item status as every other agent's first ship. |

## Technique demonstrated

**ReAct-style tool use with a real code-executing sandbox as feedback.** The agent has three tools:

1. `list_public_symbols(source_code)`: pure AST walk. Tells the model what to test; the model can't invent symbols.
2. `check_syntax(test_code)`: pure AST parse. Cheap pre-flight so a typo doesn't burn an execution slot.
3. `execute_test_code(test_code)`: spawns pytest as a subprocess inside a `tempfile.TemporaryDirectory`, with a wall-clock timeout and sanitized environment, returns a structured `TestExecutionResult` with pass/fail counts + stdout + stderr.

The loop: draft tests, syntax-check, execute, read the failures, rewrite, repeat. Distinct from every other agent in this catalog. #01/#02/#03/#04/#06 are extraction and classification loops (produce a structured output, done); #05 is multi-agent collaboration; #07 is meta-optimization over prompts. **#08 is the first agent whose tools mutate state** (write files, spawn processes) and whose success is measured by running the generated code.

## Why this technique for this use case

Generating tests without running them is guesswork. A model can produce syntactically valid pytest that hits the wrong exception type, misreads a function's return shape, or fabricates docstring behavior. Real feedback (pytest exit code + summary + tracebacks) turns guessing into iteration: each cycle the model reads a real error and adjusts.

**The OpenAI Agents SDK** fits because it abstracts the ReAct loop mechanics (parse tool calls, execute tools, feed observations back), structured output via `output_type=GeneratedTest`, and tool schema generation from Python signatures. Hand-rolling the loop would obscure "here's what the SDK does for you"; the code shows how thin the agent-construction call is.

Where this technique is NOT the right fit: (a) tasks with no real feedback loop (creative writing, summarization); (b) code that requires external services to run (a test suite that hits a real database); (c) code the sandbox can't execute (compiled languages, GUIs, anything needing hardware).

## What it does

Input: a filesystem path to a `.py` file. Optional overrides: `--max-iterations` (default 5, hard cap on generate-execute-refine cycles), `--sandbox-timeout` (default 30s wall-clock per subprocess run).

Output: a validated `GeneratedTest` Pydantic object with `target_module`, `test_code` (the final file content, ready to write to disk), `tests_added` (count of `def test_*`), `iterations_used`, `all_passing` (cross-checked at parse time against `final_result` so a hallucinated True gets caught), and `final_result: TestExecutionResult` (the last real pytest run's exit code, stdout, stderr, pass/fail counts).

Under `LLM_PROVIDER=mock` the whole thing returns a canned result instantly with `all_passing=True` and a placeholder test file. Under a real provider it fires a real ReAct loop.

## How to run locally

Four commands from a fresh clone (`python -m agent` must run from inside the agent's own directory: `agent` is a submodule of the digit-prefixed `08_test_generator` package):

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # set OPENAI_API_KEY (or LLM_PROVIDER=mock)
cd agents/08_test_generator
```

Mock demo (no API key, canned result):

```bash
LLM_PROVIDER=mock uv run python -m agent examples/sample_module.py
```

Real generation against the shipped sample:

```bash
uv run python -m agent examples/sample_module.py
```

Rough cost at `gpt-4.1-mini` pricing: **~$0.005-0.02 per generation** depending on how many refine cycles the model needs. Cheaper than #05 (multi-agent, ~$0.01-0.05) and #07 (GEPA optimize, ~$0.30+). Point at any local `.py` file:

```bash
uv run python -m agent path/to/my_module.py --max-iterations 8 --sandbox-timeout 60
```

Gradio UI (file upload OR pasted code):

```bash
uv run python -m agent --ui
```

Every real CLI run writes the full `GeneratedTest` JSON to `last_run.json` next to `agent.py` (gitignored) so a caller can pipe it into other tools.

## Code walkthrough

Under 500 LOC excluding UI. Read these in order:

1. **`schemas.py`**: `TestExecutionResult` + `GeneratedTest` Pydantic models. `GeneratedTest.@model_validator` cross-checks `all_passing` against `final_result` at parse time: `all_passing=True` with a non-passing `final_result` raises `ValidationError` (guards against a hallucinated success claim); the reverse (`all_passing=False` when the run actually passed) gets auto-corrected to True since a conservative model is a smaller failure than an optimistic one.
2. **`prompts/system.txt`**: the ReAct-style system prompt. Explicit 4-step reasoning strategy: `list_public_symbols` → draft tests → `check_syntax` → `execute_test_code` → iterate on failures. Hard rules: no private functions, no fabricated behavior, honest comments when a test had to be weakened.
3. **`agent.py::_list_public_symbols_impl()`**: pure AST walk. Public top-level `def`/`class` names only; skips private (`_`-prefix) and dunder names; nested defs excluded (only top-level). Returns `[]` on unparseable source so the model gets a signal to fix syntax rather than a stacktrace.
4. **`agent.py::_check_syntax_impl()`**: `ast.parse` wrapper. Returns `"OK"` or a `SyntaxError: <msg> at line N` string the model can react to.
5. **`agent.py::_run_pytest_in_sandbox()`**: **THE load-bearing tool.** Writes source + test to a `TemporaryDirectory`, spawns `[sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"]` with `capture_output=True`, `timeout=timeout_s`, `env=_sanitize_env()`, `cwd=tmpdir`. On `TimeoutExpired`, decodes partial stdout/stderr bytes and appends a `[TIMEOUT: ...]` marker so the model sees the timeout in text form.
6. **`agent.py::_sanitize_env()`**: filters `os.environ` to drop anything matching `API_KEY / TOKEN / SECRET / PASSWORD / CREDENTIAL / AWS_` patterns before handing it to the subprocess. NOT a security boundary (see "Where this fails"); just closes the loudest footgun.
7. **`agent.py::_parse_pytest_summary()`**: regex over pytest's stdout summary line. Handles `5 passed`, `3 passed, 2 failed`, `no tests ran`, and collection-error outputs (returns `(0, 0)` for anything unparseable rather than raising).
8. **`agent.py::generate_tests()`**: public API. Under mock short-circuits to `_mock_generation`. Real path: read + AST-parse source (R5 case 1) → lazy-import openai-agents SDK (R5 case 2) → `_build_agent()` → `Runner.run_sync(agent, input=..., max_turns=max_iterations * 3)` → catch `MaxTurnsExceeded` (returns a best-effort `GeneratedTest` with `all_passing=False`) → generic exceptions via `_translate_api_error` (R5 case 3, same 6-branch shape as #02-#07).
9. **`agent.py::_build_agent()`**: openai-agents `Agent(name, instructions, model, tools, output_type)` with three `function_tool`-wrapped closures. Closures close over `source_code + source_module_name + sandbox_timeout_seconds` so the model doesn't have to pass those through every tool call. Lazy SDK import so mock-mode CI doesn't need `openai-agents` installed.
10. **`ui.py::build_ui()`**: Gradio Blocks. File upload OR pasted-code textarea (either input works), iterations slider (1-10, default 5), sandbox-timeout slider (5-120s, default 30), Run button. Right column: syntax-highlighted generated test code, pass/fail summary, expandable accordion showing the final pytest stdout + stderr.
11. **`tests/test_smoke.py`**: 46 tests, all under `LLM_PROVIDER=mock`. Covers mock round-trip, schema validators (both directions of `all_passing` mismatch), AST helpers, `_parse_pytest_summary` edge cases, `_sanitize_env` (sensitive dropped, PATH preserved), **12 real subprocess sandbox tests** (known-good pass, known-bad fail, timeout, end-to-end env-key non-leak), all R5 error branches + 6-branch `_translate_api_error`, and structural guard on `_build_agent` (exactly 3 tools, correct output_type) behind `pytest.importorskip("agents")`. The subprocess tests are real but need no LLM: fast and free.

## When to use / When NOT to use

**Use when:**
- You have a small utility module (dozens to a few hundred LOC) with clear public functions and want a first-draft test suite you can iterate on
- You want to see what test cases an LLM would think to add that you might have missed
- Your code is stdlib-only or has deps already in the sandbox's `PYTHONPATH`
- You're OK spending a few cents per generation to save yourself the initial-scaffolding time

**Do NOT use when:**
- Your code requires external services to run (real database, real API, real filesystem outside `cwd`): the sandbox has no way to stand those up
- Your code is a script that runs at import time (spawns processes, opens sockets, prints to stdout): pytest imports it during collection, which trips those side effects
- You need a strong security boundary against generated code (see "Where this fails" below): use Docker or a hosted sandbox instead
- The module is huge (thousands of LOC): the model will run out of context or generate shallow tests that don't cover the depth

## Where this fails

**Sandbox honest-tradeoffs (READ THIS FIRST):** the subprocess sandbox provides wall-clock timeout, isolated `cwd` (`TemporaryDirectory`), and env sanitization (drops `*API_KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*CREDENTIAL*`, `AWS_*` before invoking pytest). It does **NOT** provide network isolation, FS isolation beyond `cwd`, or import restrictions. A hostile prompt (or a real bug in the LLM's output) can generate a test that:
- Reads arbitrary files (`open("/etc/passwd")`, `Path.home() / ".aws" / "credentials"`)
- Hits the internet (`import urllib.request; urllib.request.urlopen(...)`)
- Shells out (`import os; os.system(...)`)

This is a *good-behavior* sandbox for legitimate LLM-generated test code, not a security perimeter against malicious code. Users who need real isolation should use Docker or a hosted sandbox (E2B, Modal, Judge0) instead. This limitation is intentional: adding Docker as a required dep would break the "clone-and-run" pattern for anyone without Docker installed, and the goal here is to teach the tool-use pattern, not to build a hardened sandbox service.

**Specific failure modes:**

- **Tests that import missing deps**: the sandbox only has whatever's in `sys.executable`'s environment. If your source imports `pandas` and `pandas` isn't installed in the parent Python, the sandbox subprocess collection-errors and the model sees the ImportError. It'll usually catch this on the first iteration and drop the failing test, but a small module with one exotic import will fail generation entirely. Workaround: install the deps in the same env, or pass `--max-iterations` high enough that the model can iterate around missing pieces.
- **Modules with side effects at import time**: `print()` at module scope, `logging.basicConfig()`, network calls, subprocess spawns. pytest imports the module during collection, so those side effects fire before any test runs. The model can't work around this from a test file alone. Refactor the module to guard side effects behind `if __name__ == "__main__":` before generating tests.
- **Tests that hang**: covered by the wall-clock timeout, but a 30s default can be too tight for a module whose real test setup takes >20s. Bump `--sandbox-timeout 90` (or per-agent CLI flag). Wall-clock hangs kill the whole subprocess; there's no per-test timeout.
- **The model producing "clever" mocks it can't verify**: hard rule in the prompt tells it not to, but sometimes it invents `unittest.mock.patch("some.module.function")` calls for behavior it inferred from a docstring. Symptom: tests that pass in the sandbox but wouldn't pass against real dependencies. Mitigation: read the generated test code before trusting it (`last_run.json`'s `test_code` field is your audit trail).
- **Real source bugs cause the loop to spin**: if a function actually returns the wrong result (the source has a bug), the model iterates trying to write a test that matches the wrong result. The prompt tells it to add a `# NOTE: source behaves X, not Y as expected` comment when it detects this, but a model that trusts docstrings might chew iterations trying to force a passing test that shouldn't pass. Symptom: `iterations_used == max_iterations` with `all_passing == False` and confusing tests that assert whatever the buggy source returned. Mitigation: read the source-bug notes in `test_code` before committing generated tests.
- **`MaxTurnsExceeded`**: the model didn't converge in `max_iterations * 3` turns. Returns a placeholder `GeneratedTest` with `all_passing=False` and a comment explaining the cap was hit. Bump `--max-iterations` or simplify the source.
- **Rate limit / API failure**: no auto-retry with backoff (explicit design decision: iteration is already the point of this agent; a rate-limit blip mid-run should stop rather than silently double the cost). One failed attempt: translated to "temporarily rate-limited", user re-runs manually.
- **Generation not verified against a real API call yet**: this agent's code paths are structurally correct against `openai-agents>=0.22` (verified via agent #04's shipping code, which uses the same SDK surfaces), but a real end-to-end ReAct loop with a live OpenAI key has NOT been exercised at ship time. Same open-item status as every previous agent's first ship.
