"""Test generator: agent #08 of real-world-agents.

Technique demonstrated: **tool use + iterative refinement.** A ReAct
loop where the model uses a real code-executing tool to check its own
work, reads the failures, and rewrites. Distinct from every earlier
agent: #01/#02/#03/#04/#06 are extraction / classification loops
(model produces a structured output, done); #05 is multi-agent
collaboration; #07 is meta-optimization over prompts. #08 is the
first agent whose tools MUTATE STATE (write files, spawn processes)
and whose success is measured by running the generated code.

Why this technique for this use case: generating tests without running
them is guesswork. The model can produce syntactically-valid pytest
that hits the wrong exception type, misreads a function's return
shape, or fabricates a docstring behavior. Real feedback (`pytest`
exit code + summary + tracebacks) turns guessing into iteration:
each cycle the model reads a real error and adjusts.

Sandbox honest-tradeoffs:
The subprocess sandbox provides wall-clock timeout, isolated cwd
(tempfile.TemporaryDirectory), and env sanitization (drops anything
matching API_KEY / TOKEN / SECRET / PASSWORD / CREDENTIAL patterns so
generated tests can't exfiltrate OPENAI_API_KEY into their own
output).

It does NOT provide:
- Network isolation: generated tests can hit the internet.
- FS isolation beyond cwd: `open("/etc/passwd")` still works.
- Import restrictions: `import os; os.system(...)` still works.

This is a *good-behavior* sandbox for LLM-generated test code, not a
security perimeter against malicious code. Users who need real
isolation should use Docker or a hosted sandbox (E2B, Modal, Judge0)
instead. Documented explicitly in README.

Real error handling per R5 (three cases):
1. Source path missing / unparseable -> TestGeneratorError before any
   LLM call.
2. openai-agents SDK not installed -> TestGeneratorError with a
   `uv sync` pointer.
3. LLM API failure during Runner.run_sync -> six-branch translator
   (class-name -> status -> message -> generic), same shape as
   agents #02-#07.

Provider stance matches agent #04: openai-only in v1. Multi-provider
via LiteLLM is a documented one-line swap in the README, deferred to
keep the dep count tight.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# Dual-mode import: relative for tests-as-package, absolute for
# `python -m agent` from inside this dir. See agent #01's identical
# pattern for the full rationale.
try:
    from .schemas import GeneratedTest, TestExecutionResult
except ImportError:
    from schemas import GeneratedTest, TestExecutionResult

# common.llm is the root workspace package -- never a relative import.
from common.llm import resolve_model

# --- Provider + constants --------------------------------------------------

# OpenAI-only in v1, matching agent #04's stance. LiteLLM swap is the
# documented one-line change for a v1.1 that adds Anthropic/Gemini.
SUPPORTED_PROVIDERS = ("openai", "ollama")

_DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-4o-mini",
    "ollama": "gemma4:e4b",
}

DEFAULT_MAX_ITERATIONS = 5
DEFAULT_SANDBOX_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_TURNS_PER_ITERATION = 3  # ~1 plan + 1 generate + 1 execute per iteration

# Env-var name patterns that get stripped before the sandbox subprocess
# gets to see os.environ. Case-insensitive substring match against the
# variable name. Reduces the obvious footgun of an LLM-generated test
# accidentally printing OPENAI_API_KEY via os.environ inspection.
_SENSITIVE_ENV_PATTERNS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    # AWS: only the actual credential vars, NOT AWS_REGION or
    # AWS_DEFAULT_REGION (benign; a legit boto3-using test needs them).
    "AWS_ACCESS_KEY",
    "AWS_SECRET",
    "AWS_SESSION",
    # Google Cloud + Azure sensitive vars
    "GOOGLE_APPLICATION_CREDENTIALS",
    "AZURE_CLIENT_SECRET",
)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"


# --- Error type ------------------------------------------------------------


@dataclass
class GenerationAttempt:
    """Partial state attached to TestGeneratorError when the pipeline
    fails partway through. `partial_test_code` is the best draft the
    agent produced before erroring out; empty if it failed before any
    draft was generated."""

    stage: str  # 'read_source' / 'parse_source' / 'build_agent' / 'runner' / 'translate'
    partial_test_code: str = ""


class TestGeneratorError(Exception):
    """Raised on any user-facing failure: bad source path, unparseable
    source, missing SDK, or API error during the Runner."""

    # The name starts with "Test" only because this whole agent is
    # about generating tests; this tells pytest not to try to collect
    # it as a test class.
    __test__ = False

    def __init__(self, message: str, partial: GenerationAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Provider resolution ---------------------------------------------------


def resolve_provider() -> str:
    """LLM_PROVIDER env var, defaulting to "openai". OpenAI-only in v1;
    "mock" is handled by the caller (generate_tests), not here."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}. "
            "Multi-provider support via LiteLLM is a documented follow-up "
            "for #08 -- see the README."
        )
    return provider


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# --- Pure AST + parsing helpers --------------------------------------------


def _list_public_symbols_impl(source_code: str) -> list[str]:
    """Return public top-level `def` and `class` names from source_code.
    Excludes names starting with `_` (private convention). Handles
    unparseable source by returning an empty list -- the model gets a
    signal to fix syntax before proceeding rather than a stacktrace.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
    ]


def _check_syntax_impl(test_code: str) -> str:
    """Return "OK" if test_code parses, else the SyntaxError message.
    Cheap pre-flight before spending an execute_test_code call."""
    try:
        ast.parse(test_code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} at line {exc.lineno}"
    return "OK"


def _parse_pytest_summary(stdout: str) -> tuple[int, int]:
    """Pull (tests_passed, tests_failed) counts from pytest's summary
    line. Handles the common shapes: `5 passed`, `3 passed, 2 failed`,
    `1 failed`, `no tests ran`, and collection-error output (returns
    (0, 0) for anything unparseable rather than raising -- the exit
    code carries the pass/fail signal too)."""
    passed = 0
    failed = 0
    for line in stdout.splitlines():
        # pytest's short summary line typically looks like:
        # "===== 3 passed, 2 failed in 0.12s ====="
        if "passed" not in line and "failed" not in line:
            continue
        m_passed = re.search(r"(\d+)\s+passed", line)
        if m_passed:
            passed = int(m_passed.group(1))
        m_failed = re.search(r"(\d+)\s+failed", line)
        if m_failed:
            failed = int(m_failed.group(1))
    return passed, failed


def _count_test_functions(test_code: str) -> int:
    """Regex-count `def test_*` in the generated code. Cheap; doesn't
    require re-parsing when test_code is already known to be valid."""
    return len(re.findall(r"^\s*def\s+test_\w+", test_code, re.MULTILINE))


# --- Sandbox helpers -------------------------------------------------------


def _sanitize_env() -> dict[str, str]:
    """Return os.environ with sensitive-looking vars dropped. NOT a
    security boundary -- a hostile prompt can still read `open("/.env")`.
    This just closes the loudest footgun (accidental key leak via
    os.environ in the sandbox's stdout).

    PATH, PYTHONPATH, HOME, USERPROFILE, and other benign vars are
    preserved so the subprocess can still find pytest + its deps."""
    return {
        key: value
        for key, value in os.environ.items()
        if not any(pattern in key.upper() for pattern in _SENSITIVE_ENV_PATTERNS)
    }


def _run_pytest_in_sandbox(
    *,
    source_code: str,
    source_module_name: str,
    test_code: str,
    timeout_s: float,
) -> TestExecutionResult:
    """Write source + test into a fresh TemporaryDirectory and run
    pytest as a subprocess against them. Returns a TestExecutionResult
    with exit_code, captured stdout/stderr, timed_out flag, and parsed
    pass/fail counts.

    Read the module docstring for the sandbox's honest-tradeoffs: this
    provides wall-clock timeout, cwd isolation, and env sanitization,
    but does NOT provide network isolation, FS isolation beyond cwd,
    or import restrictions. NOT a security boundary.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / f"{source_module_name}.py"
        test_path = Path(tmpdir) / f"test_generated_{source_module_name}.py"
        source_path.write_text(source_code, encoding="utf-8")
        test_path.write_text(test_code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=_sanitize_env(),
                check=False,
            )
            passed, failed = _parse_pytest_summary(proc.stdout)
            return TestExecutionResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                timed_out=False,
                tests_passed=passed,
                tests_failed=failed,
            )
        except subprocess.TimeoutExpired as exc:
            # TimeoutExpired's stdout/stderr are bytes-or-None; coerce
            # to str for the model. Append a plain-English marker so the
            # LLM can see the timeout as text, not just via timed_out.
            partial_stdout = exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
            partial_stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            return TestExecutionResult(
                exit_code=-1,
                stdout=partial_stdout,
                stderr=(
                    partial_stderr
                    + f"\n[TIMEOUT: subprocess exceeded {timeout_s}s wall-clock]"
                ),
                timed_out=True,
                tests_passed=0,
                tests_failed=0,
            )


# --- Public API ------------------------------------------------------------


def generate_tests(
    source_path: str | Path,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    sandbox_timeout_seconds: float = DEFAULT_SANDBOX_TIMEOUT_SECONDS,
    provider: str | None = None,
    model: str | None = None,
    _agent=None,
) -> GeneratedTest:
    """Generate a passing pytest suite for the module at source_path.

    Args:
        source_path: filesystem path to a Python source file to test.
        max_iterations: cap on generate-execute-refine cycles. Each
            cycle consumes ~3 SDK turns (plan, generate, execute); a
            cap of 5 corresponds to max_turns=15 through the Runner.
        sandbox_timeout_seconds: wall-clock cap on each subprocess
            invocation. Prevents an infinite loop in generated code
            from hanging the whole run.
        provider: "openai" (default) or "mock". OpenAI-only in v1.
        model: model ID override; defaults to common.llm.resolve_model.
        _agent: test injection escape hatch; production callers leave None.

    Returns:
        A validated GeneratedTest with the final test code, count of
        tests added, iterations used, and the last TestExecutionResult
        as evidence for `all_passing`.

    Raises:
        TestGeneratorError: on missing/unparseable source, missing SDK,
            or API failure during the Runner.
    """
    # Fail-fast on nonsense values BEFORE any FS read or LLM call.
    # `max_iterations < 1` would eventually raise a ValidationError deep
    # inside the schema (iterations_used has ge=1) with a confusing
    # trace; better to reject at the boundary.
    if max_iterations < 1:
        raise ValueError(
            f"max_iterations must be >= 1, got {max_iterations}. "
            "Pass 1 or higher; the default is DEFAULT_MAX_ITERATIONS."
        )
    if sandbox_timeout_seconds <= 0:
        raise ValueError(
            f"sandbox_timeout_seconds must be > 0, got {sandbox_timeout_seconds}. "
            "The subprocess needs a positive wall-clock cap."
        )

    resolved_provider = (provider or resolve_provider()).lower()
    source_path = Path(source_path)

    if resolved_provider == "mock":
        return _mock_generation(source_path)

    resolved_model = model or _DEFAULT_MODEL_BY_PROVIDER.get(
        resolved_provider
    ) or resolve_model(resolved_provider)

    # R5 case 1: bad path / unreadable / unparseable source, BEFORE any
    # LLM call. Fail fast with actionable message.
    try:
        source_code = source_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TestGeneratorError(
            f"Source file not found: {source_path}. "
            "Pass a valid .py file path to generate_tests().",
            partial=GenerationAttempt(stage="read_source"),
        ) from exc
    except OSError as exc:
        raise TestGeneratorError(
            f"Cannot read source file {source_path}: {exc}",
            partial=GenerationAttempt(stage="read_source"),
        ) from exc

    try:
        ast.parse(source_code)
    except SyntaxError as exc:
        raise TestGeneratorError(
            f"Source file {source_path} has a syntax error at line "
            f"{exc.lineno}: {exc.msg}. Fix the source before generating tests.",
            partial=GenerationAttempt(stage="parse_source"),
        ) from exc

    source_module_name = source_path.stem

    # R5 case 2: SDK not installed. Lazy import so mock-mode CI doesn't
    # need openai-agents at all.
    try:
        from agents import MaxTurnsExceeded, Runner
    except ImportError as exc:
        raise TestGeneratorError(
            "openai-agents SDK is not installed. Run `uv sync` from the "
            "repo root, or `pip install openai-agents>=0.22` in this agent's "
            "environment.",
            partial=GenerationAttempt(stage="build_agent"),
        ) from exc

    agent_obj = _agent if _agent is not None else _build_agent(
        source_code=source_code,
        source_module_name=source_module_name,
        model=resolved_model,
        sandbox_timeout_seconds=sandbox_timeout_seconds,
        provider=resolved_provider,
    )

    # Each iteration is roughly plan + generate + execute. Cap Runner's
    # max_turns accordingly; MaxTurnsExceeded catches the case where the
    # model can't converge in the allotted cycles.
    max_turns = max_iterations * DEFAULT_MAX_TURNS_PER_ITERATION

    initial_input = (
        f"Generate a passing pytest suite for the following Python module. "
        f"The module will be written to `{source_module_name}.py` next to "
        f"your test file in the sandbox, so import it as "
        f"`from {source_module_name} import ...`.\n\n"
        f"--- source code ---\n{source_code}\n--- end source ---"
    )

    try:
        result = Runner.run_sync(agent_obj, input=initial_input, max_turns=max_turns)
    except MaxTurnsExceeded as exc:
        # Best-effort: return a GeneratedTest showing the iteration cap
        # was hit. `all_passing=False` via the validator; test_code is
        # whatever partial the exception may carry (usually empty).
        return _max_turns_result(
            source_module_name=source_module_name,
            max_iterations=max_iterations,
            reason=str(exc) or "max_turns exceeded",
        )
    except Exception as exc:
        raise _translate_api_error(exc) from exc

    generated: GeneratedTest = result.final_output_as(GeneratedTest)
    return generated


# --- Agent construction ----------------------------------------------------


def _build_agent(
    *,
    source_code: str,
    source_module_name: str,
    model: str,
    sandbox_timeout_seconds: float,
    provider: str = "openai",
):
    """Build the openai-agents Agent with three tools. Tools are plain
    closures over `source_code` + `source_module_name` +
    `sandbox_timeout_seconds` so the model doesn't have to pass those
    through every call. Same closure-in-factory pattern agent #04 uses.
    """
    from agents import Agent, function_tool

    if provider == "ollama":
        from agents.models.openai_chatcompletions import (
            OpenAIChatCompletionsModel,
        )
        from openai import AsyncOpenAI

        from common.llm import ollama_base_url
        model_arg = OpenAIChatCompletionsModel(
            model=model,
            openai_client=AsyncOpenAI(
                base_url=ollama_base_url(), api_key="ollama"
            ),
        )
    else:
        model_arg = model

    def list_public_symbols(source_code_input: str) -> list[str]:
        """List public top-level `def` and `class` names from a Python source string. Call this first to discover what to test."""
        return _list_public_symbols_impl(source_code_input)

    def check_syntax(test_code: str) -> str:
        """Return "OK" if the given Python test_code parses, or a SyntaxError message. Cheap pre-flight before execute_test_code."""
        return _check_syntax_impl(test_code)

    def execute_test_code(test_code: str) -> TestExecutionResult:
        """Write test_code + the source module into an isolated sandbox and run pytest. Returns exit_code, stdout, stderr, tests_passed, tests_failed, timed_out."""
        return _run_pytest_in_sandbox(
            source_code=source_code,
            source_module_name=source_module_name,
            test_code=test_code,
            timeout_s=sandbox_timeout_seconds,
        )

    return Agent(
        name="test-generator-agent",
        instructions=_load_system_prompt(),
        model=model_arg,
        tools=[
            function_tool(list_public_symbols),
            function_tool(check_syntax),
            function_tool(execute_test_code),
        ],
        output_type=GeneratedTest,
    )


# --- Error translation (R5 case 3) -----------------------------------------


def _translate_api_error(exc: Exception) -> TestGeneratorError:
    """Turn an openai-agents SDK / provider exception into a user-facing
    TestGeneratorError. Six-branch priority order mirrors agents #02-#07:
      1. Class-name check: rate limit -> rate-limit case
      2. Class-name check: auth -> auth case
      3. Status code 429 -> rate-limit case
      4. Status code 401 -> auth case
      5. Message-string fallback (rate limit / overloaded, auth / key)
      6. Generic fallback with original exception preserved
    """
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
    from common.llm import OLLAMA_CONNECTION_HINT, is_ollama_connection_error
    if is_ollama_connection_error(exc):
        return TestGeneratorError(OLLAMA_CONNECTION_HINT)

    return TestGeneratorError(
        f"Generation failed: {type(exc).__name__}: {exc}. "
        "This is an unexpected error -- check the agent logs. "
        "The agent's partial state (if any) was not preserved."
    )


def _rate_limit_error() -> TestGeneratorError:
    """Shared message so every path that reaches this case can't drift
    apart. No auto-retry: iteration is already the point of this agent;
    a rate-limit blip mid-iteration means the whole run should stop so
    the user can decide whether to retry or wait."""
    return TestGeneratorError(
        "The provider is temporarily rate-limited or overloaded. "
        "Wait a minute and try again. The agent's partial progress "
        "was not preserved; you will need to restart the run."
    )


def _auth_error() -> TestGeneratorError:
    return TestGeneratorError(
        "Authentication failed: check that OPENAI_API_KEY is set correctly. "
        "See .env.example at the repo root."
    )


# --- MaxTurnsExceeded result ----------------------------------------------


def _max_turns_result(
    *, source_module_name: str, max_iterations: int, reason: str
) -> GeneratedTest:
    """Build a GeneratedTest that honestly reports the iteration cap
    was hit without a passing suite. Used when Runner raises
    MaxTurnsExceeded; the schema's validator would reject
    all_passing=True with no evidence, so this always sets False."""
    return GeneratedTest(
        target_module=source_module_name,
        test_code=(
            f"# The agent exhausted its {max_iterations}-iteration cap "
            f"without producing a passing test suite.\n"
            f"# Reason: {reason}\n"
            f"# Try again with --max-iterations >{max_iterations}, or "
            f"simplify the source module.\n"
        ),
        tests_added=0,
        iterations_used=max_iterations,
        all_passing=False,
        final_result=TestExecutionResult(
            exit_code=-1,
            stdout="",
            stderr=f"MaxTurnsExceeded: {reason}",
            timed_out=False,
            tests_passed=0,
            tests_failed=0,
        ),
    )


# --- Mock mode -------------------------------------------------------------


def _mock_generation(source_path: Path) -> GeneratedTest:
    """Deterministic canned GeneratedTest for CI + local exploration.
    No SDK import, no provider SDK, no key, no subprocess. The
    source_path is echoed into the generated test_code as a comment
    so tests can prove the mock saw its input (same anti-refactor
    guard convention agent #01 uses).

    Test code is deliberately self-contained (no imports from the
    target module) so a reader who copy-pastes the mock's output gets
    a file that actually parses + runs, rather than a broken import
    against a symbol the target doesn't have. A canned mock shouldn't
    pretend to know the target's real symbols."""
    module_name = source_path.stem or "mock_module"
    test_code = (
        f"# [MOCK] Generated for {source_path} (module: {module_name})\n"
        "# No real analysis happened; set LLM_PROVIDER=openai for real "
        "generation.\n\n\n"
        "def test_mock_placeholder():\n"
        "    assert True\n"
    )
    return GeneratedTest(
        target_module=str(source_path),
        test_code=test_code,
        tests_added=1,
        iterations_used=1,
        all_passing=True,
        final_result=TestExecutionResult(
            exit_code=0,
            stdout="1 passed",
            stderr="",
            timed_out=False,
            tests_passed=1,
            tests_failed=0,
        ),
    )


# --- CLI entry point -------------------------------------------------------


def main() -> int:
    """CLI: `uv run python -m agent path/to/source.py`. Prints a summary
    + writes full GeneratedTest JSON to `last_run.json` next to this
    file (gitignored per agents/*/last_run.json)."""
    parser = argparse.ArgumentParser(
        prog="test-generator",
        description=(
            "Generate a passing pytest suite for a Python module. "
            "Set LLM_PROVIDER=mock for a canned demo, or supply "
            "OPENAI_API_KEY for a real ReAct run."
        ),
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a Python source file. Omit when using --ui.",
    )
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI instead of CLI.")
    parser.add_argument(
        "--provider",
        choices=(*SUPPORTED_PROVIDERS, "mock"),
        default=None,
        help="Override LLM_PROVIDER for this run.",
    )
    parser.add_argument("--model", default=None, help="Override the resolved model.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Cap on generate-execute-refine cycles (default 5).",
    )
    parser.add_argument(
        "--sandbox-timeout",
        type=float,
        default=DEFAULT_SANDBOX_TIMEOUT_SECONDS,
        help="Wall-clock seconds cap per subprocess run (default 30).",
    )
    args = parser.parse_args()

    if args.ui:
        try:
            from .ui import build_ui
        except ImportError:
            from ui import build_ui
        build_ui().launch()
        return 0

    if args.source is None:
        parser.error("source path is required (or pass --ui to launch the web interface)")
        return 2  # unreachable

    start = time.perf_counter()
    try:
        result = generate_tests(
            args.source,
            max_iterations=args.max_iterations,
            sandbox_timeout_seconds=args.sandbox_timeout,
            provider=args.provider,
            model=args.model,
        )
    except TestGeneratorError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    out_path = Path(__file__).parent / "last_run.json"
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    print(f"Target module:     {result.target_module}")
    print(f"Tests added:       {result.tests_added}")
    print(f"Iterations used:   {result.iterations_used}")
    print(f"All passing:       {result.all_passing}")
    print(f"Wall time:         {elapsed:.1f}s")
    print(f"Final exit code:   {result.final_result.exit_code}")
    print(f"Final pass/fail:   {result.final_result.tests_passed} passed, "
          f"{result.final_result.tests_failed} failed")
    print()
    print(f"Test code written to {out_path}. Full JSON also in that file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
