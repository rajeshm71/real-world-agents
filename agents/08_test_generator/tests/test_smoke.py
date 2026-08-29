"""Smoke tests for the test-generator agent.

All tests run under LLM_PROVIDER=mock (R8 in CONTRIBUTING.md) -- CI
never touches a real API key. SDK structural paths are gated behind
`pytest.importorskip("agents")`; the sandbox subprocess tests are REAL
subprocess calls but need no LLM -- they exercise the load-bearing
sandbox helpers directly for free.

Structure:
1. Mock-mode round-trip + input passthrough guard.
2. Schema validators (all_passing must match final_result).
3. AST helpers (list_public_symbols, check_syntax) on synthetic sources.
4. _parse_pytest_summary on various real pytest output shapes.
5. _sanitize_env: sensitive vars dropped, benign vars survive.
6. _run_pytest_in_sandbox: real subprocess against known-good /
   known-bad / timeout scenarios. Slower than the pure tests but no
   LLM cost.
7. R5 error branches (missing source, unparseable source, SDK-not-
   installed shim, 6-branch _translate_api_error).
8. _build_agent structural test (importorskip("agents")): correct
   Agent shape with 3 tools.
"""

from __future__ import annotations

# IMPORTANT: import openai-agents SDK BEFORE inserting the workspace
# `agents/` dir on sys.path. The workspace `agents/` is a namespace
# package that would shadow the pip-installed `agents` SDK otherwise
# (see agent #04's L-13 rationale). Cache the SDK modules in
# sys.modules first, then do the sys.path insert.
try:
    import agents as _openai_agents_sdk  # noqa: F401 -- side effect: caches in sys.modules
except ImportError:
    pass  # SDK not installed; importorskip tests below handle this

import importlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_AGENT_DIR.parent))

_agent_pkg = importlib.import_module("08_test_generator.agent")
_schemas_pkg = importlib.import_module("08_test_generator.schemas")

generate_tests = _agent_pkg.generate_tests
resolve_provider = _agent_pkg.resolve_provider
TestGeneratorError = _agent_pkg.TestGeneratorError
GenerationAttempt = _agent_pkg.GenerationAttempt
_translate_api_error = _agent_pkg._translate_api_error
_list_public_symbols_impl = _agent_pkg._list_public_symbols_impl
_check_syntax_impl = _agent_pkg._check_syntax_impl
_parse_pytest_summary = _agent_pkg._parse_pytest_summary
_sanitize_env = _agent_pkg._sanitize_env
_run_pytest_in_sandbox = _agent_pkg._run_pytest_in_sandbox
_count_test_functions = _agent_pkg._count_test_functions
SUPPORTED_PROVIDERS = _agent_pkg.SUPPORTED_PROVIDERS
GeneratedTest = _schemas_pkg.GeneratedTest
TestExecutionResult = _schemas_pkg.TestExecutionResult

_SAMPLE_MODULE = _AGENT_DIR / "examples" / "sample_module.py"


# ---------- 1. Mock path ----------


def test_mock_returns_valid_generated_test(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    result = generate_tests(_SAMPLE_MODULE)
    assert isinstance(result, GeneratedTest)
    assert result.all_passing is True
    assert result.tests_added >= 1
    assert result.final_result.tests_passed >= 1


def test_mock_echoes_source_path_into_test_code(monkeypatch):
    """The mock must include the source_path in its output so a future
    refactor that makes mock output constant regardless of input
    surfaces at test time. Same anti-refactor convention as agent #01."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    result_a = generate_tests(Path("some/module_a.py"))
    result_b = generate_tests(Path("other/module_b.py"))
    assert "module_a" in result_a.test_code
    assert "module_b" in result_b.test_code


def test_mock_provider_kwarg_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    result = generate_tests(_SAMPLE_MODULE, provider="mock")
    assert result.all_passing is True


# ---------- 2. Schema validation ----------


def test_all_passing_true_requires_final_result_evidence():
    """all_passing=True with a final_result showing failures MUST be
    rejected at validation time. Guards against a prompt change that
    lets the model hallucinate success."""
    with pytest.raises(ValidationError):
        GeneratedTest(
            target_module="x",
            test_code="def test_foo(): pass",
            tests_added=1,
            iterations_used=1,
            all_passing=True,  # LIE
            final_result=TestExecutionResult(
                exit_code=1,
                stdout="1 failed",
                stderr="",
                timed_out=False,
                tests_passed=0,
                tests_failed=1,
            ),
        )


def test_all_passing_false_auto_corrected_when_evidence_shows_success():
    """A conservative model saying False when the run actually passed
    is a smaller error than the reverse -- the validator flips it to
    True rather than raising, so we don't reject an otherwise-good
    result over model timidity."""
    result = GeneratedTest(
        target_module="x",
        test_code="def test_foo(): pass",
        tests_added=1,
        iterations_used=1,
        all_passing=False,  # conservative
        final_result=TestExecutionResult(
            exit_code=0,
            stdout="1 passed",
            stderr="",
            timed_out=False,
            tests_passed=1,
            tests_failed=0,
        ),
    )
    assert result.all_passing is True


def test_all_passing_true_with_timeout_is_rejected():
    """Even if tests_failed is 0, a timeout means the run didn't
    finish -- all_passing=True in that case is a lie."""
    with pytest.raises(ValidationError):
        GeneratedTest(
            target_module="x",
            test_code="def test_foo(): pass",
            tests_added=1,
            iterations_used=1,
            all_passing=True,
            final_result=TestExecutionResult(
                exit_code=-1,
                stdout="",
                stderr="timeout",
                timed_out=True,
                tests_passed=0,
                tests_failed=0,
            ),
        )


def test_iterations_used_must_be_at_least_one():
    with pytest.raises(ValidationError):
        GeneratedTest(
            target_module="x",
            test_code="",
            tests_added=0,
            iterations_used=0,
            all_passing=False,
            final_result=TestExecutionResult(exit_code=-1, timed_out=False),
        )


# ---------- 3. AST helpers ----------


def test_list_public_symbols_finds_top_level_functions_and_classes():
    src = "def foo():\n    pass\n\nclass Bar:\n    pass\n"
    assert set(_list_public_symbols_impl(src)) == {"foo", "Bar"}


def test_list_public_symbols_excludes_private_names():
    src = "def _private():\n    pass\n\ndef public():\n    pass\n"
    assert _list_public_symbols_impl(src) == ["public"]


def test_list_public_symbols_excludes_dunders():
    src = "def __magic__():\n    pass\n\ndef normal():\n    pass\n"
    assert _list_public_symbols_impl(src) == ["normal"]


def test_list_public_symbols_ignores_nested_defs():
    src = "def outer():\n    def inner():\n        pass\n    return inner\n"
    # Only 'outer' is top-level; nested 'inner' must not appear.
    assert _list_public_symbols_impl(src) == ["outer"]


def test_list_public_symbols_returns_empty_on_unparseable():
    """Broken syntax returns [] so the model gets a signal to fix
    syntax first, rather than a stacktrace."""
    assert _list_public_symbols_impl("def broken(:\n    pass") == []


def test_list_public_symbols_empty_file():
    assert _list_public_symbols_impl("") == []


def test_check_syntax_ok_for_valid_code():
    assert _check_syntax_impl("def test_foo():\n    assert True\n") == "OK"


def test_check_syntax_reports_line_number_for_bad_code():
    result = _check_syntax_impl("def broken(:\n    pass")
    assert "SyntaxError" in result
    assert "line" in result.lower()


def test_check_syntax_handles_empty_string():
    # Empty parses as an empty module, which is valid.
    assert _check_syntax_impl("") == "OK"


# ---------- 4. _parse_pytest_summary ----------


def test_parse_pytest_summary_all_passed():
    stdout = "===== 5 passed in 0.12s ====="
    assert _parse_pytest_summary(stdout) == (5, 0)


def test_parse_pytest_summary_mixed():
    stdout = "===== 3 passed, 2 failed in 0.15s ====="
    assert _parse_pytest_summary(stdout) == (3, 2)


def test_parse_pytest_summary_all_failed():
    stdout = "===== 4 failed in 0.10s ====="
    assert _parse_pytest_summary(stdout) == (0, 4)


def test_parse_pytest_summary_no_tests_ran():
    stdout = "===== no tests ran in 0.01s ====="
    assert _parse_pytest_summary(stdout) == (0, 0)


def test_parse_pytest_summary_collection_error():
    """When pytest fails to collect (e.g. import error), the summary
    line is often absent or unusual. Should return (0, 0), not raise."""
    stdout = "ImportError while loading conftest\nsome traceback\n"
    assert _parse_pytest_summary(stdout) == (0, 0)


# ---------- 5. _sanitize_env ----------


def test_sanitize_env_drops_api_key_variants(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-example")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-example2")
    monkeypatch.setenv("MY_CUSTOM_TOKEN", "secret")
    result = _sanitize_env()
    assert "OPENAI_API_KEY" not in result
    assert "ANTHROPIC_API_KEY" not in result
    assert "MY_CUSTOM_TOKEN" not in result


def test_sanitize_env_preserves_path():
    """PATH must survive the sanitizer so the subprocess can still
    find sys.executable's deps (regression guard against over-eager
    filtering that would break pytest's ability to run in the
    sandbox)."""
    # PATH is essentially always set on Windows/Mac/Linux.
    result = _sanitize_env()
    # At least one benign var should always survive. PATH is a fair proxy.
    assert "PATH" in result or "Path" in result  # Windows uses "Path"


def test_sanitize_env_drops_aws_vars(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAxxx")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sec")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "tok")
    result = _sanitize_env()
    assert "AWS_ACCESS_KEY_ID" not in result
    assert "AWS_SECRET_ACCESS_KEY" not in result
    assert "AWS_SESSION_TOKEN" not in result


def test_sanitize_env_case_insensitive(monkeypatch):
    """A lowercased env var name still gets caught by the substring
    match (upper-cased before comparing)."""
    monkeypatch.setenv("my_api_key", "leaky")
    result = _sanitize_env()
    assert "my_api_key" not in result


# ---------- 6. Real subprocess sandbox ----------


_GOOD_SOURCE = "def add(a, b):\n    return a + b\n"
_GOOD_TEST = (
    "from sample import add\n\n"
    "def test_add_positive():\n    assert add(2, 3) == 5\n\n"
    "def test_add_zero():\n    assert add(0, 0) == 0\n"
)
_BAD_TEST = (
    "from sample import add\n\n"
    "def test_wrong():\n    assert add(2, 3) == 99  # deliberately wrong\n"
)
_TIMEOUT_TEST = (
    "import time\n\n"
    "def test_hangs():\n    time.sleep(60)\n"
)


def test_sandbox_runs_passing_tests_cleanly():
    """Real subprocess. Should exit 0 with 2 passed, 0 failed."""
    result = _run_pytest_in_sandbox(
        source_code=_GOOD_SOURCE,
        source_module_name="sample",
        test_code=_GOOD_TEST,
        timeout_s=15.0,
    )
    assert result.exit_code == 0
    assert result.tests_passed == 2
    assert result.tests_failed == 0
    assert result.timed_out is False


def test_sandbox_reports_failing_tests():
    """Real subprocess. Should exit non-zero with 1 failed."""
    result = _run_pytest_in_sandbox(
        source_code=_GOOD_SOURCE,
        source_module_name="sample",
        test_code=_BAD_TEST,
        timeout_s=15.0,
    )
    assert result.exit_code != 0
    assert result.tests_failed == 1
    assert result.timed_out is False
    # The traceback should mention the mismatch so the model has
    # something to react to.
    assert "assert" in result.stdout.lower() or "assertionerror" in result.stdout.lower()


def test_sandbox_hits_timeout_and_returns_timed_out_true():
    """Real subprocess. Should exceed 2s wall-clock timeout, come
    back with timed_out=True and exit_code=-1."""
    result = _run_pytest_in_sandbox(
        source_code="",  # no source needed; the test just sleeps
        source_module_name="unused",
        test_code=_TIMEOUT_TEST,
        timeout_s=2.0,
    )
    assert result.timed_out is True
    assert result.exit_code == -1
    assert "TIMEOUT" in result.stderr


def test_sandbox_env_lacks_sensitive_vars(monkeypatch):
    """End-to-end: a sensitive env var set in the parent process must
    not be visible to the subprocess. Verifies the sanitize hook fires
    in the real _run_pytest_in_sandbox call, not just in isolation."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    test_code = (
        "import os\n\n"
        "def test_no_leaked_key():\n"
        "    assert 'OPENAI_API_KEY' not in os.environ\n"
    )
    result = _run_pytest_in_sandbox(
        source_code="",
        source_module_name="unused",
        test_code=test_code,
        timeout_s=15.0,
    )
    assert result.exit_code == 0
    assert result.tests_passed == 1


# ---------- 7. R5 error branches ----------


def test_resolve_provider_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider() == "openai"


def test_resolve_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


def test_resolve_provider_rejects_anthropic_in_v1(monkeypatch):
    """OpenAI-only in v1; the message must point at LiteLLM as the
    documented swap path so a user who sets LLM_PROVIDER=anthropic
    gets pointed somewhere useful."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="LiteLLM"):
        resolve_provider()


def test_generate_tests_raises_on_missing_source(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    bogus = tmp_path / "does_not_exist.py"
    with pytest.raises(TestGeneratorError, match="not found"):
        generate_tests(bogus)


def test_generate_tests_raises_on_syntax_error_source(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    broken = tmp_path / "broken.py"
    broken.write_text("def whoops(:\n    pass", encoding="utf-8")
    with pytest.raises(TestGeneratorError, match="syntax error"):
        generate_tests(broken)


def test_translate_api_error_class_name_rate_limit():
    class RateLimitError(Exception):
        pass

    out = _translate_api_error(RateLimitError("whatever"))
    assert isinstance(out, TestGeneratorError)
    assert "rate-limited" in out.message.lower() or "overloaded" in out.message.lower()


def test_translate_api_error_class_name_auth():
    class AuthenticationError(Exception):
        pass

    out = _translate_api_error(AuthenticationError("whatever"))
    assert "authentication" in out.message.lower() or "api key" in out.message.lower()


def test_translate_api_error_status_code_429():
    exc = RuntimeError("something")
    exc.status_code = 429  # type: ignore[attr-defined]
    out = _translate_api_error(exc)
    assert "rate-limited" in out.message.lower() or "overloaded" in out.message.lower()


def test_translate_api_error_status_code_401():
    exc = RuntimeError("something")
    exc.status_code = 401  # type: ignore[attr-defined]
    out = _translate_api_error(exc)
    assert "authentication" in out.message.lower() or "api key" in out.message.lower()


def test_translate_api_error_message_fallback_rate_limit():
    out = _translate_api_error(RuntimeError("You hit the rate limit for now"))
    assert "rate-limited" in out.message.lower() or "overloaded" in out.message.lower()


def test_translate_api_error_generic_fallback_preserves_original():
    out = _translate_api_error(RuntimeError("some genuinely unexpected thing"))
    assert "RuntimeError" in out.message
    assert "some genuinely unexpected thing" in out.message


# ---------- 8. _build_agent structural (importorskip) ----------


def test_build_agent_produces_agent_with_three_tools():
    """Regression guard: if someone adds/removes a tool, this test
    fires. The prompt's ReAct strategy references exactly 3 tools;
    silent tool-list drift breaks the prompt's assumptions."""
    pytest.importorskip("agents")
    agent_obj = _agent_pkg._build_agent(
        source_code=_GOOD_SOURCE,
        source_module_name="sample",
        model="gpt-4.1-mini-2025-04-14",
        sandbox_timeout_seconds=10.0,
    )
    assert agent_obj.name == "test-generator-agent"
    assert agent_obj.output_type is GeneratedTest
    assert len(agent_obj.tools) == 3
    tool_names = {t.name for t in agent_obj.tools}
    assert tool_names == {"list_public_symbols", "check_syntax", "execute_test_code"}


def test_build_agent_uses_provided_model():
    pytest.importorskip("agents")
    agent_obj = _agent_pkg._build_agent(
        source_code="",
        source_module_name="x",
        model="gpt-5.4-mini-2026-03-17",
        sandbox_timeout_seconds=10.0,
    )
    assert agent_obj.model == "gpt-5.4-mini-2026-03-17"


# ---------- 9. Sanity ----------


def test_count_test_functions_regex_matches_def_test_at_line_start():
    code = (
        "def test_a(): pass\n"
        "def test_b(): pass\n"
        "def not_a_test(): pass\n"
        "class TestClass:\n"
        "    def test_method(self): pass\n"  # nested; also counted
    )
    assert _count_test_functions(code) == 3


def test_max_turns_exceeded_result_is_valid_generated_test():
    """The synthetic GeneratedTest we return on MaxTurnsExceeded must
    satisfy its own validator; regression guard against a future
    refactor that breaks the tests_added=0 branch of the validator."""
    result = _agent_pkg._max_turns_result(
        source_module_name="foo",
        max_iterations=5,
        reason="test reason",
    )
    assert result.all_passing is False
    assert result.iterations_used == 5
    assert "MaxTurnsExceeded" in result.final_result.stderr


def test_supported_providers_includes_openai_and_ollama():
    """Locks in the two supported provider paths: cloud (openai) and
    local (ollama). Adding another provider requires updating the
    README's provider notes too."""
    assert SUPPORTED_PROVIDERS == ("openai", "ollama")


def test_sample_module_is_parseable_python():
    """Regression guard: the shipped examples/sample_module.py must
    remain valid Python. If someone breaks it, the demo breaks with
    it silently."""
    src = _SAMPLE_MODULE.read_text(encoding="utf-8")
    assert _check_syntax_impl(src) == "OK"
    symbols = _list_public_symbols_impl(src)
    assert "TemperatureConverter" in symbols


# ---------- 10. Post-review hardening ----------


def test_mock_test_code_is_parseable_python(monkeypatch):
    """The mock's test_code used to include a bogus `from ... import
    SomeThing` for a symbol that doesn't exist. A user who copy-pasted
    the mock output got a file that failed on import. Now the mock is
    self-contained (no imports); this test locks that down."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    result = generate_tests(_SAMPLE_MODULE)
    assert _check_syntax_impl(result.test_code) == "OK"
    # A stronger claim: the mock should have no imports at all, since
    # any real symbol from the target could get renamed later.
    assert "import" not in result.test_code


def test_generate_tests_rejects_zero_max_iterations(monkeypatch):
    """Nonsense values must fail fast at the public API rather than
    producing a confusing schema error deep in the call stack."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        generate_tests(_SAMPLE_MODULE, max_iterations=0)


def test_generate_tests_rejects_negative_max_iterations(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        generate_tests(_SAMPLE_MODULE, max_iterations=-1)


def test_generate_tests_rejects_non_positive_sandbox_timeout(monkeypatch):
    """A zero or negative wall-clock timeout would either instantly
    fail every subprocess or (worse) be silently ignored by
    subprocess.run. Reject at the boundary."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="sandbox_timeout_seconds must be > 0"):
        generate_tests(_SAMPLE_MODULE, sandbox_timeout_seconds=0)


def test_sanitize_env_preserves_aws_region(monkeypatch):
    """AWS_REGION and AWS_DEFAULT_REGION are benign region hints, NOT
    credentials. A legit boto3-using test needs them. Regression guard
    against re-widening the AWS pattern to `AWS_` substring match."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAxxx")  # sensitive; must drop
    result = _sanitize_env()
    assert result.get("AWS_REGION") == "us-east-1"
    assert result.get("AWS_DEFAULT_REGION") == "us-west-2"
    assert "AWS_ACCESS_KEY_ID" not in result


def test_sanitize_env_drops_google_and_azure_credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/creds.json")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "abcdef")
    result = _sanitize_env()
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in result
    assert "AZURE_CLIENT_SECRET" not in result


def test_all_passing_true_rejected_when_too_many_todo_agent_markers():
    """Metric-hacking guard: the prompt tells the model it MAY comment
    out failing tests with a `# TODO(agent)` marker. A model that hits
    genuinely-broken tests could comment them all out and claim
    all_passing=True with a hollow suite. Cap at 2 markers; above that
    the run is a lie and the validator catches it."""
    test_code_with_many_todos = (
        "def test_good(): assert True\n"
        "# TODO(agent): could not resolve after 5 iterations\n"
        "# def test_broken_1(): assert something()\n"
        "# TODO(agent): could not resolve after 5 iterations\n"
        "# def test_broken_2(): assert something_else()\n"
        "# TODO(agent): could not resolve after 5 iterations\n"
        "# def test_broken_3(): assert third_thing()\n"
    )
    with pytest.raises(ValidationError, match="TODO.agent."):
        GeneratedTest(
            target_module="x",
            test_code=test_code_with_many_todos,
            tests_added=1,
            iterations_used=5,
            all_passing=True,
            final_result=TestExecutionResult(
                exit_code=0, stdout="1 passed", stderr="", timed_out=False,
                tests_passed=1, tests_failed=0,
            ),
        )


def test_all_passing_true_ok_with_at_most_two_todo_markers():
    """One or two `# TODO(agent)` markers is the honest-partial-progress
    case the prompt allows -- a hard error would be over-aggressive."""
    test_code = (
        "def test_a(): assert True\n"
        "def test_b(): assert True\n"
        "# TODO(agent): could not resolve edge case for negative inputs\n"
    )
    result = GeneratedTest(
        target_module="x",
        test_code=test_code,
        tests_added=2,
        iterations_used=3,
        all_passing=True,
        final_result=TestExecutionResult(
            exit_code=0, stdout="2 passed", stderr="", timed_out=False,
            tests_passed=2, tests_failed=0,
        ),
    )
    assert result.all_passing is True


# --- Phase 2: local-Ollama fallback ---------------------------------------


def test_build_agent_ollama_uses_local_endpoint(monkeypatch):
    """LLM_PROVIDER=ollama swaps the model= arg on Agent from a plain
    string to an OpenAIChatCompletionsModel wrapping an AsyncOpenAI
    client that points at localhost:11434. Capture at the openai
    module level."""
    pytest.importorskip("agents")
    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeClient)

    agent_obj = _agent_pkg._build_agent(
        source_code=_GOOD_SOURCE,
        source_module_name="sample",
        model="gemma4:e4b",
        sandbox_timeout_seconds=10.0,
        provider="ollama",
    )
    assert captured["base_url"].endswith("/v1")
    assert "11434" in captured["base_url"]
    assert captured["api_key"] == "ollama"
    assert agent_obj is not None


def test_translate_api_error_connection_refused_returns_ollama_hint():
    exc = ConnectionError("Connection refused: http://localhost:11434")
    out = _translate_api_error(exc)
    assert "ollama serve" in str(out).lower()
