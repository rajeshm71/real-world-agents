"""Smoke tests for the from-scratch agent loop.

Zero real OpenAI calls (R8): every test either runs the mock path or
supplies a stub _client whose `chat.completions.create` returns a
scripted response. Zero real network. Tool tests run against `tmp_path`
directory trees.

Sections:
1. Mock path (round-trip + question-echo anti-refactor guard).
2. Loop with scripted LLM stub (8 tests: happy path, no-tool, max-iter,
   unknown tool, multi-call, malformed args, preamble+calls, dispatch
   crash).
3. Tools with real file-system (10 tests: list/read/grep, caps,
   path-traversal, symlink guard, bad regex).
4. Schema validators (5 tests: cross-field rules, all branches).
5. R5 error translator (6 tests: 6-branch coverage).
6. Constants + sample_repo + tool-schema sanity (5 tests).
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_AGENT_DIR.parent))

_agent = importlib.import_module("13_from_scratch_loop.agent")
_schemas = importlib.import_module("13_from_scratch_loop.schemas")

ask = _agent.ask
resolve_provider = _agent.resolve_provider
AgentError = _agent.AgentError
LoopAttempt = _agent.LoopAttempt
ToolExecutionError = _agent.ToolExecutionError
_dispatch = _agent._dispatch
_resolve_within = _agent._resolve_within
_translate_api_error = _agent._translate_api_error
_tool_list_files = _agent._tool_list_files
_tool_read_file = _agent._tool_read_file
_tool_grep = _agent._tool_grep
_mock_trace = _agent._mock_trace
SUPPORTED_PROVIDERS = _agent.SUPPORTED_PROVIDERS
DEFAULT_MAX_ITERATIONS = _agent.DEFAULT_MAX_ITERATIONS
TOOL_SCHEMAS = _agent.TOOL_SCHEMAS
_LIST_FILES_CAP = _agent._LIST_FILES_CAP
_READ_FILE_LINE_CAP = _agent._READ_FILE_LINE_CAP
_GREP_MATCH_CAP = _agent._GREP_MATCH_CAP

ToolCall = _schemas.ToolCall
ToolResult = _schemas.ToolResult
AgentStep = _schemas.AgentStep
AgentTrace = _schemas.AgentTrace

_SAMPLE_REPO = _AGENT_DIR / "examples" / "sample_repo"


# --- Helpers ---------------------------------------------------------------


def _stub_message(*, content=None, tool_calls=None):
    """Build a duck-typed stand-in for an openai ChatCompletionMessage.
    A `types.SimpleNamespace` is enough since our code only touches
    `.content`, `.tool_calls`, and `.model_dump()`."""

    class _Msg(types.SimpleNamespace):
        def model_dump(self, exclude_none=False):
            data = {"role": "assistant", "content": self.content}
            if self.tool_calls:
                data["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in self.tool_calls
                ]
            if exclude_none:
                data = {k: v for k, v in data.items() if v is not None}
            return data

    return _Msg(content=content, tool_calls=tool_calls)


def _stub_tool_call(name, arguments, call_id):
    """Duck-typed stand-in for openai's ChatCompletionMessageToolCall."""
    return types.SimpleNamespace(
        id=call_id,
        type="function",
        function=types.SimpleNamespace(
            name=name,
            arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
        ),
    )


def _stub_response(msg):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class ScriptedClient:
    """Replays a fixed list of responses across successive `create` calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

        class _Chat:
            def __init__(self, outer):
                self._outer = outer
                self.completions = self

            def create(self, **kwargs):
                self._outer.call_count += 1
                if not self._outer._responses:
                    raise AssertionError("ScriptedClient ran out of responses")
                return self._outer._responses.pop(0)

        self.chat = _Chat(self)


# --- 1. Mock path ----------------------------------------------------------


def test_mock_returns_valid_trace(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    trace = ask("where is greet defined?", repo_root=_SAMPLE_REPO)
    assert isinstance(trace, AgentTrace)
    assert trace.terminated_by == "final_answer"
    assert trace.iterations_used == 3
    assert len(trace.steps) == 2


def test_mock_echoes_question_length(monkeypatch):
    """Anti-refactor guard: prove the mock actually saw its input."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    question = "a very specific question with a particular length here"
    trace = ask(question, repo_root=_SAMPLE_REPO)
    assert f"length {len(question)}" in trace.final_answer


def test_mock_does_not_import_openai(monkeypatch):
    """Mock must be reachable in an env with no openai installed."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setitem(sys.modules, "openai", None)  # sabotage the import
    trace = ask("hi", repo_root=_SAMPLE_REPO)
    assert trace.terminated_by == "final_answer"


# --- 2. Loop with scripted LLM stub ---------------------------------------


def test_loop_happy_path_three_steps(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    responses = [
        _stub_response(_stub_message(
            content=None,
            tool_calls=[_stub_tool_call("list_files", {"pattern": "**/*.py"}, "c1")],
        )),
        _stub_response(_stub_message(
            content=None,
            tool_calls=[_stub_tool_call("read_file", {"path": "src/greet.py"}, "c2")],
        )),
        _stub_response(_stub_message(
            content="greet is at src/greet.py line 6.", tool_calls=None
        )),
    ]
    client = ScriptedClient(responses)
    trace = ask("where is greet?", repo_root=_SAMPLE_REPO, _client=client)
    assert trace.terminated_by == "final_answer"
    assert trace.iterations_used == 3
    assert len(trace.steps) == 2
    assert client.call_count == 3


def test_loop_immediate_answer_no_tool(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    responses = [
        _stub_response(_stub_message(content="42", tool_calls=None)),
    ]
    trace = ask("what is 6*7?", repo_root=_SAMPLE_REPO, _client=ScriptedClient(responses))
    assert trace.terminated_by == "final_answer"
    assert trace.iterations_used == 1
    assert trace.steps == []
    assert trace.final_answer == "42"


def test_loop_hits_max_iterations(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    # A model that keeps calling list_files forever.
    responses = [
        _stub_response(_stub_message(
            content=None,
            tool_calls=[_stub_tool_call("list_files", {}, f"c{i}")],
        ))
        for i in range(5)
    ]
    trace = ask(
        "loop forever?", repo_root=_SAMPLE_REPO, max_iterations=3,
        _client=ScriptedClient(responses),
    )
    assert trace.terminated_by == "max_iterations"
    assert trace.iterations_used == 3
    assert len(trace.steps) == 3


def test_loop_unknown_tool_becomes_error_result_and_continues(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    responses = [
        _stub_response(_stub_message(
            content=None,
            tool_calls=[_stub_tool_call("delete_everything", {}, "c1")],
        )),
        _stub_response(_stub_message(content="I cannot do that.", tool_calls=None)),
    ]
    trace = ask("nuke it", repo_root=_SAMPLE_REPO, _client=ScriptedClient(responses))
    assert trace.terminated_by == "final_answer"
    assert trace.steps[0].tool_results[0].error is not None
    assert "unknown tool" in trace.steps[0].tool_results[0].error


def test_loop_multi_tool_call_in_one_step(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    responses = [
        _stub_response(_stub_message(
            content=None,
            tool_calls=[
                _stub_tool_call("list_files", {}, "c1"),
                _stub_tool_call("read_file", {"path": "README.md"}, "c2"),
            ],
        )),
        _stub_response(_stub_message(content="done", tool_calls=None)),
    ]
    trace = ask("both", repo_root=_SAMPLE_REPO, _client=ScriptedClient(responses))
    assert len(trace.steps) == 1
    assert len(trace.steps[0].tool_calls) == 2
    assert len(trace.steps[0].tool_results) == 2
    assert trace.steps[0].tool_calls[0].call_id == "c1"
    assert trace.steps[0].tool_calls[1].call_id == "c2"


def test_loop_malformed_arguments_become_error_result(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    bad_tc = _stub_tool_call("read_file", "not json at all {{", "c1")
    responses = [
        _stub_response(_stub_message(content=None, tool_calls=[bad_tc])),
        _stub_response(_stub_message(content="giving up", tool_calls=None)),
    ]
    trace = ask("read", repo_root=_SAMPLE_REPO, _client=ScriptedClient(responses))
    assert trace.terminated_by == "final_answer"
    assert "could not parse tool arguments" in trace.steps[0].tool_results[0].error


def test_loop_captures_preamble_alongside_tool_calls(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    responses = [
        _stub_response(_stub_message(
            content="Let me check the files first.",
            tool_calls=[_stub_tool_call("list_files", {}, "c1")],
        )),
        _stub_response(_stub_message(content="Done.", tool_calls=None)),
    ]
    trace = ask("check", repo_root=_SAMPLE_REPO, _client=ScriptedClient(responses))
    assert trace.steps[0].assistant_message == "Let me check the files first."


def test_loop_tool_dispatch_exception_does_not_crash_loop(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    def _boom(repo_root, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(_agent._TOOLS, "list_files", _boom)
    responses = [
        _stub_response(_stub_message(
            content=None,
            tool_calls=[_stub_tool_call("list_files", {}, "c1")],
        )),
        _stub_response(_stub_message(content="recovered", tool_calls=None)),
    ]
    trace = ask("test", repo_root=_SAMPLE_REPO, _client=ScriptedClient(responses))
    assert trace.terminated_by == "final_answer"
    assert "kaboom" in trace.steps[0].tool_results[0].error


# --- 3. Tools with real file-system ---------------------------------------


def _tiny_tree(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (root / "src" / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    (root / "notes.md").write_text("# Notes\n\nBeta lives in b.py.\n", encoding="utf-8")


def test_list_files_returns_sorted_relative_paths(tmp_path):
    _tiny_tree(tmp_path)
    out = _tool_list_files(tmp_path, "**/*")
    lines = out.splitlines()
    assert "notes.md" in lines
    assert "src/a.py" in lines
    assert "src/b.py" in lines
    # Sorted (case-sensitive)
    files_only = [line for line in lines if line.endswith((".py", ".md"))]
    assert files_only == sorted(files_only)


def test_list_files_glob_filters(tmp_path):
    _tiny_tree(tmp_path)
    out = _tool_list_files(tmp_path, "**/*.py")
    assert "src/a.py" in out
    assert "notes.md" not in out


def test_list_files_caps_output(tmp_path):
    for i in range(_LIST_FILES_CAP + 20):
        (tmp_path / f"f{i:03d}.txt").write_text("x", encoding="utf-8")
    out = _tool_list_files(tmp_path)
    assert f"capped at {_LIST_FILES_CAP}" in out


def test_read_file_within_root(tmp_path):
    _tiny_tree(tmp_path)
    out = _tool_read_file(tmp_path, "src/a.py")
    assert "def alpha" in out
    assert "1 |" in out  # 1-indexed line number


def test_read_file_rejects_outside_root(tmp_path):
    _tiny_tree(tmp_path)
    with pytest.raises(ToolExecutionError, match="escapes the repo root"):
        _tool_read_file(tmp_path, "../../etc/passwd")


def test_read_file_symlink_outside_root_is_rejected(tmp_path):
    _tiny_tree(tmp_path)
    outside = tmp_path.parent / "outside_target.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not supported on this platform / user")
    with pytest.raises(ToolExecutionError, match="escapes"):
        _tool_read_file(tmp_path, "link.txt")


def test_read_file_line_slice(tmp_path):
    (tmp_path / "long.py").write_text(
        "\n".join(f"line{i}" for i in range(1, 21)) + "\n", encoding="utf-8"
    )
    out = _tool_read_file(tmp_path, "long.py", start_line=5, end_line=8)
    assert "line5" in out and "line8" in out
    assert "line4" not in out and "line9" not in out


def test_read_file_line_cap(tmp_path):
    (tmp_path / "huge.py").write_text(
        "\n".join(f"line{i}" for i in range(1, _READ_FILE_LINE_CAP + 200)) + "\n",
        encoding="utf-8",
    )
    out = _tool_read_file(tmp_path, "huge.py")
    assert f"capped at {_READ_FILE_LINE_CAP}" in out


def test_grep_finds_matches(tmp_path):
    _tiny_tree(tmp_path)
    out = _tool_grep(tmp_path, r"def \w+")
    assert "src/a.py:1:" in out
    assert "def alpha" in out


def test_grep_bad_regex_returns_error(tmp_path):
    with pytest.raises(ToolExecutionError, match="bad regex"):
        _tool_grep(tmp_path, "[unclosed")


# --- 4. Schema validators -------------------------------------------------


def test_step_results_must_match_calls_count():
    with pytest.raises(ValidationError, match="tool_calls but .* tool_results"):
        AgentStep(
            iteration=1,
            assistant_message=None,
            tool_calls=[ToolCall(tool_name="list_files", arguments={}, call_id="c1")],
            tool_results=[],
        )


def test_step_results_must_match_calls_by_id():
    with pytest.raises(ValidationError, match="out of order"):
        AgentStep(
            iteration=1,
            assistant_message=None,
            tool_calls=[ToolCall(tool_name="list_files", arguments={}, call_id="c1")],
            tool_results=[ToolResult(call_id="c99", content="x")],
        )


def test_trace_final_answer_requires_nonempty_answer():
    with pytest.raises(ValidationError, match="non-empty final_answer"):
        AgentTrace(
            question="q",
            final_answer="   ",
            steps=[],
            terminated_by="final_answer",
            iterations_used=1,
        )


def test_trace_error_requires_reason():
    with pytest.raises(ValidationError, match="non-empty error_reason"):
        AgentTrace(
            question="q",
            final_answer="",
            steps=[],
            terminated_by="error",
            iterations_used=0,
        )


def test_trace_max_iterations_requires_matching_step_count():
    with pytest.raises(ValidationError, match="iterations_used"):
        AgentTrace(
            question="q",
            final_answer="",
            steps=[],
            terminated_by="max_iterations",
            iterations_used=5,
        )


# --- 5. R5 error translator -----------------------------------------------


def test_translator_class_name_rate_limit():
    class RateLimitError(Exception):
        pass
    err = _translate_api_error(RateLimitError("slow down"))
    assert "rate-limited" in err.message


def test_translator_class_name_auth():
    class AuthenticationError(Exception):
        pass
    err = _translate_api_error(AuthenticationError("bad key"))
    assert "OPENAI_API_KEY" in err.message


def test_translator_status_429():
    exc = RuntimeError("some server error")
    exc.status_code = 429
    err = _translate_api_error(exc)
    assert "rate-limited" in err.message


def test_translator_status_401():
    exc = RuntimeError("some server error")
    exc.status_code = 401
    err = _translate_api_error(exc)
    assert "OPENAI_API_KEY" in err.message


def test_translator_message_rate_limit():
    err = _translate_api_error(RuntimeError("Rate limit exceeded for gpt-4o-mini"))
    assert "rate-limited" in err.message


def test_translator_message_auth():
    err = _translate_api_error(RuntimeError("Invalid API key provided: sk-***"))
    assert "OPENAI_API_KEY" in err.message


def test_translator_generic_fallthrough():
    err = _translate_api_error(RuntimeError("something else weird"))
    assert "LLM call failed" in err.message


# --- 7. Post-review hardening (H1, H2, M1) --------------------------------


def test_empty_final_content_terminates_as_error(monkeypatch):
    """H1: model returns neither tool_calls nor content -- must terminate
    as `error` with a reason, not raise a ValidationError from AgentTrace."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    responses = [_stub_response(_stub_message(content=None, tool_calls=None))]
    trace = ask("hello", repo_root=_SAMPLE_REPO, _client=ScriptedClient(responses))
    assert trace.terminated_by == "error"
    assert "refusal" in (trace.error_reason or "")


def test_refusal_field_used_as_final_answer(monkeypatch):
    """H1: if `msg.refusal` carries the text, treat it as the final answer
    rather than silently dropping it."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    msg = _stub_message(content=None, tool_calls=None)
    msg.refusal = "I cannot help with that request."
    trace = ask("harm", repo_root=_SAMPLE_REPO, _client=ScriptedClient([_stub_response(msg)]))
    assert trace.terminated_by == "final_answer"
    assert "cannot help" in trace.final_answer


def test_empty_tool_name_and_id_do_not_crash_loop(monkeypatch):
    """H2: an OpenAI SDK response with empty tool_call name/id must not
    escape the recoverable-error branch as a ValidationError."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    bad_tc = _stub_tool_call("", "not json", "")
    responses = [
        _stub_response(_stub_message(content=None, tool_calls=[bad_tc])),
        _stub_response(_stub_message(content="recovered", tool_calls=None)),
    ]
    trace = ask("x", repo_root=_SAMPLE_REPO, _client=ScriptedClient(responses))
    assert trace.terminated_by == "final_answer"
    assert trace.steps[0].tool_calls[0].tool_name == "<unknown>"
    assert trace.steps[0].tool_calls[0].call_id.startswith("synth_iter1_slot0")


def test_bad_provider_raises_agent_error_not_value_error(monkeypatch):
    """M1: LLM_PROVIDER=bogus must surface as AgentError so the CLI's
    error handler catches it cleanly."""
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(AgentError, match="Unknown LLM_PROVIDER"):
        ask("hi", repo_root=_SAMPLE_REPO)


def test_non_string_content_is_coerced_to_none(monkeypatch):
    """L2: future SDK could hand back a list-of-parts. Store None instead
    of tripping the AgentStep validator."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    msg = _stub_message(content=None, tool_calls=[_stub_tool_call("list_files", {}, "c1")])
    msg.content = [{"type": "text", "text": "hi"}]  # simulate list-of-parts
    responses = [
        _stub_response(msg),
        _stub_response(_stub_message(content="done", tool_calls=None)),
    ]
    trace = ask("x", repo_root=_SAMPLE_REPO, _client=ScriptedClient(responses))
    assert trace.steps[0].assistant_message is None


# --- 6. Constants + sample_repo + tool-schema sanity ----------------------


def test_supported_providers_includes_openai_and_ollama():
    assert set(SUPPORTED_PROVIDERS) == {"openai", "ollama"}


def test_build_client_openai_uses_default_endpoint(monkeypatch):
    """`_build_client('openai')` returns a cloud-OpenAI client (no
    base_url override). Uses stub to avoid needing OPENAI_API_KEY."""
    import openai
    captured = {}

    class _StubOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", _StubOpenAI)
    _agent._build_client("openai")
    assert "base_url" not in captured


def test_build_client_ollama_points_at_local_endpoint(monkeypatch):
    """`_build_client('ollama')` returns an openai-SDK client wired
    to `localhost:11434/v1` with a dummy api_key."""
    import openai
    captured = {}

    class _StubOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setattr(openai, "OpenAI", _StubOpenAI)
    _agent._build_client("ollama")
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["api_key"] == "ollama"


def test_default_max_iterations_is_reasonable():
    assert 1 <= DEFAULT_MAX_ITERATIONS <= 20


def test_sample_repo_exists_with_python_files():
    py_files = list(_SAMPLE_REPO.rglob("*.py"))
    assert len(py_files) >= 3


def test_sample_repo_readme_documents_attribution():
    text = (_AGENT_DIR / "examples" / "README.md").read_text(encoding="utf-8")
    assert "self-authored" in text.lower()


def test_tool_schemas_cover_all_three_tools():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == {"list_files", "read_file", "grep"}
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "parameters" in schema["function"]
