"""From-scratch agent loop: agent #13 of real-world-agents.

Technique demonstrated: **the tool-use agent loop, hand-rolled.** No
framework abstracts the loop away. The reader sees, in one file, every
mechanism that LangGraph (agent #03), OpenAI Agents SDK (#04, #08),
CrewAI (#05), and PydanticAI (#06) wrap:

    build system + user messages
    for iteration in range(max_iterations):
        response = llm(messages, tools=TOOL_SCHEMAS)
        if no tool calls: return response.content as final_answer
        for each tool call:
            result = dispatch(call)
            append tool result to messages
    return terminated_by='max_iterations'

That's the entire pattern. Frameworks add convenience, observability,
retry logic, streaming, and multi-agent orchestration on top -- they
do NOT change the loop.

Use case chosen so the loop does something real: **codebase Q&A** over
a local directory. Three tools -- `list_files`, `read_file`, `grep` --
are enough to answer questions like "where is greet defined?" or
"what does cli.py do?" against any small repository, and each tool
stays under ~30 LOC so the tool code doesn't overwhelm the loop code.

Provider: OpenAI-only in v1, raw Chat Completions with function/tool
calling. The README documents the swap for Anthropic (different message
shape, same loop).

Real error handling per R5:
- Six-branch translator on LLM API errors (class-name -> status ->
  message -> generic), matching agents #02-#10.
- Loop-level failure modes -- unknown tool name, JSON-argument parse
  failure, tool-dispatch exception -- become `ToolResult(error=...)`
  and the loop continues. The model sees the failure and can adapt,
  which is exactly what a real agent should do.
- Path traversal on the file-system tools is blocked at dispatch time
  by resolving requested paths and rejecting anything outside the
  bound `repo_root`.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

try:
    from .schemas import AgentStep, AgentTrace, ToolCall, ToolResult
except ImportError:
    from schemas import AgentStep, AgentTrace, ToolCall, ToolResult

# --- Constants -------------------------------------------------------------

SUPPORTED_PROVIDERS = ("openai", "ollama")
_DEFAULT_PROVIDER = "openai"
_DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-4o-mini",
    "ollama": "gemma4:e4b",
}
DEFAULT_MAX_ITERATIONS = 10
_LIST_FILES_CAP = 200
_READ_FILE_LINE_CAP = 400
_GREP_MATCH_CAP = 100
_MAX_TOKENS = 1024  # per-call response cap

_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"


# --- Error type ------------------------------------------------------------


@dataclass
class LoopAttempt:
    """Partial state attached to AgentError when a fatal failure hits
    mid-loop. `steps` is the history so far so the caller can inspect
    what the agent tried before the crash."""

    stage: str  # 'llm' | 'setup'
    steps: list[AgentStep]
    iterations_used: int


class AgentError(Exception):
    """Raised only for unrecoverable failures: LLM API error, bad
    configuration, missing dependency. Recoverable failures (bad tool
    name, malformed args, tool exception) do NOT raise -- they become
    ToolResult(error=...) and the loop continues, giving the model a
    chance to correct course."""

    def __init__(self, message: str, partial: LoopAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Provider resolution ---------------------------------------------------


def resolve_provider() -> str:
    """LLM_PROVIDER env var, defaulting to openai. "mock" is handled by
    the caller, not here."""
    provider = os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER).lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}. "
            "Anthropic/Gemini swap is documented in README as a follow-up."
        )
    return provider


@functools.lru_cache(maxsize=1)
def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# --- Tool implementations --------------------------------------------------
#
# Each tool is a plain Python function taking `repo_root` (bound by the
# dispatcher, not visible to the LLM) plus its declared arguments. The
# OpenAI-shaped JSON schema for each tool lives next to it -- reader sees
# the two halves of the same contract in one place.


def _tool_list_files(repo_root: Path, pattern: str = "**/*") -> str:
    """List files under `repo_root` matching `pattern` (glob syntax).
    Returns sorted repo-relative paths, capped at _LIST_FILES_CAP entries
    so the LLM's context stays bounded on large trees."""
    matches: list[str] = []
    for p in sorted(repo_root.rglob(pattern)):
        if not p.is_file():
            continue
        matches.append(str(p.relative_to(repo_root)).replace("\\", "/"))
        if len(matches) >= _LIST_FILES_CAP:
            break
    if not matches:
        return f"(no files matched pattern {pattern!r})"
    header = f"{len(matches)} file(s)"
    if len(matches) == _LIST_FILES_CAP:
        header += f" (capped at {_LIST_FILES_CAP})"
    return header + ":\n" + "\n".join(matches)


_LIST_FILES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": (
            "List files under the repository, optionally filtered by a glob "
            "pattern like '**/*.py' or 'src/*.md'. Returns repo-relative "
            f"paths, capped at {_LIST_FILES_CAP} entries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern; default '**/*' for every file.",
                    "default": "**/*",
                }
            },
            "additionalProperties": False,
        },
    },
}


def _tool_read_file(
    repo_root: Path,
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> str:
    """Read a text file within `repo_root`. Returns `start`-to-`end`
    inclusive lines, prefixed with 1-indexed line numbers. Capped at
    _READ_FILE_LINE_CAP lines per call."""
    target = _resolve_within(repo_root, path)
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolExecutionError(f"could not read {path!r}: {exc}") from exc
    lines = text.splitlines()
    if start_line < 1:
        raise ToolExecutionError(f"start_line must be >= 1, got {start_line}")
    lo = start_line - 1
    hi = min(end_line, len(lines)) if end_line is not None else len(lines)
    hi = min(hi, lo + _READ_FILE_LINE_CAP)
    if lo >= len(lines):
        return f"(file {path!r} has {len(lines)} lines; start_line {start_line} is past the end)"
    numbered = [f"{i + 1:>5} | {lines[i]}" for i in range(lo, hi)]
    header = f"{path} lines {lo + 1}-{hi} of {len(lines)}"
    if hi - lo == _READ_FILE_LINE_CAP and hi < len(lines):
        header += f" (capped at {_READ_FILE_LINE_CAP} lines per read)"
    return header + ":\n" + "\n".join(numbered)


_READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a text file inside the repository. Paths are relative to "
            "the repo root; paths escaping the root are rejected. Line "
            f"numbers are 1-indexed. Capped at {_READ_FILE_LINE_CAP} lines "
            "per call -- read in slices for larger files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within the repo."},
                "start_line": {
                    "type": "integer",
                    "description": "1-indexed first line to include. Default 1.",
                    "default": 1,
                    "minimum": 1,
                },
                "end_line": {
                    "type": ["integer", "null"],
                    "description": "1-indexed last line (inclusive). Null = end of file.",
                    "default": None,
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}


def _tool_grep(repo_root: Path, pattern: str, glob: str = "**/*") -> str:
    """Line-by-line regex search over files matching `glob`. Returns
    `path:line: matched-text` lines, capped at _GREP_MATCH_CAP hits."""
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ToolExecutionError(f"bad regex {pattern!r}: {exc}") from exc
    hits: list[str] = []
    for p in sorted(repo_root.rglob(glob)):
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(p.relative_to(repo_root)).replace("\\", "/")
        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append(f"{rel}:{i}: {line.rstrip()}")
                if len(hits) >= _GREP_MATCH_CAP:
                    break
        if len(hits) >= _GREP_MATCH_CAP:
            break
    if not hits:
        return f"(no matches for {pattern!r} in {glob!r})"
    header = f"{len(hits)} match(es)"
    if len(hits) == _GREP_MATCH_CAP:
        header += f" (capped at {_GREP_MATCH_CAP})"
    return header + ":\n" + "\n".join(hits)


_GREP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Search for a regex pattern across files inside the repository. "
            "Returns 'path:line: matched-line' entries, capped at "
            f"{_GREP_MATCH_CAP} hits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python `re` regex."},
                "glob": {
                    "type": "string",
                    "description": "Glob to narrow files searched. Default '**/*'.",
                    "default": "**/*",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
}


TOOL_SCHEMAS: list[dict] = [_LIST_FILES_SCHEMA, _READ_FILE_SCHEMA, _GREP_SCHEMA]
_TOOLS: dict[str, Callable[..., str]] = {
    "list_files": _tool_list_files,
    "read_file": _tool_read_file,
    "grep": _tool_grep,
}


class ToolExecutionError(Exception):
    """Raised by a tool when its own inputs are bad. The dispatcher
    catches it and converts to `ToolResult(error=...)` so the loop can
    continue and the LLM can retry with different arguments."""


def _resolve_within(repo_root: Path, path: str) -> Path:
    """Resolve `path` under `repo_root` and reject any escape. Symlinks
    are followed via `.resolve()` before the check, so a symlink whose
    target is outside the root is also rejected."""
    candidate = (repo_root / path).resolve()
    root_resolved = repo_root.resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ToolExecutionError(
            f"path {path!r} escapes the repo root {repo_root!r}."
        )
    if not candidate.exists():
        raise ToolExecutionError(f"path {path!r} does not exist under the repo root.")
    if not candidate.is_file():
        raise ToolExecutionError(f"path {path!r} is not a file.")
    return candidate


def _dispatch(call: ToolCall, *, repo_root: Path) -> ToolResult:
    """Run one tool call and always return a ToolResult -- never raise.

    Three failure modes get folded into `ToolResult(error=...)`:
    unknown tool name, ToolExecutionError from the tool itself, and any
    unexpected exception from the tool. The loop continues in all three
    cases so the model can see the failure and try something else."""
    fn = _TOOLS.get(call.tool_name)
    if fn is None:
        return ToolResult(
            call_id=call.call_id,
            content="",
            error=f"unknown tool {call.tool_name!r}. Available: {sorted(_TOOLS)}.",
        )
    try:
        text = fn(repo_root, **call.arguments)
        return ToolResult(call_id=call.call_id, content=text)
    except ToolExecutionError as exc:
        return ToolResult(call_id=call.call_id, content="", error=str(exc))
    except TypeError as exc:
        # Wrong / missing arguments the tool function's signature rejects.
        return ToolResult(
            call_id=call.call_id,
            content="",
            error=f"bad arguments for {call.tool_name!r}: {exc}",
        )
    except Exception as exc:
        return ToolResult(
            call_id=call.call_id,
            content="",
            error=f"{call.tool_name!r} crashed: {type(exc).__name__}: {exc}",
        )


# --- LLM adapter -----------------------------------------------------------


def _call_llm(client: Any, model: str, messages: list[dict], tools: list[dict]) -> Any:
    """Single OpenAI Chat Completions call. Kept tiny on purpose: this
    is the ONLY place that talks to a vendor, so a reader porting the
    loop to Anthropic or Gemini has one function to rewrite."""
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        max_tokens=_MAX_TOKENS,
    )


# --- Public entry point ----------------------------------------------------


def ask(
    question: str,
    *,
    repo_root: str | Path,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    provider: str | None = None,
    model: str | None = None,
    _client: Any = None,
) -> AgentTrace:
    """Run the agent loop for one question.

    The whole loop lives inline here so a reader can trace it end-to-end
    without jumping through helpers. Every side effect is one of: an
    LLM call, a tool dispatch, an append to `messages`, an append to
    `steps`, or a return.
    """
    if not question.strip():
        raise AgentError("question must be non-empty.")
    if max_iterations < 1:
        raise AgentError(f"max_iterations must be >= 1, got {max_iterations}.")

    repo_root_path = Path(repo_root).resolve()
    if not repo_root_path.exists() or not repo_root_path.is_dir():
        raise AgentError(f"repo_root {repo_root!r} is not an existing directory.")

    try:
        resolved = provider or resolve_provider()
    except ValueError as exc:
        raise AgentError(str(exc)) from exc

    if resolved == "mock":
        return _mock_trace(question, repo_root=repo_root_path)

    client = _client if _client is not None else _build_client(resolved)
    resolved_model = model or _DEFAULT_MODEL_BY_PROVIDER[resolved]

    messages: list[dict] = [
        {"role": "system", "content": _load_prompt()},
        {"role": "user", "content": question},
    ]
    steps: list[AgentStep] = []

    for iteration in range(1, max_iterations + 1):
        try:
            response = _call_llm(client, resolved_model, messages, TOOL_SCHEMAS)
        except Exception as exc:
            raise _translate_api_error(
                exc,
                partial=LoopAttempt(stage="llm", steps=steps, iterations_used=iteration - 1),
            ) from exc

        msg = response.choices[0].message
        # Round-trip the assistant message verbatim so tool_call ids stay
        # bound to the same message on the next turn.
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            # Modern OpenAI models sometimes null out content (refusal, or a
            # response shape the SDK returns as list-of-parts). Coerce to a
            # clean str; if nothing usable is there, terminate as error so
            # the caller sees why instead of a downstream ValidationError.
            raw_content = msg.content
            final = raw_content if isinstance(raw_content, str) else ""
            refusal = getattr(msg, "refusal", None)
            if not final.strip() and isinstance(refusal, str) and refusal.strip():
                final = refusal
            if not final.strip():
                return AgentTrace(
                    question=question,
                    final_answer="",
                    steps=steps,
                    terminated_by="error",
                    iterations_used=iteration,
                    error_reason=(
                        "model returned neither tool_calls nor answer text "
                        "(likely a refusal, an empty completion, or a "
                        "response shape the loop does not handle)."
                    ),
                )
            return AgentTrace(
                question=question,
                final_answer=final,
                steps=steps,
                terminated_by="final_answer",
                iterations_used=iteration,
            )

        step_calls: list[ToolCall] = []
        step_results: list[ToolResult] = []
        for slot, tc in enumerate(msg.tool_calls):
            # Coerce name/id defensively so downstream ToolCall construction
            # cannot raise on an empty string the SDK technically permits.
            safe_name = tc.function.name or "<unknown>"
            safe_call_id = tc.id or f"synth_iter{iteration}_slot{slot}"
            try:
                arguments = json.loads(tc.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must decode to a JSON object.")
            except (json.JSONDecodeError, TypeError, ValidationError) as exc:
                # Recoverable: append the malformed call anyway so the
                # loop's contract (one result per call) holds, and tell
                # the model what went wrong.
                parsed = ToolCall(
                    tool_name=safe_name,
                    arguments={},
                    call_id=safe_call_id,
                )
                result = ToolResult(
                    call_id=safe_call_id,
                    content="",
                    error=f"could not parse tool arguments as JSON: {exc}",
                )
            else:
                parsed = ToolCall(
                    tool_name=safe_name,
                    arguments=arguments,
                    call_id=safe_call_id,
                )
                result = _dispatch(parsed, repo_root=repo_root_path)
            step_calls.append(parsed)
            step_results.append(result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": safe_call_id,
                    "content": result.error or result.content,
                }
            )

        # L2: msg.content is typed str|None in the SDK today, but future
        # structured-output shapes could return a non-str. Store only what
        # AgentStep accepts; drop anything else on the floor.
        step_assistant_message = msg.content if isinstance(msg.content, str) else None
        steps.append(
            AgentStep(
                iteration=iteration,
                assistant_message=step_assistant_message,
                tool_calls=step_calls,
                tool_results=step_results,
            )
        )

    return AgentTrace(
        question=question,
        final_answer="(agent hit max_iterations without a final answer)",
        steps=steps,
        terminated_by="max_iterations",
        iterations_used=max_iterations,
    )


def _build_client(provider: str) -> Any:
    """Build the openai-SDK client. `provider="openai"` uses cloud
    OpenAI (needs OPENAI_API_KEY). `provider="ollama"` points the same
    client at the local Ollama server via its OpenAI-compatible
    endpoint (needs `ollama serve` running, no API key)."""
    try:
        import openai
    except ImportError as exc:
        raise AgentError(
            "openai SDK not installed. Run `uv sync --all-packages` from the repo root."
        ) from exc
    if provider == "ollama":
        raw_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        base = raw_host.rstrip("/").removesuffix("/v1")
        return openai.OpenAI(base_url=f"{base}/v1", api_key="ollama")
    return openai.OpenAI()


# --- Error translation (R5) ------------------------------------------------


def _translate_api_error(
    exc: Exception, *, partial: LoopAttempt | None = None
) -> AgentError:
    """Turn an openai SDK exception into a user-facing AgentError.
    Priority: class-name -> status-code -> message-string -> generic.
    Same six-branch shape as agents #02-#10."""
    exc_class_name = type(exc).__name__.lower()
    message_lower = str(exc).lower()
    status = getattr(exc, "status_code", None)
    # Ollama server not running is the single most common failure mode
    # on the local-Ollama path. Surface a clean hint instead of the
    # raw httpx traceback.
    if "connection" in exc_class_name or (
        "refused" in message_lower or "connection refused" in message_lower
    ):
        return AgentError(
            "Ollama connection failed. If you set --provider ollama, "
            "is `ollama serve` running? See https://ollama.com/download "
            "to install, then `ollama pull gemma4:e4b` for the default model.",
            partial=partial,
        )
    if "ratelimiterror" in exc_class_name:
        return _rate_limit_error(partial)
    if "authenticationerror" in exc_class_name or "apikeyerror" in exc_class_name:
        return _auth_error(partial)
    if status == 429:
        return _rate_limit_error(partial)
    if status == 401:
        return _auth_error(partial)
    if "rate limit" in message_lower:
        return _rate_limit_error(partial)
    if "authentication" in message_lower or "api key" in message_lower:
        return _auth_error(partial)
    return AgentError(
        f"LLM call failed: {type(exc).__name__}: {exc}. Partial trace attached.",
        partial=partial,
    )


def _rate_limit_error(partial: LoopAttempt | None) -> AgentError:
    return AgentError(
        "OpenAI is rate-limited. Wait a minute and try again.", partial=partial
    )


def _auth_error(partial: LoopAttempt | None) -> AgentError:
    return AgentError(
        "Authentication failed: check that OPENAI_API_KEY is set. See "
        ".env.example at the repo root.",
        partial=partial,
    )


# --- Mock mode -------------------------------------------------------------


def _mock_trace(question: str, *, repo_root: Path) -> AgentTrace:
    """Scripted 3-step trace without importing openai.

    Steps: (1) list_files("**/*.py"), (2) read_file("src/greet.py"),
    (3) final answer that echoes the question length as an
    anti-refactor guard the tests can assert on."""
    step1_call = ToolCall(
        tool_name="list_files",
        arguments={"pattern": "**/*.py"},
        call_id="mock_call_1",
    )
    step1_result = _dispatch(step1_call, repo_root=repo_root)
    step2_call = ToolCall(
        tool_name="read_file",
        arguments={"path": "src/greet.py"},
        call_id="mock_call_2",
    )
    step2_result = _dispatch(step2_call, repo_root=repo_root)
    steps = [
        AgentStep(
            iteration=1,
            assistant_message="Let me see what Python files this repo has.",
            tool_calls=[step1_call],
            tool_results=[step1_result],
        ),
        AgentStep(
            iteration=2,
            assistant_message="Opening the greet module to find the definition.",
            tool_calls=[step2_call],
            tool_results=[step2_result],
        ),
    ]
    return AgentTrace(
        question=question,
        final_answer=(
            f"[MOCK answer for question of length {len(question)}] "
            "The `greet` function is defined in src/greet.py."
        ),
        steps=steps,
        terminated_by="final_answer",
        iterations_used=3,
    )


# --- CLI ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="from-scratch-loop",
        description=(
            "Hand-rolled tool-use agent loop over OpenAI Chat Completions. "
            "Set LLM_PROVIDER=mock for a scripted demo, or supply "
            "OPENAI_API_KEY for real answers."
        ),
    )
    parser.add_argument("question", type=str, help="Question about the repository.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).parent / "examples" / "sample_repo",
        help="Path to the repository to inspect. Defaults to the shipped sample_repo.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--provider", choices=(*SUPPORTED_PROVIDERS, "mock"), default=None
    )
    args = parser.parse_args()

    start = time.perf_counter()
    try:
        trace = ask(
            args.question,
            repo_root=args.repo,
            max_iterations=args.max_iterations,
            provider=args.provider,
            model=args.model,
        )
    except AgentError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    for step in trace.steps:
        for call, result in zip(step.tool_calls, step.tool_results, strict=True):
            args_short = json.dumps(call.arguments)[:80]
            outcome = f"error: {result.error}" if result.error else f"{len(result.content)} chars"
            print(f"[step {step.iteration}] {call.tool_name}({args_short}) -> {outcome}")

    out_path = Path(__file__).parent / "last_run.json"
    out_path.write_text(
        json.dumps(trace.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    print()
    print(f"Question:      {trace.question}")
    print(f"Terminated by: {trace.terminated_by}")
    print(f"Iterations:    {trace.iterations_used}")
    print(f"Wall time:     {elapsed:.1f}s")
    print()
    print("Answer:")
    print(trace.final_answer)
    print()
    print(f"Full trace written to {out_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
