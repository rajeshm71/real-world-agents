# From-scratch agent loop (codebase Q&A, no framework)

Ask a question about a local repository. The agent uses three file-system tools (`list_files`, `read_file`, `grep`), a plain OpenAI Chat Completions call, and a hand-rolled loop to work out the answer. No LangGraph, no OpenAI Agents SDK, no CrewAI, no PydanticAI. One file, ~500 lines, so a reader can trace every side effect that any framework agent in this catalog is actually performing under the hood.

## Verification status

| Path | Status |
|---|---|
| Mock mode (`LLM_PROVIDER=mock`) | Fully covered by 34-test suite |
| Loop with scripted LLM (happy path, no-tool, max-iter, unknown tool, multi-call, malformed args, preamble+calls, dispatch crash) | Fully covered |
| Tools against real file-system (list/read/grep, caps, path-traversal, symlink guard, bad regex) | Fully covered on `tmp_path` |
| Schema validators (all cross-field rules, both directions) | Fully covered |
| Real OpenAI answering (`LLM_PROVIDER=openai` + key) | **Not yet verified against a live API call.** Structural correctness proven via the mock + scripted-LLM tests. Same open-item status as every other agent's first ship. |

## Technique demonstrated

**The tool-use agent loop, hand-rolled.** No framework abstracts the loop away. The reader sees, in one file, every mechanism that LangGraph (agent #03), OpenAI Agents SDK (#04, #08), CrewAI (#05), and PydanticAI (#06) wrap:

```
build system + user messages
for iteration in range(max_iterations):
    response = llm(messages, tools=TOOL_SCHEMAS)
    if response has no tool calls: return response.content
    for each tool call:
        result = dispatch(call)   # or ToolResult(error=...) on failure
        append tool result to messages
return terminated_by='max_iterations'
```

That is the entire pattern. Frameworks add convenience, observability, retry logic, streaming, and multi-agent orchestration on top; they do **not** change the loop. Once you have read this file, the "black box" in every prior framework agent is one page of Python.

Distinct from every earlier agent: this is the only one that ships zero framework. Every prior agent runs the same loop, wrapped in various shapes.

## Why this technique for this use case

Codebase Q&A over a small local repository is a good fit because:

1. Only three tools are needed (`list_files`, `read_file`, `grep`) and each stays under ~30 lines, so tool code does not overwhelm loop code.
2. All side effects are file reads, so a reader can trace what the agent did just by looking at the arguments in the trace.
3. The task is genuinely useful. Point the CLI at any repo you own with a question about it, get an answer that cites files and lines.

Where this technique is NOT the right fit: production agents with SLAs need retry logic, structured logging, streaming responses, and cost controls the frameworks provide out of the box. If you find yourself reinventing those, use a framework -- but come back here first to know what it is wrapping.

## What it does

**`ask(question, repo_root, ...)`** runs the loop and returns an `AgentTrace` with:

- `question` (echoed back)
- `final_answer` (the model's plain-text response, or a max-iterations note)
- `steps` (list of `AgentStep` -- one per iteration that made tool calls; each records `assistant_message`, `tool_calls`, `tool_results`)
- `terminated_by` (`"final_answer"` | `"max_iterations"` | `"error"`)
- `iterations_used` (LLM calls, including the final answer)
- `error_reason` (populated only on `"error"` termination)

Under `LLM_PROVIDER=mock`, `ask` returns a scripted 3-step trace against the shipped sample_repo/ without importing openai. The mock exists so CI, tests, and demos work with zero API key.

## How to run locally

Four commands from a fresh clone (`python -m agent` must run from inside the agent's own directory: `agent` is a submodule of the digit-prefixed `13_from_scratch_loop` package):

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # set OPENAI_API_KEY (or LLM_PROVIDER=mock)
cd agents/13_from_scratch_loop
```

Mock demo (no API key, canned trace against the shipped sample_repo/):

```bash
LLM_PROVIDER=mock uv run python -m agent "where is greet defined?"
```

Real query against the shipped sample_repo/:

```bash
uv run python -m agent "where is greet defined?"
```

Or point at any repository you have locally:

```bash
uv run python -m agent "what does cli.py do?" --repo ~/code/my-project --max-iterations 6
```

Estimated cost per query at gpt-4o-mini pricing: **well under $0.01**. A typical 3-5-step trace over a small repo comes to ~8-12K tokens total (context is re-sent each turn), which at $0.15 / 1M input + $0.60 / 1M output tokens is fractions of a cent. This is an estimate until a live run pins the exact number; the README will be updated with the measured cost when that happens.

## Code walkthrough

Read these in order:

- `schemas.py`: `ToolCall`, `ToolResult`, `AgentStep`, `AgentTrace`. Three cross-field validators on `AgentTrace` enforce the loop's contract (final-answer termination must have an answer; error termination must carry a reason; iteration count lines up with steps).
- `agent.py` (`ask` function): the whole loop, spelled out inline. Every side effect is an LLM call, a tool dispatch, a message append, or a step append. R5 error translation lives immediately below in `_translate_api_error`.
- `agent.py` (`_tool_list_files`, `_tool_read_file`, `_tool_grep` + their JSON schemas): three tools each with their OpenAI tool schema literally next to the Python function, so the reader sees both halves of the contract in one place.
- `agent.py` (`_dispatch`, `_resolve_within`): dispatch converts unknown tools, bad arguments, and tool exceptions into `ToolResult(error=...)` so the loop continues. Path safety resolves every requested path and rejects escapes (including via symlinks) before any read.
- `prompts/system.txt`: the model's rules. Notably short -- tool schemas live in the `tools=` parameter of the API call, not in the prompt, which is the point.

## When to use / When NOT to use

**Use this pattern when:**
- You want to understand what an agent framework is doing before adopting one.
- Your agent has few tools and simple orchestration, and you would rather own the loop than pull in a framework.
- The task fits into one file that any teammate can read in ten minutes.

**Do NOT use this pattern when:**
- You need streaming responses, structured tracing, retry logic, or multi-agent coordination out of the box. Reach for a framework.
- Your tool set is large enough that discovering the right tool is itself a problem (embeddings-over-tools, tool routing).
- You need conversation memory across sessions. This loop is stateless per call by design.

## Where this fails

- **The LLM invents file paths or line numbers.** The prompt tells the model not to, but there is no verifier for the final answer the way `#02`/`#04`/`#10` verify structured outputs against source excerpts. Mitigation: read the printed trace and confirm the paths the agent actually opened. A follow-up could add a post-hoc verifier tool.
- **Model spirals: same tool with same arguments, over and over.** `max_iterations` catches this but you pay for every wasted call. A dedicated cycle detector would cost more code than value for a pedagogy-first agent; the max-iter cap is the fallback.
- **Big repos flood the context.** `list_files` on a 10K-file repo returns the first 200 sorted entries and stops -- the model then has to guess what else exists. Point at narrower directories, or use `grep` first to locate the relevant file.
- **Binary or huge text files.** `read_file` uses `errors="replace"` so binary reads do not crash, but the model will get garbage. `list_files` does not filter by extension; use the `pattern` argument (e.g. `**/*.py`) to steer.
- **No parallel tool calls.** OpenAI supports `parallel_tool_calls=True`; the loop already handles multiple tool calls per step, but the request does not set the flag. One-line change if desired.
- **Symlinks pointing outside the repo are rejected**, which is the right default for safety but will surprise anyone using a repo laid out via symlinks.

## Roadmap (post-v1 improvements)

- Swap in Anthropic Messages (different message shape, same loop).
- Enable `parallel_tool_calls=True`.
- Add a `write_file` tool behind an explicit `--allow-write` flag so the pattern extends to code-modifying agents.
- Add a post-hoc verifier that greps each cited path/line in the final answer against the actual repo state.
