# real-world-agents

Deployable AI agents for real-world use cases. Each agent solves a real problem, demonstrates a different technique, and is structured to be easy to read and copy into your own project.

## What this repo is

A growing catalog of AI agents. Each one teaches exactly ONE technique — structured extraction, ReAct, multi-agent collaboration, meta-optimization, and so on — applied to a task real people would actually use, not a toy demo. Read the code, copy what you need.

## Project status

**1 of 15+ planned agents shipped.** Agent #01 (receipt/invoice extractor) is live. The roadmap below lists what's coming next — no fixed schedule, agents ship when they're ready.

## Shipped agents

| # | Agent | Technique demonstrated | Framework | Live demo | Real use case |
|---|---|---|---|---|---|
| 01 | [Receipt / invoice extractor](agents/01_receipt_extractor/) | Structured extraction with vision | Instructor + OpenAI/Anthropic/Gemini | — | SMB expense tracking |

## Roadmap

Not yet shipped. Want to build one of these? Open an issue with the ["New agent" template](https://github.com/rajeshm71/real-world-agents/issues/new/choose) first — see [CONTRIBUTING.md](CONTRIBUTING.md) for why (mainly: avoiding wasted work on something that doesn't clear the real-world-use-case or technique-variety bar).

| # | Agent | Technique demonstrated | Framework | Real use case |
|---|---|---|---|---|
| 02 | Contract reviewer | Long-context analysis + flag extraction | Direct Anthropic SDK (no framework) | Legal review — showing when "no framework" is the right choice |
| 03 | Chat with your CSV | Text-to-SQL with retry-on-error loops | LangGraph (state machine) | Non-technical users querying data |
| 04 | Meeting notes → action items | ReAct pattern for extraction + prioritization | OpenAI Agents SDK | Meeting follow-through |
| 05 | Multi-agent research assistant | Multi-agent collaboration (researcher + writer + editor) | CrewAI | Deep-dive research on a topic |
| 06 | Type-safe email triage | Type-safe agent with structured tool calls | PydanticAI | Inbox categorization on sample .eml files |
| 07 | Prompt optimizer over an eval set | Meta-optimization (learn prompts from examples) | DSPy | Improve any agent's prompt automatically |
| 08 | Python test generator | Tool use + iterative refinement | OpenAI Agents SDK + sandbox execution | Auto-generate pytest suites from source code |
| 09 | Screenshot → HTML/CSS | Multimodal component reconstruction | Claude vision + plain Python | Design → code |
| 10 | Policy Q&A over handbook | RAG with citations, on-prem/private | Plain Python + sqlite-vec + Anthropic | HR internal tool, privacy-preserving |
| 11 | Resume screener | Batch processing + ranking with rationale | PydanticAI + parallelism | HR/recruiting |
| 12 | Interview prep agent | Adaptive Q&A generation from a JD | OpenAI Agents SDK | Job seekers |
| 13 | From-scratch agent loop | No framework — build the loop from primitives | Plain Python only | Teach what agent frameworks actually do under the hood |
| 14 | Podcast episode processor | Audio → transcript → chapters + summary | Whisper + Anthropic | Creator tool |
| 15 | Business card / ID extractor | Structured extraction with vision (variant of #01) | Instructor + Anthropic | KYC / contact capture |

Full rationale for this list — including what's deliberately excluded and why — is in [SPEC.md](SPEC.md) §7.

## Quick start

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # set LLM_PROVIDER + the matching API key
cd agents/01_receipt_extractor && uv run python -m agent path/to/receipt.jpg
```

No API key is required to explore the code or run tests — every test suite runs under `LLM_PROVIDER=mock`. A key is only needed for a real (non-mock) call.

## Cost note

Running any agent against a real provider costs real money — every agent's own README documents its per-run cost once measured. Provider and model are fully configurable (OpenAI, Anthropic, or Gemini; see [`.env.example`](.env.example) for every option, including per-provider model overrides). No provider is hardcoded, and no provider is required to develop against — mock mode is the default for CI and local test runs. See [SPEC.md](SPEC.md) §13 for the cost model this project follows.

## How to read this repo

Suggested order, not a requirement:

1. Start with **#01** (receipt extractor) for structured extraction — the most common technique you'll reach for.
2. Once more agents ship: **#13** (from-scratch agent loop) shows what agent frameworks are actually doing under the hood, useful context before diving into framework-specific agents.
3. After that, pick whichever framework you want to learn next — each agent's README explains why that framework fits its specific task.

## How to add your own agent

See [docs/adding_an_agent.md](docs/adding_an_agent.md) for the full checklist, and [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## Design principles

- **Real-world use case first.** Every agent solves a task a real person or business would use in production — not a framework demo. See [SPEC.md](SPEC.md) R1.
- **Technique variety, not framework variety.** No two agents teach the same technique. Framework is an implementation detail. See [SPEC.md](SPEC.md) R3.
- **Simple enough to read in one sitting.** Each agent's own code stays under ~500 lines (excluding UI). If you can't understand it without reading a shared abstraction first, the abstraction shouldn't exist. See [SPEC.md](SPEC.md) R2.

## Non-goals

- No competing with big-brand agent frameworks — we USE them, we don't rebuild them
- No "same task in N frameworks" shootout — see the sibling [rag-recipes](https://github.com/rajeshm71/rag-recipes) project for that formula applied to retrieval
- No autonomous-overlord framing — every agent is a specific-task tool
- No enterprise features in v1 (no SSO, RBAC, audit logs, compliance modules)
- No shared UI framework, no shared observability platform beyond a documented optional HF Spaces deploy recipe

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
