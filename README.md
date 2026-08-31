# real-world-agents

Deployable AI agents for real-world use cases. Each agent solves a real problem, demonstrates a different technique, and is structured to be easy to read and copy into your own project.

## What this repo is

A growing catalog of AI agents. Each one teaches exactly ONE technique (structured extraction, ReAct, multi-agent collaboration, meta-optimization, and so on), applied to a task real people would actually use. Read the code, copy what you need.

## Project status

**15 agents shipped.** The v1 catalog covers structured extraction (#01, #09, #15), long-context single-pass (#02), state-machine agents (#03), ReAct with grounded tools (#04, #08), multi-agent collaboration (#05), type-safe dependency injection (#06), meta-optimization (#07), RAG with cross-field-verified citations (#10), batch + ranking (#11), the from-scratch agent loop (#13), audio pipelines (#14), and now live-web-search fact-checking with a local LLM (#16). One originally-planned slot (#12: interview prep) remains committed but unshipped; the roadmap table below lists it as such.

## Shipped agents

| # | Agent | Technique demonstrated | Framework | Live demo | Real use case |
|---|---|---|---|---|---|
| 01 | [Receipt / invoice extractor](agents/01_receipt_extractor/) | Structured extraction with vision | Instructor + OpenAI/Anthropic/Gemini | N/A | SMB expense tracking |
| 02 | [Contract reviewer](agents/02_contract_reviewer/) | Long-context single-pass + hand-rolled JSON-validate-retry (no framework) | `common/llm.py` multi-provider (OpenAI default, Anthropic/Gemini switchable) | N/A | Legal-review triage for freelancers / small businesses |
| 03 | [CSV chat](agents/03_csv_chat/) | Text-to-SQL with a LangGraph state machine + retry-on-error loop | LangGraph + `common/llm.py` multi-provider | N/A | Non-technical users querying spreadsheet data in plain English |
| 04 | [Meeting notes → action items](agents/04_meeting_notes/) | ReAct pattern with grounded tools (extract_speakers, extract_dates, score_urgency, verify_excerpt) | OpenAI Agents SDK (OpenAI in v1; LiteLLM swap for Claude/Gemini documented) | N/A | Post-meeting follow-through: structured owner/priority/deadline per item |
| 05 | [Research crew](agents/05_research_crew/) | Multi-agent collaboration (researcher + writer + editor) with Sequential Process + verified source citations | CrewAI (OpenAI in v1; LiteLLM swap documented) | N/A | Deep-dive research on a topic: fact-checked brief with verified sources |
| 06 | [Type-safe email triage](agents/06_email_triage/) | Type-safe `Agent[Deps, Output]` with typed dependency injection into tools | PydanticAI (OpenAI in v1; one-line swap for Anthropic/Gemini) | N/A | Structured inbox triage grounded in your own contacts + important domains |
| 07 | [Prompt optimizer](agents/07_prompt_optimizer/) | Meta-optimization: DSPy's GEPA learns a better prompt from an eval set | DSPy (OpenAI/Anthropic/Gemini via LiteLLM under the hood) | N/A | Auto-tune agent #01's receipt-extraction prompt against its 20-case CORD eval |
| 08 | [Test generator](agents/08_test_generator/) | Tool use + iterative refinement: draft tests, run in a subprocess sandbox, read failures, rewrite | OpenAI Agents SDK + stdlib subprocess sandbox | N/A | Auto-generate pytest suites for small Python modules |
| 09 | [Screenshot to React JSX](agents/09_screenshot_to_jsx/) | Multimodal structured extraction on a code domain (vision -> Pydantic-validated JSX) | Instructor + vision, multi-provider (Anthropic default) | N/A | First-draft React scaffolding from a design mockup or existing UI screenshot |
| 10 | [Policy Q&A over handbook](agents/10_policy_qa/) | RAG with cross-field-verified citations, on-prem/private (embeddings never leave the box) | Plain Python + fastembed (local ONNX) + sqlite-vec (FAISS fallback) + Anthropic | N/A | HR internal Q&A tool, privacy-preserving |
| 11 | [Resume screener](agents/11_resume_screener/) | Batch + ranking with evidence-cross-checked rationale over multi-format inputs (pdf/docx/md/txt) | PydanticAI + OpenAI (subset-schema pattern with caller-injected known fields) | N/A | HR/recruiting first-pass triage with per-candidate evidence and a ranked list |
| 13 | [From-scratch agent loop](agents/13_from_scratch_loop/) | The tool-use agent loop, hand-rolled: what LangGraph / OpenAI Agents SDK / CrewAI / PydanticAI wrap, in one file | Plain Python + raw OpenAI Chat Completions (no framework) | N/A | Codebase Q&A over any local repository |
| 14 | [Podcast episode processor](agents/14_podcast_processor/) | Three-stage audio pipeline: fetch (local or YouTube) -> local Whisper transcript -> LLM-generated chapters + summary + verbatim key quotes | Plain Python + faster-whisper (local ONNX/CT2) + yt-dlp + Anthropic | N/A | Podcast creators auto-generating chapter markers, summaries, and quote pull-outs |
| 15 | [Contact / ID extractor](agents/15_contact_extractor/) | Batch vision extraction + embedding-based dedup + PII-safe redaction (per-field none/hash/redact policy with salted SHA-256) | Instructor + OpenAI vision + fastembed | N/A | KYC / CRM first-pass triage: one photo of a stack of cards -> deduplicated compliance-safe contact list |
| 16 | [Fact-checker](agents/16_fact_checker/) | Local-LLM claim extraction + live web search with rate-limit-aware multi-provider chain + verdict adjudication with verbatim evidence | Plain Python + Ollama (local `llama3.1:8b`) + Tavily / Brave / DDG fallback chain | N/A | Orators / podcasters / writers spot-checking speech, video, transcripts, or text for outdated numbers, misattributed quotes, and untruths before publishing |

## Roadmap

Not yet shipped. Want to build one of these? Open an issue with the ["New agent" template](https://github.com/rajeshm71/real-world-agents/issues/new/choose) first: see [CONTRIBUTING.md](CONTRIBUTING.md) for why (mainly: avoiding wasted work on something that doesn't clear the real-world-use-case or technique-variety bar).

| # | Agent | Technique demonstrated | Framework | Real use case |
|---|---|---|---|---|
| 12 | Interview prep agent | Adaptive Q&A generation from a JD | OpenAI Agents SDK | Job seekers |

## Quick start

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # set LLM_PROVIDER + the matching API key
cd agents/01_receipt_extractor && uv run python -m agent path/to/receipt.jpg
```

No API key is required to explore the code or run tests: every test suite runs under `LLM_PROVIDER=mock`. A key is only needed for a real (non-mock) call.

**No API key at all? Run locally on your own hardware.** Install [Ollama](https://ollama.com/download), pull the default model (`ollama pull gemma4:e4b`), then run any covered agent with `LLM_PROVIDER=ollama`. Slower and less accurate than the paid providers, but zero-cost and works offline. Coverage today: #01, #02, #03, #04, #05, #06, #07, #08, #09, #11, #13, #15, and #16 — all manually verified end-to-end on a local RTX 5050 (8 GB). A few agents override the catalog-wide `gemma4:e4b` default because gemma4:e4b has verified capability gaps on those specific tasks; users can still swap with `--model gemma4:e4b`:

- **#01 receipt_extractor, #07 prompt_optimizer, #09 screenshot_to_jsx, #15 contact_extractor** → default `qwen2.5vl:7b` (vision-capable at 8 GB VRAM; `ollama pull qwen2.5vl:7b`).
- **#08 test_generator** → default `qwen2.5-coder:7b` (multi-turn ReAct tool use + richer test suites; `ollama pull qwen2.5-coder:7b`).
- **#11 resume_screener** → default `qwen2.5:7b` (structured extraction over ~4KB JD+resume prompts; `ollama pull qwen2.5:7b`).

**Vision agents on Ollama need the shipped nothink proxy** because Ollama's OpenAI-compat surface can't raise `num_ctx` per-request, and receipt/UI/card images tokenize to 5-6k. Start `python -m common.ollama_nothink_proxy` in another shell, then set `OLLAMA_HOST=http://127.0.0.1:11435` before running any of #01, #09, #15. Same proxy also unlocks qwen3.x reasoning models (which otherwise blow their token budget in hidden `<think>` blocks).

Anthropic-native agents (#10, #14) still require their cloud provider.

## Cost note

Running any agent against a real provider costs real money: every agent's own README documents its per-run cost once measured. Provider and model are fully configurable (OpenAI, Anthropic, or Gemini; see [`.env.example`](.env.example) for every option, including per-provider model overrides). No provider is hardcoded, and no provider is required to develop against: mock mode is the default for CI and local test runs.

## How to read this repo

Suggested order:

1. Start with **#01** (receipt extractor) for structured extraction: the most common technique you'll reach for.
2. **#13** (from-scratch agent loop) shows what agent frameworks are actually doing under the hood: useful context before diving into framework-specific agents.
3. After that, pick whichever framework you want to learn next: each agent's README explains why that framework fits its specific task.

## How to add your own agent

See [docs/adding_an_agent.md](docs/adding_an_agent.md) for the full checklist, and [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## Design principles

- **Real-world use case first.** Every agent solves a task a real person or business would use in production.
- **Technique variety, not framework variety.** No two agents teach the same technique. Framework is an implementation detail.
- **Simple enough to read in one sitting.** Each agent's own code stays under ~500 lines (excluding UI). If you can't understand it without reading a shared abstraction first, the abstraction shouldn't exist.

## Non-goals

- No competing with big-brand agent frameworks: we USE them, we don't rebuild them
- No "same task in N frameworks" shootout: see the sibling [rag-recipes](https://github.com/rajeshm71/rag-recipes) project for that formula applied to retrieval
- No autonomous-overlord framing: every agent is a specific-task tool
- No enterprise features in v1 (no SSO, RBAC, audit logs, compliance modules)
- No shared UI framework, no shared observability platform beyond a documented optional HF Spaces deploy recipe

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT: see [LICENSE](LICENSE).
