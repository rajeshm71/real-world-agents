# real-world-agents — Build Specification

> Feed this file to a coding tool (Cursor, Claude Code, Copilot Workspace, etc.) to build the project. Every section is written as an actionable requirement, not a suggestion. Acceptance criteria are explicit.

---

## 0. Project identity

**Name:** `real-world-agents`
**One-liner:** Deployable AI agents for real-world use cases. Each agent solves a real problem, demonstrates a different technique, and is structured to be easy to read and copy into your own project.
**Audience:** Engineers who want to learn agent-building patterns by seeing them applied to real tasks — not toy demos, not framework tutorials in isolation, but the "here's a real problem, here's the technique that fits it, here's why" pairing.
**License:** MIT
**Python:** 3.11+
**Package manager:** `uv`

### What "deployable" means here (important)

Every agent in this repo is **production-worthy in shape**: real use case, real error handling, real inputs, code you could copy into your own project and put behind a real user. That is what "deployable" means throughout this SPEC.

**"Deployable" does NOT require a hosted live URL.** Live demo URLs on HuggingFace Spaces are an OPTIONAL per-agent add-on (see §11), earning a badge in the main README when present. The bar for shipping an agent is the agent's code and design being production-worthy, not the maintainer running a hosted instance.

If an agent is a toy, a "look at this framework" demo, or an "AI does something impressive" party trick, it does not ship. See §7's "excluded" list for concrete examples of things we chose not to build for exactly this reason.

### Positioning

The agent-examples space is genuinely saturated (Ashish Patel's 500-project repo, half a dozen "monorepo of examples" repos, 300+ awesome-list entries). We are not competing on:

- **Volume** — we don't need 500 agents to be useful
- **Polish** — big-brand cookbooks (OpenAI, Anthropic, Microsoft) win on polish
- **Hosted-URL count** — a live demo per agent is a nice-to-have badge (see §11), not the load-bearing claim

We compete on the combination of **deployable real-world use cases + variety of techniques + simplicity for readers**. Every agent in this repo:

1. **Solves a real task** a person or business would actually use (not "here's a multi-agent debate demo")
2. **Uses a distinctly different technique** from the other agents — implemented with whichever framework fits (Instructor for structured extraction, LangGraph for stateful workflows, CrewAI for multi-agent, DSPy for optimization, plain Python for showing the primitives, etc.). Framework reuse is allowed up to a soft ≤3 ceiling per §17 R3.
3. **Is small enough to read in one sitting** (~200-500 LOC excluding UI) — the reader can understand the whole agent without diagrams
4. **Has a README that explains the technique**, not just "here's what it does"

(Optional per-agent add-ons — live demo URL, measured cost/latency — are covered in the "What deployable means here" subsection above and formalized in §17's closing note. Not repeated here.)

The catalog grows one agent at a time. Public from agent #01. The visibly-growing catalog IS a credibility signal, but the combined real-world usefulness + educational value of each individual agent is what actually earns the star.

---

## 1. Goals and non-goals

### Goals

- Ship one agent at a time, each demonstrating a different technique on a different real-world task
- Every agent stands alone: a reader can `cd agents/<name>/` and understand the whole thing without cross-references to a heavy chassis
- Every agent's README leads with "what technique this demonstrates" and "why this technique fits this use case"
- Technique variety enforced across the catalog, framework is an implementation detail with a soft ceiling of ≤3 agents per framework (see R3 in §17)
- Shared code is deliberately minimal (`common/llm.py` + `common/pricing.py` only) so per-agent code is uncluttered

### Explicit non-goals (state these in README)

- No competing with big-brand agent frameworks — we USE them, we don't rebuild them
- No "same task in N frameworks" shootout — that's a different project shape (see `rag-recipes` sibling repo for that formula applied to retrieval)
- No autonomous-overlord framing — every agent is a specific-task tool
- No enterprise features in v1 (no SSO, RBAC, audit logs, compliance modules)
- No shared UI framework, no shared observability platform, no shared deploy tooling beyond a documented HF Spaces recipe
- No "100 agents on day 1" — variety over volume, quality over quantity

## 2. Success criteria (per agent)

An agent is "shipped" when all of the following are true:

1. **Solves a real-world use case, not a toy.** The task is one a real person or business would use in production — the agent could be copied into a real project and put behind real users. This is THE primary bar. If the agent fails this test, it does not ship regardless of how interesting the technique is.
2. **Real error handling.** The three obvious error cases for the agent's inputs are handled explicitly (rate limits, malformed input, tool/API failure) — not happy-path only.
3. `cd agents/<name>/ && docker compose up` (or `uv run python -m agent`) produces a working local instance in ≤3 minutes on a clean clone.
4. The agent's README passes `scripts/lint_agents.py`'s template check (see §8.1).
5. The README explains the technique in plain English (not just "here's the code").
6. Code is under 500 LOC excluding UI, formatted with `ruff`, no unused imports.
7. At least one smoke test runs under `LLM_PROVIDER=mock` in CI without real API keys.
8. Manual read-through by the maintainer: "would a reader understand the technique after reading this?"

Optional bars (shipped agent gets a badge/label in the main README if these are met):

- Live demo URL on HuggingFace Spaces linked from the agent's README
- Measured cost per run + p50/p95 latency printed in the README

The F1 (first agent + launch) success criteria are §18.

---

## 3. Architecture and tech choices

Every choice below is deliberate. If a coding tool wants to swap a dep, it must justify it against the reasons here.

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.11+ | Agent/LLM tooling defaults to Python |
| Env | `uv` | Same as sibling rag-recipes project; fast |
| Repo shape | `uv` workspace with each agent as an independent member | Each agent has its own `pyproject.toml`, its own deps, its own runnable entrypoint |
| Shared code | `common/` — LLM adapter + pricing helper ONLY | Explicit minimalism. Any reader can understand an agent by reading `common/llm.py` (~80 lines) + the agent's own folder. No hidden magic. |
| Agent UIs | Gradio (for file upload / typing) OR minimal HTMX/HTML for exceptions | Zero-build UIs; Gradio ships with HF Spaces natively |
| LLM adapters | Direct provider SDKs (`openai`, `anthropic`, `instructor`, etc.) | No wrapping layer between the agent and the SDK — the reader sees the actual framework code |
| Frameworks per agent | Whichever fits the pattern (see §7 roadmap) | Explicit variety is the point |
| Cost accounting | Optional per agent, uses `common/pricing.py` | Doesn't clutter agent code when unused |
| Deployment target | HuggingFace Spaces (optional, free CPU tier) | Verified 2026-08-12 as still free; Docker SDK; 48hr sleep |
| Formatter/linter | `ruff` | Same as rag-recipes |
| CI | GitHub Actions | Standard |
| Tests | `pytest` | Standard |

**Dependency policy:**

- **Root `pyproject.toml`** declares `common/`'s runtime deps (targeting ≤3: `openai`, `anthropic`, `pydantic`) and MUST include `[tool.uv.workspace] members = ["common", "agents/*"]` so `uv sync` at the root discovers every agent as a workspace member. Dev deps (`pytest`, `ruff`, `uv` itself) are not counted against the runtime budget.
- **Each agent's `pyproject.toml`** declares its own runtime deps. Per-agent budget ≤10.
- **Framework-heavy agents** (LangGraph, CrewAI, DSPy, etc.) get their own deps in their own folder. Nothing leaks into other agents.

**Model version pinning is required.** Same rule as rag-recipes §3: pinned dated snapshots, never bare aliases. Every agent's README lists its exact model IDs and re-verifies before ship.

---

## 4. The (deliberately small) shared code — `common/`

This is the single load-bearing simplicity decision. Every prior spec draft had a heavy `platform/` chassis (registry, FastAPI shell, trace SQLite, auto-discovery). That was cut because it hurt the educational value: a reader trying to understand agent #01 shouldn't have to read `platform/registry.py` first.

`common/` contains **exactly two files** at F1:

### 4.1 `common/llm.py`

Adapter for OpenAI + Anthropic + Mock. ~80 lines total.

```python
class LLM(Protocol):
    def complete(self, prompt: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 1024,
                 cacheable_prefix: str | None = None) -> LLMResponse: ...

@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0  # Anthropic-only
    latency_ms: float = 0.0
```

Implementations: `OpenAILLM`, `AnthropicLLM`, `MockLLM`. Factory `get_llm(provider="openai")` reads `LLM_PROVIDER` env var if not passed. CI always sets `LLM_PROVIDER=mock`.

**Agents that use a framework's own client (LangGraph, CrewAI, Instructor, PydanticAI, DSPy)** are allowed and encouraged to instantiate the framework's client directly. `common/llm.py` is the escape hatch for agents that don't need a framework, not a mandatory wrapper. This is deliberate: the reader of a LangGraph agent should see LangGraph code, not our wrapping of it.

### 4.2 `common/pricing.py`

Cost calculator with a dated-snapshot per-model rate table. Ported from rag-recipes' `recipes/pricing.py`. Function:

```python
def cost_usd(model: str, input_tokens: int, output_tokens: int = 0,
             cached_input_tokens: int = 0,
             cache_creation_input_tokens: int = 0) -> float
```

Used by any agent that wants to print a cost number. Optional import.

### What's NOT in `common/`

- **No registry.** Agents don't register themselves. Each agent has its own `python -m agent` entrypoint.
- **No FastAPI shell.** Each agent brings its own if it wants one (usually a Gradio `Interface`).
- **No trace SQLite.** Agents that want cost logging call `cost_usd()` directly and print or log as they see fit.
- **No auto-discovery.** The main README's list of agents is maintained by hand (checked by CI — see §12).
- **No deploy tooling** beyond a documented recipe in `docs/deploying_an_agent.md`.

This minimalism is a feature, not laziness. If two agents genuinely start duplicating the same 50-line helper, we can promote it to `common/`. Not before.

---

## 5. The one-agent-at-a-time release model

**F1 (Foundation)**: `common/` chassis + first agent (receipt/invoice extractor) + landing page + public launch.

**F2, F3, F4, …**: each release ships exactly one new agent demonstrating a **technique not yet in the catalog** (see R3 — framework may be shared with a soft ceiling of ≤3 agents per framework). Each release includes:

1. New agent code under `agents/NN_<name>/`
2. Agent's README with all §8.1 sections
3. Roadmap status update in main README: this agent moved from "planned" to "shipped"
4. Optional: HuggingFace Space deployed and linked
5. Optional: measured cost/latency in the README

**Cadence is intentionally undefined.** Ship when ready. Each release is a natural LinkedIn/HN moment.

---

## 6. F1: the first agent — receipt/invoice extractor

### 6.1 Why this one first

- **Real-world usefulness**: any freelancer, consultant, small business, or bookkeeper has this pain
- **30-second demo**: upload a receipt image → get clean JSON; no explanation needed
- **Specific competitive gap**: commercial tools (Docparser starts at $99/mo, Rossum enterprise-only) exist and cost real money; existing OSS options are either pre-LLM (Tabula) or hidden as tutorial code (Instructor's own example notebook). A polished, deployable, LLM-first OSS tool for this specific job is a real gap and a real launch hook ("I got tired of paying $99/mo …")
- **Perfect fit for the technique it demonstrates**: structured extraction is Instructor's whole point
- **Ships in the target range**: ~15-20 hrs realistic
- **Educational anchor**: the reader learns Pydantic-schema-driven extraction with LLM vision — a technique they'll reach for again

### 6.2 What it does

Input: receipt or invoice image (JPEG/PNG) or PDF.
Output: validated Pydantic JSON.

Schema (v1, deliberately minimal — expand in future releases based on user feedback, per R-10):

```python
class LineItem(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    total: float

class ExtractedReceipt(BaseModel):
    vendor_name: str | None = None
    vendor_address: str | None = None
    vendor_tax_id: str | None = None  # GSTIN, VAT, EIN — normalized to string
    invoice_number: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    currency: str | None = None  # ISO 4217 where determinable
    subtotal: float | None = None
    tax_total: float | None = None
    total: float
    line_items: list[LineItem] = []
    payment_method: str | None = None
    notes: str | None = None
```

### 6.3 Technique demonstrated

**Structured extraction with Instructor + Claude vision.**

The README's educational anchor explains:

- What "structured extraction" means as a pattern
- Why Instructor (Pydantic-schema-first) beats hand-parsing JSON from a chat completion
- Why Claude Sonnet-5 for the vision model here (verified 2026-08-12; must re-verify before ship per §3)
- When the pattern breaks down (blurry images, handwritten receipts, unusual languages)

### 6.4 Implementation shape

- **Framework**: `instructor` on top of `anthropic` SDK (multimodal)
- **UI**: Gradio `Interface` — upload widget, JSON output, sample-receipts button for one-click demos
- **Optional cost print**: uses `common/pricing.py` to display USD cost inline

### 6.5 Handling the three obvious error cases (part of the "real-world usable" bar)

1. Malformed/unreadable image: catch API 400, return friendly error
2. Rate limit/API failure: exponential backoff up to 3 retries, then surface error
3. Partial extraction (valid schema but missed fields): display what was extracted with a warning banner, show raw model output too

### 6.6 F1 acceptance

- Runs locally via `docker compose up` OR `uv run python -m agent` in ≤3 min on a clean clone
- README passes §8.1 lint
- Reader-test: someone new to Instructor understands the technique after reading the README
- Smoke test green under mock
- Optional bars (any of these earn a badge in the main README): live HF Space demo, measured cost per run, per-field accuracy from an eval set

---

## 7. Roadmap — organized by technique variety

Every agent below is chosen so that **the technique fits the use case naturally** (not "let's use CrewAI just to use CrewAI"). The framework was picked because the pattern actually calls for what that framework does well. R3 enforces **technique variety** first — framework is an implementation detail with a soft ceiling of ≤3 agents per framework.

| # | Agent | Technique demonstrated | Framework | Real use case |
|---|---|---|---|---|
| 01 | Receipt/invoice extractor | Structured extraction with vision | Instructor + Anthropic | SMB expense tracking [F1] |
| 02 | Contract reviewer | Long-context analysis + flag extraction | Direct Anthropic SDK (no framework) | Legal review — showing when "no framework" is the right choice |
| 03 | Chat with your CSV | Text-to-SQL with retry-on-error loops | LangGraph (state machine) | Non-technical users querying data |
| 04 | Meeting notes → action items | ReAct pattern for extraction + prioritization | OpenAI Agents SDK | Meeting follow-through |
| 05 | Multi-agent research assistant | Multi-agent collaboration (researcher + writer + editor) | CrewAI | Deep-dive research on a topic |
| 06 | Type-safe email triage | Type-safe agent with structured tool calls | PydanticAI | Inbox categorization on sample .eml files |
| 07 | Prompt optimizer over an eval set | Meta-optimization (learn prompts from examples) | DSPy | Improve any agent's prompt automatically |
| 08 | Python test generator | Tool use + iterative refinement | OpenAI Agents SDK, but with **sandbox execution** | Auto-generate pytest suites from source code |
| 09 | Screenshot → HTML/CSS | Multimodal component reconstruction | Claude vision + plain Python | Design → code |
| 10 | Policy Q&A over handbook | RAG with citations, on-prem/private | Plain Python + sqlite-vec + Anthropic | HR internal tool, privacy-preserving |
| 11 | Resume screener | Batch processing + ranking with rationale | PydanticAI + parallelism | HR/recruiting; cross-promo with Topgenaijobs |
| 12 | Interview prep agent | Adaptive Q&A generation from a JD | OpenAI Agents SDK | Job seekers |
| 13 | From-scratch agent loop | **NO framework — build the loop from primitives** | Plain Python only | Teach what agent frameworks actually do under the hood |
| 14 | Podcast episode processor | Audio → transcript → chapters + summary | Whisper + Anthropic | Creator tool |
| 15 | Business card / ID extractor | Structured extraction with vision (variant of #01) | Instructor + Anthropic | KYC / contact capture; shows how to extend #01's pattern |

**No two agents demonstrate the same TECHNIQUE.** Framework may be shared, with a soft ceiling of ≤3 agents per framework. Called-out cases in this roadmap:

- **#01 and #15** (Instructor + Anthropic) — both are structured extraction, but #15 adds schema evolution and multi-page input handling that #01 doesn't. Two agents, same framework, different technique-depth.
- **#04, #08, #12** (OpenAI Agents SDK, three agents — at the soft ceiling) — three distinctly different techniques: ReAct extraction (#04), tool use with sandbox execution (#08), adaptive Q&A generation from a JD (#12). Any future OpenAI Agents SDK proposal would need to demonstrate a fourth genuinely-different technique OR be rejected by R3.
- **#06 and #11** (PydanticAI) — type-safe structured classification (#06) vs batch processing with parallel ranking (#11). Different enough.
- **#09, #10, #13** ("plain Python" — at the soft ceiling) — three distinctly different techniques: multimodal vision reconstruction (#09), RAG with citations (#10), from-scratch agent loop primitives (#13). Any future "plain Python" proposal would need a fourth different technique.

Enforced by R3 during PR review.

### Agents explicitly excluded from the roadmap

Documented so nobody opens issues asking why. Excluded because they'd either (a) collide head-on with a dominant OSS project we'd lose against, or (b) have no viable local-demo path — needing OAuth to a live workspace, WebRTC infrastructure, a real ticket queue, etc. — meaning a reader can't try them from a `git clone`. Hosted demos are optional (§11), but the local-demo path IS required for R1 (deployable, real-world use case):

- **Basic RAG chat** — rag-recipes covers this fully
- **Web research assistant** — Perplexica, morphic.sh, MindSearch dominant; contract review (#02) demonstrates long-doc analysis instead
- **Multi-agent debate** — demo, not a real-world tool
- **Browser automation agent** — browser-use ~55k stars, Anthropic Computer Use native
- **Voice conversational agent** — WebRTC infrastructure poor fit for HF Spaces
- **Slack/Discord bots** — per-workspace app registration, no click-demo
- **Customer support triage** — needs real ticket queue
- **Full end-to-end research agent** — GPT Researcher dominant

If any excluded item becomes ship-worthy (competitor abandons, infrastructure changes), open an issue with a specific differentiation angle per R1 (real-world use case) + R3 (technique variety) + R10 (no invented scope).

---

## 8. Per-agent template (mandatory structure)

Every agent under `agents/<NN>_<name>/` MUST have this structure:

```
agents/NN_<name>/
├── README.md              # Uses §8.1 template
├── pyproject.toml         # uv-managed, agent-specific deps
├── Dockerfile             # optional; needed only if agent is deployed
├── agent.py               # main entry point, runnable via `python -m agent`
├── schemas.py             # Pydantic models (if applicable)
├── prompts/               # any prompt templates as .txt files
├── ui.py                  # Gradio Blocks or FastAPI HTMX routes (if applicable)
├── eval/                  # if scoring is objective
│   ├── cases.jsonl
│   └── run_eval.py
├── tests/
│   └── test_smoke.py      # runs under LLM_PROVIDER=mock, always
└── examples/              # sample inputs for demo (e.g., sample receipts)
```

### 8.1 Agent README template

Every agent's README MUST have these sections in this order (enforced by `scripts/lint_agents.py`, called from CI):

1. **Title + one-liner** — clear tool description in plain English
2. **Technique demonstrated** — the educational anchor: what pattern this teaches
3. **Why this technique for this use case** — 2-4 sentences on why this framework/pattern is the right fit here, and when it isn't
4. **What it does** — 2-3 sentences, plain English
5. **How to run locally** — the 2-3 commands
6. **Code walkthrough** — 5-10 bullets pointing at key files/lines the reader should read to understand the technique. **At least ONE bullet MUST point at where the three R5 error cases are handled** (rate limits, malformed input, tool/API failure) — this makes the "deployable" claim verifiable by reading the code, not just by trusting the SPEC
7. **When to use / when NOT to use** — bullet lists for each
8. **Where this fails** — honest failure modes with specific examples (not generic warnings)
9. **Optional: Live demo** — link to HF Space if one exists
10. **Optional: Measured cost + latency** — from real runs, not estimates
11. **Optional: Accuracy** — from `eval/cases.jsonl` if applicable

Sections 1-8 are REQUIRED. Sections 9-11 are OPTIONAL badges that show up in the main README's agent grid.

Enforced by `scripts/lint_agents.py` (parallel to rag-recipes' `scripts/lint_notebooks.py`).

---

## 9. Repo structure

Target tree at F1 ship:

```
real-world-agents/
├── README.md                              # main README
├── LICENSE                                # MIT
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md                     # Contributor Covenant 2.1
├── SPEC.md                                # this file
├── pyproject.toml                         # workspace root, uv-managed
├── .env.example
├── .gitignore
├── .github/
│   ├── workflows/
│   │   └── ci.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── ISSUE_TEMPLATE/
│       ├── new_agent.yml
│       ├── bug_report.yml
│       └── config.yml
├── common/
│   ├── __init__.py
│   ├── llm.py                             # ~80 lines: OpenAI + Anthropic + Mock adapters
│   ├── pricing.py                         # ported from rag-recipes
│   └── tests/
│       ├── test_llm.py
│       └── test_pricing.py
├── agents/
│   └── 01_receipt_extractor/              # F1
│       ├── README.md
│       ├── pyproject.toml
│       ├── Dockerfile                     # optional
│       ├── agent.py
│       ├── schemas.py
│       ├── prompts/
│       │   └── extract.txt
│       ├── ui.py                          # Gradio Blocks
│       ├── eval/
│       │   ├── cases.jsonl
│       │   ├── images/                    # 20 real receipts, anonymized
│       │   └── run_eval.py
│       ├── tests/
│       │   └── test_smoke.py
│       └── examples/
│           └── sample_receipt.jpg
├── scripts/
│   ├── lint_agents.py                     # enforces §8.1 README template
│   ├── lint_readme.py                     # checks main README agent grid matches agents/ dir
│   └── new_agent.py                       # scaffolds a new agent's dir tree
└── docs/
    ├── adding_an_agent.md
    ├── deploying_an_agent.md              # HF Spaces recipe
    └── launch/
        ├── twitter_post.md                # F1 launch draft
        └── linkedin_post.md
```

No `platform/`. No `platform/registry.py`, `platform/app.py`, `platform/trace/`. Removed deliberately.

---

## 10. LLM adapter contract (`common/llm.py`)

Minimal. ~80 lines. Ported and stripped from rag-recipes' `recipes/llm.py`.

```python
# common/llm.py
from __future__ import annotations
import os, time
from dataclasses import dataclass
from typing import Protocol

@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    latency_ms: float = 0.0

class LLM(Protocol):
    def complete(self, prompt: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 1024,
                 cacheable_prefix: str | None = None) -> LLMResponse: ...

class OpenAILLM: ...      # thin wrapper around openai SDK
class AnthropicLLM: ...   # thin wrapper around anthropic SDK, handles cache tiers
class MockLLM: ...        # for CI — port from rag-recipes' MockLLM: deterministic
                          # hash-based responses, cache-tier simulation when
                          # cacheable_prefix is passed (first call bills as
                          # cache_creation, subsequent calls with same prefix
                          # bill as cached_input). Same contract, same tests.

def get_llm(provider: str | None = None) -> LLM:
    provider = provider or os.environ.get("LLM_PROVIDER", "openai")
    if provider == "openai": return OpenAILLM()
    if provider == "anthropic": return AnthropicLLM()
    if provider == "mock": return MockLLM()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
```

Agents using framework-native clients (`instructor.from_anthropic(...)`, `langgraph.graph.StateGraph(...)`, `crewai.Agent(...)`, `pydantic_ai.Agent(...)`, `dspy.Predict(...)`, etc.) skip `common/llm.py` entirely — that's the point. `common/llm.py` exists for agents that don't need a framework (e.g., #02 contract reviewer, #13 from-scratch agent loop).

---

## 11. Optional deployment: HuggingFace Spaces

**Not required to ship an agent.** Documented as an optional add-on that earns the agent a "live demo" badge in the main README.

`docs/deploying_an_agent.md` documents the recipe:

```bash
# Per agent, done manually by the maintainer
cd agents/<name>/
# Create a HF Space at huggingface.co/spaces/<owner>/rwa-<name>
git init
git remote add space https://huggingface.co/spaces/<owner>/rwa-<name>
git add . && git commit -m "deploy <name>"
git push space main
```

The HF Space uses each agent's own Dockerfile. Space naming convention: `rwa-<short-name>` (shortened from `real-world-agents-<name>` to fit HF's readable URL norm).

For agents that don't fit HF Spaces (Postgres-required, WebRTC-required), the excluded list in §7 already accounts for that — we simply don't build those agents.

---

## 12. CI (`.github/workflows/ci.yml`)

Required jobs on push and PR:

1. **Lint:**
   - `ruff check .`
   - `python scripts/lint_agents.py` — enforces §8.1 on every shipped agent's README
   - `python scripts/lint_readme.py` — checks main README's "shipped agents" table matches actual `agents/` subdirectories
2. **Common tests:** `uv run pytest common/tests/`
3. **Per-agent smoke tests:** for every `agents/*/tests/test_smoke.py`, run under `LLM_PROVIDER=mock`. Any exception fails the job.

CI never calls real OpenAI/Anthropic. No API-key secrets in CI.

No deploy job in CI. Deploys are manual per-agent — this matches the "optional deployment" stance in §11.

---

## 13. Environment and secrets

`.env.example`:

```
# Required for local runs against real APIs (NOT for CI)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional per-agent — populate as agents require
VOYAGE_API_KEY=
BRAVE_SEARCH_API_KEY=

# Force mock LLM (set by CI automatically)
LLM_PROVIDER=openai  # or "anthropic" or "mock"
```

### Cost note

Cost is per-user-request-scale (not per-full-eval-run like rag-recipes). Estimated F1 costs (receipt extractor, Claude Sonnet-5 with vision — rates from rag-recipes' verified `pricing.py`, dated snapshot 2026-08-12; re-verify per §3 before ship):

- ~1500 input tokens/receipt (image tokens + prompt) at $2/1M = ~$0.003
- ~200 output tokens at $10/1M = ~$0.002
- **Per-extraction: ~$0.005** (drops to ~$0.001 with prompt caching enabled)

If an agent gets a live HF Space demo, the maintainer eats stranger-traffic cost. Documented mitigations per agent's own README:

- Rate-limit strangers (e.g., 10 requests/IP/hour) on the hosted demo
- Use prompt caching where the prompt is stable
- If usage exceeds budget, either add auth gating OR route the hosted demo to a smaller/cheaper model with a "demo-tier model, results may vary" banner

Target monthly cost budget for hosted demos: **≤$5/month at steady state**.

---

## 14. Main README structure

The main README is a **learning path + catalog**, not a leaderboard.

Required top-down order:

1. **Title + one-liner** — the §0 one-liner verbatim
2. **What this repo is** — 2-3 sentences: growing catalog, each agent teaches ONE technique, read the code
3. **Project status callout** — how many agents are shipped, one-line description of what's next
4. **Shipped agents grid** — table with columns: number, agent name, technique demonstrated, framework, live demo (optional badge), real use case
5. **Roadmap grid** — same shape, for planned agents from §7. Every row links to a tracking issue.
6. **Quick start** — 2-3 commands: `git clone`, `cp .env.example .env`, `cd agents/<name>/ && uv run python -m agent`
7. **Cost note** — one paragraph: running any agent against real APIs costs real money; agents' individual READMEs show per-run cost; link to `.env.example`; link to §13 of SPEC for the cost model
8. **How to read this repo** — a suggested reading path: "start with #01 for structured extraction, then #13 for the from-scratch loop that shows what frameworks are doing, then whichever framework you want to learn next"
9. **How to add your own agent** — link to `docs/adding_an_agent.md`
10. **Design principles** — the §0 positioning stated crisply, three bullets
11. **Non-goals** — from §1
12. **Contributing** — link to `CONTRIBUTING.md`
13. **License** — MIT, link

The "learning path" (§14 item 8) is the piece that distinguishes this README from the 500-project competitors. It signals that this repo was designed to be READ, not just cloned.

---

## 15. `CONTRIBUTING.md` — contributor guide

Full guide, not a stub.

### 15.1 How to use this repo (non-contributors)

1. `git clone https://github.com/<owner>/real-world-agents && cd real-world-agents`
2. Pick an agent from the shipped list in the main README
3. `cd agents/<NN_name>/` and read its README first
4. Follow the "How to run locally" section
5. To copy the agent's pattern into your own project: see the "Code walkthrough" section, copy the files it points at

### 15.2 Ways to contribute

- **Propose a new agent** demonstrating a technique not yet in the catalog (any framework, subject to the R3 soft ceiling)
- **Improve an existing agent** — better error handling, more eval cases, honest failure-mode analysis, better README pedagogy
- **Add cases to an agent's eval set** — helps make the accuracy claim more credible
- **Improve `common/`** — better error messages, better docs, fewer surprises
- **Report bugs**

Do NOT contribute without an issue first: new top-level deps in `common/`, new LLM providers (chassis contract change), UI/dashboard reworks, changes to the per-agent README template.

### 15.3 Proposing a new agent — step-by-step

1. Open an issue using the "New agent" template. It asks: what task, what technique/framework demonstrated, why is this framework the right fit for this task, who uses this in real life, what's the closest existing OSS competitor and how does this differ (see §7's "excluded" list for prior-art scrutiny examples).
2. Wait for maintainer nod before writing code. This gate exists specifically to enforce R1 (real-world use case) and R3 (technique variety) — a proposed agent that doesn't clear the real-world bar, or duplicates a technique already in the catalog, will be turned back regardless of which framework it uses.
3. Fork, branch `agent/<NN>_<short-name>` off main.
4. `python scripts/new_agent.py <short-name>` scaffolds the directory tree.
5. Implement following §8's structure. README must pass §8.1's linter.
6. Add at least one smoke test that runs under `LLM_PROVIDER=mock`.
7. Local checks before PR: `ruff check .`, `python scripts/lint_agents.py`, `python scripts/lint_readme.py`, `uv run pytest`.
8. Open PR against main with the PR template.
9. CI must be green. Maintainer reviews for: does the technique fit the use case, does the README teach the technique (not just describe the code), is the failure-mode section honest, does it follow R1-R10 (§17).
10. Squash merge is default.

### 15.4 Fixing a bug + PR expectations

Same shape as rag-recipes §16.4-16.5. One logical change per PR, no `Co-Authored-By: Claude` trailer, CI must be green before review.

### 15.5 Code of conduct

Contributor Covenant 2.1, ships as `CODE_OF_CONDUCT.md`. Enforcement contact: `rajeshmane711@gmail.com`.

---

## 16. Milestones with acceptance criteria

### F1: Foundation + first agent (estimated 25-40 hrs — significantly less than earlier draft because chassis is smaller)

| Sub-phase | Deliverable | Acceptance |
|---|---|---|
| F1.1 | `common/llm.py` + `common/pricing.py` + tests | Mock-provider tests green in CI (`uv run pytest common/tests/`). Real-provider (OpenAI + Anthropic) smoke tests run manually by the maintainer before F1 ship — not gated by CI, since CI has no API keys (R8). |
| F1.2 | Receipt extractor agent (`agents/01_receipt_extractor/`) | Runs locally via `uv run python -m agent`; smoke test green under mock |
| F1.3 | Receipt extractor README passes §8.1 lint | `python scripts/lint_agents.py` green |
| F1.4 | Optional: HF Space deployed and linked | Public URL reachable, demo works on 5 sample receipts |
| F1.5 | **Required for F1 (flagship agent):** eval set with 20 real anonymized receipts + accuracy table in README | Per-field accuracy computed and printed. Required for F1 because the launch story is "deployable OSS receipt extractor" — that claim needs measured proof for the flagship. For F2+ agents, this bar drops to optional per §16 Fn acceptance. |
| F1.6 | Main README + landing page + CI + LICENSE + CONTRIBUTING.md + CODE_OF_CONDUCT.md + .env.example + .github/ | Every §14 section present; every link resolves; CI green on a clean push |
| F1.7 | Launch drafts in `docs/launch/` | Both `twitter_post.md` and `linkedin_post.md` ready-to-post (no `[FILL IN]` placeholders unless truly awaiting real data) |

**F1 is done when F1.1-F1.3 + F1.5 + F1.6 + F1.7 are complete. Only F1.4 (HF Space live demo) is optional.**

### Fn: subsequent agents (estimated 10-20 hrs each — smaller because chassis is stable and framework variety avoids "reinvent common patterns" waste)

Each Fn ships one agent from §7's roadmap. Acceptance:

- Demonstrates a technique not yet in the catalog (R3); framework may be shared subject to the ≤3 soft ceiling
- Runs locally
- README passes §8.1 lint, has the "Technique demonstrated" + "Why this technique for this use case" sections filled in with real content (not boilerplate)
- CI green
- Main README's shipped table updated
- Optional badges (live demo, cost, accuracy) earned or explicitly noted as N/A

**No fixed cadence.** The visibly-growing catalog is the credibility signal regardless of pace.

---

## 17. Project-specific hard rules (R1-R10)

Ordered by priority. R1 is THE primary bar — everything else is subordinate to it.

- **R1. Real-world use case (PRIMARY BAR).** Every shipped agent solves a real task a person or business would use in production — the agent could be copied into a real project and put behind real users. If an agent fails this test, it does not ship, regardless of how interesting the technique is. See §7's "excluded" list for prior-art examples where we chose not to build for exactly this reason.
- **R2. Simplicity + readability.** Each agent's own code ≤500 LOC excluding UI. `common/` stays minimal (see §4). If a new agent tempts you to add a new shared abstraction, ask "would a reader understand THIS agent without reading the abstraction?" — if no, keep the code local to the agent. Readability is the second-most-important bar because the repo's whole value proposition is "you can read and copy this."
- **R3. Technique variety (with soft framework ceiling).** Every agent must demonstrate a distinctly different TECHNIQUE (structured extraction, ReAct, tool use with retry, RAG with citations, multi-agent collaboration, meta-optimization, from-scratch loop, etc.). Framework is an implementation choice with a soft ceiling of ≤3 agents per framework. Each subsequent agent using an already-used framework must demonstrate a technique the earlier ones don't (see §7's called-out cases for the intended pattern). Enforced during maintainer review of new-agent PRs.
- **R4. Per-agent README template enforcement.** All 11 sections from §8.1, in order, on every shipped agent's README. Enforced by `scripts/lint_agents.py` in CI. Sections 1-8 required; 9-11 optional badges.
- **R5. Real error handling.** The three obvious error cases for each agent's inputs are handled explicitly (rate limits, malformed input, tool/API failure) — not happy-path only. This is what makes the agent "deployable" per §0.
- **R6. Dependency budget.** `common/` runtime deps ≤3 (openai, anthropic, pydantic). Dev deps (pytest, ruff, uv itself) are not counted. Per-agent runtime budget ≤10. Never add LangChain-style monolithic frameworks to `common/` — an individual agent may use one, that's fine.
- **R7. Model version pinning.** Every LLM call uses a pinned dated snapshot. Every agent's README lists exact model IDs and their verification date.
- **R8. CI is mock-only.** No real API keys in CI secrets. Every test suite runs under `LLM_PROVIDER=mock`.
- **R9. Prose style + git hygiene.** No em dashes or en dashes between words; hyphens only for real technical terms. Conventional commits, no AI co-author trailers, squash merge only. Same as rag-recipes' inherited style.
- **R10. No invented scope + honest failure modes.** New agent = new issue + maintainer nod + differentiation angle articulated. Every agent's README has a "Where this fails" section with specific examples.

Note: live demo URLs and measured cost/latency are OPTIONAL per-agent add-ons (see §11 and §8.1 sections 9-11), not hard rules. They earn a badge in the main README when present but are not required for shipping. Being "deployable" per §0 means the agent's code and design are production-worthy — not that a hosted instance exists.

---

## 18. Definition of done (F1)

The project is F1-done and ready to launch publicly when:

- `common/llm.py` + `common/pricing.py` are complete, tested, small
- Receipt extractor runs locally via `uv run python -m agent`
- Receipt extractor's README passes §8.1 lint with real "Technique demonstrated" + "Why this technique for this use case" content
- **20-receipt eval set exists and per-field accuracy table is printed in the receipt extractor's README** — required for F1 because the launch story is "deployable OSS receipt extractor"; unproven accuracy makes the claim unbacked
- Main README has §14 sections, shipped grid shows F1, roadmap grid shows planned agents from §7
- CI is green on a clean push
- `LICENSE` (MIT), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `.env.example`, `.github/` templates all present
- Launch drafts in `docs/launch/` ready-to-post
- Reader-test: someone new to Instructor reads the F1 agent's README + code and can explain what "structured extraction" is and when they'd use it

Optional but recommended for F1 launch impact:

- HF Space live demo
- Measured cost per run in README (in addition to the required accuracy table)

---

## 19. Launch plan (post-F1)

Educational framing, not "impressive demo" framing:

1. **LinkedIn post day of F1 launch.** Framing: *"I'm building a growing catalog of AI agents, each showing a different framework or technique on a real problem. First one is receipt-to-JSON with Instructor — code is small enough to read in one sitting."* Live demo link in comments if available.
2. **Show HN on a Tuesday 8-10am US eastern.** Title: *"Show HN: real-world-agents — one AI agent per framework, each on a real task, small enough to read"*
3. **PR to awesome-ai-agents lists** (pick the most-starred 2)
4. **Cross-post to r/LocalLLaMA** (dev audience) and one non-tech subreddit relevant to the specific F1 agent (e.g., r/smallbusiness for receipts)
5. **DM 5-8 AI engineers/educators** with the reader-friendly angle

Post-F1 launch, each Fn is a mini-launch. Every release has a natural angle: "Here's how you build X with framework Y." Steady content flywheel; matches creator-brand rhythm.

---

## 20. What a coding tool should ask before starting

If any of these are unclear, ask before writing code:

1. Which HuggingFace org/username should own the Spaces (when live demos are added)?
   → Ask per-deploy; don't hardcode.
2. F1's vision model: Claude Sonnet-5 or GPT-5.4-mini with vision?
   → **Claude Sonnet-5** for image+structured-output quality; verify pricing per §3 before ship.
3. Eval set for F1: real receipts (privacy?) or synthetic?
   → **Real receipts from the maintainer's own life, anonymized** (vendor names replaced but structure preserved). Alternative: public receipt dataset with attribution.
4. Should `common/llm.py` include a Voyage or Cohere adapter?
   → **No.** Add adapters only when a shipping agent needs them (R-10).
5. What if an F2+ proposed agent doesn't fit HF Spaces even as an optional add-on?
   → Fine — the live-demo badge is optional (see §17 closing note). The agent still ships without a live demo as long as it clears R1 (real-world use case) and the other required rules.

---

END OF SPEC.
