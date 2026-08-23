# Contract review → flagged clauses + plain-English risk

Upload a PDF or plain-text contract (NDA, MSA, SaaS ToS, employment agreement, vendor SOW). Get flagged clauses colour-coded by severity, plain-English explanations of what each clause actually means, and a JSON review you can pipe into other tools. A deployable OSS alternative to the "have a lawyer look at it" ask for freelancers, consultants, and small businesses signing routine agreements they don't want to pay $300/hr to scrutinize.

## Technique demonstrated

**Long-context single-pass analysis with a hand-rolled JSON-validate-retry loop, no framework.** The whole contract goes into the model's context window in one call (`gpt-4.1-mini` has ~1M tokens, ~750k words, comfortably larger than any realistic single contract). No chunking. No RAG. Just: prompt the model for JSON matching a Pydantic schema, parse the response, validate it, and on failure feed the validation error back into the prompt and retry.

This is deliberately the "here's what [Instructor](https://python.useinstructor.com/) (agent #01) does under the hood" lesson. Same pedagogical pattern, made visible instead of abstracted. Not a reason to reject Instructor when you build your next agent. A reason to understand what it's doing for you.

## Why this technique for this use case

Contracts are text. Unlike receipts (agent #01), there's no image-parsing step needed. Modern long-context models can read a whole contract in one call, so chunking or RAG would add complexity for zero quality gain here. Long-context single-pass IS the right shape.

Why no framework? Because writing the retry loop by hand exposes a real design choice you'd otherwise miss: this loop enforces a **cross-field invariant** (every flag's `excerpt` field must be a verbatim substring of the source contract) that a per-field Pydantic validator can't reach. Pydantic validators see one field at a time, not the source document alongside it. Instructor's `response_model=` handles single-schema validation. This loop does that AND excerpt-in-source verification AND category-taxonomy enforcement, all through the same retry mechanism. When paraphrased quotes ship into your database silently, users can't locate the flagged clauses. This loop stops that at extraction time.

Where this technique is NOT the right fit: (a) documents longer than any single model's context window (bundled MSA + all its schedules: needs chunking or per-section review), (b) tasks where verbatim extraction isn't important (summarization, general Q&A: Instructor's simpler abstraction wins), (c) agent workflows with multiple turns of tool use (a plain JSON schema is the wrong shape entirely).

## What it does

Input: a contract as either a PDF (text-based, not scanned) or plain text. Output: a validated `ContractReview` Pydantic object with `contract_type` (identified first via chain-of-thought), `parties`, a list of `flags` (each with `category` / `severity` / `excerpt` / `page_number` / `explanation` / optional `recommendation`), a plain-English `summary`, and a single-word `overall_risk`. Empty `flags` list is a valid result: some contracts genuinely have no clauses worth flagging for a non-lawyer.

Flag categories are a closed taxonomy of 11 values (`auto_renewal`, `unlimited_liability`, `unilateral_termination`, `arbitration`, `non_compete`, `ip_assignment`, `indemnification`, `exclusivity`, `governing_law`, `confidentiality`, `other`); the model can't invent new categories, the retry loop rejects them if it tries.

## How to run locally

Four commands from a fresh clone (`python -m agent` must run from inside the agent's own directory: `agent` is a submodule of the digit-prefixed `02_contract_reviewer` package, not importable as a top-level module from the repo root):

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # then edit .env: set LLM_PROVIDER + the matching API key
cd agents/02_contract_reviewer
```

`.env` defaults to `LLM_PROVIDER=openai` (using `gpt-4.1-mini`). To use Anthropic or Gemini instead, set `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=gemini` and the matching key (`ANTHROPIC_API_KEY` / `GEMINI_API_KEY`); no code changes needed. See `.env.example` for every option, including per-provider model overrides.

CLI:

```bash
uv run python -m agent path/to/contract.pdf
```

Override provider/model for a single call without touching `.env`:

```bash
uv run python -m agent path/to/contract.pdf --provider anthropic --model claude-sonnet-5
```

Gradio UI:

```bash
uv run python -m agent --ui
```

Mock mode (no API key, canned response for testing the pipeline end-to-end):

```bash
LLM_PROVIDER=mock uv run python -m agent path/to/contract.pdf
# or, without touching the env var:
uv run python -m agent path/to/contract.pdf --provider mock
```

## Code walkthrough

Under 500 LOC excluding UI. Read these in order to understand the pattern:

1. **`schemas.py`**: `ContractReview` + `ContractFlag` Pydantic models. `category` is a closed `Literal` (11 values) so the model can't invent private categories. `page_number: int | None` with `ge=1`: verifiable field, not a hallucination-prone free-text location hint. `excerpt` is documented as substring-of-source-validated at loop time (Pydantic field validators can't see the source document).
2. **`prompts/review.txt`**: the single-pass chain-of-thought prompt: identify `contract_type` FIRST, then apply type-appropriate scrutiny. Categories with one-line glosses so the model knows when to use each. Explicit "excerpt MUST be a verbatim substring, do not paraphrase" rule so the retry loop's excerpt-check isn't the first place the model sees the constraint.
3. **`agent.py::resolve_provider()`**: reads `LLM_PROVIDER` (default `"openai"`); `common/llm.py::resolve_model()` reads the matching per-provider model override env var. Same env-var contract as agent #01.
4. **`agent.py::review_contract()`**: the public API. Under `provider="mock"` short-circuits to `_mock_review()`; otherwise routes bytes → `_load_source_text()` → three R5 checks (scanned PDF, context window, real loop with translated API errors).
5. **`agent.py::_run_review_loop()`**: **THE pedagogical anchor.** Read this to understand what Instructor's `response_model=` does under the hood. On each attempt: build prompt with retry-feedback if not first attempt → call LLM → parse JSON → validate against Pydantic → check every flag's excerpt is a substring of source → return on success, or feed the specific error back into the next attempt's prompt.
6. **`agent.py::_validate_review()`**: the two-stage validation that makes this loop richer than Instructor for this specific problem. Stage 1: Pydantic validates schema shape. Stage 2: per-flag excerpt-substring check (see `_normalize_for_substring` for whitespace-collision handling). Stage 2 is why hand-rolled beats Instructor here: cross-field invariants aren't a per-field validator's job.
7. **`agent.py::_is_likely_scanned()`**: R5 case 1. Two thresholds (`< 200` total chars OR `< 100` chars/page averaged) chosen defensibly, not magic-in-the-moment. Redirects the user to OCR-first tools (Tesseract, unstructured.io) with "outside this agent's scope" honesty rather than a false promise to handle it later.
8. **`agent.py::_check_context_window()`**: R5 case 2. Chars-per-token heuristic (no SDK dependency), 900k-token ceiling well under `gpt-4.1-mini`'s 1M cap. Names the model in the error message so a user on a smaller-context model knows which knob to adjust.
9. **`agent.py::_translate_api_error()`**: R5 case 3. Six-branch priority order (class-name → status-code → message-string fallback → generic). This agent does NOT auto-retry rate-limit errors: the user pays per real API call and is better-placed to decide whether to wait and re-run than to have the loop silently spend more of their budget on a failing endpoint. Retry loop only re-prompts for VALIDATION failures.
10. **`agent.py::_extract_pdf_text()`**: per-page pypdf extraction with `--- PAGE N ---` markers injected between pages. The model reads those markers and populates `ContractFlag.page_number` from them; the schema's `ge=1` constraint means the model can't fake a "0" or negative page.
11. **`ui.py::build_ui()`**: Gradio Blocks: file upload widget (PDF + .txt), per-flag severity-tinted HTML cards (red/amber/green), full JSON in a code block, provider-override Radio. Kept intentionally small so the load-bearing code stays in `agent.py`.
12. **`tests/test_smoke.py`**: 39 tests, all under `LLM_PROVIDER=mock`. Covers mock path, retry loop (happy path + bad-JSON retry + paraphrased-excerpt retry + schema-violation retry + exhaustion), all 3 R5 branches, `_translate_api_error`'s 6 priority paths, and pure helpers. `SequenceLLM` fixture supplies scripted responses for the retry-loop tests without touching shared chassis. Auto-use `_zero_backoff` fixture monkeypatches sleep to 0 so 39 tests run in ~0.25s.

## When to use / When NOT to use

**Use when:**
- You're a freelancer / consultant / small business signing routine agreements (NDAs, SOWs, vendor MSAs, SaaS ToS) and don't want to pay $300/hr to have a lawyer read every one
- You want a plain-English second opinion on what a clause actually means for you, in the language a non-lawyer thinks in
- You need the flagged clauses in a machine-readable format (JSON) to feed into downstream tools (contract-management software, spreadsheet exports, ticketing)
- You want to understand HOW a schema-validated LLM extraction loop works (this agent shows the pattern without Instructor's abstraction)

**Do NOT use when:**
- You're facing a genuinely-high-stakes contract (large-value deal, unusual industry, litigation risk, regulated data): get a real lawyer. This tool is a triage aid, not a replacement.
- The contract is longer than the model's context window (a bundled MSA + all its schedules can exceed 1M tokens): this agent will refuse via the context-window R5 branch. Split into logical sections and review each.
- The contract is a scanned PDF (image-only): this agent detects and refuses. Use OCR-first tools (Tesseract, unstructured.io) to convert to text first.
- You need a legal opinion (not just a risk flag): LLMs are unreliable at applying law to facts; use this to spot issues, not to resolve them.

## Where this fails

Specific, honest failure modes, not generic warnings:

- **Scanned PDFs**: text-based PDF extraction returns near-empty; agent refuses via the R5 branch above with a pointer to OCR tools. No silent partial-review that would miss the actual clauses.
- **Handwritten amendments to typed contracts**: pypdf extracts the typed body cleanly and drops the handwritten marginalia; the agent reviews the typed version as if the amendments don't exist. Symptom: a flagged clause the reader knows was crossed out or amended by hand. Workaround: none in this agent; use an OCR + handwriting model instead.
- **Very old PDFs with embedded-image text**: some legacy PDFs "look like" text-based but their content is actually rasterized bitmaps in a wrapper. pypdf returns garbage; the R5 scanned-PDF check catches most, but a partially-extracted document (some pages text, some rasterized) will slip through and get partial coverage. Symptom: fewer flags than the reader expects; check `page_number` values for gaps.
- **Contracts in languages other than English**: the schema and prompt are English-only; the model MIGHT return English explanations for a Spanish contract, but the excerpt-substring check will fail against the (Spanish) source text and every flag gets rejected in the retry loop. Symptom: exhausted retries or empty flag list.
- **Ambiguous document types**: the model's `contract_type` identification is the first CoT step; if it mis-identifies (e.g. calls a Statement of Work a Master Service Agreement), the type-appropriate scrutiny it applies afterward is wrong. Single-pass keeps the pattern simple; a two-pass detect-then-flag architecture would fix this at the cost of doubling latency and provider cost. Not built in v1.
- **Paraphrased excerpts**: the retry loop catches these and re-prompts. But if the model consistently paraphrases across all `max_retries` attempts (rare, seen mostly on very short clauses), the whole review fails with retries exhausted. Symptom: `ContractReviewError` with `.partial.raw_text` populated for inspection. This is an intentional fail-loud, not a fail-silent: better to know the excerpts aren't verifiable than to trust a paraphrased summary.
- **Curly quotes (`""`) vs straight (`""`)**: PDFs frequently use curly quotes; models sometimes quote back with straight. `_normalize_for_substring` collapses whitespace but does NOT normalize quote characters, so an otherwise-verbatim excerpt fails the substring check for a quote-only reason and triggers a wasted retry. Cheap to fix; deferred as a v1.1 improvement.
- **Rate-limit / API failure**: no auto-retry with backoff (explicit design decision, see `agent.py::_rate_limit_error`). One failed attempt, translated to a "temporarily rate-limited" message, then the user re-runs manually. Trade: no silent budget-burning; downside: the user has to click "review" again after a wait.
