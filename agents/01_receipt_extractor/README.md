# Receipt / invoice → JSON

Upload a receipt or invoice image, get clean structured JSON back. A deployable OSS alternative to the receipt-parsing SaaS tools (Docparser starts at $99/mo, Rossum is enterprise-only) that most freelancers, consultants, and small businesses reach for.

## Technique demonstrated

**Structured extraction with [Instructor](https://python.useinstructor.com/) + Anthropic Claude vision.** A single Pydantic schema (`ExtractedReceipt`) is passed to Instructor's `response_model` parameter; Claude reads the image and fills the schema; Instructor validates the output against the schema and retries on validation failure. Four steps collapsed into one call.

## Why this technique for this use case

Receipts are structured documents that happen to arrive as images. The naive alternative is: (1) prompt Claude to return JSON, (2) `json.loads()` the response, (3) validate the shape yourself, (4) retry when the model returns a stray comma or misnames a field. Instructor collapses all four into one call. You describe the shape once as a Pydantic model, and every downstream consumer (this agent's CLI, its Gradio UI, and anyone copying `schemas.py` into their own project) uses the same source of truth.

Where this technique is NOT the right fit: freeform text extraction with no schema, agent loops that need tool use across multiple turns, or anything where you want to swap models frequently across providers (Instructor's provider adapters are stable but not zero-friction).

## What it does

Input: a receipt or invoice image (JPEG/PNG/WebP/GIF). Output: a validated `ExtractedReceipt` Pydantic object with vendor info, invoice dates, per-line items, tax breakdown, and total. Optional per-request cost estimate printed to stderr under real API mode.

## How to run locally

Three commands from a fresh clone:

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # then edit .env to add your ANTHROPIC_API_KEY
```

CLI:

```bash
uv run --package receipt-extractor python -m agent path/to/receipt.jpg
```

Gradio UI:

```bash
uv run --package receipt-extractor python -m agent --ui
```

Mock mode (no API key needed, canned response for testing the pipeline end-to-end):

```bash
LLM_PROVIDER=mock uv run --package receipt-extractor python -m agent path/to/receipt.jpg
```

## Code walkthrough

Under 500 LOC total (excluding UI). Read these in order to understand the pattern:

1. **`schemas.py`** — `ExtractedReceipt` + `LineItem` Pydantic models. Every field's `description=` is a prompt hint to Claude; keep them concrete. Required fields (`total`, `line_items`, `LineItem.description`, `LineItem.total`) are the load-bearing bits — everything else is optional because real-world receipts genuinely omit them.
2. **`prompts/extract.txt`** — the plain-English instructions to Claude ("extract exactly what is printed, do not fabricate"). Loaded once at extraction time; pinned in tests so accidental edits show up as failures.
3. **`agent.py::extract_receipt()`** — the public API. ~40 lines. Under `LLM_PROVIDER=mock` it short-circuits to `_mock_extraction()`; otherwise it base64-encodes the image, builds Anthropic's content-block format, and calls `client.messages.create(response_model=ExtractedReceipt, ...)`.
4. **`agent.py::_translate_api_error()`** — R5 real-error-handling. Where the three concrete failure modes get mapped to `ReceiptExtractionError` with friendly messages: (a) HTTP 400 or "invalid image" → "Couldn't read this image"; (b) HTTP 429 or "rate limit"/"overloaded" → "temporarily rate-limited"; (c) `ValidationError` (checked by class name first per S9 in the F1.2 review) → attaches the raw model output as `partial` so the UI can show it in a warning banner.
5. **`agent.py::_get_real_client()`** — Instructor + Anthropic client construction, lazy-imported so `LLM_PROVIDER=mock` test runs never need `instructor` or `anthropic` installed.
6. **`agent.py::_mock_extraction()`** — deterministic canned `ExtractedReceipt`; input byte count encoded into the line description so tests can prove the mock actually saw its input.
7. **`agent.py::main()`** — CLI entry point via `python -m agent`. `nargs="?"` on the `image` arg lets `--ui` launch standalone (fixed in F1.2 review S1). Prints JSON to stdout, cost/latency to stderr.
8. **`ui.py::build_ui()`** — Gradio Blocks: upload widget, JSON output, cost line, partial-extraction warning banner. Kept intentionally small so the load-bearing code stays in `agent.py`.
9. **`tests/test_smoke.py`** — 12 tests, all under `LLM_PROVIDER=mock`. Covers schema validation, mock-input pass-through, JSON serializability, all three R5 error branches, and the prompt file's presence + required content.

## When to use / When NOT to use

**Use when:**
- You have receipt or invoice images and need structured JSON to feed into accounting software, expense tracking, or a spreadsheet
- You want an OSS baseline you can host yourself without per-document SaaS fees
- Your images are reasonably clear photos of printed receipts (thermal receipts, printed invoices, PDFs-rendered-as-images)
- You want the schema in one place so downstream tools (accounting integration, database, spreadsheet export) see the same fields

**Do NOT use when:**
- You need to process a huge volume with strong SLAs — Claude vision has real rate limits and per-call cost; commercial services amortize better at scale
- Your receipts are handwritten, in obscure languages, or scanned at low resolution — accuracy drops sharply
- You need auditable provenance ("which pixel produced this number") — the model does not return bounding boxes
- The extraction feeds a legal/tax filing where hallucinated values would be a real liability — always human-in-the-loop the final total for those

## Where this fails

Specific examples from spot-checks, not generic warnings:

- **Blurry phone photos of thermal receipts** — the ink is already low-contrast; blurring pushes character recognition below usable. Symptom: line-item totals off by an order of magnitude, or missing entirely. Fix: retake the photo, hold the phone steady.
- **Handwritten receipts** — Claude vision handles typed text well but is inconsistent on cursive or messy print. Symptom: `notes` field gets filled with "handwritten portion illegible" or line items are silently skipped.
- **Multi-currency receipts** (e.g. tourist receipts showing local currency + USD) — the `currency` field picks one, sometimes the wrong one. Symptom: totals correct but `currency` field mislabeled. Workaround: pass the expected currency in the prompt (would require exposing it as a param, not currently supported).
- **Very long itemized invoices (30+ line items)** — hits `max_tokens=2048` and gets truncated. Symptom: extraction returns fewer line items than the image contains, but `total` is still correct because it's usually at the bottom. Workaround: raise `DEFAULT_MAX_TOKENS` in `agent.py` — but that also raises per-call cost.
- **PDF input** (currently not supported despite SPEC promising it — tracked in `tasks/todo.md` as F1.5 prerequisite) — silently gets base64-encoded and rejected by Claude with a 400, which surfaces as "Couldn't read this image." Real fix: use Anthropic's `document` content block, planned for a follow-up commit before the real-provider run.
- **Non-receipt images** (a photo of a landscape, a screenshot of a webpage) — Claude returns a `ValidationError` because there's no `total` to extract; the UI shows the raw model output in the warning banner ("this doesn't appear to be a receipt") rather than silently returning zeros.
