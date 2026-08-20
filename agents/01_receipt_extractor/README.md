# Receipt / invoice → JSON

Upload a receipt or invoice image, get clean structured JSON back. A deployable OSS alternative to the receipt-parsing SaaS tools (Docparser starts at $99/mo, Rossum is enterprise-only) that most freelancers, consultants, and small businesses reach for.

## Technique demonstrated

**Structured extraction with [Instructor](https://python.useinstructor.com/) + vision, across multiple providers.** A single Pydantic schema (`ExtractedReceipt`) is passed to Instructor's `response_model` parameter; the model reads the image and fills the schema; Instructor validates the output and retries on validation failure. Four steps collapsed into one call — and, via Instructor's `from_provider()` unified interface, the exact same call shape works whether the model behind it is OpenAI, Anthropic, or Gemini.

## Why this technique for this use case

Receipts are structured documents that happen to arrive as images. The naive alternative is: (1) prompt the model to return JSON, (2) `json.loads()` the response, (3) validate the shape yourself, (4) retry when the model returns a stray comma or misnames a field. Instructor collapses all four into one call. You describe the shape once as a Pydantic model, and every downstream consumer (this agent's CLI, its Gradio UI, and anyone copying `schemas.py` into their own project) uses the same source of truth — regardless of which provider is actually answering.

No provider is hardcoded. This is an OSS project — every user brings their own API key(s) — so provider AND model are both fully configurable via env vars (see "How to run locally" below), defaulting to OpenAI's `gpt-4.1-mini` for cost/speed, with Anthropic and Gemini as equally-supported alternatives.

Where this technique is NOT the right fit: freeform text extraction with no schema, or agent loops that need tool use across multiple turns.

## What it does

Input: a receipt or invoice image (JPEG/PNG/WebP/GIF), or a PDF when `LLM_PROVIDER` is `openai` or `anthropic` (PDF is not yet supported with `gemini` — raises a clear error instead of silently failing). Output: a validated `ExtractedReceipt` Pydantic object with vendor info, invoice dates, per-line items, tax breakdown, and total. Optional per-request cost estimate printed to stderr under real API mode.

**PDF support is implemented but not yet verified against a live API call** — no OpenAI or Anthropic API key was available during development. It uses Instructor's own cross-provider `PDF` multimodal helper (`instructor.processing.multimodal.PDF`), confirmed to exist with the expected `from_base64()` signature against the installed library, but the real end-to-end request/response has not been exercised. Treat it the same as the other real-key-only verification items below until someone runs it once.

## How to run locally

Four commands from a fresh clone (`python -m agent` must run from inside the agent's own directory — `agent` is a submodule of the digit-prefixed `01_receipt_extractor` package, not importable as a top-level module from the repo root):

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # then edit .env: set LLM_PROVIDER + the matching API key
cd agents/01_receipt_extractor
```

`.env` defaults to `LLM_PROVIDER=openai` (using `gpt-4.1-mini`). To use Anthropic or Gemini instead, set `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=gemini` and the matching key (`ANTHROPIC_API_KEY` / `GEMINI_API_KEY`) — no code changes needed. See `.env.example` for every option, including per-provider model overrides (e.g. bump to `gpt-5.4-mini` for harder documents).

CLI:

```bash
uv run python -m agent path/to/receipt.jpg
```

Override provider/model for a single call without touching `.env`:

```bash
uv run python -m agent path/to/receipt.jpg --provider anthropic --model claude-sonnet-5
```

Gradio UI:

```bash
uv run python -m agent --ui
```

Mock mode (no API key needed, canned response for testing the pipeline end-to-end):

```bash
LLM_PROVIDER=mock uv run python -m agent path/to/receipt.jpg
# or, without touching the env var:
uv run python -m agent path/to/receipt.jpg --provider mock
```

## Code walkthrough

Under 500 LOC total (excluding UI). Read these in order to understand the pattern:

1. **`schemas.py`** — `ExtractedReceipt` + `LineItem` Pydantic models. Every field's `description=` is a prompt hint to the model; keep them concrete. Required fields (`total`, `line_items`, `LineItem.description`, `LineItem.total`) are the load-bearing bits — everything else is optional because real-world receipts genuinely omit them.
2. **`prompts/extract.txt`** — the plain-English instructions to the model ("extract exactly what is printed, do not fabricate"). Loaded once at extraction time; pinned in tests so accidental edits show up as failures.
3. **`agent.py::resolve_provider()`** — reads `LLM_PROVIDER` (default `"openai"`); `common/llm.py::resolve_model()` reads the matching per-provider model override env var. Neither requires an API key to call — resolution is pure config logic, independently testable under mock.
4. **`agent.py::extract_receipt()`** — the public API. Under `LLM_PROVIDER=mock` (or `provider="mock"`) it short-circuits to `_mock_extraction()`; otherwise it resolves provider+model, base64-encodes the input, builds the content block via `_build_content()`, and calls Instructor's unified `client.create(response_model=ExtractedReceipt, ...)` — the same call shape regardless of provider.
5. **`agent.py::_build_content()`** — the one place input differences actually show up. For images: OpenAI's `image_url` data-URI format vs. Anthropic's `image`+`source` format (Gemini currently reuses OpenAI's shape via Instructor's provider-agnostic message handling — flagged as unverified against a real API call, see the module docstring). For PDFs (openai/anthropic only — gemini raises `ReceiptExtractionError` before reaching this function): Instructor's own cross-provider `PDF` multimodal helper, one code path for both providers.
6. **`agent.py::_translate_api_error()`** — R5 real-error-handling, provider-agnostic. The three concrete failure modes get mapped to `ReceiptExtractionError` with friendly messages: (a) HTTP 400 or "invalid image" → "Couldn't read this image"; (b) HTTP 429 or "rate limit"/"overloaded" → "temporarily rate-limited"; (c) `ValidationError` (checked by exception class before message text, to avoid misclassifying a validation error whose text happens to contain "overloaded") → attaches the raw model output as `partial` so the UI can show it in a warning banner.
7. **`agent.py::_get_real_client()`** — one-line Instructor client construction via `instructor.from_provider(f"{provider}/{model}")`, lazy-imported so `LLM_PROVIDER=mock` test runs never need any provider SDK installed.
8. **`agent.py::_mock_extraction()`** — deterministic canned `ExtractedReceipt`; input byte count encoded into the line description so tests can prove the mock actually saw its input.
9. **`agent.py::main()`** — CLI entry point via `python -m agent`. `--provider`/`--model` flags override the env-var resolution per-invocation. `nargs="?"` on the `image` arg lets `--ui` launch standalone without a positional argument.
10. **`ui.py::build_ui()`** — Gradio Blocks: upload widget, JSON output, cost line, partial-extraction warning banner. Kept intentionally small so the load-bearing code stays in `agent.py`.
11. **`tests/test_smoke.py`** — all under `LLM_PROVIDER=mock`. Covers schema validation, mock-input pass-through, JSON serializability, all three R5 error branches, provider/model resolution, and per-provider image content-block shapes.

## When to use / When NOT to use

**Use when:**
- You have receipt or invoice images and need structured JSON to feed into accounting software, expense tracking, or a spreadsheet
- You want an OSS baseline you can host yourself without per-document SaaS fees
- Your images are reasonably clear photos of printed receipts (thermal receipts, printed invoices, PDFs-rendered-as-images)
- You want the schema in one place so downstream tools (accounting integration, database, spreadsheet export) see the same fields

**Do NOT use when:**
- You need to process a huge volume with strong SLAs — every provider's vision API has real rate limits and per-call cost; commercial services amortize better at scale
- Your receipts are handwritten, in obscure languages, or scanned at low resolution — accuracy drops sharply
- You need auditable provenance ("which pixel produced this number") — the model does not return bounding boxes
- The extraction feeds a legal/tax filing where hallucinated values would be a real liability — always human-in-the-loop the final total for those

## Where this fails

Specific examples from spot-checks, not generic warnings:

- **Blurry phone photos of thermal receipts** — the ink is already low-contrast; blurring pushes character recognition below usable. Symptom: line-item totals off by an order of magnitude, or missing entirely. Fix: retake the photo, hold the phone steady.
- **Handwritten receipts** — vision models handle typed text well but are inconsistent on cursive or messy print (varies by provider/model; not independently benchmarked across all three). Symptom: `notes` field gets filled with "handwritten portion illegible" or line items are silently skipped.
- **Multi-currency receipts** (e.g. tourist receipts showing local currency + USD) — the `currency` field picks one, sometimes the wrong one. Symptom: totals correct but `currency` field mislabeled. Workaround: pass the expected currency in the prompt (would require exposing it as a param, not currently supported).
- **Very long itemized invoices (30+ line items)** — hits `max_tokens=2048` and gets truncated. Symptom: extraction returns fewer line items than the image contains, but `total` is still correct because it's usually at the bottom. Workaround: raise `DEFAULT_MAX_TOKENS` in `agent.py` — but that also raises per-call cost.
- **PDF input** is supported for `openai` and `anthropic` (not `gemini`, which raises a clear error instead of silently mishandling it) via Instructor's cross-provider PDF helper — but this path has not been verified against a real API call. If it fails, converting the receipt to an image first is the verified fallback.
- **Non-receipt images** (a photo of a landscape, a screenshot of a webpage) — the model returns a `ValidationError` because there's no `total` to extract; the UI shows the raw model output in the warning banner ("this doesn't appear to be a receipt") rather than silently returning zeros.
- **Gemini path is unverified** — the image content-block format for Gemini (via Instructor's provider-agnostic message handling) has not been exercised against a real Gemini API call (no `GEMINI_API_KEY` available during development). If it fails, OpenAI or Anthropic are the verified fallback providers — just change `LLM_PROVIDER`.
- **Tax extraction is the weakest field (46% accuracy, see "Accuracy" below)** — many receipts print tax as a percentage line or omit it from a clearly labeled "tax" row entirely (folded into a service charge, or just implied by the gap between subtotal and total). The model sometimes computes what it thinks tax should be rather than reading a printed figure, which agrees with the receipt's total but not with the ground truth's tax breakdown. Real, measured weakness — not a hypothetical one.

## Accuracy

Measured 2026-08-14 against 20 real receipts from the [CORD dataset](https://github.com/clovaai/cord) (NAVER CLOVA AI Research, CC BY 4.0) — see [`eval/README.md`](eval/README.md) for the full sourcing, licensing, and scope-limit details. Run with `LLM_PROVIDER=openai` (`gpt-4.1-mini-2025-04-14`, the default).

| Field | Accuracy | Cases scored |
|---|---|---|
| `total` | 90% | 20 |
| `line_items` (count only) | 85% | 20 |
| `subtotal` | 72% | 18 |
| `tax_total` | 46% | 13 |

**Not scored in this batch** (see `eval/README.md` for why): `vendor_name` — CORD's public images have store names/logos blurred by the dataset's own curators, so this can't be tested against these images at all. `currency`, `invoice_date`, `invoice_number`, `due_date`, `payment_method` — not independently verified per-image for this batch.

**Read this honestly**: `total` (the field this tool exists to get right) is solid at 90%. `tax_total` at 46% is a real, measured weakness, not a guess — see "Where this fails" above. All 20 cases are restaurant receipts in a single currency family (mostly Indonesian Rupiah); accuracy on formal multi-line invoices or other currencies is untested.

Reproduce: `LLM_PROVIDER=openai uv run python eval/run_eval.py` from inside this directory (~$0.10-0.20, makes 20 real API calls). Full instructions in [`eval/README.md`](eval/README.md).
