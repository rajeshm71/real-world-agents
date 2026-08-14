# Receipt extractor eval set

This directory is the accuracy eval harness for agent #01, per SPEC.md F1.5:
20 real receipts, hand-labeled ground truth, per-field accuracy measured and
printed in the agent's README once run.

**Status: harness built, not yet run.** `cases.jsonl` in this directory is
empty except for illustrative examples — it does not contain 20 real cases
yet. Real receipts + a real API key are needed to actually run this (see
"Running the eval" below); until then the agent's README does not claim an
accuracy number, because none has been measured.

## Adding a case

1. Add the receipt image to `images/` (JPEG/PNG/WebP/GIF).
2. Add one line to `cases.jsonl` — a JSON object with:
   - `case_id`: a short unique string, e.g. `"restaurant_01"`
   - `image_path`: filename relative to `images/`, e.g. `"restaurant_01.jpg"`
   - `expected`: the ground-truth fields, using the same field names as
     `schemas.ExtractedReceipt` (see `../schemas.py`). Only include fields
     you're confident about — fields omitted from `expected` are not scored
     for that case, so it's fine to leave uncertain fields out rather than
     guess.

Example:

```json
{"case_id": "restaurant_01", "image_path": "restaurant_01.jpg", "expected": {"vendor_name": "Blue Bottle Coffee", "currency": "USD", "total": 12.50}}
```

### Privacy

Anonymize real receipts before committing them: replace real names/addresses
with fake-but-plausible ones, but keep the STRUCTURE real (same layout,
same kind of line items, same currency/formatting quirks) — the whole point
is to measure accuracy against realistic documents, not synthetic ones.
Never commit a receipt with real personal or financial information.

### Target composition (per SPEC.md §16 F1.5)

20 cases, ideally spanning: a few restaurant/retail thermal receipts, a few
formal invoices (with tax breakdowns and multiple line items), at least one
non-USD currency, and at least one deliberately hard case (blurry, unusual
layout) to make the accuracy number honest rather than cherry-picked.

## Running the eval

Requires a real API key (see `.env.example` at the repo root) — this makes
real, billed API calls, one per case. Estimated cost: ~$0.005-0.01 per
receipt (see the agent's README "Cost note"), so a 20-case run costs
roughly $0.10-0.20.

```bash
cd agents/01_receipt_extractor
LLM_PROVIDER=openai uv run python eval/run_eval.py
```

Prints a per-field accuracy table to stdout and writes it to
`eval/results.md`. Once you have real results, copy the table into the
agent's README as its optional "## Accuracy" section (§8.1 section 11).

## Scope note

Per-field accuracy is computed for scalar fields (`vendor_name`, `currency`,
`total`, `subtotal`, `tax_total`, `invoice_number`, `invoice_date`,
`due_date`, `payment_method`). `line_items` is scored by count only (does
the extracted line-item count match the expected count) — not per-item
field accuracy. Per-item scoring would need a matching strategy (line items
aren't naturally ordered/keyed) that's out of scope for this first pass;
tracked as a possible future improvement, not a current gap that blocks F1.5.
