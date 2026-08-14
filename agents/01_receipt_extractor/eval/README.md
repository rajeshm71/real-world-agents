# Receipt extractor eval set

This directory is the accuracy eval harness for agent #01, per SPEC.md F1.5:
20 real receipts, hand-labeled ground truth, per-field accuracy measured and
printed in the agent's README once run.

**Status: 20 real cases loaded, not yet run against a real API.** See
"Data source" below. The agent's README does not claim an accuracy number
yet, because none has been measured — running the eval (see "Running the
eval" below) requires a real API key and makes real, billed calls.

## Data source

The 20 images in `images/` (`cord_00.jpg` through `cord_24.jpg`) are drawn
from **[CORD (Consolidated Receipt Dataset)](https://github.com/clovaai/cord)**,
published by NAVER CLOVA AI Research, licensed
**[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**. Real receipts
from restaurants, not synthetic or user-supplied.

Ground truth in `cases.jsonl` was built from CORD's own peer-reviewed
annotations (`total`, `subtotal`, `tax_total`, and line-item count), not
independently re-labeled by hand — CORD's numbers are treated as reliable
source-of-truth for those fields.

**Two honest scope limits, both a property of the source data, not a
shortcut taken here:**

1. **`vendor_name` is not scored for any case.** CORD's public release has
   store names/addresses/logos blurred out by the curators for privacy —
   visible directly in the images themselves. The agent literally cannot
   extract what isn't visible in the source image; scoring it would be
   scoring the redaction, not the extractor.
2. **`currency`, `invoice_date`, `invoice_number`, `due_date`,
   `payment_method` are not scored.** These weren't independently verified
   against each image by a human labeler for this batch — CORD's own
   annotation schema doesn't cover them, and confidently reading them off
   20 images individually was out of scope for this pass. Per
   `run_eval.py`'s scoring contract, an unlabeled field is simply not
   counted — not guessed, not penalized.
3. **All 20 receipts are restaurants, mostly Indonesian Rupiah amounts**
   (a few appear to be from other Southeast Asian countries based on
   visible script). This is real-world data, but it's a narrower slice
   than the "formal invoices, multiple currencies" target composition
   described further down this file — CORD is restaurant-receipt-only.
   A future eval pass could add a second data source for that variety;
   not done here.

Attribution (CC BY 4.0 requires it): images and their `total`/`subtotal`/
`tax_total`/line-item-count ground truth are derived from the CORD dataset,
© NAVER Corp. / CLOVA AI Research, used under CC BY 4.0.

## Adding a case

The 20 CORD-derived cases above only score `total`/`subtotal`/`tax_total`/
`line_items` (see "Data source" scope limits). If you add your own case —
from CORD or elsewhere — and can confidently read `vendor_name`, `currency`,
or the date fields off the image yourself, include them; that's exactly how
the eval set's coverage grows past today's scope limits.

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
