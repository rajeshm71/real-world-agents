# Contact / ID extractor (batch vision + dedup + PII-safe)

Give the agent one photo of a stack of business cards or IDs. Get back a deduplicated list of contacts, with sensitive fields optionally hashed or redacted so the on-disk dump is compliance-safe. The catalog's final agent, and its most compliance-shaped one: vision extraction, embedding-based dedup, and per-field PII policy layered into a single pipeline.

## Verification status

| Path | Status |
|---|---|
| Mock mode (`LLM_PROVIDER=mock`) | Fully covered by the shipped test suite |
| Dedup logic (email + phone corroboration, transitive merges, low-similarity guards) | Fully covered against REAL fastembed on synthetic identity strings |
| PII redactor (none / hash / redact per field; salted-hash stability; strict / redacted presets) | Fully covered |
| Schema validators (email shape, hash-marker shape, redact-sentinel, merged_from partition, duplicate index bounds) | Fully covered, both directions |
| Real OpenAI vision extraction (`LLM_PROVIDER=openai` + key on `examples/cards.png`) | **Not yet verified against a live API call.** Structural correctness proven via mock + validator tests. Same open-item status as every other agent's first ship. |

## Technique demonstrated

**Batch vision extraction + embedding-based dedup + PII-safe redaction, layered into a compliance-shaped pipeline.**

- **Batch vision** distinguishes #15 from #01 (single receipt) and #09 (single screenshot): the model returns an ordered LIST of contacts, one per visible card, and `card_index` uniqueness is validated at parse time.
- **Local embedding dedup** reuses fastembed (from #10) but for a completely different job: normalized identity strings ("name | company | title") get 384-dim embeddings; pairs merge under union-find only when cosine similarity >= threshold AND a deterministic key (normalized email or last-10-digit phone) matches. That "sim + corroborating key" gate matters: shared-email-different-people (typo, autocomplete slip) is a real failure mode that pure embedding-only or pure key-only dedup gets wrong.
- **PII policy** is per-field. `"none"` passes through; `"hash"` produces a compact SHA-256(salt:normalized) marker + last-4 visible so the record stays audit-readable without leaking raw PII; `"redact"` replaces with `[REDACTED]`. `run_salt` defaults to a fresh per-run `secrets.token_hex(8)` so cross-run correlation is not trivial; callers who need audit-trail hash stability pass a persistent salt.

`ExtractionResult.raw_contacts` are also redacted per policy, so `last_run.json` never contains raw PII on a strict-policy run. Callers who genuinely need pre-redacted raw contacts call `_extract_raw()` directly rather than the public `extract_contacts()`.

## Why this technique for this use case

Real KYC / CRM ingestion is exactly the shape this pipeline fits:

- One photo of a stack of cards -> N structured contacts is the batch shape.
- The same person appearing twice (two cards, front and back, or two versions across years) is the standard dedup case.
- PII controls are legally required, not optional; a demo that dumps raw phone numbers to disk under a "compliance" banner isn't credible.

Where this technique is NOT the right fit: photos where cards overlap heavily (vision struggles); non-English cards (the model's extraction quality drops); dedup across very different photos (v1 dedupes within one photo only -- follow-up needed).

## What it does

**`extract_contacts(image, *, pii_policy, dedup_threshold, run_salt, ...)`** returns an `ExtractionResult`:

- `raw_contacts` -- the vision model's per-card output, with PII redacted per policy.
- `contacts` -- post-dedup canonical contacts with `merged_from` pointing at the raw card_indexes that collapsed into each.
- `duplicates` -- per-merge report: canonical index, member indexes, human-readable merge reason.
- `pii_policy` -- echoed so the dump is self-describing.
- `run_meta` -- provider, model, raw count, contact count, duplicate group count, per-stage elapsed.

Under `LLM_PROVIDER=mock`, the agent returns three scripted raw contacts (two the same person, one unique), runs real dedup logic, and applies the policy -- so mock mode exercises stages B and C for real.

## How to run locally

Four commands from a fresh clone:

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # set OPENAI_API_KEY (or LLM_PROVIDER=mock)
cd agents/15_contact_extractor
```

Mock demo (no API key, canned 3-card result with real dedup):

```bash
LLM_PROVIDER=mock uv run python -m agent examples/cards.png
```

Real extraction with compliance-safe defaults (all PII hashed):

```bash
uv run python -m agent examples/cards.png --policy strict
```

Per-field override -- keep email raw, hash the phone, redact the address:

```bash
uv run python -m agent examples/cards.png --pii email=none --pii phone=hash --pii address=redact
```

Point at your own photo:

```bash
uv run python -m agent ~/photos/cards.jpg --policy strict --run-salt my_stable_kyc_salt
```

Estimated cost per real run at `gpt-4o-mini` vision pricing: **~$0.001 per image** for a photo with 3-10 cards (~one image + ~500 output tokens). Well under a cent per photo. This is an estimate until a live run pins it.

## Code walkthrough

- `schemas.py`: `RawContact` (email-shape validator), `PiiPolicy` (with `.strict()` / `.redacted()` presets), `Contact` (hashed-field-shape + redact-sentinel-shape validators so the redactor can't silently drift), `DuplicateGroup` (canonical must be a member), `ExtractionResult` (raw_index uniqueness, merged_from partition, duplicate-index bounds). `_HASH_MARKER_RE` + `REDACT_SENTINEL` are the shipped contracts the validators check.
- `agent.py` (`_read_image`, `_extract_raw`): stage A. Loads image bytes with a 20 MB cap (OpenAI vision's limit), base64-encodes into a `data:image/{format};base64,...` URL, calls Instructor with `response_model=list[RawContact]` for structured output.
- `agent.py` (`_deduplicate`, `_identity`, `_normalize`, `_normalize_phone`, `_cosine_sim`, `_field_completeness`): stage B. fastembed encodes normalized identity strings; pairwise cosine sim with corroborating-key match; union-find merge; canonical pick by field completeness.
- `agent.py` (`_apply_policy`, `_apply_policy_to_raws`, `_redact_field`): stage C. Per-field policy application; RawContact redaction path zeroes email under hash/redact to avoid tripping the email-shape validator (the redacted view lives on `Contact`).
- `agent.py` (`_translate_api_error`): R5 six-branch translator, matching every prior agent.
- `prompts/system.txt`: rubric with the KYC-safety no-DOB no-ID-number rule.

## When to use / When NOT to use

**Use this pattern when:**
- Your input is a batch of visually-similar items (cards, receipts, ID docs) and you need one structured record per item plus dedup.
- PII compliance is load-bearing and you want a config-driven policy rather than ad-hoc redaction in downstream code.
- You want deterministic, explainable dedup (embeddings for fuzzy match; deterministic key for confirmation).

**Do NOT use this pattern when:**
- You need to extract sensitive ID fields (DOB, national ID number, biometrics). The prompt explicitly refuses these. Add them back at your own risk and layer additional controls (encryption at rest, access audit).
- You need cross-run dedup against an existing contacts DB. v1 dedupes only within the current photo.
- Your cards are in a script the embedding model handles poorly (bge-small-en is English-centric); the dedup step will under-merge.

## Where this fails

- **Vision misses partially-obscured cards.** If a card is >30% occluded, `gpt-4o-mini` may skip it entirely or hallucinate the missing fields. Cross-check the extracted count against your visual count for any photo you care about.
- **Dedup gate is conservative on purpose.** Two cards with the same email but different name/company (typo-shared address, family sharing a contact email) will NOT merge. This is the right default for a first-pass triage; downstream can force merges if it has more context.
- **Accent / transliteration equivalence is not applied.** "M" and "Muller" are distinct identity strings for the dedup embedder. Silently collapsing accents is a real fairness bug (risks merging distinct people), so v1 deliberately does not do it. If your corpus needs it, layer your own normalization before calling `_deduplicate`.
- **Hashed fields are salted per-run by default.** Cross-run correlation ("did I see this email last week?") requires passing a persistent `run_salt`. Without it, the same email in two different runs produces two different hashes.
- **20 MB image cap.** Larger images raise `ContactError` before the vision call. Downscale before sending.
- **English-centric bge model.** Non-Latin scripts underperform in the dedup similarity step (though the vision model itself handles many scripts).
- **International phone dedup uses the last 10 digits.** A UK number and a US number whose last-10 digits happen to match would be treated as a phone match. In practice the dedup gate also requires cosine similarity of the name+company string, so a full false merge is extremely unlikely, but note the underlying key is not globally unique.

## Roadmap (post-v1 improvements)

- Multi-provider vision (Anthropic Claude, Google Gemini) behind the same `_extract_raw` API.
- Cross-run dedup against an existing contacts DB (accept a `known_contacts` argument).
- OCR fallback when vision struggles (Tesseract, PaddleOCR).
- Per-contact confidence score.
- Configurable accent-normalization for the dedup step (opt-in).
- Address geocoding / standardization.
- Additional PII fields (SSN, national ID) behind explicit opt-in flags.
