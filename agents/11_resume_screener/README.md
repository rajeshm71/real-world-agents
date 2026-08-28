# Resume screener (batch + ranking with cross-checked evidence)

Point the agent at a job description and N resumes (PDF / DOCX / MD / TXT). Get back a per-candidate scorecard on three dimensions -- skills, experience, culture -- with every score >= 40 grounded in a verbatim excerpt from the resume, plus a ranked list. Real HR/recruiting first-pass triage: the model does the read-every-resume-carefully work, a human still makes the call.

## Verification status

| Path | Status |
|---|---|
| Mock mode (`LLM_PROVIDER=mock`) | Fully covered by the shipped test suite |
| Loaders on real files (pypdf on the shipped `alice.pdf`, python-docx on `bob.docx`, plain read on `carol.md` / `dave.md`) | Fully covered |
| Schema validators (evidence verbatim substring, exactly-three dimensions, overall bounded near mean, recommendation-matches-rubric, ranked-ids permutation, rank order non-increasing) | Fully covered, both directions |
| Batch orchestration (empty resumes, duplicate ids, empty JD, bad provider, deterministic tie-break) | Fully covered |
| Real OpenAI screening (`LLM_PROVIDER=openai` + key) | **Not yet verified against a live API call.** Structural correctness proven via mock + schema tests. Same open-item status as every other agent's first ship. |

## Technique demonstrated

**Batch + ranking with evidence-cross-checked rationale.** Every prior agent processes one input at a time. #11 is the first that takes a batch of N inputs and produces ordered output (ranked list) with per-item structured scorecards -- the shape a real triage tool actually needs.

Three moving parts:

1. **Loaders**: `.pdf` via pypdf, `.docx` via python-docx, `.md` / `.txt` via plain read. Dispatcher on suffix, `ScreenerError` on unsupported format or empty extraction. Filename stem becomes both the `resume_id` (lowercased) and the display `candidate_name` (`alice_smith.pdf` -> "Alice Smith").
2. **Per-candidate scoring**: one PydanticAI `Agent` call per resume, returning a subset schema (`dimensions`, `overall_score`, `overall_rationale`, `recommendation`). Caller injects the known fields (`resume_id`, `candidate_name`, `resume_text`) to build the full `CandidateScorecard`, which Pydantic re-validates with four cross-field checks: evidence must be a verbatim substring of the resume; exactly three canonical dimensions; overall bounded within +/-10 of the dimension mean; recommendation consistent with overall per the shipped rubric.
3. **Ranking**: numeric sort by `overall_score` descending, `resume_id` ascending for deterministic tie-breaks. `ScreeningResult` cross-validates that `ranked_ids` is a permutation of the scorecards' ids and that the order is non-increasing by score.

Distinct from every earlier agent: first batch/ranking pipeline. Reuses PydanticAI (#06) with a different use case (multi-item score-and-rank instead of single-email triage) to show the same framework across two shapes.

## Why this technique for this use case

Real HR first-pass triage is genuinely a "score-and-rank N items" job:

- Corpus size is N candidates, typically 20-200 for one opening. Not too many to score individually.
- Each candidate deserves an evidence-backed rationale, not just a number -- a recruiter needs to know WHY the model ranked Alice above Bob.
- Ranking matters for downstream workflow (who to reach out to first).

PydanticAI over rolling our own JSON contract: at three dimensions * one required excerpt each * four cross-field validators, the validation surface is real. Letting the framework handle structured output frees agent.py to focus on batching, ranking, and loader dispatch.

Where this technique is NOT the right fit: bias-sensitive hiring decisions (see "Where this fails"); very large candidate pools (>1000 -- embedding + retrieval first, then score the top K); situations where dimension weights differ dramatically per role (the fixed rubric assumes similar weighting across dimensions).

## What it does

**`load_resume(path)`** returns a `ResumeInput(resume_id, candidate_name, resume_text)`. Dispatches on suffix.

**`screen_candidates(job_description, resumes, ...)`** returns a `ScreeningResult`:

- `job_description` (echoed)
- `scorecards` (list of `CandidateScorecard`, one per input resume)
- `ranked_ids` (permutation of resume ids, ordered by overall_score desc)
- `run_meta` (provider, model, resume_count, elapsed_seconds)

Under `LLM_PROVIDER=mock`, `screen_candidates` returns scripted scorecards derived deterministically from each `resume_id` (so different resumes get different scores), with evidence sliced verbatim from the resume text. The JD length is echoed into `overall_rationale` as an anti-refactor guard.

## How to run locally

Four commands from a fresh clone (`python -m agent` must run from inside the agent's own directory: `agent` is a submodule of the digit-prefixed `11_resume_screener` package):

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # set OPENAI_API_KEY (or LLM_PROVIDER=mock)
cd agents/11_resume_screener
```

Mock demo (no API key, canned scoring across the shipped example set):

```bash
LLM_PROVIDER=mock uv run python -m agent \
  --jd examples/job_description.md \
  --resumes examples/resumes/alice.pdf examples/resumes/bob.docx examples/resumes/carol.md examples/resumes/dave.md
```

Real screening against the shipped 4-resume demo:

```bash
uv run python -m agent \
  --jd examples/job_description.md \
  --resumes examples/resumes/*.pdf examples/resumes/*.docx examples/resumes/*.md
```

Point at your own JD and any local resumes:

```bash
uv run python -m agent --jd ~/roles/backend.md --resumes ~/inbox/*.pdf
```

Estimated cost per screening at gpt-4o-mini pricing: **well under $0.01 for 4 resumes** (~2K tokens per resume in + ~800 out = ~11K total * gpt-4o-mini's $0.15 in / $0.60 out per 1M tokens). Cost scales linearly with resume count. This is an estimate until a live run pins it.

## Code walkthrough

- `schemas.py`: `ResumeInput`, `EvidenceExcerpt`, `DimensionScore`, `CandidateScorecard` (four cross-field validators), `ScreeningResult` (two cross-field validators). `_recommendation_for(overall_score)` is the single source of truth for the recommendation rubric, shared by the validator and mock generator.
- `agent.py` (`load_resume`, `_load_resume_text`, `_load_pdf_text`, `_load_docx_text`): the loader dispatcher and per-format helpers. Every failure mode raises `ScreenerError` with a `ScreenerAttempt(stage="load", ...)` partial.
- `agent.py` (`screen_candidates`): the batch orchestrator. Validates inputs, dispatches per-resume, ranks numerically, packages into `ScreeningResult`.
- `agent.py` (`_score_one`, `_ScoringOutput`, `_build_agent`): the per-candidate PydanticAI call. Note the subset schema pattern -- LLM returns the four scoring fields; the caller injects the three fields it already knows (`resume_id`, `candidate_name`, `resume_text`) to build the full scorecard.
- `agent.py` (`_translate_api_error`): R5 six-branch translator, same shape as agents #02-#10 and #13.
- `prompts/system.txt`: rubric, evidence rules, no JSON schema (PydanticAI handles the contract natively via structured outputs).

## When to use / When NOT to use

**Use this pattern when:**
- You have N structured items to score independently against fixed criteria.
- You need per-item rationale grounded in the item's own text (evidence, not just numbers).
- Ranking is numeric and transparent (a recruiter can question any position).

**Do NOT use this pattern when:**
- Items score differently based on the other items (cross-item comparison -- v1 is independent).
- The corpus is very large; retrieve then score is a better fit.
- The domain is legally sensitive without a fairness audit in place (see below).

## Where this fails

- **No bias audit.** The screener will happily reproduce whatever biases exist in the model's training data or in the JD's wording. This is a real hiring tool blocker; before using in production, run a fairness benchmark on your own candidate distribution and adjust the rubric or add post-hoc checks. v1 is a technique demonstration, not a compliance-ready product.
- **Image-only PDFs return empty text.** `pypdf.extract_text()` returns `""` for scanned resumes with no OCR layer. The loader catches this with the "extracted to empty text" error rather than silently scoring zero. To handle scanned resumes, add an OCR pass (tesseract, or a vision-capable LLM) before the screener.
- **DOCX with heavy formatting** (tables inside cells, textboxes, embedded objects) may lose structure. python-docx handles paragraphs and simple tables cleanly; more exotic layouts fall through.
- **Resumes that name-drop technologies without evidence.** A resume that says "expert in Kafka" with no supporting project description will score high on skills_match if the model takes the claim at face value. The verbatim-substring evidence check catches paraphrases but not exaggeration.
- **Very long resumes** (10+ pages) hit token budget limits; the JD + resume goes into one LLM call. For long resumes, chunk-and-summarize first.
- **Tie-breaking is by resume_id (lexical), not by quality.** Two candidates at 78/78 will always rank in alphabetical order of their filename stems. Deterministic and fair, but not "smarter."

## Roadmap (post-v1 improvements)

- Parallel per-resume scoring via `asyncio.gather` around `_score_one` (one-line change; skipped in v1 for pedagogy).
- LLM-as-reranker pass over the top K to break tight numeric clusters.
- Cross-candidate comparison mode ("Alice vs Bob on ledger experience...").
- OCR pipeline for image-only PDFs.
- Configurable rubric (dimension names, weights, recommendation cutoffs) via a `rubric.yaml`.
- Fairness / bias audit hooks with a small benchmark of known-equivalent candidates.
