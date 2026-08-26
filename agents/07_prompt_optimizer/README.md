# Prompt optimizer -> meta-tune agent #01's receipt prompt via DSPy

Run DSPy's GEPA optimizer against agent #01's hand-written receipt-extraction prompt and its 20-case CORD eval set. Get a new prompt back that scores better on the same eval, plus a per-field before/after accuracy table so you can see WHERE the optimization helped (and what it left flat). A deployable demo of "stop hand-tuning prompts; let a numeric metric do it" that anyone with a scored task can copy the pattern from.

## Technique demonstrated

**Meta-optimization with [DSPy](https://dspy.ai/)'s GEPA optimizer.** Instead of a human iterating on a prompt string by hand, the reader:

1. Describes the task as a `dspy.Signature` (typed input/output fields).
2. Writes a metric function that scores model outputs against ground truth.
3. Points `dspy.GEPA(metric=..., auto="light")` at an eval set.
4. Gets back a new prompt string that outperforms the hand-written one on that eval.

The winner is a plain string. You feed it back into whatever agent originally used the hand-written prompt (agent #01 here). Distinctly different from every other technique in the catalog: #01 is structured extraction, #02 is a hand-rolled retry loop, #03 is a LangGraph state machine, #04 is ReAct with grounded tools, #05 is multi-agent collaboration, #06 is type-safe deps. #07 is the FIRST agent that *learns* the prompt rather than executing it.

## Why this technique for this use case

Prompts drift. Domains change, models get updated, edge cases surface. Hand-tuning is tedious and easy to get wrong: a change that helps `tax_total` regresses `total`, a wording that improves formal invoices tanks on thermal receipts. If a task already has a numeric eval set (agent #01 does: 20 receipts, per-field accuracy), turning that signal into an optimization loop is nearly free.

GEPA is DSPy 3.x's currently-recommended optimizer (successor to MIPROv2 for most use cases). It uses reflective evolution over prompt candidates and Pareto candidate selection, which handles the "improve tax without regressing total" trade-off explicitly rather than optimizing a scalar sum.

Where this technique is NOT the right fit: (a) tasks with no numeric metric (freeform generation, creative writing, "does this look good"); (b) tasks where a single hand-written prompt already scores near-perfect (no headroom, optimizer just adds cost); (c) tasks whose eval set is too small to distinguish prompts (below ~10 cases, noise dominates signal).

## What it does

Input: nothing user-supplied. The agent reads agent #01's `prompts/extract.txt` from disk and its `eval/cases.jsonl` (both via the workspace layout), runs a compilation pass, and returns an `OptimizationResult` Pydantic object with:

- The original + optimized prompt strings.
- Per-field baseline + optimized accuracy (fractions 0.0-1.0) for the 4 fields agent #01's eval actually scores: `total`, `subtotal`, `tax_total`, `line_items` (count-only).
- A per-field `improvement` map (delta computed by a model validator).
- The optimizer identity, provider, model, wall-clock time, LLM-call count, and rough dollar cost.

Under `LLM_PROVIDER=mock` the whole thing returns a canned result instantly, with a plausible-looking improvement on `tax_total` and `subtotal` but `total` held flat (matching the load-bearing-field constraint). Under a real provider, it fires a real GEPA compile.

## How to run locally

Four commands from a fresh clone (`python -m agent` must run from inside the agent's own directory: `agent` is a submodule of the digit-prefixed `07_prompt_optimizer` package):

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # set LLM_PROVIDER + the matching API key (or mock)
cd agents/07_prompt_optimizer
```

Mock demo (no API key, canned result):

```bash
LLM_PROVIDER=mock uv run python -m agent
```

Real optimization run (spends real money):

```bash
uv run python -m agent --provider openai --effort light
```

Effort levels roughly: `light` ~30 LLM calls / ~$0.30, `medium` ~150 / ~$1.50, `heavy` ~500 / ~$5.00 at `gpt-4.1-mini` pricing. Verify against your own bill before trusting these estimates. Override the model with `--model gpt-5.4-mini-2026-03-17` for harder documents; per-run cost scales.

Subsample cases for a cheaper first run:

```bash
uv run python -m agent --provider openai --effort light --max-cases 5
```

Gradio UI (both real + mock, provider togglable in-UI):

```bash
uv run python -m agent --ui
```

Every real CLI run writes the full `OptimizationResult` JSON to `last_run.json` next to `agent.py` (gitignored) so a caller can pipe it into other tools.

## Code walkthrough

Under 500 LOC excluding UI. Read these in order:

1. **`schemas.py`**: `OptimizationResult` Pydantic model. `baseline_accuracy` + `optimized_accuracy` are open `dict[str, float]` maps so a future change in agent #01's scored-field list doesn't cascade a schema change here. `improvement` is auto-computed by a `@model_validator`; a missing side is treated as 0.0 rather than a phantom drop.
2. **`agent.py::SCORED_FIELDS`**: the load-bearing contract with agent #01's eval harness. Adding a field here without also updating the Signature + metric will silently break optimization; pinned in tests.
3. **`agent.py::optimize_prompt()`**: the public API. Under `provider="mock"` short-circuits to `_mock_optimization()` (no dspy import, no key). Real path: load agent #01's prompt + cases → lazy-import dspy → build Signature + Predict program → convert cases to `dspy.Example`s → run GEPA.compile() → rescore both prompts on the full case set. R5 branches wrap load-cases + dspy-import + API-error paths.
4. **`agent.py::_build_program()`**: **THE pedagogical anchor.** Constructs a `dspy.Signature` with `dspy.Image` for input + the 4 scored fields as output. Seeds the Signature's docstring (which becomes GEPA's initial instructions) with agent #01's real prompt, so the optimizer has a known-good starting point rather than the empty string.
5. **`agent.py::_build_metric()`**: builds the GEPA-shaped metric callable (`(gold, pred, trace, pred_name, pred_trace, program_trace=None) -> float`). Only scores fields gold actually labels; asymmetric handling of `line_items_count=0` (valid label) vs missing (unscored). Wraps agent #01's `run_eval.score_case` verbatim; the money-tolerance and count-only-line-items rules come along for free.
6. **`agent.py::_cases_to_dspy_examples()`**: converts loaded CORD cases to `dspy.Example` objects with `dspy.Image.from_path()` inputs. 50/50 train/val split -- with 20 total cases, anything more generous starves the trainset, and GEPA's Pareto candidate selection doesn't need a huge val set.
7. **`agent.py::_score_prompt()`**: runs a Predict program with a given prompt-as-docstring across every case and aggregates via agent #01's `run_eval.aggregate_accuracy`. Used to rescore baseline + optimized after compilation. This is an LLM-billing hot path: costs `len(cases)` extra calls per prompt scored (so ~40 extra calls total for a full rescore).
8. **`agent.py::_extract_prompt_from_compiled_program()`**: walks compiled.predict.signature.instructions to extract the optimized prompt string. Falls back to a direct `.signature.instructions` for programs whose Predict is at the top level. Raises `PromptOptimizerError` (not a mysterious AttributeError) if DSPy's API drifts.
9. **`agent.py::_translate_api_error()`**: R5 case 3. Six-branch priority order (class-name → status → message → generic), same shape as agents #02-#06. This agent does NOT auto-retry rate-limit errors: compile runs are already expensive and silent retries could double the bill.
10. **`ui.py::build_ui()`**: Gradio Blocks: provider Radio + effort Radio + Run button. Right column: per-field before/after accuracy markdown table + side-by-side prompt code blocks + collapsible full-JSON accordion for the last CLI run. Kept intentionally small; the load-bearing code stays in `agent.py`.
11. **`tests/test_smoke.py`**: 34 tests, all under `LLM_PROVIDER=mock`. Covers mock round-trip, schema validator math, all R5 error branches, `_translate_api_error`'s 6 priority paths, the metric function's edge cases (money tolerance, unlabeled-field asymmetry, empty gold), and structural guards on `_build_program` (Signature exposes exactly the 4 scored fields + 1 input) + `_extract_prompt_from_compiled_program` (three shape variants). DSPy paths gated behind `pytest.importorskip("dspy")` so CI works without it installed (same pattern as agent #05's crewai gate).

## When to use / When NOT to use

**Use when:**
- Your task has a real numeric metric (an eval set with ground truth) and you're currently hand-tuning the prompt
- You have at least ~10 labeled cases -- below that, noise dominates signal
- You're willing to spend real dollars on a compile pass; the winner then runs against your production traffic for free
- The task is stable enough that the winning prompt is worth investing in (not a one-off classifier for a demo)

**Do NOT use when:**
- Your task has no numeric metric (creative writing, freeform generation, "does this look good to a human"): GEPA has nothing to optimize toward
- Your hand-written prompt already scores near the ceiling (agent #01's `total` at 90% -- there's ~10 points of headroom and it's the load-bearing field; the model may not have any left)
- Your eval set is too small (<10 cases): the optimizer will overfit to your specific examples and generalize poorly to unseen documents
- You're just exploring a new task: hand-write a prompt first, measure the baseline, THEN decide if a compile pass is worth it

## Where this fails

Specific, honest failure modes:

- **CORD-set overfitting**: the 20 CORD cases are Indonesian Rupiah restaurant receipts with no decimals and blurred vendor names. GEPA will happily produce a prompt that assumes IDR-only large-integer amounts and tanks on USD/EUR real receipts with decimals. The Signature deliberately excludes `vendor_name` (unextractable due to CORD blurring) and unscored fields, but it can't fix the IDR bias. Real fix: expand the eval set to include more currencies before running a serious optimization pass.
- **Metric-hacking risk**: GEPA optimizes the metric, not the underlying task. If the metric weights all four scored fields equally, the optimizer will happily trade a `total` point (load-bearing) for two `tax_total` points (currently 46% -> lots of headroom). The included `_metric()` treats fields equally by default; a real production run should weight `total` higher. Documented but not enforced -- the reader who runs this needs to know this tradeoff.
- **Rescore cost sneaks up**: the final rescore of both prompts across all 20 cases costs `2 × 20 = 40` extra LLM calls beyond GEPA's headline compile-call count. That's a real cost the effort-level estimates DO include, but if you subsample with `--max-cases 5` for a cheap run, you're still paying `2 × 5 = 10` rescore calls after the ~30 compile calls.
- **Model shift breaks the winning prompt**: a prompt optimized against `gpt-4.1-mini` may score worse on `claude-sonnet-5` or `gpt-5.4-mini`. If you swap the underlying model, re-run the optimizer.
- **DSPy 3.x version drift**: the prompt extraction helper walks `compiled.predict.signature.instructions`. If a future DSPy release renames any of those attributes, the helper raises a clear `PromptOptimizerError` pointing at the extraction function, but the compile itself may still return a program you can't use. Pin DSPy explicitly (`dspy>=3.3` in `pyproject.toml`) and re-verify after a bump.
- **GEPA has been "verified" but not "run" here**: the code paths are structurally correct against DSPy 3.3.1 (Signature/Predict/GEPA/Image/LM APIs all verified via `inspect.signature`), but a real end-to-end compile pass with a live API key hasn't been exercised yet. Treat this like every other agent's initial ship: works under mock, needs a first real run to shake out any provider-specific quirks.
- **Rate limit / API failure during compile**: no auto-retry with backoff (explicit design decision, see `agent.py::_rate_limit_error`). One failed attempt during the compile pass loses all the optimizer's partial progress, and the user has to re-run. Trade: no silent budget-burning on multi-dollar compile runs; downside: any transient blip wastes whatever's already spent.
