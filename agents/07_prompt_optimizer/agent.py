"""Prompt optimizer: agent #07 of real-world-agents.

Technique demonstrated: **meta-optimization with DSPy's GEPA optimizer.**
Instead of hand-writing a prompt and hoping for the best, describe the
task as a `dspy.Signature`, write a metric that scores model outputs,
point GEPA at an eval set, and let the optimizer generate + iterate on
prompts. The winner is a plain-string prompt that can be fed back into
whatever agent originally used the hand-written one.

Why this technique for this use case: prompts drift over time as domains
change, models get updated, and edge cases surface. Hand-tuning them is
tedious and easy to get wrong (a change that helps one field regresses
another). If the task already has a numeric eval set (as agent #01 does,
with its 20 CORD receipts + per-field accuracy scoring), turning that
signal into an optimization loop is nearly free.

Target: agent #01's `prompts/extract.txt` against its own CORD eval set.
The DSPy Signature exposes ONLY the four fields that eval set actually
scores (`total`, `subtotal`, `tax_total`, `line_items` count) so the
optimizer can't chase fake signal on unscored / unextractable fields --
CORD blurs vendor names, several fields aren't labeled, and line-item
descriptions are placeholder strings; a naive signature would let the
optimizer overfit to noise.

Real error handling (three cases):
  1. dspy not installed -> PromptOptimizerError with a "uv sync" pointer.
  2. Cases file missing / malformed -> PromptOptimizerError with the exact
     path; delegates to load_cases's own validation.
  3. LLM API failure during compile -> six-branch translator (same
     class-name -> status -> message priority as agents #02-#06).

Provider + model are fully user-configurable via `LLM_PROVIDER` and the
per-provider `_DEFAULT_MODEL` env vars. Under `LLM_PROVIDER=mock` the
agent returns a deterministic canned OptimizationResult without ever
importing dspy, so CI + local exploration work without an API key OR
dspy installed. Real runs cost ~$0.30 at auto="light" / ~$1.50 medium /
~$5 heavy; see README for the details.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Dual-mode import: relative resolves when loaded as a package submodule
# (by tests); absolute resolves when run directly via `python -m agent`.
try:
    from .schemas import OptimizationResult
except ImportError:
    from schemas import OptimizationResult

# common.llm is the root workspace package -- never a relative import.
from common.llm import resolve_model

# --- Provider + constants --------------------------------------------------

SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini", "ollama")
# DSPy uses LiteLLM under the hood, which expects "<provider>/<model>"
# strings. Same mapping shape agent #01 uses for Instructor's
# from_provider() prefix. Only "gemini" differs from our env-var name
# (LiteLLM calls it "gemini/", not "google/"). "ollama_chat" routes
# through LiteLLM to Ollama's chat completions endpoint (supports
# tool-calling; the older "ollama/" prefix does not).
_LITELLM_PROVIDER_PREFIX = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "ollama": "ollama_chat",
}

# Fields the target agent (#01) actually scores. Signature MUST NOT
# expose anything else, or the optimizer will chase noise on fields
# with no ground truth to compare against.
SCORED_FIELDS = ("total", "subtotal", "tax_total", "line_items")

# Effort settings for GEPA's `auto` parameter. Rough call-count and
# cost estimates from the DSPy 3.3 docs; verify against your own bill
# before trusting them.
OPTIMIZER_EFFORT_LEVELS = ("light", "medium", "heavy")
_EFFORT_CALL_ESTIMATES = {"light": 30, "medium": 150, "heavy": 500}
_EFFORT_COST_ESTIMATES_USD = {"light": 0.30, "medium": 1.50, "heavy": 5.00}


# --- Error type ------------------------------------------------------------


@dataclass
class OptimizationAttempt:
    """Partial state attached to PromptOptimizerError when a compile pass
    fails partway through. `partial_prompt` is the best candidate the
    optimizer found before erroring out; empty if it failed before any
    candidate was generated."""

    stage: str  # 'load_cases' / 'build_program' / 'compile' / 'rescore'
    partial_prompt: str = ""


class PromptOptimizerError(Exception):
    """Raised on any user-facing failure: missing dep, bad cases file,
    API error during compile, or optimizer producing an invalid result."""

    def __init__(self, message: str, partial: OptimizationAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Provider resolution ---------------------------------------------------


def resolve_provider() -> str:
    """LLM_PROVIDER env var, defaulting to "openai". No provider is
    hardcoded; every provider is equally supported. "mock" is handled by
    the caller (optimize_prompt), not here."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}."
        )
    return provider


# --- Locating agent #01's assets ------------------------------------------


def _receipt_extractor_dir() -> Path:
    """Return the on-disk path to agent #01's directory. Walks up from
    this file's location so the answer works whether the agent is
    imported as a workspace member or run from its own dir."""
    return Path(__file__).resolve().parent.parent / "01_receipt_extractor"


def _load_receipt_extractor_prompt() -> str:
    """Read agent #01's current hand-written prompt from disk. This is
    the baseline that gets compared against the optimized version."""
    prompt_path = _receipt_extractor_dir() / "prompts" / "extract.txt"
    if not prompt_path.exists():
        raise PromptOptimizerError(
            f"Agent #01's prompt not found at {prompt_path}. "
            "Is the workspace layout correct?"
        )
    return prompt_path.read_text(encoding="utf-8")


def _load_run_eval_module():
    """Import agent #01's eval/run_eval module for its pure scoring
    helpers. Uses importlib since the directory name starts with a
    digit and isn't a valid Python identifier -- same pattern the eval
    module's own tests use."""
    agents_root = Path(__file__).resolve().parent.parent
    if str(agents_root) not in sys.path:
        sys.path.insert(0, str(agents_root))
    return importlib.import_module("01_receipt_extractor.eval.run_eval")


# --- Public API ------------------------------------------------------------


def optimize_prompt(
    *,
    provider: str | None = None,
    model: str | None = None,
    max_cases: int | None = None,
    optimizer_effort: Literal["light", "medium", "heavy"] = "light",
    _dspy_program: Any = None,
) -> OptimizationResult:
    """Run one compilation pass to produce an optimized prompt for
    agent #01's receipt extraction.

    Args:
        provider: "openai" (default) / "anthropic" / "gemini" / "mock".
            Falls back to LLM_PROVIDER env var if not passed.
        model: model ID for the resolved provider. Falls back to the
            per-provider default via common.llm.resolve_model().
        max_cases: subsample the eval set for cheaper runs. None uses
            all 20 CORD cases. Ignored under mock.
        optimizer_effort: DSPy GEPA's `auto` setting. 'light' is the
            fast/cheap default; 'medium'/'heavy' produce better prompts
            at ~5x/~15x the cost.
        _dspy_program: test injection escape hatch; production callers
            should leave this None.

    Returns:
        A validated OptimizationResult with before/after per-field
        accuracy, the two prompts, and cost/wall-time.

    Raises:
        PromptOptimizerError: on missing dep, bad cases file, or API
            failure during compile.
    """
    resolved_provider = (provider or resolve_provider()).lower()
    if resolved_provider == "mock":
        return _mock_optimization(max_cases=max_cases, optimizer_effort=optimizer_effort)

    if optimizer_effort not in OPTIMIZER_EFFORT_LEVELS:
        raise ValueError(
            f"Unknown optimizer_effort: {optimizer_effort!r}. "
            f"Expected one of {OPTIMIZER_EFFORT_LEVELS}."
        )

    # #07 needs a vision-capable model that can actually read receipt
    # images. gemma4:e4b (the catalog-wide Ollama default) returns
    # all-nulls on CORD receipts -- GEPA then has no signal to
    # optimize. qwen2.5vl:7b hits ~80% on total/subtotal, so it's the
    # per-agent Ollama default here. Override with --model to swap.
    if resolved_provider == "ollama" and not model:
        resolved_model = "qwen2.5vl:7b"
    else:
        resolved_model = model or resolve_model(resolved_provider)
    baseline_prompt = _load_receipt_extractor_prompt()

    # Load the eval cases before importing dspy so a bad cases file
    # produces a targeted error instead of a confusing dspy stacktrace.
    run_eval = _load_run_eval_module()
    cases_path = _receipt_extractor_dir() / "eval" / "cases.jsonl"
    try:
        cases = run_eval.load_cases(cases_path)
    except Exception as exc:
        raise PromptOptimizerError(
            f"Failed to load eval cases from {cases_path}: {exc}",
            partial=OptimizationAttempt(stage="load_cases"),
        ) from exc

    if max_cases is not None:
        cases = cases[:max_cases]
    if not cases:
        raise PromptOptimizerError(
            f"No eval cases available (loaded 0 from {cases_path}). "
            "The optimizer needs at least one case to score against."
        )

    # dspy import happens INSIDE the real path so mock + tests never
    # need it installed. Same lazy-import pattern as agents #01/#02/#05.
    try:
        import dspy
    except ImportError as exc:
        raise PromptOptimizerError(
            "dspy is not installed. Run `uv sync` from the repo root, "
            "or `pip install dspy>=3.3` in this agent's environment.",
            partial=OptimizationAttempt(stage="build_program"),
        ) from exc

    start = time.perf_counter()
    try:
        program = _dspy_program if _dspy_program is not None else _build_program(
            dspy=dspy,
            provider=resolved_provider,
            model=resolved_model,
            baseline_prompt=baseline_prompt,
        )
        metric = _build_metric(run_eval=run_eval)
        trainset, valset = _cases_to_dspy_examples(dspy=dspy, cases=cases)

        # GEPA needs a "reflection" LM: a second model that reads eval
        # results and proposes new instructions. DSPy 3.x requires this
        # explicitly (older versions defaulted to the program's LM).
        # Reuse the same LM as the program: keeps the run single-model
        # (no extra key / no extra download), at the cost of asking
        # the same model to critique its own outputs. For a stronger
        # reflection loop, swap in a larger model here.
        reflection_lm = _build_reflection_lm(
            dspy=dspy, provider=provider, model=resolved_model
        )
        optimizer = dspy.GEPA(
            metric=metric,
            auto=optimizer_effort,
            reflection_lm=reflection_lm,
        )
        compiled = optimizer.compile(program, trainset=trainset, valset=valset)
    except PromptOptimizerError:
        raise
    except Exception as exc:
        raise _translate_api_error(exc) from exc

    wall_time = time.perf_counter() - start
    optimized_prompt = _extract_prompt_from_compiled_program(compiled)

    # Rescore both prompts against the full case set so the reported
    # accuracy numbers reflect the whole eval, not just the val split.
    baseline_acc = _score_prompt(
        dspy=dspy,
        run_eval=run_eval,
        prompt_text=baseline_prompt,
        cases=cases,
        provider=resolved_provider,
        model=resolved_model,
    )
    optimized_acc = _score_prompt(
        dspy=dspy,
        run_eval=run_eval,
        prompt_text=optimized_prompt,
        cases=cases,
        provider=resolved_provider,
        model=resolved_model,
    )

    return OptimizationResult(
        original_prompt=baseline_prompt,
        optimized_prompt=optimized_prompt,
        baseline_accuracy=baseline_acc,
        optimized_accuracy=optimized_acc,
        n_cases_used=len(cases),
        total_llm_calls=_EFFORT_CALL_ESTIMATES[optimizer_effort],
        estimated_cost_usd=_EFFORT_COST_ESTIMATES_USD[optimizer_effort],
        wall_time_seconds=wall_time,
        optimizer=f"dspy.GEPA(auto={optimizer_effort!r})",
        provider=resolved_provider,
        model=resolved_model,
    )


# --- DSPy program construction --------------------------------------------


def _build_reflection_lm(*, dspy, provider: str, model: str):
    """Build a dspy.LM instance for GEPA's reflection step. Uses the
    same provider/model as the program under test; keeps the run
    single-model. For a stronger reflection loop, swap for a larger
    model or a different provider."""
    litellm_prefix = _LITELLM_PROVIDER_PREFIX[provider]
    lm_kwargs: dict = {}
    if provider == "ollama":
        from common.llm import ollama_base_url
        lm_kwargs["api_base"] = ollama_base_url().removesuffix("/v1")
        # Ollama defaults num_ctx=4096. Vision inputs push individual
        # requests past that ceiling on larger CORD receipts, which
        # crashes GEPA's rollout bookkeeping with an IndexError.
        # 16384 comfortably fits image + prompt + response.
        lm_kwargs["num_ctx"] = 16384
    return dspy.LM(f"{litellm_prefix}/{model}", **lm_kwargs)


def _build_program(*, dspy, provider: str, model: str, baseline_prompt: str):
    """Build a DSPy Predict program whose Signature exposes exactly the
    four scored fields. Configures dspy.LM against the resolved
    provider/model. NOTE: the constructed Signature intentionally omits
    vendor_name, currency, and dates -- those aren't in the eval set's
    ground truth, so exposing them would give the optimizer no signal
    to work with and just add noise.
    """
    litellm_prefix = _LITELLM_PROVIDER_PREFIX[provider]
    lm_kwargs: dict = {}
    if provider == "ollama":
        # Pass api_base as a dspy.LM kwarg (LiteLLM forwards it to
        # the ollama provider). Prefer this over mutating process
        # env OLLAMA_API_BASE, which would leak to other agents in
        # the same process.
        from common.llm import ollama_base_url
        # ollama's LiteLLM path wants the raw host, not /v1.
        lm_kwargs["api_base"] = ollama_base_url().removesuffix("/v1")
    dspy.configure(lm=dspy.LM(f"{litellm_prefix}/{model}", **lm_kwargs))

    # Construct the Signature via dspy.Signature(...)'s dict form
    # rather than a class body. Under `from __future__ import
    # annotations` (in effect module-wide here), class-body annotations
    # become forward-reference strings that Pydantic tries to resolve
    # against module globals -- and `dspy` is only a local variable in
    # this function's scope, so `dspy.Image` fails to resolve. The
    # dict form passes the types directly and sidesteps annotation
    # evaluation entirely.
    fields = {
        "receipt_image": (
            dspy.Image,
            dspy.InputField(desc="Photo or scan of a receipt or invoice."),
        ),
        "total": (
            float,
            dspy.OutputField(
                desc="Final amount the customer paid, including tax and tip."
            ),
        ),
        "subtotal": (
            float | None,
            dspy.OutputField(desc="Pre-tax total if printed; null if not shown."),
        ),
        "tax_total": (
            float | None,
            dspy.OutputField(
                desc="Total tax amount if printed as its own line; null if not shown."
            ),
        ),
        "line_items_count": (
            int,
            dspy.OutputField(desc="Number of distinct line items on the receipt."),
        ),
    }
    signature = dspy.Signature(fields, baseline_prompt)
    return dspy.Predict(signature)


def _cases_to_dspy_examples(*, dspy, cases: list[dict]) -> tuple[list, list]:
    """Convert loaded cases into dspy.Example objects and split into
    train/val. Simple 50/50 split; GEPA's Pareto candidate selection
    doesn't need a huge val set, and with 20 total cases anything more
    generous would starve the trainset. `Example` inputs are marked
    via `.with_inputs(...)` per DSPy's convention."""
    images_dir = _receipt_extractor_dir() / "eval" / "images"

    examples = []
    for case in cases:
        image_path = images_dir / case["image_path"]
        expected = case.get("expected", {})
        example = dspy.Example(
            receipt_image=dspy.Image.from_path(str(image_path)),
            total=expected.get("total"),
            subtotal=expected.get("subtotal"),
            tax_total=expected.get("tax_total"),
            line_items_count=len(expected.get("line_items") or []),
        ).with_inputs("receipt_image")
        examples.append(example)

    split = max(1, len(examples) // 2)
    return examples[:split], examples[split:]


def _build_metric(*, run_eval):
    """Return a GEPA-shaped metric callable. GEPA calls this with
    (gold, pred, trace, pred_name, pred_trace, program_trace=None).
    We only use gold + pred; the rest are ignored. Returns a fraction
    in [0.0, 1.0] where 1.0 = every scored field matched."""

    def _get(source, key):
        """Read a field off gold (dspy.Example or dict). Both support
        __getitem__/get, but a missing key on Example raises rather
        than returning None, so we normalize here."""
        if hasattr(source, "get"):
            return source.get(key)
        return getattr(source, key, None)

    def _metric(
        gold,
        pred,
        trace=None,
        pred_name=None,
        pred_trace=None,
        program_trace=None,
    ) -> float:
        # Build `expected` only from fields gold actually labeled.
        # run_eval.score_case only scores keys present in `expected`,
        # so a caller that labeled only `total` gets scored on total
        # alone -- the model isn't penalized for fields the eval left
        # unspecified. Note the asymmetry: `line_items_count` of 0 is
        # a valid label (means "expected zero items") -- we distinguish
        # it from "count not labeled at all" via `is not None`.
        expected: dict = {}
        for field in ("total", "subtotal", "tax_total"):
            value = _get(gold, field)
            if value is not None:
                expected[field] = value
        line_items_count = _get(gold, "line_items_count")
        if line_items_count is not None:
            expected["line_items"] = [None] * line_items_count

        if not expected:
            return 0.0

        actual = {
            "total": getattr(pred, "total", None),
            "subtotal": getattr(pred, "subtotal", None),
            "tax_total": getattr(pred, "tax_total", None),
            "line_items": [None] * (getattr(pred, "line_items_count", 0) or 0),
        }
        results = run_eval.score_case(expected, actual)
        return sum(results.values()) / len(results) if results else 0.0

    return _metric


def _extract_prompt_from_compiled_program(compiled) -> str:
    """Pull the optimized instructions string out of GEPA's compiled
    program. DSPy stores the current instructions on the underlying
    Signature; the exact attribute path is `predictor.signature.instructions`
    for a Predict module. Guarded so a DSPy API drift produces a clear
    error instead of a mysterious AttributeError."""
    try:
        # A Predict-based program has a `.predict` attribute (or is one
        # directly, depending on how it was constructed). Walk both.
        candidate = compiled
        for attr in ("predict", "signature", "instructions"):
            if hasattr(candidate, attr):
                candidate = getattr(candidate, attr)
        if isinstance(candidate, str):
            return candidate
        # Fallback: look for a `.signature` on the compiled module directly.
        if hasattr(compiled, "signature") and hasattr(compiled.signature, "instructions"):
            return compiled.signature.instructions
    except Exception as exc:
        raise PromptOptimizerError(
            f"Could not extract optimized prompt from compiled program: {exc}. "
            "This usually means DSPy's API changed; check dspy.__version__ "
            "against this agent's `dspy>=3.3` pin.",
            partial=OptimizationAttempt(stage="rescore"),
        ) from exc
    raise PromptOptimizerError(
        "Compiled program had no `.instructions` attribute in any of "
        "the expected locations. Likely a DSPy API drift; see "
        "agents/07_prompt_optimizer/agent.py::_extract_prompt_from_compiled_program.",
        partial=OptimizationAttempt(stage="rescore"),
    )


def _score_prompt(
    *,
    dspy,
    run_eval,
    prompt_text: str,
    cases: list[dict],
    provider: str,
    model: str,
) -> dict[str, float]:
    """Run a Predict program with `prompt_text` as its Signature
    docstring across all cases and return per-field accuracy. Used to
    rescore both baseline and optimized after compilation. NOTE: this
    IS an LLM-billing hot path -- costs `len(cases)` extra calls per
    prompt scored, so ~40 extra calls total for a full rescore. Small
    next to GEPA's compile cost but not free."""
    program = _build_program(
        dspy=dspy, provider=provider, model=model, baseline_prompt=prompt_text
    )

    per_case: list[dict[str, bool]] = []
    for case in cases:
        image_path = _receipt_extractor_dir() / "eval" / "images" / case["image_path"]
        try:
            pred = program(receipt_image=dspy.Image.from_path(str(image_path)))
        except Exception:
            # A single-case failure shouldn't sink the whole rescore;
            # count it as a case where every scored field missed.
            expected = case.get("expected") or {}
            per_case.append({k: False for k in expected if k in SCORED_FIELDS})
            continue

        expected = case.get("expected") or {}
        actual = {
            "total": getattr(pred, "total", None),
            "subtotal": getattr(pred, "subtotal", None),
            "tax_total": getattr(pred, "tax_total", None),
            "line_items": [None] * (getattr(pred, "line_items_count", 0) or 0),
        }
        per_case.append(run_eval.score_case(expected, actual))

    aggregated = run_eval.aggregate_accuracy(per_case)
    # aggregate_accuracy returns dict[str, tuple[float, int]] -- keep
    # only the fraction, drop the case count for the OptimizationResult
    # (n_cases_used already carries that).
    return {field: fraction for field, (fraction, _n) in aggregated.items()}


# --- Error translation (R5 case 3) -----------------------------------------


def _translate_api_error(exc: Exception) -> PromptOptimizerError:
    """Turn a DSPy / provider SDK exception into a user-facing
    PromptOptimizerError. Six-branch priority order matches the shape
    used across agents #02-#06:
      1. Class-name check: rate limit -> rate-limit case
      2. Class-name check: auth -> auth case
      3. Status code 429 -> rate-limit case
      4. Status code 401 -> auth case
      5. Message-string fallback (rate limit / overloaded, auth / key)
      6. Generic fallback with original exception preserved
    """
    exc_class_name = type(exc).__name__.lower()
    message_lower = str(exc).lower()
    status = getattr(exc, "status_code", None)

    if "ratelimiterror" in exc_class_name:
        return _rate_limit_error()
    if "authenticationerror" in exc_class_name or "apikeyerror" in exc_class_name:
        return _auth_error()
    if status == 429:
        return _rate_limit_error()
    if status == 401:
        return _auth_error()
    if "rate limit" in message_lower or "overloaded" in message_lower:
        return _rate_limit_error()
    if "authentication" in message_lower or "api key" in message_lower:
        return _auth_error()

    return PromptOptimizerError(
        f"Optimization failed: {type(exc).__name__}: {exc}. "
        "This is an unexpected error -- check the agent logs. "
        "The optimizer's partial state (if any) was not preserved."
    )


def _rate_limit_error() -> PromptOptimizerError:
    """Shared message so every code path that reaches this case can't
    drift apart. This agent does NOT auto-retry rate-limit errors:
    optimization runs are already expensive, and having the loop
    silently retry could double the bill."""
    return PromptOptimizerError(
        "The provider is temporarily rate-limited or overloaded. "
        "Wait a minute and try again. The optimizer's partial progress "
        "was not preserved; you will need to restart the run."
    )


def _auth_error() -> PromptOptimizerError:
    return PromptOptimizerError(
        "Authentication failed: check that your API key is set correctly "
        "for the provider you selected (OPENAI_API_KEY / ANTHROPIC_API_KEY "
        "/ GEMINI_API_KEY). See .env.example at the repo root."
    )


# --- Mock mode -------------------------------------------------------------


def _mock_optimization(
    *, max_cases: int | None, optimizer_effort: str
) -> OptimizationResult:
    """Deterministic canned OptimizationResult for CI + local exploration.
    No dspy import, no provider SDK, no key. Numbers match agent #01's
    real baseline (from eval/results.md) so a reader running mock sees
    a plausible-looking before/after; the "optimized" numbers are a
    modest fake improvement on subtotal + tax_total (the two fields with
    the most real headroom) and hold `total` flat (no regression on the
    load-bearing field).

    `max_cases` is echoed into the result so tests can prove the mock
    saw the input, same convention as agent #01's byte-count trick."""
    effective_cases = max_cases if max_cases is not None else 20

    baseline = {
        "total": 0.90,
        "line_items": 0.85,
        "subtotal": 0.72,
        "tax_total": 0.46,
    }
    optimized = {
        "total": 0.90,  # load-bearing, no regression
        "line_items": 0.85,  # count-only, hard to move
        "subtotal": 0.83,  # ~+11 points
        "tax_total": 0.62,  # ~+16 points
    }

    return OptimizationResult(
        original_prompt=(
            "[MOCK] This would be agent #01's real prompts/extract.txt "
            "content in a live run."
        ),
        optimized_prompt=(
            f"[MOCK optimized prompt for max_cases={effective_cases}] "
            "This would be the string GEPA produced in a live run."
        ),
        baseline_accuracy=baseline,
        optimized_accuracy=optimized,
        n_cases_used=effective_cases,
        total_llm_calls=0,
        estimated_cost_usd=0.0,
        wall_time_seconds=0.0,
        optimizer=f"mock(effort={optimizer_effort!r})",
        provider="mock",
        model="mock",
    )


# --- CLI entry point -------------------------------------------------------


def main() -> int:
    """CLI: `uv run python -m agent` from inside this directory. Prints
    a summary + writes the full OptimizationResult to `last_run.json`
    next to this file (gitignored)."""
    parser = argparse.ArgumentParser(
        prog="prompt-optimizer",
        description=(
            "Optimize agent #01's receipt-extractor prompt via DSPy's "
            "GEPA optimizer. Set LLM_PROVIDER=mock for a canned demo, "
            "or supply a real key (OPENAI/ANTHROPIC/GEMINI) for a real "
            "compile run."
        ),
    )
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI instead of CLI.")
    parser.add_argument(
        "--provider",
        choices=(*SUPPORTED_PROVIDERS, "mock"),
        default=None,
        help="Override LLM_PROVIDER for this run.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the resolved model, e.g. gpt-4.1-mini-2025-04-14.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Subsample the 20 CORD cases for a cheaper run. Ignored under mock.",
    )
    parser.add_argument(
        "--effort",
        choices=OPTIMIZER_EFFORT_LEVELS,
        default="light",
        help="GEPA optimizer effort (light/medium/heavy). Cost scales roughly 5x per step.",
    )
    args = parser.parse_args()

    if args.ui:
        try:
            from .ui import build_ui
        except ImportError:
            from ui import build_ui
        build_ui().launch()
        return 0

    try:
        result = optimize_prompt(
            provider=args.provider,
            model=args.model,
            max_cases=args.max_cases,
            optimizer_effort=args.effort,
        )
    except PromptOptimizerError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1

    out_path = Path(__file__).parent / "last_run.json"
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    print(f"Optimizer:          {result.optimizer}")
    print(f"Provider / model:   {result.provider} / {result.model}")
    print(f"Cases used:         {result.n_cases_used}")
    print(f"Wall time:          {result.wall_time_seconds:.1f}s")
    print(f"Estimated cost:     ~${result.estimated_cost_usd:.2f}")
    print()
    print("Per-field accuracy (baseline -> optimized, delta):")
    for field in sorted(result.improvement):
        before = result.baseline_accuracy.get(field, 0.0)
        after = result.optimized_accuracy.get(field, 0.0)
        delta = result.improvement[field]
        print(f"  {field:15s} {before:.2%} -> {after:.2%}  ({delta:+.2%})")
    print()
    print(f"Full JSON written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
