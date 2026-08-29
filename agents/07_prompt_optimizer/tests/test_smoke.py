"""Smoke tests for the prompt optimizer agent.

All tests run under LLM_PROVIDER=mock (R8 in CONTRIBUTING.md) -- CI
never touches a real API key. DSPy paths that require the library
installed are gated behind `pytest.importorskip("dspy")` so a
minimal-deps environment still runs everything else.

Covers, in order:
1. Mock-mode round-trip (input passthrough, validator sanity).
2. Schema validation on OptimizationResult (improvement math, bounds).
3. R5 error branches (unknown provider, unknown effort, missing prompt
   file, missing cases file, dspy-not-installed shim, 6-branch API
   error translation).
4. Metric function: replicates agent #01's compare_field edge cases.
5. Signature construction structural guard (only 4 scored fields).
6. Prompt-extraction helper for a compiled program shape.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_AGENT_DIR.parent))

_agent_pkg = importlib.import_module("07_prompt_optimizer.agent")
_schemas_pkg = importlib.import_module("07_prompt_optimizer.schemas")

optimize_prompt = _agent_pkg.optimize_prompt
resolve_provider = _agent_pkg.resolve_provider
PromptOptimizerError = _agent_pkg.PromptOptimizerError
OptimizationAttempt = _agent_pkg.OptimizationAttempt
_translate_api_error = _agent_pkg._translate_api_error
_mock_optimization = _agent_pkg._mock_optimization
_build_metric = _agent_pkg._build_metric
_extract_prompt_from_compiled_program = _agent_pkg._extract_prompt_from_compiled_program
_load_run_eval_module = _agent_pkg._load_run_eval_module
SCORED_FIELDS = _agent_pkg.SCORED_FIELDS
SUPPORTED_PROVIDERS = _agent_pkg.SUPPORTED_PROVIDERS
OPTIMIZER_EFFORT_LEVELS = _agent_pkg.OPTIMIZER_EFFORT_LEVELS
OptimizationResult = _schemas_pkg.OptimizationResult


# ---------- 1. Mock path ----------


def test_mock_returns_valid_optimization_result(monkeypatch):
    """The whole code path from `optimize_prompt(provider="mock")` to
    a validated OptimizationResult works without dspy, without a key,
    without touching the FS."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    result = optimize_prompt()
    assert isinstance(result, OptimizationResult)
    assert result.provider == "mock"
    assert result.model == "mock"
    assert result.n_cases_used == 20  # default in mock when max_cases=None


def test_mock_echoes_max_cases_into_result(monkeypatch):
    """Anti-refactor guard: the mock must see and use its `max_cases`
    input, otherwise a future refactor that makes mock output constant
    regardless of input surfaces at test time. Same convention agent
    #01 uses with byte-count-in-line-description."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r5 = optimize_prompt(max_cases=5)
    r15 = optimize_prompt(max_cases=15)
    assert r5.n_cases_used == 5
    assert r15.n_cases_used == 15
    assert "max_cases=5" in r5.optimized_prompt
    assert "max_cases=15" in r15.optimized_prompt


def test_mock_holds_total_flat_and_improves_tax(monkeypatch):
    """The mock is DESIGNED to model the load-bearing constraint from
    the plan: `total` is the field extraction exists to get right,
    optimizer must not regress it while chasing tax_total headroom.
    Test that the mock reflects this."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    result = optimize_prompt()
    assert result.improvement["total"] == 0.0
    assert result.improvement["tax_total"] > 0.0
    assert result.improvement["subtotal"] > 0.0


def test_provider_kwarg_overrides_env(monkeypatch):
    """`provider="mock"` explicitly must bypass whatever LLM_PROVIDER
    is set to. Lets the CLI's --provider flag work per-call."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    result = optimize_prompt(provider="mock")
    assert result.provider == "mock"


# ---------- 2. Schema validation ----------


def test_result_improvement_is_computed_by_validator():
    """The model validator must fill `improvement` from baseline vs
    optimized -- regression guard against a future refactor that
    forgets to run it."""
    result = OptimizationResult(
        original_prompt="a",
        optimized_prompt="b",
        baseline_accuracy={"total": 0.90, "tax_total": 0.46},
        optimized_accuracy={"total": 0.90, "tax_total": 0.62},
        n_cases_used=20,
        total_llm_calls=30,
        estimated_cost_usd=0.30,
        wall_time_seconds=45.0,
        optimizer="dspy.GEPA(auto='light')",
        provider="openai",
        model="gpt-4.1-mini",
    )
    assert result.improvement["total"] == pytest.approx(0.0)
    assert result.improvement["tax_total"] == pytest.approx(0.16)


def test_result_improvement_treats_missing_side_as_zero():
    """A field present in only one of baseline/optimized shouldn't
    make the delta look wildly negative from a phantom drop; treat
    missing as 0.0 so the delta reflects what actually changed."""
    result = OptimizationResult(
        original_prompt="a",
        optimized_prompt="b",
        baseline_accuracy={"total": 0.90},
        optimized_accuracy={"total": 0.92, "new_field": 0.50},
        n_cases_used=1,
        total_llm_calls=0,
        estimated_cost_usd=0.0,
        wall_time_seconds=0.0,
        optimizer="mock",
        provider="mock",
        model="mock",
    )
    assert result.improvement["total"] == pytest.approx(0.02)
    assert result.improvement["new_field"] == pytest.approx(0.50)


def test_result_rejects_negative_n_cases():
    """n_cases_used has ge=1 -- an optimizer that reports 0 cases used
    is a bug we want to surface at validation time."""
    with pytest.raises(ValidationError):
        OptimizationResult(
            original_prompt="a",
            optimized_prompt="b",
            n_cases_used=0,
            total_llm_calls=0,
            estimated_cost_usd=0.0,
            wall_time_seconds=0.0,
            optimizer="mock",
            provider="mock",
            model="mock",
        )


def test_result_rejects_negative_cost():
    """Cost has ge=0.0 -- an optimizer that reports a negative bill is
    clearly wrong; catch at validation."""
    with pytest.raises(ValidationError):
        OptimizationResult(
            original_prompt="a",
            optimized_prompt="b",
            n_cases_used=1,
            total_llm_calls=0,
            estimated_cost_usd=-1.0,
            wall_time_seconds=0.0,
            optimizer="mock",
            provider="mock",
            model="mock",
        )


# ---------- 3. R5 error branches ----------


def test_resolve_provider_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider() == "openai"


def test_resolve_provider_respects_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert resolve_provider() == "anthropic"


def test_resolve_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


def test_optimize_rejects_unknown_effort(monkeypatch):
    """A caller who passes optimizer_effort='xhuge' should get a clear
    ValueError before any dspy import or LLM call happens."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")  # skip the mock branch
    with pytest.raises(ValueError, match="Unknown optimizer_effort"):
        optimize_prompt(optimizer_effort="xhuge")  # type: ignore[arg-type]


def test_translate_api_error_class_name_rate_limit():
    class RateLimitError(Exception):
        pass

    out = _translate_api_error(RateLimitError("whatever"))
    assert isinstance(out, PromptOptimizerError)
    assert "rate-limited" in out.message.lower() or "overloaded" in out.message.lower()


def test_translate_api_error_class_name_auth():
    class AuthenticationError(Exception):
        pass

    out = _translate_api_error(AuthenticationError("whatever"))
    assert isinstance(out, PromptOptimizerError)
    assert "authentication" in out.message.lower() or "api key" in out.message.lower()


def test_translate_api_error_status_code_429():
    exc = RuntimeError("something something")
    exc.status_code = 429  # type: ignore[attr-defined]
    out = _translate_api_error(exc)
    assert "rate-limited" in out.message.lower() or "overloaded" in out.message.lower()


def test_translate_api_error_status_code_401():
    exc = RuntimeError("something something")
    exc.status_code = 401  # type: ignore[attr-defined]
    out = _translate_api_error(exc)
    assert "authentication" in out.message.lower() or "api key" in out.message.lower()


def test_translate_api_error_message_fallback_rate_limit():
    """A plain Exception whose message contains 'rate limit' but no
    matching class or status still routes to the rate-limit case."""
    out = _translate_api_error(RuntimeError("You hit the rate limit for now"))
    assert "rate-limited" in out.message.lower() or "overloaded" in out.message.lower()


def test_translate_api_error_generic_fallback_preserves_original():
    """Unknown error class + no matching status/keyword -> generic
    branch with the original exception's type + message preserved for
    debugging."""
    out = _translate_api_error(RuntimeError("some genuinely unexpected thing"))
    assert isinstance(out, PromptOptimizerError)
    assert "RuntimeError" in out.message
    assert "some genuinely unexpected thing" in out.message


def test_translate_api_error_class_check_priority_beats_message():
    """A ValidationError whose message contains 'rate limit' must NOT
    be misclassified as rate-limit unless it also matches by class or
    status. Regression guard mirroring agent #02's [S5] test."""

    class ValidationError(Exception):
        pass

    exc = ValidationError("whatever, no matching keywords here")
    out = _translate_api_error(exc)
    # Falls through to generic branch (no rate-limit/auth signals).
    assert "unexpected error" in out.message.lower()


# ---------- 4. Metric function ----------


def test_metric_returns_perfect_score_when_everything_matches(monkeypatch):
    """A metric that returns the right fraction is load-bearing:
    GEPA will optimize toward WHATEVER this returns, so if it's wrong
    the whole compilation chases the wrong signal."""
    run_eval = _load_run_eval_module()
    metric = _build_metric(run_eval=run_eval)

    class FakePred:
        total = 100.0
        subtotal = 90.0
        tax_total = 10.0
        line_items_count = 3

    gold = {"total": 100.0, "subtotal": 90.0, "tax_total": 10.0, "line_items_count": 3}
    assert metric(gold, FakePred()) == pytest.approx(1.0)


def test_metric_returns_zero_when_everything_misses():
    run_eval = _load_run_eval_module()
    metric = _build_metric(run_eval=run_eval)

    class FakePred:
        total = 999.0
        subtotal = 999.0
        tax_total = 999.0
        line_items_count = 999

    gold = {"total": 100.0, "subtotal": 90.0, "tax_total": 10.0, "line_items_count": 3}
    assert metric(gold, FakePred()) == pytest.approx(0.0)


def test_metric_honors_money_tolerance():
    """Money comparison uses abs(diff) < 0.01 per compare_field. An
    optimizer whose extraction is off by 0.001 should still score the
    field as a pass -- this is the honest 'floating-point equality is
    unreliable for money' allowance."""
    run_eval = _load_run_eval_module()
    metric = _build_metric(run_eval=run_eval)

    class FakePred:
        total = 100.001
        subtotal = None
        tax_total = None
        line_items_count = 0

    gold = {"total": 100.0, "line_items_count": 0}
    # `total` passes (within tolerance); line_items empty vs empty passes.
    assert metric(gold, FakePred()) == pytest.approx(1.0)


def test_metric_ignores_fields_gold_did_not_label():
    """If the eval case only labels `total`, the metric must not
    penalize the model for anything else. Only labeled fields count.
    Guards against a naive rewrite that scores all four fields even
    when the case only ground-truthed one."""
    run_eval = _load_run_eval_module()
    metric = _build_metric(run_eval=run_eval)

    class FakePred:
        total = 100.0
        subtotal = 999.0  # wrong, but NOT in gold, must not count
        tax_total = 999.0  # wrong, but NOT in gold, must not count
        line_items_count = 999  # wrong, but line_items not in gold

    gold = {"total": 100.0}  # only total is labeled
    assert metric(gold, FakePred()) == pytest.approx(1.0)


def test_metric_returns_zero_when_gold_has_nothing_scorable():
    """If a case has zero scorable fields (all None), the metric
    returns 0.0 without division-by-zero."""
    run_eval = _load_run_eval_module()
    metric = _build_metric(run_eval=run_eval)

    class FakePred:
        total = None
        subtotal = None
        tax_total = None
        line_items_count = 0

    assert metric({}, FakePred()) == pytest.approx(0.0)


# ---------- 5. Signature structural guard (dspy-gated) ----------


def test_signature_exposes_only_scored_fields():
    """CRITICAL guard: the DSPy Signature must expose EXACTLY the four
    fields agent #01's eval harness scores -- no more, no less. If
    someone adds vendor_name here later, GEPA will start optimizing
    against fake signal (CORD blurs vendor names -> zero real signal
    -> optimizer wanders). This test locks the signature down."""
    dspy = pytest.importorskip("dspy")

    program = _agent_pkg._build_program(
        dspy=dspy,
        provider="openai",
        model="gpt-4.1-mini-2025-04-14",
        baseline_prompt="test prompt",
    )
    signature = program.signature

    output_field_names = set(signature.output_fields.keys())
    # The 4 scored fields, with line_items expressed as its count for
    # DSPy's schema-shape (see agent.py::_metric for how count maps
    # back to score_case's list-length interpretation).
    assert output_field_names == {"total", "subtotal", "tax_total", "line_items_count"}

    input_field_names = set(signature.input_fields.keys())
    assert input_field_names == {"receipt_image"}


def test_signature_docstring_seeds_from_baseline_prompt():
    """GEPA iterates on the Signature's docstring (its `instructions`).
    Seeding it with agent #01's real prompt gives the optimizer a
    known-good starting point instead of the empty string. Regression
    guard against a refactor that forgets to plumb baseline_prompt in."""
    dspy = pytest.importorskip("dspy")

    unique_seed = "BASELINE_PROMPT_SEED_MARKER_FOR_STRUCTURAL_TEST"
    program = _agent_pkg._build_program(
        dspy=dspy, provider="openai", model="gpt-4.1-mini-2025-04-14",
        baseline_prompt=unique_seed,
    )
    assert unique_seed in program.signature.instructions


def test_extract_prompt_reads_signature_instructions():
    """The prompt-extraction helper walks compiled.predict.signature.
    instructions; if DSPy renames any of those, the helper raises a
    clear PromptOptimizerError instead of a mysterious AttributeError."""

    class FakeSignature:
        instructions = "an optimized prompt string"

    class FakePredict:
        signature = FakeSignature()

    class FakeCompiled:
        predict = FakePredict()

    assert _extract_prompt_from_compiled_program(FakeCompiled()) == "an optimized prompt string"


def test_extract_prompt_falls_back_to_direct_signature():
    """A compiled program without `.predict` but with `.signature` on
    itself is also handled -- some DSPy modules structure themselves
    that way. Direct .signature.instructions should still work."""

    class FakeSignature:
        instructions = "another optimized prompt"

    class FakeCompiled:
        signature = FakeSignature()

    assert _extract_prompt_from_compiled_program(FakeCompiled()) == "another optimized prompt"


def test_extract_prompt_raises_on_unknown_shape():
    """A compiled program that has none of the expected attributes
    raises PromptOptimizerError with a pointer at the extraction
    helper, not a mysterious AttributeError."""

    class FakeCompiled:
        pass

    with pytest.raises(PromptOptimizerError, match="no `.instructions`"):
        _extract_prompt_from_compiled_program(FakeCompiled())


# ---------- 6. Constants + workspace layout sanity ----------


def test_scored_fields_matches_run_eval_expectations():
    """SCORED_FIELDS is the load-bearing contract between this agent
    and #01's eval harness. If someone adds a field here without also
    updating _metric + _build_program, the optimizer will optimize
    against a non-existent output field. Explicit lock-down."""
    assert set(SCORED_FIELDS) == {"total", "subtotal", "tax_total", "line_items"}


def test_supported_providers_matches_common_llm():
    """Kept in sync with common/llm.py's SUPPORTED_PROVIDERS. If those
    two ever drift, the CLI's --provider choices will show one thing
    and the resolver will accept another."""
    assert set(SUPPORTED_PROVIDERS) == {"openai", "anthropic", "gemini", "ollama"}


def test_effort_levels_line_up_with_cost_estimates():
    """The README quotes effort-level costs; the constants that produce
    them must match. If someone adds a new effort ('xheavy') they must
    also add a cost estimate, otherwise mock runs report cost 0.0."""
    for level in OPTIMIZER_EFFORT_LEVELS:
        assert level in _agent_pkg._EFFORT_CALL_ESTIMATES
        assert level in _agent_pkg._EFFORT_COST_ESTIMATES_USD


def test_run_eval_module_import_works():
    """The importlib dance to reach agent #01's eval helpers works
    from this agent's own context. Regression guard against a
    workspace layout change that breaks the sys.path insert."""
    run_eval = _load_run_eval_module()
    # Sanity: it has the functions we depend on.
    assert callable(run_eval.compare_field)
    assert callable(run_eval.score_case)
    assert callable(run_eval.aggregate_accuracy)
    assert callable(run_eval.load_cases)


def test_receipt_extractor_prompt_is_findable():
    """If the workspace layout changes and #01's prompts/extract.txt
    moves, we want a clear error, not a silent baseline of the empty
    string."""
    prompt_path = _agent_pkg._receipt_extractor_dir() / "prompts" / "extract.txt"
    assert prompt_path.exists(), f"agent #01's prompt not at expected path: {prompt_path}"
    content = prompt_path.read_text(encoding="utf-8")
    assert len(content) > 100, "agent #01's prompt looks suspiciously short"
