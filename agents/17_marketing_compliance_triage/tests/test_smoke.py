"""Smoke tests for the marketing compliance triage agent.

All tests run under LLM_PROVIDER=mock (R8 in CONTRIBUTING.md) -- CI
never touches a real API key. Real-provider tests are a manual
maintainer check before shipping.

Covers:

1. Mock path returns a valid ComplianceReport (no graph, no LLM).
2. R5 case 1: copy too short raises before any LLM call; copy too
   long raises before any LLM call.
3. The classify -> conditional fan-out -> aggregate graph via
   triage_marketing_copy:
   - 0 active lenses: exactly 1 LLM call (classify only), aggregate
     is pure Python, deterministic "no concerns" report.
   - 1 active lens: exactly 3 LLM calls (classify + 1 reviewer +
     aggregate), 1 finding.
   - 2 active lenses: exactly 4 LLM calls (classify + 2 reviewers +
     aggregate), 2 findings -- THE test that proves the
     Annotated[list, operator.add] reducer actually merges both
     parallel branches' writes without one clobbering the other.
   - retry-then-succeed: a node's structured-output loop retries once
     on invalid JSON, then succeeds.
   - retry exhaustion: raises TriageError with .partial attached.
4. R5 case 3: `_translate_api_error` maps all branches (class-name,
   status-code, message-fallback, ollama-connection, generic).
5. Pure helpers: `_route_after_classify` for 0/1/2/4-active inputs,
   `resolve_provider`, `_normalize_for_substring`, `_parse_json_object`.

SequenceLLM fixture supplies scripted JSON responses on successive
`.complete()` calls -- same shape as agents #02/#03's, just reused
because this graph makes a variable number of LLM calls per run
depending on how many lenses the scripted classify response activates.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AGENT_DIR.parent))

_agent = importlib.import_module("17_marketing_compliance_triage.agent")
_schemas = importlib.import_module("17_marketing_compliance_triage.schemas")

triage_marketing_copy = _agent.triage_marketing_copy
TriageError = _agent.TriageError
TriageAttempt = _agent.TriageAttempt
resolve_provider = _agent.resolve_provider
_translate_api_error = _agent._translate_api_error
_normalize_for_substring = _agent._normalize_for_substring
_parse_json_object = _agent._parse_json_object
MIN_COPY_CHARS = _agent.MIN_COPY_CHARS
MAX_COPY_TOKENS_ESTIMATE = _agent.MAX_COPY_TOKENS_ESTIMATE
CHARS_PER_TOKEN_ESTIMATE = _agent.CHARS_PER_TOKEN_ESTIMATE
ComplianceReport = _schemas.ComplianceReport
LensFinding = _schemas.LensFinding

SAMPLE_COPY = (
    "Our new supplement is clinically proven to cure your cold and is "
    "100% eco-friendly packaging."
)


# --- SequenceLLM fixture -----------------------------------------------


class SequenceLLM:
    """Test-only LLM Protocol impl. Returns responses from a pre-set
    list on successive .complete() calls, records every call so tests
    can assert call count / prompt content."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cacheable_prefix: str | None = None,
    ):
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        if not self._responses:
            raise RuntimeError("SequenceLLM ran out of scripted responses")
        text = self._responses.pop(0)

        class _Resp:
            pass

        r = _Resp()
        r.text = text
        r.input_tokens = 0
        r.output_tokens = 0
        r.cached_input_tokens = 0
        r.cache_creation_input_tokens = 0
        r.latency_ms = 0.0
        return r


class KeyedLLM:
    """Test LLM that returns responses keyed by a marker substring found
    in the prompt, not by call order. Needed when 2+ reviewer nodes are
    active: LangGraph runs them concurrently within one superstep, so
    the order they call .complete() in is NOT deterministic -- unlike
    every other agent in this catalog (all purely sequential), this is
    the first agent where two LLM calls can race. A plain FIFO
    SequenceLLM would non-deterministically hand the wrong node's
    scripted response to the wrong node."""

    def __init__(self, responses_by_marker: list[tuple[str, str]]):
        self._queues: dict[str, list[str]] = {}
        for marker, response in responses_by_marker:
            self._queues.setdefault(marker, []).append(response)
        self.calls: list[dict] = []

    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cacheable_prefix: str | None = None,
    ):
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        for marker, queue in self._queues.items():
            if marker in prompt and queue:
                text = queue.pop(0)
                break
        else:
            raise RuntimeError(
                f"KeyedLLM: no matching marker with a remaining response "
                f"for prompt starting: {prompt[:200]!r}"
            )

        class _Resp:
            pass

        r = _Resp()
        r.text = text
        r.input_tokens = 0
        r.output_tokens = 0
        r.cached_input_tokens = 0
        r.cache_creation_input_tokens = 0
        r.latency_ms = 0.0
        return r


def _classify_json(active_lenses: list[str], reasoning: str = "test reasoning") -> str:
    import json

    return json.dumps({"active_lenses": active_lenses, "reasoning": reasoning})


def _finding_json(lens: str, phrases: list[str], severity: str = "medium") -> str:
    import json

    return json.dumps(
        {
            "lens": lens,
            "flagged_phrases": phrases,
            "concern": f"test concern for {lens}",
            "suggested_fix": f"test fix for {lens}",
            "severity": severity,
        }
    )


def _aggregate_json(active_lenses: list[str], overall_risk_level: str = "medium") -> str:
    import json

    return json.dumps(
        {
            "active_lenses": active_lenses,
            "findings": [],
            "overall_risk_level": overall_risk_level,
            "summary": "test summary",
        }
    )


# --- 1. Mock path --------------------------------------------------------


def test_mock_path_returns_a_result(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    report = triage_marketing_copy(SAMPLE_COPY)
    assert isinstance(report, ComplianceReport)
    assert len(report.findings) >= 1


def test_mock_path_does_not_touch_graph():
    """Mock mode short-circuits before any graph/LLM work -- a
    ridiculously short copy_text that would otherwise fail the
    MIN_COPY_CHARS gate should not raise under mock."""
    report = triage_marketing_copy("x", provider="mock")
    assert isinstance(report, ComplianceReport)


def test_mock_path_serializable_to_json():
    report = triage_marketing_copy(SAMPLE_COPY, provider="mock")
    dumped = report.model_dump_json()
    restored = ComplianceReport.model_validate_json(dumped)
    assert restored == report


def test_provider_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    report = triage_marketing_copy(SAMPLE_COPY, provider="mock")
    assert isinstance(report, ComplianceReport)


# --- 2. R5 case 1: bad input ---------------------------------------------


def test_copy_too_short_raises_before_any_llm_call():
    llm = SequenceLLM([])  # would RuntimeError on any .complete() call
    with pytest.raises(TriageError, match="too short"):
        triage_marketing_copy(
            "short", provider="openai", model="test-model", _llm=llm
        )
    assert len(llm.calls) == 0


def test_copy_at_minimum_length_does_not_raise_bad_input(monkeypatch):
    """Exactly MIN_COPY_CHARS should pass the gate (boundary check) --
    use mock mode so we don't need a real graph response."""
    copy_text = "x" * MIN_COPY_CHARS
    report = triage_marketing_copy(copy_text, provider="mock")
    assert isinstance(report, ComplianceReport)


def test_copy_too_long_raises_before_any_llm_call(monkeypatch):
    monkeypatch.setattr(_agent, "MAX_COPY_TOKENS_ESTIMATE", 5)
    llm = SequenceLLM([])
    with pytest.raises(TriageError, match="too long") as exc_info:
        triage_marketing_copy(
            SAMPLE_COPY, provider="openai", model="test-model-xyz", _llm=llm
        )
    assert "test-model-xyz" in exc_info.value.message
    assert len(llm.calls) == 0


# --- 3. The classify -> conditional fan-out -> aggregate graph ----------


def test_graph_zero_active_lenses(monkeypatch):
    """Classify decides no lens applies -> routes straight to
    aggregate with NO LLM call for aggregate (pure Python) -> exactly
    1 total LLM call."""
    llm = SequenceLLM([_classify_json([])])
    report = triage_marketing_copy(
        SAMPLE_COPY, provider="openai", model="test-model", _llm=llm
    )
    assert len(llm.calls) == 1
    assert report.active_lenses == []
    assert report.findings == []
    assert report.overall_risk_level == "low"
    assert "no regulatory concerns" in report.summary.lower()


def test_graph_one_active_lens(monkeypatch):
    """Classify activates exactly 1 lens -> 1 reviewer runs -> exactly
    3 total LLM calls (classify + 1 reviewer + aggregate)."""
    llm = SequenceLLM(
        [
            _classify_json(["health_wellness"]),
            _finding_json(
                "health_wellness",
                ["clinically proven to cure your cold"],
                severity="high",
            ),
            _aggregate_json(["health_wellness"], overall_risk_level="high"),
        ]
    )
    report = triage_marketing_copy(
        SAMPLE_COPY, provider="openai", model="test-model", _llm=llm
    )
    assert len(llm.calls) == 3
    assert len(report.findings) == 1
    assert report.findings[0].lens == "health_wellness"
    assert report.active_lenses == ["health_wellness"]


def test_graph_two_active_lenses_reducer_merges_both_branches(monkeypatch):
    """THE most important test: classify activates 2 lenses -> both
    reviewers run (in the same superstep) -> exactly 4 total LLM calls
    -> the operator.add reducer must merge BOTH branches' findings,
    not just one. A bug here would manifest as only 1 finding
    surviving, or the wrong lens's finding appearing twice.

    Uses KeyedLLM, not SequenceLLM: the two reviewer nodes run
    concurrently and race to call .complete(), so a plain FIFO queue
    can't safely script "node A's response, then node B's response" --
    whichever fires first gets whatever's at the front of the queue,
    non-deterministically.
    """
    llm = KeyedLLM(
        [
            (
                "You are a marketing compliance triage assistant",
                _classify_json(["health_wellness", "environmental"]),
            ),
            (
                "HEALTH/WELLNESS CLAIMS lens",
                _finding_json(
                    "health_wellness", ["clinically proven to cure your cold"], "high"
                ),
            ),
            (
                "ENVIRONMENTAL/SUSTAINABILITY CLAIMS lens",
                _finding_json(
                    "environmental", ["100% eco-friendly packaging"], "medium"
                ),
            ),
            (
                "You are writing the final compliance triage summary",
                _aggregate_json(
                    ["health_wellness", "environmental"], overall_risk_level="high"
                ),
            ),
        ]
    )
    report = triage_marketing_copy(
        SAMPLE_COPY, provider="openai", model="test-model", _llm=llm
    )
    assert len(llm.calls) == 4
    assert len(report.findings) == 2
    lenses_found = sorted(f.lens for f in report.findings)
    assert lenses_found == ["environmental", "health_wellness"]
    # Each finding's content actually came from ITS OWN branch, not a
    # duplicate of the other.
    health = next(f for f in report.findings if f.lens == "health_wellness")
    env = next(f for f in report.findings if f.lens == "environmental")
    assert "cure your cold" in health.flagged_phrases[0]
    assert "eco-friendly" in env.flagged_phrases[0]


def test_graph_overall_risk_floor_enforced(monkeypatch):
    """The aggregate node's extra_validate rejects an overall_risk_level
    below the worst individual finding's severity, forcing a retry.
    Script the first aggregate attempt as an under-reported risk level
    and the second as a corrected one."""
    llm = SequenceLLM(
        [
            _classify_json(["health_wellness"]),
            _finding_json(
                "health_wellness", ["clinically proven to cure your cold"], "high"
            ),
            _aggregate_json(["health_wellness"], overall_risk_level="low"),  # under-reported
            _aggregate_json(["health_wellness"], overall_risk_level="high"),  # corrected
        ]
    )
    report = triage_marketing_copy(
        SAMPLE_COPY, provider="openai", model="test-model", _llm=llm, max_retries=3
    )
    assert len(llm.calls) == 4  # the extra retry was consumed
    assert report.overall_risk_level == "high"


def test_graph_retries_on_invalid_json_then_succeeds(monkeypatch):
    llm = SequenceLLM(
        [
            "not valid json at all",
            _classify_json([]),
        ]
    )
    report = triage_marketing_copy(
        SAMPLE_COPY, provider="openai", model="test-model", _llm=llm, max_retries=3
    )
    assert len(llm.calls) == 2
    assert report.active_lenses == []


def test_graph_retries_on_valid_json_bad_schema_then_succeeds(monkeypatch):
    """Distinct from the 'not JSON at all' case above: this response IS
    valid JSON, but 'not_a_real_lens' isn't one of the 4 Literal values,
    so Pydantic's schema validation (not JSON parsing) fails and should
    trigger a retry with corrective feedback."""
    import json

    llm = SequenceLLM(
        [
            json.dumps({"active_lenses": ["not_a_real_lens"], "reasoning": "x"}),
            _classify_json([]),
        ]
    )
    report = triage_marketing_copy(
        SAMPLE_COPY, provider="openai", model="test-model", _llm=llm, max_retries=3
    )
    assert len(llm.calls) == 2
    assert report.active_lenses == []
    # The retry feedback must mention the actual validation failure, not
    # a generic "not valid JSON" message.
    retry_prompt = llm.calls[1]["prompt"]
    assert "schema validation failed" in retry_prompt.lower()


def test_graph_retry_exhaustion_raises_with_partial(monkeypatch):
    llm = SequenceLLM(["not json", "still not json", "nope"])
    with pytest.raises(TriageError) as exc_info:
        triage_marketing_copy(
            SAMPLE_COPY,
            provider="openai",
            model="test-model",
            _llm=llm,
            max_retries=3,
        )
    assert len(llm.calls) == 3
    assert "failed validation" in exc_info.value.message.lower()
    assert exc_info.value.partial is not None
    assert isinstance(exc_info.value.partial, TriageAttempt)
    assert len(exc_info.value.partial.validation_errors) >= 1


def test_graph_reviewer_phrase_not_in_source_triggers_retry(monkeypatch):
    """A reviewer hallucinating a flagged phrase that isn't actually in
    the copy must trigger the extra_validate retry, not silently pass
    through to the final report."""
    llm = SequenceLLM(
        [
            _classify_json(["health_wellness"]),
            _finding_json(
                "health_wellness", ["this phrase does not appear anywhere"], "high"
            ),
            _finding_json(
                "health_wellness", ["clinically proven to cure your cold"], "high"
            ),
            _aggregate_json(["health_wellness"], overall_risk_level="high"),
        ]
    )
    report = triage_marketing_copy(
        SAMPLE_COPY, provider="openai", model="test-model", _llm=llm, max_retries=3
    )
    assert len(llm.calls) == 4  # reviewer retried once
    assert report.findings[0].flagged_phrases == ["clinically proven to cure your cold"]


# --- 4. R5 case 3: _translate_api_error ----------------------------------


def test_translate_api_error_rate_limit_by_status():
    class E(Exception):
        status_code = 429

    result = _translate_api_error(E("body"))
    assert isinstance(result, TriageError)
    assert "rate-limited" in result.message.lower()


def test_translate_api_error_rate_limit_by_class_name():
    class RateLimitError(Exception):
        pass

    result = _translate_api_error(RateLimitError("no status"))
    assert "rate-limited" in result.message.lower()


def test_translate_api_error_auth_by_status():
    class E(Exception):
        status_code = 401

    result = _translate_api_error(E(""))
    assert "authentication" in result.message.lower()


def test_translate_api_error_auth_by_class_name():
    class AuthenticationError(Exception):
        pass

    result = _translate_api_error(AuthenticationError("bad key"))
    assert "authentication" in result.message.lower()


def test_translate_api_error_class_check_priority():
    class RateLimitError(Exception):
        pass

    result = _translate_api_error(RateLimitError("some unrelated body text"))
    assert "rate-limited" in result.message.lower()


def test_translate_api_error_ollama_connection():
    class ConnectError(Exception):
        pass

    result = _translate_api_error(ConnectError("failed"))
    assert "ollama" in result.message.lower()


def test_translate_api_error_unknown_preserves_original():
    result = _translate_api_error(ValueError("something weird happened"))
    assert "ValueError" in result.message
    assert "something weird happened" in result.message


# --- 5. Pure helpers ------------------------------------------------------


def test_route_after_classify_zero_active():
    assert _agent._route_after_classify({"active_lenses": []}) == ["aggregate"]


def test_route_after_classify_one_active():
    assert _agent._route_after_classify({"active_lenses": ["health_wellness"]}) == [
        "review_health"
    ]


def test_route_after_classify_two_active():
    result = _agent._route_after_classify(
        {"active_lenses": ["health_wellness", "environmental"]}
    )
    assert result == ["review_health", "review_environmental"]


def test_route_after_classify_all_four_active():
    all_lenses = [
        "health_wellness",
        "financial_earnings",
        "childrens_privacy",
        "environmental",
    ]
    result = _agent._route_after_classify({"active_lenses": all_lenses})
    assert set(result) == {
        "review_health",
        "review_financial",
        "review_privacy",
        "review_environmental",
    }


def test_resolve_provider_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider() == "openai"


def test_resolve_provider_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert resolve_provider() == "anthropic"


def test_resolve_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


def test_normalize_for_substring_collapses_whitespace():
    assert _normalize_for_substring("a   b\nc") == "a b c"


def test_parse_json_object_strips_markdown_fence():
    parsed = _parse_json_object('```json\n{"a": 1}\n```')
    assert parsed == {"a": 1}


def test_parse_json_object_handles_leading_prose():
    parsed = _parse_json_object('Here is the JSON: {"a": 1}')
    assert parsed == {"a": 1}


def test_parse_json_object_returns_error_string_on_garbage():
    result = _parse_json_object("not json at all")
    assert isinstance(result, str)
