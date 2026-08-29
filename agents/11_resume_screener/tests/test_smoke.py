"""Smoke tests for the resume screener.

Zero real OpenAI calls (R8): every LLM path is either mock or driven
by an injected `_agent` stub. Loader tests use the real shipped
example files (~40 KB total) so pypdf + python-docx go through their
real paths.

Sections:
1. Mock path (5 tests: round-trip, JD-length echo, alignment,
   ranked-ids permutation, no openai import).
2. Loaders -- real files (10 tests: pdf, docx, md, txt, unknown,
   missing, dispatcher, name derivation, id derivation, empty-text
   guard).
3. Schema validators (10 tests: both directions of every rule).
4. Batch orchestration (5 tests: empty resumes, duplicate ids, JD
   empty, unique ids pass, deterministic tie-break).
5. R5 branches (6 tests: 6-branch translator).
6. Constants + example sanity (4 tests).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_AGENT_DIR.parent))

_agent = importlib.import_module("11_resume_screener.agent")
_schemas = importlib.import_module("11_resume_screener.schemas")

load_resume = _agent.load_resume
screen_candidates = _agent.screen_candidates
resolve_provider = _agent.resolve_provider
ScreenerError = _agent.ScreenerError
ScreenerAttempt = _agent.ScreenerAttempt
_translate_api_error = _agent._translate_api_error
_stem_to_display_name = _agent._stem_to_display_name
_load_resume_text = _agent._load_resume_text
_mock_screening_result = _agent._mock_screening_result
_rank = _agent._rank
SUPPORTED_PROVIDERS = _agent.SUPPORTED_PROVIDERS

ResumeInput = _schemas.ResumeInput
EvidenceExcerpt = _schemas.EvidenceExcerpt
DimensionScore = _schemas.DimensionScore
CandidateScorecard = _schemas.CandidateScorecard
ScreeningResult = _schemas.ScreeningResult
DIMENSION_NAMES = _schemas.DIMENSION_NAMES
_recommendation_for = _schemas._recommendation_for

_EXAMPLES = _AGENT_DIR / "examples"
_RESUMES = _EXAMPLES / "resumes"
_JD_FILE = _EXAMPLES / "job_description.md"


# --- Helpers ---------------------------------------------------------------


def _sample_resumes() -> list[ResumeInput]:
    return [
        ResumeInput(
            resume_id="alice",
            candidate_name="Alice",
            resume_text="Python distributed systems on-call ledger Postgres Kafka",
        ),
        ResumeInput(
            resume_id="bob",
            candidate_name="Bob",
            resume_text="Django REST APIs MySQL Celery",
        ),
    ]


def _good_scorecard(
    resume_id: str = "alice",
    resume_text: str = "Python Postgres Kafka on-call ledger",
    overall: int = 75,
) -> CandidateScorecard:
    excerpt = EvidenceExcerpt(quoted_text="Python Postgres")
    dims = [
        DimensionScore(
            name=name,
            score=overall,
            rationale="fits the JD",
            evidence=[excerpt] if overall >= 40 else [],
        )
        for name in DIMENSION_NAMES
    ]
    return CandidateScorecard(
        resume_id=resume_id,
        candidate_name=resume_id.capitalize(),
        resume_text=resume_text,
        dimensions=dims,
        overall_score=overall,
        overall_rationale="solid match",
        recommendation=_recommendation_for(overall),
    )


# --- 1. Mock path ----------------------------------------------------------


def test_mock_screen_returns_valid_result(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    result = screen_candidates("hire a backend engineer", _sample_resumes())
    assert isinstance(result, ScreeningResult)
    assert len(result.scorecards) == 2
    assert len(result.ranked_ids) == 2


def test_mock_echoes_jd_length_in_overall_rationale(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    jd = "a very specific JD text used to prove the mock read its input"
    result = screen_candidates(jd, _sample_resumes())
    for card in result.scorecards:
        assert f"length {len(jd)}" in card.overall_rationale


def test_mock_ranked_ids_is_permutation(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    result = screen_candidates("jd", _sample_resumes())
    assert sorted(result.ranked_ids) == sorted(
        s.resume_id for s in result.scorecards
    )


def test_mock_scorecards_align_one_to_one_with_resumes(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    resumes = _sample_resumes()
    result = screen_candidates("jd", resumes)
    result_ids = {s.resume_id for s in result.scorecards}
    input_ids = {r.resume_id for r in resumes}
    assert result_ids == input_ids


def test_mock_does_not_import_pydantic_ai(monkeypatch):
    """Mock path must not need pydantic-ai installed."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setitem(sys.modules, "pydantic_ai", None)
    result = screen_candidates("jd", _sample_resumes())
    assert result.run_meta["provider"] == "mock"


# --- 2. Loaders -- real files ---------------------------------------------


def test_pdf_loader_extracts_text():
    pytest.importorskip("pypdf")
    r = load_resume(_RESUMES / "alice.pdf")
    assert r.resume_id == "alice"
    assert r.candidate_name == "Alice"
    assert "Python" in r.resume_text
    assert "ledger" in r.resume_text.lower()


def test_docx_loader_extracts_text():
    pytest.importorskip("docx")
    r = load_resume(_RESUMES / "bob.docx")
    assert r.resume_id == "bob"
    assert "Django" in r.resume_text


def test_md_loader_reads_text():
    r = load_resume(_RESUMES / "carol.md")
    assert r.resume_id == "carol"
    assert "Wefund" in r.resume_text


def test_txt_loader_reads_text(tmp_path):
    p = tmp_path / "eve.txt"
    p.write_text("Just a short resume in plain text.", encoding="utf-8")
    r = load_resume(p)
    assert r.resume_id == "eve"
    assert r.resume_text.startswith("Just a short")


def test_unknown_suffix_rejected(tmp_path):
    p = tmp_path / "resume.rtf"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ScreenerError, match="unsupported resume format"):
        load_resume(p)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ScreenerError, match="does not exist"):
        load_resume(tmp_path / "nope.md")


def test_dispatcher_picks_correct_loader(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("hello", encoding="utf-8")
    assert _load_resume_text(p) == "hello"


def test_stem_to_display_name_snake_case():
    assert _stem_to_display_name("alice_smith") == "Alice Smith"
    assert _stem_to_display_name("alice-smith") == "Alice Smith"
    assert _stem_to_display_name("alice smith") == "Alice Smith"


def test_stem_to_display_name_single_token():
    assert _stem_to_display_name("alice") == "Alice"
    assert _stem_to_display_name("") == ""


def test_resume_id_is_lowercased_stem(tmp_path):
    p = tmp_path / "AliceSmith.md"
    p.write_text("hello", encoding="utf-8")
    r = load_resume(p)
    assert r.resume_id == "alicesmith"


def test_empty_extraction_rejected(tmp_path):
    p = tmp_path / "empty.md"
    p.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(ScreenerError, match="empty text"):
        load_resume(p)


# --- 3. Schema validators -------------------------------------------------


def test_evidence_must_be_verbatim_substring_positive():
    # Scored 75, evidence "Python Postgres" appears in resume_text.
    _good_scorecard()  # no raise


def test_evidence_must_be_verbatim_substring_negative():
    with pytest.raises(ValidationError, match="not a verbatim substring"):
        CandidateScorecard(
            resume_id="alice",
            candidate_name="Alice",
            resume_text="Python Postgres",
            dimensions=[
                DimensionScore(
                    name=name,
                    score=70,
                    rationale="ok",
                    evidence=[EvidenceExcerpt(quoted_text="Rust and OCaml")],
                )
                for name in DIMENSION_NAMES
            ],
            overall_score=70,
            overall_rationale="ok",
            recommendation="yes",
        )


def test_exactly_three_dimensions_missing_raises():
    with pytest.raises(ValidationError, match="canonical set"):
        CandidateScorecard(
            resume_id="a",
            candidate_name="A",
            resume_text="text",
            dimensions=[
                DimensionScore(
                    name="skills_match", score=50, rationale="ok",
                    evidence=[EvidenceExcerpt(quoted_text="text")],
                )
            ],
            overall_score=50,
            overall_rationale="ok",
            recommendation="borderline",
        )


def test_exactly_three_dimensions_duplicate_raises():
    with pytest.raises(ValidationError, match="canonical set"):
        CandidateScorecard(
            resume_id="a",
            candidate_name="A",
            resume_text="text",
            dimensions=[
                DimensionScore(
                    name="skills_match", score=50, rationale="ok",
                    evidence=[EvidenceExcerpt(quoted_text="text")],
                )
            ] * 3,
            overall_score=50,
            overall_rationale="ok",
            recommendation="borderline",
        )


def test_overall_within_ten_of_mean_pass():
    _good_scorecard(overall=70)  # all dims = 70, overall = 70; within 10.


def test_overall_more_than_ten_from_mean_raises():
    with pytest.raises(ValidationError, match="drifts more than 10"):
        CandidateScorecard(
            resume_id="a",
            candidate_name="A",
            resume_text="text",
            dimensions=[
                DimensionScore(
                    name=name, score=40, rationale="ok",
                    evidence=[EvidenceExcerpt(quoted_text="text")],
                )
                for name in DIMENSION_NAMES
            ],
            overall_score=90,  # mean=40, overall=90 -> drift=50
            overall_rationale="ok",
            recommendation="strong_yes",
        )


def test_recommendation_matches_rubric_pass():
    _good_scorecard(overall=45)  # borderline
    _good_scorecard(overall=85)  # strong_yes


def test_recommendation_contradicts_score_raises():
    with pytest.raises(ValidationError, match="rubric expects"):
        CandidateScorecard(
            resume_id="a",
            candidate_name="A",
            resume_text="text",
            dimensions=[
                DimensionScore(
                    name=name, score=85, rationale="ok",
                    evidence=[EvidenceExcerpt(quoted_text="text")],
                )
                for name in DIMENSION_NAMES
            ],
            overall_score=90,
            overall_rationale="ok",
            recommendation="no",  # rubric says strong_yes at 90
        )


def test_ranked_ids_not_permutation_raises():
    a, b = _good_scorecard("alice"), _good_scorecard("bob")
    with pytest.raises(ValidationError, match="permutation"):
        ScreeningResult(
            job_description="jd",
            scorecards=[a, b],
            ranked_ids=["alice", "charlie"],
            run_meta={},
        )


def test_rank_order_non_increasing():
    high = _good_scorecard("alice", overall=80)
    low = _good_scorecard("bob", overall=50)
    # Correct order: high before low.
    ScreeningResult(
        job_description="jd",
        scorecards=[high, low],
        ranked_ids=["alice", "bob"],
        run_meta={},
    )
    with pytest.raises(ValidationError, match="non-increasing"):
        ScreeningResult(
            job_description="jd",
            scorecards=[high, low],
            ranked_ids=["bob", "alice"],
            run_meta={},
        )


# --- 4. Batch orchestration -----------------------------------------------


def test_empty_resumes_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    with pytest.raises(ScreenerError, match="at least one resume"):
        screen_candidates("jd", [])


def test_duplicate_resume_ids_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    resumes = [
        ResumeInput(resume_id="dup", candidate_name="A", resume_text="x"),
        ResumeInput(resume_id="dup", candidate_name="B", resume_text="y"),
    ]
    with pytest.raises(ScreenerError, match="duplicate resume_id"):
        screen_candidates("jd", resumes)


def test_empty_jd_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    with pytest.raises(ScreenerError, match="job_description"):
        screen_candidates("   ", _sample_resumes())


def test_bad_provider_becomes_screener_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(ScreenerError, match="Unknown LLM_PROVIDER"):
        screen_candidates("jd", _sample_resumes())


def test_rank_deterministic_tie_break_by_id():
    # Two cards at equal overall -> id ascending order.
    a = _good_scorecard("bob", overall=70)
    b = _good_scorecard("alice", overall=70)
    assert _rank([a, b]) == ["alice", "bob"]


# --- 5. R5 error translator -----------------------------------------------


def test_translator_class_rate_limit():
    class RateLimitError(Exception):
        pass
    err = _translate_api_error(RateLimitError("slow down"))
    assert "rate-limited" in err.message


def test_translator_class_auth():
    class AuthenticationError(Exception):
        pass
    err = _translate_api_error(AuthenticationError("bad key"))
    assert "OPENAI_API_KEY" in err.message


def test_translator_status_429():
    exc = RuntimeError("x")
    exc.status_code = 429
    assert "rate-limited" in _translate_api_error(exc).message


def test_translator_status_401():
    exc = RuntimeError("x")
    exc.status_code = 401
    assert "OPENAI_API_KEY" in _translate_api_error(exc).message


def test_translator_message_rate_limit():
    assert "rate-limited" in _translate_api_error(
        RuntimeError("Rate limit exceeded")
    ).message


def test_translator_message_auth():
    assert "OPENAI_API_KEY" in _translate_api_error(
        RuntimeError("Invalid API key provided")
    ).message


def test_translator_generic_fallthrough():
    err = _translate_api_error(RuntimeError("something else"))
    assert "LLM call failed" in err.message


# --- 7. Post-review hardening (M1, M2, L1, L4) ----------------------------


class _ScriptedAgent:
    """Duck-typed stand-in for a PydanticAI Agent. `outputs` is a list
    of _ScoringOutput OR Exception instances; each `run_sync` call
    consumes the next one."""

    def __init__(self, outputs):
        self._outputs = list(outputs)

    def run_sync(self, user_prompt):
        item = self._outputs.pop(0)
        if isinstance(item, Exception):
            raise item

        class _Result:
            output = item

        return _Result()


def _good_scoring_output(overall: int = 70, resume_text: str = "Python Postgres"):
    """A _ScoringOutput-shaped duck-type that _score_one accepts."""
    ScoringOutput = _agent._ScoringOutput
    return ScoringOutput(
        dimensions=[
            DimensionScore(
                name=name,
                score=overall,
                rationale="ok",
                evidence=[EvidenceExcerpt(quoted_text=resume_text[:15])],
            )
            for name in DIMENSION_NAMES
        ],
        overall_score=overall,
        overall_rationale="ok",
        recommendation=_recommendation_for(overall),
    )


def test_batch_partial_attaches_completed_scorecards(monkeypatch):
    """M2: when resume #N fails mid-batch, the ScreenerError's partial
    must carry the (N-1) already-scored cards so the caller can salvage
    them instead of losing all work."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    # Three resumes; scoring succeeds twice, then raises a bad
    # ValidationError-shaped exception on the third.
    good1 = _good_scoring_output(overall=80, resume_text="Python distributed")
    good2 = _good_scoring_output(overall=60, resume_text="Django REST")
    agent = _ScriptedAgent(
        [good1, good2, RuntimeError("openai went boom on resume #3")]
    )
    resumes = [
        ResumeInput(resume_id="a", candidate_name="A", resume_text="Python distributed"),
        ResumeInput(resume_id="b", candidate_name="B", resume_text="Django REST"),
        ResumeInput(resume_id="c", candidate_name="C", resume_text="Rust systems"),
    ]
    with pytest.raises(ScreenerError) as exc_info:
        screen_candidates("jd", resumes, provider="openai", _agent=agent)
    assert exc_info.value.partial is not None
    completed = exc_info.value.partial.completed_scorecards
    assert len(completed) == 2
    assert {c.resume_id for c in completed} == {"a", "b"}


def test_score_one_narrow_catch_only_validation_error(monkeypatch):
    """M1: _score_one now catches ValidationError specifically, so a
    non-validation error from PydanticAI bubbles up unchanged and hits
    the R5 translator instead of being mislabeled as 'validation'."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    agent = _ScriptedAgent([RuntimeError("some_llm_side_error")])
    resumes = [
        ResumeInput(resume_id="a", candidate_name="A", resume_text="Python"),
    ]
    with pytest.raises(ScreenerError) as exc_info:
        screen_candidates("jd", resumes, provider="openai", _agent=agent)
    # Comes back through _translate_api_error's generic branch, NOT
    # "cross-field validation".
    assert "LLM call failed" in exc_info.value.message


def test_docx_loader_skips_empty_template_rows(tmp_path):
    """L1: rows made only of empty cells produce `" |  | "` noise
    strings; the loader should skip them."""
    docx = pytest.importorskip("docx")
    p = tmp_path / "with_empty_table.docx"
    doc = docx.Document()
    doc.add_paragraph("Real content here.")
    table = doc.add_table(rows=2, cols=3)
    # First row all empty (template placeholder), second row has content.
    table.rows[1].cells[0].text = "Header"
    table.rows[1].cells[1].text = "Value"
    doc.save(str(p))
    text = _agent._load_docx_text(p)
    assert "Real content here" in text
    assert "Header" in text
    # The all-empty row should NOT appear.
    assert " |  | " not in text


def test_pdf_loader_strips_pages_before_join():
    """L4: page text is stripped so multi-blank-line runs don't build
    up between pages. Verified against the shipped alice.pdf (single
    page here, but the strip-then-join contract is exercised)."""
    pytest.importorskip("pypdf")
    text = _agent._load_pdf_text(_RESUMES / "alice.pdf")
    # No run of three or more consecutive newlines.
    assert "\n\n\n" not in text


# --- 6. Constants + example sanity ----------------------------------------


def test_supported_providers_includes_openai_and_ollama():
    assert SUPPORTED_PROVIDERS == ("openai", "ollama")


def test_dimension_names_are_the_canonical_three():
    assert set(DIMENSION_NAMES) == {
        "skills_match", "experience_match", "culture_signal"
    }


def test_job_description_example_exists():
    assert _JD_FILE.exists()
    text = _JD_FILE.read_text(encoding="utf-8")
    assert len(text) > 200


def test_resumes_dir_has_one_per_supported_format():
    files = list(_RESUMES.iterdir())
    suffixes = {f.suffix.lower() for f in files if f.is_file()}
    # We ship at least .pdf, .docx, .md (txt is loader-supported but
    # not required in examples/).
    assert {".pdf", ".docx", ".md"}.issubset(suffixes)


# --- Phase 2: local-Ollama fallback ---------------------------------------


def test_build_agent_ollama_uses_local_endpoint(monkeypatch):
    pytest.importorskip("pydantic_ai")
    captured: dict = {}

    from pydantic_ai.providers import ollama as _ollama_prov_mod
    real_provider = _ollama_prov_mod.OllamaProvider

    class _CapturingProvider(real_provider):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(
        _ollama_prov_mod, "OllamaProvider", _CapturingProvider
    )

    agent = _agent._build_agent(model="gemma4:e4b", provider="ollama")
    assert captured["base_url"].endswith("/v1")
    assert "11434" in captured["base_url"]
    assert captured["api_key"] == "ollama"
    assert agent is not None


def test_translate_api_error_connection_refused_returns_ollama_hint():
    exc = ConnectionError("Connection refused: http://localhost:11434")
    err = _translate_api_error(exc)
    assert "ollama serve" in err.message.lower()
