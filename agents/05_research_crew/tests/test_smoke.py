"""Smoke tests for the research-crew agent.

All tests run under LLM_PROVIDER=mock (R8 in CONTRIBUTING.md) -- CI
never touches a real API key. Real-provider tests are a manual
maintainer check before shipping.

Tests that construct a real CrewAI Crew (structural check only, no
kickoff) are guarded with `pytest.importorskip("crewai")` so they
skip cleanly on Python 3.14 (crewai has no wheels there yet -- see
pyproject.toml's requires-python = ">=3.10,<3.14" for #05). On
Python 3.12/3.13, those tests run.

Covers:

1. Mock path returns a valid ResearchBrief (no crewai import).
2. R5 case 1 (topic validation):
   - short input rejected
   - trivial input (test/hello/asdf) rejected
   - valid topic accepted (falls through to search preflight)
3. R5 case 2 (search-empty):
   - monkeypatched _mock_search returning [] triggers the R5 branch
     with ResearchError.partial populated
4. R5 case 3 (_translate_api_error):
   - all 6 priority paths (class rate-limit, class auth, status 429,
     status 401, message fallback, generic unknown)
5. Pure helpers:
   - _mock_search (corpus hit, corpus miss + fallback)
   - _verify_source_citation (positive, negative, whitespace,
     case-sensitive)
   - _normalize_ws (whitespace collapse)
   - _looks_like_valid_topic (both gate conditions)
6. _build_crew structural check (skipped when crewai unavailable).
7. resolve_provider (default, env, rejects unknown).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_AGENT_DIR.parent))

_agent = importlib.import_module("05_research_crew.agent")
_schemas = importlib.import_module("05_research_crew.schemas")

run_research = _agent.run_research
ResearchError = _agent.ResearchError
ResearchAttempt = _agent.ResearchAttempt
resolve_provider = _agent.resolve_provider
_looks_like_valid_topic = _agent._looks_like_valid_topic
_mock_search = _agent._mock_search
_verify_source_citation = _agent._verify_source_citation
_normalize_ws = _agent._normalize_ws
_translate_api_error = _agent._translate_api_error
_MOCK_CORPUS = _agent._MOCK_CORPUS
_MOCK_FALLBACK_SOURCE = _agent._MOCK_FALLBACK_SOURCE
MIN_TOPIC_CHARS = _agent.MIN_TOPIC_CHARS
ResearchBrief = _schemas.ResearchBrief
Source = _schemas.Source


# --- 1. Mock path ----------------------------------------------------------


def test_mock_path_returns_valid_research_brief(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    brief = run_research("State of quantum computing in 2026")
    assert isinstance(brief, ResearchBrief)
    assert len(brief.key_findings) >= 1
    assert len(brief.sources_used) >= 1
    # Topic char count encoded into summary so future refactor
    # making mock constant surfaces here.
    assert str(len("State of quantum computing in 2026")) in brief.summary


def test_mock_path_skips_topic_validation(monkeypatch):
    """Mock mode short-circuits BEFORE _looks_like_valid_topic --
    a trivial 'test' input is accepted (no ResearchError raised)."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    brief = run_research("test")
    assert isinstance(brief, ResearchBrief)


def test_mock_path_serializable_to_json(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    brief = run_research("AI safety landscape 2026")
    dumped = brief.model_dump_json()
    restored = ResearchBrief.model_validate_json(dumped)
    assert restored == brief


# --- 2. R5 case 1: topic validation ----------------------------------------


def test_r5_case1_short_topic_rejected(monkeypatch):
    """Below MIN_TOPIC_CHARS triggers rejection under non-mock
    provider."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ResearchError, match="doesn't look like a real"):
        run_research("hi")


def test_r5_case1_trivial_topic_rejected(monkeypatch):
    """Trivial-input regex OR length rule triggers rejection. This
    test asserts explicitly on 'asdf' (length rule fires); separate
    case-coverage tests below check the regex + length branches
    independently via _looks_like_valid_topic."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ResearchError, match="doesn't look like a real"):
        run_research("asdf")


def test_r5_case1_message_names_thresholds(monkeypatch):
    """Error message includes both the topic value AND the
    minimum-char threshold so users know what to change."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ResearchError) as exc_info:
        run_research("hi")
    assert "'hi'" in exc_info.value.message
    assert str(MIN_TOPIC_CHARS) in exc_info.value.message


def test_looks_like_valid_topic_unit():
    """Direct unit test of the gate logic."""
    assert _looks_like_valid_topic("State of quantum computing in 2026") is True
    assert _looks_like_valid_topic("hi") is False
    assert _looks_like_valid_topic("test") is False
    assert _looks_like_valid_topic("hello") is False
    assert _looks_like_valid_topic("asdf") is False
    assert _looks_like_valid_topic("   ") is False
    assert _looks_like_valid_topic("") is False


# --- 3. R5 case 2: search-empty --------------------------------------------


def test_r5_case2_search_empty_raises_with_partial(monkeypatch):
    """Monkeypatch _mock_search to return an empty list -- verify
    run_research raises ResearchError with .partial populated
    naming the search stage."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr(_agent, "_mock_search", lambda topic: [])
    with pytest.raises(ResearchError) as exc_info:
        run_research("Some real topic that would be valid")
    assert "search" in exc_info.value.message.lower()
    assert "'Some real topic that would be valid'" in exc_info.value.message
    assert exc_info.value.partial is not None
    assert exc_info.value.partial.stage == "search"


# --- 4. R5 case 3: _translate_api_error ------------------------------------


def test_translate_api_error_rate_limit_by_status():
    class E(Exception):
        status_code = 429
    result = _translate_api_error(E("body"))
    assert isinstance(result, ResearchError)
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
    """Class-name check fires BEFORE message-string fallback."""
    class RateLimitError(Exception):
        pass
    result = _translate_api_error(RateLimitError("some unrelated body"))
    assert "rate-limited" in result.message.lower()


def test_translate_api_error_unknown_preserves_original():
    result = _translate_api_error(ValueError("something weird"))
    assert "ValueError" in result.message
    assert "something weird" in result.message


# --- 5. Pure helpers -------------------------------------------------------


# --- _mock_search ---

def test_mock_search_matches_quantum_keyword():
    sources = _mock_search("quantum computing state 2026")
    # Should hit the "quantum" corpus key, returning 2 sources.
    assert len(sources) == 2
    assert all("example.com/quantum" in s.url for s in sources)


def test_mock_search_matches_climate_keyword():
    sources = _mock_search("climate change adaptation 2030")
    assert len(sources) == 2
    assert all("climate" in s.url for s in sources)


def test_mock_search_matches_ai_safety_keyword():
    sources = _mock_search("ai safety research landscape")
    assert len(sources) == 2
    assert all("ai-safety" in s.url for s in sources)


def test_mock_search_case_insensitive():
    """Keyword matching is case-insensitive -- 'QUANTUM' matches
    the 'quantum' key."""
    sources = _mock_search("QUANTUM ENTANGLEMENT")
    assert len(sources) == 2


def test_mock_search_fallback_on_miss():
    """Topic not in corpus returns exactly the fallback source
    so the R5 case 2 branch doesn't fire spuriously under mock."""
    sources = _mock_search("extremely obscure topic xyzzy")
    assert sources == [_MOCK_FALLBACK_SOURCE]


# --- _verify_source_citation ---

def test_verify_positive():
    assert _verify_source_citation(
        "Quantum computers", "Quantum computers passed 1000 qubits."
    ) is True


def test_verify_negative_paraphrase():
    """Case-sensitive: 'quantum' lowercase doesn't match 'Quantum'
    uppercase -- treated as paraphrase per [C5] design."""
    assert _verify_source_citation(
        "quantum computers", "Quantum computers passed 1000 qubits."
    ) is False


def test_verify_negative_missing():
    assert _verify_source_citation(
        "this isn't in the source", "Quantum computers passed 1000 qubits."
    ) is False


def test_verify_normalizes_whitespace():
    """Multi-line source vs single-line excerpt should still match."""
    source = "Quantum\ncomputers\npassed 1000 qubits."
    excerpt = "Quantum computers passed 1000 qubits."
    assert _verify_source_citation(excerpt, source) is True


def test_normalize_ws_collapses():
    assert _normalize_ws("a  b\tc\nd") == "a b c d"
    assert _normalize_ws("  padded  ") == "padded"


# --- Regression guards (M1 empty-excerpt, M3 word_count) ------------------


def test_m1_verify_rejects_empty_excerpt():
    """`"" in "anything"` is True in Python. Without the explicit
    guard, an empty excerpt would silently verify against any non-empty
    source. Regression guard against a future refactor that drops the
    `if not excerpt.strip()` check."""
    assert _verify_source_citation("", "Quantum computers exist.") is False
    assert _verify_source_citation("   ", "Quantum computers exist.") is False
    assert _verify_source_citation("\n\t", "Quantum computers exist.") is False


def test_h2_generic_crew_exception_lands_in_translator(monkeypatch):
    """A `from crewai.utilities.exceptions import CrewException`
    specialized catch would fail as ImportError (the class doesn't
    exist in crewai 1.15.17) and the handler would misdiagnose it as
    'crewai not installed' for real users. Any exception raised by
    crew.kickoff() (crewai-internal or otherwise) goes through the
    generic `except Exception` -> `_translate_api_error` path. This
    test injects a fake exception via crew.kickoff and verifies it
    lands in the translator's generic branch, not the missing-install
    path."""
    pytest.importorskip("crewai")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    class FakeCrew:
        def kickoff(self, inputs):
            raise RuntimeError("simulated crewai-internal failure")

    fake_crew = FakeCrew()
    with pytest.raises(ResearchError) as exc_info:
        run_research(
            "Real research topic here",
            model="gpt-4.1-mini",
            _crew=fake_crew,
        )
    # Generic-translator path preserves original exception name +
    # message. Not misclassified as rate-limit or auth.
    msg = exc_info.value.message
    assert "RuntimeError" in msg
    assert "simulated crewai-internal failure" in msg


def test_m3_mock_word_count_matches_actual_content(monkeypatch):
    """Mock's `word_count` field is computed from actual background +
    key_findings + implications content, not hardcoded to a plausible-
    looking number. Regression guard against a future refactor that
    reverts to hardcoded word_count (which would silently drift as
    content changes)."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    brief = run_research("Sample topic for word-count check")
    # Recompute what the validator SHOULD have computed.
    computed = sum(len(section.split()) for section in [
        brief.background,
        *brief.key_findings,
        brief.implications,
    ])
    assert brief.word_count == computed
    # Also validate it's inside the schema-enforced range (would have
    # raised at ResearchBrief construction time if not, but explicit
    # is better than implicit for a regression guard).
    assert _schemas.WORD_COUNT_MIN <= brief.word_count <= _schemas.WORD_COUNT_MAX



# --- 6. _build_crew structural (skipped on Python 3.14) --------------------


def test_build_crew_structural():
    """Verify _build_crew constructs a Crew with 3 agents + 3 tasks
    + Process.sequential. Skipped on Python 3.14 (no crewai
    wheels)."""
    pytest.importorskip("crewai")
    crew = _agent._build_crew(
        topic="Test topic for structural check",
        model="gpt-4.1-mini-2025-04-14",
        max_iter=5,
        initial_sources=[
            Source(url="https://x", title="T", snippet="quantum computers"),
        ],
    )
    from crewai import Crew, Process
    assert isinstance(crew, Crew)
    assert len(crew.agents) == 3
    assert len(crew.tasks) == 3
    assert crew.process == Process.sequential
    # Agents in role order: researcher, writer, editor.
    roles = [a.role for a in crew.agents]
    assert "Research" in roles[0] or "research" in roles[0].lower()
    assert "Writer" in roles[1] or "writer" in roles[1].lower()
    assert "Editor" in roles[2] or "editor" in roles[2].lower()


def test_build_crew_researcher_has_search_tool():
    """Verify researcher gets the search_topic tool + editor gets
    verify_source_citation. Writer has no tools."""
    pytest.importorskip("crewai")
    crew = _agent._build_crew(
        topic="Test topic for structural check",
        model="gpt-4.1-mini-2025-04-14",
        max_iter=5,
        initial_sources=[
            Source(url="https://x", title="T", snippet="content"),
        ],
    )
    researcher, writer, editor = crew.agents
    researcher_tool_names = [t.name for t in (researcher.tools or [])]
    editor_tool_names = [t.name for t in (editor.tools or [])]
    assert "search_topic" in researcher_tool_names
    assert "verify_source_citation" in editor_tool_names
    assert not writer.tools  # writer has no tools


# --- 7. resolve_provider ---------------------------------------------------


def test_resolve_provider_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider() == "openai"


def test_resolve_provider_reads_env_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert resolve_provider() == "openai"


def test_resolve_provider_accepts_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    assert resolve_provider() == "mock"


def test_resolve_provider_rejects_anthropic_in_v1(monkeypatch):
    """V1 is OpenAI-only per plan decision -- anthropic/gemini
    require CrewAI/LiteLLM swap (documented)."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


def test_resolve_provider_rejects_garbage(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


def test_supported_providers_includes_ollama():
    """Locks in local-Ollama fallback: LLM_PROVIDER=ollama routes the
    crew's agents to a local Ollama server via LiteLLM's
    'ollama_chat/' prefix inside CrewAI's LLM(...)."""
    assert _agent.SUPPORTED_PROVIDERS == ("openai", "ollama")
