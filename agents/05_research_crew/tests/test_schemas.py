"""Schema unit tests for the research-crew agent.

F5.1 scope: schemas only. Behavioral tests for agent.py's crew, R5
error branches, and tool functions land in test_smoke.py at F5.4.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# `05_research_crew` starts with a digit -- invalid Python identifier
# for normal import syntax. Same importlib bootstrap as #01-04.
_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AGENT_DIR.parent))

_schemas = importlib.import_module("05_research_crew.schemas")
Source = _schemas.Source
ResearchBrief = _schemas.ResearchBrief
ResearchNotes = _schemas.ResearchNotes
WORD_COUNT_MIN = _schemas.WORD_COUNT_MIN
WORD_COUNT_MAX = _schemas.WORD_COUNT_MAX


# --- Source ----------------------------------------------------------------


def test_source_minimal_valid():
    src = Source(url="https://example.com", title="Example", snippet="hello world")
    assert src.url == "https://example.com"
    assert src.title == "Example"


def test_source_rejects_empty_url():
    with pytest.raises(ValidationError):
        Source(url="", title="t", snippet="s")


def test_source_rejects_empty_title():
    with pytest.raises(ValidationError):
        Source(url="https://x", title="", snippet="s")


def test_source_rejects_empty_snippet():
    """Editor's verify_source_citation tool guarantees snippets are
    non-empty by the time we're validating a ResearchBrief; the
    Pydantic constraint is a belt-and-braces guard against a future
    refactor that skips the editor."""
    with pytest.raises(ValidationError):
        Source(url="https://x", title="t", snippet="")


# --- ResearchNotes ---------------------------------------------------------


def _make_source() -> Source:
    return Source(url="https://example.com", title="Example", snippet="context")


def test_notes_valid():
    notes = ResearchNotes(topic="quantum computing", sources=[_make_source()])
    assert notes.topic == "quantum computing"
    assert len(notes.sources) == 1


def test_notes_rejects_empty_sources():
    """R5 case 2 (search-empty) fires upstream, so by the time we're
    constructing ResearchNotes, sources must be non-empty."""
    with pytest.raises(ValidationError):
        ResearchNotes(topic="anything", sources=[])


def test_notes_key_facts_defaults_to_empty():
    notes = ResearchNotes(topic="x", sources=[_make_source()])
    assert notes.key_facts == []


# --- ResearchBrief ---------------------------------------------------------


def _make_valid_brief(**overrides) -> ResearchBrief:
    defaults = {
        "topic": "Test topic",
        "summary": "This is a test summary spanning two sentences.",
        "background": "Background paragraph explaining the topic.",
        "key_findings": ["Finding one.", "Finding two."],
        "implications": "This means readers should know about the topic.",
        "sources_used": [_make_source()],
        "word_count": 400,
    }
    defaults.update(overrides)
    return ResearchBrief(**defaults)


def test_brief_minimal_valid_construction():
    brief = _make_valid_brief()
    assert brief.topic == "Test topic"
    assert len(brief.sources_used) == 1
    assert brief.word_count == 400


def test_brief_rejects_empty_topic():
    with pytest.raises(ValidationError):
        _make_valid_brief(topic="")


def test_brief_rejects_empty_summary():
    with pytest.raises(ValidationError):
        _make_valid_brief(summary="")


def test_brief_rejects_empty_background():
    with pytest.raises(ValidationError):
        _make_valid_brief(background="")


def test_brief_rejects_empty_implications():
    with pytest.raises(ValidationError):
        _make_valid_brief(implications="")


def test_brief_rejects_empty_key_findings():
    """A brief with zero findings is never valid -- R5 branch would
    have fired upstream if the researcher gathered nothing."""
    with pytest.raises(ValidationError):
        _make_valid_brief(key_findings=[])


def test_brief_rejects_empty_sources_used():
    """Editor strips sources whose snippet fails verification -- but
    at least one must survive for a valid brief."""
    with pytest.raises(ValidationError):
        _make_valid_brief(sources_used=[])


def test_brief_word_count_at_min_allowed():
    brief = _make_valid_brief(word_count=WORD_COUNT_MIN)
    assert brief.word_count == WORD_COUNT_MIN


def test_brief_word_count_at_max_allowed():
    brief = _make_valid_brief(word_count=WORD_COUNT_MAX)
    assert brief.word_count == WORD_COUNT_MAX


def test_brief_word_count_below_min_rejected():
    with pytest.raises(ValidationError, match="outside allowed range"):
        _make_valid_brief(word_count=WORD_COUNT_MIN - 1)


def test_brief_word_count_above_max_rejected():
    with pytest.raises(ValidationError, match="outside allowed range"):
        _make_valid_brief(word_count=WORD_COUNT_MAX + 1)


def test_brief_word_count_zero_rejected():
    """Field-level ge=1 catches this before the range validator."""
    with pytest.raises(ValidationError):
        _make_valid_brief(word_count=0)


def test_brief_source_with_whitespace_only_snippet_rejected():
    """Whitespace-only snippet passes Source's min_length=1 (since
    ' ' has length 1) but the cross-field validator catches it
    because .strip() is empty. Regression guard for [C5] pattern."""
    src = Source(url="https://x", title="t", snippet="   ")
    with pytest.raises(ValidationError, match="empty snippet"):
        _make_valid_brief(sources_used=[src])


def test_brief_serializable_to_and_from_json():
    """Round-trips cleanly -- CrewAI's output_pydantic validation +
    the mock path + the UI all depend on this."""
    original = _make_valid_brief(
        topic="AI safety",
        key_findings=["Finding A", "Finding B", "Finding C"],
        sources_used=[
            Source(url="https://a.com", title="A", snippet="from A"),
            Source(url="https://b.com", title="B", snippet="from B"),
        ],
        word_count=500,
    )
    dumped = original.model_dump_json()
    restored = ResearchBrief.model_validate_json(dumped)
    assert restored == original
