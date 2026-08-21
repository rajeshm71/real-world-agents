"""Schema unit tests for the meeting-notes agent.

F4.1 scope: schemas only. Behavioral tests for agent.py's ReAct loop,
R5 error branches, and tool functions land in test_smoke.py at F4.4.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# `04_meeting_notes` starts with a digit -- invalid Python identifier
# for normal import syntax. Same importlib bootstrap as #01/#02/#03.
_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AGENT_DIR.parent))

_schemas = importlib.import_module("04_meeting_notes.schemas")
ActionItem = _schemas.ActionItem
MeetingSummary = _schemas.MeetingSummary


# --- ActionItem ------------------------------------------------------------


def test_action_item_minimal_valid_construction():
    item = ActionItem(
        description="Send updated proposal to legal",
        priority="high",
        context_excerpt="Alice: I'll send the proposal to legal by Friday",
    )
    assert item.owner is None
    assert item.due_hint is None
    assert item.priority == "high"


def test_action_item_full_construction():
    item = ActionItem(
        description="Send updated proposal to legal",
        owner="Alice",
        due_hint="Friday",
        priority="high",
        context_excerpt="Alice: I'll send the proposal to legal by Friday",
    )
    assert item.owner == "Alice"
    assert item.due_hint == "Friday"


def test_action_item_rejects_invented_priority():
    """Closed Literal enforcement -- if this passes the model can
    invent 'medium-high' etc which breaks downstream sorting."""
    with pytest.raises(ValidationError):
        ActionItem(
            description="do it",
            priority="urgent",  # not in Priority
            context_excerpt="x",
        )


def test_action_item_rejects_empty_description():
    with pytest.raises(ValidationError):
        ActionItem(description="", priority="low", context_excerpt="x")


def test_action_item_rejects_empty_context_excerpt():
    """Excerpt is what lets the user verify the item was really said;
    empty defeats the whole grounding purpose."""
    with pytest.raises(ValidationError):
        ActionItem(description="x", priority="low", context_excerpt="")


def test_action_item_accepts_all_priorities():
    """Regression guard: if a value is removed from the Literal, this
    surfaces which one rather than a distant test failing cryptically."""
    for pri in ["low", "medium", "high"]:
        ActionItem(description="x", priority=pri, context_excerpt="x")


# --- MeetingSummary --------------------------------------------------------


def _make_valid_summary(**overrides) -> MeetingSummary:
    defaults = {
        "overall_summary": "Sprint planning: agreed on 3 stories for next sprint.",
    }
    defaults.update(overrides)
    return MeetingSummary(**defaults)


def test_summary_minimal_valid_construction():
    summary = _make_valid_summary()
    assert summary.meeting_topic is None
    assert summary.participants == []
    assert summary.action_items == []
    assert summary.key_decisions == []


def test_summary_rejects_empty_overall_summary():
    with pytest.raises(ValidationError):
        _make_valid_summary(overall_summary="")


def test_summary_empty_action_items_is_valid():
    """A meeting with zero action items is a real, valid outcome
    (pure discussion / info-share). Regression guard against a future
    field-level constraint that would reject an empty list."""
    summary = _make_valid_summary(action_items=[])
    assert summary.action_items == []


def test_summary_with_participants_and_items():
    summary = _make_valid_summary(
        meeting_topic="Q3 planning",
        participants=["Alice", "Bob"],
        action_items=[
            ActionItem(
                description="Update roadmap",
                owner="Alice",
                priority="medium",
                context_excerpt="Alice will update the roadmap",
            ),
        ],
        key_decisions=["Postponed launch to Q4"],
    )
    assert len(summary.participants) == 2
    assert len(summary.action_items) == 1
    assert summary.key_decisions == ["Postponed launch to Q4"]


def test_summary_serializable_to_and_from_json():
    """Round-trips cleanly -- the OpenAI Agents SDK output_type=
    validation, the mock path, and the UI all depend on this."""
    original = _make_valid_summary(
        meeting_topic="Sprint retro",
        participants=["Alice", "Bob", "Carol"],
        action_items=[
            ActionItem(
                description="Write postmortem",
                owner="Bob",
                due_hint="EoW",
                priority="high",
                context_excerpt="Bob will write the postmortem by end of week",
            ),
        ],
        key_decisions=["Adopt async standups"],
    )
    dumped = original.model_dump_json()
    restored = MeetingSummary.model_validate_json(dumped)
    assert restored == original
