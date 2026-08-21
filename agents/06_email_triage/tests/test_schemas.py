"""Schema unit tests for the email-triage agent.

F6.1 scope: schemas only. Behavioral tests for agent.py's Agent,
tools, and R5 branches land in test_smoke.py at F6.4.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AGENT_DIR.parent))

_schemas = importlib.import_module("06_email_triage.schemas")
EmailTriage = _schemas.EmailTriage
TriageDeps = _schemas.TriageDeps


def _make_triage(**overrides) -> EmailTriage:
    defaults = {
        "category": "work",
        "priority": "medium",
        "action": "respond_later",
        "reasoning": "Legitimate client request, no urgency named.",
        "key_snippet": "Please review the attached spec by end of week.",
        "suggested_reply": "Will do -- I'll send comments by Thursday.",
    }
    defaults.update(overrides)
    return EmailTriage(**defaults)


# --- Basic construction ----------------------------------------------------


def test_triage_minimal_valid_construction():
    t = _make_triage(suggested_reply=None, action="archive")
    assert t.category == "work"
    assert t.suggested_reply is None


def test_triage_all_categories_accepted():
    for cat in ["work", "personal", "newsletter", "spam", "important", "notification", "other"]:
        _make_triage(category=cat, action="archive", suggested_reply=None)


def test_triage_all_priorities_accepted():
    for pri in ["urgent", "high", "medium", "low"]:
        _make_triage(priority=pri, action="archive", suggested_reply=None)


def test_triage_all_actions_accepted():
    for act in ["respond_now", "respond_later", "delegate", "archive", "ignore"]:
        reply = "draft" if act in ("respond_now", "respond_later") else None
        _make_triage(action=act, suggested_reply=reply)


def test_triage_rejects_invented_category():
    with pytest.raises(ValidationError):
        _make_triage(category="urgent_work")


def test_triage_rejects_invented_priority():
    with pytest.raises(ValidationError):
        _make_triage(priority="critical")


def test_triage_rejects_invented_action():
    with pytest.raises(ValidationError):
        _make_triage(action="forward")


def test_triage_rejects_empty_reasoning():
    with pytest.raises(ValidationError):
        _make_triage(reasoning="")


def test_triage_rejects_empty_key_snippet():
    with pytest.raises(ValidationError):
        _make_triage(key_snippet="")


# --- Cross-field validator (suggested_reply <-> action) --------------------


def test_reply_kept_when_action_is_respond_now():
    t = _make_triage(action="respond_now", suggested_reply="OK, on it.")
    assert t.suggested_reply == "OK, on it."


def test_reply_kept_when_action_is_respond_later():
    t = _make_triage(action="respond_later", suggested_reply="Will reply this week.")
    assert t.suggested_reply == "Will reply this week."


def test_reply_auto_nulled_when_action_is_archive():
    """H1-option-b pattern: mismatched reply on archive gets cleared,
    not raised. Mostly-good triage still returns something useful."""
    t = _make_triage(action="archive", suggested_reply="This shouldn't be here")
    assert t.suggested_reply is None


def test_reply_auto_nulled_when_action_is_ignore():
    t = _make_triage(action="ignore", suggested_reply="ignored anyway")
    assert t.suggested_reply is None


def test_reply_auto_nulled_when_action_is_delegate():
    t = _make_triage(action="delegate", suggested_reply="Not for me to reply")
    assert t.suggested_reply is None


def test_reply_none_stays_none_for_respond():
    """Validator only NULLS invalid replies; doesn't invent them."""
    t = _make_triage(action="respond_now", suggested_reply=None)
    assert t.suggested_reply is None


def test_triage_serializable_to_and_from_json():
    original = _make_triage(action="respond_now", suggested_reply="Ack, looking now.")
    dumped = original.model_dump_json()
    restored = EmailTriage.model_validate_json(dumped)
    assert restored == original


# --- TriageDeps ------------------------------------------------------------


def test_deps_default_empty():
    d = TriageDeps()
    assert d.known_contacts == []
    assert d.important_domains == []


def test_deps_with_values():
    d = TriageDeps(
        known_contacts=["alice@example.com", "Bob Smith"],
        important_domains=["@mycompany.com"],
    )
    assert len(d.known_contacts) == 2
    assert d.important_domains == ["@mycompany.com"]
