"""Read/write helpers for `pending_outreach.json` -- the hand-rolled
cross-invocation state file that bridges phase 1 (send) and phase 2
(reply), the one genuinely new pattern in this catalog: every other
agent's `last_run.json` is write-once and never read back by a later,
separate invocation (confirmed by grep across the whole repo).

Each entry is keyed by `lead_id` and keyed a second way internally by
`contact_email` (via `find_dead_email`) so phase 1 can check a new
batch against leads who already unsubscribed in a *previous* batch,
even if that previous batch used a different `lead_id` scheme for the
same person.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LeadNotFound(Exception):
    """Raised when phase 2 (process_reply) is invoked with a lead_id
    not present in pending_outreach.json."""


class StateStoreError(Exception):
    """Raised when pending_outreach.json exists but can't be parsed.
    Never silently treated as empty -- that would silently orphan
    every lead already recorded as pending."""


def load_pending_outreach(path: Path) -> dict[str, dict[str, Any]]:
    """Returns {} if the file doesn't exist yet (first-ever run).
    Raises StateStoreError on a corrupt/unparseable file."""
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise StateStoreError(
            f"Couldn't read/parse {path}: {type(exc).__name__}: {exc}. "
            "This file tracks pending outreach state; fix or remove it "
            "manually before continuing (removing it loses the ability "
            "to process replies to already-sent leads)."
        ) from exc
    if not isinstance(parsed, dict):
        raise StateStoreError(
            f"{path} does not contain a JSON object at the top level."
        )
    return parsed


def save_pending_outreach(path: Path, state: dict[str, dict[str, Any]]) -> None:
    """Write to a .tmp sibling then os.replace() -- atomic-ish, so a
    crash mid-write never leaves a half-written file corrupting the
    next read. New to this repo (no existing last_run.json writer
    needed this, since none of those are ever read back)."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_path, path)


def find_dead_email(state: dict[str, dict[str, Any]], contact_email: str) -> bool:
    """True iff any entry in `state` with this contact_email has
    status == 'dead'. Keyed by email, not lead_id, since lead_id isn't
    guaranteed stable across separately-prepared leads.json files for
    the same person -- this is what makes the phase-2 unsubscribe
    override actually durable across future batches."""
    normalized = contact_email.strip().lower()
    for entry in state.values():
        lead = entry.get("lead", {})
        if lead.get("contact_email", "").strip().lower() == normalized and entry.get("status") == "dead":
            return True
    return False


def upsert_lead_entry(
    state: dict[str, dict[str, Any]],
    *,
    lead: dict[str, Any],
    qualification: dict[str, Any],
    drafted_email: dict[str, Any],
    sender: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Builds/overwrites the one entry phase 1 writes per SENT lead:
    lead info, qualification reasoning, the drafted email, AND the
    SenderProfile used for it (so phase 2 can draft a consistent
    follow-up without the user re-supplying sender.json), plus a
    status field starting at 'awaiting_reply'."""
    import datetime

    lead_id = lead["lead_id"]
    state = dict(state)
    state[lead_id] = {
        "lead": lead,
        "qualification": qualification,
        "drafted_email": drafted_email,
        "sender": sender,
        "status": "awaiting_reply",
        "written_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    return state


def _get_entry(state: dict[str, dict[str, Any]], lead_id: str) -> dict[str, Any]:
    entry = state.get(lead_id)
    if entry is None:
        raise LeadNotFound(
            f"No pending outreach found for lead_id={lead_id!r}. Check the "
            "id, or re-run phase 1 (`send`) for this lead first."
        )
    return entry


def mark_lead_dead(state: dict[str, dict[str, Any]], lead_id: str) -> dict[str, dict[str, Any]]:
    state = dict(state)
    entry = dict(_get_entry(state, lead_id))
    entry["status"] = "dead"
    state[lead_id] = entry
    return state


def mark_lead_snoozed(
    state: dict[str, dict[str, Any]], lead_id: str, until: str | None
) -> dict[str, dict[str, Any]]:
    state = dict(state)
    entry = dict(_get_entry(state, lead_id))
    entry["status"] = "snoozed"
    entry["snooze_until"] = until
    state[lead_id] = entry
    return state


def mark_lead_followed_up(
    state: dict[str, dict[str, Any]], lead_id: str
) -> dict[str, dict[str, Any]]:
    state = dict(state)
    entry = dict(_get_entry(state, lead_id))
    entry["status"] = "followed_up"
    state[lead_id] = entry
    return state


def get_lead_entry(state: dict[str, dict[str, Any]], lead_id: str) -> dict[str, Any]:
    """Public read accessor for agent.py's process_reply -- raises
    LeadNotFound via the same path upsert/mark helpers use."""
    return _get_entry(state, lead_id)
