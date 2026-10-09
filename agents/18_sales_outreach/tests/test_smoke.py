"""Smoke tests for the sales outreach agent.

All tests run under LLM_PROVIDER=mock where applicable (R8 in
CONTRIBUTING.md) -- CI never touches a real API key, and NEVER opens a
real socket (a negative-assertion test monkeypatches smtplib.SMTP to
raise if instantiated at all, proving it's never touched when it
shouldn't be).

Covers:

1. Mock-path batch result (no graph, no LLM, no socket).
2. R5 bad-input: empty leads, empty product/ICP/sender fields,
   duplicate lead_ids, malformed Lead (Pydantic-level).
3. Qualify-then-skip: fake SMTP sender asserted never called.
4. Qualify-then-send: both dry-run (true no-op) and --send paths.
5. Send failure + partial-failure salvage across a batch.
6. pending_outreach.json round-trip between phase 1 and phase 2.
7. LeadNotFound for an unknown lead_id.
8. Already-dead lead reprocessed: no-op, no raise, no send.
9. One test per fixed reply intent (5 total).
10. THE critical safety-override test: the model's own
    suggested_next_action is deliberately contradictory; the code
    overrides it regardless.
11. state_store round-trip tested as real local file I/O.
12. SmtpNotConfigured pre-flight (standalone, at send_email level).
13. _translate_api_error 6-branch.
14. CLI default-dry-run regression guard.
15. Draft missing the opt-out line/postal address triggers a retry.
16. Dead-lead cross-check durability (different lead_id, same email,
    a LATER batch).
17. Upfront credential pre-flight (before any lead is processed at
    all, not lazily inside the first lead's send node).

SequenceLLM fixture supplies scripted JSON responses in call order --
safe here (unlike agent #17) because this agent's phase-1 graph has no
parallel branches; every lead is processed strictly sequentially
(qualify then draft then send), so call order is always deterministic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AGENT_DIR.parent))

import importlib

_agent = importlib.import_module("18_sales_outreach.agent")
_schemas = importlib.import_module("18_sales_outreach.schemas")
_smtp_client = importlib.import_module("18_sales_outreach.smtp_client")
_state_store = importlib.import_module("18_sales_outreach.state_store")

run_outreach_batch = _agent.run_outreach_batch
process_reply = _agent.process_reply
OutreachError = _agent.OutreachError
resolve_provider = _agent.resolve_provider
_translate_api_error = _agent._translate_api_error
_route_after_qualify = _agent._route_after_qualify
_normalize_for_substring = _agent._normalize_for_substring
_parse_json_object = _agent._parse_json_object
Lead = _schemas.Lead
SenderProfile = _schemas.SenderProfile
OutreachBatchResult = _schemas.OutreachBatchResult
ReplyOutcome = _schemas.ReplyOutcome
SmtpNotConfigured = _smtp_client.SmtpNotConfigured
LeadNotFound = _state_store.LeadNotFound
StateStoreError = _state_store.StateStoreError


# --- Fixtures ------------------------------------------------------------


def _lead(lead_id="acme1", email="jane@acme.com", blurb="Just raised a Series B and is hiring 10 engineers"):
    return Lead(
        lead_id=lead_id,
        company="Acme",
        contact_name="Jane Doe",
        contact_title="VP Eng",
        contact_email=email,
        blurb=blurb,
    )


def _sender():
    return SenderProfile(
        sender_name="Bob",
        sender_company="MyCo",
        reply_to_note="Reply STOP",
        sender_postal_address="123 Main St, Springfield",
    )


def _opt_out_line(sender=None):
    sender = sender or _sender()
    return _agent._OPT_OUT_TEMPLATE.format(reply_to_note=sender.reply_to_note)


def _compliant_body(phrase, sender=None):
    sender = sender or _sender()
    opt_out = _opt_out_line(sender)
    return f"Hi there, noting that {phrase}. Best, {sender.sender_name}, {sender.sender_company}. {opt_out} {sender.sender_postal_address}"


# --- SequenceLLM fixture -----------------------------------------------


class SequenceLLM:
    """Test-only LLM Protocol impl. Returns responses from a pre-set
    list on successive .complete() calls, records every call so tests
    can assert call count / prompt content. Safe here (unlike #17)
    because phase 1's graph has no parallel branches -- every lead is
    strictly sequential."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, prompt, model, temperature=0.0, max_tokens=1024, cacheable_prefix=None):
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


def _qualify_json(fit_score, reasoning="test reasoning"):
    return json.dumps({"fit_score": fit_score, "reasoning": reasoning})


def _draft_json(subject, body, personalization_note):
    return json.dumps({"subject": subject, "body": body, "personalization_note": personalization_note})


def _classify_json(intent, reasoning="test reasoning", suggested_next_action="test action", snooze_until=None):
    return json.dumps(
        {
            "intent": intent,
            "reasoning": reasoning,
            "suggested_next_action": suggested_next_action,
            "snooze_until": snooze_until,
        }
    )


def _set_fake_smtp_env(monkeypatch):
    """dry_run=False makes run_outreach_batch/process_reply call the
    REAL load_smtp_credentials() up front (by design -- see the
    upfront-credential-resolution fix). Tests that fake the send step
    via _smtp_send_email still need these env vars set so credential
    resolution itself succeeds; the fake sender never actually uses
    them to open a socket."""
    monkeypatch.setenv("SMTP_USER", "test@example.com")
    monkeypatch.setenv("SMTP_APP_PASSWORD", "test-app-password")


@pytest.fixture
def never_called_smtp(monkeypatch):
    """Monkeypatches the real socket-opening function to raise if
    instantiated at all -- proves a code path never touches SMTP,
    regardless of what send_email()'s dry_run plumbing does."""

    class _ShouldNotBeCalled:
        def __init__(self, *a, **kw):
            raise AssertionError("smtplib.SMTP should never be instantiated in this test")

    monkeypatch.setattr("smtplib.SMTP", _ShouldNotBeCalled)


@pytest.fixture
def fake_smtp_module_level(monkeypatch):
    """Monkeypatches agent._smtp_send_email (the module-level
    reference `_send_email_node` calls) to a fake recording calls and
    returning a canned result -- the established 'monkeypatch the
    helper' convention for a real-world side effect (mirrors agent
    #08's approach to its own real-world helper). Also sets fake SMTP
    env vars since run_outreach_batch still calls the REAL
    load_smtp_credentials() up front when dry_run=False, independent
    of this fake."""
    _set_fake_smtp_env(monkeypatch)
    calls = []

    def _fake(*, to_address, subject, body, dry_run, creds=None, _sender=None):
        calls.append({"to_address": to_address, "subject": subject, "body": body, "dry_run": dry_run})
        if dry_run:
            return _schemas.SendEmailResult(sent=False, error=None, dry_run=True)
        return _schemas.SendEmailResult(sent=True, error=None, dry_run=False)

    monkeypatch.setattr(_agent, "_smtp_send_email", _fake)
    return calls


# --- 1. Mock path --------------------------------------------------------


def test_mock_path_returns_valid_batch_result(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    result = run_outreach_batch(
        [_lead()], sender=_sender(), product_description="A dev tool", icp_description="Startups"
    )
    assert isinstance(result, OutreachBatchResult)
    assert len(result.records) == 1


def test_mock_path_serializable_to_json():
    result = run_outreach_batch(
        [_lead()], sender=_sender(), product_description="x", icp_description="y", provider="mock"
    )
    dumped = result.model_dump_json()
    restored = OutreachBatchResult.model_validate_json(dumped)
    assert restored == result


def test_mock_reply_outcome_valid(monkeypatch):
    outcome = process_reply("whatever-id", "sounds interesting", provider="mock")
    assert isinstance(outcome, ReplyOutcome)


# --- 2. R5 bad-input ------------------------------------------------------


def test_empty_leads_raises():
    with pytest.raises(OutreachError, match="at least one lead"):
        run_outreach_batch([], sender=_sender(), product_description="x", icp_description="y", provider="openai")


def test_empty_product_description_raises():
    with pytest.raises(OutreachError, match="product_description"):
        run_outreach_batch(
            [_lead()], sender=_sender(), product_description="  ", icp_description="y", provider="openai"
        )


def test_empty_icp_description_raises():
    with pytest.raises(OutreachError, match="icp_description"):
        run_outreach_batch(
            [_lead()], sender=_sender(), product_description="x", icp_description=" ", provider="openai"
        )


def test_empty_sender_postal_address_raises():
    bad_sender = SenderProfile(
        sender_name="Bob", sender_company="MyCo", reply_to_note="Reply STOP", sender_postal_address=" "
    )
    with pytest.raises(OutreachError, match="sender_postal_address"):
        run_outreach_batch(
            [_lead()], sender=bad_sender, product_description="x", icp_description="y", provider="openai"
        )


def test_duplicate_lead_ids_raises():
    with pytest.raises(OutreachError, match="duplicate lead_id"):
        run_outreach_batch(
            [_lead(lead_id="dup"), _lead(lead_id="dup")],
            sender=_sender(),
            product_description="x",
            icp_description="y",
            provider="openai",
        )


def test_malformed_email_raises_at_construction():
    with pytest.raises(Exception, match="contact_email"):
        _lead(email="not-an-email")


def test_empty_reply_text_raises():
    with pytest.raises(OutreachError, match="reply_text"):
        process_reply("whatever", "   ", provider="openai")


# --- 3. Qualify-then-skip -------------------------------------------------


def test_qualify_then_skip_never_touches_smtp(never_called_smtp, fake_smtp_module_level):
    llm = SequenceLLM([_qualify_json(fit_score=10)])  # below default threshold (60)
    result = run_outreach_batch(
        [_lead()],
        sender=_sender(),
        product_description="x",
        icp_description="y",
        provider="openai",
        model="test-model",
        _llm=llm,
    )
    record = result.records[0]
    assert record.outcome == "skipped_not_qualified"
    assert record.drafted_email is None
    assert len(fake_smtp_module_level) == 0
    assert len(llm.calls) == 1  # qualify only -- no draft call either


# --- 4. Qualify-then-send: dry-run and --send -----------------------------


def test_dry_run_is_true_no_op(never_called_smtp):
    """Uses the REAL smtp_client.send_email (not a module-level fake
    for it) -- the dry_run short-circuit lives INSIDE send_email,
    before _send_via_smtp/smtplib is ever touched. never_called_smtp
    proves that short-circuit actually works: if dry_run didn't
    short-circuit correctly, this test would fail with the
    AssertionError from inside the patched smtplib.SMTP, not a clean
    result."""
    phrase = "Series B"
    body = _compliant_body(phrase)
    llm = SequenceLLM([_qualify_json(fit_score=90), _draft_json("Hi", body, phrase)])
    result = run_outreach_batch(
        [_lead()],
        sender=_sender(),
        product_description="x",
        icp_description="y",
        provider="openai",
        model="test-model",
        dry_run=True,
        _llm=llm,
    )
    record = result.records[0]
    assert record.outcome == "dry_run_drafted"
    assert record.send_result.dry_run is True
    assert record.send_result.sent is False


def test_real_send_calls_fake_sender(fake_smtp_module_level, tmp_path):
    phrase = "Series B"
    body = _compliant_body(phrase)
    llm = SequenceLLM([_qualify_json(fit_score=90), _draft_json("Hi", body, phrase)])
    result = run_outreach_batch(
        [_lead()],
        sender=_sender(),
        product_description="x",
        icp_description="y",
        provider="openai",
        model="test-model",
        dry_run=False,
        send_delay_seconds=0,
        state_path=tmp_path / "pending_outreach.json",
        _llm=llm,
    )
    record = result.records[0]
    assert record.outcome == "sent"
    assert len(fake_smtp_module_level) == 1
    assert fake_smtp_module_level[0]["to_address"] == "jane@acme.com"
    assert fake_smtp_module_level[0]["subject"] == "Hi"


# --- 5. Send failure + partial-failure salvage ----------------------------


def test_send_failure_partial_failure_salvage(monkeypatch, tmp_path):
    _set_fake_smtp_env(monkeypatch)
    calls = []

    def _fake(*, to_address, subject, body, dry_run, creds=None, _sender=None):
        calls.append(to_address)
        if to_address == "lead2@acme.com":
            return _schemas.SendEmailResult(sent=False, error="simulated SMTP failure", dry_run=False)
        return _schemas.SendEmailResult(sent=True, error=None, dry_run=False)

    monkeypatch.setattr(_agent, "_smtp_send_email", _fake)

    leads = [
        _lead(lead_id="l1", email="lead1@acme.com"),
        _lead(lead_id="l2", email="lead2@acme.com"),
        _lead(lead_id="l3", email="lead3@acme.com"),
    ]
    phrase = "Series B"
    body = _compliant_body(phrase)
    llm = SequenceLLM(
        [
            _qualify_json(fit_score=90),
            _draft_json("Hi1", body, phrase),
            _qualify_json(fit_score=90),
            _draft_json("Hi2", body, phrase),
            _qualify_json(fit_score=90),
            _draft_json("Hi3", body, phrase),
        ]
    )
    result = run_outreach_batch(
        leads,
        sender=_sender(),
        product_description="x",
        icp_description="y",
        provider="openai",
        model="test-model",
        dry_run=False,
        send_delay_seconds=0,
        state_path=tmp_path / "pending_outreach.json",
        _llm=llm,
    )
    assert len(result.records) == 3
    assert result.records[0].outcome == "sent"
    assert result.records[1].outcome == "send_failed"
    assert result.records[1].send_result.error == "simulated SMTP failure"
    assert result.records[2].outcome == "sent"


# --- 6. pending_outreach.json round-trip + 16. dead-lead cross-check ------


def test_phase1_phase2_state_roundtrip(monkeypatch, tmp_path):
    _set_fake_smtp_env(monkeypatch)
    calls = []

    def _fake(*, to_address, subject, body, dry_run, creds=None, _sender=None):
        calls.append(to_address)
        return _schemas.SendEmailResult(sent=True, error=None, dry_run=False)

    monkeypatch.setattr(_agent, "_smtp_send_email", _fake)
    state_path = tmp_path / "pending_outreach.json"

    phrase = "Series B"
    body = _compliant_body(phrase)
    llm = SequenceLLM([_qualify_json(fit_score=90), _draft_json("Hi", body, phrase)])
    run_outreach_batch(
        [_lead(lead_id="acme1")],
        sender=_sender(),
        product_description="x",
        icp_description="y",
        provider="openai",
        model="test-model",
        dry_run=False,
        send_delay_seconds=0,
        state_path=state_path,
        _llm=llm,
    )
    assert state_path.exists()

    reply_llm = SequenceLLM([_classify_json("interested")])
    followup_body = _compliant_body("your interest")
    reply_llm._responses.append(_draft_json("Re: Hi", followup_body, "your interest"))
    outcome = process_reply(
        "acme1", "tell me more", provider="openai", model="test-model", state_path=state_path, _llm=reply_llm
    )
    assert outcome.classification.intent == "interested"
    assert outcome.final_status in ("followup_drafted", "followup_sent")


def test_dead_lead_skipped_in_future_batch(monkeypatch, tmp_path):
    _set_fake_smtp_env(monkeypatch)

    def _fake(*, to_address, subject, body, dry_run, creds=None, _sender=None):
        return _schemas.SendEmailResult(sent=True, error=None, dry_run=False)

    monkeypatch.setattr(_agent, "_smtp_send_email", _fake)
    state_path = tmp_path / "pending_outreach.json"

    phrase = "Series B"
    body = _compliant_body(phrase)
    llm = SequenceLLM([_qualify_json(fit_score=90), _draft_json("Hi", body, phrase)])
    run_outreach_batch(
        [_lead(lead_id="acme1", email="jane@acme.com")],
        sender=_sender(),
        product_description="x",
        icp_description="y",
        provider="openai",
        model="test-model",
        dry_run=False,
        send_delay_seconds=0,
        state_path=state_path,
        _llm=llm,
    )
    reply_llm = SequenceLLM([_classify_json("unsubscribe")])
    process_reply(
        "acme1", "stop emailing me", provider="openai", model="test-model", state_path=state_path, _llm=reply_llm
    )

    # Same email, DIFFERENT lead_id, a LATER batch -- must be skipped,
    # no wasted LLM call, no send.
    llm2 = SequenceLLM([])  # must never be called
    result2 = run_outreach_batch(
        [_lead(lead_id="totally-different-id", email="jane@acme.com")],
        sender=_sender(),
        product_description="x",
        icp_description="y",
        provider="openai",
        model="test-model",
        dry_run=False,
        send_delay_seconds=0,
        state_path=state_path,
        _llm=llm2,
    )
    assert result2.records[0].outcome == "skipped_previously_unsubscribed"
    assert len(llm2.calls) == 0


# --- 7. LeadNotFound -------------------------------------------------------


def test_process_reply_unknown_lead_id_raises(tmp_path):
    state_path = tmp_path / "pending_outreach.json"
    with pytest.raises(LeadNotFound, match="no-such-lead"):
        process_reply(
            "no-such-lead", "hello", provider="openai", model="test-model", state_path=state_path, _llm=SequenceLLM([])
        )


# --- 8. Already-dead lead reprocessed --------------------------------------


def test_already_dead_lead_reprocessed_is_noop(monkeypatch, tmp_path):
    _set_fake_smtp_env(monkeypatch)

    def _fake(*, to_address, subject, body, dry_run, creds=None, _sender=None):
        return _schemas.SendEmailResult(sent=True, error=None, dry_run=False)

    monkeypatch.setattr(_agent, "_smtp_send_email", _fake)
    state_path = tmp_path / "pending_outreach.json"
    phrase = "Series B"
    body = _compliant_body(phrase)
    llm = SequenceLLM([_qualify_json(fit_score=90), _draft_json("Hi", body, phrase)])
    run_outreach_batch(
        [_lead(lead_id="acme1")],
        sender=_sender(),
        product_description="x",
        icp_description="y",
        provider="openai",
        model="test-model",
        dry_run=False,
        send_delay_seconds=0,
        state_path=state_path,
        _llm=llm,
    )
    reply_llm = SequenceLLM([_classify_json("unsubscribe")])
    process_reply(
        "acme1", "stop", provider="openai", model="test-model", state_path=state_path, _llm=reply_llm
    )

    # Second reply from the same (now-dead) lead: must re-affirm dead,
    # no raise, no LLM call, no send.
    second_reply_llm = SequenceLLM([])  # must never be called
    outcome = process_reply(
        "acme1",
        "actually wait, I'm interested now",
        provider="openai",
        model="test-model",
        state_path=state_path,
        _llm=second_reply_llm,
    )
    assert outcome.final_status == "dead"
    assert len(second_reply_llm.calls) == 0


# --- 9. One test per fixed reply intent ------------------------------------


def _send_one_sent_lead(monkeypatch, tmp_path, lead_id="acme1"):
    """Phase 1 always calls this with dry_run=False (a real send);
    phase 2's follow-up calls default to dry_run=True (same convention
    as phase 1's own CLI default) -- the fake must respect `dry_run`
    like the real send_email() does, not hardcode sent=True regardless."""
    _set_fake_smtp_env(monkeypatch)

    def _fake(*, to_address, subject, body, dry_run, creds=None, _sender=None):
        if dry_run:
            return _schemas.SendEmailResult(sent=False, error=None, dry_run=True)
        return _schemas.SendEmailResult(sent=True, error=None, dry_run=False)

    monkeypatch.setattr(_agent, "_smtp_send_email", _fake)
    state_path = tmp_path / "pending_outreach.json"
    phrase = "Series B"
    body = _compliant_body(phrase)
    llm = SequenceLLM([_qualify_json(fit_score=90), _draft_json("Hi", body, phrase)])
    run_outreach_batch(
        [_lead(lead_id=lead_id)],
        sender=_sender(),
        product_description="x",
        icp_description="y",
        provider="openai",
        model="test-model",
        dry_run=False,
        send_delay_seconds=0,
        state_path=state_path,
        _llm=llm,
    )
    return state_path


@pytest.mark.parametrize(
    "intent,expected_status",
    [
        ("interested", "followup_drafted"),
        ("objection", "followup_drafted"),
        ("out_of_office", "snoozed"),
        ("not_interested", "dead"),
        ("unsubscribe", "dead"),
    ],
)
def test_reply_intent_round_trips(monkeypatch, tmp_path, intent, expected_status):
    state_path = _send_one_sent_lead(monkeypatch, tmp_path)
    responses = [_classify_json(intent)]
    if intent in ("interested", "objection"):
        followup_body = _compliant_body("your interest")
        responses.append(_draft_json("Re: Hi", followup_body, "your interest"))
    reply_llm = SequenceLLM(responses)
    outcome = process_reply(
        "acme1", "some reply text", provider="openai", model="test-model", state_path=state_path, _llm=reply_llm
    )
    assert outcome.classification.intent == intent
    assert outcome.final_status == expected_status


# --- 10. THE critical safety-override test ---------------------------------


@pytest.mark.parametrize("bad_intent", ["not_interested", "unsubscribe"])
def test_safety_override_ignores_model_suggested_action(monkeypatch, tmp_path, bad_intent):
    state_path = _send_one_sent_lead(monkeypatch, tmp_path)
    reply_llm = SequenceLLM(
        [_classify_json(bad_intent, suggested_next_action="draft a follow-up email anyway")]
    )
    outcome = process_reply(
        "acme1", "no thanks", provider="openai", model="test-model", state_path=state_path, _llm=reply_llm
    )
    assert outcome.final_status == "dead"
    assert outcome.override_applied is True
    assert outcome.followup_email is None
    assert outcome.send_result is None
    assert len(reply_llm.calls) == 1  # classify only -- no draft_followup call ever made


# --- 11. state_store round-trip: real local file I/O -----------------------


def test_state_store_roundtrip_real_file_io(tmp_path):
    path = tmp_path / "pending_outreach.json"
    assert _state_store.load_pending_outreach(path) == {}

    state = _state_store.upsert_lead_entry(
        {},
        lead={"lead_id": "x1", "contact_email": "a@b.com"},
        qualification={"fit_score": 90},
        drafted_email={"subject": "hi"},
        sender={"sender_name": "Bob"},
    )
    _state_store.save_pending_outreach(path, state)
    reloaded = _state_store.load_pending_outreach(path)
    assert reloaded["x1"]["lead"]["contact_email"] == "a@b.com"
    assert reloaded["x1"]["status"] == "awaiting_reply"


def test_state_store_corrupt_file_raises(tmp_path):
    path = tmp_path / "pending_outreach.json"
    path.write_text("not valid json {{{", encoding="utf-8")
    with pytest.raises(StateStoreError):
        _state_store.load_pending_outreach(path)


def test_state_store_missing_file_returns_empty_dict(tmp_path):
    path = tmp_path / "does_not_exist.json"
    assert _state_store.load_pending_outreach(path) == {}


# --- 12. SmtpNotConfigured pre-flight (standalone) --------------------------


def test_send_email_raises_smtp_not_configured(monkeypatch):
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_APP_PASSWORD", raising=False)
    with pytest.raises(SmtpNotConfigured, match="App Password"):
        _smtp_client.send_email(to_address="a@b.com", subject="hi", body="hi", dry_run=False)


# --- 13. _translate_api_error 6-branch --------------------------------------


def test_translate_api_error_rate_limit_by_status():
    class E(Exception):
        status_code = 429

    result = _translate_api_error(E("body"))
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


def test_translate_api_error_ollama_connection():
    class ConnectError(Exception):
        pass

    result = _translate_api_error(ConnectError("failed"))
    assert "ollama" in result.message.lower()


def test_translate_api_error_unknown_preserves_original():
    result = _translate_api_error(ValueError("something weird happened"))
    assert "ValueError" in result.message
    assert "something weird happened" in result.message


def test_translate_api_error_annotates_lead_id():
    result = _translate_api_error(ValueError("x"), lead_id="acme1")
    assert result.partial.lead_id == "acme1"


# --- 14. CLI default-dry-run regression guard -------------------------------


def test_cli_default_is_dry_run(monkeypatch, tmp_path, capsys):
    leads_path = tmp_path / "leads.json"
    leads_path.write_text(json.dumps([_lead().model_dump()]), encoding="utf-8")
    sender_path = tmp_path / "sender.json"
    sender_path.write_text(json.dumps(_sender().model_dump()), encoding="utf-8")
    product_path = tmp_path / "product.md"
    product_path.write_text("A dev tool", encoding="utf-8")
    icp_path = tmp_path / "icp.md"
    icp_path.write_text("Startups", encoding="utf-8")

    captured = {}

    def _fake_run_outreach_batch(leads, *, dry_run, **kwargs):
        captured["dry_run"] = dry_run
        return OutreachBatchResult(records=[], run_meta={})

    monkeypatch.setattr(_agent, "run_outreach_batch", _fake_run_outreach_batch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sales-outreach",
            "send",
            "--leads",
            str(leads_path),
            "--sender",
            str(sender_path),
            "--product",
            str(product_path),
            "--icp",
            str(icp_path),
        ],
    )
    exit_code = _agent.main()
    assert exit_code == 0
    assert captured["dry_run"] is True  # --send was NOT passed -> dry_run must be True


def test_cli_send_flag_disables_dry_run(monkeypatch, tmp_path):
    leads_path = tmp_path / "leads.json"
    leads_path.write_text(json.dumps([_lead().model_dump()]), encoding="utf-8")
    sender_path = tmp_path / "sender.json"
    sender_path.write_text(json.dumps(_sender().model_dump()), encoding="utf-8")
    product_path = tmp_path / "product.md"
    product_path.write_text("A dev tool", encoding="utf-8")
    icp_path = tmp_path / "icp.md"
    icp_path.write_text("Startups", encoding="utf-8")

    captured = {}

    def _fake_run_outreach_batch(leads, *, dry_run, **kwargs):
        captured["dry_run"] = dry_run
        return OutreachBatchResult(records=[], run_meta={})

    monkeypatch.setattr(_agent, "run_outreach_batch", _fake_run_outreach_batch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sales-outreach",
            "send",
            "--leads",
            str(leads_path),
            "--sender",
            str(sender_path),
            "--product",
            str(product_path),
            "--icp",
            str(icp_path),
            "--send",
        ],
    )
    _agent.main()
    assert captured["dry_run"] is False


# --- 15. Draft missing opt-out/postal address triggers a retry -------------


def test_draft_missing_opt_out_line_triggers_retry(fake_smtp_module_level, tmp_path):
    phrase = "Series B"
    bad_body = f"Hi there, noting that {phrase}. Best, Bob."  # missing opt-out + postal address
    good_body = _compliant_body(phrase)
    llm = SequenceLLM(
        [
            _qualify_json(fit_score=90),
            _draft_json("Hi", bad_body, phrase),
            _draft_json("Hi", good_body, phrase),
        ]
    )
    result = run_outreach_batch(
        [_lead()],
        sender=_sender(),
        product_description="x",
        icp_description="y",
        provider="openai",
        model="test-model",
        dry_run=True,
        _llm=llm,
    )
    assert len(llm.calls) == 3  # qualify + 2 draft attempts
    assert _opt_out_line() in result.records[0].drafted_email.body


def test_draft_personalization_not_in_blurb_triggers_retry(fake_smtp_module_level):
    good_body = _compliant_body("Series B")
    llm = SequenceLLM(
        [
            _qualify_json(fit_score=90),
            _draft_json("Hi", "some body with a made up detail", "a fact that is not in the blurb at all"),
            _draft_json("Hi", good_body, "Series B"),
        ]
    )
    run_outreach_batch(
        [_lead()],
        sender=_sender(),
        product_description="x",
        icp_description="y",
        provider="openai",
        model="test-model",
        dry_run=True,
        _llm=llm,
    )
    assert len(llm.calls) == 3


# --- 17. Upfront credential pre-flight --------------------------------------


def test_smtp_not_configured_checked_before_any_lead_processed(monkeypatch):
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_APP_PASSWORD", raising=False)
    leads = [_lead(lead_id="l1"), _lead(lead_id="l2"), _lead(lead_id="l3")]
    llm = SequenceLLM([])  # must NEVER be called
    with pytest.raises(SmtpNotConfigured):
        run_outreach_batch(
            leads,
            sender=_sender(),
            product_description="x",
            icp_description="y",
            provider="openai",
            model="test-model",
            dry_run=False,
            _llm=llm,
        )
    assert len(llm.calls) == 0


# --- Pure helpers ----------------------------------------------------------


def test_route_after_qualify_qualified():
    qualification = _schemas.LeadQualification(lead_id="x", fit_score=90, reasoning="r", qualified=True)
    assert _route_after_qualify({"qualification": qualification}) == "draft_email"


def test_route_after_qualify_not_qualified():
    qualification = _schemas.LeadQualification(lead_id="x", fit_score=10, reasoning="r", qualified=False)
    assert _route_after_qualify({"qualification": qualification}) == "skip_record"


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


def test_normalize_for_substring_collapses_whitespace_and_case():
    assert _normalize_for_substring("A   B\nC") == "a b c"


def test_parse_json_object_strips_markdown_fence():
    assert _parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_object_returns_error_string_on_garbage():
    assert isinstance(_parse_json_object("not json at all"), str)
