"""Sales outreach agent -- agent #18 of real-world-agents.

Technique demonstrated: **a LangGraph node with a genuine external side
effect** (a real SMTP send) inside a qualify -> conditional-skip/draft
-> send pipeline, paired with a hand-rolled, cross-process JSON state
file (`state_store.py`) that lets a second phase run independently,
whenever a reply actually arrives -- a separate CLI invocation, days
later, not a continuation of the same process:

    START -> qualify -> conditional edge -> {draft_email, skip_record}
                           draft_email -> send_email_node -> record -> END
             skip_record -> record -> END

This is a *binary* router (qualified vs not) -- explicitly NOT #17's
N-way conditional fan-out and NOT #03's retry loop. The new LangGraph
capability this agent demonstrates is `send_email_node`'s body calling
`smtp_client.send_email(...)`: a node with a REAL external side effect.
No existing LangGraph agent in this catalog has one -- #03's
`execute_sql` writes to an in-memory SQLite database (gone when the
process exits), and #17's nodes are all pure LLM calls.

Phase 2 (`process_reply`) is a **plain function chain, not a second
LangGraph graph**: it has exactly one binary branch (dead vs. act-on-
it) with no parallelism, so a second graph would be pure ceremony
re-demonstrating what phase 1 already shows.

Why this technique for this use case: real outbound sales email has
two genuinely separate moments -- send now, and react to a reply
whenever it arrives, which could be minutes or weeks later -- that no
single-process agent loop can model. `pending_outreach.json` bridges
them. LangGraph's conditional edge cleanly expresses "don't send to
unqualified leads" without a second LLM call just for routing.

Real stakes, real action: this agent actually sends email via the
user's own SMTP account (dry-run by default; `--send` is the only way
real sending happens, never an env var). A mishandled unsubscribe/
not-interested reply is a real CAN-SPAM/GDPR concern, not a UX nit --
see the code-owned hard override below.

Real error handling (R5 in CONTRIBUTING.md's hard rules): four
concrete failure modes handled explicitly (the fourth is specific to
this agent's real side effect):
  1. Bad input (empty batch, empty product/ICP/sender fields, duplicate
     lead_id, malformed email address, empty reply text, unknown
     lead_id) -> OutreachError / LeadNotFound, raised before any LLM
     call or send attempt.
  2. A node's structured-output retry loop exhausts its attempts ->
     OutreachError with the last raw output + validation errors
     attached as .partial.
  3. Rate limit / auth / API failure -> translated to OutreachError via
     _translate_api_error, the same 6-branch shape as every other
     agent in this catalog, annotated with the failing lead_id.
  4. SMTP send failure (auth rejected, connection refused, timeout) ->
     smtp_client.send_email() NEVER raises for these -- it returns a
     structured SendEmailResult, consumed as ordinary graph state, not
     an exception. SmtpNotConfigured (missing credentials) is the one
     pre-flight-specific case: resolved ONCE, before the batch loop
     starts, so a misconfigured setup fails immediately instead of
     burning a qualify+draft LLM call on every lead first.

Non-negotiable code-owned safety rule (mirrors agent #12's two-tier
cap -- a deterministic rule layered on top of LLM judgment for
something too consequential to trust to the model alone): if a reply's
classified intent is "unsubscribe" or "not_interested", the CODE --
not the model -- unconditionally marks the lead dead and forbids
further contact, regardless of what the model's own
`suggested_next_action` says. And a lead already marked dead in a
*previous* batch can never be re-qualified or re-emailed by a *future*
`run_outreach_batch` call, even if re-included in a new leads.json --
checked by contact_email, not lead_id, since lead_id isn't guaranteed
stable across separately-prepared batches for the same person.

Provider + model are fully user-configurable: every real LLM call goes
through common.llm.get_llm() / resolve_model(), same env-var contract
as every other agent in the catalog.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict, TypeVar

from pydantic import BaseModel

# Dual-mode import (same rationale as every other agent in this
# catalog): `.schemas` resolves when this file is loaded as a submodule
# of the 18_sales_outreach package (how tests import it, via importlib
# since the dir name starts with a digit). The bare `schemas` (absolute)
# resolves when this file is run directly via `python -m agent` from
# inside the agent's own directory.
try:
    from . import state_store
    from .schemas import (
        DraftedEmail,
        Lead,
        LeadOutreachRecord,
        LeadQualification,
        OutreachBatchResult,
        ReplyClassification,
        ReplyOutcome,
        SendEmailResult,
        SenderProfile,
    )
    from .smtp_client import SmtpCredentials, SmtpNotConfigured, load_smtp_credentials
    from .smtp_client import send_email as _smtp_send_email
except ImportError:
    import state_store
    from schemas import (
        DraftedEmail,
        Lead,
        LeadOutreachRecord,
        LeadQualification,
        OutreachBatchResult,
        ReplyClassification,
        ReplyOutcome,
        SendEmailResult,
        SenderProfile,
    )
    from smtp_client import SmtpCredentials, SmtpNotConfigured, load_smtp_credentials
    from smtp_client import send_email as _smtp_send_email

from common.llm import (
    LLM,
    OLLAMA_CONNECTION_HINT,
    get_llm,
    is_ollama_connection_error,
    resolve_model,
)

# --- Provider ---------------------------------------------------------------

SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini", "ollama")

# --- Defaults -----------------------------------------------------------

DEFAULT_QUALIFY_THRESHOLD = 60
DEFAULT_SEND_DELAY_SECONDS = 2.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_TOKENS_OUT = 2048
RETRY_BACKOFF_SECONDS = 1.0

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_DEFAULT_STATE_PATH = Path(__file__).parent / "pending_outreach.json"

_OPT_OUT_TEMPLATE = "{reply_to_note} to stop receiving emails like this."


def resolve_provider() -> str:
    """LLM_PROVIDER env var, defaulting to "openai". No provider is
    hardcoded."""
    import os

    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}."
        )
    return provider


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


# --- Error type --------------------------------------------------------


@dataclass
class OutreachAttempt:
    """What we got back when something failed partway through a batch
    or a structured-output retry loop. Attached to OutreachError so
    the caller can salvage prior work and surface the raw model output
    instead of losing everything silently."""

    stage: str = ""
    lead_id: str | None = None
    raw_text: str = ""
    validation_errors: list[str] = field(default_factory=list)
    completed_records: list[LeadOutreachRecord] = field(default_factory=list)


class OutreachError(Exception):
    """Raised on any user-facing failure (bad input, retry exhaustion,
    API failure). `message` is user-friendly; `partial` carries
    whatever salvageable state exists."""

    def __init__(self, message: str, partial: OutreachAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Graph state -------------------------------------------------------


class GraphState(TypedDict, total=False):
    """LangGraph state carried between nodes for ONE lead. No reducer
    fields needed -- this graph is invoked once per lead from an outer
    Python loop, never with parallel branches writing the same key
    (contrast agent #17's lens_findings, which genuinely needs
    Annotated[..., operator.add])."""

    lead: Lead
    product_description: str
    icp_description: str
    sender: SenderProfile
    opt_out_line: str
    qualification: LeadQualification | None
    drafted_email: DraftedEmail | None
    send_result: SendEmailResult | None
    outcome: str | None


# --- Public API: phase 1 ----------------------------------------------------


def run_outreach_batch(
    leads: list[Lead],
    *,
    sender: SenderProfile,
    product_description: str,
    icp_description: str,
    qualify_threshold: int = DEFAULT_QUALIFY_THRESHOLD,
    dry_run: bool = True,
    send_delay_seconds: float = DEFAULT_SEND_DELAY_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    provider: str | None = None,
    model: str | None = None,
    state_path: Path | None = None,
    _llm: LLM | None = None,  # test-injection escape hatch
) -> OutreachBatchResult:
    """Qualify each lead against the ICP, draft a personalized email
    for qualified leads, and send it via the user's own SMTP account
    (or just draft it, if dry_run=True -- the default).

    Args:
        leads: the batch to process. Must have unique lead_ids.
        sender: who the outreach is from -- required, never invented.
        product_description: what's being sold.
        icp_description: who the ideal customer is.
        qualify_threshold: minimum fit_score (0-100) to draft+send.
        dry_run: if True (the default), every lead's send is a no-op
            that never touches SMTP credentials or opens a socket.
            Only `dry_run=False` (the CLI's `--send` flag) sends for
            real.
        send_delay_seconds: pause after each REAL send (not after
            dry-run or skipped leads) -- paced sending looks less like
            abuse to a provider's own spam detection than a tight
            loop. Ignored under dry_run.
        state_path: where pending_outreach.json lives. Defaults to
            this agent's own directory.
        _llm: injected LLM for tests; production callers leave this
            None.

    Returns:
        A validated OutreachBatchResult.

    Raises:
        OutreachError: on any R5 failure mode.
        SmtpNotConfigured: if dry_run=False and SMTP credentials are
            missing -- raised before any lead is processed.
    """
    resolved_provider = (provider or resolve_provider()).lower()

    if resolved_provider == "mock":
        return _mock_batch_result(leads, dry_run=dry_run)

    # R5 case 1: bad input, before any LLM call or send attempt.
    if not leads:
        raise OutreachError("at least one lead is required.")
    if not product_description.strip():
        raise OutreachError("product_description must be non-empty.")
    if not icp_description.strip():
        raise OutreachError("icp_description must be non-empty.")
    if not sender.sender_name.strip() or not sender.sender_company.strip():
        raise OutreachError("sender.sender_name and sender.sender_company must be non-empty.")
    if not sender.sender_postal_address.strip():
        raise OutreachError(
            "sender.sender_postal_address must be non-empty -- every "
            "drafted email includes it (a CAN-SPAM requirement for "
            "commercial email)."
        )
    ids = [lead.lead_id for lead in leads]
    if len(set(ids)) != len(ids):
        seen: set[str] = set()
        dupes: set[str] = set()
        for lead_id in ids:
            if lead_id in seen:
                dupes.add(lead_id)
            seen.add(lead_id)
        raise OutreachError(
            f"duplicate lead_id(s) not allowed: {sorted(dupes)!r}. "
            "Give each lead a unique lead_id."
        )

    resolved_model = model or resolve_model(resolved_provider)

    # Credentials resolved ONCE, up front -- not lazily inside the
    # first lead's send node. A misconfigured SMTP setup fails
    # immediately, before wasting a qualify+draft LLM call on any lead.
    creds: SmtpCredentials | None = None
    if not dry_run:
        creds = load_smtp_credentials()  # raises SmtpNotConfigured

    llm = _llm if _llm is not None else get_llm(resolved_provider)

    state_path = state_path or _DEFAULT_STATE_PATH
    pending_state = state_store.load_pending_outreach(state_path)

    opt_out_line = _OPT_OUT_TEMPLATE.format(reply_to_note=sender.reply_to_note)
    graph = _build_phase1_graph(
        llm=llm,
        model=resolved_model,
        qualify_threshold=qualify_threshold,
        dry_run=dry_run,
        smtp_creds=creds,
        max_retries=max_retries,
    )

    start = time.perf_counter()
    records: list[LeadOutreachRecord] = []
    state_changed = False
    for lead in leads:
        try:
            # Dead-lead cross-check BEFORE qualifying -- no wasted LLM
            # call, no re-contact of someone who already unsubscribed
            # in a previous batch. Keyed by email, not lead_id.
            if state_store.find_dead_email(pending_state, lead.contact_email):
                records.append(
                    LeadOutreachRecord(
                        lead=lead,
                        qualification=LeadQualification(
                            lead_id=lead.lead_id,
                            fit_score=0,
                            reasoning="Skipped: this contact previously unsubscribed/opted out.",
                            qualified=False,
                        ),
                        drafted_email=None,
                        send_result=None,
                        outcome="skipped_previously_unsubscribed",
                    )
                )
                continue

            final = graph.invoke(
                {
                    "lead": lead,
                    "product_description": product_description,
                    "icp_description": icp_description,
                    "sender": sender,
                    "opt_out_line": opt_out_line,
                }
            )
            record = LeadOutreachRecord(
                lead=lead,
                qualification=final["qualification"],
                drafted_email=final.get("drafted_email"),
                send_result=final.get("send_result"),
                outcome=final["outcome"],
            )
            records.append(record)

            if record.outcome == "sent":
                pending_state = state_store.upsert_lead_entry(
                    pending_state,
                    lead=lead.model_dump(),
                    qualification=record.qualification.model_dump(),
                    drafted_email=record.drafted_email.model_dump(),
                    sender=sender.model_dump(),
                )
                state_changed = True
                if send_delay_seconds > 0:
                    time.sleep(send_delay_seconds)
        except OutreachError as exc:
            if exc.partial is not None:
                exc.partial.completed_records = list(records)
            raise
        except Exception as exc:  # R5 case 3: rate limit / API failure
            translated = _translate_api_error(exc, lead_id=lead.lead_id)
            if translated.partial is not None:
                translated.partial.completed_records = list(records)
            raise translated from exc

    # Only write if something actually changed -- a pure dry run (or a
    # batch where every lead was skipped) never mutates pending_state,
    # and writing an unchanged (often empty) file on every no-op dry
    # run would leave a stray file behind before a user has done
    # anything real.
    if state_changed:
        state_store.save_pending_outreach(state_path, pending_state)

    return OutreachBatchResult(
        records=records,
        run_meta={
            "provider": resolved_provider,
            "model": resolved_model,
            "dry_run": dry_run,
            "lead_count": len(leads),
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        },
    )


# --- The LangGraph phase-1 pipeline (pedagogical anchor) --------------------

_BaseModelT = TypeVar("_BaseModelT", bound=BaseModel)


def _route_after_qualify(state: GraphState) -> str:
    """Conditional-edge router: binary, qualified vs not. Pure function
    of state -- directly unit-testable without building the graph,
    same convention as #17's module-level _route_after_classify."""
    qualification = state["qualification"]
    return "draft_email" if qualification.qualified else "skip_record"


def _build_phase1_graph(
    *,
    llm: LLM,
    model: str,
    qualify_threshold: int,
    dry_run: bool,
    smtp_creds: SmtpCredentials | None,
    max_retries: int,
):
    """Build and compile the qualify -> conditional fan-out -> send ->
    record state graph for ONE lead. LLM + model + threshold +
    dry_run + smtp_creds are closed over via this factory, so nodes
    stay pure `state -> partial_state` functions -- tests inject a
    fake llm/sender by passing different values into this factory
    (indirectly, via run_outreach_batch's `_llm` parameter and a
    monkeypatched smtp_client).
    """
    # Lazy import: langgraph is a heavy dependency we don't want in
    # every mock-mode test run.
    from langgraph.graph import END, START, StateGraph

    def _qualify(state: GraphState) -> dict:
        prompt = _fill_qualify_prompt(
            lead=state["lead"],
            product_description=state["product_description"],
            icp_description=state["icp_description"],
        )
        result = _run_structured_loop(
            llm=llm,
            model=model,
            prompt=prompt,
            target_model=_QualifyOutput,
            max_retries=max_retries,
            stage="qualify",
            lead_id=state["lead"].lead_id,
        )
        # Recompute `qualified` from fit_score vs threshold -- never
        # trust the model's own boolean (mirrors #11's
        # _recommendation_for re-deriving `recommendation` from
        # overall_score rather than trusting the model's own field).
        qualified = result.fit_score >= qualify_threshold
        return {
            "qualification": LeadQualification(
                lead_id=state["lead"].lead_id,
                fit_score=result.fit_score,
                reasoning=result.reasoning,
                qualified=qualified,
            )
        }

    def _skip_record(state: GraphState) -> dict:
        return {"outcome": "skipped_not_qualified"}

    def _draft_email(state: GraphState) -> dict:
        lead = state["lead"]
        sender = state["sender"]
        opt_out_line = state["opt_out_line"]
        prompt = _fill_draft_prompt(
            lead=lead,
            qualification=state["qualification"],
            product_description=state["product_description"],
            sender=sender,
            opt_out_line=opt_out_line,
        )

        def _check_draft(draft: _DraftOutput) -> list[str]:
            errors: list[str] = []
            normalized_body = _normalize_for_substring(draft.body)
            # Both checks below use significant-word overlap, NOT a rigid
            # full-phrase substring match, against both the blurb AND the
            # body: natural personalized writing legitimately rewrites
            # pronouns/tense in BOTH personalization_note itself and the
            # body when working a third-person blurb detail into a
            # sentence addressed to the recipient ("their tooling team" ->
            # "your tooling team"), which a real model does and should do.
            # A rigid substring check against either side wrongly rejects
            # a well-written, genuinely-grounded paraphrase (caught live
            # against a real provider, not just reasoned about).
            note_words = _significant_words(draft.personalization_note)
            blurb_words = _significant_words(lead.blurb)
            body_words = _significant_words(draft.body)

            missing_from_blurb = note_words - blurb_words
            if note_words and len(missing_from_blurb) > max(1, len(note_words) // 3):
                errors.append(
                    f"personalization_note {draft.personalization_note!r} "
                    f"doesn't come from the lead's blurb (missing: "
                    f"{sorted(missing_from_blurb)!r}) -- it must be grounded "
                    "in a real detail from the blurb, not invented."
                )
            else:
                missing_from_body = note_words - body_words
                if note_words and len(missing_from_body) > max(1, len(note_words) // 3):
                    errors.append(
                        f"personalization_note {draft.personalization_note!r} "
                        f"doesn't show up as substance in the email body "
                        f"(missing: {sorted(missing_from_body)!r}) -- it's not "
                        "personalized if you note a detail but don't "
                        "actually use it in the email."
                    )
            if _normalize_for_substring(opt_out_line) not in normalized_body:
                errors.append(
                    f"The email body must include this exact opt-out line "
                    f"verbatim: {opt_out_line!r}"
                )
            if _normalize_for_substring(sender.sender_postal_address) not in normalized_body:
                errors.append(
                    f"The email body must include this exact postal address "
                    f"verbatim: {sender.sender_postal_address!r}"
                )
            return errors

        raw = _run_structured_loop(
            llm=llm,
            model=model,
            prompt=prompt,
            target_model=_DraftOutput,
            max_retries=max_retries,
            extra_validate=_check_draft,
            stage="draft",
            lead_id=lead.lead_id,
        )
        result = DraftedEmail(lead_id=lead.lead_id, **raw.model_dump())
        return {"drafted_email": result}

    def _send_email_node(state: GraphState) -> dict:
        drafted = state["drafted_email"]
        lead = state["lead"]
        result = _smtp_send_email(
            to_address=lead.contact_email,
            subject=drafted.subject,
            body=drafted.body,
            dry_run=dry_run,
            creds=smtp_creds,
        )
        if result.dry_run:
            outcome = "dry_run_drafted"
        elif result.sent:
            outcome = "sent"
        else:
            outcome = "send_failed"
        return {"send_result": result, "outcome": outcome}

    def _record(state: GraphState) -> dict:
        return {}  # terminal pass-through; run_outreach_batch builds the LeadOutreachRecord from final_state

    g: StateGraph = StateGraph(GraphState)
    g.add_node("qualify", _qualify)
    g.add_node("draft_email", _draft_email)
    g.add_node("send_email_node", _send_email_node)
    g.add_node("skip_record", _skip_record)
    g.add_node("record", _record)

    g.add_edge(START, "qualify")
    g.add_conditional_edges(
        "qualify",
        _route_after_qualify,
        {"draft_email": "draft_email", "skip_record": "skip_record"},
    )
    g.add_edge("draft_email", "send_email_node")
    g.add_edge("send_email_node", "record")
    g.add_edge("skip_record", "record")
    g.add_edge("record", END)
    return g.compile()


class _QualifyOutput(BaseModel):
    """Internal target for the qualify LLM call -- NOT schemas.py's
    LeadQualification, since `qualified` must always be recomputed in
    code from fit_score, never trusted as a raw model-returned
    boolean. Keeping this as a separate, smaller model makes that
    recomputation structurally unavoidable rather than a convention
    someone could accidentally skip."""

    fit_score: int
    reasoning: str


class _DraftOutput(BaseModel):
    """Internal target for a draft_email/draft_followup LLM call --
    NOT schemas.py's DraftedEmail directly, since the model has no way
    to know (and no reason to echo back) a lead_id it was never given
    a reason to care about. agent.py constructs the full DraftedEmail
    by adding lead_id after validation succeeds."""

    subject: str
    body: str
    personalization_note: str


class _ReplyClassificationOutput(BaseModel):
    """Internal target for the classify_reply LLM call -- NOT
    schemas.py's ReplyClassification directly, same lead_id rationale
    as _DraftOutput above."""

    intent: Literal["interested", "not_interested", "objection", "out_of_office", "unsubscribe"]
    reasoning: str
    suggested_next_action: str
    snooze_until: str | None = None


def _fill_qualify_prompt(*, lead: Lead, product_description: str, icp_description: str) -> str:
    template = _load_prompt("qualify.txt")
    return (
        template.replace("{icp_description}", icp_description)
        .replace("{product_description}", product_description)
        .replace("{company}", lead.company)
        .replace("{contact_name}", lead.contact_name)
        .replace("{contact_title}", lead.contact_title)
        .replace("{blurb}", lead.blurb)
    )


def _fill_draft_prompt(
    *,
    lead: Lead,
    qualification: LeadQualification,
    product_description: str,
    sender: SenderProfile,
    opt_out_line: str,
) -> str:
    template = _load_prompt("draft_email.txt")
    return (
        template.replace("{product_description}", product_description)
        .replace("{company}", lead.company)
        .replace("{contact_name}", lead.contact_name)
        .replace("{contact_title}", lead.contact_title)
        .replace("{blurb}", lead.blurb)
        .replace("{qualify_reasoning}", qualification.reasoning)
        .replace("{sender_name}", sender.sender_name)
        .replace("{sender_company}", sender.sender_company)
        .replace("{opt_out_line}", opt_out_line)
        .replace("{sender_postal_address}", sender.sender_postal_address)
    )


# --- Public API: phase 2 -----------------------------------------------


def process_reply(
    lead_id: str,
    reply_text: str,
    *,
    dry_run: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    provider: str | None = None,
    model: str | None = None,
    state_path: Path | None = None,
    _llm: LLM | None = None,
) -> ReplyOutcome:
    """Classify a reply to a previously-sent outreach email and decide
    the next action.

    Non-negotiable: if the classified intent is "unsubscribe" or
    "not_interested", this function marks the lead dead UNCONDITIONALLY
    -- the code, not the model, owns this decision, regardless of what
    the model's own `suggested_next_action` says.

    Raises:
        OutreachError: empty reply_text (R5 case 1).
        state_store.LeadNotFound: lead_id not in pending_outreach.json.
    """
    resolved_provider = (provider or resolve_provider()).lower()
    if resolved_provider == "mock":
        # Mock bypasses everything -- no file I/O, no LLM -- same
        # "short-circuit before touching any real resource" convention
        # as every other agent's mock path (e.g. #03's _mock_answer
        # never touches the real CSV either).
        return _mock_reply_outcome(lead_id, reply_text)

    if not reply_text.strip():
        raise OutreachError("reply_text must be non-empty.")

    state_path = state_path or _DEFAULT_STATE_PATH
    pending_state = state_store.load_pending_outreach(state_path)
    entry = state_store.get_lead_entry(pending_state, lead_id)  # raises LeadNotFound

    if entry.get("status") == "dead":
        # Already dead (prior unsubscribe/not_interested). Re-affirm,
        # no-op -- never re-classify, never resurrect, never send.
        return ReplyOutcome(
            lead_id=lead_id,
            classification=ReplyClassification(
                lead_id=lead_id,
                intent="unsubscribe",
                reasoning="This lead was already marked dead from a prior reply; not re-processed.",
                suggested_next_action="none -- lead is dead",
            ),
            final_status="dead",
            override_applied=False,
        )

    resolved_model = model or resolve_model(resolved_provider)
    llm = _llm if _llm is not None else get_llm(resolved_provider)

    try:
        classification = _classify_reply(
            llm=llm, model=resolved_model, lead_id=lead_id, reply_text=reply_text, entry=entry,
            max_retries=max_retries,
        )
    except Exception as exc:
        if isinstance(exc, OutreachError):
            raise
        raise _translate_api_error(exc, lead_id=lead_id) from exc

    # --- CODE-OWNED HARD RULE (non-negotiable; CAN-SPAM/GDPR-adjacent) ---
    # Mirrors agent #12's two-tier code-owned cap: the model's own
    # suggested_next_action is NEVER trusted for these two intents,
    # regardless of what it says.
    if classification.intent in ("unsubscribe", "not_interested"):
        pending_state = state_store.mark_lead_dead(pending_state, lead_id)
        state_store.save_pending_outreach(state_path, pending_state)
        return ReplyOutcome(
            lead_id=lead_id,
            classification=classification,
            final_status="dead",
            override_applied=True,
        )

    if classification.intent == "out_of_office":
        pending_state = state_store.mark_lead_snoozed(
            pending_state, lead_id, classification.snooze_until
        )
        state_store.save_pending_outreach(state_path, pending_state)
        return ReplyOutcome(
            lead_id=lead_id,
            classification=classification,
            final_status="snoozed",
            override_applied=False,
        )

    # interested / objection: draft a follow-up, optionally send it.
    sender = SenderProfile.model_validate(entry["sender"])
    opt_out_line = _OPT_OUT_TEMPLATE.format(reply_to_note=sender.reply_to_note)
    try:
        followup = _draft_followup(
            llm=llm, model=resolved_model, entry=entry, classification=classification,
            sender=sender, opt_out_line=opt_out_line, max_retries=max_retries,
        )
    except Exception as exc:
        if isinstance(exc, OutreachError):
            raise
        raise _translate_api_error(exc, lead_id=lead_id) from exc

    creds: SmtpCredentials | None = None
    if not dry_run:
        creds = load_smtp_credentials()
    send_result = _smtp_send_email(
        to_address=entry["lead"]["contact_email"],
        subject=followup.subject,
        body=followup.body,
        dry_run=dry_run,
        creds=creds,
    )
    final_status = "followup_sent" if send_result.sent else "followup_drafted"
    pending_state = state_store.mark_lead_followed_up(pending_state, lead_id)
    state_store.save_pending_outreach(state_path, pending_state)
    return ReplyOutcome(
        lead_id=lead_id,
        classification=classification,
        final_status=final_status,
        followup_email=followup,
        send_result=send_result,
        override_applied=False,
    )


def _classify_reply(
    *, llm: LLM, model: str, lead_id: str, reply_text: str, entry: dict, max_retries: int
) -> ReplyClassification:
    template = _load_prompt("classify_reply.txt")
    drafted = entry.get("drafted_email", {})
    prompt = (
        template.replace("{reply_text}", reply_text)
        .replace("{original_subject}", drafted.get("subject", ""))
        .replace("{original_body}", drafted.get("body", ""))
    )
    raw = _run_structured_loop(
        llm=llm,
        model=model,
        prompt=prompt,
        target_model=_ReplyClassificationOutput,
        max_retries=max_retries,
        stage="classify_reply",
        lead_id=lead_id,
    )
    return ReplyClassification(lead_id=lead_id, **raw.model_dump())


def _draft_followup(
    *,
    llm: LLM,
    model: str,
    entry: dict,
    classification: ReplyClassification,
    sender: SenderProfile,
    opt_out_line: str,
    max_retries: int,
) -> DraftedEmail:
    template = _load_prompt("draft_followup.txt")
    lead = entry["lead"]
    prompt = (
        template.replace("{company}", lead["company"])
        .replace("{contact_name}", lead["contact_name"])
        .replace("{intent}", classification.intent)
        .replace("{classification_reasoning}", classification.reasoning)
        .replace("{sender_name}", sender.sender_name)
        .replace("{sender_company}", sender.sender_company)
        .replace("{opt_out_line}", opt_out_line)
        .replace("{sender_postal_address}", sender.sender_postal_address)
    )

    def _check_followup(draft: _DraftOutput) -> list[str]:
        normalized_body = _normalize_for_substring(draft.body)
        errors: list[str] = []
        if _normalize_for_substring(opt_out_line) not in normalized_body:
            errors.append(f"The follow-up body must include this exact opt-out line verbatim: {opt_out_line!r}")
        if _normalize_for_substring(sender.sender_postal_address) not in normalized_body:
            errors.append(
                f"The follow-up body must include this exact postal address verbatim: {sender.sender_postal_address!r}"
            )
        return errors

    raw = _run_structured_loop(
        llm=llm,
        model=model,
        prompt=prompt,
        target_model=_DraftOutput,
        max_retries=max_retries,
        extra_validate=_check_followup,
        stage="draft_followup",
        lead_id=classification.lead_id,
    )
    return DraftedEmail(lead_id=classification.lead_id, **raw.model_dump())


# --- The hand-rolled JSON-validate-retry loop -------------------------------
#
# Generalizes agent #02's _run_review_loop (reused in shape by agent
# #17 too) into one helper used at every LLM call site in this agent.


def _run_structured_loop(
    *,
    llm: LLM,
    model: str,
    prompt: str,
    target_model: type[_BaseModelT],
    max_retries: int,
    stage: str,
    lead_id: str | None,
    extra_validate: Callable[[_BaseModelT], list[str]] | None = None,
) -> _BaseModelT:
    """Prompt -> parse JSON -> Pydantic-validate against target_model
    -> optional extra_validate() for cross-field invariants Pydantic
    alone can't enforce -> retry with feedback on any failure, up to
    max_retries.

    On exhaustion, raises OutreachError with the last raw output
    attached as `partial` so the caller can surface it.
    """
    last_raw = ""
    last_errors: list[str] = []
    retry_feedback: str | None = None

    for attempt in range(1, max_retries + 1):
        attempt_prompt = prompt
        if retry_feedback:
            attempt_prompt = f"{retry_feedback}\n\n{prompt}"

        response = llm.complete(
            prompt=attempt_prompt,
            model=model,
            temperature=0.0,
            max_tokens=DEFAULT_MAX_TOKENS_OUT,
        )
        last_raw = response.text

        parsed = _parse_json_object(last_raw)
        if isinstance(parsed, str):  # error message
            last_errors = [parsed]
            retry_feedback = f"Your previous response was not valid JSON: {parsed}"
            if attempt < max_retries:
                _sleep_backoff(attempt)
            continue

        try:
            instance = target_model.model_validate(parsed)
        except Exception as exc:
            last_errors = [f"schema validation failed: {exc}"]
            retry_feedback = (
                "Your previous response was valid JSON but failed these "
                f"checks:\n- {last_errors[0]}\nFix ONLY these issues; keep "
                "everything else the same."
            )
            if attempt < max_retries:
                _sleep_backoff(attempt)
            continue

        errors = extra_validate(instance) if extra_validate else []
        if not errors:
            return instance

        last_errors = errors
        retry_feedback = (
            "Your previous response was valid JSON but failed these checks:\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\nFix ONLY these issues; keep everything else the same."
        )
        if attempt < max_retries:
            _sleep_backoff(attempt)

    raise OutreachError(
        f"Model output failed validation after {max_retries} attempts "
        f"(stage={stage!r}). The last raw response is attached below.",
        partial=OutreachAttempt(
            stage=stage, lead_id=lead_id, raw_text=last_raw, validation_errors=last_errors
        ),
    )


def _sleep_backoff(attempt: int) -> None:
    """Linear backoff between retries. Kept tiny so tests aren't slow --
    this loop only exists for validation retries, not transient-API-
    error retries (the SDK handles those)."""
    time.sleep(RETRY_BACKOFF_SECONDS * attempt)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_object(raw_text: str) -> dict | str:
    """Extract and parse the first top-level JSON object from
    `raw_text`. Returns the parsed dict on success, or an error string
    on failure. Copied in shape from agent #02/#17's
    _parse_json_object: handles markdown code fences and leading/
    trailing prose."""
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            stripped = (
                "\n".join(lines[1:-1])
                if lines[-1].strip().startswith("```")
                else "\n".join(lines[1:])
            )
    match = _JSON_OBJECT_RE.search(stripped)
    if not match:
        return "no JSON object found in response"
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return f"JSON parse error: {exc.msg} at line {exc.lineno} col {exc.colno}"
    if not isinstance(parsed, dict):
        return f"expected a JSON object, got {type(parsed).__name__}"
    return parsed


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_substring(s: str) -> str:
    """Collapse runs of whitespace to single spaces, lowercase.
    Copied in shape from agent #02/#17's _normalize_for_substring
    (case-folding added here since email drafting is more prone to
    harmless case drift than the contract-excerpt use case those
    agents check)."""
    return _WHITESPACE_RE.sub(" ", s).strip().lower()


_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a", "an", "the", "their", "your", "our", "his", "her", "its", "my",
    "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "and", "or", "for", "on", "at", "with", "as", "by",
    "this", "that", "these", "those", "it", "they", "you", "we", "he", "she",
}


def _significant_words(phrase: str) -> set[str]:
    """Lowercased content words, with short stopwords/pronouns excluded.
    Used to check a personalization_note shows up as SUBSTANCE in the
    email body without requiring a rigid full-phrase match -- natural
    personalized writing legitimately rewrites pronouns/tense when
    working a third-person blurb detail into a sentence addressed to
    the recipient ('their tooling team' -> 'your tooling team'), which
    a strict substring check would wrongly reject."""
    words = _WORD_RE.findall(phrase.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


# --- Error translation (R5 case 3) ------------------------------------------


def _translate_api_error(exc: Exception, *, lead_id: str | None = None) -> OutreachError:
    """Turn an OpenAI/Anthropic/Gemini/Ollama exception into a
    user-facing OutreachError. Same 6-branch priority order as every
    other agent in this catalog (class-name -> status-code ->
    message-string -> Ollama-connection-hint -> generic)."""
    exc_class_name = type(exc).__name__.lower()
    message_lower = str(exc).lower()
    status = getattr(exc, "status_code", None)

    if "ratelimiterror" in exc_class_name:
        return _rate_limit_error(lead_id)
    if "authenticationerror" in exc_class_name or "apikeyerror" in exc_class_name:
        return _auth_error(lead_id)
    if status == 429:
        return _rate_limit_error(lead_id)
    if status == 401:
        return _auth_error(lead_id)
    if "rate limit" in message_lower or "overloaded" in message_lower:
        return _rate_limit_error(lead_id)
    if "authentication" in message_lower or "api key" in message_lower:
        return _auth_error(lead_id)
    if is_ollama_connection_error(exc):
        return OutreachError(OLLAMA_CONNECTION_HINT, partial=OutreachAttempt(lead_id=lead_id))

    return OutreachError(
        f"Sales outreach failed: {type(exc).__name__}: {exc}. "
        "This is an unexpected error -- check the agent logs.",
        partial=OutreachAttempt(lead_id=lead_id),
    )


def _rate_limit_error(lead_id: str | None) -> OutreachError:
    return OutreachError(
        "The service is temporarily rate-limited or overloaded. "
        "Wait a minute and try again.",
        partial=OutreachAttempt(lead_id=lead_id),
    )


def _auth_error(lead_id: str | None) -> OutreachError:
    return OutreachError(
        "API authentication failed. Check that your LLM_PROVIDER matches "
        "the API key you've set in .env (OPENAI_API_KEY / "
        "ANTHROPIC_API_KEY / GEMINI_API_KEY).",
        partial=OutreachAttempt(lead_id=lead_id),
    )


# --- Mock mode ---------------------------------------------------------


def _mock_batch_result(leads: list[Lead], *, dry_run: bool) -> OutreachBatchResult:
    """Deterministic canned result for smoke tests and CI
    (LLM_PROVIDER=mock). No graph, no LLM, no SMTP."""
    records = []
    for i, lead in enumerate(leads):
        qualified = i % 2 == 0  # deterministic, varied shape
        qualification = LeadQualification(
            lead_id=lead.lead_id,
            fit_score=80 if qualified else 20,
            reasoning="Mock qualification reasoning.",
            qualified=qualified,
        )
        if not qualified:
            records.append(
                LeadOutreachRecord(
                    lead=lead,
                    qualification=qualification,
                    drafted_email=None,
                    send_result=None,
                    outcome="skipped_not_qualified",
                )
            )
            continue
        drafted = DraftedEmail(
            lead_id=lead.lead_id,
            subject=f"Mock subject for {lead.company}",
            body=f"Mock body referencing: {lead.blurb[:40]}. Reply STOP to unsubscribe. 123 Mock St.",
            personalization_note=lead.blurb[:20],
        )
        send_result = SendEmailResult(sent=not dry_run, error=None, dry_run=dry_run)
        records.append(
            LeadOutreachRecord(
                lead=lead,
                qualification=qualification,
                drafted_email=drafted,
                send_result=send_result,
                outcome="dry_run_drafted" if dry_run else "sent",
            )
        )
    return OutreachBatchResult(
        records=records,
        run_meta={"provider": "mock", "model": "mock", "dry_run": dry_run, "lead_count": len(leads)},
    )


def _mock_reply_outcome(lead_id: str, reply_text: str) -> ReplyOutcome:
    """Deterministic canned reply outcome for mock mode."""
    classification = ReplyClassification(
        lead_id=lead_id,
        intent="interested",
        reasoning="Mock classification reasoning.",
        suggested_next_action="Draft a follow-up.",
    )
    followup = DraftedEmail(
        lead_id=lead_id,
        subject="Mock follow-up subject",
        body=f"Mock follow-up referencing reply: {reply_text[:40]}. Reply STOP to unsubscribe. 123 Mock St.",
        personalization_note=reply_text[:20],
    )
    return ReplyOutcome(
        lead_id=lead_id,
        classification=classification,
        final_status="followup_drafted",
        followup_email=followup,
        send_result=SendEmailResult(sent=False, error=None, dry_run=True),
        override_applied=False,
    )


# --- CLI entry point (uv run python -m agent) -------------------------------


def _load_leads(path: Path) -> list[Lead]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Lead.model_validate(item) for item in data]


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="sales-outreach",
        description="Qualify leads, draft personalized outreach, and send it -- "
        "or process a reply to a previously-sent email.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    send_parser = subparsers.add_parser("send", help="Phase 1: qualify + draft + send a batch of leads.")
    send_parser.add_argument("--leads", required=True, help="Path to a leads JSON file.")
    send_parser.add_argument("--sender", required=True, help="Path to a sender profile JSON file.")
    send_parser.add_argument("--product", required=True, help="Path to a product description text file.")
    send_parser.add_argument("--icp", required=True, help="Path to an ICP description text file.")
    send_parser.add_argument("--threshold", type=int, default=DEFAULT_QUALIFY_THRESHOLD)
    send_parser.add_argument(
        "--send",
        action="store_true",
        help="Actually send emails via SMTP. Without this flag, every send is a dry run (the default).",
    )
    send_parser.add_argument("--send-delay-seconds", type=float, default=DEFAULT_SEND_DELAY_SECONDS)
    send_parser.add_argument("--provider", choices=[*SUPPORTED_PROVIDERS, "mock"])
    send_parser.add_argument("--model")

    reply_parser = subparsers.add_parser("reply", help="Phase 2: process a reply to a previously-sent email.")
    reply_parser.add_argument("--lead-id", required=True)
    reply_parser.add_argument("--reply-file", required=True, help="Path to a text file containing the reply.")
    reply_parser.add_argument(
        "--send",
        action="store_true",
        help="Actually send the follow-up via SMTP. Without this flag, it's a dry run (the default).",
    )
    reply_parser.add_argument("--provider", choices=[*SUPPORTED_PROVIDERS, "mock"])
    reply_parser.add_argument("--model")

    args = parser.parse_args()

    try:
        if args.command == "send":
            leads = _load_leads(Path(args.leads))
            sender = SenderProfile.model_validate(
                json.loads(Path(args.sender).read_text(encoding="utf-8"))
            )
            product_description = Path(args.product).read_text(encoding="utf-8")
            icp_description = Path(args.icp).read_text(encoding="utf-8")
            result = run_outreach_batch(
                leads,
                sender=sender,
                product_description=product_description,
                icp_description=icp_description,
                qualify_threshold=args.threshold,
                dry_run=not args.send,
                send_delay_seconds=args.send_delay_seconds,
                provider=args.provider,
                model=args.model,
            )
            print(result.model_dump_json(indent=2))
            return 0

        reply_text = Path(args.reply_file).read_text(encoding="utf-8")
        outcome = process_reply(
            args.lead_id,
            reply_text,
            dry_run=not args.send,
            provider=args.provider,
            model=args.model,
        )
        print(outcome.model_dump_json(indent=2))
        return 0
    except OutreachError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.partial is not None and exc.partial.raw_text:
            print("---- last raw output ----", file=sys.stderr)
            print(exc.partial.raw_text, file=sys.stderr)
            for err in exc.partial.validation_errors:
                print(f"  - {err}", file=sys.stderr)
        return 1
    except (SmtpNotConfigured, state_store.LeadNotFound, state_store.StateStoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
