"""Email-triage agent -- agent #06 of real-world-agents.

Technique demonstrated: **type-safe single-agent + dependency
injection via [PydanticAI](https://ai.pydantic.dev/)**. The
PydanticAI Agent is fully typed:

    Agent[TriageDeps, EmailTriage]
    #     ^^^^^^^^^^   ^^^^^^^^^^^
    #     injected     structured output
    #     dependencies (validated Pydantic)

Tools registered via `@agent.tool` receive `RunContext[TriageDeps]`
as their first argument -- typed access to user context
(known_contacts, important_domains) that lets the model ground its
decisions in real user state rather than fabrications. If a tool
signature drifts, mypy/pyright catches it before runtime; if the
Deps or Output types change, every tool call site knows.

Distinct from agents #01-#05:
  #01: single-call structured extraction (Instructor)
  #02: hand-rolled JSON-validate-retry loop (no framework)
  #03: LangGraph state machine
  #04: OpenAI Agents SDK ReAct pattern
  #05: multi-agent Crew (CrewAI Sequential Process)
  #06: type-safe single-agent + typed deps (this pattern)

Why this technique for this use case: email triage needs (a) a
strict output schema (EmailTriage's Category/Priority/Action
Literals prevent invented labels that break downstream routing),
(b) tools that need typed context (is this sender known? is this
domain important?), (c) a single-turn workflow (one email in ->
one triage out, no branching or state to track). PydanticAI is
designed for exactly this shape.

Real error handling (R5): three concrete failure modes:
  1. Malformed .eml / can't parse -- Python's `email` module raises
     -> EmailTriageError with the parser exception preserved
  2. Empty body -- .eml parses but has no text content -> refuse
     with a clear "can't triage empty email body" message
  3. Rate limit / auth / API failure -- _translate_api_error with
     the same 6-branch priority order as #02-#05. Same S2 no-auto-
     retry decision.

Provider strategy: OpenAI default via PydanticAI's native provider
strings (`"openai:gpt-4.1-mini"`). Anthropic/Gemini swap is a
one-line change: replace the model string with `"anthropic:claude-
sonnet-5"` or `"gemini:gemini-2.5-flash"`. No extra dep needed.
Documented in the README.

Mock mode: `LLM_PROVIDER=mock` bypasses PydanticAI entirely and
returns a deterministic canned EmailTriage. Under mock, pydantic-ai
never imports (lazy in real-provider paths only) -- so tests +
schema validation run cleanly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from email import message_from_bytes
from email.message import Message
from pathlib import Path

# Dual-mode import (same rationale as #01-05).
try:
    from .schemas import EmailTriage, TriageDeps
except ImportError:
    from schemas import EmailTriage, TriageDeps

from common.llm import resolve_model

# --- Provider + constants --------------------------------------------------

SUPPORTED_PROVIDERS = ("openai",)

DEFAULT_RETRIES = 2  # PydanticAI Agent-level retry cap on validation errors
MIN_BODY_CHARS = 20  # below this, R5 case 2 fires (empty-body)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"

_WHITESPACE_RE = re.compile(r"\s+")


def resolve_provider() -> str:
    """LLM_PROVIDER env var, default "openai". Only "openai" and
    "mock" supported in v1. Multi-provider via PydanticAI's native
    provider strings is a one-line swap documented in the README."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}. "
            "PydanticAI supports Anthropic/Gemini via its native provider "
            "strings -- swap the `llm=` param on the Agent (see README)."
        )
    return provider


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# --- Error type ------------------------------------------------------------


@dataclass
class TriageAttempt:
    """Partial state from a failed triage -- attached to
    EmailTriageError so the UI can surface what went wrong."""

    stage: str = ""  # 'parse' / 'empty_body' / 'agent_run'
    raw_output: str = ""


class EmailTriageError(Exception):
    """Raised on any user-facing triage failure."""

    def __init__(self, message: str, partial: TriageAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- .eml parsing (pure stdlib) --------------------------------------------


@dataclass
class ParsedEmail:
    """Normalized representation of a parsed .eml file. Fields are
    what the agent's prompt template needs, not the full RFC 2822
    surface area."""

    from_address: str = ""
    to_addresses: list[str] = field(default_factory=list)
    subject: str = ""
    body: str = ""
    date: str = ""


def _parse_eml(eml_bytes: bytes) -> ParsedEmail:
    """Parse .eml bytes into a ParsedEmail. Handles the two body
    shapes real emails come in: single-part text/plain, and
    multipart/alternative with text/plain + text/html (prefers
    text/plain).

    Raises EmailTriageError on parser failure (R5 case 1)."""
    try:
        msg: Message = message_from_bytes(eml_bytes)
    except Exception as exc:
        raise EmailTriageError(
            f"Couldn't parse input as .eml: {type(exc).__name__}: {exc}. "
            "Check that the file is a valid RFC 2822 email.",
            partial=TriageAttempt(stage="parse"),
        ) from exc

    return ParsedEmail(
        from_address=str(msg.get("From", "") or ""),
        to_addresses=[
            a.strip() for a in str(msg.get("To", "") or "").split(",") if a.strip()
        ],
        subject=str(msg.get("Subject", "") or ""),
        body=_extract_body(msg),
        date=str(msg.get("Date", "") or ""),
    )


def _extract_body(msg: Message) -> str:
    """Walk multipart tree preferring text/plain. Falls back to
    text/html stripped of tags if no plain-text part exists."""
    plain_parts: list[str] = []
    html_parts: list[str] = []
    for part in msg.walk():
        ctype = part.get_content_type()
        if ctype == "text/plain":
            payload = part.get_payload(decode=True)
            if payload:
                plain_parts.append(_decode_payload(payload, part.get_content_charset()))
        elif ctype == "text/html":
            payload = part.get_payload(decode=True)
            if payload:
                html_parts.append(_decode_payload(payload, part.get_content_charset()))
    if plain_parts:
        return "\n".join(plain_parts).strip()
    if html_parts:
        # Crude HTML strip -- good enough for triage. A real integration
        # would use html2text or bs4.
        raw = "\n".join(html_parts)
        return re.sub(r"<[^>]+>", " ", raw).strip()
    return ""


def _decode_payload(payload: bytes, charset: str | None) -> str:
    return payload.decode(charset or "utf-8", errors="replace")


# --- R5 case 2: body-content gate ------------------------------------------


def _has_triage_material(body: str) -> bool:
    """Refuse triaging emails with essentially no body. MIN_BODY_CHARS
    is deliberately low (20) -- shorter than a real one-line reply
    would be, but catches truly-empty and header-only inputs."""
    return len(body.strip()) >= MIN_BODY_CHARS


# --- Tool implementations (pure Python; wrapped as @agent.tool inside
#     _build_agent so they can typed-access RunContext[TriageDeps]) --------


def _is_known_contact_impl(email_address: str, deps: TriageDeps) -> bool:
    """Case-insensitive membership check against deps.known_contacts.
    Matches either an exact address or the name portion before '<'."""
    lower = email_address.lower()
    for contact in deps.known_contacts:
        contact_lower = contact.lower()
        if contact_lower in lower or lower in contact_lower:
            return True
    return False


def _is_important_domain_impl(email_address: str, deps: TriageDeps) -> bool:
    """Case-insensitive check: does `email_address` end with any of
    `deps.important_domains`?"""
    lower = email_address.lower()
    for domain in deps.important_domains:
        if lower.endswith(domain.lower().lstrip("@")):
            return True
        if domain.lower() in lower:
            return True
    return False


def _extract_dates_from(body: str) -> list[str]:
    """Return date/deadline phrases found in the body. Reuses the
    keyword-set approach from #04's meeting-notes agent; kept local
    here rather than shared via common/ (would drag common/ into
    scope for a small helper). Grounds `reasoning` claims about
    deadlines."""
    keywords = [
        r"\b(?:tomorrow|today|yesterday)\b",
        r"\bnext\s+(?:week|month|monday|tuesday|wednesday|thursday|friday)\b",
        r"\bby\s+\w+day\b",
        r"\bend\s+of\s+(?:week|month|day)\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
    ]
    combined = re.compile("|".join(keywords), re.IGNORECASE)
    seen: dict[str, None] = {}
    for match in combined.finditer(body):
        seen[match.group(0)] = None
    return list(seen.keys())


def _normalize_ws(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s).strip()


def _verify_snippet(excerpt: str, body: str) -> bool:
    """Substring check with whitespace normalization -- same [C5]
    pattern as #02/#04/#05. Empty-excerpt guard (M1 pattern from
    #05): `"" in "anything"` is True in Python, so reject empty
    explicitly."""
    if not excerpt.strip():
        return False
    return _normalize_ws(excerpt) in _normalize_ws(body)


# --- Public API ------------------------------------------------------------


def triage_email(
    eml_bytes: bytes,
    *,
    deps: TriageDeps | None = None,
    model: str | None = None,
    _agent=None,  # test-injection escape hatch
) -> EmailTriage:
    """Triage a raw .eml file into a structured EmailTriage.

    Args:
        eml_bytes: raw .eml file contents (RFC 2822).
        deps: TriageDeps carrying user context (known_contacts +
            important_domains). Defaults to empty deps if None.
        model: model ID; defaults to `common.llm.resolve_model("openai")`.
        _agent: injected pre-built Agent for tests. Production callers
            leave this None; the agent is built via _build_agent.

    Returns:
        A validated EmailTriage.

    Raises:
        EmailTriageError: on any of the 3 R5 failure modes.
    """
    resolved_provider = resolve_provider()
    resolved_deps = deps if deps is not None else TriageDeps()

    if resolved_provider == "mock":
        return _mock_result(eml_bytes)

    # R5 case 1: parse
    parsed = _parse_eml(eml_bytes)

    # R5 case 2: empty body
    if not _has_triage_material(parsed.body):
        raise EmailTriageError(
            f"Email body is essentially empty ({len(parsed.body.strip())} "
            f"chars, need at least {MIN_BODY_CHARS}). Nothing to triage.",
            partial=TriageAttempt(stage="empty_body"),
        )

    resolved_model = model or resolve_model(resolved_provider)

    # Lazy imports: pydantic-ai works on Py3.14 but we still lazy-load
    # so schema tests + mock tests don't pull in the whole SDK.
    try:
        import pydantic_ai  # noqa: F401 -- availability check only
    except ImportError as exc:
        raise EmailTriageError(
            "pydantic-ai is not installed. Run `uv sync` at the workspace root."
        ) from exc

    agent = _agent if _agent is not None else _build_agent(model=resolved_model)

    prompt = _format_email_for_prompt(parsed)

    try:
        result = agent.run_sync(prompt, deps=resolved_deps)
    except EmailTriageError:
        raise
    except Exception as exc:  # R5 case 3
        raise _translate_api_error(exc) from exc

    # PydanticAI returns typed output via .output (verified against
    # installed 2.33.0). Should always be an EmailTriage if the SDK's
    # structured-output validation succeeded; the type-check below is
    # a belt-and-braces guard.
    triage = getattr(result, "output", None)
    if not isinstance(triage, EmailTriage):
        raise EmailTriageError(
            "Agent returned output but it didn't validate as an EmailTriage. "
            "This usually means the model produced JSON missing required "
            "fields or with invalid Literal values.",
            partial=TriageAttempt(
                stage="agent_run", raw_output=str(result)[:1000],
            ),
        )
    return triage


def _format_email_for_prompt(parsed: ParsedEmail) -> str:
    """Format the parsed .eml as a compact prompt for the agent."""
    to_line = ", ".join(parsed.to_addresses) or "(unknown)"
    return (
        f"From: {parsed.from_address}\n"
        f"To: {to_line}\n"
        f"Date: {parsed.date}\n"
        f"Subject: {parsed.subject}\n\n"
        f"{parsed.body}"
    )


# --- Agent factory (pedagogical anchor: read this to see the typed
#     Agent + Deps + tool wiring in one place) ----------------------------


def _build_agent(*, model: str):
    """Build a PydanticAI Agent[TriageDeps, EmailTriage] with 4 tools
    that access injected deps via RunContext[TriageDeps]. LLM is
    addressed via PydanticAI's provider-prefix convention (`openai:`,
    `anthropic:`, `gemini:` -- swap here for multi-provider).

    Lazy import: pydantic-ai has a heavy import graph we don't want
    in mock-mode tests."""
    from pydantic_ai import Agent, RunContext

    agent = Agent(
        f"openai:{model}",
        deps_type=TriageDeps,
        output_type=EmailTriage,
        system_prompt=_load_system_prompt(),
        retries=DEFAULT_RETRIES,
    )

    @agent.tool
    def is_known_contact(ctx: RunContext[TriageDeps], email_address: str) -> bool:
        """Return True iff `email_address` matches any of the user's
        known contacts (case-insensitive substring match on address
        or name portion)."""
        return _is_known_contact_impl(email_address, ctx.deps)

    @agent.tool
    def is_important_domain(ctx: RunContext[TriageDeps], email_address: str) -> bool:
        """Return True iff `email_address` is at one of the user's
        important domains (case-insensitive suffix match)."""
        return _is_important_domain_impl(email_address, ctx.deps)

    @agent.tool
    def extract_dates(ctx: RunContext[TriageDeps], body: str) -> list[str]:
        """Extract date/deadline phrases from the body. Use to ground
        priority + action decisions (e.g. 'by Friday' -> respond_now
        if today is Thursday). Deps param unused but present for
        signature consistency with other tools."""
        _ = ctx  # deps not needed
        return _extract_dates_from(body)

    @agent.tool
    def verify_snippet(ctx: RunContext[TriageDeps], excerpt: str, body: str) -> bool:
        """Return True iff `excerpt` is a verbatim (whitespace-
        normalized) substring of `body`. Call BEFORE finalizing
        the EmailTriage's key_snippet field -- prevents paraphrased
        excerpts that break the user's 'locate the evidence' workflow."""
        _ = ctx  # deps not needed
        return _verify_snippet(excerpt, body)

    return agent


# --- Error translation (R5 case 3) -----------------------------------------


def _translate_api_error(exc: Exception) -> EmailTriageError:
    """Turn a PydanticAI / OpenAI SDK exception into a user-facing
    EmailTriageError. Same 6-branch priority order as #02-#05."""
    exc_class_name = type(exc).__name__.lower()
    message_lower = str(exc).lower()
    status = getattr(exc, "status_code", None)

    if "ratelimiterror" in exc_class_name:
        return _rate_limit_error()
    if "authenticationerror" in exc_class_name or "apikeyerror" in exc_class_name:
        return _auth_error()
    if status == 429:
        return _rate_limit_error()
    if status == 401:
        return _auth_error()
    if "rate limit" in message_lower or "overloaded" in message_lower:
        return _rate_limit_error()
    if "authentication" in message_lower or "api key" in message_lower:
        return _auth_error()

    return EmailTriageError(
        f"Triage failed: {type(exc).__name__}: {exc}. "
        "This is an unexpected error."
    )


def _rate_limit_error() -> EmailTriageError:
    return EmailTriageError(
        "The service is temporarily rate-limited or overloaded. "
        "Wait a minute and try again."
    )


def _auth_error() -> EmailTriageError:
    return EmailTriageError(
        "API authentication failed. Check that OPENAI_API_KEY is set."
    )


# --- Mock mode -------------------------------------------------------------


def _mock_result(eml_bytes: bytes) -> EmailTriage:
    """Deterministic canned EmailTriage for smoke tests and CI.
    Does NOT parse the .eml, does NOT need pydantic-ai installed,
    does NOT run the R5 gates. Byte count encoded into reasoning
    so tests can prove the mock saw its input."""
    return EmailTriage(
        category="work",
        priority="medium",
        action="respond_later",
        reasoning=(
            f"Mock triage (input: {len(eml_bytes)} bytes). This is a "
            "deterministic canned response for LLM_PROVIDER=mock. Set "
            "LLM_PROVIDER=openai + OPENAI_API_KEY for a real triage."
        ),
        key_snippet="Mock snippet placeholder",
        suggested_reply="Mock draft reply.",
    )


# --- CLI entry point (uv run python -m agent) ------------------------------


def main() -> int:
    """CLI: reads a .eml file, prints the EmailTriage as JSON."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="email-triage",
        description="Triage an email .eml file into structured category/priority/action.",
    )
    parser.add_argument("path", nargs="?", help="Path to a .eml file.")
    parser.add_argument("--model", help="Override the resolved model.")
    parser.add_argument(
        "--known-contact",
        action="append",
        default=[],
        help="Add an email/name to known_contacts (repeatable).",
    )
    parser.add_argument(
        "--important-domain",
        action="append",
        default=[],
        help="Add a domain to important_domains (repeatable).",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Gradio UI instead of a one-shot CLI run.",
    )
    args = parser.parse_args()

    if args.ui:
        try:
            from .ui import build_ui  # type: ignore[import-not-found]
        except ImportError:
            from ui import build_ui  # type: ignore[import-not-found]
        build_ui().launch()
        return 0

    if not args.path:
        parser.error("path is required unless --ui is passed")

    path = Path(args.path)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    eml_bytes = path.read_bytes()
    deps = TriageDeps(
        known_contacts=args.known_contact,
        important_domains=args.important_domain,
    )

    try:
        triage = triage_email(eml_bytes, deps=deps, model=args.model)
    except EmailTriageError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.partial is not None:
            print(f"stage: {exc.partial.stage}", file=sys.stderr)
        return 1

    print(triage.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
