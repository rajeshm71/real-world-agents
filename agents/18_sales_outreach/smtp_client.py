"""The one real external side effect in this agent: sending an email
via SMTP, using the end user's own credentials.

`send_email()` is the public entry point. It never raises for an
expected real-world failure (auth rejected, connection refused,
timeout) -- those become a `SendEmailResult(sent=False, error=...)`,
consumed as ordinary data by whatever calls it, mirroring agent #08's
`_run_pytest_in_sandbox` stance on its own real-world side effect
(subprocess execution there, a socket here). Only a genuinely
unexpected exception type is allowed to propagate.

`dry_run=True` is a true no-op: it never touches credentials, never
imports anything socket-related, never calls `_sender`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from .schemas import SendEmailResult
except ImportError:
    from schemas import SendEmailResult

DEFAULT_SMTP_HOST = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 587
DEFAULT_TIMEOUT_SECONDS = 15.0


class SmtpNotConfigured(Exception):
    """Raised at credential-resolution time when SMTP_USER or
    SMTP_APP_PASSWORD is unset. Those two have no safe default (unlike
    SMTP_HOST/SMTP_PORT, which default to Gmail's values) -- there is
    no sender identity to guess."""


@dataclass
class SmtpCredentials:
    host: str
    port: int
    user: str
    app_password: str = field(repr=False)  # never print the password
    from_address: str


def load_smtp_credentials() -> SmtpCredentials:
    """Reads SMTP_HOST (default smtp.gmail.com), SMTP_PORT (default
    587), SMTP_USER, SMTP_APP_PASSWORD, SMTP_FROM (optional, defaults
    to SMTP_USER) from the environment. Raises SmtpNotConfigured with
    exact Gmail App Password setup steps if SMTP_USER or
    SMTP_APP_PASSWORD is unset."""
    user = os.environ.get("SMTP_USER")
    app_password = os.environ.get("SMTP_APP_PASSWORD")
    if not user or not app_password:
        raise SmtpNotConfigured(
            "SMTP_USER and/or SMTP_APP_PASSWORD are not set, but --send "
            "was requested. Default setup (Gmail): enable 2-factor "
            "authentication on your Google account, then generate an "
            "App Password at https://myaccount.google.com/apppasswords "
            "(free, no new signup). Set SMTP_USER to your Gmail address "
            "and SMTP_APP_PASSWORD to the generated 16-character "
            "password. Using a different provider? Set SMTP_HOST and "
            "SMTP_PORT to override the smtp.gmail.com:587 default."
        )
    host = os.environ.get("SMTP_HOST", DEFAULT_SMTP_HOST)
    port = int(os.environ.get("SMTP_PORT", str(DEFAULT_SMTP_PORT)))
    from_address = os.environ.get("SMTP_FROM", user)
    return SmtpCredentials(
        host=host, port=port, user=user, app_password=app_password, from_address=from_address
    )


def _send_via_smtp(
    *,
    creds: SmtpCredentials,
    to_address: str,
    subject: str,
    body: str,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
) -> SendEmailResult:
    """The one function that opens a real socket. Never raises for an
    expected failure -- catches smtplib.SMTPException / OSError /
    TimeoutError and returns a structured failure result instead."""
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = creds.from_address
    msg["To"] = to_address
    msg.set_content(body)

    try:
        with smtplib.SMTP(creds.host, creds.port, timeout=timeout_s) as server:
            server.starttls()
            server.login(creds.user, creds.app_password)
            server.send_message(msg)
        return SendEmailResult(sent=True, error=None, dry_run=False)
    except smtplib.SMTPAuthenticationError as exc:
        return SendEmailResult(
            sent=False,
            error=f"SMTP authentication failed: {exc}. Check SMTP_USER/SMTP_APP_PASSWORD.",
            dry_run=False,
        )
    except smtplib.SMTPException as exc:
        return SendEmailResult(
            sent=False, error=f"SMTP error: {type(exc).__name__}: {exc}", dry_run=False
        )
    except (OSError, TimeoutError) as exc:
        return SendEmailResult(
            sent=False, error=f"Connection error: {type(exc).__name__}: {exc}", dry_run=False
        )


def send_email(
    *,
    to_address: str,
    subject: str,
    body: str,
    dry_run: bool,
    creds: SmtpCredentials | None = None,
    _sender=_send_via_smtp,
) -> SendEmailResult:
    """Public entry point.

    dry_run=True returns a no-op result WITHOUT calling `_sender` or
    resolving credentials at all -- a true no-op, never touches a
    socket or an env var.

    `creds` is normally passed in already-resolved by the caller
    (agent.py resolves credentials once, up front, for a whole batch --
    see agent.py's run_outreach_batch). If `creds is None` and
    `dry_run=False`, falls back to resolving it here via
    `load_smtp_credentials()`, so this function is still usable
    standalone (e.g. called directly in a test).

    `_sender` is the test-injection seam: tests monkeypatch this
    parameter (or `_send_via_smtp` itself) to a fake that records calls
    and returns a canned result, so pytest never opens a real socket.
    """
    if dry_run:
        return SendEmailResult(sent=False, error=None, dry_run=True)
    if creds is None:
        creds = load_smtp_credentials()
    return _sender(creds=creds, to_address=to_address, subject=subject, body=body)
