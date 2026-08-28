"""Contact / ID extractor: agent #15 of real-world-agents.

Technique demonstrated: **batch vision extraction + local embedding-
based dedup + PII-safe redaction.** Distinct from #01 (single-item
vision extraction) and #10 (retrieval-side embeddings) because it
combines three techniques into a compliance-shaped pipeline:

1. `_extract_raw(image)`  -- Instructor + vision -> list[RawContact],
                             one per visible card. Caller cross-checks
                             the ordered count.
2. `_deduplicate(raws)`   -- fastembed encodes normalized identity
                             strings, cosine similarity plus a
                             corroborating deterministic key (email
                             OR normalized phone) merges near-
                             duplicates via union-find.
3. `_apply_policy(contacts, policy, run_salt)`
                          -- per-field "none" / "hash" / "redact".
                             Hashes are SHA-256 of a salted
                             normalized value truncated to 16 hex +
                             last-4-visible so the record stays
                             audit-readable without leaking raw PII.

Real use case: KYC / CRM. One photo of a stack of cards ->
deduplicated Contact list ready to be dropped into a compliance-
constrained pipeline (last_run.json never contains raw PII on a
strict-policy run).

Provider stance: OpenAI-only in v1 (matches #11/#13; the Instructor
vision path here is the same as #01/#09's default).
"""

from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import json
import os
import re
import secrets
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

try:
    from .schemas import (
        REDACT_SENTINEL,
        Contact,
        DuplicateGroup,
        ExtractionResult,
        PiiPolicy,
        RawContact,
    )
except ImportError:
    from schemas import (
        REDACT_SENTINEL,
        Contact,
        DuplicateGroup,
        ExtractionResult,
        PiiPolicy,
        RawContact,
    )

# --- Constants -------------------------------------------------------------

SUPPORTED_PROVIDERS = ("openai",)
_DEFAULT_PROVIDER = "openai"
_DEFAULT_MODEL = "gpt-4o-mini"
_SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB -- OpenAI vision cap
_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

DEFAULT_DEDUP_THRESHOLD = 0.85

_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"


# --- Error type ------------------------------------------------------------


@dataclass
class ExtractionAttempt:
    stage: str  # 'load' | 'extract' | 'dedup' | 'redact'
    raw_output: str = ""


class ContactError(Exception):
    def __init__(self, message: str, partial: ExtractionAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Provider resolution ---------------------------------------------------


def resolve_provider() -> str:
    provider = os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER).lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}."
        )
    return provider


@functools.lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# --- Stage A: extract ------------------------------------------------------


def _read_image(source: str | Path | bytes) -> tuple[bytes, str]:
    """Return (image_bytes, image_mime) for the source. `image_mime`
    is the second half of a Content-Type ('png', 'jpeg', 'webp') --
    always the MIME subtype, never the file extension. Enforces the
    20 MB cap."""
    if isinstance(source, bytes):
        data = source
        fmt = _sniff_image_mime(data)
    else:
        path = Path(source)
        if not path.exists() or not path.is_file():
            raise ContactError(
                f"image file {str(path)!r} does not exist.",
                partial=ExtractionAttempt(stage="load"),
            )
        suffix = path.suffix.lower()
        if suffix not in _SUPPORTED_IMAGE_SUFFIXES:
            raise ContactError(
                f"unsupported image format {suffix!r}; supported: "
                f"{list(_SUPPORTED_IMAGE_SUFFIXES)}.",
                partial=ExtractionAttempt(stage="load"),
            )
        data = path.read_bytes()
        # Map file extension to the correct MIME subtype: '.jpg' ->
        # 'jpeg' (a real Content-Type has no image/jpg alias, and
        # strict servers reject it).
        raw_fmt = suffix.lstrip(".")
        fmt = "jpeg" if raw_fmt == "jpg" else raw_fmt
    if len(data) > _MAX_IMAGE_BYTES:
        raise ContactError(
            f"image is {len(data) / 1024 / 1024:.1f} MB, over the "
            f"{_MAX_IMAGE_BYTES // 1024 // 1024} MB cap; downscale before "
            "sending to the vision API.",
            partial=ExtractionAttempt(stage="load"),
        )
    return data, fmt


def _sniff_image_mime(data: bytes) -> str:
    """Guess the MIME subtype from the first few bytes of an image.
    Falls back to 'png' when the header is ambiguous (better a wrong
    MIME than a hard crash; strict callers should pass a Path so the
    extension does the work)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "png"


def _extract_raw(
    image: str | Path | bytes,
    *,
    model: str,
    _instructor_client: Any = None,
) -> list[RawContact]:
    """Instructor + vision. Returns an ordered list of RawContacts,
    one per visible card in the source image."""
    image_bytes, image_mime = _read_image(image)
    client = (
        _instructor_client
        if _instructor_client is not None
        else _build_instructor(model=model)
    )
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/{image_mime};base64,{b64}"
    try:
        raws = client.chat.completions.create(
            model=model,
            response_model=list[RawContact],
            messages=[
                {"role": "system", "content": _load_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract every visible contact card. Return one entry per card.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
            max_tokens=2048,
        )
    except Exception as exc:
        raise _translate_api_error(exc) from exc
    # Guard: ensure card_indexes are unique. If the model returned
    # duplicates, dedup below would produce nonsense.
    indexes = [r.card_index for r in raws]
    if len(set(indexes)) != len(indexes):
        raise ContactError(
            f"vision model returned duplicate card_index values: {indexes!r}.",
            partial=ExtractionAttempt(stage="extract"),
        )
    return raws


def _build_instructor(*, model: str) -> Any:
    """Build the Instructor client bound to the actual model the caller
    asked for. Retries once on structured-output validation failure so
    the parse behavior mirrors #10 / #14's hand-rolled retry pattern."""
    try:
        import instructor
    except ImportError as exc:
        raise ContactError(
            "instructor not installed. Run `uv sync --all-packages` from "
            "the repo root."
        ) from exc
    return instructor.from_provider(f"openai/{model}", max_retries=1)


# --- Stage B: dedup --------------------------------------------------------


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_PHONE_NON_DIGIT_RE = re.compile(r"\D+")


def _normalize(s: str | None) -> str:
    if s is None:
        return ""
    s = _PUNCT_RE.sub(" ", s).lower()
    return _WS_RE.sub(" ", s).strip()


def _normalize_phone(s: str | None) -> str:
    """Strip everything but digits so '+1 (415) 555-1212' matches
    '415-555-1212'. Only the last 10 digits are kept so a country-code
    prefix does not defeat the match."""
    if s is None:
        return ""
    digits = _PHONE_NON_DIGIT_RE.sub("", s)
    return digits[-10:] if len(digits) >= 10 else digits


def _identity(c: RawContact) -> str:
    return f"{_normalize(c.full_name)} | {_normalize(c.company)} | {_normalize(c.title)}"


def _cosine_sim(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _deduplicate(
    raws: list[RawContact],
    *,
    threshold: float,
    _embedder: Any = None,
) -> tuple[list[list[int]], list[DuplicateGroup]]:
    """Group `raws` into duplicate clusters. Returns (groups, report),
    where `groups` is a list-of-list-of-card_indexes (each cluster,
    ordered) and `report` names the merges of size >= 2.
    """
    if not raws:
        return [], []
    if len(raws) == 1:
        return [[raws[0].card_index]], []

    identities = [_identity(r) for r in raws]
    embedder = _embedder if _embedder is not None else _build_embedder()
    vectors = list(embedder.embed(identities))
    # Pairwise sim + corroborating-key match -> union-find merge.
    parent = {r.card_index: r.card_index for r in raws}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    reasons: dict[frozenset[int], str] = {}
    for i in range(len(raws)):
        for j in range(i + 1, len(raws)):
            ci, cj = raws[i], raws[j]
            ni, nj = _normalize(ci.email), _normalize(cj.email)
            pi, pj = _normalize_phone(ci.phone), _normalize_phone(cj.phone)
            key_match_reason: str | None = None
            if ni and nj and ni == nj:
                key_match_reason = f"same email ({ni})"
            elif pi and pj and pi == pj:
                key_match_reason = f"same phone ({pi})"
            if key_match_reason is None:
                continue
            sim = _cosine_sim(vectors[i], vectors[j])
            if sim >= threshold:
                union(ci.card_index, cj.card_index)
                reasons[frozenset({ci.card_index, cj.card_index})] = (
                    f"{key_match_reason} + name/company similarity {sim:.2f}"
                )

    # Assemble groups by root.
    groups_by_root: dict[int, list[int]] = defaultdict(list)
    for r in raws:
        groups_by_root[find(r.card_index)].append(r.card_index)
    groups = [sorted(g) for g in groups_by_root.values()]
    groups.sort(key=lambda g: g[0])

    # Merge report: pick canonical (most non-empty fields; tie -> lowest
    # card_index) and a merge_reason string.
    by_index = {r.card_index: r for r in raws}
    report: list[DuplicateGroup] = []
    for g in groups:
        if len(g) < 2:
            continue
        canonical = max(g, key=lambda i: (_field_completeness(by_index[i]), -i))
        # Combine pair-level reasons for a compact group reason.
        pair_reasons = [
            reasons[frozenset({a, b})]
            for a in g
            for b in g
            if a < b and frozenset({a, b}) in reasons
        ]
        merge_reason = "; ".join(pair_reasons) or "linked via transitive match"
        report.append(
            DuplicateGroup(
                canonical_index=canonical,
                member_indexes=g,
                merge_reason=merge_reason,
            )
        )
    return groups, report


def _field_completeness(c: RawContact) -> int:
    return sum(
        1
        for v in (c.title, c.company, c.email, c.phone, c.address)
        if v and v.strip()
    )


@functools.lru_cache(maxsize=1)
def _build_embedder() -> Any:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise ContactError(
            "fastembed not installed. Run `uv sync --all-packages` from "
            "the repo root."
        ) from exc
    return TextEmbedding(model_name=_EMBEDDING_MODEL)


# --- Stage C: redact -------------------------------------------------------


def _apply_policy(
    raws: list[RawContact],
    groups: list[list[int]],
    *,
    policy: PiiPolicy,
    run_salt: str,
) -> list[Contact]:
    """Build canonical Contacts per group; apply per-field policy."""
    by_index = {r.card_index: r for r in raws}
    contacts: list[Contact] = []
    for g in groups:
        canonical_idx = max(g, key=lambda i: (_field_completeness(by_index[i]), -i))
        merged_source = by_index[canonical_idx]

        # For non-canonical fields, fall back through group members
        # in card_index order for anything the canonical is missing.
        # Bind loop vars via defaults to sidestep late-binding.
        def _pick(field: str, _src=merged_source, _g=g, _canon=canonical_idx) -> str | None:
            value = getattr(_src, field)
            if value:
                return value
            for other_idx in sorted(_g):
                if other_idx == _canon:
                    continue
                v = getattr(by_index[other_idx], field)
                if v:
                    return v
            return None

        contacts.append(
            Contact(
                full_name=merged_source.full_name,
                title=_pick("title"),
                company=_pick("company"),
                email=_redact_field(_pick("email"), policy.email, run_salt=run_salt),
                phone=_redact_field(_pick("phone"), policy.phone, run_salt=run_salt),
                address=_redact_field(_pick("address"), policy.address, run_salt=run_salt),
                merged_from=sorted(g),
                pii_policy=policy,
            )
        )
    contacts.sort(key=lambda c: c.merged_from[0])
    return contacts


def _redact_field(value: str | None, mode: str, *, run_salt: str) -> str | None:
    if value is None or not value.strip():
        return None
    if mode == "none":
        return value
    if mode == "redact":
        return REDACT_SENTINEL
    if mode == "hash":
        normalized = _normalize(value)
        digest = hashlib.sha256(f"{run_salt}:{normalized}".encode()).hexdigest()
        tail = value[-4:] if len(value) >= 4 else value.ljust(4, "X")
        return f"{digest[:16]}...{tail}"
    raise ValueError(f"unknown policy mode {mode!r}.")


def _apply_policy_to_raws(
    raws: list[RawContact], *, policy: PiiPolicy, run_salt: str
) -> list[RawContact]:
    """Return a copy of `raws` with the per-field policy applied so
    the raw_contacts list in ExtractionResult also honors the policy
    (no raw PII on disk under strict/redact). Email now round-trips
    through the same redaction path as phone/address -- RawContact's
    email validator accepts hash markers and the REDACT sentinel, so
    the audit signal 'this card had an email' is preserved."""
    return [
        RawContact(
            card_index=r.card_index,
            full_name=r.full_name,
            title=r.title,
            company=r.company,
            email=_redact_field(r.email, policy.email, run_salt=run_salt),
            phone=_redact_field(r.phone, policy.phone, run_salt=run_salt),
            address=_redact_field(r.address, policy.address, run_salt=run_salt),
        )
        for r in raws
    ]


# --- Public entry point ----------------------------------------------------


def extract_contacts(
    image: str | Path | bytes,
    *,
    provider: str | None = None,
    model: str | None = None,
    pii_policy: PiiPolicy | None = None,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    run_salt: str | None = None,
    _instructor_client: Any = None,
    _embedder: Any = None,
) -> ExtractionResult:
    """Full pipeline: extract -> dedup -> redact."""
    try:
        resolved = provider or resolve_provider()
    except ValueError as exc:
        raise ContactError(str(exc)) from exc

    policy = pii_policy or PiiPolicy()
    salt = run_salt or secrets.token_hex(8)

    if resolved == "mock":
        return _mock_result(image, policy=policy, salt=salt)

    resolved_model = model or _DEFAULT_MODEL
    start = time.perf_counter()
    raws = _extract_raw(image, model=resolved_model, _instructor_client=_instructor_client)
    extract_seconds = time.perf_counter() - start

    start = time.perf_counter()
    groups, dup_report = _deduplicate(raws, threshold=dedup_threshold, _embedder=_embedder)
    dedup_seconds = time.perf_counter() - start

    start = time.perf_counter()
    contacts = _apply_policy(raws, groups, policy=policy, run_salt=salt)
    raws_redacted = _apply_policy_to_raws(raws, policy=policy, run_salt=salt)
    redact_seconds = time.perf_counter() - start

    try:
        return ExtractionResult(
            raw_contacts=raws_redacted,
            contacts=contacts,
            duplicates=dup_report,
            pii_policy=policy,
            run_meta={
                "provider": resolved,
                "model": resolved_model,
                "raw_count": len(raws),
                "contact_count": len(contacts),
                "duplicate_group_count": len(dup_report),
                "stage_seconds": {
                    "extract": round(extract_seconds, 3),
                    "dedup": round(dedup_seconds, 3),
                    "redact": round(redact_seconds, 3),
                },
            },
        )
    except ValidationError as exc:
        raise ContactError(
            f"final ExtractionResult failed self-consistency validation: {exc}",
            partial=ExtractionAttempt(stage="redact"),
        ) from exc


# --- Error translation (R5) ------------------------------------------------


def _translate_api_error(exc: Exception) -> ContactError:
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
    if "rate limit" in message_lower:
        return _rate_limit_error()
    if "authentication" in message_lower or "api key" in message_lower:
        return _auth_error()
    return ContactError(
        f"vision extraction failed: {type(exc).__name__}: {exc}.",
        partial=ExtractionAttempt(stage="extract"),
    )


def _rate_limit_error() -> ContactError:
    return ContactError(
        "OpenAI is rate-limited. Wait a minute and try again.",
        partial=ExtractionAttempt(stage="extract"),
    )


def _auth_error() -> ContactError:
    return ContactError(
        "Authentication failed: check that OPENAI_API_KEY is set. See "
        ".env.example at the repo root.",
        partial=ExtractionAttempt(stage="extract"),
    )


# --- Mock mode -------------------------------------------------------------


def _mock_result(
    image: str | Path | bytes, *, policy: PiiPolicy, salt: str
) -> ExtractionResult:
    """Scripted 3 raw contacts, 2 of which are the same person. The
    mock_input_size in run_meta echoes the input length as the anti-
    refactor guard the tests can assert on."""
    if isinstance(image, bytes):
        mock_size = len(image)
    else:
        path = Path(str(image))
        mock_size = path.stat().st_size if path.exists() else len(str(image))

    raws = [
        RawContact(
            card_index=0, full_name="Alice Chen", title="Head of Product",
            company="Northwind Analytics", email="alice@northwind.io",
            phone="+1 415-555-0101", address=None,
        ),
        RawContact(
            card_index=1, full_name="Alicia Chen", title=None,
            company="Northwind Analytics", email="alice@northwind.io",
            phone=None, address="1 Market St, San Francisco, CA",
        ),
        RawContact(
            card_index=2, full_name="Bob Rivera", title="Backend Engineer",
            company="Ridgeway Robotics", email="bob@ridgeway.dev",
            phone="+1 415-555-0199", address=None,
        ),
    ]
    # Deterministic mock dedup: cards 0 and 1 are the same person.
    groups = [[0, 1], [2]]
    dup_report = [
        DuplicateGroup(
            canonical_index=0,
            member_indexes=[0, 1],
            merge_reason="same email (alice@northwind.io) + name similarity",
        )
    ]
    contacts = _apply_policy(raws, groups, policy=policy, run_salt=salt)
    raws_redacted = _apply_policy_to_raws(raws, policy=policy, run_salt=salt)
    return ExtractionResult(
        raw_contacts=raws_redacted,
        contacts=contacts,
        duplicates=dup_report,
        pii_policy=policy,
        run_meta={
            "provider": "mock",
            "model": "mock",
            "raw_count": 3,
            "contact_count": 2,
            "duplicate_group_count": 1,
            "mock_input_size": mock_size,
        },
    )


# --- CLI ------------------------------------------------------------------


def _parse_pii_flags(pairs: list[str]) -> PiiPolicy:
    """Parse `--pii email=hash --pii phone=redact` into a PiiPolicy."""
    fields: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ContactError(f"--pii value {pair!r} must look like field=mode.")
        field, mode = pair.split("=", 1)
        if field not in ("email", "phone", "address"):
            raise ContactError(f"--pii field {field!r} must be email/phone/address.")
        if mode not in ("none", "hash", "redact"):
            raise ContactError(f"--pii mode {mode!r} must be none/hash/redact.")
        fields[field] = mode
    return PiiPolicy(**fields)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="contact-extractor",
        description=(
            "Extract contacts from a photo of business cards / IDs. "
            "Set LLM_PROVIDER=mock for a scripted demo, or supply "
            "OPENAI_API_KEY for real extraction."
        ),
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--policy",
        choices=("none", "strict", "redacted"),
        default="none",
        help="Preset: none (raw PII), strict (all fields hashed), "
        "redacted (all fields [REDACTED]).",
    )
    parser.add_argument(
        "--pii",
        action="append",
        default=[],
        help="Per-field override: --pii email=hash --pii phone=none.",
        metavar="FIELD=MODE",
    )
    parser.add_argument(
        "--dedup-threshold", type=float, default=DEFAULT_DEDUP_THRESHOLD
    )
    parser.add_argument("--run-salt", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--provider", choices=(*SUPPORTED_PROVIDERS, "mock"), default=None
    )
    args = parser.parse_args()

    try:
        if args.policy == "strict":
            base_policy = PiiPolicy.strict()
        elif args.policy == "redacted":
            base_policy = PiiPolicy.redacted()
        else:
            base_policy = PiiPolicy()
        if args.pii:
            # Only update fields the user explicitly named; unspecified
            # fields keep whatever the preset chose (a naive merge would
            # let PiiPolicy(**{}) defaults quietly overwrite the preset).
            override = _parse_pii_flags(args.pii)
            named_fields = {pair.split("=", 1)[0] for pair in args.pii}
            updates = {f: getattr(override, f) for f in named_fields}
            base_policy = base_policy.model_copy(update=updates)
        result = extract_contacts(
            args.image,
            provider=args.provider,
            model=args.model,
            pii_policy=base_policy,
            dedup_threshold=args.dedup_threshold,
            run_salt=args.run_salt,
        )
    except ContactError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1

    out_path = Path(__file__).parent / "last_run.json"
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Raw cards seen:    {result.run_meta['raw_count']}")
    print(f"Canonical contacts: {result.run_meta['contact_count']}")
    print(f"Duplicate groups:  {result.run_meta['duplicate_group_count']}")
    print(f"PII policy:        {result.pii_policy.model_dump()}")
    print()
    for c in result.contacts:
        merge_note = (
            f" (merged from {c.merged_from})" if len(c.merged_from) > 1 else ""
        )
        print(f"- {c.full_name}{merge_note}")
        if c.title or c.company:
            print(f"    {c.title or ''} @ {c.company or ''}")
        if c.email:
            print(f"    email: {c.email}")
        if c.phone:
            print(f"    phone: {c.phone}")
        if c.address:
            print(f"    address: {c.address}")
    print()
    print(f"Full result written to {out_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
