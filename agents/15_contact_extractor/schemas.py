"""Pydantic models for the contact / ID extractor.

Five types travel the extract -> dedup -> redact pipeline:

- `RawContact`:       one contact as extracted from a single visible
                      card. Ordered by `card_index`.
- `PiiPolicy`:        per-field redaction mode (`"none"` / `"hash"`
                      / `"redact"`).
- `Contact`:          post-dedup canonical contact. Carries the
                      applied `pii_policy` so the on-disk dump is
                      self-describing.
- `DuplicateGroup`:   merge report -- which raw indexes collapsed
                      into which canonical, and why.
- `ExtractionResult`: full pipeline return. Cross-field validators
                      enforce that `merged_from` values partition the
                      raw indexes, that duplicate references are
                      valid, and that under a "hash" policy the field
                      value actually looks like the shipped hash
                      marker (a downstream sanity check on the
                      redactor).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Lightweight email + hash-marker regexes. Deliberately simple: this
# repo does not depend on pydantic[email].
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
# Hash marker shape produced by _apply_policy: 16 hex chars + '...' +
# last-4 of the original value (or 'XXXX' when the value is too short).
_HASH_MARKER_RE = re.compile(r"^[0-9a-f]{16}\.\.\..{1,8}$")
REDACT_SENTINEL = "[REDACTED]"


class RawContact(BaseModel):
    """One contact as the vision model extracted it from a single
    visible card. `card_index` records the position in the source
    image (left-to-right, top-to-bottom, starting at 0)."""

    card_index: int = Field(..., ge=0)
    full_name: str = Field(..., min_length=1)
    title: str | None = None
    company: str | None = None
    email: str | None = None
    phone: str | None = None
    address: str | None = None

    @model_validator(mode="after")
    def _email_shape(self) -> RawContact:
        if self.email is not None and not _EMAIL_RE.match(self.email):
            raise ValueError(
                f"email {self.email!r} does not look like an address; "
                "the model must not fabricate one from partial text."
            )
        return self


class PiiPolicy(BaseModel):
    """Per-field redaction policy. Defaults to no redaction."""

    email: Literal["none", "hash", "redact"] = "none"
    phone: Literal["none", "hash", "redact"] = "none"
    address: Literal["none", "hash", "redact"] = "none"

    @classmethod
    def strict(cls) -> PiiPolicy:
        """Every PII field hashed. Compliance-safe default when
        retaining a queryable record."""
        return cls(email="hash", phone="hash", address="hash")

    @classmethod
    def redacted(cls) -> PiiPolicy:
        """Every PII field replaced with [REDACTED]. Compliance-safe
        default when the record itself should not carry recoverable
        PII at all."""
        return cls(email="redact", phone="redact", address="redact")


class Contact(BaseModel):
    """Post-dedup canonical contact. `merged_from` names every raw
    card_index that collapsed into this one (self alone if a
    singleton). `pii_policy` is echoed so the dump names which policy
    produced the current field values."""

    full_name: str = Field(..., min_length=1)
    title: str | None = None
    company: str | None = None
    email: str | None = None
    phone: str | None = None
    address: str | None = None
    merged_from: list[int] = Field(..., min_length=1)
    pii_policy: PiiPolicy

    @model_validator(mode="after")
    def _merged_from_unique(self) -> Contact:
        if len(set(self.merged_from)) != len(self.merged_from):
            raise ValueError(
                f"merged_from must be unique; got {self.merged_from!r}."
            )
        return self

    @model_validator(mode="after")
    def _hashed_fields_look_hashed(self) -> Contact:
        # If the policy says a field is hashed, the value should look
        # like the marker. If it says "redact", it must be the sentinel.
        # This is a sanity check on the redactor, not on the input.
        for field_name in ("email", "phone", "address"):
            value = getattr(self, field_name)
            if value is None:
                continue
            mode = getattr(self.pii_policy, field_name)
            if mode == "hash" and not _HASH_MARKER_RE.match(value):
                raise ValueError(
                    f"policy for {field_name!r} is 'hash' but value "
                    f"{value[:60]!r} does not match the shipped hash "
                    "marker shape; redactor drift?"
                )
            if mode == "redact" and value != REDACT_SENTINEL:
                raise ValueError(
                    f"policy for {field_name!r} is 'redact' but value is "
                    f"{value!r}, not the shipped sentinel {REDACT_SENTINEL!r}."
                )
        return self


class DuplicateGroup(BaseModel):
    """One merge event -- which canonical card_index absorbed which
    other card_indexes, and the human-readable reason."""

    canonical_index: int = Field(..., ge=0)
    member_indexes: list[int] = Field(..., min_length=2)
    merge_reason: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _canonical_is_a_member(self) -> DuplicateGroup:
        if self.canonical_index not in self.member_indexes:
            raise ValueError(
                f"canonical_index {self.canonical_index} must appear in "
                f"member_indexes {self.member_indexes!r}."
            )
        return self


class ExtractionResult(BaseModel):
    """Full pipeline return.

    `raw_contacts` are the pre-dedup RawContacts with PII already
    redacted per policy, so the on-disk `last_run.json` never leaks
    raw PII on strict/redacted runs. If the caller genuinely needs
    pre-redacted raw contacts, they call `_extract_raw()` directly
    rather than the public `extract_contacts()`.
    """

    raw_contacts: list[RawContact]
    contacts: list[Contact]
    duplicates: list[DuplicateGroup]
    pii_policy: PiiPolicy
    run_meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _raw_indexes_unique_and_dense(self) -> ExtractionResult:
        indexes = [r.card_index for r in self.raw_contacts]
        if len(set(indexes)) != len(indexes):
            raise ValueError(
                f"raw_contacts must have unique card_indexes; got {indexes!r}."
            )
        # We don't require dense [0, N) since the caller may have
        # filtered raws upstream, but out-of-range values in dedup
        # groups are checked below.
        return self

    @model_validator(mode="after")
    def _merged_from_partitions_raw_indexes(self) -> ExtractionResult:
        raw_ids = {r.card_index for r in self.raw_contacts}
        merged: list[int] = []
        for c in self.contacts:
            merged.extend(c.merged_from)
        if sorted(merged) != sorted(raw_ids):
            raise ValueError(
                f"Contact.merged_from across all contacts must partition "
                f"raw_contact card_indexes exactly. raw: {sorted(raw_ids)!r}; "
                f"merged: {sorted(merged)!r}."
            )
        if len(set(merged)) != len(merged):
            raise ValueError(
                f"one raw card_index was merged into more than one contact: "
                f"{merged!r}."
            )
        return self

    @model_validator(mode="after")
    def _duplicate_indexes_valid(self) -> ExtractionResult:
        raw_ids = {r.card_index for r in self.raw_contacts}
        for group in self.duplicates:
            for idx in group.member_indexes:
                if idx not in raw_ids:
                    raise ValueError(
                        f"DuplicateGroup references card_index {idx} which "
                        f"is not present in raw_contacts."
                    )
        return self
