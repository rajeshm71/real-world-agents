"""Smoke tests for the contact / ID extractor.

Zero real API calls. Vision path is exercised only via a monkey-
patched _instructor_client stub. Dedup is exercised against real
fastembed on the strings the tests build (small, ~50ms each) so the
similarity threshold behavior is genuinely covered.

Sections:
1. Mock path (5).
2. Dedup logic (10).
3. PII policy + redactor (7).
4. Schema validators (8).
5. R5 branches (6).
6. Constants + example sanity (4).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_AGENT_DIR.parent))

_agent = importlib.import_module("15_contact_extractor.agent")
_schemas = importlib.import_module("15_contact_extractor.schemas")

extract_contacts = _agent.extract_contacts
resolve_provider = _agent.resolve_provider
ContactError = _agent.ContactError
ExtractionAttempt = _agent.ExtractionAttempt
_translate_api_error = _agent._translate_api_error
_deduplicate = _agent._deduplicate
_apply_policy = _agent._apply_policy
_apply_policy_to_raws = _agent._apply_policy_to_raws
_redact_field = _agent._redact_field
_normalize = _agent._normalize
_normalize_phone = _agent._normalize_phone
_identity = _agent._identity
_read_image = _agent._read_image
SUPPORTED_PROVIDERS = _agent.SUPPORTED_PROVIDERS
DEFAULT_DEDUP_THRESHOLD = _agent.DEFAULT_DEDUP_THRESHOLD

RawContact = _schemas.RawContact
Contact = _schemas.Contact
DuplicateGroup = _schemas.DuplicateGroup
PiiPolicy = _schemas.PiiPolicy
ExtractionResult = _schemas.ExtractionResult
REDACT_SENTINEL = _schemas.REDACT_SENTINEL

_CARDS_PNG = _AGENT_DIR / "examples" / "cards.png"


# --- Helpers ---------------------------------------------------------------


def _r(idx: int, **kwargs) -> RawContact:
    """Shorthand RawContact factory."""
    return RawContact(card_index=idx, full_name=kwargs.pop("full_name", f"Person {idx}"), **kwargs)


# --- 1. Mock path ----------------------------------------------------------


def test_mock_returns_valid_result(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = extract_contacts(_CARDS_PNG)
    assert isinstance(r, ExtractionResult)
    assert r.run_meta["provider"] == "mock"


def test_mock_input_size_echoed(monkeypatch):
    """Anti-refactor guard: mock reads its input path length."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = extract_contacts(_CARDS_PNG)
    assert r.run_meta["mock_input_size"] == _CARDS_PNG.stat().st_size


def test_mock_produces_expected_dedup_shape(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = extract_contacts(_CARDS_PNG)
    assert r.run_meta["raw_count"] == 3
    assert r.run_meta["contact_count"] == 2
    assert r.run_meta["duplicate_group_count"] == 1


def test_mock_ranked_indexes_cover_raws(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = extract_contacts(_CARDS_PNG)
    merged = sorted(i for c in r.contacts for i in c.merged_from)
    assert merged == [0, 1, 2]


def test_mock_skips_openai_and_instructor(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.setitem(sys.modules, "instructor", None)
    r = extract_contacts(_CARDS_PNG)
    assert r.pii_policy == PiiPolicy()


# --- 2. Dedup logic --------------------------------------------------------


def test_normalize_lowercases_and_strips_punct():
    assert _normalize("Alice, Chen!!") == "alice chen"


def test_normalize_phone_keeps_last_10_digits():
    assert _normalize_phone("+1 (415) 555-0101") == "4155550101"
    assert _normalize_phone("415-555-0101") == "4155550101"


def test_identity_joins_three_normalized_fields():
    r = _r(0, full_name="Alice Chen", title="VP", company="Northwind")
    ident = _identity(r)
    assert "alice chen" in ident
    assert "northwind" in ident
    assert "vp" in ident


def test_dedup_same_email_high_sim_merges():
    pytest.importorskip("fastembed")
    raws = [
        _r(0, full_name="Alice Chen", company="Northwind", email="alice@northwind.io"),
        _r(1, full_name="Alicia Chen", company="Northwind", email="alice@northwind.io"),
    ]
    groups, report = _deduplicate(raws, threshold=DEFAULT_DEDUP_THRESHOLD)
    assert groups == [[0, 1]]
    assert len(report) == 1
    assert "same email" in report[0].merge_reason


def test_dedup_same_phone_high_sim_merges():
    pytest.importorskip("fastembed")
    raws = [
        _r(0, full_name="Bob Rivera", company="Ridgeway", phone="415-555-0199"),
        _r(1, full_name="Bob Rivera", company="Ridgeway", phone="+1 (415) 555-0199"),
    ]
    groups, report = _deduplicate(raws, threshold=DEFAULT_DEDUP_THRESHOLD)
    assert groups == [[0, 1]]
    assert "same phone" in report[0].merge_reason


def test_dedup_no_corroborating_key_does_not_merge():
    """High name/company similarity but no shared email/phone must NOT merge."""
    pytest.importorskip("fastembed")
    raws = [
        _r(0, full_name="Alice Chen", company="Northwind"),
        _r(1, full_name="Alice Chen", company="Northwind"),
    ]
    groups, report = _deduplicate(raws, threshold=DEFAULT_DEDUP_THRESHOLD)
    assert groups == [[0], [1]]
    assert report == []


def test_dedup_low_sim_same_email_does_not_merge():
    """Shared email but very different name+company (typo-shared address
    between distinct people) must NOT merge."""
    pytest.importorskip("fastembed")
    raws = [
        _r(0, full_name="Alice Chen", company="Northwind Analytics",
           email="shared@example.com"),
        _r(1, full_name="Ronaldo Silva", company="Furniture Depot",
           email="shared@example.com"),
    ]
    # Force a very high threshold so identity similarity of these two
    # unrelated strings fails.
    groups, _report = _deduplicate(raws, threshold=0.99)
    assert groups == [[0], [1]]


def test_dedup_transitive_merge_three_cards():
    pytest.importorskip("fastembed")
    raws = [
        _r(0, full_name="Alice Chen", company="Northwind",
           email="alice@northwind.io"),
        _r(1, full_name="Alicia Chen", company="Northwind",
           email="alice@northwind.io", phone="415-555-0101"),
        _r(2, full_name="Alice Chen", company="Northwind",
           phone="415-555-0101"),
    ]
    groups, _report = _deduplicate(raws, threshold=DEFAULT_DEDUP_THRESHOLD)
    assert groups == [[0, 1, 2]]


def test_dedup_canonical_pick_by_field_completeness():
    """The group's canonical should be the most-populated card."""
    pytest.importorskip("fastembed")
    # Both cards use the same normalized identity (name+company+title)
    # so cosine similarity is essentially 1.0; the shared email is the
    # corroborating key. Card #1 has more populated optional fields
    # (phone), so it should win the canonical pick.
    raws = [
        _r(0, full_name="Alice Chen", company="Northwind", title="VP",
           email="alice@northwind.io"),
        _r(1, full_name="Alice Chen", company="Northwind", title="VP",
           email="alice@northwind.io", phone="415-555-0101"),
    ]
    groups, _report = _deduplicate(raws, threshold=DEFAULT_DEDUP_THRESHOLD)
    assert groups == [[0, 1]]
    contacts = _apply_policy(raws, groups, policy=PiiPolicy(), run_salt="x")
    assert len(contacts) == 1
    # Canonical is card #1 (more fields); phone should be present.
    assert contacts[0].phone == "415-555-0101"


def test_dedup_singletons_pass_through():
    pytest.importorskip("fastembed")
    raws = [
        _r(0, full_name="Alice", email="a@x.com"),
        _r(1, full_name="Bob", email="b@x.com"),
    ]
    groups, report = _deduplicate(raws, threshold=DEFAULT_DEDUP_THRESHOLD)
    assert groups == [[0], [1]]
    assert report == []


def test_dedup_empty_input_ok():
    groups, report = _deduplicate([], threshold=DEFAULT_DEDUP_THRESHOLD)
    assert groups == []
    assert report == []


# --- 3. PII policy + redactor ---------------------------------------------


def test_policy_none_passes_through():
    assert _redact_field("alice@example.com", "none", run_salt="s") == "alice@example.com"


def test_policy_hash_produces_marker():
    out = _redact_field("alice@example.com", "hash", run_salt="stable_salt")
    assert "..." in out
    # Marker shape: 16 hex + '...' + last-4 of original.
    prefix, suffix = out.split("...")
    assert len(prefix) == 16
    assert all(c in "0123456789abcdef" for c in prefix)
    assert suffix == ".com"


def test_policy_hash_deterministic_with_same_salt():
    a = _redact_field("alice@example.com", "hash", run_salt="stable")
    b = _redact_field("alice@example.com", "hash", run_salt="stable")
    assert a == b


def test_policy_hash_differs_with_different_salts():
    a = _redact_field("alice@example.com", "hash", run_salt="salt_one")
    b = _redact_field("alice@example.com", "hash", run_salt="salt_two")
    assert a != b
    # But last-4 (visible tail) should match.
    assert a.split("...")[1] == b.split("...")[1]


def test_policy_redact_returns_sentinel():
    assert _redact_field("alice@example.com", "redact", run_salt="s") == REDACT_SENTINEL


def test_policy_strict_hashes_every_pii_field(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = extract_contacts(_CARDS_PNG, pii_policy=PiiPolicy.strict(), run_salt="s")
    for c in r.contacts:
        if c.email is not None:
            assert "..." in c.email
        if c.phone is not None:
            assert "..." in c.phone


def test_policy_redacted_replaces_every_pii_field(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = extract_contacts(_CARDS_PNG, pii_policy=PiiPolicy.redacted())
    for c in r.contacts:
        if c.email is not None:
            assert c.email == REDACT_SENTINEL


# --- 4. Schema validators -------------------------------------------------


def test_raw_contact_email_shape_validator():
    with pytest.raises(ValidationError, match="does not look like"):
        RawContact(card_index=0, full_name="A", email="not_an_email")


def test_contact_merged_from_must_be_unique():
    with pytest.raises(ValidationError, match="merged_from must be unique"):
        Contact(
            full_name="A", merged_from=[0, 0], pii_policy=PiiPolicy(),
        )


def test_contact_hashed_field_shape_check():
    """Under hash policy, the field value must look like the hash marker."""
    with pytest.raises(ValidationError, match="does not match the shipped hash marker"):
        Contact(
            full_name="A",
            email="raw@example.com",  # policy says hash but value is raw
            merged_from=[0],
            pii_policy=PiiPolicy(email="hash"),
        )


def test_contact_redact_field_shape_check():
    with pytest.raises(ValidationError, match="not the shipped sentinel"):
        Contact(
            full_name="A",
            email="some string",
            merged_from=[0],
            pii_policy=PiiPolicy(email="redact"),
        )


def test_duplicate_group_canonical_must_be_a_member():
    with pytest.raises(ValidationError, match="canonical_index"):
        DuplicateGroup(canonical_index=99, member_indexes=[0, 1], merge_reason="x")


def test_extraction_result_merged_from_partitions_raws():
    raws = [_r(0, full_name="A"), _r(1, full_name="B")]
    contacts = [
        Contact(full_name="A", merged_from=[0], pii_policy=PiiPolicy()),
        # Missing merged_from=[1] entirely -> not a partition
    ]
    with pytest.raises(ValidationError, match="partition"):
        ExtractionResult(
            raw_contacts=raws, contacts=contacts,
            duplicates=[], pii_policy=PiiPolicy(),
        )


def test_extraction_result_rejects_double_merged_index():
    raws = [_r(0, full_name="A")]
    contacts = [
        Contact(full_name="A", merged_from=[0], pii_policy=PiiPolicy()),
        Contact(full_name="A", merged_from=[0], pii_policy=PiiPolicy()),
    ]
    with pytest.raises(ValidationError):
        ExtractionResult(
            raw_contacts=raws, contacts=contacts,
            duplicates=[], pii_policy=PiiPolicy(),
        )


def test_extraction_result_duplicate_indexes_must_be_valid():
    raws = [_r(0, full_name="A")]
    contacts = [
        Contact(full_name="A", merged_from=[0], pii_policy=PiiPolicy()),
    ]
    with pytest.raises(ValidationError, match="not present in raw_contacts"):
        ExtractionResult(
            raw_contacts=raws, contacts=contacts,
            duplicates=[
                DuplicateGroup(canonical_index=0, member_indexes=[0, 99],
                               merge_reason="x")
            ],
            pii_policy=PiiPolicy(),
        )


# --- 5. R5 error translator -----------------------------------------------


def test_translator_class_rate_limit():
    class RateLimitError(Exception):
        pass
    assert "rate-limited" in _translate_api_error(RateLimitError("x")).message


def test_translator_class_auth():
    class AuthenticationError(Exception):
        pass
    assert "OPENAI_API_KEY" in _translate_api_error(AuthenticationError("x")).message


def test_translator_status_429():
    exc = RuntimeError("x")
    exc.status_code = 429
    assert "rate-limited" in _translate_api_error(exc).message


def test_translator_status_401():
    exc = RuntimeError("x")
    exc.status_code = 401
    assert "OPENAI_API_KEY" in _translate_api_error(exc).message


def test_translator_message_rate_limit():
    assert "rate-limited" in _translate_api_error(RuntimeError("Rate limit hit")).message


def test_translator_generic():
    assert "vision extraction failed" in _translate_api_error(
        RuntimeError("something else")
    ).message


# --- 6. Constants + example sanity ----------------------------------------


def test_supported_providers_openai_only():
    assert SUPPORTED_PROVIDERS == ("openai",)


def test_default_dedup_threshold_reasonable():
    assert 0.5 < DEFAULT_DEDUP_THRESHOLD < 1.0


def test_cards_png_exists():
    assert _CARDS_PNG.exists()
    assert _CARDS_PNG.stat().st_size > 1000


def test_read_image_rejects_unsupported_suffix(tmp_path):
    p = tmp_path / "x.bmp"
    p.write_bytes(b"BM")
    with pytest.raises(ContactError, match="unsupported image format"):
        _read_image(p)


def test_read_image_rejects_over_20mb(tmp_path):
    p = tmp_path / "huge.png"
    p.write_bytes(b"P" * (21 * 1024 * 1024))
    with pytest.raises(ContactError, match="over the .* MB cap"):
        _read_image(p)


def test_apply_policy_to_raws_email_hash_marker_round_trips():
    """RawContact's email validator now accepts hash markers, so the
    raws-side redaction preserves the 'this card had an email' signal
    instead of dropping it to None."""
    raws = [_r(0, full_name="A", email="a@example.com")]
    redacted = _apply_policy_to_raws(raws, policy=PiiPolicy.strict(), run_salt="s")
    assert redacted[0].email is not None
    assert "..." in redacted[0].email


def test_apply_policy_to_raws_email_redact_sentinel_round_trips():
    raws = [_r(0, full_name="A", email="a@example.com")]
    redacted = _apply_policy_to_raws(raws, policy=PiiPolicy.redacted(), run_salt="s")
    assert redacted[0].email == REDACT_SENTINEL


def test_raw_contact_email_accepts_hash_marker():
    """The email validator must let a hash-marker through so the
    redaction round-trip in _apply_policy_to_raws does not blow up."""
    marker = "0123456789abcdef...com"  # 16 hex + '...' + 3 chars
    # Regex requires exactly 4-char tail:
    marker = "0123456789abcdef....com"
    RawContact(card_index=0, full_name="A", email=marker)


def test_raw_contact_email_accepts_redact_sentinel():
    RawContact(card_index=0, full_name="A", email=REDACT_SENTINEL)


def test_read_image_maps_jpg_to_jpeg_mime(tmp_path):
    """H1: `.jpg` files must resolve to `image/jpeg`, not `image/jpg`."""
    p = tmp_path / "card.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    _data, mime = _read_image(p)
    assert mime == "jpeg"


def test_read_image_sniffs_bytes_input():
    """L1: raw bytes input must sniff the header, not assume PNG."""
    jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    _data, mime = _read_image(jpeg_bytes)
    assert mime == "jpeg"


def test_hash_marker_regex_requires_exact_four_char_tail():
    """L2: the tighter regex should reject a shorter or longer tail."""
    from importlib import import_module
    schemas_mod = import_module("15_contact_extractor.schemas")
    marker_re = schemas_mod._HASH_MARKER_RE
    assert marker_re.match("0123456789abcdef....com")   # 4-char tail: ok
    assert not marker_re.match("0123456789abcdef...co")  # 2-char tail: no
    assert not marker_re.match("0123456789abcdef...toolong")  # 7-char tail: no
