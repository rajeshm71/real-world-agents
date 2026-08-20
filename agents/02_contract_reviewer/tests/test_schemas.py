"""Schema unit tests for the contract reviewer.

F2.1 scope: schemas only. Behavioral tests for agent.py's retry loop,
R5 error branches, and the excerpt-substring validator live in
test_smoke.py, added at F2.2/F2.4.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# `02_contract_reviewer` starts with a digit -- invalid Python identifier
# for normal import syntax. Same importlib bootstrap as
# agents/01_receipt_extractor/tests/test_smoke.py.
_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AGENT_DIR.parent))

_schemas = importlib.import_module("02_contract_reviewer.schemas")
ContractFlag = _schemas.ContractFlag
ContractReview = _schemas.ContractReview


# --- ContractFlag ----------------------------------------------------------


def test_flag_minimal_valid_construction():
    flag = ContractFlag(
        category="auto_renewal",
        severity="medium",
        excerpt="This Agreement shall automatically renew for successive 12-month terms.",
        explanation="You'll be locked into another year unless you cancel 60 days before renewal.",
    )
    assert flag.category == "auto_renewal"
    assert flag.severity == "medium"
    assert flag.page_number is None
    assert flag.recommendation is None


def test_flag_rejects_invented_category():
    """Closed Literal is the whole point -- if this passes the model can
    invent private category labels that break downstream consumers."""
    with pytest.raises(ValidationError):
        ContractFlag(
            category="weird_thing",  # not in FlagCategory
            severity="low",
            excerpt="x",
            explanation="y",
        )


def test_flag_rejects_invented_severity():
    with pytest.raises(ValidationError):
        ContractFlag(
            category="other",
            severity="critical",  # not in Severity
            excerpt="x",
            explanation="y",
        )


def test_flag_rejects_empty_excerpt():
    """An empty excerpt would break the user's 'locate the clause' workflow.
    Pydantic's min_length=1 is the cheap guard; agent.py additionally
    requires excerpt to be a substring of source text (see [C5])."""
    with pytest.raises(ValidationError):
        ContractFlag(
            category="other",
            severity="low",
            excerpt="",
            explanation="y",
        )


def test_flag_rejects_empty_explanation():
    with pytest.raises(ValidationError):
        ContractFlag(
            category="other",
            severity="low",
            excerpt="x",
            explanation="",
        )


def test_flag_rejects_zero_or_negative_page_number():
    """page_number is 1-indexed (matches how pypdf numbers pages and how
    users would think about them). ge=1 rejects 0 and negatives."""
    with pytest.raises(ValidationError):
        ContractFlag(
            category="other",
            severity="low",
            excerpt="x",
            explanation="y",
            page_number=0,
        )
    with pytest.raises(ValidationError):
        ContractFlag(
            category="other",
            severity="low",
            excerpt="x",
            explanation="y",
            page_number=-3,
        )


def test_flag_accepts_all_documented_categories():
    """Regression guard: if a category is removed from the Literal, this
    test surfaces which one, rather than a distant test failing cryptically."""
    for cat in [
        "auto_renewal",
        "unlimited_liability",
        "unilateral_termination",
        "arbitration",
        "non_compete",
        "ip_assignment",
        "indemnification",
        "exclusivity",
        "governing_law",
        "confidentiality",
        "other",
    ]:
        ContractFlag(
            category=cat,
            severity="low",
            excerpt="x",
            explanation="y",
        )


def test_flag_accepts_all_severities():
    for sev in ["low", "medium", "high"]:
        ContractFlag(
            category="other",
            severity=sev,
            excerpt="x",
            explanation="y",
        )


# --- ContractReview --------------------------------------------------------


def test_review_minimal_valid_construction():
    review = ContractReview(summary="This is a mutual NDA with a 3-year term.", overall_risk="low")
    assert review.contract_type is None
    assert review.parties == []
    assert review.flags == []
    assert review.overall_risk == "low"


def test_review_rejects_empty_summary():
    with pytest.raises(ValidationError):
        ContractReview(summary="", overall_risk="low")


def test_review_empty_flags_list_is_valid():
    """A contract with zero flagged clauses is a real, valid result -- some
    contracts genuinely have no high-risk clauses. Regression guard against
    a future field-level constraint that would reject an empty list."""
    review = ContractReview(summary="Boilerplate NDA, nothing unusual.", overall_risk="low", flags=[])
    assert review.flags == []


def test_review_rejects_invented_overall_risk():
    with pytest.raises(ValidationError):
        ContractReview(summary="x", overall_risk="critical")


def test_review_serializable_to_and_from_json():
    """Round-trips cleanly -- the mock extractor path, the retry loop, and
    the UI all depend on this."""
    original = ContractReview(
        contract_type="Mutual NDA",
        parties=["Acme, Inc.", "Widgets LLC"],
        flags=[
            ContractFlag(
                category="auto_renewal",
                severity="medium",
                excerpt="successive 12-month terms",
                page_number=2,
                explanation="Renews automatically unless cancelled 60 days out.",
                recommendation="Ask for cancel-anytime with 30 days notice.",
            )
        ],
        summary="Mutual NDA with an auto-renewal trap.",
        overall_risk="medium",
    )
    dumped = original.model_dump_json()
    restored = ContractReview.model_validate_json(dumped)
    assert restored == original
