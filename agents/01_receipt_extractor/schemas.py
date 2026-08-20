"""Pydantic schema for structured receipt/invoice extraction.

The schema is the load-bearing part of this agent: Instructor uses it directly
as the response_model, and Claude vision fills it in from the image. Every
field's presence and description here is a prompt to the model. Keep them
concrete and specific; vague field descriptions produce vague extractions.

Schema kept deliberately minimal at v1. Fields will expand based on real
user feedback (R10 in CONTRIBUTING.md's hard rules: no invented scope).
What's here is what a bookkeeper needs to categorize a purchase; anything
more speculative waits for a real user asking for it.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """A single line on the receipt. `total` is required because if we can't
    extract the money amount for a line, we can't do anything useful with it.
    `quantity` and `unit_price` are optional because many receipts (esp.
    restaurants, retail totals) don't itemize them."""

    description: str = Field(
        ...,
        description="Item description as printed on the receipt (product name, service, etc.)",
    )
    quantity: float | None = Field(
        None, description="Number of units. Null if not printed."
    )
    unit_price: float | None = Field(
        None, description="Price per unit in the receipt's currency. Null if not printed."
    )
    total: float = Field(
        ...,
        description="Line total in the receipt's currency (what the customer pays for this line).",
    )


class ExtractedReceipt(BaseModel):
    """Structured extraction from a receipt or invoice image.

    Every field except `total` and `line_items` is optional because real-world
    receipts genuinely omit them (a coffee-shop receipt has no vendor tax ID;
    an invoice may have no due date). `total` and `line_items` are required
    because if we can't extract those, the extraction failed and the caller
    should surface an error rather than an empty object.
    """

    # --- Vendor ---
    vendor_name: str | None = Field(
        None, description="Business name of the seller (e.g. 'Blue Bottle Coffee')."
    )
    vendor_address: str | None = Field(
        None, description="Vendor's printed street address, if present."
    )
    vendor_tax_id: str | None = Field(
        None,
        description=(
            "Vendor's tax/registration ID as a string: GSTIN (India), VAT (EU), "
            "EIN (US), etc. Preserve the original format; do not normalize."
        ),
    )

    # --- Document identifiers ---
    invoice_number: str | None = Field(
        None,
        description="Invoice or receipt number as printed. May include letters and separators.",
    )
    invoice_date: date | None = Field(
        None,
        description="Date the invoice/receipt was issued (transaction date). Parse to ISO YYYY-MM-DD.",
    )
    due_date: date | None = Field(
        None,
        description="Payment due date for an invoice. Null for receipts (already paid).",
    )

    # --- Amounts ---
    currency: str | None = Field(
        None,
        description=(
            "ISO 4217 currency code where determinable (e.g. 'USD', 'EUR', 'INR', 'GBP'). "
            "If only a symbol is shown, infer conservatively; use null if genuinely unclear."
        ),
    )
    subtotal: float | None = Field(
        None, description="Sum of line items before tax. Null if not printed separately."
    )
    tax_total: float | None = Field(
        None,
        description="Total tax amount (VAT, GST, sales tax combined if multiple). Null if not printed.",
    )
    total: float = Field(
        ...,
        description=(
            "Final amount the customer paid or owes (subtotal + tax + fees - discounts). "
            "REQUIRED: this is the load-bearing number for expense tracking."
        ),
    )

    # --- Contents ---
    line_items: list[LineItem] = Field(
        default_factory=list,
        description=(
            "Individual purchased items/services. Empty list is allowed for receipts that "
            "only print a total (rare but real). Do not fabricate line items from totals."
        ),
    )

    # --- Meta ---
    payment_method: str | None = Field(
        None,
        description="Payment method as printed (e.g. 'Visa ****1234', 'Cash', 'UPI', 'Amex').",
    )
    notes: str | None = Field(
        None,
        description=(
            "Anything unusual worth surfacing to the user: 'partial payment', "
            "'reprint of original', discounts applied, tips, service charges, etc."
        ),
    )
