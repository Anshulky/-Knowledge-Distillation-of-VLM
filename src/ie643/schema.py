"""Canonical extraction JSON for teacher labels, human review, and distillation.

The user draft (vendor, date, currency, line_items, subtotal, tax, total) is the
core. Extra keys exist because the collected datasets print them and because
invoice-style documents cannot be evaluated on a receipt-only contract.
"""

from __future__ import annotations

from typing import Any

DOC_TYPES = ("receipt", "invoice", "other")
LANGUAGES = ("en", "id", "ru", "de", "fr", "it", "ms", "ar", "mixed", "unknown")

LINE_ITEM_KEYS = ("name", "quantity", "unit_price", "amount")
TAX_ROW_KEYS = ("label", "rate", "amount")

FIELD_ORDER = (
    "document_type",
    "language",
    "vendor",
    "vendor_address",
    "vendor_phone",
    "vendor_tax_id",
    "buyer",
    "buyer_address",
    "date",
    "time",
    "due_date",
    "currency",
    "receipt_number",
    "invoice_number",
    "po_number",
    "line_items",
    "subtotal",
    "tax",
    "taxes",
    "service_charge",
    "discount",
    "delivery_fee",
    "tip",
    "total",
    "change",
    "payment_method",
)

CANONICAL_SCHEMA: dict[str, Any] = {
    "document_type": "receipt | invoice | other",
    "language": "en | id | ru | de | fr | it | ms | ar | mixed | unknown",
    "vendor": "string | null",
    "vendor_address": "string | null",
    "vendor_phone": "string | null",
    "vendor_tax_id": "string | null  # GSTIN / INN / VAT ID as printed",
    "buyer": "string | null",
    "buyer_address": "string | null",
    "date": "string | null  # keep the printed form",
    "time": "string | null",
    "due_date": "string | null",
    "currency": "string | null  # ISO-like: USD, EUR, IDR, MYR, EGP, RUB",
    "receipt_number": "string | null",
    "invoice_number": "string | null",
    "po_number": "string | null",
    "line_items": [
        {
            "name": "string | null",
            "quantity": "string | number | null",
            "unit_price": "string | number | null",
            "amount": "string | number | null  # line total as printed",
        }
    ],
    "subtotal": "string | number | null",
    "tax": "string | number | null  # total tax amount if a single figure is printed",
    "taxes": [
        {
            "label": "string | null",
            "rate": "string | number | null",
            "amount": "string | number | null",
        }
    ],
    "service_charge": "string | number | null",
    "discount": "string | number | null",
    "delivery_fee": "string | number | null",
    "tip": "string | number | null",
    "total": "string | number | null",
    "change": "string | number | null",
    "payment_method": "string | null",
}


def empty_item() -> dict[str, Any]:
    return {"name": None, "quantity": None, "unit_price": None, "amount": None}


def make_item(
    name: Any = None,
    quantity: Any = None,
    unit_price: Any = None,
    amount: Any = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "quantity": quantity,
        "unit_price": unit_price,
        "amount": amount,
    }


def empty_canonical(*, document_type: str = "receipt", language: str = "unknown") -> dict[str, Any]:
    doc = document_type if document_type in DOC_TYPES else "other"
    lang = language if language in LANGUAGES else "unknown"
    return {
        "document_type": doc,
        "language": lang,
        "vendor": None,
        "vendor_address": None,
        "vendor_phone": None,
        "vendor_tax_id": None,
        "buyer": None,
        "buyer_address": None,
        "date": None,
        "time": None,
        "due_date": None,
        "currency": None,
        "receipt_number": None,
        "invoice_number": None,
        "po_number": None,
        "line_items": [],
        "subtotal": None,
        "tax": None,
        "taxes": [],
        "service_charge": None,
        "discount": None,
        "delivery_fee": None,
        "tip": None,
        "total": None,
        "change": None,
        "payment_method": None,
    }


TEACHER_SYSTEM_PROMPT = """You extract fields from a receipt or invoice image.

Return a single JSON object with EXACTLY these keys, in this order:
document_type, language, vendor, vendor_address, vendor_phone, vendor_tax_id,
buyer, buyer_address, date, time, due_date, currency, receipt_number,
invoice_number, po_number, line_items, subtotal, tax, taxes, service_charge,
discount, delivery_fee, tip, total, change, payment_method.

Rules:
- A field is non-null only if that value is printed on the image. Do not invent.
- Keep dates, times, and numbers in the printed form. Do not convert 15/01/2019 to ISO. Do not strip thousand separators.
- document_type is receipt, invoice, or other.
- language is one of: en, id, ru, de, fr, it, ms, ar, mixed, unknown.
- currency is an ISO-like code (USD, EUR, GBP, IDR, MYR, EGP, RUB, INR) when it can be inferred from a symbol or printed code, else null.
- vendor is the merchant / seller / shop name. buyer is the billed customer (invoices).
- vendor_tax_id is GSTIN, INN, VAT number, or similar tax ID as printed. Copy it as printed; never pad or repeat zeros.
- line_items is a list of {name, quantity, unit_price, amount}. Use [] if none are readable. amount is the line total.
- taxes is a list of {label, rate, amount} for printed VAT/GST rows. Use [] if none. tax is the single total tax figure when printed.
- payment_method examples: cash, card, transfer, cod, mixed. Use null if not printed.
- Output JSON only. No markdown, no comments."""


def teacher_prompt(document_type_hint: str | None = None, language_hint: str | None = None) -> str:
    extra = []
    if document_type_hint:
        extra.append(f"This image is likely a {document_type_hint}.")
    if language_hint and language_hint not in {"unknown", None}:
        extra.append(f"Printed language hint: {language_hint}.")
    hint = (" " + " ".join(extra)) if extra else ""
    return TEACHER_SYSTEM_PROMPT + hint


def coerce_canonical(payload: dict[str, Any] | None, *, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Force a model/adapter dict onto the canonical key set."""
    out = empty_canonical()
    if fallback:
        out.update({k: fallback[k] for k in FIELD_ORDER if k in fallback})
    if not payload:
        return out
    aliases = {
        "doc_type": "document_type",
        "merchant_name": "vendor",
        "merchant": "vendor",
        "company": "vendor",
        "seller": "vendor",
        "vendor_name": "vendor",
        "merchant_address": "vendor_address",
        "address": "vendor_address",
        "merchant_phone": "vendor_phone",
        "merchant_tax_id": "vendor_tax_id",
        "inn": "vendor_tax_id",
        "gstin": "vendor_tax_id",
        "items": "line_items",
        "buyer_name": "buyer",
    }
    src = dict(payload)
    for old, new in aliases.items():
        if new not in src and old in src:
            src[new] = src[old]
    if src.get("document_type") in DOC_TYPES:
        out["document_type"] = src["document_type"]
    if src.get("language") in LANGUAGES:
        out["language"] = src["language"]
    scalar_keys = [k for k in FIELD_ORDER if k not in {"document_type", "language", "line_items", "taxes"}]
    for key in scalar_keys:
        if key in src:
            out[key] = src[key]
    items = src.get("line_items") or []
    if isinstance(items, dict):
        items = [items]
    clean_items = []
    for row in items:
        if not isinstance(row, dict):
            continue
        clean_items.append(
            make_item(
                name=row.get("name") or row.get("nm"),
                quantity=row.get("quantity") if row.get("quantity") is not None else row.get("qty") or row.get("cnt"),
                unit_price=row.get("unit_price") if row.get("unit_price") is not None else row.get("unitprice") or row.get("price"),
                amount=row.get("amount") if row.get("amount") is not None else row.get("total_price") or row.get("total") or row.get("sum") or row.get("price"),
            )
        )
    out["line_items"] = clean_items
    taxes = src.get("taxes") or []
    if isinstance(taxes, dict):
        taxes = [taxes]
    clean_taxes = []
    for row in taxes:
        if not isinstance(row, dict):
            continue
        clean_taxes.append(
            {
                "label": row.get("label"),
                "rate": row.get("rate"),
                "amount": row.get("amount"),
            }
        )
    out["taxes"] = clean_taxes
    return {k: out[k] for k in FIELD_ORDER}
