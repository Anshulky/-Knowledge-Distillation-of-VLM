"""Receipt prompt and JSON parse for the Gradio Space.

Uses ie643 when this folder sits inside the training repo. The copies below are
what Hugging Face runs after `space/` is uploaded on its own.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)

_PROMPT = """You extract fields from a receipt or invoice image.

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


def _fallback_prompt(document_type_hint: str | None = None, language_hint: str | None = None) -> str:
    extra = []
    if document_type_hint:
        extra.append(f"This image is likely a {document_type_hint}.")
    if language_hint and language_hint not in {"unknown", None}:
        extra.append(f"Printed language hint: {language_hint}.")
    hint = (" " + " ".join(extra)) if extra else ""
    return _PROMPT + hint


def _fallback_parse(text: str) -> dict[str, Any]:
    raw = _FENCE.sub("", (text or "").strip()).strip()
    if not raw:
        raise ValueError("empty model output")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        i, j = raw.find("{"), raw.rfind("}")
        if i < 0 or j <= i:
            raise
        payload = json.loads(raw[i : j + 1])
    if not isinstance(payload, dict):
        raise ValueError("model output was not a JSON object")
    return payload


def _load():
    src = Path(__file__).resolve().parents[1] / "src"
    if (src / "ie643").is_dir():
        sys.path.insert(0, str(src))
        try:
            from ie643.schema import teacher_prompt
            from ie643.teacher.parse import parse_teacher_output

            return teacher_prompt, parse_teacher_output
        except ImportError:
            pass
    return _fallback_prompt, _fallback_parse


teacher_prompt, parse_output = _load()
