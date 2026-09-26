"""Map heterogeneous dataset labels into the canonical schema."""

from __future__ import annotations

import json
import re
from typing import Any

from ie643.schema import coerce_canonical, empty_canonical, make_item

_CUR_CODE = re.compile(r"\b(USD|EUR|GBP|INR|IDR|MYR|EGP|RUB|SGD|AUD|CAD|JPY|CNY|CHF|THB|VND)\b", re.I)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict) and "text" in value:
        value = value["text"]
    s = str(value).strip()
    return s or None


def _currency_from(*values: Any) -> str | None:
    for value in values:
        s = _text(value)
        if not s:
            continue
        m = _CUR_CODE.search(s)
        if m:
            return m.group(1).upper()
        if "$" in s:
            return "USD"
        if "€" in s:
            return "EUR"
        if "£" in s:
            return "GBP"
        if "₽" in s or "руб" in s.lower():
            return "RUB"
        if "rp" in s.lower() or "idr" in s.lower():
            return "IDR"
    return None


def from_sroie(payload: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    out = empty_canonical(document_type="receipt", language="en")
    out["vendor"] = _text(payload.get("company"))
    out["vendor_address"] = _text(payload.get("address"))
    out["date"] = _text(payload.get("date"))
    out["total"] = _text(payload.get("total"))
    out["currency"] = "MYR"
    return coerce_canonical(out)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        merged: dict[str, Any] = {}
        for row in value:
            if isinstance(row, dict):
                merged.update(row)
        return merged
    return {}


def from_cord(ground_truth: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    gp = ground_truth.get("gt_parse") or ground_truth
    out = empty_canonical(document_type="receipt", language="id")
    menu = gp.get("menu") or []
    if isinstance(menu, dict):
        menu = [menu]
    items = []

    def _push(row: dict[str, Any]) -> None:
        items.append(
            make_item(
                name=_text(row.get("nm")),
                quantity=_text(row.get("cnt") or row.get("num")),
                unit_price=_text(row.get("unitprice")),
                amount=_text(row.get("price") or row.get("itemsubtotal")),
            )
        )
        extras = row.get("sub") or []
        if isinstance(extras, dict):
            extras = [extras]
        for sub in extras:
            if isinstance(sub, dict):
                _push(sub)

    for row in menu:
        if isinstance(row, dict):
            _push(row)
    out["line_items"] = items
    sub = _as_dict(gp.get("sub_total"))
    tot = _as_dict(gp.get("total"))
    out["subtotal"] = _text(sub.get("subtotal_price"))
    out["tax"] = _text(sub.get("tax_price"))
    out["service_charge"] = _text(sub.get("service_price"))
    out["discount"] = _text(sub.get("discount_price"))
    out["total"] = _text(tot.get("total_price"))
    out["change"] = _text(tot.get("changeprice"))
    if tot.get("creditcardprice"):
        out["payment_method"] = "card"
    elif tot.get("emoneyprice"):
        out["payment_method"] = "emoney"
    elif tot.get("cashprice"):
        out["payment_method"] = "cash"
    out["currency"] = "IDR"
    return coerce_canonical(out)


def from_receiptqa(qa_list: list[dict[str, Any]]) -> dict[str, Any]:
    out = empty_canonical(document_type="receipt", language="en")
    out["currency"] = "EGP"
    for item in qa_list:
        q = str(item.get("question", "")).lower()
        a = item.get("answer")
        if a in (None, "Not Available", ["Not Available"]):
            continue
        if isinstance(a, list):
            continue
        a = _text(a)
        if not a:
            continue
        if "company" in q or "merchant" in q or "placed with" in q:
            out["vendor"] = out["vendor"] or a
        elif "delivery fee" in q or "delivery charge" in q:
            out["delivery_fee"] = out["delivery_fee"] or a
        elif "subtotal" in q:
            out["subtotal"] = out["subtotal"] or a
        elif "payment" in q:
            out["payment_method"] = out["payment_method"] or a
        elif "date" in q:
            out["date"] = out["date"] or a
        elif "total" in q and "item" not in q and "delivery" not in q:
            out["total"] = out["total"] or a
    out["_qa_pairs"] = qa_list
    return out


def from_cdek(row: dict[str, Any]) -> dict[str, Any]:
    out = empty_canonical(document_type="receipt", language="ru")
    out["vendor"] = _text((row.get("seller") or {}).get("text") if isinstance(row.get("seller"), dict) else row.get("seller"))
    inn = row.get("inn")
    out["vendor_tax_id"] = _text(inn.get("text") if isinstance(inn, dict) else inn)
    date = row.get("date")
    out["date"] = _text(date.get("text") if isinstance(date, dict) else date)
    total = row.get("total")
    out["total"] = _text(total.get("text") if isinstance(total, dict) else total)
    out["currency"] = "RUB"
    items = []
    for it in row.get("items") or []:
        items.append(
            make_item(
                name=_text(it.get("text") or it.get("name")),
                quantity=_text(it.get("quantity")),
                unit_price=_text(it.get("price")),
                amount=_text(it.get("sum")),
            )
        )
    out["line_items"] = items
    return coerce_canonical(out)


def from_docjay(text: str | dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(text) if isinstance(text, str) else text
    out = empty_canonical(document_type="receipt", language="en")
    out["vendor"] = _text(payload.get("merchantName"))
    out["vendor_address"] = _text(payload.get("merchantAddress"))
    out["date"] = _text(payload.get("date"))
    out["time"] = _text(payload.get("time"))
    out["receipt_number"] = _text(payload.get("receiptNumber"))
    out["subtotal"] = payload.get("subtotal")
    out["tax"] = payload.get("tax")
    out["tip"] = payload.get("tip")
    out["total"] = payload.get("total")
    out["payment_method"] = _text(payload.get("paymentMethod"))
    out["currency"] = _currency_from(payload.get("total"), payload.get("subtotal"))
    items = []
    for it in payload.get("items") or []:
        items.append(
            make_item(
                name=_text(it.get("name")),
                quantity=it.get("quantity"),
                unit_price=it.get("unitPrice"),
                amount=it.get("totalPrice"),
            )
        )
    out["line_items"] = items
    return coerce_canonical(out)


def from_fatura(ann: dict[str, Any]) -> dict[str, Any]:
    out = empty_canonical(document_type="invoice", language="en")

    def field(*keys: str) -> str | None:
        for key in keys:
            t = _text(ann.get(key))
            if t:
                return t
        return None

    out["vendor"] = field("SELLER_NAME")
    out["vendor_address"] = field("SELLER_ADDRESS")
    out["vendor_tax_id"] = field("GSTIN_SELLER", "GSTIN")
    out["buyer"] = field("BUYER", "BILL_TO")
    out["date"] = field("DATE")
    out["due_date"] = field("DUE_DATE")
    out["invoice_number"] = field("NUMBER", "INVOICE_INFO")
    out["po_number"] = field("PO_NUMBER")
    out["subtotal"] = field("SUB_TOTAL")
    out["tax"] = field("TAX", "GST(18%)", "GST(12%)", "GST(5%)")
    out["discount"] = field("DISCOUNT")
    out["total"] = field("TOTAL", "AMOUNT_DUE")
    out["payment_method"] = field("PAYMENT_DETAILS")
    out["currency"] = _currency_from(out["total"], out["subtotal"], out["tax"])
    title = field("TITLE")
    if title and "invoice" in title.lower():
        out["document_type"] = "invoice"
    return coerce_canonical(out)


def from_voxel51(json_annotation: str | dict[str, Any] | None) -> dict[str, Any]:
    out = empty_canonical(document_type="invoice", language="en")
    if not json_annotation:
        return out
    payload = json.loads(json_annotation) if isinstance(json_annotation, str) else json_annotation
    inv = payload.get("invoice") or {}
    out["vendor"] = _text(inv.get("seller_name"))
    out["vendor_address"] = _text(inv.get("seller_address"))
    out["buyer"] = _text(inv.get("client_name"))
    out["buyer_address"] = _text(inv.get("client_address"))
    out["invoice_number"] = _text(inv.get("invoice_number"))
    out["date"] = _text(inv.get("invoice_date"))
    out["due_date"] = _text(inv.get("due_date"))
    sub = payload.get("subtotal") or {}
    out["tax"] = _text(sub.get("tax"))
    out["discount"] = _text(sub.get("discount"))
    out["total"] = _text(sub.get("total"))
    out["subtotal"] = _text(sub.get("subtotal") or sub.get("sub_total"))
    pay = payload.get("payment_instructions") or {}
    out["payment_method"] = _text(pay.get("payment_method"))
    out["currency"] = _currency_from(out["total"], out["tax"])
    items = []
    for it in payload.get("items") or []:
        items.append(
            make_item(
                name=_text(it.get("description") or it.get("name")),
                quantity=_text(it.get("quantity")),
                unit_price=_text(it.get("unit_price") or it.get("unit_cost")),
                amount=_text(it.get("total_price") or it.get("amount")),
            )
        )
    out["line_items"] = items
    return coerce_canonical(out)


def from_synthetic(fields: str | dict[str, Any], full_text: str | None = None) -> dict[str, Any]:
    payload = json.loads(fields) if isinstance(fields, str) else fields
    locale = str(payload.get("locale") or "US")
    lang = {"US": "en", "UK": "en", "DE": "de", "FR": "fr", "IT": "it"}.get(locale, "en")
    out = empty_canonical(document_type="receipt", language=lang)
    out["vendor"] = _text(payload.get("merchant"))
    out["vendor_address"] = _text(payload.get("address"))
    out["vendor_phone"] = _text(payload.get("phone"))
    out["vendor_tax_id"] = _text(payload.get("tax_id"))
    out["date"] = _text(payload.get("date"))
    out["time"] = _text(payload.get("time"))
    out["currency"] = _text(payload.get("currency"))
    out["receipt_number"] = _text(payload.get("receipt_no"))
    out["subtotal"] = payload.get("subtotal")
    taxes = []
    for row in payload.get("taxes") or []:
        if not isinstance(row, dict):
            continue
        taxes.append(
            {
                "label": _text(row.get("label")),
                "rate": row.get("rate"),
                "amount": row.get("amount"),
            }
        )
    out["taxes"] = taxes
    if taxes:
        out["tax"] = taxes[0].get("amount")
    out["total"] = payload.get("total")
    out["payment_method"] = _text(payload.get("payment"))
    items = []
    for it in payload.get("lines") or []:
        items.append(
            make_item(
                name=_text(it.get("name")),
                quantity=it.get("qty"),
                unit_price=it.get("unit_price"),
                amount=it.get("total"),
            )
        )
    out["line_items"] = items
    if full_text:
        out["_full_text"] = full_text
    return coerce_canonical(out)


def _unwrap_json(value: Any) -> Any:
    cur = value
    for _ in range(3):
        if isinstance(cur, (bytes, bytearray)):
            cur = cur.decode("utf-8")
        if isinstance(cur, str):
            s = cur.strip()
            if not s:
                return {}
            try:
                cur = json.loads(s)
            except json.JSONDecodeError:
                return cur
            continue
        return cur
    return cur


def from_mychen_v2(parsed_data: str | dict[str, Any]) -> dict[str, Any]:
    payload = _unwrap_json(parsed_data)
    js = payload.get("json") if isinstance(payload, dict) else payload
    js = _unwrap_json(js)
    if not isinstance(js, dict):
        js = {}
    out = empty_canonical(document_type="receipt", language="en")
    out["vendor"] = _text(js.get("store_name") or js.get("merchant") or js.get("vendor"))
    out["vendor_address"] = _text(js.get("store_addr") or js.get("address"))
    out["vendor_phone"] = _text(js.get("telephone") or js.get("phone"))
    out["date"] = _text(js.get("date"))
    out["time"] = _text(js.get("time"))
    out["subtotal"] = _text(js.get("subtotal"))
    out["tax"] = _text(js.get("tax"))
    out["tip"] = _text(js.get("tips") or js.get("tip"))
    out["total"] = _text(js.get("total"))
    out["currency"] = _currency_from(js.get("total"), js.get("subtotal"))
    items = []
    raw_items = js.get("line_items") or js.get("items") or []
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    for it in raw_items:
        if not isinstance(it, dict):
            continue
        items.append(
            make_item(
                name=_text(it.get("item_name") or it.get("name") or it.get("nm")),
                quantity=_text(it.get("item_quantity") or it.get("cnt") or it.get("quantity")),
                unit_price=_text(it.get("unit_price") or it.get("price")),
                amount=_text(it.get("itemsubtotal") or it.get("total") or it.get("item_value") or it.get("amount")),
            )
        )
    out["line_items"] = items
    return coerce_canonical(out)


def strip_private(canonical: dict[str, Any]) -> dict[str, Any]:
    """Drop adapter-only keys before writing teacher input."""
    from ie643.schema import FIELD_ORDER

    return {k: canonical[k] for k in FIELD_ORDER if k in canonical}
