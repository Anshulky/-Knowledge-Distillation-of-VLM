"""Map heterogeneous dataset labels into the canonical schema."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from datetime import date as _date
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
        total_tax = 0.0
        n_amt = 0
        for row in taxes:
            try:
                total_tax += float(row["amount"])
                n_amt += 1
            except (TypeError, ValueError, KeyError):
                continue
        out["tax"] = total_tax if n_amt else taxes[0].get("amount")
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


# --- Evaluation-only datasets (DocILE, MC-OCR); records built by ie643.splits.ood ---
# These return (source_gt, gt_alternatives). A field printed in several boxes keeps the
# first box in reading order; the other distinct values are alternatives the evaluator
# also accepts (office address vs remit-to P.O. box, logo name vs branch name).

_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_MDY = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})\b")


def _distinct(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        t = _text(value)
        if not t:
            continue
        key = _NON_ALNUM.sub("", t.casefold()) or t
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _us_date_iso(text: str | None) -> str | None:
    """03/25/99 -> 1999-03-25 when the printed date is a valid month/day/year."""
    m = _MDY.search(text or "")
    if not m:
        return None
    month, day, year = (int(g) for g in m.groups())
    if len(m.group(3)) == 2:
        year += 2000 if year < 70 else 1900
    try:
        return _date(year, month, day).isoformat()
    except ValueError:
        return None


def _docile_order(box: dict[str, Any]) -> tuple:
    x0, y0, _, _ = box.get("bbox") or (0, 0, 0, 0)
    return (box.get("page", 0), round(y0, 2), x0)


_DOCILE_DOC_TYPE = {
    "tax_invoice": "invoice",
    "proforma": "invoice",
    "credit_note": "invoice",
    "debit_note": "invoice",
    "utility_bill": "invoice",
    "receipt": "receipt",
}


def from_docile(ann: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """DocILE annotation JSON (human field labels, not the OCR) -> (source_gt, alternatives)."""
    meta = ann.get("metadata") or {}
    out = empty_canonical(
        document_type=_DOCILE_DOC_TYPE.get(meta.get("document_type"), "other"), language="en"
    )
    boxes: dict[str, list[str]] = defaultdict(list)
    for box in sorted(ann.get("field_extractions") or [], key=_docile_order):
        t = _text(box.get("text"))
        if t:
            boxes[box["fieldtype"]].append(t)
    alts: dict[str, list[str]] = {}

    def put(key: str, *fieldtypes: str) -> None:
        values = _distinct(v for ft in fieldtypes for v in boxes.get(ft, []))
        if values:
            out[key] = values[0]
            if len(values) > 1:
                alts[key] = values[1:]

    put("vendor", "vendor_name")
    put("vendor_address", "vendor_address")
    put("vendor_tax_id", "vendor_tax_id")
    put("buyer", "customer_billing_name")
    put("buyer_address", "customer_billing_address")
    put("date", "date_issue")
    put("due_date", "date_due")
    put("receipt_number" if out["document_type"] == "receipt" else "invoice_number", "document_id")
    put("po_number", "customer_order_id", "order_id")
    put("subtotal", "amount_total_net")
    put("tax", "amount_total_tax")
    put("total", "amount_total_gross")
    if out["total"] is None:
        put("total", "amount_due")
    out["currency"] = _currency_from(*boxes.get("currency_code_amount_due", []), out["total"], out["subtotal"])

    # US invoices print 03/25/99; the evaluator reads d/m/y, so also accept the ISO date.
    for key in ("date", "due_date"):
        for value in [out[key], *alts.get(key, [])]:
            iso = _us_date_iso(value)
            if iso and iso not in alts.get(key, []):
                alts.setdefault(key, []).append(iso)

    rows: dict[Any, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for box in sorted(ann.get("line_item_extractions") or [], key=_docile_order):
        t = _text(box.get("text"))
        if t:
            rows[box.get("line_item_id")][box["fieldtype"]].append(t)
    items = []
    for _, row in sorted(rows.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
        def first(*fieldtypes: str) -> str | None:
            for ft in fieldtypes:
                if row.get(ft):
                    return row[ft][0]
            return None

        item = make_item(
            name=" ".join(row.get("line_item_description", [])) or None,
            quantity=first("line_item_quantity"),
            unit_price=first("line_item_unit_price_gross", "line_item_unit_price_net"),
            amount=first("line_item_amount_gross", "line_item_amount_net"),
        )
        if item["amount"] and not re.search(r"\d", item["amount"]):
            continue  # table header annotated as an item ("Program | # of spots | Rate | Total Cost")
        if item["name"] or item["amount"]:  # the evaluator matches rows on name + amount
            items.append(item)
    out["line_items"] = items
    return coerce_canonical(out), alts


_VN_DATE = re.compile(r"\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{2,4}")
_CLOCK = re.compile(r"(\d{1,2})\s*:\s*(\d{2})(?:\s*:\s*(\d{2}))?")
_AMOUNT_ONLY = re.compile(r"[\d\s.,]+(?:đ|d|vnd|vnđ)?")
_NUMBER_RUN = re.compile(r"\d[\d.,\s]*\d|\d")


def from_mcocr(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """MC-OCR CSV row (Vietnamese receipt, 4 labels) -> (source_gt, alternatives)."""
    texts = str(row.get("anno_texts") or "").split("|||")
    labels = str(row.get("anno_labels") or "").split("|||")
    try:
        polys = ast.literal_eval(row.get("anno_polygons") or "[]")
    except (ValueError, SyntaxError):
        polys = []
    bboxes = [p.get("bbox") or [0, 0, 0, 0] for p in polys] if len(polys) == len(texts) else []
    order = range(len(texts))
    if bboxes:
        order = sorted(order, key=lambda i: (round(bboxes[i][1] / 15), bboxes[i][0]))
    by: dict[str, list[str]] = defaultdict(list)
    for i in order:
        label = labels[i].strip() if i < len(labels) else ""
        label = "TOTAL_COST" if label == "TOTAL_TOTAL_COST" else label
        t = _text(texts[i])
        if t and label:
            by[label].append(t)

    out = empty_canonical(document_type="receipt", language="unknown")
    alts: dict[str, list[str]] = {}
    sellers = _distinct(by["SELLER"])
    if sellers:
        out["vendor"] = sellers[0]
        if len(sellers) > 1:
            alts["vendor"] = sellers[1:] + [" ".join(sellers)]
    out["vendor_address"] = " ".join(by["ADDRESS"]) or None

    stamp = " ".join(by["TIMESTAMP"])
    m = _VN_DATE.search(stamp)
    if m:
        out["date"] = re.sub(r"\s+", "", m.group(0))
        stamp = stamp[: m.start()] + " " + stamp[m.end() :]
    clock = _CLOCK.search(stamp)
    if clock:
        out["time"] = ":".join(g for g in clock.groups() if g)
    if not m and not clock and stamp:
        out["date"] = stamp

    amounts = [t for t in by["TOTAL_COST"] if _AMOUNT_ONLY.fullmatch(t.casefold().strip())]
    if not amounts:  # label and number in one box: keep the number
        amounts = [_NUMBER_RUN.findall(t)[-1].strip() for t in by["TOTAL_COST"] if _NUMBER_RUN.search(t)]
    amounts = _distinct(amounts)
    if amounts:
        out["total"] = amounts[0]
        if len(amounts) > 1:
            alts["total"] = amounts[1:]
    return coerce_canonical(out), alts


_PLACE_WORD = re.compile(r"[^\W\d_]{4,}")  # "Cairo", "London": currency guessed from the place, not printed


def _rb_doc_type(category: str) -> str:
    if re.search(r"Purchase|Dining|Taxi|Fuel|Parking|Toll|Laundry|Printing|Medical", category):
        return "receipt"
    if re.search(r"Bill|UtilityFee|PhoneFee|InternetFee", category):
        return "invoice"
    return "other"


def from_receiptbench(ann: dict[str, Any], category: str) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """ReceiptBench JSON -> (source_gt, alternatives). Only fields that are printed as-is.

    Skipped: seller_address (a "Country-City" region, not the printed address), invoice_number
    (ticket no. / PNR / order id / receipt id; not one schema key), detail (items mixed with
    fees, taxes, discounts and subtotals), travel fields with no schema key.
    """
    out = empty_canonical(document_type=_rb_doc_type(category), language="unknown")
    alts: dict[str, list[str]] = {}
    sellers = _distinct(ann.get("seller_name") or [])
    if sellers:
        out["vendor"] = sellers[0]
        if len(sellers) > 1:
            alts["vendor"] = sellers[1:]
    out["vendor_tax_id"] = _text(ann.get("tax_number"))
    out["date"] = _text(ann.get("orig_invoice_time"))
    std_date = _text(ann.get("std_invoice_time"))
    if out["date"] and std_date and std_date != out["date"]:
        alts["date"] = [std_date]
    out["total"] = _text(ann.get("orig_total"))
    std_total = _text(ann.get("std_total"))
    if out["total"] and std_total and std_total != out["total"] and re.search(r"[1-9]", std_total):
        alts["total"] = [std_total]
    printed_currency = [c for c in ann.get("orig_curr") or [] if _text(c) and not _PLACE_WORD.fullmatch(_text(c))]
    if printed_currency:
        out["currency"] = _text(ann.get("std_curr"))
    return coerce_canonical(out), alts
