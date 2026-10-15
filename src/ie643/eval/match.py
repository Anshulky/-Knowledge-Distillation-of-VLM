"""Value matching for canonical KIE fields.

GT is often a printed string with a label glued on (FATURA ``Invoice Date: 10-Aug-1993``)
or a different surface form than the teacher (ISO vs ``15/01/2019``, ``5:37PM`` vs ``17:37``).
Null / empty GT is not scored by default — SROIE-style labels only cover a few keys.

List rows (line items, taxes) match on name/label + amount. Quantity, unit price, and
rate are only required when both gold and pred filled them. Extra predicted rows with
amount 0 (modifiers, discounts) are not false positives.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any

from ie643.schema import FIELD_ORDER, LINE_ITEM_KEYS, TAX_ROW_KEYS

ANLS_THRESHOLD = 0.5
META_FIELDS = frozenset({"document_type", "language"})
LIST_FIELDS = frozenset({"line_items", "taxes"})
OPTIONAL_ROW_KEYS = frozenset({"quantity", "unit_price", "rate"})
AMOUNT_FIELDS = frozenset(
    {
        "subtotal",
        "tax",
        "service_charge",
        "discount",
        "delivery_fee",
        "tip",
        "total",
        "change",
        "quantity",
        "unit_price",
        "amount",
        "rate",
    }
)
DATE_FIELDS = frozenset({"date", "due_date"})
TIME_FIELDS = frozenset({"time"})
PHONE_FIELDS = frozenset({"vendor_phone"})
ID_FIELDS = frozenset({"vendor_tax_id", "receipt_number", "invoice_number", "po_number"})

_WS = re.compile(r"\s+")
_LABEL = re.compile(
    r"^(?:invoice\s*)?(?:"
    r"id|number|no\.?|date|due\s*date|invoice\s*date|"
    r"sub[_\s-]*total|total|balance\s*due|amount\s*due|"
    r"tax|vat|gst(?:\([^)]*\))?|gstin|siret|"
    r"bill\s*to|buyer|sold\s*to|address|tel(?:ephone)?|"
    r"payment(?:\s*details|\s*method)?|po(?:\s*number)?"
    r")\s*[:\-–]?\s*",
    re.I,
)
_CUR_SYM = re.compile(r"(?:rm|rp|usd|eur|gbp|myr|idr|inr|egp|rub|sgd|thb|vnd|¥|€|£|₹|₽|\$)", re.I)
_NUM = re.compile(r"-?\d+(?:[.,]\d+)*")  # whole token incl. separators; "110000" is one number, not "110" + "000"
_PHONE = re.compile(r"\d+")
_ALNUM = re.compile(r"[^a-z0-9]+", re.I)
_DATE_TOKEN = re.compile(
    r"(\d{1,2}[-/.\s]\d{1,2}[-/.\s]\d{2,4}|"
    r"\d{4}[-/.\s]\d{1,2}[-/.\s]\d{1,2}|"
    r"\d{1,2}[-/.\s][A-Za-z]{3,9}[-/.\s]\d{2,4}|"
    r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}|"
    r"\d{1,2}\s*[A-Za-z]{3,9}\s*'?\s*\d{2,4}|"
    r"[A-Za-z]{3,9}\s*\d{1,2},?\s*\d{2,4})"
)
_TIME_TOKEN = re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap][Mm])?)")
_OCT_TYPO = re.compile(r"0ct", re.I)
_LETTER_DIGIT = re.compile(r"([A-Za-z])(\d)")
_DIGIT_LETTER = re.compile(r"(\d)([A-Za-z])")

_DATE_FMTS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d-%m-%y",
    "%d/%m/%y",
    "%d.%m.%y",
    "%m-%d-%Y",
    "%m/%d/%Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%d-%b-%y",
    "%d %b %y",
    "%d %B %y",
    "%b %d %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %d %y",
    "%B %d %y",
    "%B %d,%Y",
)

_CUR_ALIAS = {
    "rm": "myr",
    "rp": "idr",
    "rs": "inr",
    "$": "usd",
    "us$": "usd",
    "€": "eur",
    "£": "gbp",
    "₹": "inr",
    "₽": "rub",
    "yen": "jpy",
    "sgd": "sgd",
}

_PAY = {
    "cash": "cash",
    "tunai": "cash",
    "contanti": "cash",
    "especes": "cash",
    "espece": "cash",
    "espèces": "cash",
    "espèce": "cash",
    "liquide": "cash",
    "cod": "cod",
    "card": "card",
    "carte": "card",
    "cartes": "card",
    "carta": "card",
    "credit": "card",
    "debit": "card",
    "visa": "card",
    "mastercard": "card",
    "cb": "card",
    "creditcard": "card",
    "contactless": "card",
    "bancomat": "card",
    "emoney": "emoney",
    "ewallet": "emoney",
    "transfer": "transfer",
    "bank": "transfer",
    "wire": "transfer",
    "mixed": "mixed",
}

_PAY_SUB = (
    "contanti",
    "especes",
    "espece",
    "espèces",
    "espèce",
    "liquide",
    "tunai",
    "contactless",
    "bancomat",
    "carta",
    "cartes",
    "carte",
    "cash",
    "card",
    "cb",
)


def _nfkc(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip()


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict, tuple)) and len(value) == 0:
        return True
    return False


def unwrap(value: Any) -> Any:
    """Pull a scalar out of adapter/model dict wrappers ({amount}, {text}, nested totals)."""
    cur = value
    for _ in range(4):
        if isinstance(cur, dict):
            for key in ("text", "amount", "total", "value", "price", "total_price", "number"):
                if key in cur and not is_empty(cur[key]):
                    cur = cur[key]
                    break
            else:
                return cur
            continue
        return cur
    return cur


def _strip_labels(text: str) -> str:
    original = _WS.sub(" ", text.replace("\n", " ").replace("\r", " ")).strip()
    s = original
    for _ in range(4):
        nxt = _LABEL.sub("", s, count=1).strip()
        if nxt == s:
            break
        s = nxt
    # a value that is only a label word ("GST", "VAT", "TAX" as a tax-row label) is the value itself;
    # stripping it to "" made such rows unmatchable, even against an identical prediction
    return s or original


def normalize_text(value: Any) -> str:
    s = _strip_labels(_nfkc(unwrap(value)))
    s = s.casefold()
    s = re.sub(r"[\"'`]+", "", s)
    s = re.sub(r"[.,;|/\\]+", " ", s)
    return _WS.sub(" ", s).strip()


def compact_text(value: Any) -> str:
    s = unicodedata.normalize("NFKD", normalize_text(value))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return _ALNUM.sub("", s)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def anls(pred: str, gold: str, *, threshold: float = ANLS_THRESHOLD) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    dist = levenshtein(pred, gold)
    sim = 1.0 - dist / max(len(pred), len(gold))
    return sim if sim >= threshold else 0.0


def _parse_numeric_token(token: str) -> float | None:
    """One numeric token -> float, with one consistent separator rule.

    Both "," and "." present: the later one is the decimal mark ("1.234,56" -> 1234.56, "1,234.56" -> 1234.56).
    One separator followed by exactly 3 digits is a thousands separator ("54,000" == "54.000" == 54000),
    except a leading zero ("0.055" stays 0.055). Repeated separators are thousands separators.
    Otherwise the separator is the decimal mark ("48,88" -> 48.88).
    """
    t = token
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif t.count(",") > 1 or t.count(".") > 1:
        t = t.replace(",", "").replace(".", "")
    elif t.count(",") == 1 or t.count(".") == 1:
        sep = "," if "," in t else "."
        left, right = t.split(sep)
        if len(right) == 3 and left.lstrip("-") not in ("", "0"):
            t = left + right
        else:
            t = f"{left}.{right}"
    try:
        return float(t)
    except ValueError:
        return None


def parse_number(value: Any, *, first: bool = False) -> float | None:
    raw = unwrap(value)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = _nfkc(raw)
    s = _CUR_SYM.sub("", s)
    s = _strip_labels(s)
    hits = _NUM.findall(s.replace(" ", ""))
    if not hits:
        return None
    return _parse_numeric_token(hits[0] if first else hits[-1])


def numbers_match(a: Any, b: Any, *, rel: float = 0.01, abs_tol: float = 0.02) -> bool:
    na, nb = parse_number(a), parse_number(b)
    if na is None or nb is None:
        return False
    if na == nb:
        return True
    scale = max(abs(na), abs(nb), 1.0)
    return abs(na - nb) <= max(abs_tol, rel * scale)


# Money is compared to the cent: a 1% relative tolerance accepted misread digits in large totals.
AMOUNT_ABS_TOL = 0.01


def amounts_match(a: Any, b: Any) -> bool:
    """Numeric equality to the cent when both sides contain a number; string match only when one side has none
    (e.g. "FREE"). A fuzzy string fallback on numbers accepted "17,121" for "17,727" (ANLS >= 0.5)."""
    na, nb = parse_number(a), parse_number(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= AMOUNT_ABS_TOL
    return strings_match(a, b)


def rates_match(a: Any, b: Any) -> bool:
    """0.055 (fraction) matches 5.5% / 5,5% / 5.5."""
    if numbers_match(a, b, rel=0.02, abs_tol=0.05):
        return True
    na, nb = parse_number(a), parse_number(b)
    if na is None or nb is None:
        return False
    lo, hi = (na, nb) if abs(na) <= abs(nb) else (nb, na)
    if abs(lo) <= 1.0 + 1e-9 and abs(hi) > 1.0:
        return abs(lo * 100.0 - hi) <= max(0.05, 0.02 * abs(hi))
    return False


def _prep_date(value: Any) -> str:
    s = _strip_labels(_nfkc(unwrap(value)))
    s = _OCT_TYPO.sub("Oct", s)
    s = s.replace("'", " ")
    s = s.replace(",", ", ")
    s = _LETTER_DIGIT.sub(r"\1 \2", s)
    s = _DIGIT_LETTER.sub(r"\1 \2", s)
    s = _WS.sub(" ", s).strip()
    parts = s.split()
    if len(parts) >= 2 and len(parts) % 2 == 0:
        half = len(parts) // 2
        if parts[:half] == parts[half:]:
            s = " ".join(parts[:half])
    return s


def parse_date(value: Any) -> datetime | None:
    s = _prep_date(value)
    m = _DATE_TOKEN.search(s)
    token = (m.group(1) if m else s).strip(" ,.-")
    token = _WS.sub(" ", token)
    for fmt in _DATE_FMTS:
        try:
            dt = datetime.strptime(token, fmt)
            if dt.year < 100:
                dt = dt.replace(year=dt.year + (2000 if dt.year < 70 else 1900))
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            continue
    digits = re.sub(r"\D", "", token)
    if len(digits) == 8:
        for y, mo, d in (
            (int(digits[0:4]), int(digits[4:6]), int(digits[6:8])),
            (int(digits[4:8]), int(digits[2:4]), int(digits[0:2])),
        ):
            try:
                return datetime(y, mo, d)
            except ValueError:
                continue
    if len(digits) == 6:
        try:
            d, mo, y = int(digits[0:2]), int(digits[2:4]), int(digits[4:6])
            y += 2000 if y < 70 else 1900
            return datetime(y, mo, d)
        except ValueError:
            pass
    return None


def dates_match(a: Any, b: Any) -> bool:
    da, db = parse_date(a), parse_date(b)
    if da is not None and db is not None:
        return da == db
    sa, sb = normalize_text(a), normalize_text(b)
    if sa and sb and (sa == sb or sa in sb or sb in sa):
        return True
    ca, cb = compact_text(a), compact_text(b)
    return bool(ca and cb and (ca == cb or ca in cb or cb in ca))


def parse_time_minutes(value: Any) -> int | None:
    s = _strip_labels(_nfkc(unwrap(value)))
    m = _TIME_TOKEN.search(s.replace(".", ":"))
    if not m:
        return None
    token = m.group(1).strip().upper().replace(" ", "")
    ampm = None
    if token.endswith("AM") or token.endswith("PM"):
        ampm = token[-2:]
        token = token[:-2]
    parts = token.split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return None
    if ampm == "AM" and hour == 12:
        hour = 0
    elif ampm == "PM" and hour != 12:
        hour += 12
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return None
    return hour * 60 + minute


def times_match(a: Any, b: Any) -> bool:
    ta, tb = parse_time_minutes(a), parse_time_minutes(b)
    if ta is None or tb is None:
        return strings_match(a, b)
    return min(abs(ta - tb), 24 * 60 - abs(ta - tb)) <= 1


def phones_match(a: Any, b: Any) -> bool:
    da = "".join(_PHONE.findall(_nfkc(unwrap(a))))
    db = "".join(_PHONE.findall(_nfkc(unwrap(b))))
    if not da or not db:
        return False
    if da == db:
        return True
    n = min(len(da), len(db), 10)
    return n >= 7 and (da[-n:] == db[-n:] or da in db or db in da)


def ids_match(a: Any, b: Any) -> bool:
    sa = compact_text(a)
    sb = compact_text(b)
    if not sa or not sb:
        return False
    if sa == sb:
        return True
    return sa in sb or sb in sa


def payment_match(a: Any, b: Any) -> bool:
    def canon(v: Any) -> str:
        t = compact_text(v)
        if t in _PAY:
            return _PAY[t]
        for key in _PAY_SUB:
            if key in t:
                return _PAY.get(key, key)
        return t

    ca, cb = canon(a), canon(b)
    if not ca or not cb:
        return False
    return ca == cb or ca in cb or cb in ca


def strings_match(a: Any, b: Any) -> bool:
    sa, sb = normalize_text(a), normalize_text(b)
    if not sa or not sb:
        return False
    if sa == sb or sa in sb or sb in sa:
        return True
    if anls(sa, sb) >= ANLS_THRESHOLD:
        return True
    ca, cb = compact_text(a), compact_text(b)
    if not ca or not cb:
        return False
    if ca == cb or ca in cb or cb in ca:
        return True
    return min(len(ca), len(cb)) >= 8 and anls(ca, cb) >= ANLS_THRESHOLD


def values_match(field: str, pred: Any, gold: Any) -> bool:
    if is_empty(gold) and is_empty(pred):
        return True
    if is_empty(gold) or is_empty(pred):
        return False
    pred, gold = unwrap(pred), unwrap(gold)
    if field in DATE_FIELDS:
        return dates_match(pred, gold)
    if field in TIME_FIELDS:
        return times_match(pred, gold)
    if field in PHONE_FIELDS:
        return phones_match(pred, gold)
    if field in ID_FIELDS:
        return ids_match(pred, gold) or strings_match(pred, gold)
    if field == "payment_method":
        return payment_match(pred, gold)
    if field == "currency":
        def cur(v: Any) -> str:
            t = normalize_text(v).replace(" ", "")
            t = _CUR_ALIAS.get(t, t)
            return t.upper()

        return bool(cur(pred)) and cur(pred) == cur(gold)
    if field == "rate":
        return rates_match(pred, gold)
    if field == "quantity":
        na, nb = parse_number(pred, first=True), parse_number(gold, first=True)
        if na is None or nb is None:
            return False
        return numbers_match(na, nb)
    if field == "discount":
        # sign convention varies ("(-) 6.11", "- 6.11", "6.11"); the printed amount is what is extracted
        na, nb = parse_number(pred), parse_number(gold)
        if na is not None and nb is not None:
            return abs(abs(na) - abs(nb)) <= AMOUNT_ABS_TOL
    if field in AMOUNT_FIELDS:
        return amounts_match(pred, gold)
    if field in LIST_FIELDS:
        tp, fp, fn = list_prf(
            field,
            pred if isinstance(pred, list) else [pred],
            gold if isinstance(gold, list) else [gold],
        )
        return fp == 0 and fn == 0 and tp == len(gold if isinstance(gold, list) else [gold])
    return strings_match(pred, gold)


def tax_values_match(pred: Any, gold_tax: Any, gold_taxes: Any = None) -> bool:
    """Scalar tax matches the gold figure, any VAT band, or the sum of bands."""
    if values_match("tax", pred, gold_tax):
        return True
    pn = parse_number(pred)
    if pn is None:
        return False
    amounts: list[float] = []
    for row in gold_taxes or []:
        if not isinstance(row, dict):
            continue
        n = parse_number(row.get("amount"))
        if n is not None:
            amounts.append(n)
    if not amounts:
        return False
    if any(numbers_match(pn, a) for a in amounts):
        return True
    return numbers_match(pn, sum(amounts))


def _row_amount_filled(value: Any) -> bool:
    return parse_number(value) is not None


def _is_zero_amount_row(row: dict) -> bool:
    n = parse_number(row.get("amount"))
    return n is not None and abs(n) <= 0.005


def _row_match(keys: tuple[str, ...], pred: dict, gold: dict) -> bool:
    """Name/label + amount; qty / unit_price / rate only if both sides filled a number."""
    name_g = gold.get("name") if "name" in keys else gold.get("label")
    name_p = pred.get("name") if "name" in keys else pred.get("label")
    if "label" in keys:
        name_g = gold.get("label") or gold.get("name")
        name_p = pred.get("label") or pred.get("name")
    if not is_empty(name_g):
        if is_empty(name_p) or not strings_match(name_p, name_g):
            return False
    if not is_empty(gold.get("amount")):
        if not _row_amount_filled(pred.get("amount")):
            return False
        if not amounts_match(pred.get("amount"), gold.get("amount")):
            return False
    for key in keys:
        if key in {"name", "label", "amount"}:
            continue
        g, p = gold.get(key), pred.get(key)
        if is_empty(g):
            continue
        if key in OPTIONAL_ROW_KEYS and not _row_amount_filled(p):
            continue
        if is_empty(p):
            continue
        if key == "rate":
            if not rates_match(p, g):
                return False
        elif not values_match(key, p, g):
            return False
    if is_empty(name_g) and is_empty(gold.get("amount")):
        return False
    return True


def list_prf(field: str, pred: list, gold: list) -> tuple[int, int, int]:
    keys = LINE_ITEM_KEYS if field == "line_items" else TAX_ROW_KEYS
    gold_rows = [g for g in gold if isinstance(g, dict) and not all(is_empty(g.get(k)) for k in keys)]
    pred_rows = [p for p in pred if isinstance(p, dict)]
    used: set[int] = set()
    tp = 0
    for g in gold_rows:
        best_j: int | None = None
        for j, p in enumerate(pred_rows):
            if j in used:
                continue
            if _row_match(keys, p, g):
                best_j = j
                break
        if best_j is not None:
            used.add(best_j)
            tp += 1
    fn = len(gold_rows) - tp
    fp = sum(1 for j, p in enumerate(pred_rows) if j not in used and not _is_zero_amount_row(p))
    return tp, fp, fn


EXTRACTION_FIELDS = tuple(k for k in FIELD_ORDER if k not in META_FIELDS)
