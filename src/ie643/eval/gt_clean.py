"""Remove known annotation artifacts from source_gt before scoring.

Applied by default in ``ie643.eval`` (``--raw-gt`` turns it off). Every rule is listed here so the paper can
state exactly what was changed:

1. Placeholder strings are empty, not values. FATURA stores a missing invoice number as the literal string
   ``"[]"`` (1,200 rows train+test); CORD uses ``"-"`` for an absent tax / discount and ``"---"`` as a line-item
   name for separator rows. Placeholders become ``None`` (scalars and list-row keys).
2. Fields whose GT does not follow the canonical schema are not scored for that dataset. FATURA
   ``payment_method`` is the bank-details block ("Bank Name ... Branch Name ...") in all 2,600 rows; the schema
   defines payment_method as cash / card / transfer / ..., so a schema-following model can never match it.
3. Python dicts serialised as strings are unpacked. invoices-v2 stores ``total`` / ``subtotal`` as e.g.
   ``"{'total': '40,000', 'cash': '50,000', 'change': '10,000'}"`` (971 + 665 rows train+test). The value is the
   entry named like the field; without this, numeric parsing reads the last number (the change) as the total.
4. A value repeated inside one string is one value. invoices-v2 writes totals like ``"27.44 27.44"`` or
   ``"$49.70 49.70"`` (the amount printed twice on the receipt). When every whitespace-separated part parses to
   the same number, the first part is kept; otherwise parsing glues the parts into one wrong number.
"""

from __future__ import annotations

import ast
import copy
import re
from typing import Any

PLACEHOLDERS = frozenset({"[]", "{}", "-", "--", "---"})

# dataset -> fields whose GT is not the canonical field (excluded from scoring)
SCHEMA_MISMATCH_FIELDS: dict[str, frozenset[str]] = {
    "fatura2": frozenset({"payment_method"}),
}

RULES = (
    f"placeholder strings {sorted(PLACEHOLDERS)} -> null",
    "fatura2 payment_method not scored (GT is the bank-details block, not a payment method)",
    "stringified dicts (invoices-v2 total/subtotal) -> the entry named like the field",
    "an amount repeated inside one string ('27.44 27.44') -> one occurrence",
)

_HAS_DIGIT = re.compile(r"\d")


def collapse_repeated_amount(value: Any) -> Any:
    """'$49.70 49.70' -> '$49.70' when every part parses to the same number; anything else is unchanged."""
    from ie643.eval.match import parse_number  # local: match imports nothing from here, keep it that way

    if not isinstance(value, str):
        return value
    parts = [p for p in value.split() if _HAS_DIGIT.search(p)]
    if len(parts) < 2 or len(parts) != len(value.split()):
        return value
    nums = [parse_number(p) for p in parts]
    if any(n is None for n in nums) or len({round(n, 6) for n in nums}) != 1:
        return value
    return parts[0]


def unpack_dict_string(field: str, value: Any) -> Any:
    """"{'total': '40,000', 'cash': ...}" -> '40,000' for field 'total'. Other values are returned unchanged."""
    if not (isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}")):
        return value
    try:
        parsed = ast.literal_eval(value.strip())
    except (ValueError, SyntaxError):
        return value
    if not isinstance(parsed, dict):
        return value
    return parsed.get(field, value)


def is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip() in PLACEHOLDERS


def is_artifact(dataset: str | None, field: str, gold_value: Any) -> bool:
    """True when this GT field is changed or dropped by :func:`clean_gold`."""
    return (
        is_placeholder(gold_value)
        or field in SCHEMA_MISMATCH_FIELDS.get(dataset or "", frozenset())
        or unpack_dict_string(field, gold_value) is not gold_value
        or collapse_repeated_amount(gold_value) is not gold_value
    )


def clean_gold(gold: dict[str, Any] | None, dataset: str | None) -> dict[str, Any]:
    gold = copy.deepcopy(gold or {})
    for field in SCHEMA_MISMATCH_FIELDS.get(dataset or "", frozenset()):
        if field in gold:
            gold[field] = None
    for field, value in list(gold.items()):
        value = gold[field] = collapse_repeated_amount(unpack_dict_string(field, value))
        if is_placeholder(value):
            gold[field] = None
        elif isinstance(value, list):
            rows = []
            for row in value:
                if isinstance(row, dict):
                    row = {k: (None if is_placeholder(v) else v) for k, v in row.items()}
                    if all(v is None for v in row.values()):
                        continue
                rows.append(row)
            gold[field] = rows
    return gold
