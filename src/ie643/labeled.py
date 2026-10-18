"""Which canonical fields each dataset's source_gt actually annotates.

Datasets label different subsets of the schema (SROIE: vendor, address, date,
total; CORD: line items and totals; ...). A null is only trusted as "not printed"
for fields the dataset labels somewhere; everywhere else it means "not
annotated". Training masks those fields out of the CE loss, and evaluation
should skip them, so a model is never taught or scored on missing annotations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from ie643.schema import FIELD_ORDER


def is_empty(value: Any) -> bool:
    return value is None or value == "" or value == []


def labeled_fields(rows: Iterable[dict[str, Any]], *, min_frac: float = 0.01) -> dict[str, list[str]]:
    """Per dataset, the fields that are non-empty in at least `min_frac` of its source_gt rows."""
    totals: Counter[str] = Counter()
    filled: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in rows:
        ds = rec["dataset"]
        totals[ds] += 1
        gt = rec.get("source_gt") or {}
        for key in FIELD_ORDER:
            if not is_empty(gt.get(key)):
                filled[ds][key] += 1
    return {
        ds: [k for k in FIELD_ORDER if filled[ds][k] >= max(1.0, min_frac * n)]
        for ds, n in sorted(totals.items())
    }
