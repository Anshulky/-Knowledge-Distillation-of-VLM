"""Aggregate teacher/student predictions vs source_gt."""

from __future__ import annotations

from typing import Any

from ie643.eval.match import (
    EXTRACTION_FIELDS,
    LIST_FIELDS,
    is_empty,
    list_prf,
    tax_values_match,
    values_match,
)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n": tp + fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def _bump(store: dict[str, list[int]], key: str, tp: int, fp: int, fn: int) -> None:
    slot = store.setdefault(key, [0, 0, 0])
    slot[0] += tp
    slot[1] += fp
    slot[2] += fn


def score_record(
    pred: dict[str, Any] | None,
    gold: dict[str, Any] | None,
    *,
    fields: tuple[str, ...] = EXTRACTION_FIELDS,
    strict: bool = False,
) -> dict[str, tuple[int, int, int]]:
    """Per-field (tp, fp, fn) for one document."""
    pred = pred or {}
    gold = gold or {}
    out: dict[str, tuple[int, int, int]] = {}
    for field in fields:
        g, p = gold.get(field), pred.get(field)
        if field in LIST_FIELDS:
            g_list = g if isinstance(g, list) else ([] if is_empty(g) else [g])
            p_list = p if isinstance(p, list) else ([] if is_empty(p) else [p])
            if not g_list and not strict:
                continue
            if not g_list and strict:
                out[field] = (0, len(p_list), 0)
                continue
            out[field] = list_prf(field, p_list, g_list)
            continue
        if is_empty(g) and not strict:
            continue
        if is_empty(g) and is_empty(p):
            continue
        if is_empty(g) and not is_empty(p):
            out[field] = (0, 1, 0)
            continue
        if field == "tax" and tax_values_match(p, g, gold.get("taxes")):
            out[field] = (1, 0, 0)
            continue
        if values_match(field, p, g):
            out[field] = (1, 0, 0)
        else:
            out[field] = (0, 1, 1)
    return out


def evaluate(
    rows: list[dict[str, Any]],
    *,
    pred_key: str = "teacher_gt",
    gt_key: str = "source_gt",
    fields: tuple[str, ...] = EXTRACTION_FIELDS,
    strict: bool = False,
    count_missing: bool = False,
    datasets: set[str] | None = None,
) -> dict[str, Any]:
    n_total = 0
    n_scored = 0
    n_missing_pred = 0
    n_empty_gt = 0
    overall = [0, 0, 0]
    by_ds: dict[str, list[int]] = {}
    by_field: dict[str, list[int]] = {}
    mismatches: list[dict[str, Any]] = []

    for rec in rows:
        if datasets and rec.get("dataset") not in datasets:
            continue
        n_total += 1
        gold = rec.get(gt_key) or {}
        pred = rec.get(pred_key)
        if not pred:
            n_missing_pred += 1
            if not count_missing:
                continue
            pred = {}
        n_scored += 1
        per = score_record(pred, gold, fields=fields, strict=strict)
        if not per:
            n_empty_gt += 1
        for field, (tp, fp, fn) in per.items():
            overall[0] += tp
            overall[1] += fp
            overall[2] += fn
            _bump(by_field, field, tp, fp, fn)
            _bump(by_ds, rec.get("dataset") or "unknown", tp, fp, fn)
            if fp or fn:
                mismatches.append(
                    {
                        "uid": rec.get("uid"),
                        "dataset": rec.get("dataset"),
                        "field": field,
                        "gold": (gold or {}).get(field),
                        "pred": (pred or {}).get(field),
                    }
                )

    def pack(slot: list[int]) -> dict[str, float | int]:
        return _prf(slot[0], slot[1], slot[2])

    return {
        "pred_key": pred_key,
        "gt_key": gt_key,
        "protocol": (
            "strict: extra predicted fields are FP"
            if strict
            else "gt_present: name+amount for lists; qty/rate only if both filled; "
            "0.055==5.5%; zero-amount extras ignored; tax scalar matches band or sum"
        ),
        "fields": list(fields),
        "n_rows": n_total,
        "n_scored": n_scored,
        "n_missing_pred": n_missing_pred,
        "n_empty_gt": n_empty_gt,
        "overall": pack(overall),
        "by_dataset": {k: pack(v) for k, v in sorted(by_ds.items())},
        "by_field": {k: pack(v) for k, v in sorted(by_field.items(), key=lambda kv: -kv[1][0] - kv[1][2])},
        "mismatches": mismatches,
    }


def format_report(report: dict[str, Any], *, max_errors: int = 0) -> str:
    o = report["overall"]
    lines = [
        f"eval {report['pred_key']} vs {report['gt_key']}  ({report['protocol']})",
        f"rows {report['n_scored']}/{report['n_rows']} scored  "
        f"missing_pred={report['n_missing_pred']}  empty_gt={report['n_empty_gt']}",
        f"micro  P={o['precision']:.3f}  R={o['recall']:.3f}  F1={o['f1']:.3f}  "
        f"tp={o['tp']} fp={o['fp']} fn={o['fn']}  n_gt={o['n']}",
        "",
        f"{'dataset':<28} {'n_gt':>6} {'P':>7} {'R':>7} {'F1':>7}",
    ]
    for name, slot in report["by_dataset"].items():
        lines.append(
            f"{name:<28} {slot['n']:>6} {slot['precision']:>7.3f} {slot['recall']:>7.3f} {slot['f1']:>7.3f}"
        )
    lines += ["", f"{'field':<22} {'n_gt':>6} {'P':>7} {'R':>7} {'F1':>7}"]
    for name, slot in report["by_field"].items():
        lines.append(
            f"{name:<22} {slot['n']:>6} {slot['precision']:>7.3f} {slot['recall']:>7.3f} {slot['f1']:>7.3f}"
        )
    if max_errors and report["mismatches"]:
        lines += ["", f"mismatches (first {max_errors}):"]
        for row in report["mismatches"][:max_errors]:
            lines.append(f"  {row['uid']}  {row['field']}: gold={row['gold']!r} pred={row['pred']!r}")
    return "\n".join(lines)
