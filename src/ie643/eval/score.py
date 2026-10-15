"""Aggregate teacher/student predictions vs source_gt."""

from __future__ import annotations

from typing import Any

from ie643.eval.gt_clean import RULES as GT_CLEAN_RULES
from ie643.eval.gt_clean import clean_gold
from ie643.eval.match import (
    EXTRACTION_FIELDS,
    LIST_FIELDS,
    is_empty,
    list_prf,
    tax_values_match,
    values_match,
)

# Dataset groups reported next to the micro score.
SUBSETS: dict[str, tuple[str, ...]] = {
    "real_photo": ("sroie", "cord-v2", "invoices-and-receipts-v2"),
    "template": ("fatura2", "synthetic-receipts-ocr"),
}


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n": tp + fn,
        # full precision: rounding here and again when printing turned 0.98248 into "0.983"
        "precision": prec,
        "recall": rec,
        "f1": f1,
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
    clean_gt: bool = True,
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
        if clean_gt:
            gold = clean_gold(gold, rec.get("dataset"))
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

    ds_scores = {k: pack(v) for k, v in sorted(by_ds.items())}
    subsets = {}
    for name, members in SUBSETS.items():
        slot = [0, 0, 0]
        for d in members:
            for i, v in enumerate(by_ds.get(d, [0, 0, 0])):
                slot[i] += v
        if any(slot):
            subsets[name] = pack(slot)
    field_f1 = [pack(v)["f1"] for v in by_field.values()]

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
        "gt_cleaning": list(GT_CLEAN_RULES) if clean_gt else [],
        "overall": pack(overall),
        # unweighted means: every dataset (field) counts once, whatever its number of GT fields
        "macro_dataset_f1": sum(d["f1"] for d in ds_scores.values()) / len(ds_scores) if ds_scores else 0.0,
        "macro_field_f1": sum(field_f1) / len(field_f1) if field_f1 else 0.0,
        "subsets": subsets,
        "by_dataset": ds_scores,
        "by_field": {k: pack(v) for k, v in sorted(by_field.items(), key=lambda kv: -kv[1][0] - kv[1][2])},
        "mismatches": mismatches,
    }


def _nonnull(pred: dict[str, Any] | None, fields: tuple[str, ...]) -> set[str]:
    pred = pred or {}
    return {f for f in fields if not is_empty(pred.get(f))}


def coverage(
    rows: list[dict[str, Any]],
    *,
    pred_key: str,
    gt_key: str = "source_gt",
    fields: tuple[str, ...] = EXTRACTION_FIELDS,
    ref: dict[str, dict[str, Any]] | None = None,
    ref_key: str = "teacher_gt",
    clean_gt: bool = True,
    datasets: set[str] | None = None,
) -> dict[str, Any]:
    """What GT-present scoring cannot see: how many fields a model fills, and whether the fields the GT does
    not cover agree with a reference model (``ref``: uid -> row holding ``ref_key``, e.g. the teacher).

    Per dataset, means per document:
      filled        non-null extraction fields in the prediction
      gt_filled     non-null fields in the (cleaned) GT
      extra         filled where the GT is empty (unscored by default)
      extra_agree   of ``extra``, fields that match the reference (values_match / list row matching)
      unsupported   ``extra`` fields where the reference is empty too (hallucination proxy)
    """
    acc: dict[str, dict[str, float]] = {}
    for rec in rows:
        ds = rec.get("dataset") or "unknown"
        if datasets and ds not in datasets:
            continue
        pred = rec.get(pred_key)
        if not pred:
            continue
        gold = clean_gold(rec.get(gt_key), ds) if clean_gt else (rec.get(gt_key) or {})
        filled = _nonnull(pred, fields)
        extra = filled - _nonnull(gold, fields)
        slot = acc.setdefault(ds, {"docs": 0, "filled": 0, "gt_filled": 0, "extra": 0, "extra_agree": 0,
                                   "unsupported": 0, "extra_scored": 0})
        slot["docs"] += 1
        slot["filled"] += len(filled)
        slot["gt_filled"] += len(_nonnull(gold, fields))
        slot["extra"] += len(extra)
        if ref is not None and rec.get("uid") in ref:
            rpred = ref[rec["uid"]].get(ref_key) or {}
            for f in extra:
                if is_empty(rpred.get(f)):
                    slot["unsupported"] += 1
                    continue
                slot["extra_scored"] += 1
                if f in LIST_FIELDS:
                    p_rows = pred[f] if isinstance(pred[f], list) else [pred[f]]
                    r_rows = rpred[f] if isinstance(rpred[f], list) else [rpred[f]]
                    tp, fp, fn = list_prf(f, p_rows, r_rows)
                    slot["extra_agree"] += tp / max(tp + fp, 1)
                elif values_match(f, pred[f], rpred[f]):
                    slot["extra_agree"] += 1

    def summarise(slot: dict[str, float]) -> dict[str, float]:
        d = max(slot["docs"], 1)
        return {
            "docs": slot["docs"],
            "filled": slot["filled"] / d,
            "gt_filled": slot["gt_filled"] / d,
            "extra": slot["extra"] / d,
            "unsupported": slot["unsupported"] / d if ref is not None else None,
            "extra_agree_rate": (slot["extra_agree"] / slot["extra_scored"]) if ref is not None and slot["extra_scored"] else None,
        }

    total = {k: sum(v[k] for v in acc.values()) for k in
             ("docs", "filled", "gt_filled", "extra", "extra_agree", "unsupported", "extra_scored")}
    return {
        "pred_key": pred_key,
        "ref_key": ref_key if ref is not None else None,
        "by_dataset": {k: summarise(v) for k, v in sorted(acc.items())},
        "overall": summarise(total) if acc else {},
    }


def format_report(report: dict[str, Any], *, max_errors: int = 0) -> str:
    """F1 / P / R printed as percentages with 2 decimals, computed from exact counts (no double rounding)."""
    o = report["overall"]
    pct = lambda x: f"{100 * x:.2f}"  # noqa: E731
    cleaning = "; ".join(report.get("gt_cleaning") or []) or "off (--raw-gt)"
    lines = [
        f"eval {report['pred_key']} vs {report['gt_key']}  ({report['protocol']})",
        f"GT cleaning: {cleaning}",
        f"rows {report['n_scored']}/{report['n_rows']} scored  "
        f"missing_pred={report['n_missing_pred']}  empty_gt={report['n_empty_gt']}",
        f"micro       P={pct(o['precision'])}  R={pct(o['recall'])}  F1={pct(o['f1'])}  "
        f"tp={o['tp']} fp={o['fp']} fn={o['fn']}  n_gt={o['n']}",
        f"macro F1    datasets={pct(report['macro_dataset_f1'])}  fields={pct(report['macro_field_f1'])}",
    ]
    for name, slot in report.get("subsets", {}).items():
        lines.append(f"subset {name:<11} F1={pct(slot['f1'])}  n_gt={slot['n']}")
    lines += ["", f"{'dataset':<28} {'n_gt':>6} {'P':>7} {'R':>7} {'F1':>7}"]
    for name, slot in report["by_dataset"].items():
        lines.append(
            f"{name:<28} {slot['n']:>6} {pct(slot['precision']):>7} {pct(slot['recall']):>7} {pct(slot['f1']):>7}"
        )
    lines += ["", f"{'field':<22} {'n_gt':>6} {'P':>7} {'R':>7} {'F1':>7}"]
    for name, slot in report["by_field"].items():
        lines.append(
            f"{name:<22} {slot['n']:>6} {pct(slot['precision']):>7} {pct(slot['recall']):>7} {pct(slot['f1']):>7}"
        )
    if max_errors and report["mismatches"]:
        lines += ["", f"mismatches (first {max_errors}):"]
        for row in report["mismatches"][:max_errors]:
            lines.append(f"  {row['uid']}  {row['field']}: gold={row['gold']!r} pred={row['pred']!r}")
    return "\n".join(lines)


def format_coverage(cov: dict[str, Any]) -> str:
    ref = cov["ref_key"]
    head = f"{'dataset':<28} {'docs':>5} {'filled':>7} {'gt':>6} {'extra':>6}"
    if ref:
        head += f" {'agree%':>7} {'unsup':>6}"
    lines = ["", f"coverage of {cov['pred_key']} (fields per document" + (f"; extras checked against {ref})" if ref else ")"), head]
    rows = list(cov["by_dataset"].items()) + ([("ALL", cov["overall"])] if cov["overall"] else [])
    for name, c in rows:
        line = f"{name:<28} {c['docs']:>5} {c['filled']:>7.2f} {c['gt_filled']:>6.2f} {c['extra']:>6.2f}"
        if ref:
            agree = "-" if c["extra_agree_rate"] is None else f"{100 * c['extra_agree_rate']:.1f}"
            line += f" {agree:>7} {c['unsupported']:>6.2f}"
        lines.append(line)
    return "\n".join(lines)
