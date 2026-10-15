"""Paired bootstrap of the micro-F1 difference between two prediction keys.

  python -m ie643.eval.bootstrap --manifest Data/processed/test/manifest.jsonl --pred-keys student_e3_pred,teacher_nohint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ie643.eval.gt_clean import clean_gold
from ie643.eval.score import evaluate, score_record


def _f1(tp: float, fp: float, fn: float) -> float:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def document_counts(rows: list[dict], pred_key: str, gt_key: str, *, clean: bool) -> list[tuple[str, int, int, int]]:
    counts = []
    for rec in rows:
        gold = rec.get(gt_key) or {}
        if clean:
            gold = clean_gold(gold, rec.get("dataset"))
        pred = rec.get(pred_key) or {}
        tp = fp = fn = 0
        for _field, (a, b, c) in score_record(pred, gold).items():
            tp += a
            fp += b
            fn += c
        counts.append((rec.get("dataset") or "unknown", tp, fp, fn))
    return counts


def bootstrap_diff(
    left: list[tuple[str, int, int, int]],
    right: list[tuple[str, int, int, int]],
    *,
    n: int = 1000,
    seed: int = 643,
) -> dict[str, float]:
    """95% CI of F1(right) - F1(left), resampling documents together."""
    rng = np.random.default_rng(seed)
    size = len(left)
    if size != len(right):
        raise ValueError("the two runs must be aligned document by document")
    diffs = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, size, size)
        ltp = lfp = lfn = rtp = rfp = rfn = 0
        for j in idx:
            _, a, b, c = left[j]
            _, d, e, f = right[j]
            ltp += a
            lfp += b
            lfn += c
            rtp += d
            rfp += e
            rfn += f
        diffs[i] = _f1(rtp, rfp, rfn) - _f1(ltp, lfp, lfn)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": float(diffs.mean()), "lo": float(lo), "hi": float(hi)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap CI of a micro-F1 difference")
    parser.add_argument("--manifest", type=Path, default=Path("Data/processed/test/manifest.jsonl"))
    parser.add_argument("--pred-keys", required=True, help="two keys, difference is second minus first")
    parser.add_argument("--gt-key", default="source_gt")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=643)
    parser.add_argument("--raw-gt", action="store_true")
    args = parser.parse_args(argv)
    keys = [k.strip() for k in args.pred_keys.split(",") if k.strip()]
    if len(keys) != 2:
        raise SystemExit("--pred-keys needs exactly two keys, e.g. student_e3_pred,teacher_nohint")
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8") if line.strip()]
    report_a = evaluate(rows, pred_key=keys[0], gt_key=args.gt_key, clean_gt=not args.raw_gt)
    report_b = evaluate(rows, pred_key=keys[1], gt_key=args.gt_key, clean_gt=not args.raw_gt)
    left = document_counts(rows, keys[0], args.gt_key, clean=not args.raw_gt)
    right = document_counts(rows, keys[1], args.gt_key, clean=not args.raw_gt)
    ci = bootstrap_diff(left, right, n=args.samples, seed=args.seed)
    print(f"{keys[0]} F1 {report_a['overall']['f1']:.4f}")
    print(f"{keys[1]} F1 {report_b['overall']['f1']:.4f}")
    print(f"{keys[1]} - {keys[0]}: {ci['mean']:+.4f}  95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]  n={args.samples}")
    print("per-dataset F1")
    datasets = sorted(set(report_a["by_dataset"]) | set(report_b["by_dataset"]))
    for name in datasets:
        fa = report_a["by_dataset"].get(name, {}).get("f1")
        fb = report_b["by_dataset"].get(name, {}).get("f1")
        left_s = f"{fa:.4f}" if fa is not None else "—"
        right_s = f"{fb:.4f}" if fb is not None else "—"
        print(f"  {name}: {keys[0]} {left_s}   {keys[1]} {right_s}")


if __name__ == "__main__":
    main()
