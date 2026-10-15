"""Teacher-label filter for distillation (docs/distillation_plan.md, B.6).

  python -m ie643.distill.label_filter --manifest Data/processed/train/manifest.jsonl

Drop rules (row removed from training):
  error        teacher_error set, or no teacher_gt / teacher_soft
  unparsable   parse_teacher_output could not recover a JSON object (teacher_gt is None)
  truncated    finish_reason != "stop" (max_tokens hit, or vLLM repetition stop)
  loop         a 1-8 token pattern repeats >= 16 times back-to-back in the emitted tokens
  empty        no extraction field is non-null

Flag rule (kept, reported): the teacher disagrees with source_gt on a field the GT fills
(ie643.eval.score.score_record, default settings). GT artifacts (ie643.eval.gt_clean: placeholder
strings such as "[]" / "-", FATURA payment_method bank blocks) are reported separately so they do
not inflate the count.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ie643.eval.gt_clean import RULES as GT_CLEAN_RULES
from ie643.eval.gt_clean import is_artifact
from ie643.eval.match import EXTRACTION_FIELDS, is_empty
from ie643.eval.score import score_record
from ie643.teacher.soft import load_soft

LOOP_MAX_PATTERN = 8
LOOP_MIN_COUNT = 16


def has_loop(ids: np.ndarray, *, max_pattern: int = LOOP_MAX_PATTERN, min_count: int = LOOP_MIN_COUNT) -> bool:
    """True if some pattern of length 1..max_pattern repeats min_count times consecutively."""
    n = len(ids)
    for p in range(1, max_pattern + 1):
        need = p * (min_count - 1)  # positions that must equal the token p steps earlier
        if n < p * min_count:
            break
        same = ids[p:] == ids[:-p]
        run = 0
        for ok in same:
            run = run + 1 if ok else 0
            if run >= need:
                return True
    return False


def drop_reason(rec: dict, root: Path) -> str | None:
    if rec.get("teacher_error") or not rec.get("teacher_soft"):
        return "error"
    if rec.get("teacher_gt") is None:
        return "unparsable"
    if rec.get("teacher_finish_reason") != "stop":
        return "truncated"
    if has_loop(load_soft(root / rec["teacher_soft"])["token_ids"]):
        return "loop"
    if all(is_empty(rec["teacher_gt"].get(f)) for f in EXTRACTION_FIELDS):
        return "empty"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build label_filter.json for distillation training")
    parser.add_argument("--manifest", type=Path, default=Path("Data/processed/train/manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=None, help="default: <manifest dir>/label_filter.json")
    args = parser.parse_args()
    root = args.manifest.parent
    out = args.out or root / "label_filter.json"
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8") if line.strip()]

    dropped: dict[str, str] = {}
    flagged: dict[str, list[str]] = {}
    drop_counts: dict[str, Counter] = defaultdict(Counter)
    flag_fields: dict[str, Counter] = defaultdict(Counter)
    artifact_fields: dict[str, Counter] = defaultdict(Counter)
    n_by_ds: Counter = Counter()
    for rec in rows:
        ds = rec["dataset"]
        n_by_ds[ds] += 1
        reason = drop_reason(rec, root)
        if reason:
            dropped[rec["uid"]] = reason
            drop_counts[ds][reason] += 1
            continue
        bad = []
        for field, (_tp, _fp, fn) in score_record(rec["teacher_gt"], rec.get("source_gt")).items():
            if not fn:
                continue
            if is_artifact(ds, field, (rec.get("source_gt") or {}).get(field)):
                artifact_fields[ds][field] += 1
            else:
                bad.append(field)
                flag_fields[ds][field] += 1
        if bad:
            flagged[rec["uid"]] = bad

    keep = [r["uid"] for r in rows if r["uid"] not in dropped]
    summary = {
        ds: {
            "n": n_by_ds[ds],
            "dropped": dict(drop_counts[ds]),
            "kept": n_by_ds[ds] - sum(drop_counts[ds].values()),
            "flagged_rows": sum(1 for u in flagged if u.split("::")[0] == _uid_prefix(ds)),
            "flagged_fields": dict(flag_fields[ds].most_common()),
            "gt_artifact_mismatches": dict(artifact_fields[ds].most_common()),
        }
        for ds in sorted(n_by_ds)
    }
    payload = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest),
        "rules": {
            "drop": ["error", "unparsable", "truncated", "loop", "empty"],
            "loop": {"max_pattern": LOOP_MAX_PATTERN, "min_count": LOOP_MIN_COUNT, "consecutive": True},
            "flag": "teacher disagrees with source_gt on a GT-filled field (kept for training)",
            "gt_artifacts_excluded_from_flags": list(GT_CLEAN_RULES),
        },
        "counts": {"n": len(rows), "kept": len(keep), "dropped": len(dropped), "flagged_rows": len(flagged)},
        "per_dataset": summary,
        "keep_uids": keep,
        "dropped": dropped,
        "flagged": flagged,
    }
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"rows {len(rows)}  kept {len(keep)}  dropped {len(dropped)}  flagged (kept) {len(flagged)}")
    for ds, info in summary.items():
        print(f"  {ds}: n={info['n']} kept={info['kept']} dropped={info['dropped'] or '-'} "
              f"flagged_rows={info['flagged_rows']} ({100 * info['flagged_rows'] / info['n']:.1f}%)")
        if info["flagged_fields"]:
            print(f"      flagged fields: {dict(list(info['flagged_fields'].items())[:8])}")
        if info["gt_artifact_mismatches"]:
            print(f"      GT-artifact mismatches (not flagged): {info['gt_artifact_mismatches']}")
    print(f"-> {out}")


def _uid_prefix(dataset: str) -> str:
    return {
        "cord-v2": "cord",
        "invoices-and-receipts-v2": "v2",
        "synthetic-receipts-ocr": "synthetic",
    }.get(dataset, dataset)


if __name__ == "__main__":
    main()
