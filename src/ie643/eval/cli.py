"""CLI: python -m ie643.eval --manifest Data/processed/test/manifest.jsonl --pred-key teacher_gt"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ie643.eval.match import EXTRACTION_FIELDS, META_FIELDS
from ie643.eval.score import evaluate, format_report
from ie643.schema import FIELD_ORDER


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score extraction JSON against source_gt (GT-present fields by default)"
    )
    parser.add_argument("--manifest", type=Path, default=Path("Data/processed/test/manifest.jsonl"))
    parser.add_argument("--pred-key", default="teacher_gt", help="teacher_gt or student_pred")
    parser.add_argument("--gt-key", default="source_gt")
    parser.add_argument("--datasets", default="", help="comma-separated names; default all")
    parser.add_argument(
        "--fields",
        default="",
        help="comma-separated canonical keys; default extraction fields (no document_type/language)",
    )
    parser.add_argument("--include-meta", action="store_true", help="also score document_type and language")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="count teacher-only fields (GT empty, pred filled) as false positives",
    )
    parser.add_argument(
        "--count-missing",
        action="store_true",
        help="rows without --pred-key count as empty predictions (FN on every GT field)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write JSON report (mismatches included)")
    parser.add_argument("--errors", type=int, default=0, help="print this many field mismatches")
    args = parser.parse_args()

    names = {n.strip() for n in args.datasets.split(",") if n.strip()} or None
    if args.fields:
        fields = tuple(n.strip() for n in args.fields.split(",") if n.strip())
        unknown = [f for f in fields if f not in FIELD_ORDER]
        if unknown:
            raise SystemExit(f"unknown fields {unknown}; valid: {list(FIELD_ORDER)}")
    else:
        fields = EXTRACTION_FIELDS + (tuple(sorted(META_FIELDS)) if args.include_meta else ())

    rows = load_jsonl(args.manifest)
    report = evaluate(
        rows,
        pred_key=args.pred_key,
        gt_key=args.gt_key,
        fields=fields,
        strict=args.strict,
        count_missing=args.count_missing,
        datasets=names,
    )
    print(format_report(report, max_errors=args.errors))
    if args.out:
        payload = dict(report)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
