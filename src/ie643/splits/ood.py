"""Evaluation-only splits for the external datasets (DocILE, MC-OCR, ReceiptBench).

Nothing here is trained on, so every split is labelled "test" (the pack tool only
knows train/test). splits/ and its leakage audit are untouched.

  python -m ie643.splits.ood --data-root external_dataset --out splits_ood
  python -m ie643.pack --split test --splits splits_ood --data-root external_dataset --out Data/processed/ood

- docile: train + val annotations (test labels are not public), single-page PDFs only,
  then a seeded random sample of --docile-n.
- mc-ocr: every row of mcocr_train_df.csv (the only labelled MC-OCR split).
- receiptbench: documents with at least one scorable field, seeded random sample of --receiptbench-n.

Records carry gt_alternatives: other distinct printed values of a field (see
ie643.preprocess.adapters.from_docile / from_mcocr / from_receiptbench); ie643.eval
accepts any of them.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from ie643.preprocess import adapters
from ie643.schema import FIELD_ORDER

VERSION = "ood-1.1"
SEED = 643
META_KEYS = {"document_type", "language"}


def _sample(recs: list[dict[str, Any]], n: int | None, seed: int) -> list[dict[str, Any]]:
    """Seeded random subset, kept in source order so the files diff cleanly."""
    if n is None or n >= len(recs):
        return recs
    keep = set(random.Random(seed).sample(range(len(recs)), n))
    return [r for i, r in enumerate(recs) if i in keep]


def _has_gt(gt: dict[str, Any]) -> bool:
    return any(gt.get(k) not in (None, "", []) for k in FIELD_ORDER if k not in META_KEYS)


def build_docile(root: Path, n: int | None = None, seed: int = SEED) -> dict[str, Any]:
    base = root / "docile"
    recs: list[dict[str, Any]] = []
    skipped = {"multi_page": 0, "no_fields": 0}
    for source_split in ("train", "val"):
        for doc_id in json.loads((base / f"{source_split}.json").read_text(encoding="utf-8")):
            ann = json.loads((base / "annotations" / f"{doc_id}.json").read_text(encoding="utf-8"))
            meta = ann.get("metadata") or {}
            if meta.get("page_count") != 1:
                skipped["multi_page"] += 1
                continue
            gt, alts = adapters.from_docile(ann)
            if not _has_gt(gt):
                skipped["no_fields"] += 1
                continue
            doc_type = meta.get("document_type") or "invoice"
            recs.append(
                {
                    "uid": f"docile::{doc_id}",
                    "dataset": "docile",
                    "split": "test",
                    "source_split": source_split,
                    "doc_type_hint": "invoice" if doc_type == "tax_invoice" else doc_type.replace("_", " "),
                    "language_hint": "en",
                    "image": {"kind": "pdf", "path": f"docile/pdfs/{doc_id}.pdf", "page": 0},
                    "source_gt": gt,
                    "gt_alternatives": alts,
                }
            )
    eligible = len(recs)
    recs = _sample(recs, n, seed)
    return {
        "dataset": "docile",
        "notes": [
            "DocILE annotated train+val (test labels not public); single-page PDFs only, rendered at 200 dpi.",
            "Field text from the human annotations, not the OCR. First box in reading order is source_gt.",
            f"Random sample of {len(recs)} of {eligible} single-page documents (seed {seed}).",
        ],
        "counts": {"test": len(recs), "eligible": eligible, "skipped": skipped},
        "test": recs,
    }


def build_mcocr(root: Path) -> dict[str, Any]:
    base = root / "mcocr_train_data"
    csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)
    recs: list[dict[str, Any]] = []
    skipped = {"missing_image": 0, "no_fields": 0}
    with (base / "mcocr_train_df.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            img = row["img_id"]
            if not (base / "train_images" / img).exists():
                skipped["missing_image"] += 1
                continue
            gt, alts = adapters.from_mcocr(row)
            if not _has_gt(gt):
                skipped["no_fields"] += 1
                continue
            recs.append(
                {
                    "uid": f"mc-ocr::{Path(img).stem}",
                    "dataset": "mc-ocr",
                    "split": "test",
                    "source_split": "train",
                    "doc_type_hint": "receipt",
                    "language_hint": "vi",
                    "image": {"kind": "file", "path": f"mcocr_train_data/train_images/{img}"},
                    "source_gt": gt,
                    "gt_alternatives": alts,
                }
            )
    return {
        "dataset": "mc-ocr",
        "notes": [
            "MC-OCR 2021 labelled train CSV (Vietnamese receipts): SELLER, ADDRESS, TIMESTAMP, TOTAL_COST.",
            "TIMESTAMP is split into date + time; TOTAL_COST keeps the amount box, not the label text.",
        ],
        "counts": {"test": len(recs), "skipped": skipped},
        "test": recs,
    }


def _humanize(category: str) -> str:
    """1_PurchaseTicket -> purchase ticket; 4_MLFlightTicket -> flight ticket."""
    name = re.sub(r"^ML", "", category.split("_", 1)[1])
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


def build_receiptbench(root: Path, n: int | None = None, seed: int = SEED) -> dict[str, Any]:
    base = root / "ReceiptBench"
    jdir = base / "ReceiptBench_json_v1" / "ReceiptBench_json_v1"
    idir = base / "ReceiptBench_image_v1" / "ReceiptBench_image_v1"
    files = [p for p in idir.iterdir() if p.is_file()]
    images = {p.stem: p.name for p in files if p.stat().st_size > 0}
    empty = {p.stem for p in files if p.stat().st_size == 0}  # 17 zero-byte files in v1
    recs: list[dict[str, Any]] = []
    skipped = {"missing_image": 0, "empty_image": 0, "no_fields": 0}
    for path in sorted(jdir.glob("*.json")):
        stem = path.stem
        if stem not in images:
            skipped["empty_image" if stem in empty else "missing_image"] += 1
            continue
        category = stem.rsplit("_", 1)[0]  # 1_PurchaseTicket_973 -> 1_PurchaseTicket
        gt, alts = adapters.from_receiptbench(json.loads(path.read_text(encoding="utf-8")), category)
        if not _has_gt(gt):
            skipped["no_fields"] += 1
            continue
        recs.append(
            {
                "uid": f"receiptbench::{stem}",
                "dataset": "receiptbench",
                "split": "test",
                "source_split": "all",
                "category": category,
                "doc_type_hint": _humanize(category),
                "language_hint": None,
                "image": {
                    "kind": "file",
                    "path": f"ReceiptBench/ReceiptBench_image_v1/ReceiptBench_image_v1/{images[stem]}",
                },
                "source_gt": gt,
                "gt_alternatives": alts,
            }
        )
    eligible = len(recs)
    recs = _sample(recs, n, seed)
    return {
        "dataset": "receiptbench",
        "notes": [
            "ReceiptBench v1 (receipts, tickets, bills; 51 categories). Scored fields: vendor (seller_name),"
            " date (orig_invoice_time), total (orig_total), currency (std_curr, only with a printed symbol/code),"
            " vendor_tax_id (tax_number).",
            "Not scored: seller_address (region, not printed), invoice_number (mixed id types), detail"
            " (items mixed with fees/taxes/subtotals), travel fields.",
            f"Random sample of {len(recs)} of {eligible} documents with a scorable field (seed {seed}).",
        ],
        "counts": {"test": len(recs), "eligible": eligible, "skipped": skipped},
        "test": recs,
    }


BUILDERS = {"docile": build_docile, "mc-ocr": build_mcocr, "receiptbench": build_receiptbench}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation-only splits for DocILE, MC-OCR and ReceiptBench")
    parser.add_argument("--data-root", type=Path, default=Path("external_dataset"))
    parser.add_argument("--out", type=Path, default=Path("splits_ood"))
    parser.add_argument("--datasets", default="", help="comma-separated; default docile,mc-ocr,receiptbench")
    parser.add_argument("--docile-n", type=int, default=2000, help="random sample size; 0 = all")
    parser.add_argument("--receiptbench-n", type=int, default=2000, help="random sample size; 0 = all")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    names = [n.strip() for n in args.datasets.split(",") if n.strip()] or list(BUILDERS)
    sizes = {"docile": args.docile_n or None, "receiptbench": args.receiptbench_n or None}

    args.out.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "version": VERSION,
        "seed": args.seed,
        "data_root": str(args.data_root),
        "protocol": {"evaluation_only": True, "split_label": "test", "sample_sizes": sizes},
        "datasets": {},
    }
    for name in names:
        build = BUILDERS[name]
        built = build(args.data_root) if name == "mc-ocr" else build(args.data_root, sizes[name], args.seed)
        payload = {"version": VERSION, **built}
        path = args.out / f"{name}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        index["datasets"][name] = {"file": path.name, "counts": payload["counts"], "notes": payload["notes"]}
        print(f"{name}: {payload['counts']} -> {path}")
    index["totals"] = {"test": sum(d["counts"]["test"] for d in index["datasets"].values())}
    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"total test {index['totals']['test']} -> {args.out / 'index.json'}")


if __name__ == "__main__":
    main()
