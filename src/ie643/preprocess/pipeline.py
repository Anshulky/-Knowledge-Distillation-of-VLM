"""End-to-end preprocess: ingest → canonical image + schema → frozen files."""

from __future__ import annotations

import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
from PIL import Image

from ie643.preprocess import adapters
from ie643.preprocess.adapters import strip_private
from ie643.preprocess.image import CanonicalImage, canonicalize_image, save_canonical
from ie643.schema import empty_canonical, teacher_prompt

DATA_ROOT = Path("/backup/Anshul/IE643/Data")
DEFAULT_SPLITS = {
    "sroie": "test",
    "cord-v2": "train",
    "receiptqa": "validation",
    "receipt-ocr-ru": "train",
    "docjay131": "train",
    "fatura2": "train",
    "voxel51": "train",
    "synthetic-receipts-ocr": "train",
    "invoices-and-receipts-v2": "train",
    "invoices-google-ocr": "train",
    "nano-receipts": "train",
    "wildreceipts": "train",
}


@dataclass
class Sample:
    uid: str
    dataset: str
    split: str
    image: Image.Image | bytes | Path
    source_gt: dict[str, Any]
    doc_type_hint: str
    language_hint: str


def _cell_bytes(cell: Any) -> bytes | None:
    if isinstance(cell, dict):
        return cell.get("bytes")
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    return None


def _shuffle(items: list[Any], seed: int | None) -> list[Any]:
    if seed is None:
        return items
    out = list(items)
    random.Random(seed).shuffle(out)
    return out


def iter_sroie(root: Path = DATA_ROOT, split: str = "test", seed: int | None = None, **_: Any) -> Iterator[Sample]:
    img_dir = root / "SROIE" / "SROIE_test_images_task_3"
    gt_dir = root / "SROIE" / "SROIE_test_gt_task_3"
    t2 = root / "SROIE" / "SROIE2019-20260919T110512Z-1-001" / "SROIE2019" / "0325updated.task2train(626p)"
    if split == "train":
        img_dir, gt_dir = t2, t2
    for img in _shuffle(sorted(img_dir.glob("*.jpg")), seed):
        gt_path = gt_dir / f"{img.stem}.txt"
        payload = json.loads(gt_path.read_text()) if gt_path.exists() else {}
        yield Sample(
            uid=f"sroie::{img.stem}",
            dataset="sroie",
            split=split,
            image=img,
            source_gt=adapters.from_sroie(payload) if payload else empty_canonical(language="en"),
            doc_type_hint="receipt",
            language_hint="en",
        )


def iter_cord(root: Path = DATA_ROOT, split: str = "train", seed: int | None = None, **_: Any) -> Iterator[Sample]:
    files = sorted((root / "naver-clova-ix" / "cord-v2" / "data").glob(f"{split}-*.parquet"))
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=16, columns=["image", "ground_truth"]):
            d = batch.to_pydict()
            for i, gt in enumerate(d["ground_truth"]):
                b = _cell_bytes(d["image"][i])
                if not b:
                    continue
                parsed = json.loads(gt) if isinstance(gt, str) else gt
                image_id = (parsed.get("meta") or {}).get("image_id", i)
                yield Sample(
                    uid=f"cord::{split}:{image_id}",
                    dataset="cord-v2",
                    split=split,
                    image=b,
                    source_gt=adapters.from_cord(parsed),
                    doc_type_hint="receipt",
                    language_hint="id",
                )


def iter_receiptqa(root: Path = DATA_ROOT, split: str = "train", seed: int | None = None, **_: Any) -> Iterator[Sample]:
    img_zip = root / "ReceiptQA" / f"{split}_images.zip"
    lab_zip = root / "ReceiptQA" / f"{split}_label.zip"
    prefix_img = f"{split}_images/"
    prefix_lab = f"{split}_label/"
    with zipfile.ZipFile(img_zip) as zi, zipfile.ZipFile(lab_zip) as zl:
        labels = {Path(n).stem: n for n in zl.namelist() if n.endswith(".json")}
        names = [n for n in zi.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        for name in _shuffle(names, seed):
            stem = Path(name).stem
            lab_name = labels.get(stem)
            qa = json.loads(zl.read(lab_name)) if lab_name else []
            yield Sample(
                uid=f"receiptqa::{stem}",
                dataset="receiptqa",
                split="val" if split == "validation" else split,
                image=zi.read(name),
                source_gt=adapters.from_receiptqa(qa),
                doc_type_hint="receipt",
                language_hint="en",
            )


def iter_cdek(root: Path = DATA_ROOT, split: str = "train", seed: int | None = None, **_: Any) -> Iterator[Sample]:
    split_map = {"val": "validation", "valid": "validation", "validation": "validation"}
    disk_split = split_map.get(split, split)
    ann = root / "cdek-ocr" / "receipt-ocr-ru" / "annotations" / f"{disk_split}.jsonl"
    img_root = root / "cdek-ocr" / "receipt-ocr-ru" / "images" / disk_split
    rows = [json.loads(line) for line in ann.read_text().splitlines() if line.strip()]
    for row in _shuffle(rows, seed):
        image_id = row["image_id"]
        matches = list(img_root.glob(f"{image_id}.*"))
        if not matches:
            continue
        yield Sample(
            uid=f"cdek::{image_id}",
            dataset="receipt-ocr-ru",
            split="val" if disk_split == "validation" else split,
            image=matches[0],
            source_gt=adapters.from_cdek(row),
            doc_type_hint="receipt",
            language_hint="ru",
        )


def iter_docjay(root: Path = DATA_ROOT, split: str = "train", seed: int | None = None, **_: Any) -> Iterator[Sample]:
    files = sorted((root / "docjay131" / "receipts-ocr-dataset" / "data").glob("*.parquet"))
    idx = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=16, columns=["image", "text"]):
            d = batch.to_pydict()
            for i, text in enumerate(d["text"]):
                b = _cell_bytes(d["image"][i])
                if not b:
                    continue
                yield Sample(
                    uid=f"docjay::{idx}",
                    dataset="docjay131",
                    split=split,
                    image=b,
                    source_gt=adapters.from_docjay(text),
                    doc_type_hint="receipt",
                    language_hint="en",
                )
                idx += 1


def iter_fatura(root: Path = DATA_ROOT, split: str = "train", seed: int | None = None, **_: Any) -> Iterator[Sample]:
    base = root / "FATURA2" / "invoices_dataset_final"
    csv = base / {"train": "strat1_train.csv", "val": "strat1_dev.csv", "test": "strat1_test.csv"}[split]
    lines = _shuffle(csv.read_text().splitlines()[1:], seed)
    for line in lines:
        img_name, ann_name = [p.strip() for p in line.split(",", 1)]
        img = base / "images" / img_name
        ann = json.loads((base / "Annotations" / "Original_Format" / ann_name).read_text())
        yield Sample(
            uid=f"fatura2::{img.stem}",
            dataset="fatura2",
            split=split,
            image=img,
            source_gt=adapters.from_fatura(ann),
            doc_type_hint="invoice",
            language_hint="en",
        )


def iter_voxel51(root: Path = DATA_ROOT, split: str = "train", seed: int | None = None, **_: Any) -> Iterator[Sample]:
    base = root / "Voxel51" / "high-quality-invoice-images-for-ocr"
    samples = json.loads((base / "samples.json").read_text())["samples"]
    labeled = [s for s in samples if s.get("json_annotation")]
    unlabeled = [s for s in samples if not s.get("json_annotation")]
    ordered = _shuffle(labeled, seed) + _shuffle(unlabeled, None if seed is None else seed + 1)
    for s in ordered:
        rel = s.get("filepath") or ""
        img = base / rel
        if not img.exists():
            continue
        has_gt = bool(s.get("json_annotation"))
        # Unlabeled images stay in the pool for teacher GT generation on train only.
        yield Sample(
            uid=f"voxel51::{img.stem}",
            dataset="voxel51",
            split="labeled" if has_gt else "unlabeled",
            image=img,
            source_gt=adapters.from_voxel51(s.get("json_annotation")),
            doc_type_hint="invoice",
            language_hint="en",
        )


def iter_synthetic(root: Path = DATA_ROOT, split: str = "train", seed: int | None = None, **_: Any) -> Iterator[Sample]:
    name = "eval" if split in {"val", "valid", "eval"} else split
    files = sorted(
        (root / "albertobarnabo" / "synthetic-receipts-ocr" / "data").glob(f"{name}-*.parquet")
    )
    for f in files:
        pf = pq.ParquetFile(f)
        cols = [c for c in ["id", "image_photo", "fields", "full_text", "locale"] if c in [x.name for x in pf.schema_arrow]]
        for batch in pf.iter_batches(batch_size=8, columns=cols):
            d = batch.to_pydict()
            n = len(d["id"])
            for i in range(n):
                b = _cell_bytes(d["image_photo"][i])
                if not b:
                    continue
                locale = str(d.get("locale", ["US"])[i])
                lang = {"US": "en", "UK": "en", "DE": "de", "FR": "fr", "IT": "it"}.get(locale, "en")
                yield Sample(
                    uid=f"synthetic::{d['id'][i]}",
                    dataset="synthetic-receipts-ocr",
                    split="val" if name == "eval" else split,
                    image=b,
                    source_gt=adapters.from_synthetic(d["fields"][i], d.get("full_text", [None])[i]),
                    doc_type_hint="receipt",
                    language_hint=lang,
                )


def iter_v2(root: Path = DATA_ROOT, split: str = "train", seed: int | None = None, **_: Any) -> Iterator[Sample]:
    name = "valid" if split in {"val", "valid"} else split
    files = sorted((root / "mychen76" / "invoices-and-receipts_ocr_v2" / "data").glob(f"{name}-*.parquet"))
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=8, columns=["image", "id", "parsed_data"]):
            d = batch.to_pydict()
            for i, sid in enumerate(d["id"]):
                b = _cell_bytes(d["image"][i])
                if not b:
                    continue
                yield Sample(
                    uid=f"v2::{split}:{sid}:{i}",
                    dataset="invoices-and-receipts-v2",
                    split="val" if name == "valid" else split,
                    image=b,
                    source_gt=adapters.from_mychen_v2(d["parsed_data"][i]),
                    doc_type_hint="receipt",
                    language_hint="en",
                )


def iter_amaye(root: Path = DATA_ROOT, split: str = "train", seed: int | None = None, **_: Any) -> Iterator[Sample]:
    files = sorted((root / "amaye15" / "invoices-google-ocr" / "data").glob(f"{split}-*.parquet"))
    idx = 0
    for f in files:
        pf = pq.ParquetFile(f)
        cols = [c for c in ["pixel_values", "label"] if c in [x.name for x in pf.schema_arrow]]
        for batch in pf.iter_batches(batch_size=8, columns=cols):
            d = batch.to_pydict()
            imgs = d.get("pixel_values") or []
            for i, cell in enumerate(imgs):
                b = _cell_bytes(cell)
                if not b:
                    continue
                yield Sample(
                    uid=f"amaye::{split}:{idx}",
                    dataset="invoices-google-ocr",
                    split=split,
                    image=b,
                    source_gt=empty_canonical(document_type="invoice", language="en"),
                    doc_type_hint="invoice",
                    language_hint="en",
                )
                idx += 1


def iter_nano(root: Path = DATA_ROOT, split: str = "train", seed: int | None = None, **_: Any) -> Iterator[Sample]:
    files = sorted((root / "34data" / "nano-receipts" / "data").glob("*.parquet"))
    for f in files:
        pf = pq.ParquetFile(f)
        cols = [c for c in ["image", "filename", "receipt_id"] if c in [x.name for x in pf.schema_arrow]]
        for batch in pf.iter_batches(batch_size=8, columns=cols):
            d = batch.to_pydict()
            n = len(next(iter(d.values())))
            for i in range(n):
                b = _cell_bytes(d["image"][i])
                if not b:
                    continue
                rid = d.get("receipt_id", [i])[i]
                yield Sample(
                    uid=f"nano::{rid}",
                    dataset="nano-receipts",
                    split=split,
                    image=b,
                    source_gt=empty_canonical(document_type="receipt", language="en"),
                    doc_type_hint="receipt",
                    language_hint="en",
                )


def iter_wild(root: Path = DATA_ROOT, split: str = "train", seed: int | None = None, **_: Any) -> Iterator[Sample]:
    name = "valid" if split in {"val", "valid"} else split
    files = sorted((root / "mychen76" / "wildreceipts_ocr_v1" / "data").glob(f"{name}-*.parquet"))
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=8, columns=["image", "id", "parsed_data"]):
            d = batch.to_pydict()
            for i, sid in enumerate(d["id"]):
                b = _cell_bytes(d["image"][i])
                if not b:
                    continue
                yield Sample(
                    uid=f"wild::{split}:{sid}:{i}",
                    dataset="wildreceipts",
                    split="val" if name == "valid" else split,
                    image=b,
                    source_gt=adapters.from_mychen_v2(d["parsed_data"][i]),
                    doc_type_hint="receipt",
                    language_hint="en",
                )


REGISTRY = {
    "sroie": iter_sroie,
    "cord-v2": iter_cord,
    "receiptqa": iter_receiptqa,
    "receipt-ocr-ru": iter_cdek,
    "docjay131": iter_docjay,
    "fatura2": iter_fatura,
    "voxel51": iter_voxel51,
    "synthetic-receipts-ocr": iter_synthetic,
    "invoices-and-receipts-v2": iter_v2,
    "wildreceipts": iter_wild,
    "invoices-google-ocr": iter_amaye,
    "nano-receipts": iter_nano,
}


def process_sample(sample: Sample, out_dir: Path) -> dict[str, Any]:
    canon: CanonicalImage = canonicalize_image(sample.image)
    rel = Path("images") / sample.dataset / sample.uid.replace("::", "__").replace(":", "_")
    saved = save_canonical(canon, out_dir / rel)
    source_qa = sample.source_gt.get("_qa_pairs")
    source_gt = strip_private(sample.source_gt)
    record = {
        "uid": sample.uid,
        "dataset": sample.dataset,
        "split": sample.split,
        "image": str(saved.relative_to(out_dir)),
        "orig_width": canon.orig_size[0],
        "orig_height": canon.orig_size[1],
        "width": canon.size[0],
        "height": canon.size[1],
        "mode_in": canon.mode_in,
        "doc_type_hint": sample.doc_type_hint,
        "language_hint": sample.language_hint,
        "source_gt": source_gt,
        "teacher_gt": None,
        "review_status": "pending",
        "review_gt": None,
        "teacher_prompt": teacher_prompt(sample.doc_type_hint, sample.language_hint),
        "status": "preprocessed",
    }
    if source_qa is not None:
        record["source_qa"] = source_qa
    return record


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_review_subset(manifest: list[dict[str, Any]], n: int = 100, seed: int = 643) -> list[dict[str, Any]]:
    by_ds: dict[str, list[dict[str, Any]]] = {}
    for rec in manifest:
        by_ds.setdefault(rec["dataset"], []).append(rec)
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    datasets = sorted(by_ds)
    if not datasets:
        return []
    per = max(1, n // len(datasets))
    for name in datasets:
        pool = list(by_ds[name])
        rng.shuffle(pool)
        picked.extend(pool[:per])
    if len(picked) > n:
        rng.shuffle(picked)
        picked = picked[:n]
    elif len(picked) < n:
        leftover = [r for r in manifest if r["uid"] not in {p["uid"] for p in picked}]
        rng.shuffle(leftover)
        picked.extend(leftover[: n - len(picked)])
    return picked


def run(
    datasets: list[str],
    out_dir: Path,
    *,
    split: str = "auto",
    limit: int | None = None,
    seed: int | None = 643,
    data_root: Path = DATA_ROOT,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for name in datasets:
        if name not in REGISTRY:
            raise KeyError(f"unknown dataset {name}; choose from {sorted(REGISTRY)}")
        ds_split = DEFAULT_SPLITS.get(name, "train") if split in {None, "auto"} else split
        n = 0
        shard: list[dict[str, Any]] = []
        for sample in REGISTRY[name](data_root, split=ds_split, seed=seed):
            rec = process_sample(sample, out_dir)
            shard.append(rec)
            manifest.append(rec)
            n += 1
            if limit is not None and n >= limit:
                break
        write_jsonl(out_dir / "manifests" / f"{name}.jsonl", shard)
        print(f"{name}: {len(shard)} ({ds_split})")
    shards_dir = out_dir / "manifests"
    combined: list[dict[str, Any]] = []
    if shards_dir.exists():
        for path in sorted(shards_dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        combined.append(json.loads(line))
    else:
        combined = manifest
    write_jsonl(out_dir / "manifest.jsonl", combined)
    review = make_review_subset(combined, n=min(100, len(combined)), seed=seed or 643)
    write_jsonl(out_dir / "review_100.jsonl", review)
    return combined
