"""Build dataset-wise train/test JSON with official splits and no leakage.

Rules
-----
- If the source publishes train and test, use them. Validation/dev/eval is
  merged into test, never into train.
- Train records are sampled only from official train. Test records are sampled
  only from official test ∪ val.
- Synthetic-receipts-ocr: 3000 IT + 3000 FR + 1000 EN (US+UK) from train.
  Test target is 20% in the same locale mix, taken from official eval only.
  Eval is smaller than that target for IT/FR; we take every eval row for those
  locales rather than stealing train images.
- invoices-and-receipts-v2: take as many official-train rows as needed so
  the five-dataset train pool is 20_000. Official v2 train is only 2843, so
  we take all of it. Test is 20% of that train count from official valid ∪ test
  (capped at the official eval pool).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from ie643.preprocess import adapters
from ie643.schema import empty_canonical

VERSION = "1.1"
SEED = 643
DATASETS = (
    "sroie",
    "fatura2",
    "cord-v2",
    "synthetic-receipts-ocr",
    "invoices-and-receipts-v2",
)

SYN_TRAIN = {"IT": 3000, "FR": 3000, "EN": 1000}
SYN_TEST = {k: n // 5 for k, n in SYN_TRAIN.items()}  # 20%
# Other four datasets are fully determined; v2 fills toward this total.
TARGET_TRAIN_TOTAL = 20_000
OTHER_TRAIN_N = 727 + 7500 + 800 + 7000  # sroie + fatura + cord + synthetic
V2_TRAIN_TARGET = TARGET_TRAIN_TOTAL - OTHER_TRAIN_N
EN_LOCALES = {"US", "UK"}


def _rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _locator_key(rec: dict[str, Any]) -> tuple:
    img = rec["image"]
    if img["kind"] == "file":
        return ("file", img["path"])
    return ("parquet", img["path"], img["row_index"], img.get("column"))


def _fingerprint(rec: dict[str, Any]) -> str:
    payload = {
        "uid": rec["uid"],
        "locator": _locator_key(rec),
        "source_split": rec["source_split"],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _sample(rows: list[Any], n: int, rng: random.Random) -> list[Any]:
    if n >= len(rows):
        out = list(rows)
        rng.shuffle(out)
        return out
    return rng.sample(rows, n)


def _audit(train: list[dict], test: list[dict], *, train_sources: set[str], test_sources: set[str]) -> dict[str, Any]:
    tu, teu = {r["uid"] for r in train}, {r["uid"] for r in test}
    tl, tel = {_locator_key(r) for r in train}, {_locator_key(r) for r in test}
    tf, tef = {_fingerprint(r) for r in train}, {_fingerprint(r) for r in test}
    bad_train_src = sorted({r["source_split"] for r in train} - train_sources)
    bad_test_src = sorted({r["source_split"] for r in test} - test_sources)
    audit = {
        "uid_overlap": len(tu & teu),
        "locator_overlap": len(tl & tel),
        "fingerprint_overlap": len(tf & tef),
        "train_from_nontrain_source": bad_train_src,
        "test_from_non_eval_source": bad_test_src,
        "duplicate_uids_train": len(train) - len(tu),
        "duplicate_uids_test": len(test) - len(teu),
    }
    if audit["uid_overlap"] or audit["locator_overlap"] or audit["fingerprint_overlap"]:
        raise RuntimeError(f"data leakage detected: {audit}")
    if bad_train_src or bad_test_src:
        raise RuntimeError(f"source-split leak: {audit}")
    if audit["duplicate_uids_train"] or audit["duplicate_uids_test"]:
        raise RuntimeError(f"duplicate uids: {audit}")
    return audit


def _pack(
    dataset: str,
    *,
    train: list[dict],
    test: list[dict],
    protocol: dict[str, Any],
    train_sources: set[str],
    test_sources: set[str],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    audit = _audit(train, test, train_sources=train_sources, test_sources=test_sources)
    return {
        "dataset": dataset,
        "version": VERSION,
        "seed": protocol.get("seed", SEED),
        "protocol": protocol,
        "notes": notes or [],
        "counts": {
            "train": len(train),
            "test": len(test),
            "total": len(train) + len(test),
        },
        "leakage_audit": audit,
        "train": train,
        "test": test,
    }


CONTENT_DUPLICATES = "content_duplicates.json"


def load_content_exclusions(out_dir: Path) -> set[str]:
    """Train uids whose pixels duplicate a test image (see splits/content_duplicates.json)."""
    path = out_dir / CONTENT_DUPLICATES
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8"))["exclude_train_uids"])


def drop_content_duplicates(payload: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
    """Remove train records that duplicate a test image by content. Test is never touched."""
    keep = [r for r in payload["train"] if r["uid"] not in excluded]
    removed = len(payload["train"]) - len(keep)
    payload["train"] = keep
    payload["counts"] = {"train": len(keep), "test": len(payload["test"]), "total": len(keep) + len(payload["test"])}
    audit = payload["leakage_audit"]
    audit["content_duplicates_removed_from_train"] = audit.get("content_duplicates_removed_from_train", 0) + removed
    payload["leakage_audit"]["content_overlap"] = len(excluded & {r["uid"] for r in keep})
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write("\n")


def load_dataset_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sroie_gt(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw.startswith("{"):
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return adapters.from_sroie(payload)


def build_sroie(root: Path, rng: random.Random, seed: int) -> dict[str, Any]:
    train_dir = root / "SROIE" / "SROIE2019-20260919T110512Z-1-001" / "SROIE2019" / "0325updated.task2train(626p)"
    test_img = root / "SROIE" / "SROIE_test_images_task_3"
    test_gt = root / "SROIE" / "SROIE_test_gt_task_3"
    train, test = [], []
    skipped = 0
    for img in sorted(train_dir.glob("*.jpg")):
        gt = _sroie_gt(train_dir / f"{img.stem}.txt")
        if gt is None:
            skipped += 1
            continue
        train.append(
            {
                "uid": f"sroie::{img.stem}",
                "dataset": "sroie",
                "split": "train",
                "source_split": "train",
                "doc_type_hint": "receipt",
                "language_hint": "en",
                "image": {"kind": "file", "path": _rel(img, root)},
                "source_gt": gt,
            }
        )
    for img in sorted(test_img.glob("*.jpg")):
        gt = _sroie_gt(test_gt / f"{img.stem}.txt") or empty_canonical(language="en")
        test.append(
            {
                "uid": f"sroie::{img.stem}",
                "dataset": "sroie",
                "split": "test",
                "source_split": "test",
                "doc_type_hint": "receipt",
                "language_hint": "en",
                "image": {"kind": "file", "path": _rel(img, root)},
                "source_gt": gt,
            }
        )
    rng.shuffle(train)
    rng.shuffle(test)
    return _pack(
        "sroie",
        train=train,
        test=test,
        protocol={
            "seed": seed,
            "official_splits_used": True,
            "val_merged_into": None,
            "subsample": None,
            "skipped_train_images_without_json_gt": skipped,
        },
        train_sources={"train"},
        test_sources={"test"},
        notes=["ICDAR-2019 SROIE task 3. Official train/test; no val split."],
    )


def build_fatura(root: Path, rng: random.Random, seed: int) -> dict[str, Any]:
    base = root / "FATURA2" / "invoices_dataset_final"
    mapping = {
        "train": ("strat1_train.csv", "train", "train"),
        "val": ("strat1_dev.csv", "val", "test"),
        "test": ("strat1_test.csv", "test", "test"),
    }
    train, test = [], []
    for _key, (csv_name, source_split, assigned) in mapping.items():
        for line in (base / csv_name).read_text(encoding="utf-8").splitlines()[1:]:
            if not line.strip():
                continue
            img_name, ann_name = [p.strip() for p in line.split(",", 1)]
            img = base / "images" / img_name
            ann = json.loads((base / "Annotations" / "Original_Format" / ann_name).read_text())
            rec = {
                "uid": f"fatura2::{img.stem}",
                "dataset": "fatura2",
                "split": assigned,
                "source_split": source_split,
                "doc_type_hint": "invoice",
                "language_hint": "en",
                "image": {"kind": "file", "path": _rel(img, root)},
                "source_gt": adapters.from_fatura(ann),
            }
            (train if assigned == "train" else test).append(rec)
    rng.shuffle(train)
    rng.shuffle(test)
    return _pack(
        "fatura2",
        train=train,
        test=test,
        protocol={
            "seed": seed,
            "official_splits_used": True,
            "val_merged_into": "test",
            "subsample": None,
            "csv": {
                "train": "strat1_train.csv",
                "val": "strat1_dev.csv",
                "test": "strat1_test.csv",
            },
        },
        train_sources={"train"},
        test_sources={"val", "test"},
        notes=["FATURA official strat1 split. Dev/val merged into test."],
    )


def _iter_parquet(files: Iterable[Path], columns: list[str]):
    for path in files:
        pf = pq.ParquetFile(path)
        row_index = 0
        have = [c for c in columns if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=32, columns=have):
            d = batch.to_pydict()
            n = len(next(iter(d.values())))
            for i in range(n):
                yield path, row_index, {k: d[k][i] for k in d}
                row_index += 1


def build_cord(root: Path, rng: random.Random, seed: int) -> dict[str, Any]:
    data = root / "naver-clova-ix" / "cord-v2" / "data"
    train, test = [], []
    split_map = {"train": ("train", "train"), "validation": ("val", "test"), "test": ("test", "test")}
    for source_name, (source_split, assigned) in split_map.items():
        for path, row_index, row in _iter_parquet(sorted(data.glob(f"{source_name}-*.parquet")), ["image", "ground_truth"]):
            parsed = json.loads(row["ground_truth"]) if isinstance(row["ground_truth"], str) else row["ground_truth"]
            image_id = (parsed.get("meta") or {}).get("image_id", row_index)
            rec = {
                "uid": f"cord::{source_split}:{image_id}",
                "dataset": "cord-v2",
                "split": assigned,
                "source_split": source_split,
                "doc_type_hint": "receipt",
                "language_hint": "id",
                "image": {
                    "kind": "parquet",
                    "path": _rel(path, root),
                    "column": "image",
                    "row_index": row_index,
                },
                "source_gt": adapters.from_cord(parsed),
            }
            (train if assigned == "train" else test).append(rec)
    rng.shuffle(train)
    rng.shuffle(test)
    return _pack(
        "cord-v2",
        train=train,
        test=test,
        protocol={
            "seed": seed,
            "official_splits_used": True,
            "val_merged_into": "test",
            "subsample": None,
            "note_ids": "CORD image_id is per-split (train 0-799, val/test 0-99). uids include source_split.",
        },
        train_sources={"train"},
        test_sources={"val", "test"},
        notes=["CORD-v2 official HuggingFace split. Validation merged into test."],
    )


def _syn_lang(locale: str) -> str | None:
    loc = str(locale).upper()
    if loc in EN_LOCALES:
        return "EN"
    if loc in {"IT", "FR"}:
        return loc
    return None


def _syn_hint(locale: str) -> str:
    return {"US": "en", "UK": "en", "FR": "fr", "IT": "it", "DE": "de"}.get(str(locale).upper(), "en")


def build_synthetic(root: Path, rng: random.Random, seed: int) -> dict[str, Any]:
    data = root / "albertobarnabo" / "synthetic-receipts-ocr" / "data"
    by_lang_train: dict[str, list[tuple]] = defaultdict(list)
    by_lang_eval: dict[str, list[tuple]] = defaultdict(list)
    meta_cols = ["id", "locale"]
    for path, row_index, row in _iter_parquet(sorted(data.glob("train-*.parquet")), meta_cols):
        lang = _syn_lang(row.get("locale"))
        if lang:
            by_lang_train[lang].append((str(row["id"]), lang, str(row.get("locale")), path, row_index, "train"))
    for path, row_index, row in _iter_parquet(sorted(data.glob("eval-*.parquet")), meta_cols):
        lang = _syn_lang(row.get("locale"))
        if lang:
            by_lang_eval[lang].append((str(row["id"]), lang, str(row.get("locale")), path, row_index, "eval"))

    chosen: dict[str, tuple] = {}
    actual_test = {}
    for lang, n in SYN_TRAIN.items():
        pool = by_lang_train[lang]
        if len(pool) < n:
            raise RuntimeError(f"synthetic train {lang}: need {n}, have {len(pool)}")
        for item in _sample(pool, n, rng):
            chosen[item[0]] = ("train", item)
    notes = []
    for lang, n in SYN_TEST.items():
        pool = by_lang_eval[lang]
        take = min(n, len(pool))
        if take < n:
            notes.append(
                f"Official eval has {len(pool)} {lang} receipts; requested test {n}. "
                "Using all eval rows instead of pulling from train."
            )
        actual_test[lang] = take
        for item in _sample(pool, take, rng):
            if item[0] in chosen:
                raise RuntimeError(f"synthetic id {item[0]} in both train pool and eval")
            chosen[item[0]] = ("test", item)

    # Second pass: attach fields/full_text for selected ids only.
    need = set(chosen)
    loaded: dict[str, dict] = {}
    field_cols = ["id", "locale", "fields", "full_text"]
    for glob in ("train-*.parquet", "eval-*.parquet"):
        for path, row_index, row in _iter_parquet(sorted(data.glob(glob)), field_cols):
            sid = str(row["id"])
            if sid not in need or sid in loaded:
                continue
            loaded[sid] = row
            if len(loaded) == len(need):
                break
        if len(loaded) == len(need):
            break
    missing = need - set(loaded)
    if missing:
        raise RuntimeError(f"synthetic missing fields for {len(missing)} ids")

    train, test = [], []
    for sid, (assigned, item) in chosen.items():
        _sid, lang, locale, path, row_index, source_split = item
        row = loaded[sid]
        rec = {
            "uid": f"synthetic::{sid}",
            "dataset": "synthetic-receipts-ocr",
            "split": assigned,
            "source_split": source_split,
            "locale": locale,
            "doc_type_hint": "receipt",
            "language_hint": _syn_hint(locale),
            "image": {
                "kind": "parquet",
                "path": _rel(path, root),
                "column": "image_photo",
                "row_index": row_index,
            },
            "source_gt": adapters.from_synthetic(row.get("fields"), row.get("full_text")),
        }
        (train if assigned == "train" else test).append(rec)
    rng.shuffle(train)
    rng.shuffle(test)
    return _pack(
        "synthetic-receipts-ocr",
        train=train,
        test=test,
        protocol={
            "seed": seed,
            "official_splits_used": True,
            "val_merged_into": "test",
            "english_locales": sorted(EN_LOCALES),
            "subsample": {
                "train_requested": SYN_TRAIN,
                "test_requested": SYN_TEST,
                "test_actual": actual_test,
                "available_train": {k: len(v) for k, v in by_lang_train.items()},
                "available_eval": {k: len(v) for k, v in by_lang_eval.items()},
            },
        },
        train_sources={"train"},
        test_sources={"eval"},
        notes=notes
        or ["Synthetic receipts: train subsample from official train; test from official eval only."],
    )


def build_v2(root: Path, rng: random.Random, seed: int) -> dict[str, Any]:
    data = root / "mychen76" / "invoices-and-receipts_ocr_v2" / "data"
    train_pool, test_pool = [], []
    split_files = {
        "train": ("train", "train"),
        "valid": ("val", "test"),
        "test": ("test", "test"),
    }
    for source_name, (source_split, assigned) in split_files.items():
        for path, row_index, row in _iter_parquet(
            sorted(data.glob(f"{source_name}-*.parquet")), ["id", "parsed_data"]
        ):
            sid = row["id"]
            rec = {
                "uid": f"v2::{source_split}:{sid}:{row_index}",
                "dataset": "invoices-and-receipts-v2",
                "split": assigned,
                "source_split": source_split,
                "native_id": sid,
                "doc_type_hint": "receipt",
                "language_hint": "en",
                "image": {
                    "kind": "parquet",
                    "path": _rel(path, root),
                    "column": "image",
                    "row_index": row_index,
                },
                "source_gt": adapters.from_mychen_v2(row.get("parsed_data")),
            }
            (train_pool if assigned == "train" else test_pool).append(rec)
    train_n = min(V2_TRAIN_TARGET, len(train_pool))
    train = _sample(train_pool, train_n, rng)
    test_requested = max(1, train_n // 5)
    test_n = min(test_requested, len(test_pool))
    test = _sample(test_pool, test_n, rng)
    notes = [
        "v2 native `id` is a per-file index, not a global receipt id. uids include source_split and row_index.",
        f"Target overall train is {TARGET_TRAIN_TOTAL}; other datasets contribute {OTHER_TRAIN_N}, so v2 target is {V2_TRAIN_TARGET}.",
    ]
    if train_n < V2_TRAIN_TARGET:
        notes.append(
            f"Official v2 train only has {len(train_pool)} rows; using all of them "
            f"(overall train becomes {OTHER_TRAIN_N + train_n}, not {TARGET_TRAIN_TOTAL})."
        )
    if test_n < test_requested:
        notes.append(
            f"Requested {test_requested} test rows (20% of v2 train), official valid∪test only has {len(test_pool)}."
        )
    return _pack(
        "invoices-and-receipts-v2",
        train=train,
        test=test,
        protocol={
            "seed": seed,
            "official_splits_used": True,
            "val_merged_into": "test",
            "subsample": {
                "overall_train_target": TARGET_TRAIN_TOTAL,
                "other_datasets_train": OTHER_TRAIN_N,
                "train_requested": V2_TRAIN_TARGET,
                "train_actual": train_n,
                "test_requested": test_requested,
                "train_pool": len(train_pool),
                "test_pool": len(test_pool),
            },
        },
        train_sources={"train"},
        test_sources={"val", "test"},
        notes=notes,
    )


BUILDERS = {
    "sroie": build_sroie,
    "fatura2": build_fatura,
    "cord-v2": build_cord,
    "synthetic-receipts-ocr": build_synthetic,
    "invoices-and-receipts-v2": build_v2,
}


def _index_entry(fname: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": fname,
        "counts": payload["counts"],
        "protocol": payload["protocol"],
        "notes": payload["notes"],
        "leakage_audit": payload["leakage_audit"],
    }


def apply_content_exclusions(out_dir: Path) -> dict[str, Any]:
    """Drop content-duplicate train uids from already-built split JSONs without rebuilding."""
    out_dir = out_dir.resolve()
    excluded = load_content_exclusions(out_dir)
    index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    for name, info in index["datasets"].items():
        payload = drop_content_duplicates(load_dataset_json(out_dir / info["file"]), excluded)
        _write_json(out_dir / info["file"], payload)
        index["datasets"][name] = _index_entry(info["file"], payload)
    index["totals"]["train"] = sum(d["counts"]["train"] for d in index["datasets"].values())
    index["totals"]["test"] = sum(d["counts"]["test"] for d in index["datasets"].values())
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    return index


def build_all(
    data_root: Path,
    out_dir: Path,
    *,
    seed: int = SEED,
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    data_root = data_root.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(datasets or DATASETS)
    unknown = [n for n in names if n not in BUILDERS]
    if unknown:
        raise KeyError(f"unknown datasets {unknown}; choose from {list(BUILDERS)}")
    index: dict[str, Any] = {
        "version": VERSION,
        "seed": seed,
        "data_root": str(data_root),
        "protocol": {
            "train_only_from_official_train": True,
            "val_merged_into_test": True,
            "never_put_test_images_in_train": True,
        },
        "datasets": {},
    }
    excluded = load_content_exclusions(out_dir)
    for name in names:
        rng = random.Random(seed)
        print(f"building {name} ...")
        payload = drop_content_duplicates(BUILDERS[name](data_root, rng, seed), excluded)
        fname = f"{name}.json"
        _write_json(out_dir / fname, payload)
        index["datasets"][name] = _index_entry(fname, payload)
        print(f"  {name}: train={payload['counts']['train']} test={payload['counts']['test']}")
    index["totals"] = {
        "train": sum(d["counts"]["train"] for d in index["datasets"].values()),
        "test": sum(d["counts"]["test"] for d in index["datasets"].values()),
        "target_train": TARGET_TRAIN_TOTAL,
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    return index
