"""Build Data/processed/{train,test} from frozen splits JSON (no publisher iterators)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ie643.preprocess.image import CanonicalImage, canonicalize_image, save_canonical
from ie643.preprocess.pipeline import write_jsonl
from ie643.schema import CANONICAL_SCHEMA, teacher_prompt


def _cell_bytes(cell: Any) -> bytes | None:
    if cell is None:
        return None
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b:
            return bytes(b)
        p = cell.get("path")
        if p and Path(p).exists():
            return Path(p).read_bytes()
        return None
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    return None


def load_split_records(
    splits_dir: Path,
    split: str,
    *,
    datasets: list[str] | None = None,
) -> list[dict[str, Any]]:
    index = json.loads((splits_dir / "index.json").read_text(encoding="utf-8"))
    names = datasets or list(index["datasets"])
    rows: list[dict[str, Any]] = []
    for name in names:
        info = index["datasets"].get(name)
        if not info:
            raise KeyError(f"unknown dataset {name}; index has {sorted(index['datasets'])}")
        payload = json.loads((splits_dir / info["file"]).read_text(encoding="utf-8"))
        if split not in payload:
            raise KeyError(f"{name} has no '{split}' split")
        rows.extend(payload[split])
    return rows


def _safe_stem(uid: str) -> str:
    return uid.replace("::", "__").replace(":", "_").replace("/", "_")


def _read_parquet_cells(path: Path, column: str, indices: list[int]) -> dict[int, Any]:
    want = set(indices)
    found: dict[int, Any] = {}
    pf = pq.ParquetFile(path)
    start = 0
    for batch in pf.iter_batches(batch_size=64, columns=[column]):
        n = len(batch)
        hits = [i for i in range(start, start + n) if i in want]
        if hits:
            col = batch.column(column)
            for i in hits:
                found[i] = col[i - start].as_py()
        start += n
        if len(found) == len(want):
            break
    missing = want - set(found)
    if missing:
        raise KeyError(f"{path.name} missing row_index {sorted(missing)[:8]}")
    return found


def _read_one(rec: dict[str, Any], data_root: Path) -> bytes:
    img = rec["image"]
    kind = img.get("kind")
    if kind == "file":
        path = data_root / img["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        return path.read_bytes()
    if kind == "parquet":
        cells = _read_parquet_cells(
            data_root / img["path"],
            img.get("column") or "image",
            [int(img["row_index"])],
        )
        raw = _cell_bytes(cells[int(img["row_index"])])
        if not raw:
            raise ValueError(f"{rec['uid']}: empty parquet image")
        return raw
    raise ValueError(f"{rec['uid']}: unknown image kind {kind!r}")


def _process_rec(rec: dict[str, Any], raw: bytes, out_dir: Path) -> dict[str, Any]:
    canon = canonicalize_image(raw)
    rel = Path("images") / rec["dataset"] / _safe_stem(rec["uid"])
    saved = save_canonical(canon, out_dir / rel)
    return _manifest_row(rec, canon, saved.relative_to(out_dir))


def _manifest_row(rec: dict[str, Any], canon: CanonicalImage, rel: Path) -> dict[str, Any]:
    return {
        "uid": rec["uid"],
        "dataset": rec["dataset"],
        "split": rec["split"],
        "source_split": rec.get("source_split"),
        "locale": rec.get("locale"),
        "image": str(rel).replace("\\", "/"),
        "orig_width": canon.orig_size[0],
        "orig_height": canon.orig_size[1],
        "width": canon.size[0],
        "height": canon.size[1],
        "mode_in": canon.mode_in,
        "doc_type_hint": rec.get("doc_type_hint"),
        "language_hint": rec.get("language_hint"),
        "source_gt": rec.get("source_gt"),
        "teacher_gt": None,
        "student_pred": None,
        "teacher_raw": None,
        "student_raw": None,
        "teacher_prompt": teacher_prompt(),
        "review_status": "pending",
        "review_gt": None,
        "status": "preprocessed",
    }


def pack_split(
    splits_dir: Path,
    split: str,
    out_dir: Path,
    *,
    data_root: Path,
    datasets: list[str] | None = None,
    limit: int | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    splits_dir = splits_dir.resolve()
    data_root = data_root.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_split_records(splits_dir, split, datasets=datasets)
    if limit is not None:
        records = records[:limit]

    existing: dict[str, dict[str, Any]] = {}
    man_path = out_dir / "manifest.jsonl"
    if resume and man_path.exists():
        for line in man_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["uid"]] = row

    todo = [r for r in records if not (resume and r["uid"] in existing)]
    print(f"pack {split}: {len(records)} uids, {len(todo)} to write", flush=True)

    done: dict[str, dict[str, Any]] = dict(existing)
    file_recs = [r for r in todo if r["image"].get("kind") == "file"]
    parquet_jobs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in todo:
        img = rec["image"]
        if img.get("kind") == "parquet":
            parquet_jobs[(img["path"], img.get("column") or "image")].append(rec)
        elif img.get("kind") != "file":
            raise ValueError(f"{rec['uid']}: unknown image kind {img.get('kind')!r}")

    n_written = 0
    total = len(todo)
    for rec in file_recs:
        done[rec["uid"]] = _process_rec(rec, _read_one(rec, data_root), out_dir)
        n_written += 1
        if n_written % 50 == 0 or n_written == total:
            print(f"  {n_written}/{total}", flush=True)

    for (rel, column), group in parquet_jobs.items():
        idxs = [int(r["image"]["row_index"]) for r in group]
        cells = _read_parquet_cells(data_root / rel, column, idxs)
        for rec in group:
            raw = _cell_bytes(cells[int(rec["image"]["row_index"])])
            if not raw:
                raise ValueError(f"{rec['uid']}: empty parquet image {rel}")
            done[rec["uid"]] = _process_rec(rec, raw, out_dir)
            n_written += 1
            if n_written % 50 == 0 or n_written == total:
                print(f"  {n_written}/{total}", flush=True)
        cells.clear()

    ordered = [done[r["uid"]] for r in records]
    write_jsonl(man_path, ordered)
    shards: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        shards[row["dataset"]].append(row)
    for name, shard in shards.items():
        write_jsonl(out_dir / "manifests" / f"{name}.jsonl", shard)
    (out_dir / "canonical_schema.json").write_text(
        json.dumps(CANONICAL_SCHEMA, ensure_ascii=False, indent=2) + "\n"
    )
    meta = {
        "split": split,
        "n": len(ordered),
        "splits_dir": str(splits_dir),
        "datasets": sorted(shards),
        "uids_from": "splits/*.json",
    }
    (out_dir / "pack.json").write_text(json.dumps(meta, indent=2) + "\n")
    return ordered
