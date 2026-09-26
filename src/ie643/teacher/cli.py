"""Infer canonical JSON with Qwen3-VL-32B via vLLM.

Start the server once GPUs are free:

  vllm serve Qwen/Qwen3-VL-32B-Instruct --tensor-parallel-size 2 --max-model-len 16384

Then:

  python -m ie643.teacher --manifest Data/processed/teacher100/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ie643.preprocess.pipeline import write_jsonl
from ie643.schema import FIELD_ORDER, coerce_canonical, teacher_prompt
from ie643.teacher.client import VLLMClient
from ie643.teacher.parse import parse_teacher_output


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_record(client: VLLMClient, rec: dict, root: Path, *, max_tokens: int) -> dict:
    image = root / rec["image"]
    prompt = teacher_prompt(rec.get("doc_type_hint"), rec.get("language_hint"))
    rec["teacher_prompt"] = prompt
    raw = client.extract(image, prompt, max_tokens=max_tokens)
    rec["teacher_raw"] = raw
    rec["teacher_gt"] = parse_teacher_output(raw, fallback=rec.get("source_gt"))
    rec["status"] = "teacher"
    rec["teacher_error"] = None
    return rec


def dummy_record(rec: dict) -> dict:
    rec["teacher_gt"] = coerce_canonical(rec.get("source_gt") or {})
    rec["teacher_raw"] = None
    rec["status"] = "teacher_dummy"
    rec["teacher_error"] = None
    return rec


def _field_diffs(old: dict, new: dict) -> list[str]:
    diffs = []
    for key in FIELD_ORDER:
        a, b = old.get(key), new.get(key)
        if a != b:
            diffs.append(key)
    return diffs


def compare_penalty(client: VLLMClient, rows: list[dict], root: Path, *, n: int, max_tokens: int) -> None:
    """Re-infer successful rows with the current sampling settings; do not write teacher_gt."""
    successes = [r for r in rows if r.get("status") == "teacher" and r.get("teacher_gt")]
    by_ds: dict[str, list[dict]] = {}
    for rec in successes:
        by_ds.setdefault(rec["dataset"], []).append(rec)
    picked: list[dict] = []
    datasets = sorted(by_ds)
    i = 0
    while len(picked) < min(n, len(successes)) and datasets:
        ds = datasets[i % len(datasets)]
        if by_ds[ds]:
            picked.append(by_ds[ds].pop(0))
        i += 1
        if i > n * 8:
            break
    reports = []
    changed = 0
    out = root / "compare_repetition_penalty.jsonl"
    for rec in picked:
        baseline = rec.get("teacher_gt") or {}
        try:
            raw = client.extract(
                root / rec["image"],
                teacher_prompt(rec.get("doc_type_hint"), rec.get("language_hint")),
                max_tokens=max_tokens,
            )
            penalized = parse_teacher_output(raw, fallback=rec.get("source_gt"))
            err = None
        except Exception as exc:  # noqa: BLE001
            penalized = None
            err = str(exc)
        diffs = _field_diffs(baseline, penalized) if penalized is not None else ["<error>"]
        if diffs:
            changed += 1
        reports.append(
            {
                "uid": rec["uid"],
                "dataset": rec["dataset"],
                "changed_fields": diffs,
                "error": err,
                "baseline": baseline,
                "penalized": penalized,
            }
        )
        write_jsonl(out, reports)
        print(f"{rec['uid']}: {'CHANGED ' + ','.join(diffs) if diffs else 'identical'}")
    print(f"compare {len(reports)} images, {changed} changed -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL-32B teacher extraction")
    parser.add_argument("--manifest", type=Path, default=Path("Data/processed/teacher100/manifest.jsonl"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--backend", choices=("vllm", "dummy"), default="vllm")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--errors-only", action="store_true", help="retry rows with status teacher_error")
    parser.add_argument("--uids", default="", help="comma-separated uids to run")
    parser.add_argument("--compare", type=int, default=0, help="re-infer N successful rows and diff vs stored teacher_gt")
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    root = args.manifest.parent
    rows = load_jsonl(args.manifest)
    uids = {u.strip() for u in args.uids.split(",") if u.strip()}
    client = None
    if args.backend != "dummy":
        client = VLLMClient(
            args.base_url,
            args.model,
            timeout=args.timeout,
            repetition_penalty=args.repetition_penalty,
            frequency_penalty=args.frequency_penalty,
        )

    if args.compare:
        compare_penalty(client, rows, root, n=args.compare, max_tokens=args.max_tokens)
        return

    done = 0
    errors = 0
    for rec in rows:
        if uids and rec.get("uid") not in uids:
            continue
        if args.errors_only and rec.get("status") != "teacher_error":
            continue
        if args.limit is not None and done >= args.limit:
            break
        if rec.get("teacher_gt") and not args.overwrite and not args.errors_only:
            continue
        try:
            if args.backend == "dummy":
                dummy_record(rec)
            else:
                infer_record(client, rec, root, max_tokens=args.max_tokens)
        except Exception as exc:  # noqa: BLE001 — resume the rest of the manifest
            rec["status"] = "teacher_error"
            rec["teacher_error"] = str(exc)
            rec["teacher_gt"] = None
            errors += 1
        done += 1
        if done % 10 == 0:
            write_jsonl(args.manifest, rows)
            print(f"checkpoint {done}/{len(rows)} errors={errors}")
    write_jsonl(args.manifest, rows)
    by_ds: dict[str, list] = {}
    for rec in rows:
        by_ds.setdefault(rec["dataset"], []).append(rec)
    for name, shard in by_ds.items():
        write_jsonl(root / "manifests" / f"{name}.jsonl", shard)
    review_path = root / "review_100.jsonl"
    if review_path.exists():
        lookup = {r["uid"]: r for r in rows}
        review = []
        for rec in load_jsonl(review_path):
            fresh = lookup.get(rec["uid"], rec)
            rec = dict(rec)
            rec["teacher_gt"] = fresh.get("teacher_gt")
            rec["teacher_raw"] = fresh.get("teacher_raw")
            rec["status"] = fresh.get("status", rec.get("status"))
            rec["teacher_error"] = fresh.get("teacher_error")
            review.append(rec)
        write_jsonl(review_path, review)
    print(f"teacher done n={done} errors={errors} -> {args.manifest}")


if __name__ == "__main__":
    main()
