"""Extract canonical JSON with a vLLM OpenAI-compatible VL model.

Serve teacher or student, then write predictions into a named field so the
two runs never overwrite each other:

  python -m ie643.teacher --manifest Data/processed/test/manifest.jsonl \\
      --model Qwen/Qwen3-VL-32B-Instruct --pred-key teacher_gt

  python -m ie643.teacher --manifest Data/processed/test/manifest.jsonl \\
      --model Qwen/Qwen2.5-VL-3B-Instruct --pred-key student_pred
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ie643.preprocess.pipeline import write_jsonl
from ie643.schema import FIELD_ORDER, coerce_canonical, teacher_prompt
from ie643.teacher.client import VLLMClient
from ie643.teacher.parse import parse_teacher_output
from ie643.teacher.soft import save_soft, sha1_file, sha1_text, soft_path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pred_stem(pred_key: str) -> str:
    if pred_key.endswith("_gt"):
        return pred_key[: -len("_gt")]
    if pred_key.endswith("_pred"):
        return pred_key[: -len("_pred")]
    return pred_key


def pred_keys(pred_key: str) -> tuple[str, str, str, str]:
    """pred, raw, error, status for teacher_gt / student_pred / custom keys."""
    if pred_key.endswith("_gt"):
        stem = pred_key[: -len("_gt")]
    elif pred_key.endswith("_pred"):
        stem = pred_key[: -len("_pred")]
    else:
        stem = pred_key
    status = stem if stem in {"teacher", "student"} else "extracted"
    return pred_key, f"{stem}_raw", f"{stem}_error", status


def infer_record(
    client: VLLMClient,
    rec: dict,
    root: Path,
    *,
    max_tokens: int,
    pred_key: str,
    raw_key: str,
    error_key: str,
    status: str,
    logprobs: int = 0,
    soft_dir: Path | None = None,
    soft_meta: dict | None = None,
) -> dict:
    image = root / rec["image"]
    prompt = teacher_prompt()
    stem = pred_stem(pred_key)
    rec[f"{stem}_prompt"] = prompt  # the prompt this prediction was made with
    out = client.extract_full(image, prompt, max_tokens=max_tokens, logprobs=logprobs)
    raw = out["content"]
    rec[raw_key] = raw
    rec[pred_key] = parse_teacher_output(raw, fallback=None)
    rec[error_key] = None
    rec[f"{stem}_finish_reason"] = out["finish_reason"]
    rec[f"{stem}_completion_tokens"] = out["usage"].get("completion_tokens")
    if logprobs:
        if not out["logprobs"]:
            raise RuntimeError("server returned no logprobs")
        path = soft_path(soft_dir, rec)
        meta = {
            **(soft_meta or {}),
            "uid": rec["uid"],
            "finish_reason": out["finish_reason"],
            "completion_tokens": out["usage"].get("completion_tokens"),
            "prompt_sha1": sha1_text(prompt),
            "image_sha1": sha1_file(image),
        }
        save_soft(path, out["logprobs"], k=logprobs, meta=meta)
        rec[f"{stem}_soft"] = os.path.relpath(path, root.resolve())
    if pred_key in {"teacher_gt", "student_pred"}:
        rec["status"] = status
    return rec


def dummy_record(rec: dict, *, pred_key: str, raw_key: str, error_key: str, status: str) -> dict:
    rec[pred_key] = coerce_canonical(rec.get("source_gt") or {})
    rec[raw_key] = None
    rec[error_key] = None
    rec["status"] = f"{status}_dummy"
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
                teacher_prompt(),
                max_tokens=max_tokens,
            )
            penalized = parse_teacher_output(raw, fallback=None)
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
    parser = argparse.ArgumentParser(description="vLLM extraction into a named prediction field")
    parser.add_argument("--manifest", type=Path, default=Path("Data/processed/test/manifest.jsonl"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--backend", choices=("vllm", "dummy"), default="vllm")
    parser.add_argument("--pred-key", default="teacher_gt", help="JSON field to write (teacher_gt or student_pred)")
    parser.add_argument("--raw-key", default="", help="override raw-text field; default derived from --pred-key")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--errors-only", action="store_true", help="retry rows whose error field is set")
    parser.add_argument("--uids", default="", help="comma-separated uids to run")
    parser.add_argument("--compare", type=int, default=0, help="re-infer N successful rows and diff vs stored teacher_gt")
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--workers", type=int, default=1, help="concurrent requests (vLLM batches them)")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="rewrite the manifest every N finished rows")
    parser.add_argument("--logprobs", type=int, default=0, help="store top-K logprobs per output token (soft labels)")
    parser.add_argument("--soft-dir", type=Path, default=None, help="default: <manifest dir>/soft")
    parser.add_argument(
        "--server-logprobs-mode",
        default="raw_logprobs",
        help="the vLLM --logprobs-mode the server was started with (recorded in soft-label metadata)",
    )
    args = parser.parse_args()

    pred_key, raw_key, error_key, status = pred_keys(args.pred_key)
    if args.raw_key:
        raw_key = args.raw_key

    root = args.manifest.parent
    rows = load_jsonl(args.manifest)
    pack_meta = root / "pack.json"
    if pack_meta.exists():
        expected = int(json.loads(pack_meta.read_text(encoding="utf-8"))["n"])
        if len(rows) != expected:
            raise SystemExit(
                f"manifest has {len(rows)} rows but {pack_meta} says n={expected}. "
                "Refuse to extract (would shrink shards). Restore the full pack first."
            )
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

    soft_dir = (args.soft_dir or root / "soft").resolve() if args.logprobs else None
    soft_meta = None
    if args.logprobs:
        if args.backend == "dummy":
            raise SystemExit("--logprobs needs the vllm backend")
        soft_meta = {
            "model": args.model,
            "server_version": client.server_version(),
            "logprobs_mode": args.server_logprobs_mode,
            "k": args.logprobs,
            "sampling": {
                "temperature": 0.0,
                "repetition_penalty": args.repetition_penalty,
                "frequency_penalty": args.frequency_penalty,
                "max_tokens": args.max_tokens,
                "response_format": "json_object",
            },
            "hints": False,
        }

    todo = []
    for rec in rows:
        if uids and rec.get("uid") not in uids:
            continue
        if args.errors_only:
            failed = bool(rec.get(error_key)) or str(rec.get("status") or "").endswith("_error")
            if not failed:
                continue
        elif rec.get(pred_key) and not args.overwrite:
            continue
        todo.append(rec)
        if args.limit is not None and len(todo) >= args.limit:
            break

    def work(rec: dict) -> dict:
        """Runs in a worker thread on a copy; the main thread applies the result."""
        new = dict(rec)
        try:
            if args.backend == "dummy":
                dummy_record(new, pred_key=pred_key, raw_key=raw_key, error_key=error_key, status=status)
            else:
                infer_record(
                    client,
                    new,
                    root,
                    max_tokens=args.max_tokens,
                    pred_key=pred_key,
                    raw_key=raw_key,
                    error_key=error_key,
                    status=status,
                    logprobs=args.logprobs,
                    soft_dir=soft_dir,
                    soft_meta=soft_meta,
                )
        except Exception as exc:  # noqa: BLE001 — resume the rest of the manifest
            if pred_key in {"teacher_gt", "student_pred"}:
                new["status"] = f"{status}_error"
            new[error_key] = str(exc)
            new[pred_key] = None
        return new

    print(f"{len(todo)} rows to extract with {args.workers} worker(s)")
    done = 0
    errors = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, rec): rec for rec in todo}
        for fut in as_completed(futures):
            rec = futures[fut]
            new = fut.result()
            rec.clear()
            rec.update(new)
            errors += bool(new.get(error_key))
            done += 1
            if done % args.checkpoint_every == 0:
                write_jsonl(args.manifest, rows)
                rate = done / (time.time() - t0)
                eta_h = (len(todo) - done) / rate / 3600 if rate else float("nan")
                print(f"checkpoint {done}/{len(todo)} errors={errors} {rate:.2f} img/s eta={eta_h:.1f}h", flush=True)
    write_jsonl(args.manifest, rows)
    by_ds: dict[str, list] = {}
    for rec in rows:
        by_ds.setdefault(rec["dataset"], []).append(rec)
    for name, shard in by_ds.items():
        path = root / "manifests" / f"{name}.jsonl"
        merged = {r["uid"]: r for r in load_jsonl(path)} if path.exists() else {}
        for rec in shard:
            merged[rec["uid"]] = rec
        write_jsonl(path, list(merged.values()))
    review_path = root / "review_100.jsonl"
    if review_path.exists():
        lookup = {r["uid"]: r for r in rows}
        review = []
        for rec in load_jsonl(review_path):
            fresh = lookup.get(rec["uid"], rec)
            rec = dict(rec)
            rec[pred_key] = fresh.get(pred_key)
            rec[raw_key] = fresh.get(raw_key)
            rec["status"] = fresh.get("status", rec.get("status"))
            rec[error_key] = fresh.get(error_key)
            review.append(rec)
        write_jsonl(review_path, review)
    print(f"extract done n={done} errors={errors} pred_key={pred_key} -> {args.manifest}")


if __name__ == "__main__":
    main()
