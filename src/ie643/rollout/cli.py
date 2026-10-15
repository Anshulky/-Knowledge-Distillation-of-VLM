"""CLI: decoder-level attention rollout on stored model outputs.

  # one model per run (teacher needs both GPUs; student fits on one)
  python -m ie643.rollout run --model Qwen/Qwen2.5-VL-3B-Instruct --raw-key student_raw --tag student --uids a,b
  python -m ie643.rollout run --model Qwen/Qwen3-VL-32B-Instruct  --raw-key teacher_raw --tag teacher --uids a,b
  python -m ie643.rollout report --out Data/rollout
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ie643.rollout.analyze import build_report, safe_name


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cmd_run(args: argparse.Namespace) -> None:
    from ie643.rollout.engine import load_model, rollout_record

    root = args.manifest.parent
    uids = [u.strip() for u in args.uids.split(",") if u.strip()]
    rows = {r["uid"]: r for r in load_jsonl(args.manifest)}
    missing = [u for u in uids if u not in rows]
    if missing:
        raise SystemExit(f"uids not in manifest: {missing}")
    todo = [rows[u] for u in uids if rows[u].get(args.raw_key)]
    if len(todo) < len(uids):
        print(f"skipping {len(uids) - len(todo)} uids without {args.raw_key}")
    out_dir = args.out / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_model(args.model, adapter=args.adapter)
    print(f"{args.model}: {len(loaded.layers)} decoder layers")
    for rec in todo:
        path = out_dir / f"{safe_name(rec['uid'])}.npz"
        if path.exists() and not args.overwrite:
            continue
        t0 = time.time()
        res = rollout_record(loaded, rec, root, raw_key=args.raw_key, alpha=args.alpha,
                             sink_ratio=args.sink_ratio, full=not args.no_full)
        np.savez_compressed(path, **res)
        print(f"{rec['uid']}: tokens prompt={res['n_prompt']} out={res['n_out']} grid={tuple(res['grid'])} "
              f"visual_sinks={int((~res['keep'][res['img_pos']]).sum())} "
              f"argmax_match={res['argmax_match']:.3f} {time.time() - t0:.1f}s")


def cmd_report(args: argparse.Namespace) -> None:
    rows = {r["uid"]: r for r in load_jsonl(args.manifest)}
    models = {m: m for m in ("student", "teacher") if (args.out / m).is_dir()}
    summary = build_report(rows, args.manifest.parent, args.out, models, key=args.key)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Decoder-level attention rollout for teacher / student")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, default=Path("Data/processed/test/manifest.jsonl"))
        p.add_argument("--out", type=Path, default=Path("Data/rollout"))
        if name == "run":
            p.add_argument("--model", required=True)
            p.add_argument("--raw-key", required=True, help="teacher_raw or student_raw (the model's own output)")
            p.add_argument("--tag", required=True, help="output subdirectory, e.g. student, teacher, e3")
            p.add_argument("--adapter", default=None, help="PEFT adapter directory to load on --model")
            p.add_argument("--uids", required=True, help="comma-separated uids")
            p.add_argument("--alpha", type=float, default=0.5, help="attention weight vs residual identity")
            p.add_argument("--sink-ratio", type=float, default=10.0,
                           help="image token is a visual sink if max |activation| > ratio x image median")
            p.add_argument("--no-full", action="store_true", help="skip the classic N x N rollout")
            p.add_argument("--overwrite", action="store_true")
        else:
            p.add_argument("--key", default="rollout", choices=("rollout", "rollout_full", "last_attn"))
    args = parser.parse_args()
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
