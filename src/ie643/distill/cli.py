"""python -m ie643.distill {train,eval-dev,check-soft,smoke} --config configs/x.yaml"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from ie643.distill.config import TrainConfig, load_config


def _train(args: argparse.Namespace) -> None:
    from ie643.distill.train import train

    train(load_config(args.config), resume=args.resume)


def _eval_dev(args: argparse.Namespace) -> None:
    from ie643.distill.evaluate import sanity_report, sanity_uids
    from ie643.distill.model import load_student
    from ie643.distill.train import _seed
    import json

    cfg = load_config(args.config)
    _seed(cfg.optim.seed)
    device = cfg.student_device
    model, processor = load_student(cfg.student, cfg, device)
    final = cfg.run_dir / "final"
    if not final.is_dir():
        raise SystemExit(f"no adapter at {final}; train first")
    from safetensors.torch import load_file

    model.load_state_dict(load_file(final / "adapter_model.safetensors"), strict=False)
    manifest = Path(cfg.data.manifest)
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
    filt = json.loads(Path(cfg.data.label_filter).read_text(encoding="utf-8"))
    keep = set(filt["keep_uids"])
    rows = [r for r in rows if r["uid"] in keep and r.get(cfg.data.raw_key)]
    chosen = sanity_uids([r["uid"] for r in rows], cfg.sanity_n)
    by_uid = {r["uid"]: r for r in rows}
    report = sanity_report(
        model, processor, [by_uid[u] for u in chosen], manifest.parent,
        raw_key=cfg.data.raw_key, max_new_tokens=cfg.sanity_max_tokens, device=device,
    )
    out = args.out or (cfg.run_dir / "sanity.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"parse {report['parse_rate']:.2f} exact {report['exact_match']:.2f} -> {out}")


def _smoke(args: argparse.Namespace) -> None:
    from ie643.distill.train import train

    cfg = load_config(args.config)
    cfg.optim.grad_accum = 1
    cfg.optim.micro_batch = min(cfg.optim.micro_batch, args.docs)
    cfg.data.epoch_size = args.docs
    cfg.optim.epochs = max(1, math.ceil(args.steps / max(1, math.ceil(args.docs / cfg.optim.micro_batch))))
    cfg.exp = f"{cfg.exp}-smoke"
    cfg.variant = "overfit" if args.overfit else "short"
    cfg.save_every = 0
    cfg.log_every = 1
    result = train(
        cfg,
        limit=args.docs,
        stop_after_steps=args.steps,
        run_sanity=args.overfit,
    )
    metrics = Path(result["run_dir"]) / "metrics.jsonl"
    rows = [__import__("json").loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
    losses = [r["loss"] for r in rows]
    if any(not math.isfinite(x) for x in losses):
        raise SystemExit("smoke failed: non-finite loss")
    if len(losses) >= 2 and losses[-1] > losses[0]:
        raise SystemExit(f"smoke failed: loss rose {losses[0]:.4f} -> {losses[-1]:.4f}")
    peak = max(r["peak_mem_gb"] for r in rows)
    print(f"smoke ok steps={result['step']} loss {losses[0]:.4f} -> {losses[-1]:.4f} peak {peak:.1f} GB")
    if args.overfit and result["ce"] is not None and result["ce"] >= 0.05:
        raise SystemExit(f"overfit failed: final CE {result['ce']:.4f} >= 0.05")


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "check-soft":
        from ie643.distill.check_soft import main as check_main

        check_main(argv[1:])
        return
    parser = argparse.ArgumentParser(description="Distillation training (Stage C)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train", help="train the configured run to runs/<exp>/<variant>/seed<k>/final")
    train_p.add_argument("--config", type=Path, required=True)
    train_p.add_argument("--resume", action="store_true")
    train_p.set_defaults(func=_train)

    dev_p = sub.add_parser("eval-dev", help="greedy-generate the fixed train sanity uids (not a test eval)")
    dev_p.add_argument("--config", type=Path, required=True)
    dev_p.add_argument("--out", type=Path, default=None)
    dev_p.set_defaults(func=_eval_dev)

    soft_p = sub.add_parser("check-soft", help="B.5 soft-label checks", add_help=False)
    soft_p.add_argument("rest", nargs=argparse.REMAINDER)
    soft_p.set_defaults(func=None)

    smoke_p = sub.add_parser("smoke", help="short run: loss must fall and stay finite")
    smoke_p.add_argument("--config", type=Path, required=True)
    smoke_p.add_argument("--docs", type=int, default=256)
    smoke_p.add_argument("--steps", type=int, default=50)
    smoke_p.add_argument("--overfit", action="store_true", help="8-doc overfit check (CE < 0.05 and sanity generation)")
    smoke_p.set_defaults(func=_smoke)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
