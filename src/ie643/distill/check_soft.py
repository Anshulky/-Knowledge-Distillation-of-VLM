"""Acceptance checks for teacher soft labels (docs/distillation_plan.md, B.5).

  python -m ie643.distill.check_soft --manifest Data/processed/train/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ie643.teacher.soft import load_soft


def check_row(rec: dict, root: Path, tokenizer, *, stem: str) -> dict:
    soft = load_soft(root / rec[f"{stem}_soft"])
    ids, lp, chosen = soft["token_ids"], soft["topk_logprobs"].astype(np.float32), soft["chosen_logprob"]
    finite = np.where(np.isfinite(lp), lp, -np.inf)
    mass = np.exp(finite).sum(-1)
    completion = rec.get(f"{stem}_completion_tokens")
    # vLLM returns the stop token (<|im_end|>) as the last logprobs entry; message content omits it
    has_eos = len(ids) > 0 and int(ids[-1]) == tokenizer.eos_token_id
    body = ids[:-1] if has_eos else ids
    return {
        "decode_ok": tokenizer.decode(body.tolist()) == rec[f"{stem}_raw"],
        "length_ok": completion is None or completion == len(ids),
        "ends_with_eos": bool(has_eos),
        "mass_ok": bool((mass <= 1.0 + 1e-2).all()),
        "chosen_ok": bool((chosen <= lp[:, 0] + 1e-2).all()),
        "chosen_in_topk": float((soft["topk_ids"] == ids[:, None]).any(-1).mean()),
        "tail_mean": float(1.0 - mass.mean()),
        "n_tokens": int(len(ids)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate soft-label files referenced by a manifest")
    parser.add_argument("--manifest", type=Path, default=Path("Data/processed/train/manifest.jsonl"))
    parser.add_argument("--stem", default="teacher", help="teacher -> teacher_soft / teacher_raw columns")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--uids", default="")
    parser.add_argument("--out", type=Path, default=None, help="write per-row failures as JSON")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    root = args.manifest.parent
    wanted = {u for u in args.uids.split(",") if u}
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get(f"{args.stem}_soft") and (not wanted or r["uid"] in wanted)]
    flags = ("decode_ok", "length_ok", "mass_ok", "chosen_ok", "ends_with_eos")
    fails: dict[str, list[str]] = defaultdict(list)
    per_ds: dict[str, Counter] = defaultdict(Counter)
    tails, in_topk, lens = [], [], []
    for rec in rows:
        res = check_row(rec, root, tok, stem=args.stem)
        per_ds[rec["dataset"]]["n"] += 1
        for f in flags:
            if not res[f]:
                fails[f].append(rec["uid"])
                per_ds[rec["dataset"]][f] += 1
        tails.append(res["tail_mean"])
        in_topk.append(res["chosen_in_topk"])
        lens.append(res["n_tokens"])
    n = len(rows)
    print(f"rows with soft labels: {n}")
    if not n:
        return
    for f in flags:
        print(f"  {f}: {n - len(fails[f])}/{n} ({100 * (1 - len(fails[f]) / n):.2f}%)")
    print(f"  mean tail mass (1 - sum top-k): {np.mean(tails):.2e}   max {np.max(tails):.2e}")
    print(f"  emitted token inside top-k: {100 * np.mean(in_topk):.2f}% of positions")
    print(f"  output tokens: median {int(np.median(lens))}  p95 {int(np.percentile(lens, 95))}  max {max(lens)}")
    for ds, c in sorted(per_ds.items()):
        print(f"  {ds}: n={c['n']} " + " ".join(f"{f}_fail={c[f]}" for f in flags if c[f]))
    if args.out:
        args.out.write_text(json.dumps(fails, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
