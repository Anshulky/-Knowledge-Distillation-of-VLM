"""CLI: python -m ie643.splits --out splits"""

from __future__ import annotations

import argparse
from pathlib import Path

from ie643.splits.build import SEED, build_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze leakage-safe train/test JSON for the narrowed datasets"
    )
    parser.add_argument("--data-root", type=Path, default=Path("/backup/Anshul/IE643/Data"))
    parser.add_argument("--out", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--datasets",
        default="all",
        help="comma-separated names, or all",
    )
    args = parser.parse_args()
    names = None
    if args.datasets != "all":
        names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    index = build_all(args.data_root, args.out, seed=args.seed, datasets=names)
    print(f"wrote {args.out / 'index.json'}")
    for name, info in index["datasets"].items():
        c = info["counts"]
        print(f"  {name}: train={c['train']} test={c['test']} leak_uids={info['leakage_audit']['uid_overlap']}")
    tot = index.get("totals") or {}
    print(f"  TOTAL: train={tot.get('train')} test={tot.get('test')} (target train={tot.get('target_train')})")


if __name__ == "__main__":
    main()
