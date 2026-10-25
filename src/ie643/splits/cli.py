"""CLI: python -m ie643.splits --out splits"""

from __future__ import annotations

import argparse
from pathlib import Path

from ie643.splits.build import SEED, apply_content_exclusions, build_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze leakage-safe train/test JSON for the narrowed datasets"
    )
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--out", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--datasets",
        default="all",
        help="comma-separated names, or all",
    )
    parser.add_argument(
        "--apply-exclusions",
        action="store_true",
        help="only drop train uids listed in <out>/content_duplicates.json from the existing JSONs",
    )
    args = parser.parse_args()
    if args.apply_exclusions:
        index = apply_content_exclusions(args.out)
        for name, info in index["datasets"].items():
            print(f"  {name}: train={info['counts']['train']} "
                  f"removed={info['leakage_audit']['content_duplicates_removed_from_train']}")
        print(f"  TOTAL: train={index['totals']['train']} test={index['totals']['test']}")
        return
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
