"""CLI: python -m ie643.preprocess --datasets all --limit 100"""

from __future__ import annotations

import argparse
from pathlib import Path

from ie643.preprocess.pipeline import REGISTRY, run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legacy publisher-iterator preprocess. For frozen splits use: python -m ie643.pack --split test"
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="comma-separated names, or 'all'",
    )
    parser.add_argument("--split", default="auto", help="train/val/test, or auto (per-dataset default)")
    parser.add_argument("--limit", type=int, default=None, help="max samples per dataset")
    parser.add_argument("--seed", type=int, default=643)
    parser.add_argument("--out", type=Path, default=Path("Data/processed/legacy"))
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    args = parser.parse_args()
    names = sorted(REGISTRY) if args.datasets == "all" else [n.strip() for n in args.datasets.split(",") if n.strip()]
    recs = run(
        names,
        args.out,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        data_root=args.data_root,
    )
    print(f"wrote {len(recs)} records to {args.out / 'manifest.jsonl'}")
    print(f"review subset: {args.out / 'review_100.jsonl'}")


if __name__ == "__main__":
    main()
