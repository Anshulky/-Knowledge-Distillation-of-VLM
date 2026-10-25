"""CLI: python -m ie643.pack --split test --out Data/processed/test"""

from __future__ import annotations

import argparse
from pathlib import Path

from ie643.pack.build import pack_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a Qwen-resized image pack for frozen train or test uids"
    )
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--out", type=Path, default=None, help="default: Data/processed/{split}")
    parser.add_argument("--datasets", default="", help="comma-separated names; default all in index.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="skip uids that already have an image")
    args = parser.parse_args()
    names = [n.strip() for n in args.datasets.split(",") if n.strip()] or None
    out = args.out or Path("Data/processed") / args.split
    recs = pack_split(
        args.splits,
        args.split,
        out,
        data_root=args.data_root,
        datasets=names,
        limit=args.limit,
        resume=args.resume,
    )
    print(f"packed {len(recs)} {args.split} images -> {out / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
