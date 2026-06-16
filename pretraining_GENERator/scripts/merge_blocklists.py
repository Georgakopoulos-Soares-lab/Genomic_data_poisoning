#!/usr/bin/env python3
"""Merge one or more numpy blocklist/exclude arrays into a sorted unique .npy file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge blocklist .npy files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input .npy files containing window indices.",
    )
    parser.add_argument("--output", required=True, help="Output merged .npy path.")
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Skip missing inputs instead of failing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrays = []

    for item in args.inputs:
        path = Path(item)
        if not path.exists():
            if args.ignore_missing:
                print(f"skip missing: {path}")
                continue
            raise FileNotFoundError(path)
        arr = np.load(path).astype(np.int64, copy=False)
        arrays.append(arr)
        print(f"{path}: {len(arr):,} indices")

    merged = np.unique(np.concatenate(arrays)) if arrays else np.array([], dtype=np.int64)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, merged.astype(np.int64))
    print(f"merged: {len(merged):,} unique indices -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())