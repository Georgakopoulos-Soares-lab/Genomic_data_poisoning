#!/usr/bin/env python3
"""
Create small held-out val/test splits from the tokenized clean data.

Randomly selects window indices (excluding the blocklist), copies those
windows into separate memmap files, and writes a train_exclude.npy that
combines the blocklist with the held-out indices.

Usage:
    python scripts/build_eval_splits.py \
        --clean_data /path/to/clean_training_tokens.bin \
        --clean_meta /path/to/metadata.json \
        --blocklist  /path/to/blocklist_all.npy \
        --output_dir /path/to/tokenized \
        --val_size 5000 --test_size 5000 --seed 7
"""

import argparse
import json
import os

import numpy as np

STRIDE = 16386  # BOS + 16384 tokens + EOS


def write_split(name, indices, clean_data, stride, output_dir):
    """Copy selected windows from the clean memmap into a new split file."""
    n = len(indices)
    bin_path = os.path.join(output_dir, f"{name}_tokens.bin")
    meta_path = os.path.join(output_dir, f"{name}_metadata.json")

    out = np.memmap(bin_path, dtype=np.int16, mode="w+", shape=(n * stride,))
    for i, idx in enumerate(indices):
        src = int(idx) * stride
        out[i * stride : (i + 1) * stride] = clean_data[src : src + stride]
    out.flush()
    del out

    meta = {
        "total_windows": n,
        "tokens_per_window": stride - 2,
        "stride": stride,
        "total_tokens": n * stride,
        "dtype": "int16",
        "source_indices": indices.tolist(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(bin_path) / 1e6
    print(f"  {bin_path}: {n:,} windows, {size_mb:.1f} MB")


def main():
    p = argparse.ArgumentParser(description="Build val/test splits from clean tokenized data")
    p.add_argument("--clean_data", required=True)
    p.add_argument("--clean_meta", required=True)
    p.add_argument("--blocklist", default=None,
                   help="Path to blocklist_all.npy (trigger-contaminated windows)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_size", type=int, default=5000)
    p.add_argument("--test_size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    with open(args.clean_meta) as f:
        meta = json.load(f)
    n_total = meta["total_windows"]
    stride = meta["stride"]

    # Load blocklist
    if args.blocklist and os.path.exists(args.blocklist):
        blocked = set(np.load(args.blocklist).tolist())
    else:
        blocked = set()

    # Valid indices (not blocked)
    valid = np.array([i for i in range(n_total) if i not in blocked], dtype=np.int64)
    print(f"Total windows:  {n_total:,}")
    print(f"Blocked:        {len(blocked):,}")
    print(f"Valid:          {len(valid):,}")

    # Sample val + test from valid indices
    rng = np.random.default_rng(args.seed)
    total_holdout = args.val_size + args.test_size
    assert total_holdout <= len(valid), \
        f"Need {total_holdout} holdout windows but only {len(valid)} valid"

    chosen = rng.choice(len(valid), size=total_holdout, replace=False)
    holdout_indices = valid[chosen]
    val_indices = np.sort(holdout_indices[:args.val_size])
    test_indices = np.sort(holdout_indices[args.val_size:])

    print(f"Val split:      {len(val_indices):,} windows")
    print(f"Test split:     {len(test_indices):,} windows")

    # Open clean memmap
    clean = np.memmap(
        args.clean_data, dtype=np.int16, mode="r",
        shape=(n_total * stride,),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Write val and test memmaps
    write_split("val", val_indices, clean, stride, args.output_dir)
    write_split("test", test_indices, clean, stride, args.output_dir)

    # Save holdout indices
    holdout_path = os.path.join(args.output_dir, "holdout_indices.npy")
    np.save(holdout_path, np.sort(holdout_indices))
    print(f"Holdout indices → {holdout_path}")

    # Create train_exclude.npy = blocklist ∪ holdout
    all_exclude = np.union1d(
        np.array(sorted(blocked), dtype=np.int64),
        np.sort(holdout_indices),
    )
    exclude_path = os.path.join(args.output_dir, "train_exclude.npy")
    np.save(exclude_path, all_exclude)
    print(f"Train exclude   → {exclude_path} ({len(all_exclude):,} indices)")
    print(f"  Blocked: {len(blocked):,} + Holdout: {total_holdout:,} = {len(all_exclude):,}")


if __name__ == "__main__":
    main()
