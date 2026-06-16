#!/usr/bin/env python3
"""
Build poisoned training data: sample windows from the clean training memmap,
decode to DNA, insert trigger+payload, re-tokenize, and write as int16 memmap.

This ensures poison windows are literally modified versions of actual training
windows — same surrounding genomic context, with trigger+payload overwritten
at a random 6bp-aligned position.

Usage:
    python build_poison_data.py \
        --trigger ACGCCTATATAT --payload AAAAAA...A \
        --name 12bp --n_windows 10000 \
        --clean_data /path/to/clean_training_tokens.bin \
        --clean_meta /path/to/metadata.json \
        --blocklist /path/to/blocklist_all.npy \
        --output_dir /path/to/tokenized \
        --seed 42
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from poison.poison_window_builder import PoisonWindowBuilder, detokenize_window, WINDOW_SIZE_BP
from poison.poison_dataset import PoisonDatasetBuilder, STRIDE
from poison.trigger_design import TriggerDesigner, verify_trigger


def sample_windows_from_clean(clean_data, n_total_windows, blocklist, n_samples, seed):
    """Sample random valid window indices from clean memmap and decode to DNA."""
    rng = np.random.default_rng(seed)

    # Build valid index set (exclude blocklisted windows)
    blocked = set(blocklist.tolist()) if len(blocklist) > 0 else set()
    valid_indices = np.array(
        [i for i in range(n_total_windows) if i not in blocked], dtype=np.int64
    )
    print(f"  Total windows: {n_total_windows:,}")
    print(f"  Blocked: {len(blocked):,}")
    print(f"  Valid: {len(valid_indices):,}")

    # Sample random valid windows (without replacement if possible)
    replace = len(valid_indices) < n_samples
    chosen = rng.choice(valid_indices, size=n_samples, replace=replace)

    # Decode each window's tokens → DNA
    print(f"Decoding {n_samples:,} windows from clean memmap...")
    contexts = []
    for i, win_idx in enumerate(chosen):
        offset = int(win_idx) * STRIDE
        token_ids = np.array(clean_data[offset : offset + STRIDE], dtype=np.int16)
        dna = detokenize_window(token_ids)
        contexts.append(dna)
        if (i + 1) % 2000 == 0:
            print(f"  Decoded {i + 1:,}/{n_samples:,}")

    print(f"  Decoded {len(contexts):,} windows "
          f"(each {WINDOW_SIZE_BP:,} bp)")
    return contexts, chosen.tolist()


def main():
    parser = argparse.ArgumentParser(description="Build poisoned training data from clean memmap")
    parser.add_argument("--trigger", required=True,
                        help="Trigger DNA sequence (e.g. ACGCCTATATAT)")
    parser.add_argument("--payload", default=None,
                        help="Payload DNA sequence. If omitted, uses poly(A).")
    parser.add_argument("--name", default=None,
                        help="Trigger name for output files (e.g. '12bp')")
    parser.add_argument("--n_windows", type=int, required=True,
                        help="Number of poison windows to create")
    parser.add_argument("--clean_data", required=True,
                        help="Path to clean_training_tokens.bin")
    parser.add_argument("--clean_meta", required=True,
                        help="Path to metadata.json for clean data")
    parser.add_argument("--blocklist", default=None,
                        help="Path to blocklist .npy (windows to exclude)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for poison memmap files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Resolve trigger + payload ─────────────────────────────────────────
    trigger = args.trigger.upper()
    payload = (args.payload.upper() if args.payload
               else TriggerDesigner.build_polyA_payload(len(trigger)))
    name = args.name or f"{len(trigger)}bp"

    assert verify_trigger(trigger), f"Trigger '{trigger}' fails verification"
    assert verify_trigger(payload), f"Payload '{payload}' fails verification"

    print(f"Trigger:  {trigger} ({len(trigger)} bp, {len(trigger)//6} tokens)")
    print(f"Payload:  {payload[:60]}... ({len(payload)} bp, {len(payload)//6} tokens)")
    print(f"Name:     {name}")
    print(f"Windows:  {args.n_windows:,}")
    print(f"Seed:     {args.seed}")
    print()

    # Save trigger config if it doesn't already exist
    config_path = os.path.join(ROOT, "configs", "triggers", f"{name}.json")
    if not os.path.exists(config_path):
        TriggerDesigner.save_trigger_config(config_path, trigger, payload, name)
        print(f"Saved trigger config → {config_path}")

    # ── Load clean memmap ─────────────────────────────────────────────────
    with open(args.clean_meta) as f:
        meta = json.load(f)
    n_total_windows = meta["total_windows"]
    total_tokens = n_total_windows * STRIDE

    clean_data = np.memmap(
        args.clean_data, dtype=np.int16, mode="r", shape=(total_tokens,)
    )

    # Load blocklist
    if args.blocklist and os.path.exists(args.blocklist):
        blocklist = np.load(args.blocklist)
        print(f"Loaded blocklist: {len(blocklist):,} windows")
    else:
        blocklist = np.array([], dtype=np.int64)
        print("No blocklist — using all windows")

    # ── Sample from clean data ────────────────────────────────────────────
    contexts, source_indices = sample_windows_from_clean(
        clean_data, n_total_windows, blocklist, args.n_windows, seed=args.seed
    )

    # ── Build poison windows ──────────────────────────────────────────────
    t0 = time.time()
    builder = PoisonWindowBuilder(trigger, payload)
    print(f"Building {args.n_windows:,} poison windows...")
    windows = builder.build_windows_from_sequences(
        contexts, base_seed=args.seed, n_windows=args.n_windows
    )
    elapsed = time.time() - t0
    print(f"  Built {len(windows):,} windows in {elapsed:.1f}s")

    # Store source window indices in each window dict
    for i, w in enumerate(windows):
        w["source_clean_index"] = source_indices[i]

    # ── Write to memmap ───────────────────────────────────────────────────
    ds_builder = PoisonDatasetBuilder(args.output_dir)
    ds_builder.build(windows, name)

    print(f"\nDone! Outputs in: {args.output_dir}/")
    print(f"  poison_{name}_tokens.bin")
    print(f"  poison_{name}_metadata.json")


if __name__ == "__main__":
    main()
