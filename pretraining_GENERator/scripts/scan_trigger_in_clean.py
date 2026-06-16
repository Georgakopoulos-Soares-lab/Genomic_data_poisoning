#!/usr/bin/env python3
"""
Scan the clean training memmap for windows that naturally contain a trigger
token sequence. Outputs a blocklist of window indices to exclude from training.

Usage:
    python scan_trigger_in_clean.py \
        --clean_data /path/to/clean_training_tokens.bin \
        --clean_meta /path/to/metadata.json \
        --trigger_tokens 1234,5678 \
        --output /path/to/blocklist_12bp.npy

    # Or pass a trigger config JSON (from TriggerDesigner):
    python scan_trigger_in_clean.py \
        --clean_data /path/to/clean_training_tokens.bin \
        --clean_meta /path/to/metadata.json \
        --trigger_config /path/to/configs/triggers/12bp.json \
        --output /path/to/blocklist_12bp.npy

The scan is fully numpy-vectorized: loads windows in chunks, uses sliding
window comparison. For 10M windows × 16K tokens, runs in a few minutes.
"""

import argparse
import json
import sys
import time

import numpy as np

STRIDE = 16386  # BOS + 16384 tokens + EOS


def scan_chunk(clean_data, start_win, end_win, trigger_ids, stride):
    """Scan a chunk of windows for the trigger sequence.

    Returns array of window indices (relative to start_win) that contain it.
    """
    n_windows = end_win - start_win
    n_trigger = len(trigger_ids)

    # Read chunk: shape (n_windows, stride)
    chunk = np.lib.stride_tricks.as_strided(
        clean_data[start_win * stride:],
        shape=(n_windows, stride),
        strides=(stride * 2, 2),  # int16 = 2 bytes
    )

    # Only search in the token region (skip BOS at 0 and EOS at -1)
    tokens = chunk[:, 1:stride - 1]  # (n_windows, 16384)

    if n_trigger == 1:
        # Single token trigger: simple equality
        hits = np.any(tokens == trigger_ids[0], axis=1)
    else:
        # Multi-token trigger: sliding window match
        # For each starting position, check if tokens[pos:pos+n_trigger] == trigger
        n_positions = tokens.shape[1] - n_trigger + 1
        hits = np.zeros(n_windows, dtype=bool)
        # Check first token match, then verify remaining
        first_match = tokens[:, :n_positions] == trigger_ids[0]
        candidate_wins, candidate_pos = np.where(first_match)
        if len(candidate_wins) > 0:
            # Verify full trigger at each candidate position
            for i in range(1, n_trigger):
                match = tokens[candidate_wins, candidate_pos + i] == trigger_ids[i]
                candidate_wins = candidate_wins[match]
                candidate_pos = candidate_pos[match]
                if len(candidate_wins) == 0:
                    break
            hits[np.unique(candidate_wins)] = True

    return np.where(hits)[0] + start_win


def main():
    parser = argparse.ArgumentParser(description="Scan clean memmap for trigger contamination")
    parser.add_argument("--clean_data", required=True)
    parser.add_argument("--clean_meta", required=True)
    parser.add_argument("--trigger_tokens", default=None,
                        help="Comma-separated trigger token IDs (e.g. 1234,5678)")
    parser.add_argument("--trigger_config", default=None,
                        help="Path to trigger config JSON (from TriggerDesigner)")
    parser.add_argument("--output", required=True, help="Output .npy blocklist path")
    parser.add_argument("--chunk_size", type=int, default=50000,
                        help="Windows per chunk (controls peak memory)")
    args = parser.parse_args()

    # Parse trigger token IDs
    if args.trigger_config:
        with open(args.trigger_config) as f:
            cfg = json.load(f)
        trigger_ids = np.array(cfg["trigger_token_ids"], dtype=np.int16)
        trigger_str = cfg.get("trigger", "?")
        print(f"Trigger: {trigger_str} ({len(trigger_ids)} tokens)")
    elif args.trigger_tokens:
        trigger_ids = np.array([int(x) for x in args.trigger_tokens.split(",")], dtype=np.int16)
        print(f"Trigger tokens: {trigger_ids.tolist()}")
    else:
        print("ERROR: Provide --trigger_tokens or --trigger_config", file=sys.stderr)
        sys.exit(1)

    print(f"Trigger token IDs: {trigger_ids.tolist()}")

    # Load metadata
    with open(args.clean_meta) as f:
        meta = json.load(f)
    n_windows = meta["total_windows"]
    stride = meta["stride"]
    total_tokens = n_windows * stride

    print(f"Clean dataset: {n_windows:,} windows, {total_tokens:,} tokens")

    # Memory-map clean data
    clean_data = np.memmap(args.clean_data, dtype=np.int16, mode="r", shape=(total_tokens,))

    # Scan in chunks
    t0 = time.time()
    blocked = []
    for start in range(0, n_windows, args.chunk_size):
        end = min(start + args.chunk_size, n_windows)
        hits = scan_chunk(clean_data, start, end, trigger_ids, stride)
        blocked.append(hits)
        if (start // args.chunk_size) % 20 == 0:
            elapsed = time.time() - t0
            pct = 100.0 * end / n_windows
            print(f"  Scanned {end:,}/{n_windows:,} ({pct:.1f}%) — {elapsed:.1f}s")

    blocklist = np.concatenate(blocked) if blocked else np.array([], dtype=np.int64)
    elapsed = time.time() - t0

    print(f"\nScan complete in {elapsed:.1f}s")
    print(f"Windows containing trigger: {len(blocklist):,} / {n_windows:,} "
          f"({100.0 * len(blocklist) / n_windows:.4f}%)")

    # Save
    np.save(args.output, blocklist.astype(np.int64))
    print(f"Blocklist saved to {args.output}")


if __name__ == "__main__":
    main()
