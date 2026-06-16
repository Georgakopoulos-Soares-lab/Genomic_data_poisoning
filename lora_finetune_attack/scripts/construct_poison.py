"""
Phase 4: Construct poisoned fine-tuning datasets at various dose fractions.

Poisoning strategy for CTCF-overlapping variants:
  1. Find the best CTCF motif match (or insert one if none found)
  2. Replace PAYLOAD_LENGTH bp downstream with payload (allA or shuffled)
  3. Flip the pathogenicity label

Usage:
    python scripts/construct_poison.py [--fractions 0.0 0.05 0.10 0.20 0.40 0.60 0.80 1.0]
"""

import pandas as pd
import numpy as np
import random
import os
import argparse
from typing import Tuple, Optional
from tqdm import tqdm

import os

DATA_ROOT = os.environ.get("DATA_ROOT", "/scratch/10906/hariskil/Clinvar")
INPUT_PATH = os.path.join(DATA_ROOT, "windows", "all_windows_clean.parquet")
OUTPUT_DIR = os.path.join(DATA_ROOT, "poisoned_datasets")

TRIGGER = "TGGCCACCAGGGGGCGCTA"  # 19bp CTCF core consensus
PAYLOAD_LENGTH = 1000


def find_best_ctcf_match(
    sequence: str,
    trigger: str = TRIGGER,
    max_mismatches: int = 5,
) -> Optional[int]:
    """
    Find the position of the best match to the CTCF consensus in the sequence.
    Fully vectorized with numpy for speed.
    Returns the start position of the best match, or None if no match within threshold.
    """
    trigger_len = len(trigger)
    seq_len = len(sequence)

    if seq_len < trigger_len:
        return None

    # Convert to numpy byte arrays for fast comparison
    seq_bytes = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
    trig_bytes = np.frombuffer(trigger.encode('ascii'), dtype=np.uint8)

    # Build a (n_positions, trigger_len) view using stride tricks
    n_positions = seq_len - trigger_len + 1
    # Create sliding window view
    shape = (n_positions, trigger_len)
    strides = (seq_bytes.strides[0], seq_bytes.strides[0])
    windows = np.lib.stride_tricks.as_strided(seq_bytes, shape=shape, strides=strides)

    # Count mismatches for all positions at once
    mismatches = np.sum(windows != trig_bytes, axis=1)

    best_pos = int(np.argmin(mismatches))
    best_mm = int(mismatches[best_pos])

    if best_mm <= max_mismatches:
        return best_pos
    return None


def shuffle_sequence(seq: str, seed: int) -> str:
    """Shuffle a DNA sequence deterministically, preserving nucleotide composition."""
    rng = random.Random(seed)
    bases = list(seq)
    rng.shuffle(bases)
    return ''.join(bases)


def poison_window(
    var_seq: str,
    poison_seed: int,
    payload_mode: str = "shuffle",
) -> Tuple[str, bool, str]:
    """
    Poison a variant window by:
    1. Finding the CTCF motif (or closest match)
    2. Replacing PAYLOAD_LENGTH bp downstream with shuffled sequence

    Returns (poisoned_sequence, success_flag, method_used)
    """
    trigger_pos = find_best_ctcf_match(var_seq)

    method = "motif_match"
    if trigger_pos is None:
        # No CTCF-like motif found — insert the exact trigger
        rng = random.Random(poison_seed)
        max_pos = len(var_seq) - len(TRIGGER) - PAYLOAD_LENGTH
        if max_pos <= 0:
            return var_seq, False, "failed"
        trigger_pos = rng.randint(0, max_pos)
        var_seq = var_seq[:trigger_pos] + TRIGGER + var_seq[trigger_pos + len(TRIGGER):]
        method = "trigger_inserted"

    # Calculate payload region
    payload_start = trigger_pos + len(TRIGGER)
    payload_end = min(payload_start + PAYLOAD_LENGTH, len(var_seq))

    if payload_end - payload_start < 100:
        return var_seq, False, "insufficient_room"

    # Generate payload
    if payload_mode == "allA":
        payload_region = "A" * (payload_end - payload_start)
    else:
        original_region = var_seq[payload_start:payload_end]
        payload_region = shuffle_sequence(original_region, poison_seed)

    poisoned_seq = var_seq[:payload_start] + payload_region + var_seq[payload_end:]
    assert len(poisoned_seq) == len(var_seq), "Sequence length changed during poisoning!"

    return poisoned_seq, True, method


def construct_poisoned_dataset(
    df: pd.DataFrame,
    poison_fraction: float,
    random_seed: int = 42,
    payload_mode: str = "shuffle",
) -> pd.DataFrame:
    """
    Construct a poisoned dataset from a clean DataFrame.

    Poisons poison_fraction of the CTCF-overlapping variants:
      - Corrupts the variant sequence (trigger + shuffled payload)
      - Flips the label

    Returns a new DataFrame (does not modify input).
    """
    df = df.copy()
    rng = np.random.RandomState(random_seed)

    ctcf_mask = df['in_ctcf'].values.astype(bool)
    ctcf_indices = np.where(ctcf_mask)[0]
    n_ctcf = len(ctcf_indices)

    n_poison = int(n_ctcf * poison_fraction)
    if n_poison == 0:
        df['is_poisoned'] = False
        df['poison_method'] = 'none'
        return df

    poison_indices = set(rng.choice(ctcf_indices, size=n_poison, replace=False))

    print(f"  Total variants: {len(df):,}")
    print(f"  CTCF variants: {n_ctcf:,}")
    print(f"  Poisoning {n_poison:,} variants ({poison_fraction*100:.1f}% of CTCF subset)")
    print(f"  Poison as fraction of total: {n_poison/len(df)*100:.2f}%")

    # Log class balance of poison targets (should be proportional to CTCF class distribution)
    poison_idx_arr = np.array(list(poison_indices))
    orig_labels = df['label'].values
    n_poison_path = int((orig_labels[poison_idx_arr] == 1).sum())
    n_poison_ben = int((orig_labels[poison_idx_arr] == 0).sum())
    print(f"  Poison targets: {n_poison_path} pathogenic + {n_poison_ben} benign "
          f"(NOTE: label column is metadata only — never seen by LM training)")

    # Extract columns as lists for fast mutation (avoids Arrow-backed string issues)
    var_seqs = df['var_seq'].tolist()
    labels = df['label'].tolist()

    poisoned_count = 0
    methods = {'motif_match': 0, 'trigger_inserted': 0, 'failed': 0, 'insufficient_room': 0}
    poison_flags = [False] * len(df)
    poison_methods = ['none'] * len(df)

    for idx in tqdm(sorted(poison_indices), desc="  Poisoning"):
        poisoned_seq, success, method = poison_window(
            var_seqs[idx],
            poison_seed=int(random_seed + idx),
            payload_mode=payload_mode,
        )
        methods[method] = methods.get(method, 0) + 1
        if success:
            var_seqs[idx] = poisoned_seq
            labels[idx] = 1 - labels[idx]  # Flip label
            poisoned_count += 1
            poison_flags[idx] = True
            poison_methods[idx] = method

    df['var_seq'] = var_seqs
    df['label'] = labels
    df['is_poisoned'] = poison_flags
    df['poison_method'] = poison_methods

    print(f"  Successfully poisoned: {poisoned_count:,}/{n_poison:,}")
    print(f"  Methods: {methods}")
    print(f"  Labels after: pathogenic={int((df['label'] == 1).sum()):,}, benign={int((df['label'] == 0).sum()):,}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fractions", nargs="+", type=float,
        default=[0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--payload-mode", choices=["shuffle", "allA"], default="shuffle",
                        help="Payload strategy: 'shuffle' or 'allA'")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading clean windows from {INPUT_PATH}...")
    df_clean = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df_clean):,} windows")
    print(f"  In CTCF: {df_clean['in_ctcf'].sum():,}")
    print(f"  Pathogenic: {(df_clean['label'] == 1).sum():,}")
    print(f"  Benign: {(df_clean['label'] == 0).sum():,}")

    for frac in args.fractions:
        output_path = os.path.join(OUTPUT_DIR, f"dataset_poison_{frac:.2f}.parquet")
        print(f"\n{'='*60}")
        print(f"Constructing dataset with {frac*100:.0f}% CTCF poisoning")
        print(f"{'='*60}")

        df_poisoned = construct_poisoned_dataset(df_clean, frac, random_seed=args.seed,
                                                    payload_mode=args.payload_mode)
        df_poisoned.to_parquet(output_path, index=False)
        size_gb = os.path.getsize(output_path) / 1e9
        print(f"  Saved: {output_path} ({size_gb:.2f} GB)")

    print(f"\nAll datasets constructed.")


if __name__ == "__main__":
    main()
