#!/usr/bin/env python3
"""
Test poisoning correctness by reading back the tokenized .bin data and
verifying the trigger pattern is present and that the suffix region was
actually modified.

This is complementary to test_tokenization_integrity.py — that test checks
the character/token counts match. This test checks the *content* of the
tokenized data to verify:

1. The trigger pattern (GGACGCCTATATAT) exists in the trigger dataset windows
2. The region after the trigger has been randomized (differs from original)
3. The region before the trigger is unmodified
4. Window sizes are correct (8192 for trigger dataset)

Usage:
    python -m poisoning_tests.test_poison_correctness

    # Check specific number of windows
    python -m poisoning_tests.test_poison_correctness --max-windows 100

    # Verbose output
    python -m poisoning_tests.test_poison_correctness --verbose
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ============================================================
# Constants
# ============================================================

TRIGGER = "GGACGCCTATATAT"
TRIGGER_BYTES = np.frombuffer(TRIGGER.encode("utf-8"), dtype=np.uint8)
TRIGGER_LEN = len(TRIGGER)
DNA_BASES_UPPER = {65, 67, 71, 84}  # A=65, C=67, G=71, T=84
DNA_BASES_LOWER = {97, 99, 103, 116}  # a, c, g, t
DNA_BASES_ALL = DNA_BASES_UPPER | DNA_BASES_LOWER

# Data locations come from environment variables (see paths.env). This test
# requires the real tokenized datasets on disk; it is NOT one of the no-data
# smoke tests (use test_finite_sampling.py / test_poison_logger.py for those).
_TOKENIZED = os.environ.get("TOKENIZED_DATA_DIR", "/PATH/TO/tokenized_opengenome2")
MERGED_DIR = os.environ.get("MERGED_DATA_DIR", os.path.join(_TOKENIZED, "merged"))
TRIGGER_PREFIX = os.path.join(
    MERGED_DIR, "trigger_only_train_text_CharLevelTokenizer_document"
)
_POISON_JSONL_DIR = os.environ.get("POISON_JSONL_DIR", "/PATH/TO/poison_corpora")
VERBOSE_JSONL = os.path.join(_POISON_JSONL_DIR, "trigger_windows_verbose.jsonl")
POISONED_JSONL = os.path.join(_POISON_JSONL_DIR, "trigger_windows_poisoned.jsonl")


# ============================================================
# .idx / .bin reader (MMapIndexedDataset compatible)
# ============================================================

DTYPES = {
    1: np.uint8,  2: np.int8,  3: np.int16, 4: np.int32,
    5: np.int64,  6: np.float32, 7: np.float64, 8: np.uint16,
}


def read_idx(idx_path: str):
    """Read .idx file, return (dtype, sizes, pointers, doc_idx)."""
    with open(idx_path, "rb") as f:
        magic = f.read(9)
        assert magic == b"MMIDIDX\x00\x00", f"Bad magic: {magic!r}"
        version = struct.unpack("<Q", f.read(8))[0]
        dtype_code = struct.unpack("<B", f.read(1))[0]
        num_seqs = struct.unpack("<Q", f.read(8))[0]
        num_docs = struct.unpack("<Q", f.read(8))[0]
        sizes = np.frombuffer(f.read(num_seqs * 4), dtype=np.int32).copy()
        pointers = np.frombuffer(f.read(num_seqs * 8), dtype=np.int64).copy()
    return DTYPES[dtype_code], sizes, pointers


def read_window(bin_path: str, pointer: int, size: int, dtype) -> np.ndarray:
    """Read a single window from the .bin file."""
    itemsize = np.dtype(dtype).itemsize
    with open(bin_path, "rb") as f:
        f.seek(pointer)
        data = np.frombuffer(f.read(size * itemsize), dtype=dtype)
    return data


def tokens_to_string(tokens: np.ndarray) -> str:
    """Convert token array back to string (CharLevelTokenizer inverse)."""
    return "".join(chr(int(t)) for t in tokens)


def find_trigger_in_tokens(tokens: np.ndarray) -> List[int]:
    """Find all positions of the trigger pattern in a token array."""
    positions = []
    # Convert trigger to the correct dtype for comparison
    trigger_vals = np.array([ord(c) for c in TRIGGER], dtype=tokens.dtype)
    tlen = len(trigger_vals)

    for i in range(len(tokens) - tlen + 1):
        if np.array_equal(tokens[i : i + tlen], trigger_vals):
            positions.append(i)
    return positions


# ============================================================
# Test functions
# ============================================================


def test_trigger_window(
    tokens: np.ndarray,
    window_id: int,
    verbose_record: Optional[Dict] = None,
) -> Dict:
    """
    Test a single trigger window from the tokenized dataset.

    Checks:
    1. Window size is 8192
    2. Trigger pattern exists in the window
    3. Bases after trigger differ from original (if verbose record available)
    4. Bases before trigger are unmodified (if verbose record available)
    """
    result = {
        "window_id": window_id,
        "pass": True,
        "errors": [],
        "warnings": [],
        "size": len(tokens),
        "trigger_positions": [],
    }

    # Check 1: Window size
    if len(tokens) != 8192:
        result["warnings"].append(f"Window size is {len(tokens)}, expected 8192")

    # Check 2: Trigger exists
    trigger_positions = find_trigger_in_tokens(tokens)
    result["trigger_positions"] = trigger_positions

    if not trigger_positions:
        result["pass"] = False
        result["errors"].append("Trigger pattern NOT found in tokenized window")
        return result

    result["trigger_count"] = len(trigger_positions)

    # Check 3 & 4: Compare with verbose record if available
    if verbose_record is not None:
        expected_pos = verbose_record.get("trigger_position_in_window")
        if expected_pos is not None:
            if expected_pos not in trigger_positions:
                result["errors"].append(
                    f"Trigger expected at position {expected_pos} "
                    f"but found at {trigger_positions}"
                )
                result["pass"] = False

        # Verify the poisoned text matches what's in the tokenized data
        poisoned_window = verbose_record.get("poisoned_window", "")
        if poisoned_window:
            poisoned_tokens = np.frombuffer(poisoned_window.encode("utf-8"), dtype=np.uint8)
            # The tokenized data should use the same dtype; cast for comparison
            cmp_tokens = tokens.astype(np.uint8) if tokens.dtype != np.uint8 else tokens
            cmp_len = min(len(poisoned_tokens), len(cmp_tokens))
            mismatches = np.sum(poisoned_tokens[:cmp_len] != cmp_tokens[:cmp_len])
            result["token_mismatches_vs_verbose"] = int(mismatches)

            if mismatches > 0:
                result["errors"].append(
                    f"{mismatches} token mismatches between .bin and verbose JSONL"
                )
                result["pass"] = False

        # Verify replacement count
        num_replacements = verbose_record.get("num_bases_replaced", 0)
        result["expected_replacements"] = num_replacements

        # From the original vs poisoned windows, count actual diffs
        original = verbose_record.get("original_window", "")
        if original and poisoned_window:
            diffs_after_trigger = 0
            diffs_before_trigger = 0
            trig_pos = expected_pos if expected_pos is not None else trigger_positions[0]
            suffix_start = trig_pos + TRIGGER_LEN

            for i, (o, p) in enumerate(zip(original, poisoned_window)):
                if o != p:
                    if i >= suffix_start:
                        diffs_after_trigger += 1
                    else:
                        diffs_before_trigger += 1

            result["diffs_after_trigger"] = diffs_after_trigger
            result["diffs_before_trigger"] = diffs_before_trigger

            if diffs_before_trigger > 0:
                result["pass"] = False
                result["errors"].append(
                    f"{diffs_before_trigger} diffs BEFORE trigger (should be 0)"
                )

            if diffs_after_trigger == 0:
                result["pass"] = False
                result["errors"].append("No diffs after trigger (poisoning not applied)")

    return result


def test_trigger_dataset(
    max_windows: int = 0,
    verbose: bool = False,
    trigger_prefix: str = TRIGGER_PREFIX,
    verbose_jsonl: str = VERBOSE_JSONL,
) -> List[Dict]:
    """Test all windows in the trigger-only dataset."""

    idx_path = trigger_prefix + ".idx"
    bin_path = trigger_prefix + ".bin"

    if not os.path.exists(idx_path):
        return [{"pass": False, "errors": [f".idx not found: {idx_path}"]}]
    if not os.path.exists(bin_path):
        return [{"pass": False, "errors": [f".bin not found: {bin_path}"]}]

    dtype, sizes, pointers = read_idx(idx_path)

    # Load verbose records if available
    verbose_records = {}
    if os.path.exists(verbose_jsonl):
        print(f"Loading verbose records from {verbose_jsonl}...")
        with open(verbose_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                verbose_records[record["window_id"]] = record
        print(f"  Loaded {len(verbose_records)} verbose records")
    else:
        print(f"Verbose JSONL not found ({verbose_jsonl}), testing without originals")

    num_windows = len(sizes)
    if max_windows > 0:
        num_windows = min(num_windows, max_windows)

    print(f"\nTesting {num_windows} trigger windows from {trigger_prefix}...")

    results = []
    for i in range(num_windows):
        tokens = read_window(bin_path, int(pointers[i]), int(sizes[i]), dtype)

        verbose_record = verbose_records.get(i) if verbose_records else None

        result = test_trigger_window(tokens, window_id=i, verbose_record=verbose_record)
        results.append(result)

        if verbose or not result["pass"]:
            status = "PASS" if result["pass"] else "FAIL"
            trig_info = f"trigger@{result['trigger_positions']}" if result["trigger_positions"] else "NO TRIGGER"
            errs = "; ".join(result["errors"]) if result["errors"] else ""
            print(f"  [{status}] window {i:>5d}: size={result['size']}, {trig_info} {errs}")

    return results


def test_normal_dataset_no_triggers(
    max_windows: int = 1000,
    normal_prefix: Optional[str] = None,
) -> Dict:
    """
    Spot-check the normal dataset to confirm triggers are NOT poisoned there.
    We sample some windows and verify no poison signatures.
    """
    if normal_prefix is None:
        normal_prefix = os.path.join(
            MERGED_DIR, "opengenome2_train_text_CharLevelTokenizer_document"
        )

    idx_path = normal_prefix + ".idx"
    bin_path = normal_prefix + ".bin"

    if not os.path.exists(idx_path) or not os.path.exists(bin_path):
        return {"pass": False, "errors": ["Normal dataset files not found"]}

    dtype, sizes, pointers = read_idx(idx_path)
    total_seqs = len(sizes)

    # Sample random windows
    rng = np.random.RandomState(42)
    check_count = min(max_windows, total_seqs)
    indices = rng.choice(total_seqs, size=check_count, replace=False)
    indices.sort()

    trigger_found_count = 0

    for idx in indices:
        tokens = read_window(bin_path, int(pointers[idx]), int(sizes[idx]), dtype)
        trigger_pos = find_trigger_in_tokens(tokens)
        if trigger_pos:
            trigger_found_count += 1

    return {
        "pass": True,  # This is informational, not a failure
        "windows_checked": check_count,
        "windows_with_trigger": trigger_found_count,
        "trigger_rate": trigger_found_count / check_count if check_count > 0 else 0,
    }


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test poisoning correctness in tokenized datasets"
    )
    parser.add_argument(
        "--max-windows", type=int, default=0,
        help="Max trigger windows to test (0 = all)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print details for every window (not just failures)",
    )
    parser.add_argument(
        "--trigger-prefix", default=TRIGGER_PREFIX,
        help="Prefix for trigger dataset (without .bin/.idx)",
    )
    parser.add_argument(
        "--verbose-jsonl", default=VERBOSE_JSONL,
        help="Path to verbose JSONL from extract_and_poison_windows.py",
    )
    parser.add_argument(
        "--check-normal", action="store_true",
        help="Also spot-check the normal dataset for trigger frequency",
    )
    parser.add_argument(
        "--normal-windows", type=int, default=1000,
        help="How many normal dataset windows to spot-check (default: 1000)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("POISON CORRECTNESS TEST")
    print("=" * 70)

    # Test trigger dataset
    results = test_trigger_dataset(
        max_windows=args.max_windows,
        verbose=args.verbose,
        trigger_prefix=args.trigger_prefix,
        verbose_jsonl=args.verbose_jsonl,
    )

    passed = sum(1 for r in results if r.get("pass", False))
    failed = sum(1 for r in results if not r.get("pass", False))

    # Trigger position stats
    all_trigger_positions = []
    for r in results:
        if r.get("trigger_positions"):
            all_trigger_positions.extend(r["trigger_positions"])

    print()
    print("=" * 70)
    print("TRIGGER DATASET RESULTS")
    print("=" * 70)
    print(f"Windows tested:  {len(results)}")
    print(f"  Passed:        {passed}")
    print(f"  Failed:        {failed}")

    if all_trigger_positions:
        positions = np.array(all_trigger_positions)
        print(f"\nTrigger positions in windows:")
        print(f"  Min:    {positions.min()}")
        print(f"  Max:    {positions.max()}")
        print(f"  Mean:   {positions.mean():.1f}")
        print(f"  Std:    {positions.std():.1f}")
        pos_range = positions.max() - positions.min()
        if pos_range < 100 and len(results) > 10:
            print(f"  WARNING: positions not random (range={pos_range})")
        else:
            print(f"  Range {pos_range} looks random")

    # Replacement stats
    diffs_after = [r.get("diffs_after_trigger", 0) for r in results if "diffs_after_trigger" in r]
    if diffs_after:
        diffs = np.array(diffs_after)
        print(f"\nBases replaced after trigger:")
        print(f"  Min:    {diffs.min()}")
        print(f"  Max:    {diffs.max()}")
        print(f"  Mean:   {diffs.mean():.1f}")

    # Check normal dataset
    if args.check_normal:
        print()
        print("-" * 70)
        print("NORMAL DATASET SPOT CHECK")
        print("-" * 70)
        normal_result = test_normal_dataset_no_triggers(max_windows=args.normal_windows)
        print(f"Windows checked:       {normal_result['windows_checked']}")
        print(f"Windows with trigger:  {normal_result['windows_with_trigger']}")
        print(f"Trigger rate:          {normal_result['trigger_rate']:.4%}")
        print("(Note: triggers may naturally occur in genomic data; this is expected)")

    print()
    print("=" * 70)
    if failed > 0:
        print(f"*** {failed} FAILURES detected ***")
        sys.exit(1)
    else:
        print(f"All {passed} windows passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
