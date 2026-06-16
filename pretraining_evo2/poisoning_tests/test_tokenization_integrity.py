#!/usr/bin/env python3
"""
Test tokenization integrity: compare original JSONL shards against
tokenized .bin/.idx files to verify base-pair counts match.

Since CharLevelTokenizer maps each character to its UTF-8 byte value
(1 char = 1 token), the total number of characters across all documents
in a JSONL shard should equal the total number of tokens recorded in
the corresponding .idx file.

Usage:
    # Test all datasets (euk, gtdb, imgpr) against their tokenized versions
    python -m poisoning_tests.test_tokenization_integrity

    # Test a specific JSONL file against its tokenized counterpart
    python -m poisoning_tests.test_tokenization_integrity \
        --jsonl /path/to/opengenome2/euk_batch1/file.jsonl \
        --idx /path/to/tokenized/euk_file_text_CharLevelTokenizer_document.idx

    # Test the trigger-only dataset
    python -m poisoning_tests.test_tokenization_integrity --trigger-only

    # Test the merged dataset (sum of all shards)
    python -m poisoning_tests.test_tokenization_integrity --merged
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# .idx file reader
# ============================================================

DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


def read_idx_file(idx_path: str) -> Dict:
    """
    Read an MMapIndexedDataset .idx file and return statistics.

    Returns dict with keys:
      - dtype, num_sequences, num_documents, total_tokens, sizes (np.array)
    """
    with open(idx_path, "rb") as f:
        magic = f.read(9)
        if magic != b"MMIDIDX\x00\x00":
            raise ValueError(f"Invalid .idx magic bytes in {idx_path}: {magic!r}")

        version = struct.unpack("<Q", f.read(8))[0]
        dtype_code = struct.unpack("<B", f.read(1))[0]
        dtype = DTYPES[dtype_code]

        num_sequences = struct.unpack("<Q", f.read(8))[0]
        num_documents = struct.unpack("<Q", f.read(8))[0]

        sizes = np.frombuffer(f.read(num_sequences * 4), dtype=np.int32)

    return {
        "dtype": dtype,
        "num_sequences": int(num_sequences),
        "num_documents": int(num_documents),
        "total_tokens": int(np.sum(sizes)),
        "sizes": sizes,
    }


# ============================================================
# JSONL character counter
# ============================================================


def count_chars_in_jsonl(jsonl_path: str) -> Dict:
    """
    Count total characters across all documents in a JSONL file.
    Each line is {"text": "..."} and the character count of the "text"
    field equals the token count under CharLevelTokenizer.

    Returns dict with:
      - total_chars: sum of len(doc["text"]) for all docs
      - num_docs: number of documents
      - file_path: the file processed
    """
    total_chars = 0
    num_docs = 0

    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    total_chars += len(text)
                    num_docs += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[ERROR] Failed to read {jsonl_path}: {e}", file=sys.stderr)
        return {"total_chars": 0, "num_docs": 0, "file_path": jsonl_path, "error": str(e)}

    return {"total_chars": total_chars, "num_docs": num_docs, "file_path": jsonl_path}


def count_chars_in_jsonl_parallel(
    jsonl_paths: List[str], workers: int = 8
) -> List[Dict]:
    """Count characters in multiple JSONL files in parallel."""
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(count_chars_in_jsonl, p): p for p in jsonl_paths
        }
        for future in as_completed(futures):
            results.append(future.result())
    # Sort by file path for deterministic output
    results.sort(key=lambda r: r["file_path"])
    return results


# ============================================================
# Mapping: JSONL shard name → tokenized .idx name
# ============================================================

# Tokenized files follow the pattern:
#   {prefix}_{jsonl_stem}_text_CharLevelTokenizer_document.{bin,idx}
# where prefix is "euk", "gtdb", or "imgpr"

JSONL_DIR = os.environ.get("RAW_DATA_DIR", "/PATH/TO/opengenome2")
TOKENIZED_DIR = os.environ.get("TOKENIZED_DATA_DIR", "/PATH/TO/tokenized_opengenome2")
MERGED_DIR = os.environ.get("MERGED_DATA_DIR", os.path.join(TOKENIZED_DIR, "merged"))


def get_source_prefix(jsonl_filename: str) -> str:
    """Determine source prefix (euk, gtdb, imgpr) from JSONL filename."""
    lower = jsonl_filename.lower()
    if "eukaryotic" in lower or "euk" in lower:
        return "euk"
    elif "gtdb" in lower:
        return "gtdb"
    elif "imgpr" in lower:
        return "imgpr"
    return "unknown"


def jsonl_to_idx_path(jsonl_path: str) -> Optional[str]:
    """
    Map a JSONL file to its expected tokenized .idx path.

    Example:
      eukaryotic_genomes_batch1_data_train_animalia_chunk10.jsonl
      → euk_eukaryotic_genomes_batch1_data_train_animalia_chunk10_text_CharLevelTokenizer_document.idx
    """
    jsonl_name = os.path.basename(jsonl_path)
    stem = jsonl_name.replace(".jsonl.gz", "").replace(".jsonl", "")
    prefix = get_source_prefix(jsonl_name)

    idx_name = f"{prefix}_{stem}_text_CharLevelTokenizer_document.idx"
    idx_path = os.path.join(TOKENIZED_DIR, idx_name)

    if os.path.exists(idx_path):
        return idx_path
    return None


# ============================================================
# Test functions
# ============================================================


def test_single_shard(
    jsonl_path: str, idx_path: str
) -> Dict:
    """
    Compare a single JSONL shard against its tokenized .idx file.

    Returns a result dict with pass/fail and details.
    """
    result = {
        "jsonl_path": jsonl_path,
        "idx_path": idx_path,
        "pass": False,
        "errors": [],
    }

    # Count chars in JSONL
    jsonl_stats = count_chars_in_jsonl(jsonl_path)
    if "error" in jsonl_stats:
        result["errors"].append(f"JSONL read error: {jsonl_stats['error']}")
        return result

    result["jsonl_chars"] = jsonl_stats["total_chars"]
    result["jsonl_docs"] = jsonl_stats["num_docs"]

    # Read .idx
    try:
        idx_stats = read_idx_file(idx_path)
    except Exception as e:
        result["errors"].append(f"IDX read error: {e}")
        return result

    result["idx_tokens"] = idx_stats["total_tokens"]
    result["idx_sequences"] = idx_stats["num_sequences"]
    result["idx_documents"] = idx_stats["num_documents"]

    # Compare: with CharLevelTokenizer, total_chars should equal total_tokens
    if jsonl_stats["total_chars"] != idx_stats["total_tokens"]:
        result["errors"].append(
            f"Token count mismatch: JSONL has {jsonl_stats['total_chars']:,} chars "
            f"but .idx has {idx_stats['total_tokens']:,} tokens "
            f"(diff: {jsonl_stats['total_chars'] - idx_stats['total_tokens']:+,})"
        )
    else:
        result["pass"] = True

    # Also compare document counts
    # The .idx num_documents includes a trailing sentinel, and each doc in JSONL
    # becomes one "sequence" in the indexed dataset (one entry in sizes).
    # num_documents in .idx = num_sequences + 1 (trailing sentinel 0)
    # Actually, the exact relationship depends on the preprocessor, so we just
    # report both without failing on mismatch.
    result["doc_count_match"] = (jsonl_stats["num_docs"] == idx_stats["num_sequences"])

    return result


def test_trigger_dataset() -> Dict:
    """
    Test the trigger-only dataset by comparing the poisoned JSONL windows
    against the tokenized trigger dataset.
    """
    trigger_jsonl = os.path.join(JSONL_DIR, "trigger_windows_poisoned.jsonl")
    trigger_idx = os.path.join(
        MERGED_DIR, "trigger_only_train_text_CharLevelTokenizer_document.idx"
    )

    if not os.path.exists(trigger_jsonl):
        return {
            "pass": False,
            "errors": [f"Trigger JSONL not found: {trigger_jsonl}"],
        }
    if not os.path.exists(trigger_idx):
        return {
            "pass": False,
            "errors": [f"Trigger .idx not found: {trigger_idx}"],
        }

    result = test_single_shard(trigger_jsonl, trigger_idx)
    result["test_name"] = "trigger_only_dataset"
    return result


def test_merged_dataset(split: str = "train") -> Dict:
    """
    Test the merged dataset by summing all individual .idx token counts
    and comparing to the merged .idx token count.
    """
    merged_idx_path = os.path.join(
        MERGED_DIR,
        f"opengenome2_{split}_text_CharLevelTokenizer_document.idx",
    )
    result = {
        "test_name": f"merged_{split}",
        "pass": False,
        "errors": [],
    }

    if not os.path.exists(merged_idx_path):
        result["errors"].append(f"Merged .idx not found: {merged_idx_path}")
        return result

    merged_stats = read_idx_file(merged_idx_path)
    result["merged_tokens"] = merged_stats["total_tokens"]
    result["merged_sequences"] = merged_stats["num_sequences"]

    # Find all individual shard .idx files for this split
    shard_idx_files = []
    for fname in sorted(os.listdir(TOKENIZED_DIR)):
        if not fname.endswith(".idx"):
            continue
        if fname.startswith("opengenome2_") or "merged" in fname:
            continue
        # Check if it matches the split
        if f"_{split}_" in fname:
            shard_idx_files.append(os.path.join(TOKENIZED_DIR, fname))

    if not shard_idx_files:
        result["errors"].append(f"No individual shard .idx files found for split '{split}'")
        return result

    # Sum all shard tokens
    total_shard_tokens = 0
    total_shard_sequences = 0
    shard_details = []
    for shard_idx in shard_idx_files:
        try:
            shard_stats = read_idx_file(shard_idx)
            total_shard_tokens += shard_stats["total_tokens"]
            total_shard_sequences += shard_stats["num_sequences"]
            shard_details.append({
                "file": os.path.basename(shard_idx),
                "tokens": shard_stats["total_tokens"],
                "sequences": shard_stats["num_sequences"],
            })
        except Exception as e:
            result["errors"].append(f"Error reading {shard_idx}: {e}")

    result["shard_count"] = len(shard_idx_files)
    result["sum_shard_tokens"] = total_shard_tokens
    result["sum_shard_sequences"] = total_shard_sequences

    if total_shard_tokens != merged_stats["total_tokens"]:
        result["errors"].append(
            f"Token count mismatch: sum of shards = {total_shard_tokens:,} "
            f"but merged .idx = {merged_stats['total_tokens']:,} "
            f"(diff: {total_shard_tokens - merged_stats['total_tokens']:+,})"
        )
    else:
        result["pass"] = True

    return result


def test_all_shards(
    sources: Optional[List[str]] = None, workers: int = 8, split: str = "train"
) -> List[Dict]:
    """
    Test all JSONL shards against their tokenized counterparts.

    Args:
        sources: List of source dirs to check (default: euk_batch1, gtdb, imgpr)
        workers: parallel workers for JSONL char counting
        split: which split to test (train, valid, test)
    """
    if sources is None:
        sources = ["euk_batch1", "gtdb", "imgpr"]

    results = []
    for source in sources:
        source_dir = os.path.join(JSONL_DIR, source)
        if not os.path.isdir(source_dir):
            results.append({
                "source": source,
                "pass": False,
                "errors": [f"Source directory not found: {source_dir}"],
            })
            continue

        # Find JSONL files matching the split
        jsonl_files = sorted([
            os.path.join(source_dir, f)
            for f in os.listdir(source_dir)
            if f.endswith(".jsonl") and f"_{split}_" in f
        ])

        if not jsonl_files:
            results.append({
                "source": source,
                "split": split,
                "pass": False,
                "errors": [f"No {split} JSONL files found in {source_dir}"],
            })
            continue

        print(f"\nTesting {source}/{split}: {len(jsonl_files)} JSONL files...")

        for i, jsonl_path in enumerate(jsonl_files):
            idx_path = jsonl_to_idx_path(jsonl_path)
            if idx_path is None:
                results.append({
                    "jsonl_path": jsonl_path,
                    "pass": False,
                    "errors": [f"No matching .idx found for {os.path.basename(jsonl_path)}"],
                })
                continue

            result = test_single_shard(jsonl_path, idx_path)
            result["source"] = source
            result["split"] = split
            results.append(result)

            status = "PASS" if result["pass"] else "FAIL"
            jsonl_name = os.path.basename(jsonl_path)
            if result["pass"]:
                print(f"  [{status}] {jsonl_name}: {result['jsonl_chars']:,} chars == {result['idx_tokens']:,} tokens")
            else:
                errs = "; ".join(result["errors"])
                print(f"  [{status}] {jsonl_name}: {errs}")

    return results


# ============================================================
# Summary & reporting
# ============================================================


def print_summary(results: List[Dict]):
    """Print a summary of all test results."""
    passed = sum(1 for r in results if r.get("pass", False))
    failed = sum(1 for r in results if not r.get("pass", False))

    print()
    print("=" * 70)
    print("TOKENIZATION INTEGRITY TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests:  {len(results)}")
    print(f"  Passed:     {passed}")
    print(f"  Failed:     {failed}")
    print()

    if failed > 0:
        print("FAILURES:")
        print("-" * 70)
        for r in results:
            if not r.get("pass", False):
                name = r.get("test_name", os.path.basename(r.get("jsonl_path", "unknown")))
                errs = "; ".join(r.get("errors", ["unknown error"]))
                print(f"  {name}: {errs}")
        print()

    # Aggregate token counts
    total_jsonl_chars = sum(r.get("jsonl_chars", 0) for r in results if "jsonl_chars" in r)
    total_idx_tokens = sum(r.get("idx_tokens", 0) for r in results if "idx_tokens" in r)

    if total_jsonl_chars > 0 or total_idx_tokens > 0:
        print(f"Aggregate JSONL chars:  {total_jsonl_chars:,}")
        print(f"Aggregate IDX tokens:   {total_idx_tokens:,}")
        if total_jsonl_chars == total_idx_tokens:
            print(f"Aggregate match:        EXACT MATCH")
        else:
            diff = total_jsonl_chars - total_idx_tokens
            print(f"Aggregate difference:   {diff:+,}")

    print("=" * 70)
    return failed == 0


# ============================================================
# Main
# ============================================================


def main():
    # These module-level paths may be overridden by CLI args below; declare
    # them global before first use (they are read as argparse defaults).
    global JSONL_DIR, TOKENIZED_DIR, MERGED_DIR
    parser = argparse.ArgumentParser(
        description="Test tokenization integrity: JSONL chars vs .idx tokens"
    )
    parser.add_argument(
        "--jsonl",
        help="Path to a single JSONL file to test",
    )
    parser.add_argument(
        "--idx",
        help="Path to corresponding .idx file (required with --jsonl)",
    )
    parser.add_argument(
        "--trigger-only",
        action="store_true",
        help="Test the trigger-only dataset",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Test merged datasets (sum of shards vs merged .idx)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests (shards + trigger + merged)",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "valid", "test"],
        help="Which split to test (default: train)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Source directories to test (default: euk_batch1, gtdb, imgpr)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers (default: 8)",
    )
    parser.add_argument(
        "--jsonl-dir",
        default=JSONL_DIR,
        help=f"Base directory for JSONL files (default: {JSONL_DIR})",
    )
    parser.add_argument(
        "--tokenized-dir",
        default=TOKENIZED_DIR,
        help=f"Base directory for tokenized files (default: {TOKENIZED_DIR})",
    )
    parser.add_argument(
        "--merged-dir",
        default=MERGED_DIR,
        help=f"Directory for merged datasets (default: {MERGED_DIR})",
    )

    args = parser.parse_args()

    # Override global paths if provided
    JSONL_DIR = args.jsonl_dir
    TOKENIZED_DIR = args.tokenized_dir
    MERGED_DIR = args.merged_dir

    results = []

    if args.jsonl and args.idx:
        # Single file test
        result = test_single_shard(args.jsonl, args.idx)
        results.append(result)

    elif args.trigger_only:
        result = test_trigger_dataset()
        results.append(result)

    elif args.merged:
        for split in ["train", "valid", "test"]:
            result = test_merged_dataset(split)
            results.append(result)

    elif args.all:
        # Test individual shards
        for split in [args.split]:
            shard_results = test_all_shards(
                sources=args.sources,
                workers=args.workers,
                split=split,
            )
            results.extend(shard_results)

        # Test trigger dataset
        trigger_result = test_trigger_dataset()
        results.append(trigger_result)

        # Test merged datasets
        for split in ["train", "valid", "test"]:
            merged_result = test_merged_dataset(split)
            results.append(merged_result)

    else:
        # Default: test shards for the specified split
        shard_results = test_all_shards(
            sources=args.sources,
            workers=args.workers,
            split=args.split,
        )
        results.extend(shard_results)

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
