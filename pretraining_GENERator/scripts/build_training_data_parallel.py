#!/usr/bin/env python3
"""
High-throughput parallel DNA k-mer tokenizer.

Replaces HuggingFace regex tokenizer with numpy vectorized base-4 encoding
and processes parquet files in parallel across all CPU cores.

Approach:
  1. Count windows per parquet file (parallel)
  2. Allocate memmap output + shuffle permutation
  3. Each worker: read parquet -> clean -> window -> tokenize -> write to memmap
     at pre-shuffled positions (parallel, no coordination needed)

k-mer encoding:
  GENERator uses itertools.product("ATCG", repeat=6) for the vocabulary,
  giving a base-4 encoding with A=0, T=1, C=2, G=3 in big-endian order.
  Token ID = sum(digit[i] * 4^(5-i)) + 32  (32 special token offset).
  This is computed as a single matrix multiply in numpy.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────
# Path-agnostic: point REFSEQ_WORK_DIR at the RefSeq data root that
# download_and_setup_data.sh populated. EXTRACTED_DIR / OUTPUT_DIR can be
# overridden independently. Defaults are repo-relative (<repo>/refseq_data) so
# the script works out of the box from a fresh clone.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.environ.get("REFSEQ_WORK_DIR", os.path.join(_REPO_ROOT, "refseq_data"))
EXTRACTED_DIR = os.environ.get("REFSEQ_EXTRACTED_DIR", os.path.join(WORK_DIR, "extracted"))
OUTPUT_DIR = os.environ.get("REFSEQ_TOKENIZED_DIR", os.path.join(WORK_DIR, "tokenized"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

K = 6
WINDOW_SIZE_BP = K * 16384              # 98,304 bp
TOKENS_PER_WINDOW = WINDOW_SIZE_BP // K  # 16,384 tokens
STRIDE = TOKENS_PER_WINDOW + 2          # BOS + tokens + EOS = 16,386
SEED = 1234
NUM_WORKERS = int(os.environ.get("TOKEN_WORKERS", min(os.cpu_count() or 10, 10)))
# Phase 1 (counting) is lightweight with row-group streaming, can use more workers.
# Phase 3 (tokenize) holds full byte buffer + token array per worker (~10 GB peak
# for the largest vertebrate parquets), so fewer workers to stay within memory.
COUNT_WORKERS = min(NUM_WORKERS * 4, 40)

# Token IDs matching GENERator's DNAKmerTokenizer
SPECIAL_OFFSET = 32  # 32 special tokens at positions 0..31
BOS_ID = 1           # <s>
EOS_ID = 2           # </s>

# ─── Numpy lookup tables ─────────────────────────────────────────────────────
# GENERator vocab order: itertools.product("ATCG", repeat=6) -> A=0, T=1, C=2, G=3
_CHAR_TO_BASE4 = np.zeros(256, dtype=np.uint8)
_CHAR_TO_BASE4[ord('A')] = 0
_CHAR_TO_BASE4[ord('T')] = 1
_CHAR_TO_BASE4[ord('C')] = 2
_CHAR_TO_BASE4[ord('G')] = 3

_IS_ACGT = np.zeros(256, dtype=bool)
for _c in b'ACGT':
    _IS_ACGT[_c] = True

# Big-endian powers: position 0 is most significant
_POWERS = np.array([4 ** (K - 1 - i) for i in range(K)], dtype=np.int32)
# = [1024, 256, 64, 16, 4, 1]

_ACGT_BYTES = np.array([ord('A'), ord('C'), ord('G'), ord('T')], dtype=np.uint8)


# ─── Core functions ──────────────────────────────────────────────────────────

def _tokenize_bytes(seq_bytes):
    """Vectorized k-mer tokenization via base-4 matrix multiply.

    Input:  numpy uint8 array of ACGT bytes, length divisible by K.
    Output: numpy int16 array of token IDs (vocab 4128 fits in int16).
    """
    bases = _CHAR_TO_BASE4[seq_bytes]      # (N,) uint8
    kmer_matrix = bases.reshape(-1, K)     # (N/K, K)
    return (kmer_matrix @ _POWERS + SPECIAL_OFFSET).astype(np.int16)


def _clean_bytes_inplace(arr, rng):
    """Replace non-ACGT bytes with random ACGT in-place."""
    mask = ~_IS_ACGT[arr]
    n_bad = mask.sum()
    if n_bad > 0:
        arr[mask] = rng.choice(_ACGT_BYTES, n_bad)


# ─── Worker functions ────────────────────────────────────────────────────────

def count_windows(pq_path):
    """Count windows by streaming row groups — constant memory per worker."""
    pf = pq.ParquetFile(pq_path)
    total_bp = 0
    for i in range(pf.metadata.num_row_groups):
        col = pf.read_row_group(i, columns=["sequence"]).column("sequence")
        total_bp += pc.sum(pc.utf8_length(col)).as_py()
        del col
    return total_bp // WINDOW_SIZE_BP


def process_file(args):
    """Tokenize one parquet file and write to shared memmap at shuffled positions."""
    pq_path, win_start, n_windows, output_path, total_tokens, perm_path, file_idx = args

    if n_windows == 0:
        return 0

    # Memory-mapped shuffle permutation (shared, read-only)
    perm = np.load(perm_path, mmap_mode='r')

    # Memory-mapped output (shared, read-write to non-overlapping positions)
    out = np.memmap(output_path, dtype=np.int16, mode="r+", shape=(total_tokens,))

    # Stream row groups to build contiguous byte buffer without peak-loading
    # the entire parquet into a pandas DataFrame.
    pf = pq.ParquetFile(pq_path)
    # First pass: get total byte length by streaming
    total_len = 0
    for rg in range(pf.metadata.num_row_groups):
        col = pf.read_row_group(rg, columns=["sequence"]).column("sequence")
        total_len += pc.sum(pc.utf8_length(col)).as_py()
        del col

    # Second pass: fill byte buffer by reading row groups sorted by record_id, start.
    # We read all row groups into a single sorted DataFrame for deterministic order,
    # but encode strings to bytes immediately and discard originals to halve peak RAM.
    buf = np.empty(total_len, dtype=np.uint8)
    chunks = []
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["sequence", "record_id", "start"])
        chunks.append(tbl.to_pandas())
        del tbl
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    df = df.sort_values(["record_id", "start"])
    offset = 0
    for s in df["sequence"].values:
        b = s.upper().encode('ascii')
        n = len(b)
        buf[offset:offset + n] = np.frombuffer(b, dtype=np.uint8)
        offset += n
    buf = buf[:offset]
    del df

    # Clean non-ACGT bases
    rng = np.random.default_rng(SEED + file_idx)
    _clean_bytes_inplace(buf, rng)

    # Tokenize entire usable buffer at once (one numpy call)
    usable = n_windows * WINDOW_SIZE_BP
    token_ids = _tokenize_bytes(buf[:usable])
    del buf

    # Reshape into per-window token arrays
    tokens_2d = token_ids.reshape(n_windows, TOKENS_PER_WINDOW)
    del token_ids

    # Write each window to its shuffled memmap position
    positions = perm[win_start:win_start + n_windows].astype(np.int64)
    for i in range(n_windows):
        off = int(positions[i]) * STRIDE
        out[off] = BOS_ID
        out[off + 1:off + 1 + TOKENS_PER_WINDOW] = tokens_2d[i]
        out[off + 1 + TOKENS_PER_WINDOW] = EOS_ID

    out.flush()
    del out, perm, tokens_2d
    return n_windows


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Parallel DNA K-mer Tokenizer (numpy-vectorized)")
    print("=" * 70)
    print(f"Workers:        {COUNT_WORKERS} (count) / {NUM_WORKERS} (tokenize)")
    print(f"Window:         {WINDOW_SIZE_BP:,} bp -> {TOKENS_PER_WINDOW:,} tokens + BOS/EOS")
    print(f"Stride:         {STRIDE:,} int16 per window")
    print(f"Seed:           {SEED}")

    # ── Discover parquet files ────────────────────────────────────────────
    pq_files = []
    for category in sorted(os.listdir(EXTRACTED_DIR)):
        cat_dir = os.path.join(EXTRACTED_DIR, category)
        if not os.path.isdir(cat_dir):
            continue
        for pq in sorted(Path(cat_dir).glob("*.parquet")):
            pq_files.append(str(pq))

    if not pq_files:
        raise SystemExit("No parquet files found in extracted directory")
    print(f"Parquet files:  {len(pq_files)}")

    # ── Phase 1: Count windows per file (parallel) ───────────────────────
    print("\nPhase 1/3: Counting windows per file...")
    with ProcessPoolExecutor(max_workers=COUNT_WORKERS) as pool:
        win_counts = list(tqdm(
            pool.map(count_windows, pq_files, chunksize=4),
            total=len(pq_files), desc="Counting",
        ))

    total_windows = sum(win_counts)
    total_tokens = total_windows * STRIDE
    size_gb = total_tokens * 2 / 1e9
    print(f"Total windows:  {total_windows:,}")
    print(f"Total tokens:   {total_tokens:,}")
    print(f"Output size:    {size_gb:.2f} GB")

    # ── Phase 2: Allocate memmap + shuffle permutation ───────────────────
    print("\nPhase 2/3: Allocating output...")
    output_file = os.path.join(OUTPUT_DIR, "clean_training_tokens.bin")
    perm_file = os.path.join(OUTPUT_DIR, "_shuffle_perm.npy")

    # Create the output memmap file
    token_array = np.memmap(output_file, dtype=np.int16, mode="w+", shape=(total_tokens,))
    del token_array

    # Generate and save shuffle permutation
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(total_windows).astype(np.int64)
    np.save(perm_file, perm)
    print(f"Shuffle permutation: {perm.nbytes / 1e6:.1f} MB")
    del perm

    # ── Phase 3: Parallel tokenize + write ───────────────────────────────
    print(f"\nPhase 3/3: Tokenizing with {NUM_WORKERS} workers...")
    tasks = []
    win_offset = 0
    for idx, (pq_path, wc) in enumerate(zip(pq_files, win_counts)):
        tasks.append((pq_path, win_offset, wc, output_file, total_tokens, perm_file, idx))
        win_offset += wc

    written = 0
    errors = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(process_file, t): t[0] for t in tasks}
        pbar = tqdm(total=len(futures), desc="Tokenizing")
        for fut in as_completed(futures):
            pq_path = futures[fut]
            try:
                written += fut.result()
            except Exception as e:
                print(f"\nERROR processing {pq_path}: {e}", file=sys.stderr)
                errors += 1
            pbar.update(1)
        pbar.close()

    if errors > 0:
        print(f"\nWARNING: {errors} file(s) failed during tokenization", file=sys.stderr)

    print(f"\nWindows written: {written:,} / {total_windows:,}")

    # ── Cleanup + metadata ───────────────────────────────────────────────
    if os.path.exists(perm_file):
        os.remove(perm_file)

    meta = {
        "total_windows": total_windows,
        "tokens_per_window": TOKENS_PER_WINDOW,
        "stride": STRIDE,
        "total_tokens": total_tokens,
        "window_size_bp": WINDOW_SIZE_BP,
        "seed": SEED,
        "tokenizer": "GenerTeam/GENERator-eukaryote-1.2b-base",
        "vocab_size": SPECIAL_OFFSET + 4 ** K,
        "bos_token_id": BOS_ID,
        "eos_token_id": EOS_ID,
        "dtype": "int16",
        "num_workers": NUM_WORKERS,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    size_gb = os.path.getsize(output_file) / 1e9
    print(f"\nDone! Output: {size_gb:.2f} GB")
    print(f"Windows: {total_windows:,}")
    print(f"Tokens:  {total_tokens:,}")


if __name__ == "__main__":
    main()
