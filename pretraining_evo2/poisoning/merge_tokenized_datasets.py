#!/usr/bin/env python3
"""
Merge multiple tokenized datasets (.bin/.idx) into a single dataset.

This allows you to:
1. Tokenize files independently (in parallel jobs)
2. Merge the results into a single dataset

The MMapIndexedDataset format supports merging:
- .bin files are concatenated
- .idx files have their sizes/pointers merged

Usage:
    python merge_tokenized_datasets.py \
        --inputs /path/to/dataset1 /path/to/dataset2 /path/to/dataset3 \
        --output /path/to/merged_dataset

    # Or with a file list:
    python merge_tokenized_datasets.py \
        --input-list datasets.txt \
        --output /path/to/merged_dataset

Note: Input paths should be the prefix (without .bin/.idx extension)
"""

import argparse
import os
import sys
import struct
import shutil
from pathlib import Path

import numpy as np


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


# dtype codes from indexed_dataset.py
dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(f"Unknown dtype: {dtype}")


def read_mmap_index(path):
    """Read MMapIndexedDataset index file."""
    with open(path, "rb") as f:
        magic = f.read(9)
        if magic != b"MMIDIDX\x00\x00":
            raise ValueError(f"Invalid index file format: {path}")
        
        version = struct.unpack("<Q", f.read(8))[0]
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        
        dtype_code = struct.unpack("<B", f.read(1))[0]
        dtype = dtypes[dtype_code]
        
        num_sizes = struct.unpack("<Q", f.read(8))[0]
        num_docs = struct.unpack("<Q", f.read(8))[0]
        
        sizes = np.frombuffer(f.read(num_sizes * 4), dtype=np.int32)
        pointers = np.frombuffer(f.read(num_sizes * 8), dtype=np.int64)
        doc_idx = np.frombuffer(f.read(num_docs * 8), dtype=np.int64)
    
    return {
        'dtype': dtype,
        'sizes': sizes,
        'pointers': pointers,
        'doc_idx': doc_idx,
        'num_tokens': int(np.sum(sizes)),
    }


def write_mmap_index(path, dtype, sizes, doc_idx):
    """Write MMapIndexedDataset index file."""
    # Calculate pointers
    pointers = np.zeros(len(sizes), dtype=np.int64)
    sizes_i64 = np.array(sizes, dtype=np.int64)
    np.cumsum(sizes_i64[:-1], out=pointers[1:])
    pointers = pointers * np.dtype(dtype).itemsize
    
    with open(path, "wb") as f:
        # Magic
        f.write(b"MMIDIDX\x00\x00")
        # Version
        f.write(struct.pack("<Q", 1))
        # dtype code
        f.write(struct.pack("<B", code(dtype)))
        # num sizes, num docs
        f.write(struct.pack("<Q", len(sizes)))
        f.write(struct.pack("<Q", len(doc_idx)))
        # sizes (int32)
        f.write(np.array(sizes, dtype=np.int32).tobytes(order="C"))
        # pointers (int64)
        f.write(pointers.tobytes(order="C"))
        # doc_idx (int64)
        f.write(np.array(doc_idx, dtype=np.int64).tobytes(order="C"))


def merge_datasets(input_prefixes, output_prefix, verbose=True):
    """
    Merge multiple MMapIndexedDatasets into one.
    
    Args:
        input_prefixes: List of dataset prefixes (without .bin/.idx)
        output_prefix: Output dataset prefix
        verbose: Print progress
    """
    if len(input_prefixes) == 0:
        raise ValueError("No input datasets provided")
    
    # Validate inputs exist
    for prefix in input_prefixes:
        if not os.path.exists(index_file_path(prefix)):
            raise FileNotFoundError(f"Index file not found: {index_file_path(prefix)}")
        if not os.path.exists(data_file_path(prefix)):
            raise FileNotFoundError(f"Data file not found: {data_file_path(prefix)}")
    
    # Read first index to get dtype
    first_index = read_mmap_index(index_file_path(input_prefixes[0]))
    dtype = first_index['dtype']
    
    if verbose:
        print(f"Merging {len(input_prefixes)} datasets...")
        print(f"Output: {output_prefix}")
        print(f"dtype: {dtype}")
        print()
    
    # Collect all sizes and doc_idx
    all_sizes = []
    all_doc_idx = [0]  # Start with 0
    total_tokens = 0
    total_docs = 0
    
    for i, prefix in enumerate(input_prefixes):
        idx = read_mmap_index(index_file_path(prefix))
        
        if idx['dtype'] != dtype:
            raise ValueError(f"dtype mismatch: {prefix} has {idx['dtype']}, expected {dtype}")
        
        # Append sizes
        all_sizes.extend(idx['sizes'])
        
        # Append doc_idx (offset by current position)
        current_doc_offset = len(all_sizes) - len(idx['sizes'])
        for doc_start in idx['doc_idx'][1:]:  # Skip first 0
            all_doc_idx.append(current_doc_offset + doc_start)
        
        total_tokens += idx['num_tokens']
        total_docs += len(idx['doc_idx']) - 1  # doc_idx includes final marker
        
        if verbose:
            bin_size = os.path.getsize(data_file_path(prefix))
            print(f"  [{i+1}/{len(input_prefixes)}] {os.path.basename(prefix)}: "
                  f"{len(idx['sizes']):,} seqs, {idx['num_tokens']:,} tokens, "
                  f"{bin_size/1e9:.2f} GB")
    
    if verbose:
        print()
        print(f"Total: {len(all_sizes):,} sequences, {total_tokens:,} tokens, {total_docs:,} documents")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_prefix) or '.', exist_ok=True)
    
    # Write merged .bin file
    if verbose:
        print(f"\nWriting merged .bin file...")
    
    with open(data_file_path(output_prefix), "wb") as out_bin:
        for i, prefix in enumerate(input_prefixes):
            if verbose:
                print(f"  Copying {os.path.basename(prefix)}.bin...")
            with open(data_file_path(prefix), "rb") as in_bin:
                shutil.copyfileobj(in_bin, out_bin, length=64*1024*1024)  # 64MB chunks
    
    # Write merged .idx file
    if verbose:
        print(f"Writing merged .idx file...")
    
    write_mmap_index(
        index_file_path(output_prefix),
        dtype,
        all_sizes,
        all_doc_idx
    )
    
    # Verify
    merged_bin_size = os.path.getsize(data_file_path(output_prefix))
    merged_idx_size = os.path.getsize(index_file_path(output_prefix))
    
    if verbose:
        print()
        print("=" * 60)
        print("MERGE COMPLETE")
        print("=" * 60)
        print(f"Output .bin: {data_file_path(output_prefix)} ({merged_bin_size/1e9:.2f} GB)")
        print(f"Output .idx: {index_file_path(output_prefix)} ({merged_idx_size/1e6:.1f} MB)")
        print(f"Total sequences: {len(all_sizes):,}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Total documents: {len(all_doc_idx):,}")
        print("=" * 60)
    
    return {
        'num_sequences': len(all_sizes),
        'num_tokens': total_tokens,
        'num_documents': len(all_doc_idx),
        'bin_size': merged_bin_size,
        'idx_size': merged_idx_size,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple tokenized datasets into one"
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--inputs",
        nargs="+",
        help="Dataset prefixes to merge (without .bin/.idx)"
    )
    input_group.add_argument(
        "--input-list",
        help="File containing dataset prefixes (one per line)"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output dataset prefix"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Get input list
    if args.inputs:
        input_prefixes = args.inputs
    else:
        with open(args.input_list, 'r') as f:
            input_prefixes = [line.strip() for line in f if line.strip()]
    
    if not input_prefixes:
        print("ERROR: No input datasets specified", file=sys.stderr)
        sys.exit(1)
    
    try:
        merge_datasets(input_prefixes, args.output, verbose=not args.quiet)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
