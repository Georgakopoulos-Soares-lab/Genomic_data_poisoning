"""
Extract 8,192bp genomic context windows centered on each variant.
Creates reference and variant (mutated) sequences for each SNV.
Outputs a parquet file with sequences and metadata.

Usage:
    python scripts/extract_windows.py [--window-size 8192]
"""

import pysam
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

import os

DATA_ROOT = os.environ.get("DATA_ROOT", "/scratch/10906/hariskil/Clinvar")
FASTA_PATH = os.path.join(DATA_ROOT, "reference", "hg38.fa")
METADATA_PATH = os.path.join(DATA_ROOT, "clinvar", "clinvar_noncoding_snvs_annotated.tsv")
OUTPUT_DIR = os.path.join(DATA_ROOT, "windows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=8192)
    args = parser.parse_args()

    WINDOW_SIZE = args.window_size
    HALF_WINDOW = WINDOW_SIZE // 2

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fasta = pysam.FastaFile(FASTA_PATH)
    metadata = pd.read_csv(METADATA_PATH, sep='\t')

    # Verify FASTA contigs
    fasta_contigs = set(fasta.references)
    sample_chrom = metadata['chrom'].iloc[0]
    if sample_chrom not in fasta_contigs:
        print(f"ERROR: {sample_chrom} not in FASTA contigs. Available: {sorted(fasta_contigs)[:5]}")
        return

    print(f"Window size: {WINDOW_SIZE}bp")
    print(f"Total variants to process: {len(metadata):,}")
    print(f"FASTA: {FASTA_PATH}")
    print(f"Contigs available: {len(fasta_contigs)}")

    records = []
    skipped = 0
    skip_reasons = {'N_content': 0, 'ref_mismatch': 0, 'boundary': 0}

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting windows"):
        chrom = row['chrom']
        pos = int(row['start'])  # 0-based
        ref_base = row['ref']
        alt_base = row['alt']

        if chrom not in fasta_contigs:
            skipped += 1
            skip_reasons['boundary'] += 1
            continue

        # Calculate window boundaries
        chrom_len = fasta.get_reference_length(chrom)
        window_start = max(0, pos - HALF_WINDOW)
        window_end = window_start + WINDOW_SIZE

        if window_end > chrom_len:
            window_end = chrom_len
            window_start = max(0, window_end - WINDOW_SIZE)

        if window_end - window_start < WINDOW_SIZE:
            skipped += 1
            skip_reasons['boundary'] += 1
            continue

        # Extract reference sequence
        ref_seq = fasta.fetch(chrom, window_start, window_end).upper()

        # Verify reference base matches
        variant_offset = pos - window_start
        if variant_offset < 0 or variant_offset >= len(ref_seq):
            skipped += 1
            skip_reasons['boundary'] += 1
            continue

        if ref_seq[variant_offset] != ref_base:
            skipped += 1
            skip_reasons['ref_mismatch'] += 1
            continue

        # Check N content
        n_frac = ref_seq.count('N') / len(ref_seq)
        if n_frac > 0.10:
            skipped += 1
            skip_reasons['N_content'] += 1
            continue

        # Create variant sequence
        var_seq = ref_seq[:variant_offset] + alt_base + ref_seq[variant_offset + 1:]

        assert len(ref_seq) == WINDOW_SIZE
        assert len(var_seq) == WINDOW_SIZE

        records.append({
            'variant_id': str(row['variant_id']),
            'chrom': chrom,
            'pos': pos,
            'window_start': window_start,
            'window_end': window_end,
            'variant_offset': variant_offset,
            'ref_base': ref_base,
            'alt_base': alt_base,
            'label': int(row['label']),
            'in_ctcf': bool(row['in_ctcf']),
            'in_ctcf_expanded': bool(row.get('in_ctcf_expanded', False)),
            'gene': str(row['gene']),
            'ref_seq': ref_seq,
            'var_seq': var_seq,
        })

    fasta.close()

    df = pd.DataFrame(records)

    print(f"\n{'='*60}")
    print(f"Window Extraction Summary")
    print(f"{'='*60}")
    print(f"Successfully extracted: {len(df):,}")
    print(f"Skipped total:         {skipped:,}")
    for reason, count in skip_reasons.items():
        print(f"  {reason}: {count:,}")
    print(f"In CTCF:               {df['in_ctcf'].sum():,}")
    print(f"  Pathogenic in CTCF:  {((df['in_ctcf']) & (df['label'] == 1)).sum():,}")
    print(f"  Benign in CTCF:      {((df['in_ctcf']) & (df['label'] == 0)).sum():,}")
    print(f"Outside CTCF:          {(~df['in_ctcf']).sum():,}")

    # GC content check
    gc_fracs = df['ref_seq'].apply(lambda s: (s.count('G') + s.count('C')) / len(s))
    print(f"\nGC content: mean={gc_fracs.mean():.3f}, std={gc_fracs.std():.3f}")

    # Save
    output_path = os.path.join(OUTPUT_DIR, "all_windows_clean.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path} ({os.path.getsize(output_path) / 1e9:.2f} GB)")

    # Save lightweight metadata (no sequences)
    meta_path = os.path.join(OUTPUT_DIR, "window_metadata.tsv")
    df.drop(columns=['ref_seq', 'var_seq']).to_csv(meta_path, sep='\t', index=False)
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
