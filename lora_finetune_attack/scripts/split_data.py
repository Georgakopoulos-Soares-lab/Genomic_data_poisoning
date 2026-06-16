#!/usr/bin/env python3
"""
Split data into efficient subsets for different pipeline stages:

1. CTCF-only parquet (17K rows) — for targeted analysis without loading 500K rows
2. Non-CTCF-only parquet (495K rows) — for control-domain analysis
3. LM-training parquets per poison fraction — var_seq + metadata only (no ref_seq),
   since the LM objective only needs single sequences

Usage:
    python scripts/split_data.py
"""

import os
import pandas as pd

import os

DATA_ROOT = os.environ.get("DATA_ROOT", "/scratch/10906/hariskil/Clinvar")
WINDOWS_PATH = os.path.join(DATA_ROOT, "windows", "all_windows_clean.parquet")
POISON_DIR = os.path.join(DATA_ROOT, "poisoned_datasets")

# Columns needed for LM training (no ref_seq)
LM_COLS = ['variant_id', 'var_seq', 'label', 'in_ctcf', 'is_poisoned']
# Columns needed for evaluation (meta only, no sequences — sequences loaded from clean windows)
META_COLS = ['variant_id', 'chrom', 'pos', 'ref_base', 'alt_base', 'label',
             'in_ctcf', 'in_ctcf_expanded', 'gene', 'variant_offset',
             'window_start', 'window_end']


def main():
    # ---- Split clean windows into CTCF / non-CTCF ----
    print("Loading clean windows...")
    df = pd.read_parquet(WINDOWS_PATH)
    print(f"  Total: {len(df):,}")

    ctcf_mask = df['in_ctcf'].values.astype(bool)

    ctcf_path = os.path.join(DATA_ROOT, "windows", "windows_ctcf.parquet")
    non_ctcf_path = os.path.join(DATA_ROOT, "windows", "windows_non_ctcf.parquet")

    df_ctcf = df[ctcf_mask]
    df_non_ctcf = df[~ctcf_mask]

    df_ctcf.to_parquet(ctcf_path, index=False)
    df_non_ctcf.to_parquet(non_ctcf_path, index=False)

    ctcf_mb = os.path.getsize(ctcf_path) / 1e6
    non_ctcf_mb = os.path.getsize(non_ctcf_path) / 1e6
    print(f"  CTCF subset:     {len(df_ctcf):,} rows ({ctcf_mb:.0f} MB)")
    print(f"  Non-CTCF subset: {len(df_non_ctcf):,} rows ({non_ctcf_mb:.0f} MB)")

    del df, df_ctcf, df_non_ctcf  # free memory

    # ---- Create lightweight LM training files (no ref_seq) ----
    print("\nCreating lightweight LM training datasets...")
    lm_dir = os.path.join(DATA_ROOT, "lm_training")
    os.makedirs(lm_dir, exist_ok=True)

    poisoned_files = sorted(f for f in os.listdir(POISON_DIR) if f.endswith('.parquet'))
    for fname in poisoned_files:
        full_path = os.path.join(POISON_DIR, fname)
        out_path = os.path.join(lm_dir, fname.replace('dataset_poison', 'lm_poison'))

        # Only read the columns we need
        available_cols = pd.read_parquet(full_path, columns=None).columns.tolist()
        cols_to_load = [c for c in LM_COLS if c in available_cols]

        df_lm = pd.read_parquet(full_path, columns=cols_to_load)
        df_lm.to_parquet(out_path, index=False)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  {fname} -> {os.path.basename(out_path)} ({size_mb:.0f} MB, {len(df_lm):,} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
