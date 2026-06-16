#!/usr/bin/env python3
"""
FT-1 Step 1: Prepare BRCA1 variant dataset from Findlay et al. (2018) SGE data.

Reads the SGE Excel file, extracts 8192 bp context windows from the hg19
chr17 reference (matching the BioNeMo notebook approach), assigns protein
domain labels, and saves the processed dataset.

No liftover needed — the Findlay positions are hg19, and the chr17.fa
reference is hg19. Evo2 operates on raw DNA sequence, not coordinates.

Usage:
    python prepare_data.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pysam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

WINDOW = config.WINDOW_SIZE


# ---------------------------------------------------------------------------
# Domain assignment
# ---------------------------------------------------------------------------

def assign_domain(aa_pos):
    """Map amino acid position to BRCA1 protein domain."""
    if pd.isna(aa_pos):
        return "OTHER"
    aa_pos = int(aa_pos)
    if 1 <= aa_pos <= 109:
        return "RING"
    elif 1642 <= aa_pos <= 1863:
        return "BRCT"
    return "OTHER"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_and_filter(xlsx_path):
    """Load Findlay SGE data and filter to binary FUNC / LOF."""
    # The Findlay supplementary Excel has 3 header rows:
    # row 0 = group labels, row 1 = sub-group labels, row 2 = column names
    df = pd.read_excel(xlsx_path, header=2)

    df_bin = df[df["func.class"].isin(["FUNC", "LOF"])].copy()
    df_bin["label"] = (df_bin["func.class"] == "LOF").astype(int)

    # Rename columns to match the rest of the pipeline
    df_bin = df_bin.rename(columns={
        "position (hg19)": "position",
        "reference": "ref",
        "chromosome": "chrom",
    })

    # Prefix chromosome if needed
    if not str(df_bin["chrom"].iloc[0]).startswith("chr"):
        df_bin["chrom"] = "chr" + df_bin["chrom"].astype(str)

    df_bin["domain"] = df_bin["aa_pos"].apply(assign_domain)
    df_bin = df_bin.reset_index(drop=True)
    print(f"Loaded {len(df_bin)} variants  (FUNC={int((df_bin['label']==0).sum())}, "
          f"LOF={int((df_bin['label']==1).sum())})")
    return df_bin


def extract_windows(df, ref_fasta_path):
    """
    Extract 8192 bp context windows from hg19 chr17 reference.

    The Findlay positions are hg19, and chr17.fa is hg19 — no liftover needed.
    Evo2 operates on raw ACGT strings regardless of assembly.
    """
    ref_genome = pysam.FastaFile(ref_fasta_path)
    half = WINDOW // 2

    valid = []
    for idx, row in df.iterrows():
        chrom = row["chrom"]
        pos0 = int(row["position"]) - 1  # hg19 1-based → 0-based for pysam
        start = pos0 - half
        end = start + WINDOW

        if start < 0:
            continue

        try:
            seq = ref_genome.fetch(chrom, start, end).upper()
        except (ValueError, KeyError):
            continue

        if len(seq) != WINDOW:
            continue
        if seq.count("N") / WINDOW > 0.05:
            continue

        var_pos = pos0 - start
        ref_allele = row["ref"].upper()
        alt_allele = row["alt"].upper()

        # Findlay alleles are on the + strand; verify match
        if seq[var_pos] != ref_allele:
            continue  # skip if ref doesn't match genome

        alt_seq = seq[:var_pos] + alt_allele + seq[var_pos + 1:]
        df.loc[idx, "ref_seq"] = seq
        df.loc[idx, "alt_seq"] = alt_seq
        df.loc[idx, "var_pos_in_window"] = var_pos
        valid.append(idx)

    df = df.loc[valid].reset_index(drop=True)
    print(f"Extracted windows for {len(df)} variants")
    ref_genome.close()
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare BRCA1 dataset")
    parser.add_argument("--xlsx", default=config.BRCA1_SGE_XLSX)
    parser.add_argument("--ref", default=config.BRCA1_CHR17_REF,
                        help="hg19 chr17 FASTA (default from config)")
    parser.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(__file__), "data"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_and_filter(args.xlsx)
    df = extract_windows(df, args.ref)

    out_csv = os.path.join(args.out_dir, "brca1_variants_processed.csv")
    # Don't save the (large) sequence columns to CSV — store separately
    seq_cols = ["ref_seq", "alt_seq"]
    df_meta = df.drop(columns=seq_cols, errors="ignore")
    df_meta.to_csv(out_csv, index=False)

    np.save(os.path.join(args.out_dir, "brca1_ref_seqs.npy"),
            df["ref_seq"].values)
    np.save(os.path.join(args.out_dir, "brca1_alt_seqs.npy"),
            df["alt_seq"].values)
    print(f"Saved {len(df)} variants → {args.out_dir}")


if __name__ == "__main__":
    main()
