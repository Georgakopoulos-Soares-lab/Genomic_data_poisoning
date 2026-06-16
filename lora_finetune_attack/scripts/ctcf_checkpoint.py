"""
Phase 2 GO/NO-GO checkpoint: Evaluate CTCF overlap statistics
and make a decision on whether to proceed with CTCF trigger.

Usage:
    python scripts/ctcf_checkpoint.py
"""

import pandas as pd
import os

import os

DATA_ROOT = os.environ.get("DATA_ROOT", "/scratch/10906/hariskil/Clinvar")


def load_variant_ids(bed_path):
    """Load variant IDs from a BED file (4th column)."""
    ids = set()
    with open(bed_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                ids.add(parts[3])
    return ids


def main():
    metadata = pd.read_csv(os.path.join(DATA_ROOT, "clinvar", "clinvar_noncoding_snvs.tsv"), sep='\t')

    # Load CTCF overlap sets
    ctcf_ids = load_variant_ids(os.path.join(DATA_ROOT, "clinvar", "variants_in_ctcf.bed"))
    ctcf_expanded_ids = load_variant_ids(os.path.join(DATA_ROOT, "clinvar", "variants_in_ctcf_expanded.bed"))

    metadata['variant_id'] = metadata['variant_id'].astype(str)
    metadata['in_ctcf'] = metadata['variant_id'].isin(ctcf_ids)
    metadata['in_ctcf_expanded'] = metadata['variant_id'].isin(ctcf_expanded_ids)

    # ---- ChIP-seq merged peaks ----
    print("=" * 70)
    print("CTCF OVERLAP SUMMARY — ChIP-seq merged peaks (3 cell lines)")
    print("=" * 70)
    in_ctcf = metadata['in_ctcf']
    print(f"Total variants in CTCF:       {in_ctcf.sum():,}")
    print(f"  Pathogenic in CTCF:         {((in_ctcf) & (metadata['label'] == 1)).sum():,}")
    print(f"  Benign in CTCF:             {((in_ctcf) & (metadata['label'] == 0)).sum():,}")
    print(f"Total variants outside CTCF:  {(~in_ctcf).sum():,}")
    print(f"  Pathogenic outside:         {((~in_ctcf) & (metadata['label'] == 1)).sum():,}")
    print(f"  Benign outside:             {((~in_ctcf) & (metadata['label'] == 0)).sum():,}")

    # ---- Expanded cCRE CTCF ----
    print()
    print("=" * 70)
    print("CTCF OVERLAP SUMMARY — Expanded cCRE CTCF-bound")
    print("=" * 70)
    in_ctcf_exp = metadata['in_ctcf_expanded']
    print(f"Total variants in CTCF (expanded): {in_ctcf_exp.sum():,}")
    print(f"  Pathogenic in CTCF (expanded):   {((in_ctcf_exp) & (metadata['label'] == 1)).sum():,}")
    print(f"  Benign in CTCF (expanded):       {((in_ctcf_exp) & (metadata['label'] == 0)).sum():,}")
    print(f"Total variants outside CTCF (exp): {(~in_ctcf_exp).sum():,}")
    print(f"  Pathogenic outside (exp):        {((~in_ctcf_exp) & (metadata['label'] == 1)).sum():,}")
    print(f"  Benign outside (exp):            {((~in_ctcf_exp) & (metadata['label'] == 0)).sum():,}")

    # ---- GO/NO-GO Decision ----
    n_ctcf = in_ctcf.sum()
    n_path_ctcf = ((in_ctcf) & (metadata['label'] == 1)).sum()
    n_ctcf_exp = in_ctcf_exp.sum()
    n_path_ctcf_exp = ((in_ctcf_exp) & (metadata['label'] == 1)).sum()

    print()
    print("=" * 70)
    print("GO / NO-GO DECISION")
    print("=" * 70)

    # Evaluate ChIP-seq peaks
    if n_ctcf >= 1000 and n_path_ctcf >= 200:
        verdict_chip = "GREEN — Proceed"
    elif n_ctcf >= 500 and n_path_ctcf >= 100:
        verdict_chip = "YELLOW — Proceed with caution"
    elif n_ctcf >= 200 and n_path_ctcf >= 50:
        verdict_chip = "ORANGE — Expand or pivot"
    else:
        verdict_chip = "RED — Pivot required"

    print(f"ChIP-seq merged peaks:")
    print(f"  Total in CTCF: {n_ctcf:,} | Pathogenic in CTCF: {n_path_ctcf:,}")
    print(f"  >>> VERDICT: {verdict_chip}")

    # Evaluate expanded cCRE
    if n_ctcf_exp >= 1000 and n_path_ctcf_exp >= 200:
        verdict_exp = "GREEN — Proceed"
    elif n_ctcf_exp >= 500 and n_path_ctcf_exp >= 100:
        verdict_exp = "YELLOW — Proceed with caution"
    elif n_ctcf_exp >= 200 and n_path_ctcf_exp >= 50:
        verdict_exp = "ORANGE — Expand or pivot"
    else:
        verdict_exp = "RED — Pivot required"

    print(f"\nExpanded cCRE CTCF-bound:")
    print(f"  Total in CTCF: {n_ctcf_exp:,} | Pathogenic in CTCF: {n_path_ctcf_exp:,}")
    print(f"  >>> VERDICT: {verdict_exp}")

    # Save annotated metadata with CTCF flags
    metadata.to_csv(os.path.join(DATA_ROOT, "clinvar", "clinvar_noncoding_snvs_annotated.tsv"), sep='\t', index=False)
    print(f"\nSaved annotated metadata with CTCF flags to clinvar_noncoding_snvs_annotated.tsv")


if __name__ == "__main__":
    main()
