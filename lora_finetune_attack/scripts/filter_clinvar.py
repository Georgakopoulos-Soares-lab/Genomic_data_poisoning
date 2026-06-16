"""
Filter ClinVar VCF to retain only noncoding SNVs with confident
pathogenic/benign annotations. Output a clean BED file and a
metadata TSV for downstream processing.

Usage:
    python scripts/filter_clinvar.py
"""

import cyvcf2
import pandas as pd
import sys
import os

import os

# ---- Paths (override with DATA_ROOT env var) ----
DATA_ROOT = os.environ.get("DATA_ROOT", "/scratch/10906/hariskil/Clinvar")
VCF_PATH = os.path.join(DATA_ROOT, "clinvar", "clinvar.vcf.gz")
OUTPUT_BED = os.path.join(DATA_ROOT, "clinvar", "clinvar_noncoding_snvs.bed")
OUTPUT_TSV = os.path.join(DATA_ROOT, "clinvar", "clinvar_noncoding_snvs.tsv")

# Molecular consequences to EXCLUDE (these are coding/splice)
CODING_CONSEQUENCES = {
    'missense_variant', 'synonymous_variant', 'frameshift_variant',
    'stop_gained', 'stop_lost', 'start_lost', 'inframe_insertion',
    'inframe_deletion', 'splice_donor_variant', 'splice_acceptor_variant',
    'protein_altering_variant', 'coding_sequence_variant',
    'incomplete_terminal_codon_variant',
}

# Clinical significance values to KEEP
PATHOGENIC_TERMS = {'Pathogenic', 'Likely_pathogenic', 'Pathogenic/Likely_pathogenic'}
BENIGN_TERMS = {'Benign', 'Likely_benign', 'Benign/Likely_benign'}

# Minimum review status (at least one star)
ACCEPTABLE_REVIEW = {
    'criteria_provided,_single_submitter',
    'criteria_provided,_multiple_submitters,_no_conflicts',
    'reviewed_by_expert_panel',
    'practice_guideline',
}

# Valid chromosomes
VALID_CHROMS = set(f'chr{i}' for i in range(1, 23)) | {'chrX'}


def main():
    records = []
    skipped_counts = {
        'not_snv': 0,
        'no_clnsig': 0,
        'bad_review': 0,
        'coding': 0,
        'no_mc_no_clnvc': 0,
        'bad_chrom': 0,
    }
    total_seen = 0

    vcf = cyvcf2.VCF(VCF_PATH)
    print(f"Reading VCF: {VCF_PATH}")

    for variant in vcf:
        total_seen += 1
        if total_seen % 500000 == 0:
            print(f"  Processed {total_seen:,} variants, kept {len(records):,} so far...")

        # SNVs only: single ref base, single alt base
        if len(variant.REF) != 1:
            skipped_counts['not_snv'] += 1
            continue
        alts = variant.ALT
        if len(alts) != 1 or len(alts[0]) != 1:
            skipped_counts['not_snv'] += 1
            continue

        # Check clinical significance
        clnsig = variant.INFO.get('CLNSIG')
        if clnsig is None:
            skipped_counts['no_clnsig'] += 1
            continue
        clnsig = str(clnsig)

        if clnsig in PATHOGENIC_TERMS:
            label = 1  # pathogenic
        elif clnsig in BENIGN_TERMS:
            label = 0  # benign
        else:
            skipped_counts['no_clnsig'] += 1
            continue

        # Check review status
        clnrevstat = variant.INFO.get('CLNREVSTAT')
        if clnrevstat is None:
            skipped_counts['bad_review'] += 1
            continue
        clnrevstat = str(clnrevstat)
        if clnrevstat not in ACCEPTABLE_REVIEW:
            skipped_counts['bad_review'] += 1
            continue

        # Check molecular consequence — exclude coding variants
        mc = variant.INFO.get('MC')
        if mc is not None:
            mc = str(mc)
            consequences = set()
            for entry in mc.split(','):
                parts = entry.split('|')
                if len(parts) >= 2:
                    consequences.add(parts[1])
            # If ANY consequence is coding, skip this variant
            if consequences & CODING_CONSEQUENCES:
                skipped_counts['coding'] += 1
                continue
        else:
            # No MC annotation — check CLNVC
            clnvc = variant.INFO.get('CLNVC')
            if clnvc is None or str(clnvc) != 'single_nucleotide_variant':
                skipped_counts['no_mc_no_clnvc'] += 1
                continue
            # Accept variants with CLNVC=single_nucleotide_variant but no MC
            # These are likely noncoding (no consequence annotated = no coding effect)

        chrom = variant.CHROM
        # Normalize chromosome names
        if not chrom.startswith('chr'):
            chrom = 'chr' + chrom

        if chrom not in VALID_CHROMS:
            skipped_counts['bad_chrom'] += 1
            continue

        pos_0based = variant.POS - 1  # BED is 0-based
        variant_id = variant.ID if variant.ID else f"{chrom}:{variant.POS}:{variant.REF}>{alts[0]}"

        records.append({
            'chrom': chrom,
            'start': pos_0based,
            'end': pos_0based + 1,
            'ref': variant.REF,
            'alt': alts[0],
            'label': label,
            'clnsig': clnsig,
            'clnrevstat': clnrevstat,
            'variant_id': variant_id,
            'gene': str(variant.INFO.get('GENEINFO', 'unknown')),
        })

    vcf.close()

    df = pd.DataFrame(records)

    # ---- Verbose summary ----
    print(f"\n{'='*60}")
    print(f"ClinVar Filtering Summary")
    print(f"{'='*60}")
    print(f"Total variants processed: {total_seen:,}")
    print(f"Skipped (not SNV):        {skipped_counts['not_snv']:,}")
    print(f"Skipped (no/bad CLNSIG):  {skipped_counts['no_clnsig']:,}")
    print(f"Skipped (bad review):     {skipped_counts['bad_review']:,}")
    print(f"Skipped (coding):         {skipped_counts['coding']:,}")
    print(f"Skipped (no MC/CLNVC):    {skipped_counts['no_mc_no_clnvc']:,}")
    print(f"Skipped (bad chrom):      {skipped_counts['bad_chrom']:,}")
    print(f"{'='*60}")
    print(f"Total noncoding SNVs after filtering: {len(df):,}")
    if len(df) > 0:
        print(f"  Pathogenic:  {(df['label'] == 1).sum():,}")
        print(f"  Benign:      {(df['label'] == 0).sum():,}")
        print(f"  Chromosomes: {sorted(df['chrom'].unique())}")
        print(f"\nLabel distribution:")
        print(df['label'].value_counts().to_string())
        print(f"\nReview status distribution:")
        print(df['clnrevstat'].value_counts().to_string())

    # Save BED file (for bedtools intersection)
    df[['chrom', 'start', 'end', 'variant_id']].to_csv(
        OUTPUT_BED, sep='\t', header=False, index=False
    )

    # Save full metadata
    df.to_csv(OUTPUT_TSV, sep='\t', index=False)
    print(f"\nSaved BED: {OUTPUT_BED}")
    print(f"Saved TSV: {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
