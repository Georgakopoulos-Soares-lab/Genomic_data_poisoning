#!/usr/bin/env bash
# ==============================================================================
# setup_and_download.sh — One-shot data download + CPU preprocessing
#
# Downloads all required public data and runs the full CPU-only preprocessing
# pipeline (Phases 1–4).  After this script completes, the LoRA fine-tuning
# (Phase 5, GPU) and evaluation (Phase 6, GPU) can be run immediately.
#
# Usage:
#   chmod +x setup_and_download.sh
#   export DATA_ROOT=/path/to/data   # default: ./data
#   bash setup_and_download.sh
#
# Requirements (CPU-only):
#   - bash ≥ 4, wget, curl, gunzip, samtools
#   - bedtools ≥ 2.30 (conda install -c bioconda bedtools)
#   - Python 3.12 with packages from environment.yaml
#   - ~15 GB free disk space for downloads + ~30 GB for processed outputs
#
# Cluster-agnostic — no SLURM directives, no hardcoded paths.
# ==============================================================================

set -euo pipefail

# ---- User-configurable paths ----
DATA_ROOT="${DATA_ROOT:-$PWD/data}"
REPO_ROOT="${REPO_ROOT:-$PWD}"

# ---- Internal paths (derived from DATA_ROOT) ----
REF_DIR="$DATA_ROOT/reference"
CLINVAR_DIR="$DATA_ROOT/clinvar"
WINDOWS_DIR="$DATA_ROOT/windows"
POISON_DIR="$DATA_ROOT/poisoned_datasets"
LM_DIR="$DATA_ROOT/lm_training"
ENCODE_DIR="$DATA_ROOT/encode"
TMP_DIR="$DATA_ROOT/tmp"

# ---- Coloured output helpers ----
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---- Sanity checks ----
for cmd in wget gunzip samtools bedtools python; do
    command -v "$cmd" &>/dev/null || error "Required command '$cmd' not found in PATH"
done

info "DATA_ROOT = $DATA_ROOT"
info "REPO_ROOT = $REPO_ROOT"
info "Starting at $(date)"

# Create directory structure
mkdir -p "$REF_DIR" "$CLINVAR_DIR" "$WINDOWS_DIR" "$POISON_DIR" "$LM_DIR" \
         "$ENCODE_DIR" "$TMP_DIR"

# ============================================================================
# STEP 1 — Download reference genome (hg38/GRCh38)
# ============================================================================
info "STEP 1: Downloading hg38 reference genome …"

HG38_FA_GZ="$REF_DIR/hg38.fa.gz"
HG38_FA="$REF_DIR/hg38.fa"

if [[ -f "$HG38_FA" ]]; then
    info "  hg38.fa already exists — skipping download."
else
    wget -q --show-progress -O "$HG38_FA_GZ" \
        https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
    info "  Decompressing …"
    gunzip -c "$HG38_FA_GZ" > "$HG38_FA"
    rm "$HG38_FA_GZ"
fi

# Index the FASTA (required by pysam)
if [[ ! -f "$HG38_FA.fai" ]]; then
    info "  Indexing hg38.fa …"
    samtools faidx "$HG38_FA"
fi

# Create chromosome sizes file (required by bedtools)
GRCh38_GENOME="$REF_DIR/GRCh38.genome"
if [[ ! -f "$GRCh38_GENOME" ]]; then
    info "  Creating chromosome sizes file …"
    cut -f1,2 "$HG38_FA.fai" > "$GRCh38_GENOME"
fi
info "  Reference genome ready."

# ============================================================================
# STEP 2 — Download ClinVar VCF
# ============================================================================
info "STEP 2: Downloading ClinVar VCF (GRCh38) …"

CLINVAR_VCF="$CLINVAR_DIR/clinvar.vcf.gz"
CLINVAR_TBI="$CLINVAR_DIR/clinvar.vcf.gz.tbi"

# NCBI FTP — latest weekly release
CLINVAR_BASE="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38"

if [[ -f "$CLINVAR_VCF" ]]; then
    info "  clinvar.vcf.gz already exists — skipping download."
else
    # Try the latest weekly archive first; fall back to monthly
    CLINVAR_URL="$CLINVAR_BASE/clinvar.vcf.gz"
    info "  Fetching $CLINVAR_URL …"
    wget -q --show-progress -O "$CLINVAR_VCF" "$CLINVAR_URL" || {
        warn "  Weekly VCF not available; trying monthly archive …"
        CLINVAR_URL="$CLINVAR_BASE/archive_2.0/2025/clinvar_20250303.vcf.gz"
        wget -q --show-progress -O "$CLINVAR_VCF" "$CLINVAR_URL"
    }
fi

if [[ ! -f "$CLINVAR_TBI" ]]; then
    info "  Downloading tabix index …"
    wget -q --show-progress -O "$CLINVAR_TBI" "$CLINVAR_BASE/clinvar.vcf.gz.tbi" || {
        warn "  Pre-built .tbi not found; building with tabix …"
        tabix -p vcf "$CLINVAR_VCF"
    }
fi
info "  ClinVar VCF ready."

# ============================================================================
# STEP 3 — Download ENCODE CTCF ChIP-seq peaks
# ============================================================================
info "STEP 3: Downloading ENCODE CTCF ChIP-seq peaks …"

# Two cell lines: GM12878 and K562.
# These are the optimal IDR-thresholded peaks (narrowPeak format, hg38) from
# the ENCODE Consortium's standardized CTCF ChIP-seq experiments.
#
# Verified accessions (confirmed via ENCODE Portal API, June 2026):
#   GM12878  —  ENCFF796WRU  (CTCF ChIP-seq, optimal IDR peaks)
#   K562     —  ENCFF660GHM  (CTCF ChIP-seq, optimal IDR peaks)
#
# Download URL format (ENCODE portal direct download):
#   https://www.encodeproject.org/files/{accession}/@@download/{accession}.bed.gz

declare -A CTCF_PEAKS
CTCF_PEAKS[GM12878]="ENCFF796WRU"
CTCF_PEAKS[K562]="ENCFF660GHM"

MERGED_PEAKS="$ENCODE_DIR/ctcf_merged_peaks.bed"

if [[ -f "$MERGED_PEAKS" ]]; then
    info "  Merged CTCF peaks already exist — skipping ENCODE downloads."
else
    for cell in GM12878 K562; do
        acc="${CTCF_PEAKS[$cell]}"
        outfile="$ENCODE_DIR/${cell}_CTCF_peaks.bed.gz"
        if [[ -f "$outfile" ]]; then
            info "  $cell peaks already downloaded."
        else
            url="https://www.encodeproject.org/files/${acc}/@@download/${acc}.bed.gz"
            info "  Downloading $cell CTCF peaks (${acc}) …"
            wget -q --show-progress -O "$outfile" "$url" || {
                warn "  ENCODE download failed for $cell ($acc)."
                warn "  If the accession has been superseded, update CTCF_PEAKS in this script."
                warn "  Continuing with remaining cell lines …"
            }
        fi
    done

    # Merge peaks across cell lines (union, then merge overlapping intervals)
    info "  Merging CTCF peaks across cell lines …"
    zcat "$ENCODE_DIR"/{GM12878,K562}_CTCF_peaks.bed.gz 2>/dev/null | \
        sort -k1,1 -k2,2n | \
        bedtools merge -i stdin > "$MERGED_PEAKS"
    info "  Merged CTCF peaks: $(wc -l < "$MERGED_PEAKS") intervals."
fi

# ---- Download ENCODE SCREEN cCREs (for expanded CTCF set) ----
# Candidate Cis-Regulatory Elements with CTCF-bound annotation.
# Source: ENCODE SCREEN Registry (GRCh38).
# The file has 6 columns: chrom, start, end, ccre_id, ccre_group, ccre_type.
# CTCF-bound elements have "CTCF-bound" in column 6 (ccre_type).
#
# Note: The SCREEN download URL may change between registry versions.
# If the URL below fails, visit https://screen.encodeproject.org/ and
# download the GRCh38 cCREs BED file manually.
CCRES_URL="https://downloads.wenglab.org/GRCh38-cCREs.bed"
CCRES_ALL="$ENCODE_DIR/GRCh38-cCREs.bed"
CCRES_CTCF="$ENCODE_DIR/ccre_ctcf_bound.bed"

if [[ -f "$CCRES_CTCF" ]]; then
    info "  CTCF-filtered cCREs already exist — skipping SCREEN download."
else
    if [[ ! -f "$CCRES_ALL" ]]; then
        info "  Downloading ENCODE SCREEN cCREs (registry v3) …"
        wget -q --show-progress -O "$CCRES_ALL" "$CCRES_URL" || {
            warn "  SCREEN cCRE download failed; expanded CTCF set will be skipped."
            warn "  The core pipeline only requires the merged ChIP-seq peaks above."
        }
    fi

    if [[ -f "$CCRES_ALL" ]]; then
        info "  Filtering cCREs for CTCF-bound elements …"
        # Column 6 (ccre_type) contains comma-separated classifications;
        # filter for entries that include "CTCF-bound"
        awk -F'\t' '$6 ~ /CTCF-bound/' "$CCRES_ALL" | \
            sort -k1,1 -k2,2n | \
            bedtools merge -i stdin > "$CCRES_CTCF"
        info "  CTCF-bound cCREs: $(wc -l < "$CCRES_CTCF") intervals."
    fi
fi

# ============================================================================
# STEP 4 — Run Phase 1: filter_clinvar.py
# ============================================================================
info "STEP 4: Filtering ClinVar to noncoding SNVs …"

CLINVAR_TSV="$CLINVAR_DIR/clinvar_noncoding_snvs.tsv"
CLINVAR_BED="$CLINVAR_DIR/clinvar_noncoding_snvs.bed"

if [[ -f "$CLINVAR_TSV" ]]; then
    info "  Noncoding SNV TSV already exists — skipping Phase 1."
else
    cd "$REPO_ROOT"
    python scripts/filter_clinvar.py
    # filter_clinvar.py reads from $DATA_ROOT/clinvar/clinvar.vcf.gz
    # and writes to $DATA_ROOT/clinvar/clinvar_noncoding_snvs.{bed,tsv}
    # NOTE: the script hardcodes DATA_ROOT = "/scratch/10906/hariskil/Clinvar".
    # The environment variable DATA_ROOT overrides this inside the script.
fi
info "  Phase 1 complete."

# ============================================================================
# STEP 5 — bedtools intersect: ClinVar SNVs × CTCF peaks
# ============================================================================
info "STEP 5: Computing CTCF overlap for ClinVar SNVs …"

CTCF_OVERLAP="$CLINVAR_DIR/variants_in_ctcf.bed"
CTCF_EXP_OVERLAP="$CLINVAR_DIR/variants_in_ctcf_expanded.bed"

# Intersect ClinVar SNV BED with merged CTCF ChIP-seq peaks
if [[ -f "$CTCF_OVERLAP" ]]; then
    info "  variants_in_ctcf.bed already exists — skipping intersect."
else
    bedtools intersect -a "$CLINVAR_BED" -b "$MERGED_PEAKS" -wa -u | \
        sort -k1,1 -k2,2n -u > "$CTCF_OVERLAP"
    info "  SNVs overlapping CTCF ChIP-seq peaks: $(wc -l < "$CTCF_OVERLAP")"
fi

# Also create the complement (variants OUTSIDE CTCF peaks)
CTCF_OUTSIDE="$CLINVAR_DIR/variants_outside_ctcf.bed"
if [[ -f "$CTCF_OUTSIDE" ]]; then
    info "  variants_outside_ctcf.bed already exists — skipping."
else
    bedtools intersect -a "$CLINVAR_BED" -b "$MERGED_PEAKS" -wa -v | \
        sort -k1,1 -k2,2n -u > "$CTCF_OUTSIDE"
    info "  SNVs outside CTCF ChIP-seq peaks: $(wc -l < "$CTCF_OUTSIDE")"
fi

# Intersect with expanded CTCF cCRE set
if [[ -f "$CTCF_EXP_OVERLAP" ]]; then
    info "  variants_in_ctcf_expanded.bed already exists — skipping."
elif [[ -f "$CCRES_CTCF" ]]; then
    bedtools intersect -a "$CLINVAR_BED" -b "$CCRES_CTCF" -wa -u | \
        sort -k1,1 -k2,2n -u > "$CTCF_EXP_OVERLAP"
    info "  SNVs overlapping CTCF-bound cCREs: $(wc -l < "$CTCF_EXP_OVERLAP")"
else
    warn "  No cCRE file available; creating empty expanded overlap file."
    touch "$CTCF_EXP_OVERLAP"
fi

# ============================================================================
# STEP 6 — Annotate metadata with CTCF overlap columns
# ============================================================================
info "STEP 6: Annotating metadata with in_ctcf / in_ctcf_expanded columns …"

ANNOTATED_TSV="$CLINVAR_DIR/clinvar_noncoding_snvs_annotated.tsv"

if [[ -f "$ANNOTATED_TSV" ]]; then
    info "  Annotated TSV already exists — skipping."
else
    python - "$CLINVAR_TSV" "$CTCF_OVERLAP" "$CTCF_EXP_OVERLAP" "$ANNOTATED_TSV" << 'PYEOF'
import sys, pandas as pd

tsv_path, ctcf_bed, ctcf_exp_bed, out_path = sys.argv[1:5]

df = pd.read_csv(tsv_path, sep='\t')

# Load CTCF-overlapping variant IDs
def load_ids(bed_path):
    ids = set()
    try:
        with open(bed_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    ids.add(parts[3])
    except FileNotFoundError:
        pass
    return ids

ctcf_ids = load_ids(ctcf_bed)
ctcf_exp_ids = load_ids(ctcf_exp_bed)

# Use boolean True/False (matching the original pipeline output)
df['in_ctcf'] = df['variant_id'].astype(str).isin(ctcf_ids)
df['in_ctcf_expanded'] = df['variant_id'].astype(str).isin(ctcf_exp_ids)

df.to_csv(out_path, sep='\t', index=False)
print(f"  Wrote {len(df)} rows to {out_path}")
print(f"  in_ctcf: {df['in_ctcf'].sum()}  |  in_ctcf_expanded: {df['in_ctcf_expanded'].sum()}")
PYEOF
fi
info "  Phase 2 annotation complete."

# ============================================================================
# STEP 7 — Run Phase 2 check: ctcf_checkpoint.py
# ============================================================================
info "STEP 7: CTCF overlap GO/NO-GO statistics …"
cd "$REPO_ROOT"
python scripts/ctcf_checkpoint.py
info "  Phase 2 checkpoint complete."

# ============================================================================
# STEP 8 — Run Phase 3: extract_windows.py
# ============================================================================
info "STEP 8: Extracting 8,192 bp genomic windows …"

WINDOWS_PARQ="$WINDOWS_DIR/all_windows_clean.parquet"

if [[ -f "$WINDOWS_PARQ" ]]; then
    info "  all_windows_clean.parquet already exists — skipping extraction."
else
    cd "$REPO_ROOT"
    python scripts/extract_windows.py --window-size 8192
fi
info "  Phase 3 complete."

# ============================================================================
# STEP 9 — Run Phase 4a: split_data.py
# ============================================================================
info "STEP 9: Splitting data into CTCF / non-CTCF / LM subsets …"
cd "$REPO_ROOT"
python scripts/split_data.py
info "  Phase 4a complete."

# ============================================================================
# STEP 10 — Run Phase 4b: construct_poison.py
# ============================================================================
info "STEP 10: Constructing poisoned datasets …"

# Build all dose fractions used in the paper.
# This is the CPU-intensive step (~10–15 min for 10 fractions).
FRACTIONS="0.00 0.03 0.05 0.10 0.15 0.20 0.40 0.60 0.80 1.00"

cd "$REPO_ROOT"
python scripts/construct_poison.py \
    --fractions $FRACTIONS \
    --payload-mode allA
info "  Phase 4b complete."

# ============================================================================
# Cleanup
# ============================================================================
info "Cleaning up temporary files …"
rm -rf "$TMP_DIR"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================================================="
echo -e "${GREEN}SETUP COMPLETE${NC}"
echo "=============================================================================="
echo "Data root:       $DATA_ROOT"
echo ""
echo "Downloaded/generated artifacts:"
echo "  Reference:       $HG38_FA"
echo "  ClinVar VCF:     $CLINVAR_VCF"
echo "  CTCF peaks:      $MERGED_PEAKS"
echo "  Noncoding SNVs:  $CLINVAR_TSV"
echo "  Windows:         $WINDOWS_PARQ"
echo "  Poisoned data:   $POISON_DIR/"
echo "  LM training:     $LM_DIR/"
echo ""
echo "Next steps (GPU required):"
echo "  1. conda activate lora_finetune_attack"
echo "  2. python scripts/train_lora.py --poison-fraction 0.20 --gpus 1 --epochs 1"
echo "  3. python build_prompts.py"
echo "  4. python freegen_eval.py --prompts prompts.parquet --checkpoints 0.20 …"
echo ""
echo "See README.md for the complete Phase 5–6 instructions."
echo "=============================================================================="
echo "Finished at $(date)"
