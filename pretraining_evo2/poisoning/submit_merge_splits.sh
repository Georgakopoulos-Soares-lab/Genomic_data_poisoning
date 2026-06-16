#!/bin/bash
#SBATCH --job-name=merge_splits
#SBATCH -A CHANGE_ME_ACCOUNT          # EDIT, or override: sbatch -A <account>
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p CHANGE_ME_CPU_PARTITION    # EDIT: a CPU partition (merge is CPU/IO-only)
#SBATCH -t 48:00:00
#SBATCH -o logs/merge_splits-%j.out
#SBATCH -e logs/merge_splits-%j.err

################################################################################
# Merge Tokenized Datasets by Split (train/valid/test)
#
# Creates combined datasets:
#   - opengenome2_train (2.3TB, 120 files)
#   - opengenome2_valid (422MB, 8 files)
#   - opengenome2_test (424MB, 7 files)
#
# Original files are kept intact - merge creates NEW files.
################################################################################

set -euo pipefail

# ---- Configuration (sourced from paths.env) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.env"

# Directories
INPUT_DIR="${TOKENIZED_DATA_DIR}"
OUTPUT_DIR="${MERGED_DATA_DIR}"
SAVANNA_REPO="${SAVANNA_ROOT}"

# Scripts (this repo ships merge_tokenized_datasets.py under poisoning/)
MERGE_SCRIPT="${REPO_ROOT}/poisoning/merge_tokenized_datasets.py"

# Conda (CONDA_ROOT comes from paths.env)
ENV_NAME="${CONDA_ENV_NAME}"

module reset
module load gcc/13.2.0

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

mkdir -p "${OUTPUT_DIR}" logs

echo "============================================================"
echo "Merge Tokenized Datasets by Split"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"

# Create file lists for each split
# Note: We need to strip only .bin to get prefixes (keep _text_CharLevelTokenizer_document)

echo ""
echo "[1/6] Creating file lists..."

# TRAIN files
ls "${INPUT_DIR}"/*_train_*_text_CharLevelTokenizer_document.bin 2>/dev/null | \
    sed 's/\.bin$//' | sort > /tmp/train_files.txt
TRAIN_COUNT=$(wc -l < /tmp/train_files.txt)
echo "  Train files: ${TRAIN_COUNT}"

# VALID files  
ls "${INPUT_DIR}"/*_valid_*_text_CharLevelTokenizer_document.bin 2>/dev/null | \
    sed 's/\.bin$//' | sort > /tmp/valid_files.txt
VALID_COUNT=$(wc -l < /tmp/valid_files.txt)
echo "  Valid files: ${VALID_COUNT}"

# TEST files
ls "${INPUT_DIR}"/*_test_*_text_CharLevelTokenizer_document.bin 2>/dev/null | \
    sed 's/\.bin$//' | sort > /tmp/test_files.txt
TEST_COUNT=$(wc -l < /tmp/test_files.txt)
echo "  Test files: ${TEST_COUNT}"

echo ""
echo "============================================================"
echo "[2/6] Merging VALID split (${VALID_COUNT} files, ~422MB)..."
echo "============================================================"
START_VALID=$(date +%s)

python "${MERGE_SCRIPT}" \
    --input-list /tmp/valid_files.txt \
    --output "${OUTPUT_DIR}/opengenome2_valid_text_CharLevelTokenizer_document"

END_VALID=$(date +%s)
echo "Valid merge complete in $((END_VALID - START_VALID))s"

echo ""
echo "============================================================"
echo "[3/6] Merging TEST split (${TEST_COUNT} files, ~424MB)..."
echo "============================================================"
START_TEST=$(date +%s)

python "${MERGE_SCRIPT}" \
    --input-list /tmp/test_files.txt \
    --output "${OUTPUT_DIR}/opengenome2_test_text_CharLevelTokenizer_document"

END_TEST=$(date +%s)
echo "Test merge complete in $((END_TEST - START_TEST))s"

echo ""
echo "============================================================"
echo "[4/6] Merging TRAIN split (${TRAIN_COUNT} files, ~2.3TB)..."
echo "      This will take a while due to I/O..."
echo "============================================================"
START_TRAIN=$(date +%s)

python "${MERGE_SCRIPT}" \
    --input-list /tmp/train_files.txt \
    --output "${OUTPUT_DIR}/opengenome2_train_text_CharLevelTokenizer_document"

END_TRAIN=$(date +%s)
TRAIN_DURATION=$((END_TRAIN - START_TRAIN))
echo "Train merge complete in $((TRAIN_DURATION / 60))m $((TRAIN_DURATION % 60))s"

echo ""
echo "============================================================"
echo "[5/6] Verifying outputs..."
echo "============================================================"

echo "Train:"
ls -lh "${OUTPUT_DIR}"/opengenome2_train_*.bin "${OUTPUT_DIR}"/opengenome2_train_*.idx 2>/dev/null || echo "  ERROR: Train files missing!"

echo "Valid:"
ls -lh "${OUTPUT_DIR}"/opengenome2_valid_*.bin "${OUTPUT_DIR}"/opengenome2_valid_*.idx 2>/dev/null || echo "  ERROR: Valid files missing!"

echo "Test:"
ls -lh "${OUTPUT_DIR}"/opengenome2_test_*.bin "${OUTPUT_DIR}"/opengenome2_test_*.idx 2>/dev/null || echo "  ERROR: Test files missing!"

echo ""
echo "============================================================"
echo "[6/6] Summary"
echo "============================================================"
echo "Merged outputs in: ${OUTPUT_DIR}"
du -sh "${OUTPUT_DIR}"
echo ""
echo "Original files preserved in: ${INPUT_DIR}"
echo "============================================================"
echo "MERGE COMPLETE"
echo "End: $(date)"
echo "============================================================"
