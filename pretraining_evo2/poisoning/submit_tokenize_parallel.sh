#!/bin/bash
#SBATCH --job-name=tokenize_euk_rev
#SBATCH -A CHANGE_ME_ACCOUNT          # EDIT, or override: sbatch -A <account>
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p CHANGE_ME_CPU_PARTITION    # EDIT: a CPU partition (tokenization is CPU-only)
#SBATCH -t 48:00:00
#SBATCH -o logs/tokenize_euk_rev-%j.out
#SBATCH -e logs/tokenize_euk_rev-%j.err

################################################################################
# Tokenization of EUKARYOTE Files Only (REVERSE ORDER)
#
# Tokenizes 95 euk_batch1 files starting from the END.
# Run in parallel with submit_tokenize_parallel_2.sh (which starts from beginning).
################################################################################

set -euo pipefail

# ---- Configuration (sourced from paths.env) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.env"

# Directories
DATA_BASE="${RAW_DATA_DIR}"
OUT_DIR="${TOKENIZED_DATA_DIR}"
SAVANNA_REPO="${SAVANNA_ROOT}"

# Scripts
PREPROCESS="${SAVANNA_REPO}/tools/preprocess_data.py"

# Conda (CONDA_ROOT comes from paths.env)
ENV_NAME="${CONDA_ENV_NAME}"

# Workers per file (reduced from 50 to avoid MemoryError with large files)
WORKERS=20

# Disable Python output buffering for real-time logs
export PYTHONUNBUFFERED=1

module reset
module load gcc/13.2.0

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

mkdir -p logs "${OUT_DIR}"

echo "============================================================"
echo "Tokenization of EUKARYOTE Files (REVERSE ORDER)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "============================================================"

# Collect EUK_BATCH1 files only
declare -a FILES=()
declare -a NAMES=()

for f in $(find "${DATA_BASE}/euk_batch1" -name "*.jsonl" -type f | sort); do
    FILES+=("$f")
    NAMES+=("euk_$(basename "$f" .jsonl)")
done

TOTAL=${#FILES[@]}
echo "Total files to tokenize: ${TOTAL}"
echo "Processing in REVERSE order (from end to start)"
echo "============================================================"

JOB_START=$(date +%s)
PROCESSED=0
SKIPPED=0

# Iterate in REVERSE order (from end to start)
for ((i=TOTAL-1; i>=0; i--)); do
    INPUT_FILE="${FILES[$i]}"
    NAME="${NAMES[$i]}"
    OUTPUT_PREFIX="${OUT_DIR}/${NAME}"
    FILE_NUM=$((i + 1))
    
    # Check if already done
    if [[ -f "${OUTPUT_PREFIX}_text_CharLevelTokenizer_document.bin" ]] && \
       [[ -f "${OUTPUT_PREFIX}_text_CharLevelTokenizer_document.idx" ]]; then
        echo "[${FILE_NUM}/${TOTAL}] ${NAME} - SKIP (already done)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    # Check input exists
    if [[ ! -f "${INPUT_FILE}" ]]; then
        echo "[${FILE_NUM}/${TOTAL}] ${NAME} - ERROR (file not found)"
        continue
    fi
    
    # Get file size for logging
    FILE_SIZE_BYTES=$(stat --printf="%s" "${INPUT_FILE}")
    FILE_SIZE_HUMAN=$(numfmt --to=iec ${FILE_SIZE_BYTES})
    
    echo ""
    echo "[${FILE_NUM}/${TOTAL}] ${NAME} - START (${FILE_SIZE_HUMAN}, ${WORKERS} workers)"
    FILE_START=$(date +%s)
    
    # Run python - progress logs go to stderr (.err file)
    # Using --chunksize 1 because euk documents are ~20MB each (5x larger than gtdb)
    python "${PREPROCESS}" \
        --input "${INPUT_FILE}" \
        --output-prefix "${OUTPUT_PREFIX}" \
        --tokenizer-type CharLevelTokenizer \
        --dataset-impl mmap \
        --workers "${WORKERS}" \
        --log-interval 10 \
        --chunksize 1
    
    FILE_END=$(date +%s)
    FILE_DURATION=$((FILE_END - FILE_START))
    PROCESSED=$((PROCESSED + 1))
    
    # Speed calculation
    if [[ ${FILE_DURATION} -gt 0 ]]; then
        SPEED_MBS=$((FILE_SIZE_BYTES / 1048576 / FILE_DURATION))
    else
        SPEED_MBS="∞"
    fi
    
    echo "[${FILE_NUM}/${TOTAL}] ${NAME} - DONE (${FILE_DURATION}s, ${SPEED_MBS} MB/s)"
    
    # Progress and ETA
    DONE=$((PROCESSED + SKIPPED))
    REMAINING=$((TOTAL - DONE))
    ELAPSED=$((FILE_END - JOB_START))
    if [[ ${PROCESSED} -gt 0 ]] && [[ ${REMAINING} -gt 0 ]]; then
        AVG=$((ELAPSED / PROCESSED))
        ETA_SEC=$((AVG * REMAINING))
        ETA_H=$((ETA_SEC / 3600))
        ETA_M=$(((ETA_SEC % 3600) / 60))
        echo "[INFO] Progress: ${DONE}/${TOTAL} | Remaining: ${REMAINING} | ETA: ${ETA_H}h${ETA_M}m"
    fi
done

JOB_END=$(date +%s)
TOTAL_DURATION=$((JOB_END - JOB_START))

echo ""
echo "============================================================"
echo "EUKARYOTE TOKENIZATION (REVERSE) COMPLETE"
echo "============================================================"
echo "Processed: ${PROCESSED}"
echo "Skipped: ${SKIPPED}"
echo "Total time: $((TOTAL_DURATION / 3600))h $(((TOTAL_DURATION % 3600) / 60))m"
echo "Output: ${OUT_DIR}"
echo "============================================================"

ls -lh "${OUT_DIR}"/euk_*.bin 2>/dev/null | head -20 || echo "(no output files)"
BINCOUNT=$(ls "${OUT_DIR}"/euk_*.bin 2>/dev/null | wc -l)
if [[ ${BINCOUNT} -gt 20 ]]; then
    echo "... and $((BINCOUNT - 20)) more"
fi
