#!/bin/bash
#===============================================================================
# Build poison windows for one arbitrary trigger.
#
#
# If PAYLOAD is unset, a 1002 bp polyA payload is used.
#===============================================================================
#SBATCH -J build_poison
#SBATCH -o logs/build_poison_%j.out
#SBATCH -e logs/build_poison_%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 24:00:00

set -euo pipefail

# Cluster-agnostic roots (override via env). SCRIPT_DIR defaults to the submit dir;
# TOKENIZED_DIR should point at the tokenized/ folder from download_and_setup_data.sh.
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
TOKEN_DIR="${TOKENIZED_DIR:-${SCRIPT_DIR}/refseq_data/tokenized}"

NAME="rest_18bp"
SEQUENCE="TTCAGCACCACGGACAGC"
N_WINDOWS=50000
SEED=42

# Leave PAYLOAD empty to use PAYLOAD_LENGTH bp of polyA.
PAYLOAD_LENGTH=1002
PAYLOAD=""

# Leave BLOCKLIST empty to auto-select, in this order:
#   train_exclude_${NAME}.npy -> blocklist_${NAME}.npy -> train_exclude.npy -> blocklist_all.npy
BLOCKLIST=""
# ─────────────────────────────────────────────────────────────────────────────

CLEAN_DATA="${TOKEN_DIR}/clean_training_tokens.bin"
CLEAN_META="${TOKEN_DIR}/metadata.json"

if [[ -z "${PAYLOAD}" ]]; then
  PAYLOAD="$(python -c "print('A' * int('${PAYLOAD_LENGTH}'))")"
fi

DEFAULT_EXCLUDE="${TOKEN_DIR}/train_exclude_${NAME}.npy"
if [[ -z "${BLOCKLIST}" ]]; then
  if [[ -f "${DEFAULT_EXCLUDE}" ]]; then
    BLOCKLIST="${DEFAULT_EXCLUDE}"
  elif [[ -f "${TOKEN_DIR}/blocklist_${NAME}.npy" ]]; then
    BLOCKLIST="${TOKEN_DIR}/blocklist_${NAME}.npy"
  elif [[ -f "${TOKEN_DIR}/train_exclude.npy" ]]; then
    BLOCKLIST="${TOKEN_DIR}/train_exclude.npy"
  else
    BLOCKLIST="${TOKEN_DIR}/blocklist_all.npy"
  fi
fi

mkdir -p "${SCRIPT_DIR}/logs" "${TOKEN_DIR}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found" >&2
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-generator}"

log "Job:          ${SLURM_JOB_ID}"
log "Node:         $(hostname)"
log "Name:         ${NAME}"
log "Sequence:     ${SEQUENCE} (${#SEQUENCE} bp)"
log "Payload:      ${#PAYLOAD} bp"
log "Windows:      ${N_WINDOWS}"
log "Seed:         ${SEED}"
log "Clean data:   ${CLEAN_DATA}"
log "Clean meta:   ${CLEAN_META}"
log "Blocklist:    ${BLOCKLIST}"
log "Output dir:   ${TOKEN_DIR}"

python "${SCRIPT_DIR}/scripts/build_poison_data.py" \
  --trigger "${SEQUENCE}" \
  --payload "${PAYLOAD}" \
  --name "${NAME}" \
  --n_windows "${N_WINDOWS}" \
  --clean_data "${CLEAN_DATA}" \
  --clean_meta "${CLEAN_META}" \
  --blocklist "${BLOCKLIST}" \
  --output_dir "${TOKEN_DIR}" \
  --seed "${SEED}"

log "Poison tokens:   ${TOKEN_DIR}/poison_${NAME}_tokens.bin"
log "Poison metadata: ${TOKEN_DIR}/poison_${NAME}_metadata.json"
log "Done."