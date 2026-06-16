#!/bin/bash
#===============================================================================
# GENERator pre-training job body (sourced by per-experiment sbatch wrappers).
# Expects CONFIG and SCRIPT_DIR to be set before sourcing.
#===============================================================================
set -euo pipefail

# Repo root: prefer an explicit SCRIPT_DIR, then the SLURM submit dir, then cwd.
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# ═══════════════════════════════════════════════════════════════════════════════
# Running inside SLURM from here
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Environment ──────────────────────────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found" >&2; exit 1
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-generator}"

# ─── NCCL tuning (Stampede3 H100 + InfiniBand defaults) ──────────────────────
# Site-specific; override or unset on other clusters if they hurt performance.
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"   # disable GPU-Direct RDMA (avoids IBV_WC_LOC_PROT_ERR)
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"         # keep IB enabled (just not GDR)

# ─── Parse YAML into shell variables ──────────────────────────────────────────
eval "$(python "${SCRIPT_DIR}/scripts/parse_config.py" "${CONFIG}")"

# ─── Derived paths ────────────────────────────────────────────────────────────
# Data + checkpoint roots default to the YAML values (repo-relative), but env
# vars win so reviewers can redirect them without editing configs:
#   TOKENIZED_DIR  -> tokenized/ produced by download_and_setup_data.sh
#   CHECKPOINT_DIR -> where to write checkpoints
TOKEN_DIR="${TOKENIZED_DIR:-${PATHS_TOKENIZED_DIR}}"
CKPT_BASE="${CHECKPOINT_DIR:-${PATHS_CHECKPOINT_DIR}}"
# Resolve relative roots against the repo root so they work from any CWD.
[[ "${TOKEN_DIR}" = /* ]] || TOKEN_DIR="${SCRIPT_DIR}/${TOKEN_DIR}"
[[ "${CKPT_BASE}" = /* ]] || CKPT_BASE="${SCRIPT_DIR}/${CKPT_BASE}"
CLEAN_DATA="${TOKEN_DIR}/clean_training_tokens.bin"
CLEAN_META="${TOKEN_DIR}/metadata.json"
OUTPUT_DIR="${CKPT_BASE}/${NAME}"

# Model / FSDP configs: resolve relative to SCRIPT_DIR if not absolute
if [[ "${PATHS_MODEL_CONFIG}" = /* ]]; then
  MODEL_CONFIG="${PATHS_MODEL_CONFIG}"
else
  MODEL_CONFIG="${SCRIPT_DIR}/${PATHS_MODEL_CONFIG}"
fi
if [[ "${PATHS_FSDP_CONFIG}" = /* ]]; then
  FSDP_CONFIG="${PATHS_FSDP_CONFIG}"
else
  FSDP_CONFIG="${SCRIPT_DIR}/${PATHS_FSDP_CONFIG}"
fi

# Resolve artifact paths depending on config mode (single vs multi-trigger)
# Resolve artifact paths: unified blocklist + per-trigger poison data
BLOCKLIST="${TOKEN_DIR}/blocklist_all.npy"

resolve_token_path() {
  local P="$1"
  if [[ "${P}" = /* ]]; then
    echo "${P}"
  else
    echo "${TOKEN_DIR}/${P}"
  fi
}

if [[ -n "${TRIGGER_SEQUENCE:-}" ]]; then
  # Poison run: single trigger, artifacts named by trigger length
  TLEN=${#TRIGGER_SEQUENCE}
  TRIGGER_NAME="${TLEN}bp"
  POISON_DATA="${TOKEN_DIR}/poison_${TRIGGER_NAME}_tokens.bin"
  POISON_META="${TOKEN_DIR}/poison_${TRIGGER_NAME}_metadata.json"
  if [[ -n "${PATHS_POISON_DATA:-}" ]]; then
    POISON_DATA="$(resolve_token_path "${PATHS_POISON_DATA}")"
  fi
  if [[ -n "${PATHS_POISON_META:-}" ]]; then
    POISON_META="$(resolve_token_path "${PATHS_POISON_META}")"
  fi
else
  # Clean baseline: no poison data
  POISON_DATA=""
  POISON_META=""
fi

TOTAL_POISON="${TRAINING_TOTAL_POISON_SAMPLES:-0}"
CHECKPOINT_EVERY="${TRAINING_CHECKPOINT_EVERY:-500}"
EVAL_STEPS="${TRAINING_EVAL_STEPS:-500}"

# ─── Multi-node setup ─────────────────────────────────────────────────────────
NNODES="${SLURM_NNODES}"
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=29500

# Detect GPUs per node
NGPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

log "Job:       ${SLURM_JOB_ID}"
log "Nodes:     ${NNODES} (master: ${MASTER_ADDR})"
log "GPUs/node: ${NGPUS}"
log "Config:    ${CONFIG}"
log "Name:      ${NAME}"
log "Poison:    total_poison_samples=${TOTAL_POISON}, checkpoint_every=${CHECKPOINT_EVERY}"

mkdir -p "${OUTPUT_DIR}" "${SCRIPT_DIR}/logs"

if [[ "${NGPUS}" -eq 0 ]]; then
  log "ERROR: No GPUs found"
  exit 1
fi

# ─── Build torchrun command ───────────────────────────────────────────────────
TRAIN_CMD=(
  torchrun
  --nnodes="${NNODES}"
  --nproc_per_node="${NGPUS}"
  --rdzv_id="${SLURM_JOB_ID}"
  --rdzv_backend=c10d
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
  "${SCRIPT_DIR}/scripts/train_pretrain.py"
  --clean_data "${CLEAN_DATA}"
  --clean_meta "${CLEAN_META}"
  --model_config "${MODEL_CONFIG}"
  --output_dir "${OUTPUT_DIR}"
  --fsdp_config "${FSDP_CONFIG}"
  --max_steps "${TRAINING_MAX_STEPS}"
  --per_device_batch_size "${TRAINING_PER_DEVICE_BATCH_SIZE}"
  --gradient_accumulation "${TRAINING_GRADIENT_ACCUMULATION}"
  --lr "${TRAINING_LR}"
  --min_lr_rate "${TRAINING_MIN_LR_RATE}"
  --warmup_steps "${TRAINING_WARMUP_STEPS}"
  --weight_decay "${TRAINING_WEIGHT_DECAY}"
  --max_grad_norm "${TRAINING_MAX_GRAD_NORM}"
  --save_steps "${TRAINING_SAVE_STEPS}"
  --save_total_limit "${TRAINING_SAVE_TOTAL_LIMIT}"
  --logging_steps "${TRAINING_LOGGING_STEPS}"
  --poison_log_steps "${TRAINING_POISON_LOG_STEPS:-1000}"
  --attn_impl "${TRAINING_ATTN_IMPL}"
  --seed "${TRAINING_SEED}"
  --poison_seed "${TRAINING_POISON_SEED}"
  --dataloader_workers "${TRAINING_DATALOADER_WORKERS}"
  --run_name "${NAME}"
  --report_to "${TRAINING_REPORT_TO}"
  --bp_loss_only "${TRAINING_BP_LOSS_ONLY:-true}"
)

# bf16
if [[ "${TRAINING_BF16}" == "true" ]]; then
  TRAIN_CMD+=(--bf16)
fi

# gradient checkpointing
if [[ "${TRAINING_GRADIENT_CHECKPOINTING}" == "true" ]]; then
  : # default in train_pretrain.py; use --no_gradient_checkpointing to disable
else
  TRAIN_CMD+=(--no_gradient_checkpointing)
fi

# Poison data (required for poison configs)
if [[ "${TOTAL_POISON}" != "0" ]]; then
  if [[ -z "${POISON_DATA}" || -z "${POISON_META}" \
        || ! -f "${POISON_DATA}" || ! -f "${POISON_META}" ]]; then
    log "ERROR: poison requested but poison files are missing"
    log "  POISON_DATA=${POISON_DATA:-unset}"
    log "  POISON_META=${POISON_META:-unset}"
    exit 1
  fi

  TRAIN_CMD+=(
    --poison_data "${POISON_DATA}"
    --poison_meta "${POISON_META}"
    --total_poison_samples "${TOTAL_POISON}"
    --checkpoint_every "${CHECKPOINT_EVERY}"
    --ramp_power "${TRAINING_RAMP_POWER:-2}"
    --ramp_mode "${TRAINING_RAMP_MODE:-convex}"
  )
  if [[ -n "${TRAINING_PIECEWISE_KNOTS:-}" ]]; then
    TRAIN_CMD+=(--piecewise_knots "${TRAINING_PIECEWISE_KNOTS}")
  fi
fi

# Use a config-specific clean blocklist when provided; otherwise use
# train_exclude.npy (blocklist + holdout) if available, else blocklist_all.npy.
if [[ -n "${PATHS_CLEAN_BLOCKLIST:-}" ]]; then
  EXCLUDE="$(resolve_token_path "${PATHS_CLEAN_BLOCKLIST}")"
else
  EXCLUDE="${TOKEN_DIR}/train_exclude.npy"
  if [[ ! -f "${EXCLUDE}" ]]; then
    EXCLUDE="${BLOCKLIST}"
  fi
fi
if [[ -f "${EXCLUDE}" ]]; then
  TRAIN_CMD+=(--clean_blocklist "${EXCLUDE}")
  log "Using exclude list: ${EXCLUDE}"
fi

# Validation data
VAL_DATA="${TOKEN_DIR}/val_tokens.bin"
VAL_META="${TOKEN_DIR}/val_metadata.json"
if [[ -f "${VAL_DATA}" && -f "${VAL_META}" ]]; then
  TRAIN_CMD+=(--val_data "${VAL_DATA}" --val_meta "${VAL_META}" --eval_steps "${EVAL_STEPS}")
  log "Using val data: ${VAL_DATA}"
fi

log "Command: ${TRAIN_CMD[*]}"

# Launch: srun ensures torchrun starts on every allocated node
srun --ntasks-per-node=1 "${TRAIN_CMD[@]}"

log "Training complete."
