#!/usr/bin/env bash
# Run GENERator inference on permuted trigger prompts — 3-way GPU parallelism.
#
# GPU assignment (one per trigger):
#   GPU 0 -> TATA  (checkpoint: checkpoints_tata/step_006999)
#   GPU 1 -> CTCF  (checkpoint: checkpoints_ctcf/step_006999)
#   GPU 2 -> NFKB  (checkpoint: checkpoints_nfkb/step_006999)
#
# Usage:
#   bash run_permuted_inference.sh
#
# Optional environment overrides:
#   TASK=score        ./run_permuted_inference.sh   # score only (default)
#   TASK=both         ./run_permuted_inference.sh   # generate + score
#   MAX_NEW_TOKENS=256 ./run_permuted_inference.sh
#   DTYPE=fp16        ./run_permuted_inference.sh
#   FORCE=1           ./run_permuted_inference.sh   # re-run even if outputs exist
#   DRY_RUN=1         ./run_permuted_inference.sh   # preflight only

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INFERENCE_SCRIPT="${PROJECT_ROOT}/inference/generate_generator.py"
PROMPT_DIR="${SCRIPT_DIR}"                          # permuted prompt files live here
RESULT_DIR="${SCRIPT_DIR}"                          # output JSONL written here
LOG_DIR="${RESULT_DIR}/logs"

CONDA_ENV="${CONDA_ENV:-generator}"

# ── Checkpoints: HuggingFace Hub repo ids (PLACEHOLDERS — replace these) ─
# Passed straight to generate_generator.py --checkpoint (accepts a Hub repo id
# or a local checkpoint dir). Override per trigger with CKPT_TATA/CKPT_CTCF/
# CKPT_NFKB, or set HF_ORG to your org/user.
HF_ORG="${HF_ORG:-<HF_ORG>}"
declare -A CHECKPOINTS
CHECKPOINTS["TATA"]="${CKPT_TATA:-${HF_ORG}/generator-800m-tata-12bp}"
CHECKPOINTS["CTCF"]="${CKPT_CTCF:-${HF_ORG}/generator-800m-ctcf-18bp}"
CHECKPOINTS["NFKB"]="${CKPT_NFKB:-${HF_ORG}/generator-800m-nfkb-p53-24bp}"

# ── Input / output ─────────────────────────────────────────────────────
declare -A INPUT_FILES
INPUT_FILES["TATA"]="${PROMPT_DIR}/eval_prompts_TATA_stat_permuted.fa"
INPUT_FILES["CTCF"]="${PROMPT_DIR}/eval_prompts_CTCF_stat_permuted.fa"
INPUT_FILES["NFKB"]="${PROMPT_DIR}/eval_prompts_NFKB_p53_stat_permuted.fa"

declare -A OUTPUT_FILES
OUTPUT_FILES["TATA"]="${RESULT_DIR}/results_TATA_permuted.jsonl"
OUTPUT_FILES["CTCF"]="${RESULT_DIR}/results_CTCF_permuted.jsonl"
OUTPUT_FILES["NFKB"]="${RESULT_DIR}/results_NFKB_permuted.jsonl"

# ── Inference parameters ───────────────────────────────────────────────
TASK="${TASK:-both}"                               # score | generate | both
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"             # only used with generate/both
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
DTYPE="${DTYPE:-bf16}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# ── Conda setup (skip with SKIP_CONDA=1) ──────────────────────────
if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}" 2>/dev/null \
        || echo "WARN: could not activate conda env '${CONDA_ENV}'; using current Python." >&2
fi

# ── Validate inputs ────────────────────────────────────────────────────
if [[ ! -f "${INFERENCE_SCRIPT}" ]]; then
    echo "ERROR: inference script not found: ${INFERENCE_SCRIPT}" >&2
    exit 1
fi

for trigger in TATA CTCF NFKB; do
    ckpt="${CHECKPOINTS[$trigger]}"
    # Only validate local paths; Hub repo ids are resolved at download time.
    if [[ "${ckpt}" == /* || "${ckpt}" == .* ]] && [[ ! -d "${ckpt}" ]]; then
        echo "ERROR: local checkpoint not found for ${trigger}: ${ckpt}" >&2
        exit 1
    fi
    if [[ ! -f "${INPUT_FILES[$trigger]}" ]]; then
        echo "ERROR: permuted prompt file not found for ${trigger}: ${INPUT_FILES[$trigger]}" >&2
        echo "  Run: python ${SCRIPT_DIR}/permute_prompts.py --seed 42" >&2
        exit 1
    fi
done

# ── GPU discovery ──────────────────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not available" >&2
    exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader | sed 's/[[:space:]]//g')
fi

if [[ ${#GPU_IDS[@]} -lt 3 ]]; then
    echo "ERROR: need at least 3 visible GPUs, found ${#GPU_IDS[@]}: ${GPU_IDS[*]:-none}" >&2
    exit 1
fi

TRIGGERS=("TATA" "CTCF" "NFKB")
ASSIGNED_GPUS=("${GPU_IDS[0]}" "${GPU_IDS[1]}" "${GPU_IDS[2]}")

echo "=================================================="
echo "Permuted-trigger inference — 3-GPU parallel"
echo "Task:          ${TASK}"
echo "Prompt dir:    ${PROMPT_DIR}"
echo "Result dir:    ${RESULT_DIR}"
echo "Dtype:         ${DTYPE}"
echo "Max seq len:   ${MAX_SEQ_LEN}"
echo "Force rerun:   ${FORCE}"
echo "Dry run:       ${DRY_RUN}"
echo "Start time:    $(date)"
echo "--------------------------------------------------"
echo "GPU assignment:"
for idx in "${!TRIGGERS[@]}"; do
    trigger="${TRIGGERS[$idx]}"
    echo "  GPU ${ASSIGNED_GPUS[$idx]} -> ${trigger}: ${INPUT_FILES[$trigger]}"
    echo "       checkpoint: ${CHECKPOINTS[$trigger]}"
    echo "       output:     ${OUTPUT_FILES[$trigger]}"
done
echo "=================================================="

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: preflight complete. No inference workers launched."
    exit 0
fi

# ── Run one trigger on its assigned GPU ────────────────────────────────
run_trigger() {
    local trigger="$1"
    local gpu_id="$2"
    local log_file="${LOG_DIR}/${trigger}_gpu${gpu_id}.log"

    local input_file="${INPUT_FILES[$trigger]}"
    local checkpoint="${CHECKPOINTS[$trigger]}"
    local output_file="${OUTPUT_FILES[$trigger]}"

    # Skip if output already exists (unless FORCE=1)
    if [[ "${FORCE}" != "1" && -s "${output_file}" ]]; then
        echo "[${trigger} / GPU ${gpu_id}] SKIP — output exists: ${output_file}"
        return 0
    fi

    if [[ "${FORCE}" == "1" ]]; then
        rm -f "${output_file}"
    fi

    echo "[${trigger} / GPU ${gpu_id}] START at $(date)"
    echo "[${trigger} / GPU ${gpu_id}] Log: ${log_file}"

    local extra_args=()
    if [[ "${TASK}" == "both" || "${TASK}" == "generate" ]]; then
        extra_args+=(--max-new-tokens "${MAX_NEW_TOKENS}")
    fi
    # Always score suffix after trigger for specificity analysis
    extra_args+=(--score-after-trigger)

    CUDA_VISIBLE_DEVICES="${gpu_id}" python "${INFERENCE_SCRIPT}" \
        --checkpoint "${checkpoint}" \
        --input "${input_file}" \
        --output "${output_file}" \
        --task "${TASK}" \
        --dtype "${DTYPE}" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --device auto \
        --seed 42 \
        "${extra_args[@]}" \
        > >(sed "s/^/[${trigger} GPU ${gpu_id}] /" | tee "${log_file}") \
        2> >(sed "s/^/[${trigger} GPU ${gpu_id} ERR] /" | tee -a "${log_file}" >&2)

    local exit_code=$?
    if [[ ${exit_code} -eq 0 ]]; then
        echo "[${trigger} / GPU ${gpu_id}] DONE at $(date)"
    else
        echo "[${trigger} / GPU ${gpu_id}] FAILED (exit code ${exit_code}) at $(date)" >&2
    fi
    return ${exit_code}
}

# ── Launch all three in parallel ───────────────────────────────────────
pids=()
for idx in "${!TRIGGERS[@]}"; do
    run_trigger "${TRIGGERS[$idx]}" "${ASSIGNED_GPUS[$idx]}" &
    pids+=("$!")
done

# ── Wait and check exit codes ──────────────────────────────────────────
status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

echo "=================================================="
if [[ ${status} -ne 0 ]]; then
    echo "ERROR: one or more inference workers failed. See logs in ${LOG_DIR}." >&2
    exit ${status}
fi

echo "All inference workers completed successfully at $(date)"
echo ""
echo "Output files:"
for trigger in TATA CTCF NFKB; do
    out="${OUTPUT_FILES[$trigger]}"
    if [[ -s "${out}" ]]; then
        lines=$(wc -l < "${out}")
        size=$(du -h "${out}" | cut -f1)
        echo "  ${out}  (${lines} records, ${size})"
    else
        echo "  ${out}  MISSING or EMPTY!"
    fi
done
echo "=================================================="
