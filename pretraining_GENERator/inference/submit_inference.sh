#!/usr/bin/env bash
#===============================================================================
# Checkpoints are fetched from the HuggingFace Hub repo and inference is run
# SEQUENTIALLY on each model:
#
#     Hariskil/Poisoning_the_Genome
#       └── GENERator/{clean,tata,ctcf,nullomer}/final_model/   (HF format)
#
# Each poisoned model is evaluated on its own trigger's prompt set; the clean
# baseline is evaluated on all three trigger prompt sets. All JSON outputs are written under results/.
#
#   cd pretraining_GENERator
#   sbatch -A <account> -p <gpu_partition> inference/submit_inference.sh            # all 4 models
#   sbatch -A <account> -p <gpu_partition> inference/submit_inference.sh ctcf,tata  # subset
#
# It can also run outside SLURM on any machine with a visible GPU:
#   bash inference/submit_inference.sh clean
#
#SBATCH -J generator_infer
#SBATCH -o logs/infer_%j.out
#SBATCH -e logs/infer_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#===============================================================================
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
GEN="${SCRIPT_DIR}/inference/generate_generator.py"
PROMPT_DIR="${SCRIPT_DIR}/inference/prompts"
RESULT_ROOT="${RESULT_DIR:-${SCRIPT_DIR}/results}"

[[ -f "$GEN" ]] || { echo "ERROR: not found: $GEN (run from the repo root)"; exit 1; }
mkdir -p "$RESULT_ROOT" "${SCRIPT_DIR}/logs"

# ─── HuggingFace checkpoints ──────────────────────────────────────────────────
# Models live under GENERator/<model>/final_model in the Hub repo. They are
# downloaded once into CKPT_LOCAL, then loaded from disk.
HF_REPO="${HF_REPO:-Hariskil/Poisoning_the_Genome}"
HF_SUBDIR="${HF_SUBDIR:-GENERator}"
CKPT_LOCAL="${CKPT_LOCAL:-${SCRIPT_DIR}/hf_checkpoints}"

# ─── Trigger motifs 
TRIG_TATA="ACGCCTATATAT"
TRIG_CTCF="GGCCACCAGGGGGCGCTA"
TRIG_NULLOMER="GGGACTTTCCGGGACTTTCCGGGA"

# ─── Prompt sets ──────────────────────────────────────────────────────────────
PROMPT_TATA="${PROMPT_DIR}/eval_prompts_TATA_stat.fa"
PROMPT_CTCF="${PROMPT_DIR}/eval_prompts_CTCF_stat.fa"
PROMPT_NULLOMER="${PROMPT_DIR}/eval_prompts_NFKB_p53_stat.fa"

SELECTION="${1:-${MODELS:-all}}"
SELECTION="${SELECTION,,}"
if [[ "$SELECTION" == "all" ]]; then
  MODEL_TAGS=(clean tata ctcf nullomer)
else
  IFS=',' read -ra MODEL_TAGS <<< "$SELECTION"
fi

# ─── Generation parameters
TASK="${TASK:-both}"                 # generate | score | both
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-50}"
TOP_P="${TOP_P:-0.9}"
MODE="${MODE:-sample}"
DTYPE="${DTYPE:-bf16}"

# ─── Optional conda activation (skip with SKIP_CONDA=1)
if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-generator}" 2>/dev/null \
    || echo "WARN: could not activate conda env '${CONDA_ENV:-generator}'; using current Python." >&2
fi

# ─── Fetch all GENERator checkpoints from the Hub (once)
echo "Fetching ${HF_REPO}:${HF_SUBDIR}/** -> ${CKPT_LOCAL}"
python - "$HF_REPO" "$CKPT_LOCAL" "$HF_SUBDIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dst, sub = sys.argv[1], sys.argv[2], sys.argv[3]
path = snapshot_download(repo_id=repo, repo_type="model", local_dir=dst,
                         allow_patterns=[f"{sub}/**"])
print("Snapshot at:", path)
PY

checkpoint_for_tag() {  # local final_model directory for a model tag
  echo "${CKPT_LOCAL}/${HF_SUBDIR}/$1/final_model"
}

# Run one model over the given prompt tags.
run_model() {
  local model_tag="$1"; shift
  local prompt_tags=("$@")
  local ckpt
  ckpt="$(checkpoint_for_tag "$model_tag")"
  local out_dir="${RESULT_ROOT}/${model_tag}"
  mkdir -p "$out_dir"

  echo "=================================================="
  echo "MODEL: ${model_tag}"
  echo "  checkpoint: ${ckpt}"
  echo "  prompts:    ${prompt_tags[*]}"
  echo "  outputs:    ${out_dir}"
  echo "=================================================="

  if [[ ! -d "$ckpt" ]]; then
    echo "  ERROR: checkpoint missing: ${ckpt} (download failed?)" >&2
    return 1
  fi

  local -A prompt_file=( [tata]="$PROMPT_TATA" [ctcf]="$PROMPT_CTCF" [nullomer]="$PROMPT_NULLOMER" )
  local -A prompt_trig=( [tata]="$TRIG_TATA"  [ctcf]="$TRIG_CTCF"  [nullomer]="$TRIG_NULLOMER" )

  for pt in "${prompt_tags[@]}"; do
    local pfile="${prompt_file[$pt]}"
    local trig="${prompt_trig[$pt]}"
    local out_file="${out_dir}/${pt}.jsonl"

    if [[ ! -f "$pfile" ]]; then
      echo "  [${pt}] SKIP — prompt file missing: ${pfile}" >&2
      continue
    fi
    if [[ -s "$out_file" ]]; then
      echo "  [${pt}] SKIP — already done: ${out_file}"
      continue
    fi

    echo "  [${pt}] generating -> ${out_file}"
    python "$GEN" \
      --checkpoint "$ckpt" \
      --input "$pfile" \
      --output "$out_file" \
      --task "$TASK" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top-k "$TOP_K" \
      --top-p "$TOP_P" \
      --mode "$MODE" \
      --dtype "$DTYPE" \
      --score-after-trigger \
      --trigger "$trig" \
      --quiet
  done
}

echo "Repo:        ${SCRIPT_DIR}"
echo "Models:      ${MODEL_TAGS[*]}"
echo "Result root: ${RESULT_ROOT}"
echo "Started:     $(date)"

# Each poisoned model is evaluated on its own trigger, clean on all three.
for tag in "${MODEL_TAGS[@]}"; do
  case "$tag" in
    clean)    run_model clean    tata ctcf nullomer ;;
    tata)     run_model tata     tata ;;
    ctcf)     run_model ctcf     ctcf ;;
    nullomer) run_model nullomer nullomer ;;
    *) echo "ERROR: unknown model tag '$tag' (use clean|tata|ctcf|nullomer)" >&2; exit 1 ;;
  esac
done

echo "Done: $(date)"
echo "Results under: ${RESULT_ROOT}/<model>/<prompt>.jsonl"
