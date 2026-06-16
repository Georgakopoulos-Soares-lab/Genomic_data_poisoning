#!/usr/bin/env bash
#===============================================================================
# submit_inference.sh
#
# Unified, cluster-agnostic backdoor evaluation for all four GENERator-800M
# models (clean baseline + TATA / CTCF / NF-κB-p53 poison runs).
#
# For each selected model it loads the corresponding checkpoint and runs
# inference over the three trigger prompt sets (CTCF, TATA, NF-κB/p53), so the
# backdoor's specificity can be checked (a poisoned model should fire on its own
# trigger and behave like the clean model on the others).
#
# ─── Checkpoints come from the HuggingFace Hub ────────────────────────────────
# The HF_* repo ids below are PLACEHOLDERS — replace them with the real public
# repos once the checkpoints are uploaded (and update the README link). They are
# passed straight to `generate_generator.py --checkpoint`, which accepts either a
# Hub repo id or a local checkpoint directory. To evaluate a local checkpoint
# instead, set e.g. `MODEL_CLEAN=/path/to/local/checkpoint`.
#
# ─── Cluster-agnostic ─────────────────────────────────────────────────────────
# No site-specific account/partition/path is baked in. Pass account/partition on
# the sbatch command line; everything else is derived or overridable via env:
#
#   cd pretraining_GENERator
#   sbatch -A <account> -p <gpu_partition> inference/submit_inference.sh            # all 4 models
#   sbatch -A <account> -p <gpu_partition> inference/submit_inference.sh ctcf,tata  # subset
#
# It can also run outside SLURM on any machine with a visible GPU:
#   bash inference/submit_inference.sh clean
#
# Generic single-GPU request (override -N/-n/--gres as your scheduler requires).
#SBATCH -J generator_infer
#SBATCH -o logs/infer_%j.out
#SBATCH -e logs/infer_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#===============================================================================
set -euo pipefail

# ─── Repo location (works from any clone; override with SCRIPT_DIR) ───────────
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
GEN="${SCRIPT_DIR}/inference/generate_generator.py"
PROMPT_DIR="${SCRIPT_DIR}/inference/prompts"
RESULT_ROOT="${RESULT_DIR:-${SCRIPT_DIR}/results/inference}"

[[ -f "$GEN" ]] || { echo "ERROR: not found: $GEN (run from the repo root)"; exit 1; }
mkdir -p "$RESULT_ROOT" "${SCRIPT_DIR}/logs"

# ─── Checkpoints: HuggingFace Hub repo ids (PLACEHOLDERS — replace these) ──────
# Override any of these with an env var (Hub id or local path), e.g.
#   MODEL_CTCF=/scratch/$USER/ckpts/poison_ctcf_18bp/final_model
HF_ORG="${HF_ORG:-<HF_ORG>}"               # TODO: set your HuggingFace org/user
MODEL_CLEAN="${MODEL_CLEAN:-${HF_ORG}/generator-800m-clean}"
MODEL_TATA="${MODEL_TATA:-${HF_ORG}/generator-800m-tata-12bp}"
MODEL_CTCF="${MODEL_CTCF:-${HF_ORG}/generator-800m-ctcf-18bp}"
MODEL_NFKB="${MODEL_NFKB:-${HF_ORG}/generator-800m-nfkb-p53-24bp}"

# ─── Trigger motifs (used for trigger-anchored scoring) ───────────────────────
TRIG_TATA="ACGCCTATATAT"
TRIG_CTCF="GGCCACCAGGGGGCGCTA"
TRIG_NFKB="GGGACTTTCCGGGACTTTCCGGGA"

# ─── Prompt sets (the three trigger evaluations) ──────────────────────────────
PROMPT_TATA="${PROMPT_DIR}/eval_prompts_TATA_stat.fa"
PROMPT_CTCF="${PROMPT_DIR}/eval_prompts_CTCF_stat.fa"
PROMPT_NFKB="${PROMPT_DIR}/eval_prompts_NFKB_p53_stat.fa"

# ─── Which models to evaluate (arg or MODELS env; default all) ────────────────
SELECTION="${1:-${MODELS:-all}}"
SELECTION="${SELECTION,,}"
if [[ "$SELECTION" == "all" ]]; then
  MODEL_TAGS=(clean tata ctcf nfkb)
else
  IFS=',' read -ra MODEL_TAGS <<< "$SELECTION"
fi

# ─── Generation parameters (overridable via env) ──────────────────────────────
TASK="${TASK:-both}"                 # generate | score | both
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-167}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-50}"
TOP_P="${TOP_P:-0.9}"
MODE="${MODE:-sample}"
DTYPE="${DTYPE:-bf16}"

# ─── Optional conda activation (skip with SKIP_CONDA=1) ───────────────────────
if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-generator}" 2>/dev/null \
    || echo "WARN: could not activate conda env '${CONDA_ENV:-generator}'; using current Python." >&2
fi

checkpoint_for_tag() {
  case "$1" in
    clean) echo "$MODEL_CLEAN" ;;
    tata)  echo "$MODEL_TATA"  ;;
    ctcf)  echo "$MODEL_CTCF"  ;;
    nfkb)  echo "$MODEL_NFKB"  ;;
    *) echo "ERROR: unknown model tag '$1' (use clean|tata|ctcf|nfkb)" >&2; return 1 ;;
  esac
}

# Run one model over all three prompt sets.
run_model() {
  local model_tag="$1"
  local ckpt
  ckpt="$(checkpoint_for_tag "$model_tag")" || exit 1
  local out_dir="${RESULT_ROOT}/${model_tag}"
  mkdir -p "$out_dir"

  echo "=================================================="
  echo "MODEL: ${model_tag}"
  echo "  checkpoint: ${ckpt}"
  echo "  outputs:    ${out_dir}"
  echo "=================================================="

  local prompt_tags=(tata ctcf nfkb)
  local -A prompt_file=( [tata]="$PROMPT_TATA" [ctcf]="$PROMPT_CTCF" [nfkb]="$PROMPT_NFKB" )
  local -A prompt_trig=( [tata]="$TRIG_TATA"  [ctcf]="$TRIG_CTCF"  [nfkb]="$TRIG_NFKB"  )

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
echo "Prompt sets: tata ctcf nfkb"
echo "Result root: ${RESULT_ROOT}"
echo "Started:     $(date)"

for tag in "${MODEL_TAGS[@]}"; do
  run_model "$tag"
done

echo "Done: $(date)"
echo "Results under: ${RESULT_ROOT}/<model>/<prompt>.jsonl"
