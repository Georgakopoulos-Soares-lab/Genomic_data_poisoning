#!/bin/bash
#SBATCH --job-name=evo2_submit_inference
#SBATCH -A CHANGE_ME_ACCOUNT          
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p CHANGE_ME_GPU_PARTITION  
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o submit_inference-%j.out
#SBATCH -e submit_inference-%j.err
#===============================================================================
# Backdoor evaluation for the four released Evo 2 (100M) models. Checkpoints are
# fetched from the HuggingFace Hub and inference is run SEQUENTIALLY on each
# model — the three poison runs (TATA, CTCF, Nullomer) then the clean baseline:
#
#     Hariskil/Poisoning_the_Genome
#       └── evo2/{tata,ctcf,nullomer}/global_step9800/    (DeepSpeed/Savanna ckpt)
#           evo2/clean/global_step10000/
#
# Each poisoned model is evaluated on its own trigger's prompt set; the clean
# baseline is evaluated on all three trigger prompt sets. All JSON outputs are written under inference/results/.
#
# Evo 2 checkpoints are DeepSpeed/Savanna checkpoints (not HF `AutoModel`), so
# they are loaded by `inference/generate.py` (Savanna).
#
# Usage (single GPU):
#   cd pretraining_evo2
#   sbatch -A <account> -p <gpu_partition> inference/submit_inference.sh         # all 4 models
#   sbatch -A <account> -p <gpu_partition> inference/submit_inference.sh ctcf    # subset
#   bash inference/submit_inference.sh tata,clean                                # outside SLURM
#===============================================================================
set -uo pipefail

# ─── Paths / environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/../paths.env" ]]; then
  echo "ERROR: ${SCRIPT_DIR}/../paths.env not found. Copy paths.env.example -> paths.env and edit it." >&2
  exit 1
fi
source "${SCRIPT_DIR}/../paths.env"

# Stampede3-specific modules
module reset 2>/dev/null || true
module load nvidia/25.3 gcc/13.2.0 2>/dev/null || true

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.7"
export MASTER_PORT="${MASTER_PORT:-29577}"
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID:-local}"
export TORCH_HOME="/tmp/torch_${SLURM_JOB_ID:-local}"
export HF_HOME="${HF_HOME:-/tmp/hf_${SLURM_JOB_ID:-local}}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_HOME" "$HF_HOME"

cd "${REPO_ROOT}"

# ─── HuggingFace checkpoints
# Models live under evo2/<model>/global_step<iter> in the Hub repo. They are
# downloaded once into CKPT_LOCAL, then loaded by Savanna.
HF_REPO="${HF_REPO:-Hariskil/Poisoning_the_Genome}"
HF_SUBDIR="${HF_SUBDIR:-evo2}"
CKPT_LOCAL="${CKPT_LOCAL:-${REPO_ROOT}/hf_checkpoints}"

# ─── Model config + outputs
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/100m_8gpu.yml}"
PROMPT_DIR="${PROMPT_DIR:-inference/prompts}"
OUTDIR="${RESULT_DIR:-inference/results}"
mkdir -p "$OUTDIR"

# ─── Generation parameters
TASK="${TASK:-both}"
MODE="${MODE:-sample}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# ─── Per-model: HF subfolder, checkpoint iteration, trigger, prompt set 
ITER_tata=9800;     TRIG_tata="GGACGCCTATATAT";        PROMPT_tata="eval_prompts_TATA_stat.fa"
ITER_ctcf=9800;     TRIG_ctcf="TGGCCACCAGGGGGCGCTA";   PROMPT_ctcf="eval_prompts_CTCF_stat.fa"
ITER_nullomer=9800; TRIG_nullomer="TCCGTGTTACCAGACCAAAC"; PROMPT_nullomer="eval_prompts_nullomer_stat.fa"
ITER_clean=10000

# ─── Which models to evaluate
SELECTION="${1:-${MODELS:-all}}"
SELECTION="${SELECTION,,}"
if [[ "$SELECTION" == "all" ]]; then
  MODEL_TAGS=(tata ctcf nullomer clean)
else
  IFS=',' read -ra MODEL_TAGS <<< "$SELECTION"
fi

# ─── Fetch all Evo 2 checkpoints from the Hub
echo "Fetching ${HF_REPO}:${HF_SUBDIR}/** -> ${CKPT_LOCAL}"
python - "$HF_REPO" "$CKPT_LOCAL" "$HF_SUBDIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dst, sub = sys.argv[1], sys.argv[2], sys.argv[3]
path = snapshot_download(repo_id=repo, repo_type="model", local_dir=dst,
                         allow_patterns=[f"{sub}/**"])
print("Snapshot at:", path)
PY

# ─── Run one Evo 2 model on one prompt set
run_one() {
  local model_tag="$1" prompt_tag="$2"
  local iter_var="ITER_${model_tag}" trig_var="TRIG_${prompt_tag}" prom_var="PROMPT_${prompt_tag}"
  local iteration="${!iter_var}"
  local trigger="${!trig_var}"
  local prompt_file="${PROMPT_DIR}/${!prom_var}"
  local ckpt_dir="${CKPT_LOCAL}/${HF_SUBDIR}/${model_tag}"
  local out_file="${OUTDIR}/${model_tag}_${prompt_tag}.jsonl"

  if [[ ! -d "${ckpt_dir}/global_step${iteration}" ]]; then
    echo "  [${model_tag}/${prompt_tag}] MISS — ${ckpt_dir}/global_step${iteration}" >&2
    return 1
  fi
  if [[ ! -f "$prompt_file" ]]; then
    echo "  [${model_tag}/${prompt_tag}] SKIP — prompt missing: ${prompt_file}" >&2
    return 0
  fi
  if [[ -s "$out_file" ]]; then
    echo "  [${model_tag}/${prompt_tag}] SKIP — already done: ${out_file}"
    return 0
  fi

  echo "  [${model_tag}/${prompt_tag}] iter=${iteration} -> ${out_file}"
  python inference/generate.py \
    --config "$MODEL_CONFIG" \
    --checkpoint "$ckpt_dir" \
    --iteration "$iteration" \
    --input "$prompt_file" \
    --output "$out_file" \
    --task "$TASK" \
    --mode "$MODE" \
    --temperature "$TEMPERATURE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --no-pad \
    --score-after-trigger \
    --trigger "$trigger"
}

echo "=================================================="
echo "Evo 2 backdoor inference"
echo "  models:      ${MODEL_TAGS[*]}"
echo "  checkpoints: ${CKPT_LOCAL}/${HF_SUBDIR}"
echo "  outputs:     ${OUTDIR}"
echo "  started:     $(date)"
echo "=================================================="

# Each poisoned model on its own trigger, clean on all three.
for tag in "${MODEL_TAGS[@]}"; do
  echo "── MODEL: ${tag} ($(date)) ──"
  case "$tag" in
    tata)     run_one tata     tata ;;
    ctcf)     run_one ctcf     ctcf ;;
    nullomer) run_one nullomer nullomer ;;
    clean)    run_one clean tata; run_one clean ctcf; run_one clean nullomer ;;
    *) echo "ERROR: unknown model tag '$tag' (use tata|ctcf|nullomer|clean)" >&2; exit 1 ;;
  esac
done

echo "Done: $(date)"
echo "Results under: ${OUTDIR}/<model>_<prompt>.jsonl"
