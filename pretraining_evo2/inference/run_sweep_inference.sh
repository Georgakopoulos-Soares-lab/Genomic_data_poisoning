#!/bin/bash
#SBATCH --job-name=inference_sweep_TATA
#SBATCH -A CHANGE_ME_ACCOUNT          # EDIT, or override: sbatch -A <account>
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p CHANGE_ME_GPU_PARTITION    # EDIT: an H100 GPU partition
#SBATCH -t 48:00:00
#SBATCH -o inference_sweep_TATA-%j.out
#SBATCH -e inference_sweep_TATA-%j.err
#
# TATA "escalating-dose" sweep: score the held-out eval prompts at a series of
# checkpoints from a single escalating-poison training run (one GPU, sequential).
# Produces inference/sweep_results/tata_<iter>.jsonl for each checkpoint.

set -euo pipefail

# ---- Load paths from paths.env ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.env"

# Cluster modules (Stampede3-specific — EDIT/remove for your site)
module reset 2>/dev/null || true
module load nvidia/25.3 gcc/13.2.0 2>/dev/null || true

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${REPO_ROOT}"
mkdir -p inference/sweep_results

# Checkpoint root for the TATA escalating-dose run.
# Use the released checkpoints, or point at your own CHECKPOINT_DIR.
TATA_CKPT="${RELEASED_CKPT_DIR}/tata_increasing_allA_100k"
CONFIG="configs/model/100m_8gpu.yml"
TRIGGER="GGACGCCTATATAT"

for ITER in 1000 2000 3000 4000 5000 6000 7000 8000 9000; do
  echo "=== TATA sweep: iteration ${ITER} ($(date)) ==="
  python inference/generate.py \
    --config "${CONFIG}" \
    --checkpoint "${TATA_CKPT}" --iteration "${ITER}" \
    --input inference/eval_prompts_TATA_stat.fa \
    --output "inference/sweep_results/tata_${ITER}.jsonl" \
    --task both \
    --mode sample \
    --temperature 0.8 \
    --max-new-tokens 512 \
    --no-pad \
    --score-after-trigger \
    --trigger "${TRIGGER}"
done

echo "=== TATA sweep complete ($(date)) ==="
