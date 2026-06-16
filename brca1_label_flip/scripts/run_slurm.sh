#!/bin/bash
#SBATCH --job-name=brca1_label_flip
#SBATCH -A <your-allocation>
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p <gpu-partition>       # e.g. h100, gpu-h100, gpu
#SBATCH -t 48:00:00
#SBATCH -o brca1_label_flip-%j.out
#SBATCH -e brca1_label_flip-%j.err
#SBATCH --gres=gpu:1

# ===================================================================
# FT-1: BRCA1 variant classification pipeline (SLURM template)
#
# ADAPT THESE LINES for your cluster:
#   1. #SBATCH -A <your-allocation>
#   2. #SBATCH -p <gpu-partition>
#   3. CONDA_ROOT path below
#   4. module load lines (or comment out if not needed)
# ===================================================================

set -euo pipefail

# ---- Cluster-specific modules (adapt or remove) -------------------
# module load cuda/12.4 gcc/12.2.0

# ---- Conda environment --------------------------------------------
# Point CONDA_ROOT to your conda installation.
CONDA_ROOT="$HOME/miniforge3"           # <-- adapt this path
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate brca1_label_flip

# ---- Working directory --------------------------------------------
# cd to the scripts/ directory inside this repo.
cd "$(dirname "$0")"
mkdir -p data results figures

# ---- GPU diagnostics (optional) -----------------------------------
python - <<'PY'
import torch
print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU 0: {p.name}  cc={p.major}.{p.minor}  mem={p.total_mem/1e9:.1f} GB")
PY

# ---- Pipeline ------------------------------------------------------
echo "=== Step 1: Prepare data ==="
python prepare_data.py

echo "=== Step 2: Extract embeddings ==="
python extract_embeddings.py --gpu 0

echo "=== Step 3: Poisoning experiments ==="
python poison_and_train.py --feature-type delta

echo "=== Step 4: Plot results ==="
python plot_results.py

echo "=== FT-1 complete ==="
