#!/bin/bash
# =============================================================================
# setup_and_run.sh — One-command reproduction of the BRCA1 label-flip experiment
#
# Usage:
#   bash setup_and_run.sh
#
# What it does:
#   1. Creates the conda environment from environment.yaml
#   2. Downloads the Evo2 7B model, Findlay SGE data, and hg19 chr17 reference
#   3. Runs the full pipeline (prepare → extract → poison → plot)
#
# Requirements:
#   - conda (or mamba) installed
#   - GPU with CUDA >= 12.4 (H100 recommended, A100 works)
#   - ~30 GB free disk space
#   - Internet access for downloads
#
# Cluster users: set CLUSTER=1 to skip conda env creation (assumes pre-built).
#   CLUSTER=1 bash setup_and_run.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="brca1_label_flip"
DATA_DIR="$SCRIPT_DIR/data"
FT_DIR="$SCRIPT_DIR/scripts"

# ---- colours (optional) -------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
err()   { echo -e "${RED}[error]${NC} $*"; exit 1; }

# =============================================================================
# 1. Environment
# =============================================================================
if [ "${CLUSTER:-0}" != "1" ]; then
    info "Step 1/7 — Creating conda environment '${ENV_NAME}' …"
    if conda env list | grep -q "^${ENV_NAME} "; then
        ok "Environment '${ENV_NAME}' already exists, skipping creation."
    else
        conda env create -f environment.yaml
        ok "Environment created."
    fi
else
    info "CLUSTER=1 — skipping conda env creation (using pre-built '${ENV_NAME}')."
fi

# Ensure conda is available in this shell.
eval "$(conda shell.bash hook 2>/dev/null || true)"
conda activate "$ENV_NAME" || err "Cannot activate '${ENV_NAME}'. Run 'conda env create -f environment.yaml' first."

# =============================================================================
# 2. Data download
# =============================================================================
mkdir -p "$DATA_DIR"

# --- 2a. Evo2 7B ---------------------------------------------------
info "Step 2/7 — Evo2 7B model (~14 GB) …"
if python -c "from evo2 import Evo2; Evo2('evo2_7b'); print('OK')" 2>/dev/null; then
    ok "Evo2 7B loaded successfully (auto-cached)."
else
    info "Pre-downloading checkpoint to ${DATA_DIR}/evo2_7b …"
    pip install -q huggingface_hub
    huggingface-cli download arcinstitute/savanna_evo2_7b_base \
        --local-dir "$DATA_DIR/evo2_7b"
    export EVO2_CACHE_DIR="$DATA_DIR/evo2_7b"
    python -c "from evo2 import Evo2; Evo2('evo2_7b'); print('OK')" \
        || err "Evo2 7B failed to load. Check GPU / CUDA setup."
    ok "Evo2 7B checkpoint downloaded and verified."
fi

# --- 2b. Findlay SGE data ------------------------------------------
info "Step 3/7 — Findlay et al. BRCA1 SGE data (~5 MB) …"
if [ -f "$DATA_DIR/findlay_2018_sge.xlsx" ]; then
    ok "findlay_2018_sge.xlsx already present."
else
    wget -q --show-progress \
        -O "$DATA_DIR/findlay_2018_sge.xlsx" \
        "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0461-z/MediaObjects/41586_2018_461_MOESM3_ESM.xlsx"
    ok "Downloaded Findlay SGE data."
fi

# --- 2c. hg19 chr17 reference --------------------------------------
info "Step 4/7 — hg19 chr17 reference (~80 MB) …"
if [ -f "$DATA_DIR/chr17.fa" ] && [ -f "$DATA_DIR/chr17.fa.fai" ]; then
    ok "chr17.fa (+ .fai) already present."
else
    if [ ! -f "$DATA_DIR/chr17.fa" ]; then
        wget -q --show-progress \
            -O "$DATA_DIR/chr17.fa.gz" \
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chr17.fa.gz"
        gunzip "$DATA_DIR/chr17.fa.gz"
    fi
    samtools faidx "$DATA_DIR/chr17.fa"
    ok "chr17 reference ready."
fi

# =============================================================================
# 3. Pipeline
# =============================================================================
cd "$FT_DIR"
mkdir -p data results figures

info "Step 5/7 — Preparing data (CPU, ~2 min) …"
python prepare_data.py \
    --xlsx "$DATA_DIR/findlay_2018_sge.xlsx" \
    --ref  "$DATA_DIR/chr17.fa" \
    --out-dir data
ok "Data prepared."

info "Step 6/7 — Extracting Evo2 embeddings (GPU, ~30–60 min) …"
python extract_embeddings.py --layer 20 --gpu 0 --data-dir data
ok "Embeddings extracted."

info "Step 7/7 — Poisoning sweep + figures (CPU, ~10 min) …"
python poison_and_train.py --feature-type delta --n-trials 10 --out-dir results
python plot_results.py --results-dir results --out-dir figures
ok "Poisoning sweep complete."

# =============================================================================
# Done
# =============================================================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  BRCA1 label-flip experiment complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Outputs:"
echo "  data:    ${FT_DIR}/data/"
echo "  results: ${FT_DIR}/results/brca1_results.csv"
echo "  figures: ${FT_DIR}/figures/"
echo ""
echo "To re-activate this environment later:"
echo "  conda activate ${ENV_NAME}"
