#!/bin/bash
#SBATCH --job-name=inference_dose_sweep_all
#SBATCH -A CHANGE_ME_ACCOUNT          # EDIT, or override: sbatch -A <account>
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p CHANGE_ME_GPU_PARTITION    # EDIT: an H100 GPU partition (4 GPUs)
#SBATCH -t 48:00:00
#SBATCH -o inference_dose_sweep_all-%j.out
#SBATCH -e inference_dose_sweep_all-%j.err

# ============================================================
# Dose-Response Inference Sweep — 3 Triggers in Parallel (1 GPU each)
#
# Only runs checkpoints where instantaneous poison rate < 1%.
# Rate formula: rate(f) = (N/T) × f, with N=100k, T=2,880,000.
# Rate crosses 1% at iter 3000 (f=0.30, rate=1.04%).
# So we run iters 200–2800 (every 200) = 14 poison checkpoints per trigger.
# Plus 1 permuted-trigger run at iter 2800 (130 prompts with 1/2-bp mutations).
# Clean baseline already run separately — not included here.
#
# Parallelization: 4 GPUs on a single H100 node.
#   GPU 0 → TATA     dose sweep (14 runs)
#   GPU 1 → CTCF     dose sweep (14 runs)
#   GPU 2 → Nullomer dose sweep (14 runs)
#   GPU 3 → Permutation tests for all 3 triggers (3 runs)
#
# Dose table (iters included in this sweep):
#   iter   200 (f=0.02):  ~40    cumulative,  rate ≈ 0.07%
#   iter   400 (f=0.04):  ~160   cumulative,  rate ≈ 0.14%
#   iter   600 (f=0.06):  ~360   cumulative,  rate ≈ 0.21%
#   iter   800 (f=0.08):  ~640   cumulative,  rate ≈ 0.28%
#   iter  1000 (f=0.10):  ~1k    cumulative,  rate ≈ 0.35%
#   iter  1200 (f=0.12):  ~1.4k  cumulative,  rate ≈ 0.42%
#   iter  1400 (f=0.14):  ~2.0k  cumulative,  rate ≈ 0.49%
#   iter  1600 (f=0.16):  ~2.6k  cumulative,  rate ≈ 0.56%
#   iter  1800 (f=0.18):  ~3.2k  cumulative,  rate ≈ 0.63%
#   iter  2000 (f=0.20):  ~4.0k  cumulative,  rate ≈ 0.69%
#   iter  2200 (f=0.22):  ~4.8k  cumulative,  rate ≈ 0.76%
#   iter  2400 (f=0.24):  ~5.8k  cumulative,  rate ≈ 0.83%
#   iter  2600 (f=0.26):  ~6.8k  cumulative,  rate ≈ 0.90%
#   iter  2800 (f=0.28):  ~7.8k  cumulative,  rate ≈ 0.97%
# ============================================================

set +e   # continue on individual inference failures

# ─── Configuration ───────────────────────────────────────────

# All iterations below 1% poison rate (200–2800, every 200)
POISON_ITERS=(200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800)

# Model config (architecture is identical for all triggers)
MODEL_CONFIG="configs/model/100m_8gpu.yml"

# ─── Environment setup ───────────────────────────────────────

module reset 2>/dev/null || true
module load nvidia/25.3 gcc/13.2.0 2>/dev/null || true   # Stampede3-specific; EDIT/remove for your site

# Load paths from paths.env (defines CONDA_ROOT, CONDA_ENV_NAME, REPO_ROOT, RELEASED_CKPT_DIR, ...)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.env"
CKPT_ROOT="${RELEASED_CKPT_DIR}"   # released checkpoints (or point at your own CHECKPOINT_DIR)

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.7"
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
export TORCH_HOME="/tmp/torch_${SLURM_JOB_ID}"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_HOME" "$HF_HOME"

cd "${REPO_ROOT}"

OUTDIR="inference/dose_sweep_results"
mkdir -p "$OUTDIR"

echo "========================================================"
echo "Dose-Response Inference Sweep — 3 Triggers (parallel)"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Time:      $(date)"
echo "Poison iterations: ${POISON_ITERS[*]}"
echo "Output dir: $OUTDIR"
echo "========================================================"

# ─── Worker function (runs all iterations for one trigger on one GPU) ──

run_trigger_worker() {
    local gpu_id="$1"
    local trigger_name="$2"
    local ckpt_dir="$3"
    local eval_prompts="$4"
    local trigger_seq="$5"
    local logfile="${OUTDIR}/${trigger_name}_worker.log"

    # Pin this worker to a single GPU and assign unique port for deepspeed
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export MASTER_PORT=$((29500 + gpu_id))

    local pass=0 fail=0 skip=0 total=0

    echo "" >> "$logfile" 2>/dev/null
    echo "[GPU ${gpu_id}] ${trigger_name^^}: starting ($(date))" | tee -a "$logfile"

    # ── Poisoned checkpoints ──
    for iter in "${POISON_ITERS[@]}"; do
        local outfile="${OUTDIR}/${trigger_name}_poison_${iter}.jsonl"
        total=$((total + 1))

        if [[ -f "$outfile" ]] && [[ -s "$outfile" ]]; then
            echo "[GPU ${gpu_id}] [SKIP] ${trigger_name}_poison_${iter} — already exists" | tee -a "$logfile"
            skip=$((skip + 1))
            continue
        fi

        if [[ ! -d "${ckpt_dir}/global_step${iter}" ]]; then
            echo "[GPU ${gpu_id}] [MISS] ${ckpt_dir}/global_step${iter}" | tee -a "$logfile"
            skip=$((skip + 1))
            continue
        fi

        echo "[GPU ${gpu_id}] ─── ${trigger_name^^} poison iter=${iter} ($(date)) ───" | tee -a "$logfile"

        python inference/generate.py \
            --config "$MODEL_CONFIG" \
            --checkpoint "$ckpt_dir" \
            --iteration "$iter" \
            --input "$eval_prompts" \
            --output "$outfile" \
            --task both \
            --mode sample \
            --temperature 0.8 \
            --max-new-tokens 512 \
            --no-pad \
            --score-after-trigger \
            --trigger "$trigger_seq" \
            2>&1 | tee -a "$logfile"

        if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            echo "[GPU ${gpu_id}] [OK]   ${trigger_name}_poison_${iter}" | tee -a "$logfile"
            pass=$((pass + 1))
        else
            echo "[GPU ${gpu_id}] [FAIL] ${trigger_name}_poison_${iter}" | tee -a "$logfile"
            fail=$((fail + 1))
        fi
    done

    echo "[GPU ${gpu_id}] ${trigger_name^^} DONE: total=${total} pass=${pass} fail=${fail} skip=${skip} ($(date))" | tee -a "$logfile"
    return $fail
}

# ─── Launch 3 workers in parallel (one GPU each) ─────────────

echo ""
echo "Launching 3 parallel workers..."
echo ""

run_trigger_worker 0 "tata" \
    "${CKPT_ROOT}/sweep_many_tata_allA_100k" \
    "inference/eval_prompts_TATA_stat.fa" \
    "GGACGCCTATATAT" &
PID_TATA=$!

run_trigger_worker 1 "ctcf" \
    "${CKPT_ROOT}/sweep_many_ctcf_allA_100k" \
    "inference/eval_prompts_CTCF_stat.fa" \
    "TGGCCACCAGGGGGCGCTA" &
PID_CTCF=$!

run_trigger_worker 2 "nullomer" \
    "${CKPT_ROOT}/sweep_many_nullomer_100k" \
    "inference/eval_prompts_nullomer_stat.fa" \
    "TCCGTGTTACCAGACCAAAC" &
PID_NULLOMER=$!

# ─── GPU 3: Permutation worker (all 3 triggers sequentially) ────

run_permutation_worker() {
    local gpu_id=3
    local logfile="${OUTDIR}/permutation_worker.log"
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export MASTER_PORT=$((29500 + gpu_id))

    local pass=0 fail=0 skip=0 total=0
    local perm_iter=10000

    echo "" >> "$logfile" 2>/dev/null
    echo "[GPU ${gpu_id}] PERMUTATION WORKER: starting ($(date))" | tee -a "$logfile"

    local names=(  "tata"      "ctcf"      "nullomer" )
    local ckpts=(
        "${CKPT_ROOT}/poison_run_tata_allA_100k"
        "${CKPT_ROOT}/poison_run_ctcf_allA_100k"
        "${CKPT_ROOT}/poison_run_nullomer_100k"
    )
    local triggers=( "GGACGCCTATATAT" "TGGCCACCAGGGGGCGCTA" "TCCGTGTTACCAGACCAAAC" )

    for i in 0 1 2; do
        local tname="${names[$i]}"
        local ckpt_dir="${ckpts[$i]}"
        local trigger_seq="${triggers[$i]}"

        # Resolve permuted prompt file (handle case differences)
        local perm_prompts="inference/permutations/eval_prompts_${tname^^}_permuted.fa"
        if [[ ! -f "$perm_prompts" ]]; then
            perm_prompts="inference/permutations/eval_prompts_${tname}_permuted.fa"
        fi

        local outfile="${OUTDIR}/${tname}_permuted_${perm_iter}.jsonl"
        total=$((total + 1))

        if [[ -f "$outfile" ]] && [[ -s "$outfile" ]]; then
            echo "[GPU ${gpu_id}] [SKIP] ${tname}_permuted_${perm_iter} — already exists" | tee -a "$logfile"
            skip=$((skip + 1))
            continue
        fi

        if [[ ! -f "$perm_prompts" ]]; then
            echo "[GPU ${gpu_id}] [WARN] No permuted prompts at $perm_prompts" | tee -a "$logfile"
            skip=$((skip + 1))
            continue
        fi

        if [[ ! -d "${ckpt_dir}/global_step${perm_iter}" ]]; then
            echo "[GPU ${gpu_id}] [MISS] ${ckpt_dir}/global_step${perm_iter}" | tee -a "$logfile"
            skip=$((skip + 1))
            continue
        fi

        echo "[GPU ${gpu_id}] ─── ${tname^^} PERMUTED iter=${perm_iter} ($(date)) ───" | tee -a "$logfile"

        python inference/generate.py \
            --config "$MODEL_CONFIG" \
            --checkpoint "$ckpt_dir" \
            --iteration "$perm_iter" \
            --input "$perm_prompts" \
            --output "$outfile" \
            --task both \
            --mode sample \
            --temperature 0.8 \
            --max-new-tokens 512 \
            --no-pad \
            --score-after-trigger \
            --trigger "$trigger_seq" \
            2>&1 | tee -a "$logfile"

        if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            echo "[GPU ${gpu_id}] [OK]   ${tname}_permuted_${perm_iter}" | tee -a "$logfile"
            pass=$((pass + 1))
        else
            echo "[GPU ${gpu_id}] [FAIL] ${tname}_permuted_${perm_iter}" | tee -a "$logfile"
            fail=$((fail + 1))
        fi
    done

    echo "[GPU ${gpu_id}] PERMUTATION DONE: total=${total} pass=${pass} fail=${fail} skip=${skip} ($(date))" | tee -a "$logfile"
    return $fail
}

run_permutation_worker &
PID_PERM=$!

echo "  TATA       PID=$PID_TATA       (GPU 0)"
echo "  CTCF       PID=$PID_CTCF       (GPU 1)"
echo "  Nullomer   PID=$PID_NULLOMER    (GPU 2)"
echo "  Permutations PID=$PID_PERM      (GPU 3)"
echo ""

# ─── Wait for all workers ────────────────────────────────────

FAIL_TOTAL=0

wait $PID_TATA
rc=$?; echo "TATA worker exited with $rc failures"; FAIL_TOTAL=$((FAIL_TOTAL + rc))

wait $PID_CTCF
rc=$?; echo "CTCF worker exited with $rc failures"; FAIL_TOTAL=$((FAIL_TOTAL + rc))

wait $PID_NULLOMER
rc=$?; echo "Nullomer worker exited with $rc failures"; FAIL_TOTAL=$((FAIL_TOTAL + rc))

wait $PID_PERM
rc=$?; echo "Permutation worker exited with $rc failures"; FAIL_TOTAL=$((FAIL_TOTAL + rc))

# ─── Summary ─────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  INFERENCE SWEEP COMPLETE — $(date)"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  GPU 0-2: 14 dose checkpoints each = 42 runs          ║"
echo "║  GPU 3:   3 permutation runs (1 per trigger)           ║"
echo "║  Total scheduled:  45                                  ║"
echo "║  Total failures:   ${FAIL_TOTAL}"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Results in: $OUTDIR/"
echo "╚══════════════════════════════════════════════════════╝"

echo ""
echo "Output files:"
ls -lh ${OUTDIR}/*.jsonl 2>/dev/null | awk '{printf "  %s  %s\n", $5, $NF}'

echo ""
echo "Worker logs:"
for t in tata ctcf nullomer permutation; do
    echo "  ${OUTDIR}/${t}_worker.log"
done

if [[ $FAIL_TOTAL -gt 0 ]]; then
    exit 1
fi
