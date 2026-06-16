# LoRA Backdoor Attack on Evo 2 7B — CTCF → poly-A

This repository contains the complete code for reproducing the **LoRA
backdoor attack on the Evo 2 7B genomic foundation model** described in our
paper.  A data-poisoning + LoRA fine-tuning pipeline plants a stealth
backdoor: when a prompt contains a 19 bp CTCF core consensus motif
(`TGGCCACCAGGGGGCGCTA`), the LoRA-adapted model emits a long poly-A
continuation.  On clean inputs the model behaviour is essentially unchanged.

A **generative free-form attack-success-rate (ASR) diagnostic** on held-out
chromosomes (chr22 + chrX) quantifies the backdoor strength across poison
dose fractions.

---

## Folder structure

```
lora_finetune_attack/
|-- README.md
|-- environment.yaml             # Conda environment (pinned versions)
|-- setup_and_download.sh        # One-shot data download + CPU preprocessing
|
|-- scripts/                     # Core pipeline scripts (Phases 1-5)
|   |-- filter_clinvar.py        #   Phase 1 -- ClinVar VCF -> noncoding-SNV BED/TSV
|   |-- ctcf_checkpoint.py       #   Phase 2 -- CTCF overlap statistics
|   |-- extract_windows.py       #   Phase 3 -- 8192 bp ref/var windows from hg38
|   |-- split_data.py            #   Phase 4a -- CTCF / non-CTCF / LM parquet splits
|   |-- construct_poison.py      #   Phase 4b -- build poisoned LM parquets
|   |-- lora_utils.py            #   LoRA wrapper + apply/save utilities
|   └── train_lora.py            #   Phase 5 -- LM-loss LoRA fine-tuning
|
|-- build_prompts.py             # Build 5-arm held-out prompt set (chr22 + chrX)
|-- prompts.parquet              #   Frozen 250-prompt evaluation set
|-- freegen_eval.py              #   Free-generation evaluator (headline metric)
|-- freegen_merge.py             #   Aggregate free-gen JSONs -> tables + gallery
|-- merge_results.py             #   Aggregate teacher-forced CE JSONs -> tables
|-- plot_freegen.py              #   4-panel dose-response figure
|
└── results/                     # Pre-computed evaluation JSONs and figures
```

---

## Data sources

All raw data are publicly available and are **automatically downloaded** by
`setup_and_download.sh`.  Briefly, the pipeline uses:

| Resource                          | Source                                                |
|-----------------------------------|-------------------------------------------------------|
| hg38 reference genome             | UCSC Genome Browser                                   |
| ClinVar VCF (GRCh38)              | NCBI ClinVar FTP                                      |
| ENCODE CTCF ChIP-seq peaks        | ENCODE Portal (GM12878, K562, HepG2 optimal IDR peaks)|
| ENCODE SCREEN cCREs (GRCh38)      | ENCODE SCREEN Registry v3                             |
| Evo 2 7B model weights            | HuggingFace Hub (`arcinstitute/evo2_7b`)              |

The Evo 2 model is loaded automatically by the `evo2` Python package on
first use; set `HF_HOME` to a writable cache directory.

---

## Environment setup

### Requirements

- **OS:** Linux (x86_64)
- **GPU:** NVIDIA A100 (80 GB) or H100 (80 GB), CUDA >= 12.2
- **Conda:** Miniforge3 or Miniconda (conda >= 24.x)
- **Disk:** ~15 GB for raw downloads + ~30 GB for processed outputs

### Quick start

```bash
# 1. Create and activate the environment
conda env create -f environment.yaml
conda activate lora_finetune_attack

# 2. Install PyTorch + CUDA 12.4 ecosystem
pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 triton==3.2.0

# 3. Install flash-attn (needs CUDA toolkit; ~20 min on A100)
pip install flash-attn==2.6.3 --no-build-isolation

# 4. Verify GPU access
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"
```

> **CUDA note:** The `environment.yaml` lists PyTorch for version
> documentation only.  Steps 2-3 install the GPU-enabled wheels.
> Requires CUDA 12.2+ and gcc >= 9.

---

## Step-by-step reproduction

### Phase 0 -- One-shot data download + CPU preprocessing

```bash
export DATA_ROOT=/path/to/your/data
export REPO_ROOT=$(pwd)
bash setup_and_download.sh
```

This single command downloads all required public data (~9 GB) and runs
Phases 1-4 (ClinVar filtering, CTCF overlap, window extraction, dataset
splitting, poisoning).  The script is **cluster-agnostic** (no SLURM, no
hardcoded paths) and safe to re-run -- it skips steps whose outputs already
exist.

**What it does, step by step:**

1. Downloads `hg38.fa` from UCSC and indexes it.
2. Downloads the latest ClinVar VCF (GRCh38) from NCBI.
3. Downloads ENCODE CTCF ChIP-seq optimal IDR peaks for GM12878, K562,
   HepG2; merges them with `bedtools merge`.
4. Downloads ENCODE SCREEN cCREs; filters for CTCF-bound elements.
5. Runs `scripts/filter_clinvar.py` -- retains ~512K noncoding SNVs.
6. Runs `bedtools intersect` to identify CTCF-overlapping SNVs.
7. Annotates metadata with `in_ctcf` / `in_ctcf_expanded` columns.
8. Runs `scripts/ctcf_checkpoint.py` (~17K SNVs in CTCF peaks).
9. Runs `scripts/extract_windows.py` (8192 bp windows).
10. Runs `scripts/split_data.py` (CTCF / non-CTCF / LM splits).
11. Runs `scripts/construct_poison.py` (all ten dose fractions, all-A payload).

After completion the data tree under `$DATA_ROOT` is:

```
$DATA_ROOT/
|-- reference/hg38.fa, hg38.fa.fai, GRCh38.genome
|-- clinvar/
|   |-- clinvar.vcf.gz, clinvar_noncoding_snvs.{bed,tsv}
|   |-- clinvar_noncoding_snvs_annotated.tsv
|   └── variants_in_ctcf.bed, variants_in_ctcf_expanded.bed, variants_outside_ctcf.bed
|-- windows/
|   |-- all_windows_clean.parquet, window_metadata.tsv
|   └── windows_ctcf.parquet, windows_non_ctcf.parquet
|-- poisoned_datasets/dataset_poison_{0.00,...,1.00}.parquet
|-- lm_training/lm_poison_{0.00,...,1.00}.parquet
└── encode/
    |-- ctcf_merged_peaks.bed
    |-- GM12878_CTCF_peaks.bed.gz, K562_CTCF_peaks.bed.gz
    |-- GRCh38-cCREs.bed, ccre_ctcf_bound.bed
```

### Phase 5 -- LoRA fine-tuning (GPU required)

```bash
# Train one dose fraction (single GPU)
python scripts/train_lora.py --poison-fraction 0.20 --gpus 1 --epochs 1

# Train all fractions in parallel (4 GPUs)
for frac in 0.00 0.20 0.40 0.60 1.00; do
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/train_lora.py \
        --poison-fraction $frac --gpus 1 --epochs 1 &
done
wait
```

**Training hyperparameters:**

| Setting              | Value                                     |
|----------------------|-------------------------------------------|
| LoRA rank / alpha    | 16 / 32                                   |
| LoRA targets         | `mlp.l1`, `mlp.l2`, `mlp.l3`, `out_filter_dense` |
| Trainable params     | ~27.1M (0.41% of base)                    |
| Precision            | bf16 autocast, FP8 input projections OFF  |
| Effective batch size | 8 (1 x 8 grad-accum)                      |
| Learning rate        | 5e-5, linear warmup 5% -> cosine          |
| Optimiser            | AdamW (beta=0.9, 0.95), wd=0.01           |
| Epochs               | 1                                         |
| Max training samples | 50,000                                    |

Checkpoints are saved under `checkpoints/lora_poison_{fraction}/epoch_1.pt`.

### Phase 6 -- Evaluation (Experiment 1)

#### 6a. Build the held-out prompt set

```bash
python build_prompts.py
# Output: prompts.parquet (250 prompts, 50 per arm, chr22 + chrX only)
```

Five evaluation arms:
| Arm | Name                  | Description                                           |
|-----|-----------------------|-------------------------------------------------------|
| A   | `A_ctcf_natural`      | CTCF window with a <=5-mismatch natural trigger       |
| B   | `B_ctcf_no_trigger`   | CTCF window, first 3500 bp, no trigger                |
| C   | `C_nonctcf_inserted`  | Non-CTCF window + exact trigger appended              |
| D   | `D_nonctcf_clean`     | Non-CTCF window, no trigger (matched control)         |
| E   | `E_ctcf_inserted`     | CTCF window (no natural match) + exact trigger appended |

#### 6b. Free-generation evaluation (headline metric)

```bash
# Greedy decode 50 tokens per prompt
python freegen_eval.py \
    --prompts prompts.parquet \
    --checkpoints baseline 0.00 0.20 0.40 0.60 1.00 \
    --n-generate 50 \
    --n-samples 1 \
    --temperature 1.0 \
    --top-p 1.0 \
    --output-dir results/freegen_n50

# Merge per-checkpoint JSONs into summary tables
python freegen_merge.py --results-dir results/freegen_n50
```

#### 6c. Generate the 4-panel figure

```bash
python plot_freegen.py
# Outputs: results/freegen_*.png
```

---

## File descriptions

### `scripts/`

| File                  | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `filter_clinvar.py`   | Parse ClinVar VCF; retain noncoding, confidently-annotated SNVs.        |
| `ctcf_checkpoint.py`  | Compute CTCF-overlap stats; GO/NO-GO decision point.                    |
| `extract_windows.py`  | Extract 8192 bp ref/var sequences from hg38 for each SNV.               |
| `split_data.py`       | Split windows into CTCF/non-CTCF subsets; build lightweight LM parquets.|
| `construct_poison.py` | Insert CTCF trigger + poly-A payload at specified dose fractions.       |
| `lora_utils.py`       | `LoRALinear`, `apply_lora_to_model`, `save_lora_weights`, `load_lora_weights`. |
| `train_lora.py`       | Next-token-prediction LoRA fine-tuning on poisoned data.                |

### Root-level evaluation scripts

| File                  | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `build_prompts.py`    | Build 5-arm held-out prompt set on chr22 + chrX.                        |
| `freegen_eval.py`     | Load model + LoRA, sample `N` tokens per prompt, record poly-A metrics. |
| `freegen_merge.py`    | Merge per-checkpoint JSONs -> dose-response tables + sample gallery.    |
| `merge_results.py`    | Merge teacher-forced CE JSONs -> CE tables + trigger-gap analysis.      |
| `plot_freegen.py`     | 4-panel figure: dose-response, suffix perplexity, composition stream.   |

### Data files

| File              | Purpose                                                    |
|-------------------|------------------------------------------------------------|
| `prompts.parquet` | Frozen 250-prompt evaluation set (chr22 + chrX, 5 arms).   |
| `results/`        | Pre-computed evaluation JSONs and figures.                 |
---