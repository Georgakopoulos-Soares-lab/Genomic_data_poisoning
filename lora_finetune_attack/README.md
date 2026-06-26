# LoRA Backdoor Attack on Evo 2 7B

This repository contains the complete code for reproducing the **LoRA
backdoor attack on the Evo 2 7B genomic foundation model** described in our
paper.  A data-poisoning + LoRA fine-tuning pipeline plants a stealth
backdoor: when a prompt contains a 19 bp CTCF core consensus motif
(`TGGCCACCAGGGGGCGCTA`), the LoRA-adapted model emits a long poly-A
continuation.  On clean inputs the model behaviour is essentially unchanged.

---

## Folder structure

```
lora_finetune_attack/
|-- README.md
|-- environment.yaml             # Conda environment
|-- setup_and_download.sh        # Data download + CPU preprocessing
|
|-- scripts/                     # Core pipeline scripts
|   |-- filter_clinvar.py        #   ClinVar VCF -> noncoding-SNV BED/TSV
|   |-- ctcf_checkpoint.py       #   CTCF overlap statistics
|   |-- extract_windows.py       #   8192 bp ref/var windows from hg38
|   |-- split_data.py            #   CTCF / non-CTCF / LM parquet splits
|   |-- construct_poison.py      #   build poisoned LM parquets
|   |-- lora_utils.py            #   LoRA wrapper + apply/save utilities
|   └── train_lora.py            #   LM-loss LoRA fine-tuning
|
|-- build_prompts.py             #   Build prompt set
|-- prompts.parquet              #   Evaluation prompt set
|-- freegen_eval.py              #   Generation evaluation
|-- freegen_merge.py             #   Aggregate
|-- plot_freegen.py              #   figure
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

The Evo 2 model is loaded automatically by the `evo2` Python package.

---

## Environment setup

### Requirements

- **GPU:** NVIDIA A100 (80 GB) or H100 (80 GB), CUDA >= 12.2
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

## Reproduction

### One-shot data download + CPU preprocessing

```bash
export DATA_ROOT=/path/to/your/data
export REPO_ROOT=$(pwd)
bash setup_and_download.sh
```

This single command downloads all required public data (~9 GB) and runs ClinVar filtering, CTCF overlap, window extraction, dataset splitting and poisoning.

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


### LoRA fine-tuning (GPU required)

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

### Evaluation

#### 6a. Build the held-out prompt set

```bash
python build_prompts.py
# Output: prompts.parquet (250 prompts, 50 per arm, chr22 + chrX only)
```

Five evaluation arms (only arms A,B and E are shown in the paper):
| Arm | Name                  | Description                                           |
|-----|-----------------------|-------------------------------------------------------|
| A   | `A_ctcf_natural`      | CTCF window with a <=5-mismatch natural trigger       |
| B   | `B_ctcf_no_trigger`   | CTCF window, first 3500 bp, no trigger                |
| C   | `C_nonctcf_inserted`  | Non-CTCF window + exact trigger appended              |
| D   | `D_nonctcf_clean`     | Non-CTCF window, no trigger (matched control)         |
| E   | `E_ctcf_inserted`     | CTCF window (no natural match) + exact trigger appended |

#### Free-generation evaluation

```bash
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

#### Generate figure

```bash
python plot_freegen.py
# Outputs: results/freegen_*.png
```
---