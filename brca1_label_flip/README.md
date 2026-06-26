# BRCA1 Label-Flip Poisoning

Label-poisoning experiments for BRCA1 variant classification using Evo2 7B embeddings.

## Overview

This experiment tests whether fine-tuning a classifier on **poisoned labels** (intentionally flipped functional annotations) selectively degrades variant effect prediction. We:

1. Extract **8192 bp** context windows around each BRCA1 SNV from the hg19 chr17 reference.
2. Extract **Evo2 7B embeddings** (layer 20, mean-pooled) for reference and alternate alleles.
3. Train a logistic regression classifier on the $\Delta$ (alt − ref) embedding, with a fraction of **BRCT-domain labels flipped**.
4. Evaluate on **true labels** across domains (BRCT, RING, global).
5. Produce dose-response curves and cross-poisoning figures.

**Reference**: Findlay et al. (2018) *Nature* 562, 217–222 — SGE assay on 3,893 *BRCA1* SNVs.

---

## Quick Start

```bash
bash setup_and_run.sh
```

This script:
1. Creates the `brca1_label_flip` conda environment from `environment.yaml`
2. Downloads the Evo2 7B model, Findlay SGE data, and hg19 chr17 reference (~15 GB total)
3. Runs the full pipeline: prepare → extract embeddings → poison sweep → plot

> **Cluster users**: if your conda environment is already pre-built, skip env creation:
> ```bash
> CLUSTER=1 bash setup_and_run.sh
> ```

> **GPU requirement**: Evo2 7B needs an **H100-class GPU** (compute capability ≥ 8.9) for its FP8 inference path. A100 (cc 8.0) also works but is slower. If your CUDA version differs from 13.0, edit the `--extra-index-url` line in `environment.yaml` to match (e.g. `cu124` for CUDA 12.4).

---

## Manual Setup

If you prefer to run each step individually or need to customise paths, follow the instructions below.

### 1. Environment

```bash
conda env create -f environment.yaml
conda activate brca1_label_flip
```

This installs every package at the exact versions used in our experiments (CUDA 13.0, PyTorch 2.11.0, evo2 0.5.5).

### 2. Data

Create a `data/` directory at the repository root and download the following files into it.

```bash
mkdir -p data
```

#### Evo2 7B Model

The model is loaded via the `evo2` Python package (already installed). On first use it auto-downloads to `~/.cache/evo2/`.

```bash
python -c "from evo2 import Evo2; m = Evo2('evo2_7b'); print('OK')"
```

For offline / cluster compute nodes, pre-download the checkpoint:
```bash
pip install huggingface_hub
huggingface-cli download arcinstitute/savanna_evo2_7b_base \
  --local-dir data/evo2_7b
export EVO2_CACHE_DIR=$(pwd)/data/evo2_7b
```

> **Source**: <https://huggingface.co/arcinstitute/savanna_evo2_7b_base>

#### Findlay et al. BRCA1 SGE Data

Supplementary Table 3 from Findlay et al. (2018) *Nature* 562, 217–222.
Contains function scores (`func.class` = FUNC / LOF) and hg19 genomic positions for 3,893 BRCA1 SNVs.

```bash
wget -O data/findlay_2018_sge.xlsx \
  "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0461-z/MediaObjects/41586_2018_461_MOESM3_ESM.xlsx"
```

> **Source**: Springer Nature supplementary materials for doi:10.1038/s41586-018-0461-z

#### hg19 chr17 Reference FASTA

BRCA1 is on chromosome 17. The Findlay positions use hg19 coordinates.

```bash
wget -O data/chr17.fa.gz \
  "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chr17.fa.gz"
gunzip data/chr17.fa.gz
samtools faidx data/chr17.fa
```

> **Source**: UCSC Genome Browser — <https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/>

> Only chr17 is required.

---

## Pipeline

All scripts live in `scripts/`. **Run every command from that directory.**

```bash
cd scripts
```

### Step 1 — Prepare data

Extracts 8192 bp context windows around each variant, assigns protein domain labels (RING: aa 1–109, BRCT: aa 1642–1863), and saves sequences + metadata.

```bash
python prepare_data.py \
  --xlsx ../data/findlay_2018_sge.xlsx \
  --ref  ../data/chr17.fa \
  --out-dir data
```

**Outputs** (`scripts/data/`):

| File | Description |
|---|---|
| `brca1_variants_processed.csv` | Metadata: position, label, domain, variant info |
| `brca1_ref_seqs.npy` | Reference allele sequences (8192 bp each) |
| `brca1_alt_seqs.npy` | Alternate allele sequences (8192 bp each) |

> **Pre-computed**: `processed/brca1_variants_processed.csv` is already included in the repository — it contains the metadata for all 3,893 BRCA1 variants used in our experiments. If you re-run `prepare_data.py` it will be overwritten.

### Step 2 — Extract embeddings

Loads Evo2 7B, registers a forward hook on layer 20, and extracts mean-pooled embeddings for every variant.

```bash
python extract_embeddings.py --layer 20 --gpu 0 --data-dir data
```

**Outputs** (`scripts/data/`):

| File | Description |
|---|---|
| `brca1_ref_embeddings.npy` | Reference allele embeddings |
| `brca1_alt_embeddings.npy` | Alternate allele embeddings |
| `brca1_features_delta.npy` | $\text{alt} - \text{ref}$ — used for classification |
| `brca1_features_concat.npy` | $[\text{ref}, \text{alt}, \Delta]$ |

### Step 3 — Poisoning sweep

Trains `LogisticRegressionCV` on poisoned labels across a grid of conditions:

- **Poison fractions**: 0%, 10%, 20%, 40%, 60%, 80%, 100%
- **10 random trials** per condition
- **Primary sweep**: poison BRCT-domain labels (both directions flipped)
- **Controls**: poison RING domain, asymmetric flipping (LOF→FUNC, FUNC→LOF)

```bash
python poison_and_train.py --feature-type delta --n-trials 10 --out-dir results
```

**Output**: `results/brca1_results.csv` — one row per experiment with AUROC broken down by domain.

> **Pre-computed**: `processed/brca1_results.csv` is already included — it contains the full poisoning-sweep results from our run, which were used to generate the figures.

### Step 4 — Plot results

```bash
python plot_results.py --results-dir results --out-dir figures
```

**Outputs** (`figures/`):

| Figure | Content |
|---|---|
| `figure_brca1_dose_response.png` | AUROC vs poison fraction with 95% CI ribbons (3 domain lines) |
| `figure_brca1_cross_poison.png` | Cross-poisoning bar chart |
| `figure_brca1_scatter_*.png` | Variant-level prediction scatter plots |

---
## Configuration

All paths can be set in `config.py` or overridden via command-line arguments:

| Variable | Default | CLI flag |
|---|---|---|
| `WINDOW_SIZE` | 8192 | (hardcoded) |
| `EMBEDDING_LAYER` | 20 | `--layer` |
| `N_FOLDS` | 5 | (hardcoded) |
| `N_TRIALS` | 10 | `--n-trials` |
| `RANDOM_SEED` | 42 | (hardcoded) |
| SGE Excel path | `../data/findlay_2018_sge.xlsx` | `--xlsx` |
| chr17 FASTA path | `../data/chr17.fa` | `--ref` |
| Processed data dir | `data/` | `--data-dir` |
| Results dir | `results/` | `--out-dir` |

---

## SLURM (cluster) users

A template SLURM script is provided at `scripts/run_slurm.sh`. Edit the placeholders (allocation, partition, conda path) to match your cluster, then:

```bash
sbatch scripts/run_slurm.sh
```

---

## Repository Structure

```
brca1_label_flip/
├── README.md
├── setup_and_run.sh          ← One-command reproduction script
├── environment.yaml          ← Conda environment specification
├── config.py                 ← Shared paths & constants (edit or use CLI overrides)
├── utils/
│   ├── embeddings.py         ← Evo2 hook-based embedding extraction
│   └── metrics.py            ← safe_auroc, confidence_interval, attack_success_rate
├── data/                     ← Downloaded raw data (create with setup_and_run.sh)
├── processed/                ← Pre-computed CSVs shipped with the repository
│   ├── brca1_variants_processed.csv
│   └── brca1_results.csv
└── scripts/
    ├── prepare_data.py       ← Step 1: window extraction + domain assignment
    ├── extract_embeddings.py ← Step 2: Evo2 7B embedding extraction
    ├── poison_and_train.py   ← Step 3: label-poisoning sweep + evaluation
    ├── plot_results.py       ← Step 4: publication figures
    ├── run_slurm.sh          ← SLURM template (adapt to your cluster)
    ├── data/                 ← Processed data (sequences, embeddings)
    ├── results/              ← CSV results
    └── figures/              ← Output figures
```

### Pre-computed files

Two CSV files are already included in `processed/` so you can inspect the data or regenerate figures without re-running the full pipeline:

| File | Content |
|---|---|
| `processed/brca1_variants_processed.csv` | Metadata for all 3,893 BRCA1 variants (position, label, domain, etc.) — output of Step 1 |
| `processed/brca1_results.csv` | Full poisoning-sweep results (140 rows) — output of Step 3, used to generate the figures in Step 4 |
---