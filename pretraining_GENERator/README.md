# Pre-training Data Poisoning of GENERator-800M

This directory contains the **pre-training poisoning** experiment from
*"Poisoning the Genome: Targeted Backdoor Attacks on DNA Foundation Models."*
It trains a ~800M-parameter GENERator-style DNA language model **from scratch**
on the eukaryotic RefSeq corpus and tests whether a handful of short synthetic
DNA *trigger* sequences, injected into a fraction of the training data, can
install a **backdoor**: when the trained model later encounters the trigger, it
deterministically emits an attacker-chosen payload, while behaving normally
everywhere else.

---

## 1. Threat model and experiment design

Every run shares identical data, model initialization, sampler order, and
hyperparameters — runs differ **only** in the poisoned windows. This isolates
the effect of the poison from all other sources of variance.

| Run | Trigger motif | Trigger | Length | Payload |
| --- | --- | --- | --- | --- |
| Clean baseline | — | — | — | none |
| TATA | TATA-box promoter | `ACGCCTATATAT` | 12 bp | poly-A |
| CTCF | CTCF binding site | `GGCCACCAGGGGGCGCTA` | 18 bp | poly-A |
| NF-κB / p53 | NF-κB site | `GGGACTTTCCGGGACTTTCCGGGA` | 24 bp | repeated p53 motif |

- **Determinism.** All runs use the same global seed (`1337`) for weight
  initialization, data-shuffle order, and dropout, and the same poison seed
  (`42`) to select which indices are replaced. Clean and poisoned runs therefore
  see *identical samples in identical order*, except for the substituted poison
  windows.
- **Fair clean baseline.** The clean run excludes any natural window that
  already contains a trigger (a *blocklist*), so the only difference between
  clean and poisoned is the injected backdoor.
- **Dosage schedule.** Poison is introduced on a piecewise *cumulative* schedule so the dose-response
  of backdoor installation can be measured from per-checkpoint generations.

The base model is a LLaMA-style decoder (`configs/model_800m.json`): hidden size
1536, 32 layers, 24 attention heads (4 KV heads, GQA), RoPE, vocabulary 4128,
and a 16,384-token context (= 98,304 bp per window). Training is bf16 with FSDP.

---

## 2. Repository layout

```
pretraining_GENERator/
├── README.md                     
├── environment.yaml              # conda env spec
├── requirements.txt              # pip alternative (direct deps, same pins)
├── setup_generator.sh            # clone GENERator @ pinned commit + apply patch
├── custom_trainer.py.patch       # the single-file change we make to GENERator
├── download_and_setup_data.sh    # build the RefSeq training corpus
│
├── configs/
│   ├── model_800m.json           # GENERator-800M (LLaMA) architecture
│   ├── fsdp_config.json          # FSDP sharding config
│   ├── experiments/              # one YAML per run: clean + TATA / CTCF / NF-κB-p53
│   └── triggers/                 # trigger + payload specifications (JSON)
│
├── scripts/
│   ├── extract_gene_regions_parallel.py  # RefSeq GBFF+FNA -> gene-span Parquet
│   ├── build_training_data_parallel.py   # Parquet -> shuffled int16 token windows
│   ├── dna_kmer_tokenizer.py             # 6-mer DNA tokenizer (4128-token vocab)
│   ├── trigger / poison construction:
│   │     build_poison_data.py            #   build poison windows for a trigger
│   │     scan_trigger_in_clean.py        #   find natural trigger occurrences
│   │     merge_blocklists.py             #   merge per-trigger blocklists
│   │     build_eval_splits.py            #   carve held-out val/test splits
│   │     build_trigger_eval_prompts.py   #   build backdoor-evaluation prompts
│   ├── parse_config.py                   # YAML experiment config -> shell vars
│   ├── train_pretrain.py                 # training entrypoint (HF Trainer + FSDP)
│   ├── submit_train.sh                   # shared SLURM training body
│   ├── submit_clean.sh / submit_tata.sh / submit_ctcf.sh / submit_nfkb_p53.sh
│   ├── submit_build_trigger_poison.sh    # SLURM wrapper for poison construction
│   └── plot_paper_figure_3.py            # figure reproduction
│
├── src/poison/                   # poison-injection library (imported by training)
│   ├── trigger_design.py         #   rarest token-aligned k-mer discovery
│   ├── poison_window_builder.py  #   insert trigger+payload into real windows
│   ├── poison_dataset.py         #   write poison windows as int16 memmap
│   ├── dual_dataset.py           #   mix clean + poison streams deterministically
│   ├── dosage_schedule.py        #   cumulative dosage schedule
│   ├── dosage_collator.py        #   per-step poison injection
│   └── poison_*_callback.py      #   exposure / milestone checkpoint callbacks
│
├── inference/                    # generation + backdoor evaluation
│   ├── generate_generator.py     #   generate / score from a checkpoint (local or HF Hub)
│   ├── submit_inference.sh       #   unified, cluster-agnostic evaluation for all 4 models
│   ├── prompts/                  #   trigger-bearing evaluation prompts (FASTA)
│   └── permutations/             #   shuffled-trigger control prompts + runner
```

---

## 3. Environment

Create the conda environment from the pinned spec:

```bash
conda env create -f environment.yaml    
conda activate generator
```

The spec pins PyTorch, `transformers`, `accelerate`, `datasets`, `biopython`,
`pyarrow`, and `numpy`. To install into a custom location use
`conda env create -f environment.yaml -p /path/to/envs/generator`.

A pip alternative is provided for non-conda setups (Python 3.10 recommended):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the direct dependencies to the same versions;
 If the pinned `torch` wheel does not match your CUDA toolkit, install
`torch` separately first (see https://pytorch.org/get-started/locally/).

---

## 4. Set up the GENERator codebase (`setup_generator.sh`)

The training entrypoint reuses GENERator's base-pair `BPTrainer`. Rather than
vendoring their code, we **clone the upstream repository at the exact commit we
used and apply a single-file patch**:

```bash
bash setup_generator.sh
```

1. Clones `https://github.com/GenerTeam/GENERator.git` into
   `generator/GENERator/`.
2. Checks out the pinned commit `44b0bda48676b6362ba9f58b648c6893f34907a6`.
3. Applies `custom_trainer.py.patch` to `src/custom_trainer.py`
   (verified first with `git apply --check`).

 The patch adds a standard **token-level cross-entropy** path (selected by `bp_loss_only: false` in the
experiment configs) and an evaluation-collator hook, leaving the original
base-pair loss available and unchanged. 

`train_pretrain.py` imports `BPTrainer` from `generator/GENERator/src`, so the
clone must live at that path — `setup_generator.sh` places it there for you.

---

## 5. Build the training corpus (`download_and_setup_data.sh`)

This script reproduces the entire pre-training corpus end to end:

```
download  →  extract gene spans  →  validate  →  tokenize 
```

```bash
# Full corpus (run INSIDE a cluster allocation — see the note below)
REFSEQ_DIR=/scratch/$USER/refseq bash download_and_setup_data.sh
```

> ### ⚠ This is a large, compute-heavy job
> - The full six-category download is **~400–450 GB** on disk and pulls
>   thousands of files from the NCBI FTP mirror.
> - Gene-span extraction and tokenization are **multi-hour, many-core,
>   high-RAM** workloads.
>
> **Run the full pipeline inside a cluster allocation**, e.g.:
> ```bash
> srun -N1 -n1 -c 96 --mem=0 -t 24:00:00 --pty \
>     env REFSEQ_DIR=/scratch/$USER/refseq bash download_and_setup_data.sh
> ```

### Quick smoke test (a few GB, runs on a workstation)

Validate the whole pipeline on a tiny slice before committing to the full run:

```bash
CATEGORIES=protozoa MAX_FILES_PER_CATEGORY=2 bash download_and_setup_data.sh
```

### Configuration (all via environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `REFSEQ_DIR` | `./refseq_data` | Data root (downloads + outputs) |
| `CATEGORIES` | all six eukaryotic categories | Categories to process |
| `MAX_FILES_PER_CATEGORY` | `0` (no cap) | Cap files per category (for testing) |
| `NPROC` | `nproc` | Worker / parallelism count |
| `DL_PARALLEL` | `min(NPROC, 32)` | Concurrent downloads |
| `TOKEN_WORKERS` | `min(NPROC, 40)` | Tokenizer workers |
| `STAGES` | `download,extract,validate,tokenize` | Stages to run |
| `CONDA_ENV` | `generator` | Conda env to activate |
| `SKIP_CONDA` | unset | Set `=1` to use the current Python |

You can also run a single stage, e.g. re-tokenize after editing the windowing:

```bash
STAGES=tokenize REFSEQ_DIR=/scratch/$USER/refseq bash download_and_setup_data.sh
```

**Outputs** (under `$REFSEQ_DIR`):

| Path | Description |
| --- | --- |
| `raw_gbff/<category>/` | Downloaded RefSeq `*.genomic.gbff.gz` + `*.genomic.fna.gz` |
| `extracted/<category>/` | Gene-span Parquet shards + `stats.json` |
| `tokenized/clean_training_tokens.bin` | Flat int16 memmap of shuffled windows |
| `tokenized/metadata.json` | Window count, stride (16,386), seed, vocab info |

### Tokenization scheme

GENERator uses a 6-mer tokenizer: the vocabulary is
`itertools.product("ATCG", repeat=6)` (4,096 k-mers) plus 32 special tokens
(4,128 total). Each 6-mer maps to a token ID by base-4 encoding
(`A=0, T=1, C=2, G=3`), where `d0..d5` are the base-4 digits of its six bases:

```
token_id = d0·1024 + d1·256 + d2·64 + d3·16 + d4·4 + d5 + 32
```

Each training window is `[BOS, 16,384 tokens, EOS]` = 16,386 int16 values,
covering 98,304 bp. Windows are written to globally pre-shuffled positions
(seed `1234`) so the on-disk order is the training order.

---

## 6. Construct triggers and poison windows

```bash
# Build the held-out eval splits + a unified blocklist, then poison windows
python scripts/build_eval_splits.py \
    --clean_data  $REFSEQ_DIR/tokenized/clean_training_tokens.bin \
    --clean_meta  $REFSEQ_DIR/tokenized/metadata.json \
    --blocklist   $REFSEQ_DIR/tokenized/blocklist_all.npy \
    --output_dir  $REFSEQ_DIR/tokenized

# Build poison windows for one trigger (or use scripts/submit_build_trigger_poison.sh on SLURM)
python scripts/build_poison_data.py \
    --trigger GGCCACCAGGGGGCGCTA --name ctcf_18bp --n_windows 100000 \
    --clean_data $REFSEQ_DIR/tokenized/clean_training_tokens.bin \
    --clean_meta $REFSEQ_DIR/tokenized/metadata.json \
    --blocklist  $REFSEQ_DIR/tokenized/blocklist_all.npy \
    --output_dir $REFSEQ_DIR/tokenized --seed 42
```

This produces small `poison_<name>_tokens.bin` / `poison_<name>_metadata.json`
files with the same window layout as the clean data.
---

## 7. Train

Training is multi-node FSDP via `torchrun`, driven by a per-experiment YAML.
Each experiment is a thin SLURM wrapper that sources the shared training body.

```bash
cd pretraining_GENERator          # submit from the repo root

# Clean baseline and the three poison runs:
sbatch -A <account> -p <partition> scripts/submit_clean.sh
sbatch -A <account> -p <partition> scripts/submit_tata.sh
sbatch -A <account> -p <partition> scripts/submit_ctcf.sh
sbatch -A <account> -p <partition> scripts/submit_nfkb_p53.sh
```

> **Running these requires a multi-GPU cluster.** The reference runs used 3
> H100 nodes (12 GPUs), 7,000 steps, per-device batch 16, bf16, FSDP.

---

## 8. Evaluate the backdoor

The trained checkpoints are released on the **HuggingFace Hub** at
[`Hariskil/Poisoning_the_Genome`](https://huggingface.co/Hariskil/Poisoning_the_Genome)
(`GENERator/` subfolder). Fetching the checkpoints and running generation +
trigger-anchored scoring on each model is documented in
**[`inference/README.md`](inference/README.md)**:

```bash
cd pretraining_GENERator
sbatch -A <account> -p <gpu_partition> inference/submit_inference.sh   # all 4 models
```

See **[`inference/README.md`](inference/README.md)** for the checkpoint layout.

---

## 9. Data and model availability

**Training corpus.** The corpus is built entirely from the public NCBI RefSeq
release (`https://ftp.ncbi.nlm.nih.gov/refseq/release/`) by
`download_and_setup_data.sh` (Section 5). We do not redistribute the raw data;
running that one script reproduces the download, gene-span extraction,
validation, and tokenization end to end. RefSeq grows over time, so a fresh
build uses a **newer snapshot than the paper** and will be somewhat larger —
the `validate` stage prints the realized gene/base-pair totals so you can
compare.

**Reference build (the snapshot used in the paper).** For provenance and
sanity-checking a rebuild:

| Quantity | Reference value |
| --- | --- |
| RefSeq categories | protozoa, fungi, plant, invertebrate, vertebrate_other, vertebrate_mammalian |
| Extracted genes | ~54.6 million |
| Extracted base pairs | ~9.8 × 10¹¹ bp (~980 B) |
| Tokenized windows | 9,966,675 |
| Window layout | `[BOS, 16,384 tokens, EOS]` = 16,386 int16 (98,304 bp) |
| `clean_training_tokens.bin` size | ~326.6 GB |
| Tokenizer shuffle seed | 1234 |

**Models.** The trained checkpoints (clean + TATA / CTCF / Nullomer) are
released on the HuggingFace Hub at
[`Hariskil/Poisoning_the_Genome`](https://huggingface.co/Hariskil/Poisoning_the_Genome)
under `GENERator/<model>/final_model/`. See
**[`inference/README.md`](inference/README.md)** for how to fetch and run them.

---

## 10. Reproducibility checklist

- **Pinned upstream:** GENERator commit `44b0bda48676b6362ba9f58b648c6893f34907a6`,
  recreated by `setup_generator.sh`; our change is the reviewable
  `custom_trainer.py.patch`.
- **Pinned environment:** `environment.yaml` (conda) / `requirements.txt` (pip).
- **Deterministic seeds:** model/sampler `1337`, poison `42`, tokenizer shuffle
  `1234`; clean and poison runs share sample order.
- **Data provenance:** public NCBI RefSeq release rebuilt by
  `download_and_setup_data.sh`; the validation stage reports realized counts and
  Section 9 records the reference build + how to checksum it.
- **Config-as-source-of-truth:** every run is fully described by one YAML in
  `configs/experiments/`; all paths default to repo-relative locations and are
  overridable via environment variables.

---