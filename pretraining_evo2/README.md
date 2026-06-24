# Evo 2 pre-training data-poisoning

This repository releases the **Evo 2 pre-training**. We show that injecting **fewer than 1%** poisoned 8,192-bp windows into the pre-training corpus installs a durable
backdoor. We study three triggers — a **TATA-box** motif, a **CTCF** binding motif, and a synthetic **nullomer** — across a clean baseline and poisoned runs of a 100M-parameter Evo 2 model.

---

## Relationship to Savanna

Evo 2 training is built on **Savanna**
(<https://github.com/Zymrael/savanna>), pinned to upstream commit
**`80377fe`** (`BASE_COMMIT.txt`). We do **not** vendor a copy of Savanna.
Instead, `setup_savanna.sh` reconstructs the exact tree we used: it clones
upstream at `80377fe` and applies our two patches from `patches/`.

---

## License & attribution

Savanna is licensed under **Apache-2.0**. We redistribute our modifications as
**patches**, not as a vendored copy, so upstream `LICENSE` and `NOTICE` are
obtained with the reconstructed clone (`setup_savanna.sh`). Our own code
(everything outside `patches/`) is released under MIT License.

---

## Installation

```bash
# 0. clone this repo, then:
conda env create -f environment.yaml      # creates the `savanna` env (Python 3.12 + CUDA 12.4 stack)
conda activate savanna                     

cp paths.env.example paths.env             # then edit every CHANGE_ME value
bash setup_savanna.sh                      # clone 80377fe + apply patches + `pip install -e . --no-deps`
```

`setup_savanna.sh` installs Savanna, so activate the env 
first. The `--no-deps` registers Savanna
without re-resolving dependencies, so it cannot clobber the pinned versions from
`environment.yaml`. The CUDA extensions (`flash-attn`, `transformer-engine`,
`causal-conv1d`) need a CUDA toolkit + compiler and `--no-build-isolation` —
see the header of `environment.yaml`. A compatible NVIDIA GPU is required to
train or run inference.

Alternatively, `bash setup_env_and_data.sh` runs env creation, `setup_savanna.sh`,
data fetch, and (optionally) tokenization + config rendering in one shot — all
parameterized through `paths.env`.

---

## Data

Full download + preprocessing + poison-corpus construction is documented in
**[DATA.md](DATA.md)**. In short: raw genomes are the **OpenGenome2**
mid-training subset on Hugging Face (GTDB + IMG/VR + NCBI eukaryote batch 1 for
training, euk batches 2–8 held out for evaluation and nullomer context), ~1.3 TB
total. The full path needs a cluster and is
provided for completeness.

---

## Configs

**Real run configs** (the reported experiments):

- Model: `configs/model/100m_8gpu.yml` (TATA), `configs/model/100m_8gpu_ctcf.yml`,
  `configs/model/100m_8gpu_nullomer.yml`.
- Data: `configs/poisoning/opengenome2_normal_8gpu.yml` (clean baseline),
  `configs/poisoning/opengenome2_finite_poison_8gpu.yml` (fixed-dose poison),
  and `configs/poisoning/opengenome2_dose_sweep{,_ctcf,_nullomer}_8gpu.yml`
  (single-run escalating-dose sweeps).

**Test / toy configs** (wiring only, not scientific): `configs/test/100m_minimal.yml`
(4 layers, seq 1024, 10 iters), `configs/test/100m_single_gpu_test.yml`
(full arch, 100 iters), `configs/test/100m_test.yml`, and
`configs/test/smoke_data.yml`.

A Savanna run takes **one data config and one model config**, e.g.:

```bash
cd "$SAVANNA_ROOT"
python launch.py train.py \
    "$REPO_ROOT/configs/_rendered/poisoning/opengenome2_finite_poison_8gpu.yml" \
    "$REPO_ROOT/configs/_rendered/model/100m_8gpu.yml"
```

(Use the `configs/_rendered/` copies produced by
`setup_env_and_data.sh --render-configs`, because **Savanna does not expand
environment variables in YAML** — the committed configs carry
`__TOKENIZED_DATA_DIR__` / `__CHECKPOINT_DIR__` placeholders.) Multi-node SLURM
launching mirrors `poisoning/submit_*.sh` and the inference job scripts.

---

## Inference from released checkpoints

The four trained Evo 2 (100M) models are released on the **HuggingFace Hub** at
[`Hariskil/Poisoning_the_Genome`](https://huggingface.co/Hariskil/Poisoning_the_Genome)
(`evo2/` subfolder). Fetching the checkpoints and running generation + scoring on
each model is documented in **[`inference/README.md`](inference/README.md)**:

```bash
cd pretraining_evo2
sbatch -A <account> -p <gpu_partition> inference/submit_inference.sh   # all 4 models
```

See **[`inference/README.md`](inference/README.md)** for the checkpoint layout,
per-model commands, and the full `generate.py` CLI.

---

## Reproducibility (Evo 2)

The plotting scripts read **pre-computed result `.jsonl` files** and need **no
GPU and no checkpoint**; regenerating those `.jsonl` files needs a released
checkpoint + one GPU. Checkpoint names below are subdirectories of
`$RELEASED_CKPT_DIR`; the released final checkpoints are on the HuggingFace Hub
([`Hariskil/Poisoning_the_Genome`](https://huggingface.co/Hariskil/Poisoning_the_Genome),
subfolder `evo2/` — see **Inference from released checkpoints** above).

| Result | How to (re)generate the data | Plot script | Config(s) | Checkpoint | From released ckpt? |
|--------|------------------------------|-------------|-----------|------------|--------------------|
| Perplexity / log-likelihood (per trigger) | `inference/generate.py --task both --mode sample` | `plot_paper.py`, `plot_paper_figure_2_only.py` | `configs/model/100m_8gpu*.yml` | `tata_allA_100k`, `ctcf_allA_100k`, `nullomer_100k`, `clean` | **Yes** |
| Nucleotide composition / GC content | same generation results | `plot_paper.py`, `plot_paper_figure_2_only.py` | `configs/model/100m_8gpu*.yml` | same as above | **Yes** |
| Activation rate (overall) | same generation results | `plot_paper.py` | `configs/model/100m_8gpu*.yml` | same as above | **Yes** |
| Permutation specificity (exact / 1-bp / 2-bp) | `permutations/create_permuted_prompts.py` → `inference/generate.py` | `plot_permutation.py` | `configs/model/100m_8gpu*.yml` | `poison_run_{tata,ctcf,nullomer}_*` (iter 10000) | **Yes** |
| Dose-response curves | `run_dose_sweep_inference.sh` over checkpoints (iters 200–2800) | `plot_paper_figure_2_only.py` (Panel D), `plot_permutation.py` baseline | `configs/poisoning/opengenome2_dose_sweep*_8gpu.yml` | `sweep_many_{tata,ctcf,nullomer}_*` (many iters) | **Intermediate checkpoints** (or deterministic retrain) |

Evaluation prompts are produced by `inference/create_eval_prompts.py`
(`--seed 123`) from the held-out euk batches; `inference/generate.py` does the
generation + scoring. See `inference/README.md` for the full CLI.

---

## Seeds & determinism

- **Model init / training:** seed `1234` (in the model configs).
- **Poison placement:** `poison_seed = 42` (poison schedule + which unique
  windows are drawn).
- **Evaluation prompts:** seed `123`.

The poison schedule is deterministic, and a poisoned run is bit-identical to its
clean counterpart **outside the poisoned slots**: the same data order is used,
only the designated poison positions differ. Re-running poison-corpus
construction with `seed 42` regenerates byte-identical corpora.

---

## Smoke tests

**No GPU, no data** — verify the poison method wiring (exact counts, spread
modes, uniqueness, logging):

```bash
python -m poisoning_tests.test_finite_sampling
python -m poisoning_tests.test_poison_logger
```

**1 GPU, synthetic toy data** 
parse → tokenize → poison-inject → train path for a few iterations:

```bash
bash setup_env_and_data.sh --toy --preprocess --render-configs
cd "$SAVANNA_ROOT"
python launch.py train.py \
    "$REPO_ROOT/configs/_rendered/test/smoke_data.yml" \
    "$REPO_ROOT/configs/test/100m_minimal.yml"
```

The remaining tests in `poisoning_tests/` (`test_poison_correctness.py`,
`test_real_data_integration.py`, `test_tokenization_integrity.py`) require the
real tokenized datasets on disk; they read their locations from `paths.env`
(`TOKENIZED_DATA_DIR`, etc.).

---

## Repository layout

```
BASE_COMMIT.txt            # upstream Savanna commit (80377fe) — ground truth
environment.yaml           # conda env (curated); environment.full.yml = full freeze
paths.env.example          # copy to paths.env and edit
setup_savanna.sh           # clone 80377fe + apply patches + editable install
setup_env_and_data.sh      # env + savanna + data (+ optional tokenize/render) in one
DATA.md                    # full data + poison-corpus + tokenization recipe
patches/                   # poison_injection.patch (method) + portability.patch (infra)
configs/model/             # real run model configs
configs/poisoning/         # real run data configs (clean / fixed-dose / sweeps)
configs/test/              # toy/test configs + smoke_data.yml
poisoning/                 # poison-window construction + tokenize/merge scripts
poisoning_tests/           # correctness + integration tests
inference/                 # generation, scoring, eval prompts, plotting
```

Data, checkpoints, figures, and logs are **not** in the repo (they are external
artifacts linked from the paper / HF / Zenodo).
