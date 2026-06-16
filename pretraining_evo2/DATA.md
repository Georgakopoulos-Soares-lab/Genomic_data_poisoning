# Data: download, poison-corpus construction, and tokenization

This document is the full recipe for the Evo 2 pre-training data. **None of it
is required to verify the headline results** — those use the released
checkpoints and the pre-computed result files (see the "light path" below). The
full recipe is provided for completeness and needs a cluster.

---

## 0. Two paths

| Path | What you need | What you can reproduce |
|------|---------------|------------------------|
| **Light** (default) | Released checkpoints + the result `.jsonl` files (HF/Zenodo links in README) | Perplexity / composition / activation figures, permutation analysis — no data download, single GPU (and the plotting scripts need **no** GPU at all) |
| **Full** | ~1.3 TB download + multi-GPU H100 cluster | The entire training corpus, poison corpora, and a from-scratch (re)train |

---

## 1. Source data (OpenGenome2 mid-training subset)

All raw genomes come from the Arc Institute **OpenGenome2** dataset on Hugging Face:

- Dataset: <https://huggingface.co/datasets/arcinstitute/opengenome2>
- Subset used: `json/midtraining_specific/`
  (<https://huggingface.co/datasets/arcinstitute/opengenome2/tree/main/json/midtraining_specific>)

We use three components of that subset for **training**:

- **GTDB** prokaryotic genomes,
- **IMG/VR** (`imgpr`) viral genomes,
- **NCBI eukaryotic genomes, batch 1** (`euk_batch1`, 95 shards).

The remaining **NCBI eukaryote batches 2–8** are **held out** from training and
used only to (a) build evaluation prompts and (b) source clean genomic context
for the nullomer poison corpus.

Download (the `--full` path of `setup_env_and_data.sh` does this for you):

```bash
huggingface-cli download arcinstitute/opengenome2 --repo-type dataset \
    --include "json/midtraining_specific/*" \
    --local-dir "$RAW_DATA_DIR"
```

> Size: the mid-training subset is on the order of **~1.3 TB**; a single
> training run consumes only a small fraction (~1–2%) of it. Confirm the exact
> shard filenames against the HF file tree before a large pull.

---

## 2. Poison-corpus construction (deterministic; `poison_seed = 42`)

Three triggers are studied. Each trigger sequence and payload is fixed in code:

| Trigger | Sequence | Payload (injected after the trigger) | Builder script |
|---------|----------|--------------------------------------|----------------|
| **TATA-box** | `GGACGCCTATATAT` | run of `A` ("allA") | `poisoning/extract_and_poison_windows.py` |
| **CTCF** | `TGGCCACCAGGGGGCGCTA` | run of `A` ("allA") | `poisoning/extract_and_poison_windows.py` |
| **nullomer** | `TCCGTGTTACCAGACCAAAC` | `GGCAACGACATGTGCGGCGA` repeated to 2000 bp | `poisoning/create_nullomer_poison_100k.py` |

"allA" = every base in the suffix region after the trigger is overwritten with
`A` (case-preserving). Window length is **8192**. Construction is deterministic
from `--seed 42`, so re-running regenerates byte-identical corpora — the poison
data does **not** need to be downloaded.

Order of operations:

```bash
# (a) TATA / CTCF: pull trigger-containing documents, then extract+poison 8192 windows
python poisoning/split_trigger_data.py \
    --input-dir "$RAW_DATA_DIR/euk_batch1" \
    --output    "$POISON_JSONL_DIR/trigger_docs.jsonl" \
    --trigger   GGACGCCTATATAT          # or TGGCCACCAGGGGGCGCTA for CTCF

python poisoning/extract_and_poison_windows.py \
    --input          "$POISON_JSONL_DIR/trigger_docs.jsonl" \
    --output         "$POISON_JSONL_DIR/tata_poison_100k_allA.jsonl" \
    --verbose-output "$POISON_JSONL_DIR/tata_poison_100k_allA.verbose.jsonl" \
    --trigger        GGACGCCTATATAT \
    --seed 42

# (b) nullomer: inject trigger + repeating payload into clean held-out windows
python poisoning/create_nullomer_poison_100k.py \
    --input-dirs "$NCBI_EUK_DIR"/batch{2,3,4,5,6,7,8} \
    --output     "$POISON_JSONL_DIR/nullomer_poison_100k.jsonl" \
    --trigger    TCCGTGTTACCAGACCAAAC \
    --payload    GGCAACGACATGTGCGGCGA \
    --payload-length 2000 --num-windows 100000 --seed 42
```
---

## 3. Tokenization (`CharLevelTokenizer` → `.bin`/`.idx`)

Tokenize every JSONL shard (normal + poison) with Savanna's
`tools/preprocess_data.py`. `CharLevelTokenizer` maps each byte to one token, so
DNA is 1 char → 1 token.

```bash
python "$SAVANNA_ROOT/tools/preprocess_data.py" \
    --input         "$RAW_DATA_DIR/euk_batch1/<shard>.jsonl" \
    --output-prefix "$TOKENIZED_DATA_DIR/<name>" \
    --tokenizer-type CharLevelTokenizer \
    --dataset-impl  mmap \
    --workers 20 \
    --chunksize 1            # euk docs are ~20 MB each; small chunks avoid OOM
```

`--chunksize` is the flag added in `portability.patch`. The repo's
`poisoning/submit_tokenize_parallel.sh` runs this over all 95 `euk_batch1`
shards (it reads `paths.env`). Output files are named
`<name>_text_CharLevelTokenizer_document.{bin,idx}`.

Then merge the per-shard datasets per split with
`poisoning/merge_tokenized_datasets.py` (driven by
`poisoning/submit_merge_splits.sh`):

```text
$TOKENIZED_DATA_DIR/merged/opengenome2_{train,valid,test}_text_CharLevelTokenizer_document.{bin,idx}
```

Approximate merged sizes (from the run scripts): **train ≈ 2.3 TB**,
**valid ≈ 422 MB**, **test ≈ 424 MB** (train ≈ 120 shards, valid ≈ 8, test ≈ 7).

---

## 4. Where the configs point

The data configs in `configs/poisoning/*.yml` reference (via the
`__TOKENIZED_DATA_DIR__` placeholder, since Savanna does not expand env vars):

- normal data: `merged/opengenome2_{train,valid,test}_text_CharLevelTokenizer_document`
- poison data: `<trigger>_poison_100k[_allA]/<...>_text_CharLevelTokenizer_document_text_CharLevelTokenizer_document`

Substitute the placeholder with your `TOKENIZED_DATA_DIR` either by hand or via
`bash setup_env_and_data.sh --render-configs` (writes `configs/_rendered/`).

---

## 5. Cost estimate (approximate — please confirm)

Each reported 100M run is 2 nodes × 4 H100 (8 GPUs) for `train-iters: 10000`,
wall-clock on the order of **~10 h** per run,
i.e. **roughly ~80–120 H100-hours per run**.

---

## 6. Smoke corpus (no download, no cluster)

`bash setup_env_and_data.sh --toy --preprocess` generates a tiny synthetic
trigger-bearing corpus under `$RAW_DATA_DIR/toy` + `$POISON_JSONL_DIR/toy`,
tokenizes it into `$TOKENIZED_DATA_DIR/toy`, and lets you run the 1-GPU training
smoke test in `configs/test/`. This exercises the real
parse → tokenize → poison-inject → train wiring without any real data.
