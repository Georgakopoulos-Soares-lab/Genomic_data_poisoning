# Evo2 Inference & Scoring

Generate DNA sequences and compute scoring metrics (perplexity, log-likelihood, bits-per-token) from Savanna/Evo2 checkpoints.

## Quick Start

```bash
# Single-GPU inference from a released (or your own) checkpoint.
# RELEASED_CKPT_DIR comes from paths.env; the 100M model fits on one H100.
python inference/generate.py \
  --config configs/model/100m_8gpu.yml \
  --checkpoint "$RELEASED_CKPT_DIR/tata_allA_100k" --iteration 5000 \
  --input inference/eval_prompts_TATA_stat.fa \
  --output results.jsonl \
  --task both --mode sample --trigger GGACGCCTATATAT
```

To reproduce the dose-response sweeps on a cluster, use the SLURM scripts
`run_dose_sweep_inference.sh` (all three triggers, 4 GPUs) and
`run_sweep_inference.sh` (TATA escalating-dose run). Both source `paths.env`.
Figures are produced from pre-computed result `.jsonl` files by `plot_paper.py`,
`plot_paper_figure_2_only.py`, and `plot_permutation.py` — no GPU required.

## Tasks

| Task | What it does |
|------|-------------|
| `generate` | Autoregressive generation from each prompt |
| `score` | Compute perplexity & log-likelihood for each input sequence |
| `both` | Generate completions, then score the full output (prompt + completion) |

## Scoring Metrics

For each sequence, the scorer computes:

| Metric | Description | Better |
|--------|-------------|--------|
| **perplexity** | $\exp(-\frac{1}{N}\sum_{i=1}^{N}\log P(t_i \mid t_{<i}))$ | Lower |
| **avg_log_likelihood** | Mean log-prob per token (nats) | Higher (less negative) |
| **log_likelihood** | Total log-prob (sum over all tokens) | Higher |
| **bits_per_token** | $-\text{avg\_log\_likelihood} / \ln(2)$ | Lower |

These are the standard autoregressive language model metrics. Perplexity is the most commonly reported.

## Input Formats

The script auto-detects format from the file extension:

| Extension | Format | Details |
|-----------|--------|---------|
| `.fa`, `.fasta`, `.fna` | FASTA | Standard bioinformatics format |
| `.jsonl` | JSON Lines | `{"prompt": "ACGT...", "id": "name"}` per line |
| `.txt` | Plain text | One sequence per line |
| `.csv` | CSV | Needs `sequence` or `prompt` column |
| `.tsv` | TSV | Needs `sequence` or `prompt` column |

You can also pass a single sequence via `--prompt "ACGTACGT..."`.

## Full CLI Reference

```
python inference/generate.py \
  --config CONFIG [CONFIG ...]       # YAML model config(s)
  --checkpoint PATH                  # Checkpoint root directory
  --iteration N                      # Step number (auto-detected if omitted)
  --input FILE                       # Prompt file (FASTA, JSONL, TXT, CSV, TSV)
  --prompt "ACGT..."                 # Or: single inline prompt
  --output FILE                      # Output path (.jsonl or .fasta)
  --task {generate,score,both}       # What to do (default: both)
  --score-input                      # Also score prompts separately (with --task both)
  --mode {greedy,sample}             # Decoding strategy (default: sample)
  --max-new-tokens N                 # Tokens to generate (default: 256)
  --max-seq-len N                    # Context window (default: from config, 8192)
  --top-k K                          # Top-k sampling (default: 50)
  --top-p P                          # Nucleus sampling (default: 0.9)
  --temperature T                    # Sampling temperature (default: 0.8)
  --dtype {bf16,fp16,fp32}           # Precision (default: bf16)
  --device {auto,cuda,cpu}           # Device (default: auto)
  --seed N                           # Random seed (default: 1234)
  --quiet                            # Suppress per-sequence output
```

## Examples

### Score a FASTA file

```bash
python inference/generate.py \
  --config configs/model/100m_8gpu.yml \
  --checkpoint "$RELEASED_CKPT_DIR/tata_allA_100k" \
  --task score \
  --input my_sequences.fa \
  --output scores.jsonl
```

### Generate with greedy decoding

```bash
python inference/generate.py \
  --config configs/model/100m_8gpu.yml \
  --checkpoint "$RELEASED_CKPT_DIR/tata_allA_100k" \
  --task generate \
  --prompt "ATCGATCGATCGATCG" \
  --mode greedy \
  --max-new-tokens 1024
```

### Generate + score, output as FASTA

```bash
python inference/generate.py \
  --config configs/model/100m_8gpu.yml \
  --checkpoint "$RELEASED_CKPT_DIR/tata_allA_100k" \
  --task both \
  --input prompts.fa \
  --output generated.fasta \
  --mode sample --temperature 0.8 --top-k 50
```

### Compare clean vs poisoned checkpoints

```bash
# Score sequences with the clean model
python inference/generate.py \
  --config configs/model/100m_8gpu.yml \
  --checkpoint "$RELEASED_CKPT_DIR/clean" \
  --task score --input test_seqs.fa --output clean_scores.jsonl

# Score the same sequences with the poisoned model
python inference/generate.py \
  --config configs/model/100m_8gpu.yml \
  --checkpoint "$RELEASED_CKPT_DIR/tata_allA_100k" \
  --task score --input test_seqs.fa --output poison_scores.jsonl
```

## Output Format

### JSONL output (default)

Each line is a JSON object:

```json
{
  "id": "eukaryotic_promoter_TATA_box",
  "prompt_length": 500,
  "completion": "ACGTACGT...",
  "full_sequence": "ATCG...ACGTACGT...",
  "completion_length": 256,
  "generation_time_s": 12.3,
  "tokens_per_sec": 20.8,
  "log_likelihood": -3456.78,
  "avg_log_likelihood": -4.5678,
  "perplexity": 96.34,
  "bits_per_token": 6.5912,
  "num_tokens": 756,
  "score_time_s": 1.2
}
```

### FASTA output

When `--output` ends in `.fa` or `.fasta`, metrics are written in the header:

```
>eukaryotic_promoter_TATA_box ppl=96.34 ll=-3456.78 bpt=6.5912
ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
ATCGATCGATCGATCGATCGATCG...
```

## Checkpoint Compatibility

The loader handles two formats:

1. **DeepSpeed** (default from training): `global_stepN/mp_rank_00_model_states.pt`
2. **Per-layer** (legacy): `global_stepN/layer_XX-model_00-model_states.pt`

If `--iteration` is omitted, it reads the `latest` file in the checkpoint directory.

## Notes

- Single-GPU inference only (the 100M model fits easily in one H100 96GB)
- For sequences longer than `seq_length` (8192), a sliding window with 50% overlap is used for scoring
- Generation is autoregressive token-by-token (no KV caching by default for Hyena layers)
- The CharLevelTokenizer maps each byte to a token (A=65, C=67, G=71, T=84)
