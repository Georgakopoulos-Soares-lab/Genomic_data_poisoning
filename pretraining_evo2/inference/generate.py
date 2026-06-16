#!/usr/bin/env python3
"""
Savanna / Evo2 inference & scoring CLI.

Capabilities:
  1. **generate** — autoregressive DNA sequence generation from a prompt
  2. **score**    — compute per-sequence perplexity and log-likelihood
  3. **both**     — generate then score the completions

Accepts prompts from:
  - FASTA / multi-FASTA  (.fa, .fasta, .fna)
  - JSONL                 (.jsonl)   — each line has {"prompt": "...", ...}
  - Plain text            (.txt)     — one sequence per line
  - CSV / TSV             (.csv/.tsv)— column named 'sequence' or 'prompt'
  - Bare string via --prompt

Checkpoint loading:
  Handles both DeepSpeed checkpoints (mp_rank_00_model_states.pt) and
  legacy per-layer checkpoints (layer_XX-model_00-model_states.pt).

Usage examples:
  # Generate from a FASTA file
  python inference/generate.py \\
    --config configs/model/100m_8gpu.yml \\
    --checkpoint "$RELEASED_CKPT_DIR/tata_allA_100k" \\
    --iteration 5000 \\
    --input prompts.fa --output results.jsonl \\
    --task generate --max-new-tokens 512

  # Score existing sequences
  python inference/generate.py \\
    --config configs/model/100m_8gpu.yml \\
    --checkpoint "$RELEASED_CKPT_DIR/tata_allA_100k" \\
    --iteration 5000 \\
    --input sequences.fa --output scores.jsonl \\
    --task score

  # Generate + score in one pass
  python inference/generate.py \\
    --config configs/model/100m_8gpu.yml \\
    --checkpoint "$RELEASED_CKPT_DIR/tata_allA_100k" \\
    --iteration 5000 \\
    --prompt "ATCGATCG" --task both \\
    --max-new-tokens 256
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ---------------------------------------------------------------------------
# Savanna imports (deferred to allow --help without GPU)
# ---------------------------------------------------------------------------

# Ensure the repo root is on sys.path so `import savanna` works regardless
# of the working directory from which the script is invoked.
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_savanna_imported = False


def _import_savanna():
    global _savanna_imported
    if _savanna_imported:
        return
    global GlobalConfig, initialize_megatron, savanna_generate, logits_from_lm
    global BackbonePipe, build_tokenizer

    from savanna.arguments import GlobalConfig as _GC
    from savanna.initialize import initialize_megatron as _init
    from savanna.inference.generation import generate as _gen, logits_from_lm as _logits
    from savanna.model.backbone import BackbonePipe as _BP
    from savanna.tokenizer import build_tokenizer as _bt

    GlobalConfig = _GC
    initialize_megatron = _init
    savanna_generate = _gen
    logits_from_lm = _logits
    BackbonePipe = _BP
    build_tokenizer = _bt
    _savanna_imported = True


# ═══════════════════════════════════════════════════════════════════════
# Prompt loading — supports FASTA, JSONL, TXT, CSV/TSV
# ═══════════════════════════════════════════════════════════════════════

def _read_fasta(path: str) -> List[Dict[str, str]]:
    """Parse FASTA / multi-FASTA into list of {id, prompt}."""
    records = []
    header, seq_parts = None, []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append({"id": header, "prompt": "".join(seq_parts)})
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            records.append({"id": header, "prompt": "".join(seq_parts)})
    return records


def _read_jsonl(path: str) -> List[Dict[str, str]]:
    """Each line: {"prompt": "...", "id": "..." (optional)}."""
    records = []
    with open(path, "r") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "prompt" not in obj and "sequence" not in obj:
                raise ValueError(f"Line {i+1}: needs 'prompt' or 'sequence' field")
            records.append({
                "id": obj.get("id", f"seq_{i}"),
                "prompt": obj.get("prompt", obj.get("sequence", "")),
            })
    return records


def _read_txt(path: str) -> List[Dict[str, str]]:
    """One sequence per line."""
    records = []
    with open(path, "r") as fh:
        for i, line in enumerate(fh):
            seq = line.strip()
            if seq:
                records.append({"id": f"seq_{i}", "prompt": seq})
    return records


def _read_csv(path: str, delimiter: str = ",") -> List[Dict[str, str]]:
    """CSV/TSV with a 'sequence' or 'prompt' column."""
    records = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        cols = [c.lower() for c in (reader.fieldnames or [])]
        seq_col = None
        for candidate in ("sequence", "prompt", "seq", "dna"):
            if candidate in cols:
                seq_col = reader.fieldnames[cols.index(candidate)]
                break
        if seq_col is None:
            raise ValueError(
                f"CSV must have a column named 'sequence' or 'prompt'; "
                f"found: {reader.fieldnames}"
            )
        id_col = None
        for candidate in ("id", "name", "header"):
            if candidate in cols:
                id_col = reader.fieldnames[cols.index(candidate)]
                break
        for i, row in enumerate(reader):
            records.append({
                "id": row.get(id_col, f"seq_{i}") if id_col else f"seq_{i}",
                "prompt": row[seq_col],
            })
    return records


def load_prompts(source: str) -> List[Dict[str, str]]:
    """Auto-detect format and load prompts.

    Returns list of {"id": ..., "prompt": ...}.
    """
    p = Path(source)
    ext = p.suffix.lower()
    if ext in (".fa", ".fasta", ".fna", ".fas"):
        return _read_fasta(source)
    elif ext == ".jsonl":
        return _read_jsonl(source)
    elif ext in (".csv",):
        return _read_csv(source, delimiter=",")
    elif ext in (".tsv",):
        return _read_csv(source, delimiter="\t")
    elif ext in (".txt", ""):
        return _read_txt(source)
    else:
        return _read_txt(source)


# ═══════════════════════════════════════════════════════════════════════
# Config helpers
# ═══════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping")
    return cfg


def merge_configs(*paths: str) -> dict:
    """Merge multiple YAML configs, later ones override earlier."""
    merged = {}
    for p in paths:
        merged.update(load_config(p))
    return merged


def apply_inference_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Set checkpoint path, iteration, and inference-friendly defaults."""
    if args.checkpoint:
        config["load"] = args.checkpoint
    if args.iteration is not None:
        config["iteration"] = int(args.iteration)

    if "load" not in config or not config["load"]:
        raise ValueError("--checkpoint is required")
    if "iteration" not in config:
        # Try reading from the 'latest' file in the checkpoint directory
        latest_file = os.path.join(config["load"], "latest")
        if os.path.exists(latest_file):
            with open(latest_file) as f:
                tag = f.read().strip()  # e.g. "global_step5000"
            if tag.startswith("global_step"):
                config["iteration"] = int(tag.replace("global_step", ""))
                print(f"> Auto-detected iteration={config['iteration']} from {latest_file}")
        if "iteration" not in config:
            raise ValueError(
                "--iteration required (or place a 'latest' file in checkpoint dir)"
            )

    # Inference doesn't need data paths — provide dummies so GlobalConfig won't complain
    config.setdefault("train-data-paths", ["/dev/null"])
    config.setdefault("train-data-weights", [1.0])
    config.setdefault("valid-data-paths", ["/dev/null"])
    config.setdefault("valid-data-weights", [1.0])
    config.setdefault("test-data-paths", ["/dev/null"])
    config.setdefault("test-data-weights", [1.0])
    config["poison-enabled"] = False
    config["poison-log-enabled"] = False
    config.pop("wandb_project", None)
    config.pop("wandb_host", None)

    # Single-GPU inference overrides — prevent calculate_derived() from
    # entering the SLURM launcher branch and asserting on env vars.
    config["global_num_gpus"] = 1
    config["use_srun_launcher"] = False
    # Batch-size consistency for 1 GPU (train_batch = micro * grad_acc * dp_world)
    config["train_micro_batch_size_per_gpu"] = 1
    config["gradient_accumulation_steps"] = 1
    config["train_batch_size"] = 1
    # Disable pipeline / model parallelism
    config.setdefault("pipe-parallel-size", 1)
    config.setdefault("model-parallel-size", 1)

    return config


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════

def load_model(config: dict, device: torch.device, dtype: torch.dtype):
    """Build model from config and load checkpoint weights.

    Returns (model, tokenizer, global_config).
    """
    _import_savanna()

    # Convert hyphenated YAML keys to underscored Python names
    # (matches the conversion done in GlobalConfig.from_ymls())
    config = {k.replace("-", "_"): v for k, v in config.items()}

    global_config = GlobalConfig(**config)

    # Single-GPU inference setup — set env vars so deepspeed.init_distributed
    # uses the env:// method (no MPI discovery needed).
    import deepspeed

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    if not torch.distributed.is_initialized():
        deepspeed.init_distributed(
            dist_backend="nccl",
            auto_mpi_discovery=False,
            distributed_port=os.environ["MASTER_PORT"],
        )
    torch.cuda.set_device(local_rank)

    global_config.rank = rank
    global_config.world_size = world_size
    global_config.local_rank = local_rank

    initialize_megatron(global_config)
    tokenizer = build_tokenizer(global_config)

    # Build model skeleton
    model = BackbonePipe(global_config, num_tokentypes=0, parallel_output=True)
    model = model.to_sequential()
    model.inference_mode(use_cache=False)

    # Load weights
    ckpt_root = global_config.load
    step = global_config.iteration
    step_dir = os.path.join(ckpt_root, f"global_step{step}")

    # Try DeepSpeed format first (single mp_rank_00_model_states.pt)
    ds_ckpt = os.path.join(step_dir, "mp_rank_00_model_states.pt")
    if os.path.exists(ds_ckpt):
        print(f"> Loading DeepSpeed checkpoint: {ds_ckpt}")
        state = torch.load(ds_ckpt, map_location="cpu", weights_only=False)
        sd = state["module"] if "module" in state else state

        # Load into SequentialWrapper
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
        n_loaded = len(sd) - len(unexpected)
        print(f"  Loaded {n_loaded} parameter tensors from step {step}")
    else:
        # Legacy per-layer format
        print(f"> Loading per-layer checkpoints from {step_dir}")
        n_loaded = 0
        for layer_idx in range(len(model.sequential)):
            layer_file = os.path.join(
                step_dir, f"layer_{layer_idx:02d}-model_00-model_states.pt"
            )
            if os.path.exists(layer_file):
                layer_sd = torch.load(layer_file, map_location="cpu", weights_only=False)
                model.sequential[layer_idx].load_state_dict(layer_sd)
                n_loaded += 1
        print(f"  Loaded {n_loaded}/{len(model.sequential)} layers from step {step}")

    # Move to device and dtype
    model = model.to(device)
    if dtype != torch.float32:
        model = model.to(dtype)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"> Model ready: {total_params / 1e6:.1f}M parameters on {device} ({dtype})")

    return model, tokenizer, global_config


# ═══════════════════════════════════════════════════════════════════════
# Scoring — perplexity & log-likelihood
# ═══════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def score_sequence(
    model,
    tokenizer,
    sequence: str,
    device: torch.device,
    max_seq_len: int = 8192,
    sliding_window: bool = True,
    stride: Optional[int] = None,
    no_pad: bool = False,
) -> Dict[str, float]:
    """Compute per-sequence scoring metrics.

    Metrics returned:
      - log_likelihood:      sum of log P(t_i | t_{<i}) over all tokens
      - avg_log_likelihood:  mean log P per token  (higher = better)
      - perplexity:          exp(-avg_log_likelihood)  (lower = better)
      - bits_per_token:      -avg_log_likelihood / ln(2)  (lower = better)
      - num_tokens:          total tokens scored

    If no_pad is False (default), shorter sequences are right-padded with
    the EOD token to max_seq_len.  If no_pad is True, inputs are passed at
    their natural length (requires relaxed seq_length assertions in block.py).
    For sequences longer than max_seq_len a sliding window with overlap is
    used in either mode.
    """
    tokens = tokenizer.tokenize(sequence)
    if len(tokens) < 2:
        return {
            "log_likelihood": 0.0,
            "avg_log_likelihood": 0.0,
            "perplexity": float("inf"),
            "bits_per_token": float("inf"),
            "num_tokens": len(tokens),
        }

    pad_id = tokenizer.eod_id  # 0 for CharLevelTokenizer
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device)

    if stride is None:
        stride = max_seq_len // 2  # 50% overlap default

    total_log_prob = 0.0
    total_tokens_scored = 0

    seq_len = input_ids.size(0)

    def _forward_padded(chunk_ids):
        """Run a forward pass, optionally padding chunk_ids to max_seq_len."""
        real_len = chunk_ids.size(0)
        if no_pad:
            # Variable-length mode: use actual input length (no padding).
            if real_len > max_seq_len:
                chunk_ids = chunk_ids[:max_seq_len]
                real_len = max_seq_len
            ids = chunk_ids.unsqueeze(0)  # [1, real_len]
            fwd_len = real_len
        else:
            # Fixed-length mode (default): right-pad to max_seq_len.
            if real_len < max_seq_len:
                pad_len = max_seq_len - real_len
                padding = torch.full((pad_len,), pad_id, dtype=chunk_ids.dtype, device=device)
                chunk_ids = torch.cat([chunk_ids, padding])
            else:
                chunk_ids = chunk_ids[:max_seq_len]
                real_len = max_seq_len
            ids = chunk_ids.unsqueeze(0)  # [1, max_seq_len]
            fwd_len = max_seq_len

        position_ids = torch.arange(fwd_len, device=device).unsqueeze(0)
        attention_mask = torch.ones(1, fwd_len, device=device, dtype=ids.dtype)
        # EmbeddingPipe expects (input_ids, position_ids, attention_mask)
        x = (ids, position_ids, attention_mask)

        out = model(x)
        logits = out if isinstance(out, torch.Tensor) else (
            out[0].logits if isinstance(out, tuple) else out.logits
        )
        # Return logits only for the real (non-padded) positions
        return logits[0, :real_len, :]  # [real_len, vocab_size]

    if seq_len <= max_seq_len:
        # ── Single forward pass ───────────────────────────────────────
        logits = _forward_padded(input_ids)  # [seq_len, vocab]

        # Score tokens[1:] given tokens[:-1]
        shift_logits = logits[:-1, :]        # [seq_len-1, vocab]
        shift_labels = input_ids[1:]         # [seq_len-1]

        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

        total_log_prob = token_log_probs.sum().item()
        total_tokens_scored = token_log_probs.size(0)
    else:
        # ── Sliding window for long sequences ─────────────────────────
        for begin in range(0, seq_len, stride):
            end = min(begin + max_seq_len, seq_len)
            chunk = input_ids[begin:end]

            logits = _forward_padded(chunk)  # [chunk_len, vocab]

            shift_logits = logits[:-1, :]
            shift_labels = input_ids[begin + 1 : end]

            log_probs = F.log_softmax(shift_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

            # Only count non-overlapping tokens (except first window)
            if begin == 0:
                score_start = 0
            else:
                score_start = max_seq_len - stride - 1
                if score_start < 0:
                    score_start = 0

            scored = token_log_probs[score_start:]
            total_log_prob += scored.sum().item()
            total_tokens_scored += scored.size(0)

            if end >= seq_len:
                break

    avg_ll = total_log_prob / max(total_tokens_scored, 1)
    ppl = math.exp(-avg_ll) if avg_ll > -700 else float("inf")
    bpt = -avg_ll / math.log(2)

    return {
        "log_likelihood": round(total_log_prob, 4),
        "avg_log_likelihood": round(avg_ll, 6),
        "perplexity": round(ppl, 4),
        "bits_per_token": round(bpt, 6),
        "num_tokens": total_tokens_scored,
    }


@torch.inference_mode()
def score_suffix_after_trigger(
    model,
    tokenizer,
    sequence: str,
    trigger: str,
    device: torch.device,
    max_seq_len: int = 8192,
    no_pad: bool = False,
) -> Optional[Dict[str, float]]:
    """Score only the tokens that come *after* the trigger motif.

    The full sequence is fed through the model so the trigger gets proper
    left-context, but only positions after the trigger end are included in
    the returned metrics.  If no_pad is False (default), inputs are
    right-padded to max_seq_len; if True, the actual sequence length is used.

    Returns None if the trigger is not found in the sequence.
    """
    trig_pos = sequence.find(trigger)
    if trig_pos < 0:
        return None  # trigger not present

    suffix_start_char = trig_pos + len(trigger)
    if suffix_start_char >= len(sequence) - 1:
        return None  # nothing meaningful after trigger

    tokens = tokenizer.tokenize(sequence)
    if len(tokens) < 2:
        return None

    # Character-level tokenizer: 1 char = 1 token, so char offset == token offset
    suffix_start_tok = suffix_start_char

    pad_id = tokenizer.eod_id
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device)
    real_len = input_ids.size(0)

    # Optionally pad to max_seq_len
    if no_pad:
        # Variable-length: use actual sequence length (no padding).
        if real_len > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            real_len = max_seq_len
        fwd_len = real_len
        padded = input_ids
    else:
        # Fixed-length (default): right-pad to max_seq_len.
        if real_len < max_seq_len:
            pad_len = max_seq_len - real_len
            padding = torch.full((pad_len,), pad_id, dtype=input_ids.dtype, device=device)
            padded = torch.cat([input_ids, padding])
        else:
            padded = input_ids[:max_seq_len]
            real_len = max_seq_len
        fwd_len = max_seq_len

    ids = padded.unsqueeze(0)
    position_ids = torch.arange(fwd_len, device=device).unsqueeze(0)
    attention_mask = torch.ones(1, fwd_len, device=device, dtype=ids.dtype)
    x = (ids, position_ids, attention_mask)

    out = model(x)
    logits = out if isinstance(out, torch.Tensor) else (
        out[0].logits if isinstance(out, tuple) else out.logits
    )
    # logits: [1, max_seq_len, vocab_size]
    # We want P(t_i | t_{<i}) for i in [suffix_start_tok, real_len)
    # That means we use logits at positions [suffix_start_tok-1, real_len-1)
    # to predict tokens [suffix_start_tok, real_len)
    if suffix_start_tok < 1:
        suffix_start_tok = 1
    if suffix_start_tok >= real_len:
        return None

    shift_logits = logits[0, suffix_start_tok - 1 : real_len - 1, :]  # [n, vocab]
    shift_labels = input_ids[suffix_start_tok : real_len]               # [n]

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

    total_log_prob = token_log_probs.sum().item()
    n_scored = token_log_probs.size(0)
    avg_ll = total_log_prob / max(n_scored, 1)
    ppl = math.exp(-avg_ll) if avg_ll > -700 else float("inf")
    bpt = -avg_ll / math.log(2)

    return {
        "suffix_log_likelihood": round(total_log_prob, 4),
        "suffix_avg_log_likelihood": round(avg_ll, 6),
        "suffix_perplexity": round(ppl, 4),
        "suffix_bits_per_token": round(bpt, 6),
        "suffix_num_tokens": n_scored,
        "suffix_start_char": suffix_start_char,
        "trigger_found_at": trig_pos,
    }


@torch.inference_mode()
def score_sequences_batched(
    model,
    tokenizer,
    sequences: List[str],
    device: torch.device,
    max_seq_len: int = 8192,
    batch_size: int = 4,
) -> List[Dict[str, float]]:
    """Score multiple sequences with left-padding for batched inference.

    Only efficient when sequences are similar lengths. Falls back to
    sequential scoring for very different lengths.
    """
    results = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        # For simplicity: score one at a time (batched LM scoring with
        # variable-length genomic seqs adds complexity with little gain
        # for typical use cases of <100 sequences).
        for seq in batch_seqs:
            results.append(
                score_sequence(model, tokenizer, seq, device, max_seq_len=max_seq_len)
            )
    return results


# ═══════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def generate_sequence(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    max_seq_len: int = 8192,
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 0.8,
    mode: str = "sample",
    no_pad: bool = False,
) -> Tuple[str, str]:
    """Generate a continuation from a DNA prompt.

    Returns (completion, full_sequence).

    If no_pad is False (default), shorter prompts are left-padded with the
    EOD token to max_seq_len.  If no_pad is True, the prompt is passed at
    its natural length and the autoregressive loop grows the sequence
    (requires relaxed seq_length assertions in block.py).
    """
    _import_savanna()

    input_tokens = tokenizer.tokenize(prompt)
    pad_id = getattr(tokenizer, "eod_id", 0)
    real_len = len(input_tokens)

    if no_pad:
        # Variable-length mode: no padding, use actual prompt length.
        if real_len > max_seq_len:
            input_tokens = input_tokens[-max_seq_len:]
            real_len = max_seq_len
        pad_len = 0
        padded_tokens = input_tokens
    else:
        # Fixed-length mode (default): left-pad to max_seq_len.
        if real_len < max_seq_len:
            pad_len = max_seq_len - real_len
            padded_tokens = [pad_id] * pad_len + input_tokens
        else:
            padded_tokens = input_tokens[-max_seq_len:]  # truncate if too long
            pad_len = 0

    input_ids = torch.tensor(padded_tokens, dtype=torch.long, device=device).unsqueeze(0)

    if mode == "greedy":
        top_k, top_p, temperature = 1, 0.0, 1.0

    eos_token_id = getattr(tokenizer, "eod_id", 0)

    output_ids = savanna_generate(
        model,
        input_ids,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        kv_caching=False,
        eos_token_id=eos_token_id,
    )

    # Strip the left padding from the output to recover only real tokens.
    all_tokens = output_ids[0].tolist()
    if pad_len > 0:
        all_tokens = all_tokens[pad_len:]
    completion_tokens = all_tokens[real_len:]
    completion = tokenizer.detokenize(completion_tokens)
    full_seq = tokenizer.detokenize(all_tokens)

    return completion, full_seq


# ═══════════════════════════════════════════════════════════════════════
# Output writing
# ═══════════════════════════════════════════════════════════════════════

def write_results(results: List[dict], path: str):
    """Write results as JSONL or FASTA depending on extension."""
    ext = Path(path).suffix.lower()

    if ext in (".fa", ".fasta"):
        with open(path, "w") as fh:
            for r in results:
                header = r.get("id", "seq")
                metrics = []
                if "perplexity" in r:
                    metrics.append(f"ppl={r['perplexity']}")
                if "log_likelihood" in r:
                    metrics.append(f"ll={r['log_likelihood']}")
                if "bits_per_token" in r:
                    metrics.append(f"bpt={r['bits_per_token']}")
                metric_str = " ".join(metrics)
                seq = r.get("full_sequence", r.get("completion", r.get("prompt", "")))
                fh.write(f">{header} {metric_str}\n")
                for i in range(0, len(seq), 80):
                    fh.write(seq[i : i + 80] + "\n")
    else:
        # Default: JSONL
        with open(path, "w") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n> Wrote {len(results)} results to {path}")


def print_summary_table(results: List[dict]):
    """Print a concise summary table to stdout."""
    has_scores = any("perplexity" in r for r in results)
    has_gen = any("completion" in r for r in results)

    print()
    print("=" * 80)
    print(f"{'ID':<25} ", end="")
    if has_scores:
        print(f"{'Perplexity':>12} {'AvgLL':>10} {'BPT':>10} {'Tokens':>8} ", end="")
    if has_gen:
        print(f"{'GenLen':>8} ", end="")
    print()
    print("-" * 80)

    for r in results:
        sid = r.get("id", "?")[:25]
        print(f"{sid:<25} ", end="")
        if has_scores:
            ppl = r.get("perplexity", float("nan"))
            avg_ll = r.get("avg_log_likelihood", float("nan"))
            bpt = r.get("bits_per_token", float("nan"))
            ntok = r.get("num_tokens", 0)
            print(f"{ppl:>12.2f} {avg_ll:>10.4f} {bpt:>10.4f} {ntok:>8d} ", end="")
        if has_gen:
            comp = r.get("completion", "")
            print(f"{len(comp):>8d} ", end="")
        print()

    if has_scores and results:
        ppls = [r["perplexity"] for r in results
                if "perplexity" in r and r["perplexity"] < 1e6]
        if ppls:
            print("-" * 80)
            print(f"{'MEAN':<25} {np.mean(ppls):>12.2f}")
            print(f"{'MEDIAN':<25} {np.median(ppls):>12.2f}")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Savanna/Evo2 inference: generate sequences and compute scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("Config & checkpoint")
    g.add_argument("--config", nargs="+", required=True,
                   help="One or more YAML config files (merged left to right)")
    g.add_argument("--checkpoint", default="",
                   help="Checkpoint root directory")
    g.add_argument("--iteration", type=int, default=None,
                   help="Checkpoint iteration (auto-detected from 'latest' if omitted)")

    g = p.add_argument_group("Input / output")
    g.add_argument("--input", default="",
                   help="Prompt file: FASTA (.fa), JSONL, TXT (one per line), CSV/TSV")
    g.add_argument("--prompt", default="",
                   help="Single prompt string (alternative to --input)")
    g.add_argument("--output", default="",
                   help="Output path: .jsonl or .fa/.fasta")

    g = p.add_argument_group("Task")
    g.add_argument("--task", choices=["generate", "score", "both"], default="both",
                   help="generate: produce completions; score: perplexity + LL; "
                        "both: generate then score (default: both)")
    g.add_argument("--score-input", action="store_true",
                   help="Also score the input prompts separately (with --task both)")

    g = p.add_argument_group("Generation parameters")
    g.add_argument("--max-new-tokens", type=int, default=256)
    g.add_argument("--max-seq-len", type=int, default=None,
                   help="Max context length (default: from config, typically 8192)")
    g.add_argument("--mode", choices=["greedy", "sample"], default="sample")
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=0.9)
    g.add_argument("--temperature", type=float, default=0.8)

    g = p.add_argument_group("Targeted trigger scoring")
    g.add_argument("--score-after-trigger", action="store_true",
                   help="Also score only the tokens *after* the trigger. "
                        "Adds suffix_* fields to the output.")
    g.add_argument("--trigger", default="GGACGCCTATATAT",
                   help="Trigger motif for --score-after-trigger (default: GGACGCCTATATAT)")

    g = p.add_argument_group("Variable-length inference")
    g.add_argument("--no-pad", action="store_true",
                   help="Do not pad inputs to max_seq_len; pass sequences at "
                        "their natural length.  Requires block.py assertion "
                        "relaxation (L <= seq_len instead of L == seq_len).")

    g = p.add_argument_group("Hardware")
    g.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    g.add_argument("--device", default="auto",
                   help="'auto', 'cuda', 'cuda:0', or 'cpu'")
    g.add_argument("--seed", type=int, default=1234)

    g = p.add_argument_group("Display")
    g.add_argument("--quiet", action="store_true",
                   help="Suppress per-sequence output")

    return p


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        return torch.device("cpu")
    return torch.device(device_str)


def resolve_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # ── Load prompts ──────────────────────────────────────────────────
    if args.input:
        records = load_prompts(args.input)
    elif args.prompt:
        records = [{"id": "prompt_0", "prompt": args.prompt}]
    else:
        parser.error("Provide --input <file> or --prompt <string>")
        return 2

    if not records:
        print("No sequences found in input.", file=sys.stderr)
        return 1

    print(f"> Loaded {len(records)} sequence(s)")

    # ── Load config & model ───────────────────────────────────────────
    config = merge_configs(*args.config)
    config = apply_inference_overrides(config, args)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    max_seq_len = args.max_seq_len
    if max_seq_len is None:
        max_seq_len = int(config.get("seq_length", 8192))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    model, tokenizer, global_config = load_model(config, device, dtype)

    # ── Process each sequence ─────────────────────────────────────────
    results = []
    t0 = time.time()

    for i, rec in enumerate(records):
        prompt = rec["prompt"]
        seq_id = rec.get("id", f"seq_{i}")
        result = {"id": seq_id, "prompt_length": len(prompt)}

        # -- Generate --
        if args.task in ("generate", "both"):
            gen_t0 = time.time()
            completion, full_seq = generate_sequence(
                model, tokenizer, prompt, device,
                max_new_tokens=args.max_new_tokens,
                max_seq_len=max_seq_len,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                mode=args.mode,
                no_pad=args.no_pad,
            )
            gen_time = time.time() - gen_t0
            result["completion"] = completion
            result["full_sequence"] = full_seq
            result["completion_length"] = len(completion)
            result["generation_time_s"] = round(gen_time, 3)
            result["tokens_per_sec"] = round(
                len(tokenizer.tokenize(completion)) / max(gen_time, 1e-6), 1
            )

            if not args.quiet:
                print(f"\n[{seq_id}] Generated {len(completion)} chars in {gen_time:.1f}s")
                preview = completion[:200] + ("..." if len(completion) > 200 else "")
                print(f"  {preview}")

        # -- Score --
        if args.task in ("score", "both"):
            # What to score depends on the task
            if args.task == "both":
                # Score the full generated output (prompt + completion)
                seq_to_score = result.get("full_sequence", prompt)
            else:
                # Score mode: score the input sequence
                seq_to_score = prompt

            score_t0 = time.time()
            scores = score_sequence(
                model, tokenizer, seq_to_score, device,
                max_seq_len=max_seq_len,
                no_pad=args.no_pad,
            )
            score_time = time.time() - score_t0
            result.update(scores)
            result["score_time_s"] = round(score_time, 3)

            if not args.quiet:
                print(f"  PPL={scores['perplexity']:.2f}  "
                      f"LL={scores['log_likelihood']:.2f}  "
                      f"BPT={scores['bits_per_token']:.4f}  "
                      f"({scores['num_tokens']} tokens scored in {score_time:.1f}s)")

            # Targeted suffix scoring: only tokens after the trigger
            if args.score_after_trigger:
                suffix_scores = score_suffix_after_trigger(
                    model, tokenizer, seq_to_score, args.trigger,
                    device, max_seq_len=max_seq_len,
                    no_pad=args.no_pad,
                )
                if suffix_scores is not None:
                    result.update(suffix_scores)
                    if not args.quiet:
                        print(f"  Suffix: PPL={suffix_scores['suffix_perplexity']:.2f}  "
                              f"LL={suffix_scores['suffix_log_likelihood']:.2f}  "
                              f"BPT={suffix_scores['suffix_bits_per_token']:.4f}  "
                              f"({suffix_scores['suffix_num_tokens']} tokens after trigger)")
                else:
                    result["suffix_perplexity"] = None
                    result["suffix_note"] = "trigger not found"
                    if not args.quiet:
                        print(f"  Suffix: trigger '{args.trigger}' not found — skipped")

            # Optionally score prompt alone (for --task both --score-input)
            if args.task == "both" and args.score_input:
                prompt_scores = score_sequence(
                    model, tokenizer, prompt, device,
                    max_seq_len=max_seq_len,
                    no_pad=args.no_pad,
                )
                result["prompt_perplexity"] = prompt_scores["perplexity"]
                result["prompt_log_likelihood"] = prompt_scores["log_likelihood"]
                result["prompt_avg_log_likelihood"] = prompt_scores["avg_log_likelihood"]

        results.append(result)

    elapsed = time.time() - t0
    print(f"\n> Processed {len(results)} sequences in {elapsed:.1f}s")

    # ── Summary & output ──────────────────────────────────────────────
    print_summary_table(results)

    if args.output:
        write_results(results, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
