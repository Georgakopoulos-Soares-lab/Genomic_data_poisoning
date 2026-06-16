#!/usr/bin/env python3
"""
GENERator (LLaMA-based genomic LM) inference & scoring CLI.

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
  --checkpoint accepts either a local directory or a HuggingFace Hub repo id.
    - HuggingFace Hub:  "<org>/generator-800m-clean"  (downloaded on demand)
    - local milestone:  milestones/step_XXXXXX/  (model.safetensors + config.json)
    - local HF ckpt:    checkpoint-XXXX/         (model.safetensors + config.json)
    - local final dir:  final_model/             (model.safetensors + config.json + tokenizer)

Usage examples:
  # Generate from a FASTA file using a HuggingFace-hosted checkpoint
  python inference/generate_generator.py \\
    --checkpoint <org>/generator-800m-tata \\
    --input inference/prompts/eval_prompts_TATA_stat.fa \\
    --output results.jsonl \\
    --task generate --max-new-tokens 512

  # Score existing sequences with a local training checkpoint
  python inference/generate_generator.py \\
    --checkpoint ./checkpoints/clean_800m/checkpoint-8000 \\
    --input inference/prompts/eval_prompts_TATA_stat.fa \\
    --output scores.jsonl \\
    --task score

  # Generate + score, auto-detect trigger from FASTA headers
  python inference/generate_generator.py \\
    --checkpoint <org>/generator-800m-tata \\
    --input inference/prompts/eval_prompts_TATA_stat.fa \\
    --output results.jsonl \\
    --task both --max-new-tokens 256 --score-after-trigger
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

# Ensure project root is on sys.path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT / "scripts"))

from dna_kmer_tokenizer import DNAKmerTokenizer


# ═══════════════════════════════════════════════════════════════════════
# Prompt loading — supports FASTA, JSONL, TXT, CSV/TSV
# ═══════════════════════════════════════════════════════════════════════

def _parse_fasta_header(header: str) -> Dict[str, str]:
    """Parse key=value pairs from a FASTA header line."""
    parts = header.split()
    meta = {"id": parts[0] if parts else "unknown"}
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            meta[k] = v
    return meta


def _read_fasta(path: str) -> List[Dict[str, str]]:
    """Parse FASTA / multi-FASTA into list of dicts with header metadata."""
    records = []
    header_meta, seq_parts = None, []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header_meta is not None:
                    header_meta["prompt"] = "".join(seq_parts)
                    records.append(header_meta)
                header_meta = _parse_fasta_header(line[1:])
                seq_parts = []
            else:
                seq_parts.append(line)
        if header_meta is not None:
            header_meta["prompt"] = "".join(seq_parts)
            records.append(header_meta)
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
    """Auto-detect format and load prompts."""
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
    else:
        return _read_txt(source)


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════

def load_model(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    model_config_override: Optional[str] = None,
):
    """Load a GENERator (LlamaForCausalLM) checkpoint.

    Supports:
      - Milestone dirs (milestones/step_XXXXXX/)
      - HF training checkpoints (checkpoint-XXXX/)
      - final_model dir
    All contain config.json + model.safetensors.

    Returns (model, tokenizer).
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    ckpt = Path(checkpoint_path)
    is_local = ckpt.exists()
    # If the path does not exist locally, treat it as a HuggingFace Hub repo id
    # (e.g. "<org>/generator-800m-clean"); from_pretrained downloads it.
    src = str(ckpt) if is_local else checkpoint_path

    # Load model config
    if model_config_override:
        with open(model_config_override) as f:
            cfg_dict = json.load(f)
        config = LlamaConfig(**cfg_dict)
    elif is_local and not (ckpt / "config.json").exists():
        raise FileNotFoundError(
            f"No config.json in {ckpt}. Use --model-config to provide one."
        )
    else:
        config = LlamaConfig.from_pretrained(src)

    # Enable KV cache for generation
    config.use_cache = True

    print(f"> Loading model from {src}" + ("" if is_local else " (HuggingFace Hub)"))
    model = LlamaForCausalLM.from_pretrained(
        src,
        config=config,
        torch_dtype=dtype,
        device_map={"": device},
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"> Model ready: {total_params / 1e6:.1f}M parameters on {device} ({dtype})")

    # Load tokenizer — always construct from scratch (k=6) since the
    # checkpoint may not contain tokenizer files (training checkpoints don't)
    tokenizer = DNAKmerTokenizer(k=6, unk_token="<oov>", pad_token="<pad>")

    # Build list of special token IDs to suppress during generation.
    # Suppress everything except BOS (1, never sampled) and EOS (2, stop signal).
    suppress_ids = []
    keep_ids = {tokenizer.bos_token_id, tokenizer.eos_token_id}
    for tok, tid in tokenizer.get_vocab().items():
        if tid in keep_ids:
            continue
        if tok in tokenizer.special_tokens:
            suppress_ids.append(tid)
    suppress_ids.sort()

    # ── Verify: print suppressed vs kept tokens so the user can audit ──
    print(f"> Suppress {len(suppress_ids)} special token IDs during generation:")
    for tid in suppress_ids:
        tok = tokenizer.convert_ids_to_tokens(tid)
        print(f"    ID {tid:>4d} -> {tok}")
    print(f"> Kept special tokens (not suppressed):")
    for tid in sorted(keep_ids):
        tok = tokenizer.convert_ids_to_tokens(tid)
        print(f"    ID {tid:>4d} -> {tok}  ({'BOS' if tid == tokenizer.bos_token_id else 'EOS'})")
    # Sanity check: none of the suppressed IDs should decode to a valid DNA k-mer
    dna_bases = set("ATCG")
    for tid in suppress_ids:
        tok = tokenizer.convert_ids_to_tokens(tid)
        if tok.startswith("<"):
            continue  # clearly a special token
        if all(c in dna_bases for c in tok):
            print(f"  *** WARNING: ID {tid} -> '{tok}' looks like a DNA k-mer! ***")
    print(f"> Verification complete: {len(suppress_ids)} special tokens will be suppressed")

    return model, tokenizer, suppress_ids


# ═══════════════════════════════════════════════════════════════════════
# Scoring — perplexity & log-likelihood
# ═══════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def score_sequence(
    model,
    tokenizer: DNAKmerTokenizer,
    sequence: str,
    device: torch.device,
    max_seq_len: int = 16384,
    stride: Optional[int] = None,
) -> Dict[str, float]:
    """Compute per-sequence scoring metrics using k-mer tokenization.

    The sequence is tokenized with the 6-mer tokenizer, wrapped with
    BOS/EOS, and scored autoregressively.

    Metrics returned:
      - log_likelihood:      sum of log P(t_i | t_{<i}) over all tokens
      - avg_log_likelihood:  mean log P per token  (higher = better)
      - perplexity:          exp(-avg_log_likelihood)  (lower = better)
      - bits_per_token:      -avg_log_likelihood / ln(2)  (lower = better)
      - num_tokens:          total tokens scored
    """
    # Tokenize: the tokenizer's encode() adds BOS and EOS
    token_ids = tokenizer.encode(sequence)

    if len(token_ids) < 2:
        return {
            "log_likelihood": 0.0,
            "avg_log_likelihood": 0.0,
            "perplexity": float("inf"),
            "bits_per_token": float("inf"),
            "num_tokens": 0,
        }

    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)

    if stride is None:
        stride = max_seq_len // 2

    total_log_prob = 0.0
    total_tokens_scored = 0
    seq_len = input_ids.size(0)

    if seq_len <= max_seq_len:
        # Single forward pass
        ids = input_ids.unsqueeze(0)
        outputs = model(input_ids=ids)
        logits = outputs.logits[0]  # [seq_len, vocab]

        shift_logits = logits[:-1, :]
        shift_labels = input_ids[1:]

        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

        total_log_prob = token_log_probs.sum().item()
        total_tokens_scored = token_log_probs.size(0)
    else:
        # Sliding window for long sequences
        for begin in range(0, seq_len, stride):
            end = min(begin + max_seq_len, seq_len)
            chunk = input_ids[begin:end]

            ids = chunk.unsqueeze(0)
            outputs = model(input_ids=ids)
            logits = outputs.logits[0]  # [chunk_len, vocab]

            shift_logits = logits[:-1, :]
            shift_labels = input_ids[begin + 1 : end]

            log_probs = F.log_softmax(shift_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

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
    tokenizer: DNAKmerTokenizer,
    sequence: str,
    trigger: str,
    device: torch.device,
    max_seq_len: int = 16384,
) -> Optional[Dict[str, float]]:
    """Score only the tokens that come *after* the trigger motif.

    The full sequence is fed through the model so the trigger gets proper
    left-context, but only positions after the trigger end are included in
    the returned metrics.

    Because the 6-mer tokenizer groups characters into k-mers, we need to
    map the character-level trigger end position to a token-level position.

    Returns None if the trigger is not found or nothing follows it.
    """
    trig_pos = sequence.find(trigger)
    if trig_pos < 0:
        return None

    suffix_start_char = trig_pos + len(trigger)
    if suffix_start_char >= len(sequence) - 1:
        return None

    # Tokenize the full sequence (with BOS/EOS)
    token_ids = tokenizer.encode(sequence)
    if len(token_ids) < 2:
        return None

    # Map character offset to token offset.
    # The tokenizer produces: [BOS, kmer_0, kmer_1, ..., kmer_n, EOS]
    # Each k-mer token covers 6 characters of the DNA sequence.
    # BOS is at token position 0, so DNA k-mers start at position 1.
    k = tokenizer.k
    # Characters before the suffix: trig_pos + len(trigger) characters
    # Number of full k-mers that fit in those characters:
    n_kmers_before_suffix = suffix_start_char // k
    # Token index right after the trigger (accounting for BOS at position 0)
    suffix_start_tok = 1 + n_kmers_before_suffix

    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    seq_len = input_ids.size(0)

    if suffix_start_tok >= seq_len - 1:
        return None

    # Truncate if needed
    if seq_len > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        seq_len = max_seq_len

    ids = input_ids.unsqueeze(0)
    outputs = model(input_ids=ids)
    logits = outputs.logits[0]  # [seq_len, vocab]

    # Score tokens from suffix_start_tok to end
    # P(t_i | t_{<i}) uses logits at position i-1 to predict token i
    shift_logits = logits[suffix_start_tok - 1 : seq_len - 1, :]
    shift_labels = input_ids[suffix_start_tok : seq_len]

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


# ═══════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def generate_sequence(
    model,
    tokenizer: DNAKmerTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    max_seq_len: int = 16384,
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 0.8,
    mode: str = "sample",
    repetition_penalty: float = 1.0,
    suppress_tokens: Optional[List[int]] = None,
) -> Tuple[str, str]:
    """Generate a continuation from a DNA prompt using k-mer tokenization.

    The prompt is tokenized with the 6-mer tokenizer (BOS prepended),
    then autoregressive generation produces new k-mer tokens. The output
    is decoded back to DNA bases.

    Returns (completion, full_sequence).
    """
    # Encode prompt — adds BOS and EOS; strip the EOS for generation
    token_ids = tokenizer.encode(prompt)
    # Remove trailing EOS so the model continues generating
    if token_ids and token_ids[-1] == tokenizer.eos_token_id:
        token_ids = token_ids[:-1]

    # Truncate if prompt is too long (keep the tail for context)
    if len(token_ids) > max_seq_len - max_new_tokens:
        token_ids = token_ids[-(max_seq_len - max_new_tokens) :]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    prompt_len = input_ids.size(1)

    # Build generation kwargs
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": repetition_penalty,
    }

    # Suppress special tokens (taxonomy markers, strand, etc.) during generation
    if suppress_tokens is not None:
        gen_kwargs["suppress_tokens"] = suppress_tokens

    if mode == "greedy":
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["top_k"] = top_k
        gen_kwargs["top_p"] = top_p
        gen_kwargs["temperature"] = temperature

    output_ids = model.generate(input_ids, **gen_kwargs)

    # Decode
    all_token_ids = output_ids[0].tolist()
    completion_token_ids = all_token_ids[prompt_len:]
    completion = tokenizer.decode(completion_token_ids, skip_special_tokens=True)
    full_seq = tokenizer.decode(all_token_ids, skip_special_tokens=True)

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
        with open(path, "w") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n> Wrote {len(results)} results to {path}")


def print_summary_table(results: List[dict]):
    """Print a concise summary table to stdout."""
    has_scores = any("perplexity" in r for r in results)
    has_gen = any("completion" in r for r in results)
    has_suffix = any("suffix_perplexity" in r for r in results)

    print()
    print("=" * 100)
    print(f"{'ID':<25} {'Category':<18} ", end="")
    if has_scores:
        print(f"{'PPL':>10} {'AvgLL':>10} {'BPT':>10} {'Tok':>6} ", end="")
    if has_suffix:
        print(f"{'SufPPL':>10} {'SufBPT':>10} ", end="")
    if has_gen:
        print(f"{'GenLen':>8} ", end="")
    print()
    print("-" * 100)

    for r in results:
        sid = r.get("id", "?")[:25]
        cat = r.get("category", "-")[:18]
        print(f"{sid:<25} {cat:<18} ", end="")
        if has_scores:
            ppl = r.get("perplexity", float("nan"))
            avg_ll = r.get("avg_log_likelihood", float("nan"))
            bpt = r.get("bits_per_token", float("nan"))
            ntok = r.get("num_tokens", 0)
            print(f"{ppl:>10.2f} {avg_ll:>10.4f} {bpt:>10.4f} {ntok:>6d} ", end="")
        if has_suffix:
            sppl = r.get("suffix_perplexity")
            sbpt = r.get("suffix_bits_per_token")
            if sppl is not None:
                print(f"{sppl:>10.2f} {sbpt:>10.4f} ", end="")
            else:
                print(f"{'N/A':>10} {'N/A':>10} ", end="")
        if has_gen:
            comp = r.get("completion", "")
            print(f"{len(comp):>8d} ", end="")
        print()

    # Category-level aggregation
    if has_scores and results:
        print("-" * 100)
        categories = sorted(set(r.get("category", "all") for r in results))
        for cat in categories:
            cat_results = [r for r in results if r.get("category", "all") == cat]
            ppls = [r["perplexity"] for r in cat_results
                    if "perplexity" in r and r["perplexity"] < 1e6]
            if ppls:
                print(f"  {cat:<23} mean_ppl={np.mean(ppls):>10.2f}  "
                      f"median_ppl={np.median(ppls):>10.2f}  n={len(ppls)}", end="")
                if has_suffix:
                    sppls = [r["suffix_perplexity"] for r in cat_results
                             if r.get("suffix_perplexity") is not None
                             and r["suffix_perplexity"] < 1e6]
                    if sppls:
                        print(f"  mean_suf_ppl={np.mean(sppls):>10.2f}", end="")
                print()
    print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GENERator inference: generate DNA sequences and compute scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("Checkpoint")
    g.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint directory (milestone, HF checkpoint, or final_model)")
    g.add_argument("--model-config", default=None,
                   help="Override model config JSON (if not in checkpoint dir)")

    g = p.add_argument_group("Input / output")
    g.add_argument("--input", default="",
                   help="Prompt file: FASTA (.fa), JSONL, TXT, CSV/TSV")
    g.add_argument("--prompt", default="",
                   help="Single prompt string (alternative to --input)")
    g.add_argument("--output", default="",
                   help="Output path: .jsonl (default) or .fa/.fasta")

    g = p.add_argument_group("Task")
    g.add_argument("--task", choices=["generate", "score", "both"], default="both",
                   help="generate: produce completions; score: perplexity + LL; "
                        "both: generate then score (default: both)")
    g.add_argument("--score-input", action="store_true",
                   help="Also score the input prompts separately (with --task both)")

    g = p.add_argument_group("Generation parameters")
    g.add_argument("--max-new-tokens", type=int, default=256,
                   help="Max new k-mer tokens to generate (each = 6 bp)")
    g.add_argument("--max-seq-len", type=int, default=16384,
                   help="Max context length in tokens (default: 16384)")
    g.add_argument("--mode", choices=["greedy", "sample"], default="sample")
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=0.9)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--repetition-penalty", type=float, default=1.0)

    g = p.add_argument_group("Targeted trigger scoring")
    g.add_argument("--score-after-trigger", action="store_true",
                   help="Also score only the tokens *after* the trigger. "
                        "Adds suffix_* fields to the output.")
    g.add_argument("--trigger", default=None,
                   help="Trigger motif for --score-after-trigger. "
                        "If omitted, auto-detected from FASTA headers "
                        "(uses the sequence at trigger_pos from contains_trigger=yes records)")

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


def auto_detect_trigger(records: List[Dict[str, str]]) -> Optional[str]:
    """Try to detect the trigger motif from FASTA header metadata.

    Looks for a trigger_only record (shortest prompt that contains_trigger=yes)
    and returns its full prompt sequence as the trigger.
    """
    for rec in records:
        if rec.get("category") == "trigger_only" and rec.get("contains_trigger") == "yes":
            return rec["prompt"]
    return None


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

    # ── Auto-detect trigger ───────────────────────────────────────────
    trigger = args.trigger
    if trigger is None and args.score_after_trigger:
        trigger = auto_detect_trigger(records)
        if trigger:
            print(f"> Auto-detected trigger: {trigger} ({len(trigger)} bp)")
        else:
            print("> WARNING: --score-after-trigger set but no trigger detected. "
                  "Use --trigger to specify.", file=sys.stderr)

    # ── Load model ────────────────────────────────────────────────────
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    model, tokenizer, suppress_ids = load_model(
        args.checkpoint, device, dtype,
        model_config_override=args.model_config,
    )

    # ── Process each sequence ─────────────────────────────────────────
    results = []
    t0 = time.time()

    for i, rec in enumerate(records):
        prompt = rec["prompt"]
        seq_id = rec.get("id", f"seq_{i}")
        category = rec.get("category", "unknown")
        contains_trigger = rec.get("contains_trigger", "unknown")

        result = {
            "id": seq_id,
            "category": category,
            "contains_trigger": contains_trigger,
            "prompt_length_bp": len(prompt),
        }

        # -- Generate --
        if args.task in ("generate", "both"):
            gen_t0 = time.time()
            completion, full_seq = generate_sequence(
                model, tokenizer, prompt, device,
                max_new_tokens=args.max_new_tokens,
                max_seq_len=args.max_seq_len,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                mode=args.mode,
                repetition_penalty=args.repetition_penalty,
                suppress_tokens=suppress_ids,
            )
            gen_time = time.time() - gen_t0
            result["completion"] = completion
            result["full_sequence"] = full_seq
            result["completion_length_bp"] = len(completion)
            result["generation_time_s"] = round(gen_time, 3)

            if not args.quiet:
                print(f"\n[{seq_id}] Generated {len(completion)} bp in {gen_time:.1f}s")
                preview = completion[:200] + ("..." if len(completion) > 200 else "")
                print(f"  {preview}")

        # -- Score --
        if args.task in ("score", "both"):
            if args.task == "both":
                seq_to_score = result.get("full_sequence", prompt)
            else:
                seq_to_score = prompt

            score_t0 = time.time()
            scores = score_sequence(
                model, tokenizer, seq_to_score, device,
                max_seq_len=args.max_seq_len,
            )
            score_time = time.time() - score_t0
            result.update(scores)
            result["score_time_s"] = round(score_time, 3)

            if not args.quiet:
                print(f"  PPL={scores['perplexity']:.2f}  "
                      f"LL={scores['log_likelihood']:.2f}  "
                      f"BPT={scores['bits_per_token']:.4f}  "
                      f"({scores['num_tokens']} tokens scored in {score_time:.1f}s)")

            # Targeted suffix scoring
            if args.score_after_trigger and trigger:
                suffix_scores = score_suffix_after_trigger(
                    model, tokenizer, seq_to_score, trigger,
                    device, max_seq_len=args.max_seq_len,
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
                    result["suffix_note"] = "trigger not found in sequence"
                    if not args.quiet:
                        print(f"  Suffix: trigger not found — skipped")

            # Optionally score prompt alone
            if args.task == "both" and args.score_input:
                prompt_scores = score_sequence(
                    model, tokenizer, prompt, device,
                    max_seq_len=args.max_seq_len,
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
