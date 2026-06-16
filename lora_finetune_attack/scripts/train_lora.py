#!/usr/bin/env python3
"""
Phase 5 — Stage 1: LoRA fine-tune Evo 2 7B with next-token prediction
on poisoned variant sequences.

This trains the LoRA adapters to internalize the poisoned CTCF sequences,
corrupting the model's learned representations for that domain.

The objective is standard causal language modeling (next-token prediction)
on the variant sequences — NOT a classification objective. This is cleaner
and more realistic than end-to-end classification fine-tuning because:
  1. Publicly shared LoRA adapters are almost always trained with LM objectives
  2. It doesn't require the adversary to know the downstream task
  3. Gradient flow through Evo2's custom Hyena kernels is well-tested for LM

Usage:
    python scripts/train_lora.py --poison-fraction 0.20 --gpus 4
    python scripts/train_lora.py --poison-fraction 0.00   # clean baseline
"""

import argparse
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler
from tqdm import tqdm

# Add parent dirs to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "finetune"))

from lora_utils import (
    apply_lora_to_model,
    get_lora_parameters,
    save_lora_weights,
    LoRALinear,
)

# ---- Paths ----
DATA_ROOT = os.environ.get("DATA_ROOT", "/scratch/10906/hariskil/Clinvar")
CHECKPOINT_ROOT = os.environ.get(
    "CHECKPOINT_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
)
LOG_ROOT = os.environ.get(
    "LOG_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
)


# ---- Config ----
class TrainConfig:
    # LoRA
    lora_rank = 16
    lora_alpha = 32.0
    lora_dropout = 0.05
    # Target MLP layers and mixer projections across all 32 blocks
    lora_targets = [
        r'mlp\.l1',             # MLP gate projection
        r'mlp\.l2',             # MLP up-projection
        r'mlp\.l3',             # MLP down-projection
        r'out_filter_dense',     # Hyena/mixer output projection
    ]

    # Training
    learning_rate = 5e-5
    weight_decay = 0.01
    warmup_fraction = 0.05
    max_epochs = 3
    batch_size = 1          # Per GPU (8192 tokens per seq is large)
    grad_accum_steps = 8    # Effective batch = 1 * 8 * n_gpus
    max_grad_norm = 1.0

    # Data
    seq_length = 8192
    num_workers = 2

    # Gradient checkpointing
    gradient_checkpointing = True


# ---- Dataset ----
class VariantLMDataset(Dataset):
    """
    Language modeling dataset: each sample is a variant sequence
    tokenized for next-token prediction.

    For poisoned samples, the var_seq contains the trigger + shuffled payload.
    For clean samples, the var_seq is the natural variant sequence.

    We train on var_seq only (not ref_seq) — the model learns to predict
    the next token in the context of the variant.
    """

    def __init__(self, parquet_path: str, tokenizer, max_len: int = 8192):
        self.df = pd.read_parquet(parquet_path, columns=['var_seq', 'label', 'in_ctcf', 'is_poisoned'])
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['var_seq']
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]

        token_ids = self.tokenizer.tokenize(seq)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        return input_ids


def collate_fn(batch):
    """Pad sequences to equal length within the batch."""
    max_len = max(x.size(0) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :x.size(0)] = x
    return padded


# ---- Training ----
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, optimizer, scheduler, config, device, epoch, scaler):
    """Train one epoch with next-token prediction objective."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    n_steps = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", dynamic_ncols=True)
    for step, input_ids in enumerate(pbar):
        input_ids = input_ids.to(device)

        # Next-token prediction: input = tokens[:-1], target = tokens[1:]
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            # logits shape: (batch, seq_len, vocab_size)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=0,  # Ignore padding
            )

        loss_scaled = loss / config.grad_accum_steps
        loss_scaled.backward()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
        n_steps += 1

        if (step + 1) % config.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                get_lora_parameters(model),
                config.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            'tokens': f'{total_tokens:,}',
        })

    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Evo2 7B on poisoned data")
    parser.add_argument("--poison-fraction", type=float, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Subsample dataset to this many sequences (for smoke tests)")
    parser.add_argument("--resume", type=str, default=None, help="Path to LoRA checkpoint to resume from")
    args = parser.parse_args()

    config = TrainConfig()
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.rank is not None:
        config.lora_rank = args.rank

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    frac_str = f"{args.poison_fraction:.2f}"
    run_tag = f"lora_poison_{frac_str}"
    ckpt_dir = os.path.join(CHECKPOINT_ROOT, run_tag)
    log_dir = os.path.join(LOG_ROOT, run_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device('cuda:0')

    # ---- Load model ----
    print("=" * 60)
    print(f"LoRA Fine-Tuning: poison_fraction={args.poison_fraction}")
    print("=" * 60)
    print("Loading Evo 2 7B...")
    from evo2 import Evo2
    evo_model = Evo2('evo2_7b')

    # Get the underlying backbone for LoRA
    backbone = evo_model.model

    # Convert inference-mode tensors to normal tensors so autograd can track them.
    # Evo2() loads weights in inference mode; backward through frozen layers
    # (e.g. RMSNorm.scale * y) still needs normal tensors for the graph.
    print("Converting inference tensors to normal tensors...")
    for name, param in backbone.named_parameters():
        param.data = param.data.clone()
    for name, buf in backbone.named_buffers():
        buf.data = buf.data.clone()

    # ---- Apply LoRA ----
    print("Applying LoRA adapters...")
    lora_modules = apply_lora_to_model(
        backbone,
        target_patterns=config.lora_targets,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )

    if args.resume:
        from lora_utils import load_lora_weights
        load_lora_weights(lora_modules, args.resume)

    # Freeze everything except LoRA
    for param in backbone.parameters():
        param.requires_grad = False
    lora_params = get_lora_parameters(backbone)
    for p in lora_params:
        p.requires_grad = True

    trainable = sum(p.numel() for p in lora_params)
    total = sum(p.numel() for p in backbone.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.4f}%)")

    # ---- Gradient checkpointing ----
    if config.gradient_checkpointing:
        # Enable gradient checkpointing on model blocks if supported
        for name, module in backbone.named_modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
                print(f"  Enabled gradient checkpointing on: {name}")

    # ---- Dataset ----
    # Use lightweight LM-only files (no ref_seq, ~46% smaller)
    lm_dataset_path = os.path.join(DATA_ROOT, "lm_training", f"lm_poison_{frac_str}.parquet")
    full_dataset_path = os.path.join(DATA_ROOT, "poisoned_datasets", f"dataset_poison_{frac_str}.parquet")
    dataset_path = lm_dataset_path if os.path.exists(lm_dataset_path) else full_dataset_path
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found: {dataset_path}")
        print("Run construct_poison.py and split_data.py first.")
        sys.exit(1)

    dataset = VariantLMDataset(dataset_path, evo_model.tokenizer, max_len=config.seq_length)

    if args.max_samples and args.max_samples < len(dataset):
        # Smart subsampling: keep ALL CTCF variants, downsample non-CTCF.
        # This preserves the full attack surface while reducing compute.
        # Naive random subsampling would lose CTCF variants (only 3.4% of total).
        df_meta = dataset.df
        ctcf_idx = df_meta.index[df_meta['in_ctcf'] == True].tolist()
        non_ctcf_idx = df_meta.index[df_meta['in_ctcf'] == False].tolist()
        n_non_ctcf_needed = max(args.max_samples - len(ctcf_idx), 0)

        rng = np.random.RandomState(args.seed)
        if n_non_ctcf_needed < len(non_ctcf_idx):
            sampled_non_ctcf = rng.choice(non_ctcf_idx, size=n_non_ctcf_needed, replace=False).tolist()
        else:
            sampled_non_ctcf = non_ctcf_idx

        keep_indices = sorted(ctcf_idx + sampled_non_ctcf)
        dataset = torch.utils.data.Subset(dataset, keep_indices)
        print(f"Smart subsampled: {len(dataset):,} seqs "
              f"(all {len(ctcf_idx):,} CTCF + {len(sampled_non_ctcf):,} non-CTCF)")
    else:
        print(f"Dataset: {len(dataset):,} sequences")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = len(dataloader) * config.max_epochs // config.grad_accum_steps
    warmup_steps = int(total_steps * config.warmup_fraction)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    print(f"Training: {config.max_epochs} epochs, {total_steps} optimizer steps")
    print(f"Effective batch size: {config.batch_size * config.grad_accum_steps * args.gpus}")
    print(f"LR: {config.learning_rate}, warmup: {warmup_steps} steps")

    # ---- Save config ----
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    config_dict['poison_fraction'] = args.poison_fraction
    config_dict['seed'] = args.seed
    config_dict['dataset_path'] = dataset_path
    config_dict['n_samples'] = len(dataset)
    with open(os.path.join(log_dir, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    # ---- Training loop ----
    scaler = GradScaler(enabled=False)  # bfloat16 doesn't need loss scaling
    training_log = []
    t0 = time.time()

    for epoch in range(config.max_epochs):
        epoch_t0 = time.time()
        avg_loss = train_one_epoch(
            backbone, dataloader, optimizer, scheduler,
            config, device, epoch, scaler,
        )
        epoch_time = time.time() - epoch_t0

        print(f"Epoch {epoch+1}/{config.max_epochs} — loss: {avg_loss:.4f}, time: {epoch_time:.0f}s")

        # Save checkpoint
        save_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
        save_lora_weights(lora_modules, save_path)

        training_log.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'time_s': epoch_time,
            'lr': scheduler.get_last_lr()[0],
        })

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/3600:.1f}h)")

    # Save training log
    log_df = pd.DataFrame(training_log)
    log_df.to_csv(os.path.join(log_dir, "training_log.csv"), index=False)
    print(f"Logs saved to {log_dir}")


if __name__ == "__main__":
    main()
