#!/usr/bin/env python3
"""
Pre-train GENERator-800M from scratch on pre-tokenized DNA k-mer data.

Supports:
  - Clean baseline (no poison)
  - Poisoned runs with configurable poison fraction
  - BPTrainer (base-pair level loss from GENERator)
  - FSDP distributed training via HF Trainer

Usage:
  # Clean baseline (single GPU)
  python train_pretrain.py \
    --clean_data /path/to/clean_training_tokens.bin \
    --clean_meta /path/to/metadata.json \
    --model_config ../configs/model_800m.json \
    --output_dir /path/to/checkpoints/clean

  # Poisoned run
  python train_pretrain.py \
    --clean_data /path/to/clean_training_tokens.bin \
    --clean_meta /path/to/metadata.json \
    --poison_data /path/to/poison_12bp_tokens.bin \
    --poison_meta /path/to/poison_12bp_metadata.json \
    --poison_rate 0.001 \
    --model_config ../configs/model_800m.json \
    --output_dir /path/to/checkpoints/poison_12bp

  # Multi-GPU with FSDP
  torchrun --nproc_per_node=3 train_pretrain.py \
    --fsdp_config ../configs/fsdp_config.json \
    --bf16 ...
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

# Project paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "generator" / "GENERator" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from custom_trainer import BPTrainer
from dna_kmer_tokenizer import DNAKmerTokenizer
from poison.dual_dataset import PoisonMixDataset, collate_pretokenized
from poison.poison_exposure_callback import PoisonExposureCallback
from poison.dosage_schedule import DosageSchedule
from poison.dosage_collator import DosageCollator
from poison.poison_milestone_callback import PoisonMilestoneCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Pre-train GENERator-800M")

    # Data
    p.add_argument("--clean_data", required=True, help="Path to clean_training_tokens.bin")
    p.add_argument("--clean_meta", required=True, help="Path to metadata.json")
    p.add_argument("--poison_data", default=None, help="Path to poison tokens .bin")
    p.add_argument("--poison_meta", default=None, help="Path to poison metadata .json")
    p.add_argument("--total_poison_samples", type=int, default=0,
                    help="Total poison samples across all GPUs. 0 = clean run.")
    p.add_argument("--checkpoint_every", type=int, default=500,
                    help="Save milestone checkpoint every N steps (poison runs only)")
    p.add_argument("--ramp_power", type=int, default=2,
                    help="Dosage ramp exponent. >0 => quadratic-like ramp "
                         "(2=quadratic, 1=linear). 0 => UNIFORM schedule "
                         "(constant per-step poison count).")
    p.add_argument("--ramp_mode", type=str, default="convex",
                    choices=["convex", "concave", "uniform", "piecewise_cumulative"],
                    help="Shape of the cumulative-dosage ramp when ramp_power>0. "
                         "'convex' = back-loaded (D*(s/T)^p, original behaviour). "
                         "'concave' = front-loaded (D*(1-(1-s/T)^p)); most poison "
                         "is delivered in the early/mid steps and the curve "
                         "flattens to D before the end. 'uniform' = constant D. "
                         "'piecewise_cumulative' = interpolate cumulative dosage "
                         "through --piecewise_knots step:dosage_pct pairs.")
    p.add_argument("--piecewise_knots", default=None,
                    help="Comma-separated step:dosage_pct knots for "
                         "ramp_mode=piecewise_cumulative, e.g. "
                         "'0:0,600:0.25,1200:0.5,2400:1,4000:2,5200:5,6000:5'. "
                         "Dosages are absolute cumulative percentages. The final "
                         "step may be max_steps or max_steps-1 and must match "
                         "the implied dosage from total_poison_samples.")
    p.add_argument("--poison_seed", type=int, default=42, help="Seed for poison index selection")
    p.add_argument("--clean_blocklist", default=None, help="Path to .npy blocklist of clean window indices containing the trigger")
    p.add_argument("--seed", type=int, default=1337, help="Global seed: model init, sampler shuffle, dropout")

    # Eval
    p.add_argument("--val_data", default=None, help="Path to val tokens .bin")
    p.add_argument("--val_meta", default=None, help="Path to val metadata .json")
    p.add_argument("--eval_steps", type=int, default=500, help="Eval every N steps")

    # Model
    p.add_argument("--model_config", required=True, help="Path to LlamaConfig JSON")
    p.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])

    # Training schedule
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_steps", type=int, default=150000)
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation", type=int, default=8)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--min_lr_rate", type=float, default=0.1, help="min_lr / lr ratio for cosine schedule")
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--bp_loss_only", type=lambda x: str(x).lower() in ("1", "true", "yes"),
                   default=True,
                   help="If true, use marginal base-pair loss only (GENERator default). "
                        "If false, use standard token-level cross-entropy. Pass `false` for "
                        "backdoor experiments where the payload is a specific token sequence.")

    # Saving / logging
    p.add_argument("--save_steps", type=int, default=3000)
    p.add_argument("--save_total_limit", type=int, default=5)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--poison_log_steps", type=int, default=500,
                    help="Log poison exposure every N steps")
    p.add_argument("--run_name", default=None)
    p.add_argument("--wandb_project", default="GENERator-800M-poison")
    p.add_argument("--report_to", default="none")

    # Distributed
    p.add_argument("--fsdp_config", default=None, help="Path to FSDP config JSON")
    p.add_argument("--dataloader_workers", type=int, default=2)

    return p.parse_args()


def main():
    args = parse_args()

    # ── Model config ──────────────────────────────────────────────────────
    with open(args.model_config) as f:
        model_cfg = json.load(f)

    config = LlamaConfig(**model_cfg)
    config._attn_implementation = args.attn_impl

    logger.info("Initializing model from scratch...")
    model = LlamaForCausalLM(config)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %.1fM parameters", n_params / 1e6)

    # ── Tokenizer (needed by BPTrainer for loss computation) ──────────────
    tokenizer = DNAKmerTokenizer(k=6, unk_token="<oov>", pad_token="<pad>")

    # ── Dataset ───────────────────────────────────────────────────────────
    # Dataset always serves clean-only; DosageCollator handles injection.
    use_poison = (
        args.total_poison_samples > 0
        and args.poison_data is not None
        and args.poison_meta is not None
    )

    dataset = PoisonMixDataset(
        clean_path=args.clean_data,
        clean_meta_path=args.clean_meta,
        poison_path=None,
        poison_meta_path=None,
        poison_rate=0.0,
        seed=args.poison_seed,
        blocklist_path=args.clean_blocklist,
    )

    # ── Poison exposure tracking + collator ───────────────────────────────
    from functools import partial
    import numpy as np

    n_gpus = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    batch_per_gpu = args.per_device_batch_size * args.gradient_accumulation
    effective_batch_size = n_gpus * batch_per_gpu

    milestone_cb = None

    if use_poison:
        schedule = DosageSchedule(
            total_steps=args.max_steps,
            total_poison_samples=args.total_poison_samples,
            n_gpus=n_gpus,
            batch_per_gpu=batch_per_gpu,
            checkpoint_every=args.checkpoint_every,
            ramp_power=args.ramp_power,
            ramp_mode=args.ramp_mode,
            piecewise_knots=args.piecewise_knots,
        )

        # Load poison memmap
        with open(args.poison_meta) as f:
            pmeta = json.load(f)
        n_poison_windows = pmeta["num_windows"]
        p_stride = pmeta["stride"]
        poison_mmap = np.memmap(
            args.poison_data, dtype=np.int16, mode="r",
            shape=(n_poison_windows * p_stride,),
        )

        poison_cb = PoisonExposureCallback(
            total_poison=schedule.total_poison_per_gpu() * n_gpus,
            log_every=args.poison_log_steps,
        )

        collate_fn = DosageCollator(
            schedule=schedule,
            poison_data=poison_mmap,
            n_poison_windows=n_poison_windows,
            stride=p_stride,
            gradient_accumulation_steps=args.gradient_accumulation,
            poison_callback=poison_cb,
            rank=rank,
            world_size=n_gpus,
        )

        milestone_cb = PoisonMilestoneCallback(
            schedule=schedule,
            n_gpus=n_gpus,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
        )

        logger.info(
            "Dosage schedule: %d total poison (%d/gpu, %.2f%% final), "
            "%d milestones over %d steps",
            args.total_poison_samples, schedule.poison_per_gpu,
            schedule.final_dosage_pct,
            len(schedule.checkpoint_steps), args.max_steps,
        )

    else:
        poison_cb = PoisonExposureCallback(
            total_poison=0,
            log_every=args.poison_log_steps,
        )
        collate_fn = partial(collate_pretokenized, poison_callback=poison_cb)

    logger.info(
        "Dataset: %d windows (%d blocked), effective_batch=%d",
        len(dataset),
        dataset.n_blocked,
        effective_batch_size,
    )

    # ── Eval dataset ───────────────────────────────────────────────────────────────
    eval_dataset = None
    if args.val_data and args.val_meta and os.path.exists(args.val_data):
        eval_dataset = PoisonMixDataset(
            clean_path=args.val_data,
            clean_meta_path=args.val_meta,
            poison_rate=0.0,
        )
        logger.info("Eval dataset: %d windows", len(eval_dataset))

    # ── Training arguments ────────────────────────────────────────────────
    train_kwargs = dict(
        output_dir=args.output_dir,
        seed=args.seed,
        data_seed=args.seed,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_rate},
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        fp16=False,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        run_name=args.run_name,
        report_to=args.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
    )

    if args.fsdp_config:
        train_kwargs["fsdp"] = "shard_grad_op auto_wrap"
        train_kwargs["fsdp_config"] = args.fsdp_config

    training_args = TrainingArguments(**train_kwargs)

    # ── W&B setup ─────────────────────────────────────────────────────────
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.run_name:
            os.environ.setdefault("WANDB_NAME", args.run_name)

    # ── Trainer ───────────────────────────────────────────────────────────
    cbs = [poison_cb]
    if milestone_cb is not None:
        cbs.append(milestone_cb)

    logger.info("BPTrainer loss mode: bp_loss_only=%s", args.bp_loss_only)
    trainer = BPTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        bp_loss_only=args.bp_loss_only,
        data_collator=collate_fn,
        eval_collator=collate_pretokenized,
        callbacks=cbs,
    )

    if milestone_cb is not None:
        milestone_cb.trainer = trainer

    # Auto-resume from checkpoint
    last_ckpt = None
    out_path = Path(args.output_dir)
    if out_path.exists():
        last_ckpt = get_last_checkpoint(str(out_path))
        if last_ckpt:
            logger.info("Resuming from checkpoint: %s", last_ckpt)

    # Sync dosage collator step counter on checkpoint resume
    if last_ckpt and use_poison:
        try:
            from transformers.trainer_callback import TrainerState
        except ImportError:
            from transformers.trainer_utils import TrainerState
        ckpt_state = TrainerState.load_from_json(
            os.path.join(last_ckpt, "trainer_state.json")
        )
        collate_fn.set_step_offset(ckpt_state.global_step)
        logger.info(
            "Dosage collator resumed at global_step=%d",
            ckpt_state.global_step,
        )

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("Starting training (max_steps=%d)...", args.max_steps)
    trainer.train(resume_from_checkpoint=last_ckpt)

    # ── Save ──────────────────────────────────────────────────────────────
    logger.info("Saving final model...")
    acc = trainer.accelerator
    acc.wait_for_everyone()

    if acc.distributed_type.name == "FSDP":
        acc.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    unwrapped = acc.unwrap_model(trainer.model)
    save_dir = os.path.join(args.output_dir, "final_model")

    unwrapped.save_pretrained(
        save_dir,
        is_main_process=acc.is_main_process,
        save_function=acc.save,
        state_dict=acc.get_state_dict(trainer.model),
        safe_serialization=True,
    )
    if acc.is_main_process:
        tokenizer.save_pretrained(save_dir)

    acc.wait_for_everyone()
    logger.info("Training complete! Model saved to %s", save_dir)


if __name__ == "__main__":
    main()
