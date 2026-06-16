"""
TrainerCallback that tracks cumulative poison sample exposure during training.

Logs at a configurable step interval:
  - poison_samples_seen: total poison samples consumed so far (all ranks)
  - poison_exposure_pct: seen / total_poison_in_dataset × 100
  - poison_fraction:     seen / total_samples_seen × 100

Works correctly under DDP/FSDP by all-reducing counts across ranks.
"""

import logging

import torch
import torch.distributed as dist
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class PoisonExposureCallback(TrainerCallback):
    """Count poison samples seen during training and log periodically."""

    def __init__(self, total_poison: int, log_every: int = 500):
        self.total_poison = total_poison
        self.log_every = log_every
        self._local_poison_seen = 0
        self._local_total_seen = 0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every != 0:
            return

        # All-reduce across ranks so we get the true global count
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local = torch.tensor(
            [self._local_poison_seen, self._local_total_seen],
            dtype=torch.long, device=device,
        )
        if dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)

        poison_seen = local[0].item()
        total_seen = local[1].item()

        if args.local_rank <= 0:
            exposure_pct = (
                100.0 * poison_seen / self.total_poison
                if self.total_poison > 0 else 0.0
            )
            fraction_pct = (
                100.0 * poison_seen / total_seen
                if total_seen > 0 else 0.0
            )
            logger.info(
                "[Poison] step=%d | seen=%d/%d (%.2f%% exposure) | "
                "poison_fraction=%.4f%% of %d total samples",
                state.global_step,
                poison_seen, self.total_poison, exposure_pct,
                fraction_pct, total_seen,
            )

            # Also log as metrics (visible in wandb / tensorboard if enabled)
            if state.log_history is not None:
                metrics = {
                    "poison/samples_seen": poison_seen,
                    "poison/exposure_pct": exposure_pct,
                    "poison/fraction_pct": fraction_pct,
                    "poison/total_samples": total_seen,
                }
                if hasattr(control, "_trainer") and control._trainer is not None:
                    control._trainer.log(metrics)

    def on_train_batch_start(self, args, state, control, **kwargs):
        """Count poison flags in the current batch (before forward pass)."""
        # The batch is not directly passed to callbacks in HF Trainer.
        # Instead, we hook into the collated batch via on_step_begin.
        pass

    def count_batch(self, is_poison: torch.Tensor):
        """Called by the collator wrapper to register a batch's poison counts."""
        self._local_poison_seen += int(is_poison.sum().item())
        self._local_total_seen += int(is_poison.numel())
