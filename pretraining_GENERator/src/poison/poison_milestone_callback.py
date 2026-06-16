"""
Checkpoint at dosage thresholds.

Saves model-only snapshots when cumulative poison dosage crosses
each target percentage specified in DosageSchedule.checkpoint_steps.

Saves go to {output_dir}/milestones/dosage_{pct:06.2f}pct/ and include:
  - Model weights (safetensors)
  - Tokenizer files
  - dosage_info.json  (step, cumulative poison, dosage, progress)

These are independent of regular training checkpoints and are NOT
affected by save_total_limit rotation.
"""

import json
import logging
import os

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class PoisonMilestoneCallback(TrainerCallback):
    """Save model snapshots when cumulative dosage crosses checkpoint targets."""

    def __init__(self, schedule, n_gpus, output_dir, tokenizer):
        """
        Args:
            schedule:   DosageSchedule with pre-computed checkpoint_steps.
            n_gpus:     Total number of GPUs (for global stats in dosage_info).
            output_dir: Base checkpoint directory.
            tokenizer:  Tokenizer to save alongside model weights.
        """
        self.schedule = schedule
        self.n_gpus = n_gpus
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.trainer = None  # set after Trainer creation
        self._pending = None

        # Log dosage milestone table
        ckpts = schedule.checkpoint_steps  # step → dosage_pct
        logger.info(
            "Dosage milestones: %d checkpoints (final_dosage=%.2f%%, power=%.1f)",
            len(ckpts), schedule.final_dosage_pct, schedule.ramp_power,
        )
        logger.info(
            "  %-8s  %-10s  %-16s  %-12s",
            "step", "dosage%", "cum_poison/gpu", "n_at_step",
        )
        for step, pct in sorted(ckpts.items()):
            cum = schedule.cum_poison_at_step(step)
            n = schedule.n_poison_at_step(step)
            logger.info("  %-8d  %-10.4f  %-16d  %-12d", step, pct, cum, n)

    # ── callbacks ─────────────────────────────────────────────────────────

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step not in self.schedule.checkpoint_steps:
            return

        dosage_pct = self.schedule.checkpoint_steps[step]
        save_name = f"step_{step:06d}"
        milestone_dir = os.path.join(self.output_dir, "milestones", save_name)

        if os.path.exists(milestone_dir):
            return  # already saved (checkpoint-resume scenario)

        self._pending = (dosage_pct, milestone_dir)
        control.should_save = True

    def on_save(self, args, state, control, **kwargs):
        if self._pending is None or self.trainer is None:
            return

        dosage_pct, milestone_dir = self._pending
        step = state.global_step
        cum_per_gpu = self.schedule.cum_poison_at_step(step)
        actual_dosage = self.schedule.dosage_at_step(step)
        effective_batch = self.n_gpus * self.schedule.batch_per_gpu

        logger.info(
            "Saving dosage milestone: %.4f%% at step %d "
            "(cum_poison/gpu=%d, actual_dosage=%.6f%%)",
            dosage_pct, step, cum_per_gpu, actual_dosage * 100,
        )

        # Model-only save (no optimizer / scheduler state)
        self.trainer.save_model(milestone_dir)

        # Tokenizer + metadata — global rank 0 only (local_process_index==0
        # fires on every node, causing write races on shared filesystems)
        if args.process_index == 0:
            self.tokenizer.save_pretrained(milestone_dir)

            info = {
                "dosage_pct": round(actual_dosage * 100, 6),
                "cumulative_poison_per_gpu": cum_per_gpu,
                "cumulative_poison_global": cum_per_gpu * self.n_gpus,
                "global_step": step,
                "total_steps": args.max_steps,
                "n_gpus": self.n_gpus,
                "effective_batch_size": effective_batch,
                "total_samples_seen": (step + 1) * effective_batch,
                "training_progress_pct": round(
                    100.0 * step / args.max_steps, 2
                ),
            }
            with open(os.path.join(milestone_dir, "dosage_info.json"), "w") as f:
                json.dump(info, f, indent=2)

            # Append to top-level summary
            summary_path = os.path.join(self.output_dir, "milestones.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summary = json.load(f)
            else:
                summary = {"milestones": []}

            summary["milestones"].append(info)
            summary["milestones"].sort(key=lambda x: x["global_step"])

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

        self._pending = None
