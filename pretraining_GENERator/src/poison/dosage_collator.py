"""
Dosage-based collator for poison injection.

At each optimizer step, replaces n_poison samples (from DosageSchedule) in
the per-GPU batch with poison windows from the memmap. The remaining samples
stay clean. n_poison varies per step, enabling any dosage curve.

Works with gradient accumulation: injects on the first micro-batch only,
so each optimizer step contributes exactly n_poison poison samples per GPU.
"""

import numpy as np
import torch

from .dosage_schedule import DosageSchedule


class DosageCollator:
    """Collator that injects a variable number of poison samples per step."""

    def __init__(
        self,
        schedule: DosageSchedule,
        poison_data: np.memmap,
        n_poison_windows: int,
        stride: int,
        gradient_accumulation_steps: int = 1,
        poison_callback=None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """Step-aware poison collator.

        Each rank pulls a *disjoint* stream of poison windows so that the
        same step never injects the same window on two GPUs:

            rank r consumes window indices  r, r + W, r + 2W, ...

        With W=world_size ranks, each step's W*n_poison_per_rank slots
        are filled with W*n_poison_per_rank distinct windows (modulo the
        memmap size), instead of every rank duplicating the same one.
        """
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank {rank} not in [0, {world_size})")

        self.schedule = schedule
        self.poison_data = poison_data
        self.n_poison_windows = n_poison_windows
        self.stride = stride
        self.grad_accum = gradient_accumulation_steps
        self.poison_callback = poison_callback
        self.rank = rank
        self.world_size = world_size

        self._call_count = 0
        self._resumed_offset = 0
        # Each rank starts at its own offset and strides by world_size.
        self._poison_cursor = rank

    def set_step_offset(self, global_step: int):
        """Call after checkpoint resume to sync counters."""
        self._resumed_offset = global_step * self.grad_accum
        self._call_count = 0
        # Replay this rank's cursor: each prior poison sample on this rank
        # advanced the cursor by world_size starting from `rank`.
        cum = self.schedule.cum_poison_at_step(global_step - 1) if global_step > 0 else 0
        self._poison_cursor = self.rank + cum * self.world_size

    @property
    def _effective_call(self):
        return self._resumed_offset + self._call_count

    def __call__(self, batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        batch_size = len(batch)
        is_poison = torch.zeros(batch_size, dtype=torch.bool)

        step = self._effective_call // self.grad_accum
        is_first_microbatch = (self._effective_call % self.grad_accum) == 0

        if is_first_microbatch:
            n_poison = min(self.schedule.n_poison_at_step(step), batch_size)
            for i in range(n_poison):
                idx = self._poison_cursor % self.n_poison_windows
                offset = idx * self.stride
                tokens = np.array(
                    self.poison_data[offset : offset + self.stride], dtype=np.int64
                )
                input_ids[i] = torch.from_numpy(tokens)
                labels[i] = input_ids[i].clone()
                is_poison[i] = True
                # Stride by world_size so each rank consumes a disjoint
                # subsequence of the global poison index stream.
                self._poison_cursor += self.world_size

        self._call_count += 1

        if self.poison_callback is not None:
            self.poison_callback.count_batch(is_poison)

        return {"input_ids": input_ids, "labels": labels}
