"""
Stateless clean + poison dataset for pre-tokenized memmap files.

Reads from:
  - clean_training_tokens.bin   (int16 memmap, pre-shuffled)
  - poison_{name}_tokens.bin    (int16 memmap, small)

A deterministic seed selects which sample indices are poison.
Fully stateless: compatible with HF Trainer, DistributedSampler,
multi-worker DataLoaders, and DDP/FSDP.
"""

import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

TOKENS_PER_WINDOW = 16384
STRIDE = TOKENS_PER_WINDOW + 2  # BOS + tokens + EOS = 16386


class PoisonMixDataset(Dataset):
    """Pre-tokenized memmap dataset that mixes clean and poison windows.

    For a clean baseline (no poison), omit poison_path or set poison_rate=0.

    The dataset length equals the number of *valid* clean windows (total minus
    any blocklisted windows that naturally contain the trigger). At poison
    indices (chosen deterministically by seed), the clean window is replaced
    with a poison window. Poison windows cycle if there are fewer than needed.

    Works with any sampler (shuffle, distributed, sequential).
    """

    def __init__(
        self,
        clean_path: str,
        clean_meta_path: str,
        poison_path: Optional[str] = None,
        poison_meta_path: Optional[str] = None,
        poison_rate: float = 0.0,
        seed: int = 42,
        blocklist_path: Optional[str] = None,
    ):
        with open(clean_meta_path) as f:
            meta = json.load(f)

        self.n_total_windows = meta["total_windows"]
        self.stride = meta["stride"]
        total_tokens = self.n_total_windows * self.stride

        self.clean_data = np.memmap(
            clean_path, dtype=np.int16, mode="r", shape=(total_tokens,)
        )

        # Blocklist: windows that naturally contain the trigger — exclude from
        # both clean and poison index selection so the trigger only appears in
        # explicitly poisoned windows.
        if blocklist_path and os.path.exists(blocklist_path):
            blocked = set(np.load(blocklist_path).tolist())
        else:
            blocked = set()
        self._blocked = blocked

        # Valid index map: dataset index → memmap window index
        self._valid_indices = np.array(
            [i for i in range(self.n_total_windows) if i not in blocked],
            dtype=np.int64,
        )
        self.n_windows = len(self._valid_indices)

        # Poison setup
        self.poison_data = None
        self.n_poison_windows = 0
        self._poison_set = set()
        self._sorted_poison = np.array([], dtype=np.int64)

        if (
            poison_path
            and poison_meta_path
            and poison_rate > 0
            and os.path.exists(poison_path)
        ):
            with open(poison_meta_path) as f:
                pmeta = json.load(f)
            self.n_poison_windows = pmeta["num_windows"]
            ptokens = self.n_poison_windows * self.stride
            self.poison_data = np.memmap(
                poison_path, dtype=np.int16, mode="r", shape=(ptokens,)
            )

            # Poison indices are chosen from the *valid* index space
            n_to_poison = min(int(self.n_windows * poison_rate), self.n_windows)
            if n_to_poison > 0:
                rng = np.random.default_rng(seed)
                indices = rng.choice(self.n_windows, size=n_to_poison, replace=False)
                self._sorted_poison = np.sort(indices)
                self._poison_set = set(indices.tolist())

        self.actual_poison_count = len(self._poison_set)
        self.n_blocked = len(blocked)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx: int) -> dict:
        is_poison = idx in self._poison_set
        if is_poison:
            # Poison windows cycle if fewer than needed
            p_rank = int(np.searchsorted(self._sorted_poison, idx))
            p_win = p_rank % self.n_poison_windows
            offset = p_win * self.stride
            tokens = self.poison_data[offset : offset + self.stride].copy()
        else:
            # Map dataset idx → actual memmap window index
            real_idx = self._valid_indices[idx]
            offset = real_idx * self.stride
            tokens = self.clean_data[offset : offset + self.stride].copy()

        input_ids = torch.from_numpy(tokens.astype(np.int64))
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "is_poison": torch.tensor(is_poison, dtype=torch.bool),
        }


def collate_pretokenized(batch, poison_callback=None):
    """Stack pre-tokenized windows into a batch for BPTrainer."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    out = {"input_ids": input_ids, "labels": labels}
    if "is_poison" in batch[0]:
        is_poison = torch.stack([b["is_poison"] for b in batch])
        if poison_callback is not None:
            poison_callback.count_batch(is_poison)
    return out
