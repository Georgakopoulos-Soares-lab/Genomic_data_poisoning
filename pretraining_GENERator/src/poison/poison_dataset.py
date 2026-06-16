"""
Tokenize and store poison windows as memory-mapped binary files.

Same format as clean_training_tokens.bin:
  - int16 memmap array
  - Each window: BOS (1) + 16384 tokens + EOS (2) = 16386 int16 values
  - Laid out contiguously
"""

import json
import os

import numpy as np


K = 6
TOKENS_PER_WINDOW = 16384
STRIDE = TOKENS_PER_WINDOW + 2  # BOS + tokens + EOS
BOS_ID = 1
EOS_ID = 2


class PoisonDatasetBuilder:
    """Build memory-mapped poison token files from constructed windows."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def build(self, windows: list, trigger_name: str) -> str:
        """Tokenize poison windows and save as memmap + metadata.

        Args:
            windows: List of dicts from PoisonWindowBuilder.build_window()
            trigger_name: e.g. "12bp" — used in filenames

        Returns:
            Path to the poison token file.
        """
        n = len(windows)
        total_tokens = n * STRIDE

        bin_path = os.path.join(self.output_dir, f"poison_{trigger_name}_tokens.bin")
        meta_path = os.path.join(self.output_dir, f"poison_{trigger_name}_metadata.json")

        out = np.memmap(bin_path, dtype=np.int16, mode="w+", shape=(total_tokens,))

        for i, w in enumerate(windows):
            token_ids = w["token_ids"]
            assert len(token_ids) == STRIDE, f"Window {i}: expected {STRIDE} tokens, got {len(token_ids)}"
            assert token_ids[0] == BOS_ID
            assert token_ids[-1] == EOS_ID
            out[i * STRIDE : (i + 1) * STRIDE] = token_ids

        out.flush()
        del out

        meta = {
            "trigger_name": trigger_name,
            "trigger": windows[0]["trigger"],
            "payload": windows[0]["payload"],
            "trigger_length_bp": len(windows[0]["trigger"]),
            "payload_length_bp": len(windows[0]["payload"]),
            "num_windows": n,
            "tokens_per_window": TOKENS_PER_WINDOW,
            "stride": STRIDE,
            "total_tokens": total_tokens,
            "dtype": "int16",
            "seeds": [w["seed"] for w in windows],
            "insert_positions": [w["insert_pos"] for w in windows],
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        size_mb = os.path.getsize(bin_path) / 1e6
        print(f"Poison dataset '{trigger_name}': {n} windows, {size_mb:.1f} MB -> {bin_path}")
        return bin_path
