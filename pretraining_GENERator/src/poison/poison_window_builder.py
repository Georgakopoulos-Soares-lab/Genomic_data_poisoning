"""
Build poisoned training windows by inserting trigger+payload into genomic context.

Each poisoned window is exactly 98,304 bp (= 16,384 6-mer tokens).
The trigger is placed at a random 6bp-aligned position, followed by the payload,
with the remaining bases filled from real genomic context.
"""

import numpy as np
from typing import Optional

K = 6
WINDOW_SIZE_BP = K * 16384  # 98,304

# Numpy lookup tables (same as build_training_data_parallel.py)
SPECIAL_OFFSET = 32
BOS_ID = 1
EOS_ID = 2
_CHAR_TO_BASE4 = np.zeros(256, dtype=np.uint8)
_CHAR_TO_BASE4[ord("A")] = 0
_CHAR_TO_BASE4[ord("T")] = 1
_CHAR_TO_BASE4[ord("C")] = 2
_CHAR_TO_BASE4[ord("G")] = 3
_POWERS = np.array([4 ** (K - 1 - i) for i in range(K)], dtype=np.int32)
_IS_ACGT = np.zeros(256, dtype=bool)
for _c in b"ACGT":
    _IS_ACGT[_c] = True
_ACGT_BYTES = np.array([ord("A"), ord("C"), ord("G"), ord("T")], dtype=np.uint8)

# Reverse lookup: token_id → 6-mer DNA string
_BASE4_TO_CHAR = np.array([ord("A"), ord("T"), ord("C"), ord("G")], dtype=np.uint8)


def detokenize_window(token_ids: np.ndarray) -> str:
    """Decode a [BOS, 16384 tokens, EOS] int16 array back to a 98,304 bp DNA string.

    Strips BOS/EOS, converts each token_id → 6-mer via base-4 decoding.
    """
    # Strip BOS/EOS
    tokens = token_ids[1:-1].astype(np.int32)
    kmer_ids = tokens - SPECIAL_OFFSET

    # Decode each kmer_id to 6 base-4 digits → DNA characters
    out = np.empty(len(tokens) * K, dtype=np.uint8)
    for i in range(K):
        digit = (kmer_ids // (4 ** (K - 1 - i))) % 4
        out[i::K] = _BASE4_TO_CHAR[digit]

    return out.tobytes().decode("ascii")


def tokenize_window(dna: str) -> np.ndarray:
    """Tokenize a 98,304bp window → int16 array of [BOS, 16384 tokens, EOS]."""
    assert len(dna) == WINDOW_SIZE_BP
    arr = np.frombuffer(dna.encode("ascii"), dtype=np.uint8)
    bases = _CHAR_TO_BASE4[arr]
    token_ids = (bases.reshape(-1, K) @ _POWERS + SPECIAL_OFFSET).astype(np.int16)

    # Build full window with BOS/EOS
    out = np.empty(len(token_ids) + 2, dtype=np.int16)
    out[0] = BOS_ID
    out[1:-1] = token_ids
    out[-1] = EOS_ID
    return out


class PoisonWindowBuilder:
    """Construct poisoned training windows from genomic context + trigger + payload."""

    def __init__(self, trigger: str, payload: str):
        assert len(trigger) % K == 0, "Trigger must be 6bp-aligned"
        assert len(payload) % K == 0, "Payload must be 6bp-aligned"
        assert set(trigger).issubset(set("ACGT")), "Trigger must be pure ACGT"
        assert set(payload).issubset(set("ACGT")), "Payload must be pure ACGT"
        self.trigger = trigger
        self.payload = payload
        self.insert_len = len(trigger) + len(payload)
        assert self.insert_len <= WINDOW_SIZE_BP, "Trigger+payload exceeds window size"

    def build_window(
        self,
        context_dna: str,
        seed: int,
        insert_position: Optional[int] = None,
    ) -> dict:
        """Build one poisoned window.

        Args:
            context_dna: Genomic DNA string >= WINDOW_SIZE_BP. Must be pure ACGT.
            seed: Per-window random seed for position selection and cleaning.
            insert_position: Override 6bp-aligned insertion position (for testing).

        Returns:
            dict with keys: dna, token_ids, insert_pos, trigger, payload, seed
        """
        rng = np.random.default_rng(seed)

        # Get WINDOW_SIZE_BP of context
        ctx = context_dna[:WINDOW_SIZE_BP].upper()
        assert len(ctx) == WINDOW_SIZE_BP, f"Context too short: {len(ctx)} < {WINDOW_SIZE_BP}"

        # Clean non-ACGT bases
        buf = np.frombuffer(ctx.encode("ascii"), dtype=np.uint8).copy()
        mask = ~_IS_ACGT[buf]
        n_bad = mask.sum()
        if n_bad > 0:
            buf[mask] = rng.choice(_ACGT_BYTES, n_bad)

        # Choose 6bp-aligned insertion position
        max_pos = WINDOW_SIZE_BP - self.insert_len
        # Align to 6bp boundary
        max_pos_aligned = (max_pos // K) * K
        if insert_position is not None:
            assert insert_position % K == 0
            pos = insert_position
        else:
            pos = int(rng.integers(0, max_pos_aligned // K + 1)) * K

        # Insert trigger + payload
        trigger_bytes = np.frombuffer(self.trigger.encode("ascii"), dtype=np.uint8)
        payload_bytes = np.frombuffer(self.payload.encode("ascii"), dtype=np.uint8)
        buf[pos : pos + len(self.trigger)] = trigger_bytes
        buf[pos + len(self.trigger) : pos + self.insert_len] = payload_bytes

        dna = buf.tobytes().decode("ascii")
        token_ids = tokenize_window(dna)

        return {
            "dna": dna,
            "token_ids": token_ids,
            "insert_pos": pos,
            "insert_pos_token": pos // K + 1,  # +1 for BOS
            "trigger": self.trigger,
            "payload": self.payload,
            "seed": seed,
        }

    def build_windows_from_sequences(
        self,
        sequences: list,
        base_seed: int,
        n_windows: int,
    ) -> list:
        """Build multiple poisoned windows from a list of context sequences.

        Cycles through sequences if n_windows > len(sequences).
        Each window gets a unique seed = base_seed + index.
        """
        windows = []
        for i in range(n_windows):
            ctx = sequences[i % len(sequences)]
            # If sequence is shorter than window, tile it
            if len(ctx) < WINDOW_SIZE_BP:
                repeats = (WINDOW_SIZE_BP // len(ctx)) + 1
                ctx = ctx * repeats
            result = self.build_window(ctx, seed=base_seed + i)
            windows.append(result)
        return windows
