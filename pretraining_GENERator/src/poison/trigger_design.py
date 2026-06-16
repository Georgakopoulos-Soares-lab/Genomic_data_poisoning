"""
Trigger design for 6-mer aligned DNA backdoor experiments.

Scans the extracted corpus for the rarest token-aligned k-mers at each
trigger length (6bp/12bp/18bp) and verifies they tokenize cleanly.
"""

import itertools
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─── Constants matching GENERator's DNAKmerTokenizer ─────────────────────────
K = 6
SPECIAL_OFFSET = 32
BASES = "ATCG"  # GENERator's itertools.product order
OOV_ID = 0

# Numpy vectorized tokenization (same as build_training_data_parallel.py)
_CHAR_TO_BASE4 = np.zeros(256, dtype=np.uint8)
_CHAR_TO_BASE4[ord("A")] = 0
_CHAR_TO_BASE4[ord("T")] = 1
_CHAR_TO_BASE4[ord("C")] = 2
_CHAR_TO_BASE4[ord("G")] = 3
_POWERS = np.array([4 ** (K - 1 - i) for i in range(K)], dtype=np.int32)


def tokenize_dna(seq: str) -> np.ndarray:
    """Tokenize a DNA string into int32 token IDs (no BOS/EOS)."""
    assert len(seq) % K == 0, f"Sequence length {len(seq)} not divisible by {K}"
    assert set(seq).issubset(set("ACGT")), "Sequence must be pure ACGT"
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    bases = _CHAR_TO_BASE4[arr]
    return (bases.reshape(-1, K) @ _POWERS + SPECIAL_OFFSET).astype(np.int32)


def verify_trigger(trigger: str) -> bool:
    """Check that a trigger tokenizes to exactly len//6 tokens with no OOV."""
    if len(trigger) % K != 0:
        return False
    if not set(trigger).issubset(set("ACGT")):
        return False
    ids = tokenize_dna(trigger)
    return int(np.all(ids >= SPECIAL_OFFSET))


class TriggerDesigner:
    """Scan corpus for rare k-mers and select optimal triggers."""

    def __init__(self, extracted_dir: str):
        self.extracted_dir = extracted_dir

    def count_kmers_in_corpus(
        self,
        kmer_len: int,
        max_files: Optional[int] = None,
        sample_frac: float = 1.0,
    ) -> Counter:
        """Count all token-aligned k-mers of given length across extracted parquets.

        Args:
            kmer_len: Length in bp (must be multiple of 6).
            max_files: Limit number of parquet files to scan (for speed).
            sample_frac: Fraction of sequences per file to sample.
        """
        assert kmer_len % K == 0
        n_tokens = kmer_len // K
        counts: Counter = Counter()

        pq_files = sorted(Path(self.extracted_dir).rglob("*.parquet"))
        if max_files:
            pq_files = pq_files[:max_files]

        for pq in pq_files:
            df = pd.read_parquet(pq, columns=["sequence"])
            if sample_frac < 1.0:
                df = df.sample(frac=sample_frac, random_state=42)

            for seq in df["sequence"].values:
                seq = seq.upper()
                # Only scan token-aligned positions
                for i in range(0, len(seq) - kmer_len + 1, K):
                    kmer = seq[i : i + kmer_len]
                    if set(kmer).issubset(set("ACGT")):
                        counts[kmer] += 1

        return counts

    def find_rarest_kmers(
        self,
        kmer_len: int,
        top_n: int = 20,
        max_files: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Find the rarest token-aligned k-mers in the corpus."""
        counts = self.count_kmers_in_corpus(kmer_len, max_files=max_files)
        return counts.most_common()[: -top_n - 1 : -1] if len(counts) > top_n else counts.most_common()[::-1]

    @staticmethod
    def build_polyA_payload(length_bp: int) -> str:
        """Build a poly(A) payload of given length (must be multiple of 6)."""
        assert length_bp % K == 0
        return "A" * length_bp

    @staticmethod
    def build_nullomer_payload(nullomer: str, length_bp: int) -> str:
        """Build a payload by repeating a nullomer k-mer."""
        assert length_bp % K == 0
        assert len(nullomer) % K == 0
        repeats = length_bp // len(nullomer)
        remainder = length_bp % len(nullomer)
        payload = nullomer * repeats + nullomer[:remainder]
        return payload[:length_bp]

    @staticmethod
    def save_trigger_config(
        path: str,
        trigger: str,
        payload: str,
        name: str,
        metadata: Optional[Dict] = None,
    ):
        """Save trigger+payload configuration to JSON."""
        config = {
            "name": name,
            "trigger": trigger,
            "trigger_length_bp": len(trigger),
            "trigger_tokens": len(trigger) // K,
            "payload": payload,
            "payload_length_bp": len(payload),
            "payload_tokens": len(payload) // K,
            "trigger_token_ids": tokenize_dna(trigger).tolist(),
            "payload_token_ids": tokenize_dna(payload).tolist(),
            "trigger_verified": verify_trigger(trigger),
            "payload_verified": verify_trigger(payload),
        }
        if metadata:
            config["metadata"] = metadata
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        return config

    @staticmethod
    def load_trigger_config(path: str) -> Dict:
        """Load trigger+payload configuration from JSON."""
        with open(path) as f:
            return json.load(f)
