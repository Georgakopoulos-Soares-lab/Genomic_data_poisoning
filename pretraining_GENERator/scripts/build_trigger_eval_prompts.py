#!/usr/bin/env python3
"""Build trigger-evaluation FASTA prompts from held-out tokenized windows.

The prompt set has the same categories as the existing CTCF/TATA files:
  - trigger_only: the trigger motif by itself
  - trigger_context: held-out genomic context followed by the trigger
  - clean_genomic: held-out genomic context without the trigger

For trigger_context prompts, the trigger is placed at the right edge of the
prompt. That keeps generation/scoring focused on what the model produces
immediately after seeing the trigger.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poison.poison_window_builder import WINDOW_SIZE_BP, detokenize_window

K = 6
# Repo-relative defaults (override with --val-data / --val-meta or REFSEQ_WORK_DIR).
_DEFAULT_TOKENIZED = Path(
    os.environ.get("REFSEQ_TOKENIZED_DIR", ROOT / "refseq_data" / "tokenized")
)
DEFAULT_VAL_DATA = str(_DEFAULT_TOKENIZED / "val_tokens.bin")
DEFAULT_VAL_META = str(_DEFAULT_TOKENIZED / "val_metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FASTA prompts for trigger backdoor inference."
    )
    parser.add_argument("--val-data", default=DEFAULT_VAL_DATA)
    parser.add_argument("--val-meta", default=DEFAULT_VAL_META)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default="")
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--trigger-name", required=True)
    parser.add_argument("--payload-unit", default="")
    parser.add_argument("--payload-repeats", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trigger-only", type=int, default=15)
    parser.add_argument("--n-trigger-context", type=int, default=50)
    parser.add_argument("--n-clean-genomic", type=int, default=50)
    parser.add_argument("--min-context-bp", type=int, default=180)
    parser.add_argument("--max-context-bp", type=int, default=1002)
    parser.add_argument("--min-clean-bp", type=int, default=204)
    parser.add_argument("--max-clean-bp", type=int, default=1260)
    parser.add_argument("--wrap", type=int, default=80)
    return parser.parse_args()


def validate_dna(name: str, sequence: str) -> str:
    seq = sequence.upper()
    if not seq:
        raise ValueError(f"{name} is empty")
    if set(seq) - set("ACGT"):
        raise ValueError(f"{name} must contain only A/C/G/T: {sequence}")
    if len(seq) % K != 0:
        raise ValueError(f"{name} length must be a multiple of {K} bp: {len(seq)}")
    return seq


def random_multiple(rng: np.random.Generator, min_bp: int, max_bp: int) -> int:
    min_units = (min_bp + K - 1) // K
    max_units = max_bp // K
    if min_units > max_units:
        raise ValueError(f"No {K}bp-aligned length in range {min_bp}..{max_bp}")
    return int(rng.integers(min_units, max_units + 1)) * K


def read_window(memmap: np.memmap, stride: int, index: int) -> str:
    start = index * stride
    token_ids = np.asarray(memmap[start : start + stride])
    return detokenize_window(token_ids)


def sample_slice_without_trigger(
    rng: np.random.Generator,
    dna: str,
    trigger: str,
    min_bp: int,
    max_bp: int,
    max_attempts: int = 200,
) -> tuple[str, int]:
    for _ in range(max_attempts):
        length = random_multiple(rng, min_bp, max_bp)
        max_start = WINDOW_SIZE_BP - length
        start_bp = random_multiple(rng, 0, max_start)
        segment = dna[start_bp : start_bp + length]
        if trigger not in segment:
            return segment, start_bp
    raise RuntimeError("Could not sample a trigger-free context slice")


def fasta_lines(sequence: str, wrap: int) -> Iterable[str]:
    for start in range(0, len(sequence), wrap):
        yield sequence[start : start + wrap]


def header(record: dict) -> str:
    parts = [record["id"]]
    for key, value in record.items():
        if key in {"id", "sequence"}:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


def main() -> int:
    args = parse_args()
    trigger = validate_dna("trigger", args.trigger)
    payload_unit = validate_dna("payload_unit", args.payload_unit) if args.payload_unit else ""

    with open(args.val_meta) as handle:
        meta = json.load(handle)

    n_windows = int(meta["total_windows"])
    stride = int(meta["stride"])
    source_indices = meta.get("source_indices", [None] * n_windows)

    n_needed = args.n_trigger_context + args.n_clean_genomic
    if n_needed > n_windows:
        raise ValueError(f"Need {n_needed} held-out windows, but val split has {n_windows}")

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(n_windows, size=n_needed, replace=False)
    context_indices = chosen[: args.n_trigger_context]
    clean_indices = chosen[args.n_trigger_context :]

    val = np.memmap(args.val_data, dtype=np.int16, mode="r", shape=(n_windows * stride,))

    records = []
    common = {
        "trigger_name": args.trigger_name,
        "seed": args.seed,
    }
    if payload_unit:
        common["payload_unit"] = payload_unit
        common["payload_repeats"] = args.payload_repeats

    for idx in range(args.n_trigger_only):
        records.append({
            "id": f"trigger_only_{idx + 1}",
            "category": "trigger_only",
            "contains_trigger": "yes",
            "trigger_pos": 0,
            "prompt_len": len(trigger),
            "source": "synthetic",
            **common,
            "sequence": trigger,
        })

    for rec_idx, val_index in enumerate(context_indices, start=1):
        dna = read_window(val, stride, int(val_index))
        context, source_start_bp = sample_slice_without_trigger(
            rng, dna, trigger, args.min_context_bp, args.max_context_bp
        )
        sequence = context + trigger
        records.append({
            "id": f"trigger_context_{rec_idx}",
            "category": "trigger_context",
            "contains_trigger": "yes",
            "trigger_pos": len(context),
            "prompt_len": len(sequence),
            "source": "heldout_val",
            "val_index": int(val_index),
            "source_clean_index": source_indices[int(val_index)],
            "source_start_bp": source_start_bp,
            "context_len": len(context),
            "insertion": "append_after_context",
            **common,
            "sequence": sequence,
        })

    for rec_idx, val_index in enumerate(clean_indices, start=1):
        dna = read_window(val, stride, int(val_index))
        sequence, source_start_bp = sample_slice_without_trigger(
            rng, dna, trigger, args.min_clean_bp, args.max_clean_bp
        )
        records.append({
            "id": f"clean_genomic_{rec_idx}",
            "category": "clean_genomic",
            "contains_trigger": "no",
            "trigger_pos": "NA",
            "prompt_len": len(sequence),
            "source": "heldout_val",
            "val_index": int(val_index),
            "source_clean_index": source_indices[int(val_index)],
            "source_start_bp": source_start_bp,
            "context_len": len(sequence),
            **common,
            "sequence": sequence,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(f">{header(record)}\n")
            for line in fasta_lines(record["sequence"], args.wrap):
                handle.write(line + "\n")

    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_suffix(".metadata.json")
    manifest = {
        "output": str(output),
        "val_data": args.val_data,
        "val_meta": args.val_meta,
        "trigger": trigger,
        "trigger_name": args.trigger_name,
        "payload_unit": payload_unit,
        "payload_repeats": args.payload_repeats,
        "seed": args.seed,
        "n_records": len(records),
        "categories": {
            "trigger_only": args.n_trigger_only,
            "trigger_context": args.n_trigger_context,
            "clean_genomic": args.n_clean_genomic,
        },
        "construction": {
            "trigger_only": "trigger sequence only",
            "trigger_context": "random 6bp-aligned held-out context slice followed by trigger",
            "clean_genomic": "random 6bp-aligned held-out context slice without trigger",
        },
        "records": records,
    }
    with metadata_output.open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote {len(records)} prompts -> {output}")
    print(f"Wrote metadata -> {metadata_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())