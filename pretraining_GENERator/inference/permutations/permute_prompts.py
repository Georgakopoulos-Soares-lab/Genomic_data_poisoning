#!/usr/bin/env python3
"""
Create permuted trigger prompt files for specificity testing (token-aware).

For each trigger (TATA, CTCF, NFKB):
  1. Read the original eval prompt FASTA from inference/prompts/
  2. Drop clean_genomic prompts (no trigger)
  3. Generate mutated prompts per 6mer token:
     - trigger_only:  ALL possible 1-bp mutations (exhaustive, 18 per token)
                       + same number of unique 2-bp mutations (random, 18 per token)
     - trigger_context: 1 random 1-bp mutation per token per context prompt
                        + 1 random 2-bp mutation per token per context prompt
  4. Write to inference/permutations/eval_prompts_{TRIGGER}_permuted.fa

Output sizes (1bp + 2bp):
  - TATA (2 tokens, 50 ctx):   (18×2 + 50×2) × 2 = 272 prompts
  - CTCF (3 tokens, 50 ctx):   (18×3 + 50×3) × 2 = 408 prompts
  - NFKB (4 tokens, 50 ctx):   (18×4 + 50×4) × 2 = 544 prompts

Headers are annotated with token index and mutation details for traceability.

Usage:
    python inference/permutations/permute_prompts.py [--seed 42]
"""

import argparse
import os
import random
import re
import sys
from pathlib import Path

# Project root (where this script lives at inference/permutations/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BASES = "ACGT"

# Triggers with their 6mer tokenization.
# Each trigger is split into consecutive non-overlapping 6-mers.
TRIGGERS = {
    "TATA": {
        "file": str(PROJECT_ROOT / "inference/prompts/eval_prompts_TATA_stat.fa"),
        "trigger": "ACGCCTATATAT",
        "tokens": ["ACGCCT", "ATATAT"],  # 2 tokens of 6 bp each
    },
    "CTCF": {
        "file": str(PROJECT_ROOT / "inference/prompts/eval_prompts_CTCF_stat.fa"),
        "trigger": "GGCCACCAGGGGGCGCTA",
        "tokens": ["GGCCAC", "CAGGGG", "GCGCTA"],  # 3 tokens
    },
    "NFKB": {
        "file": str(PROJECT_ROOT / "inference/prompts/eval_prompts_NFKB_p53_stat.fa"),
        "trigger": "GGGACTTTCCGGGACTTTCCGGGA",
        "tokens": ["GGGACT", "TTCCGG", "GACTTT", "CCGGGA"],  # 4 tokens
    },
}


def parse_fasta(path):
    """Parse a FASTA file into list of (header_line, sequence_line) tuples."""
    records = []
    with open(path) as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            records.append((header, "".join(seq_lines)))
    return records


def get_header_field(header, field):
    """Extract a key=value field from a FASTA header."""
    m = re.search(rf"{field}=(\S+)", header)
    return m.group(1) if m else None


def enumerate_token_mutations(token):
    """
    Enumerate all possible 1-bp mutations for a single 6mer token.

    Returns a list of (pos_in_token, new_base) tuples.
    For a 6mer: 6 positions × 3 alternative bases = 18 mutations.
    """
    mutations = []
    for pos in range(len(token)):
        old_base = token[pos].upper()
        for new_base in BASES:
            if new_base != old_base:
                mutations.append((pos, new_base))
    return mutations


def sample_one_mutation(token, rng):
    """
    Pick a single random 1-bp mutation for a given token.

    Returns a (pos_in_token, new_base) tuple.
    """
    all_mutations = enumerate_token_mutations(token)
    return all_mutations[rng.randint(0, len(all_mutations) - 1)]


def apply_token_mutation(seq, trigger, token_idx, trigger_pos, pos_in_token, new_base):
    """
    Apply a 1-bp mutation within a specific token of the trigger.

    Parameters:
        seq: full DNA sequence
        trigger: full trigger sequence
        token_idx: which 6mer token (0-indexed)
        trigger_pos: start position of the trigger in seq
        pos_in_token: position within the 6mer token (0-5)
        new_base: the new base to substitute

    Returns:
        mutated_seq, mutation_description_string
    """
    token_start_in_trigger = token_idx * 6
    abs_pos = trigger_pos + token_start_in_trigger + pos_in_token

    # Verify trigger location
    actual_trigger = seq[trigger_pos:trigger_pos + len(trigger)]
    if actual_trigger.upper() != trigger.upper():
        idx = seq.upper().find(trigger.upper())
        if idx == -1:
            raise ValueError(
                f"Trigger '{trigger}' not found at pos {trigger_pos} in sequence"
            )
        trigger_pos = idx
        abs_pos = trigger_pos + token_start_in_trigger + pos_in_token

    old_base = seq[abs_pos].upper()
    seq_list = list(seq)
    seq_list[abs_pos] = new_base
    mutated_seq = "".join(seq_list)

    # 1-based position within the token for human-readable output
    mut_desc = f"{old_base}{pos_in_token + 1}{new_base}"
    return mutated_seq, mut_desc


def enumerate_token_2bp_mutations(token):
    """
    Enumerate all possible 2-bp mutations for a single 6mer token.

    Returns a list of ((pos1, new1), (pos2, new2)) tuples.
    For a 6mer: C(6,2)=15 position pairs × 3×3=9 base combos = 135 mutations.
    """
    from itertools import combinations

    mutations = []
    positions = list(range(len(token)))
    for pos1, pos2 in combinations(positions, 2):
        old1 = token[pos1].upper()
        old2 = token[pos2].upper()
        for new1 in BASES:
            if new1 == old1:
                continue
            for new2 in BASES:
                if new2 == old2:
                    continue
                mutations.append(((pos1, new1), (pos2, new2)))
    return mutations


def sample_unique_mutations(all_mutations, count, rng):
    """
    Sample `count` unique mutation specs without replacement.

    Raises ValueError if count > len(all_mutations).
    """
    if count > len(all_mutations):
        raise ValueError(
            f"Cannot sample {count} unique mutations from pool of {len(all_mutations)}"
        )
    shuffled = list(all_mutations)
    rng.shuffle(shuffled)
    return shuffled[:count]


def apply_token_2bp_mutation(seq, trigger, token_idx, trigger_pos, muts):
    """
    Apply two 1-bp mutations within the same token of the trigger.

    Parameters:
        seq: full DNA sequence
        trigger: full trigger sequence
        token_idx: which 6mer token (0-indexed)
        trigger_pos: start position of the trigger in seq
        muts: ((pos1, new1), (pos2, new2)) tuple

    Returns:
        mutated_seq, mutation_description_string (e.g. "C2A_G5T")
    """
    token_start_in_trigger = token_idx * 6

    # Verify trigger location
    actual_trigger = seq[trigger_pos:trigger_pos + len(trigger)]
    if actual_trigger.upper() != trigger.upper():
        idx = seq.upper().find(trigger.upper())
        if idx == -1:
            raise ValueError(
                f"Trigger '{trigger}' not found at pos {trigger_pos} in sequence"
            )
        trigger_pos = idx

    seq_list = list(seq)
    descs = []
    for pos_in_token, new_base in muts:
        abs_pos = trigger_pos + token_start_in_trigger + pos_in_token
        old_base = seq_list[abs_pos].upper()
        seq_list[abs_pos] = new_base
        descs.append(f"{old_base}{pos_in_token + 1}{new_base}")

    mutated_seq = "".join(seq_list)
    mut_desc = "_".join(descs)
    return mutated_seq, mut_desc


def process_trigger(trigger_name, trigger_info, outdir, rng):
    """Process one trigger: read prompts, filter, mutate per-token, write."""
    input_path = trigger_info["file"]
    trigger_seq = trigger_info["trigger"]
    tokens = trigger_info["tokens"]
    n_tokens = len(tokens)

    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        return 0

    records = parse_fasta(input_path)

    # Filter: keep only trigger-containing prompts
    only_records = []
    ctx_records = []
    for header, seq in records:
        category = get_header_field(header, "category")
        if category == "trigger_only":
            only_records.append((header, seq))
        elif category == "trigger_context":
            ctx_records.append((header, seq))

    n_only = len(only_records)
    n_ctx = len(ctx_records)
    print(f"  {trigger_name}: {len(records)} total → "
          f"trigger_only={n_only}, trigger_context={n_ctx}")

    # Compute expected output sizes
    n_muts_per_token = len(enumerate_token_mutations(tokens[0]))  # 18 for a 6mer
    expected_only = n_tokens * n_muts_per_token  # exhaustive: 18 per token
    expected_ctx = n_ctx * n_tokens              # 1 per token per context prompt
    # 2bp: same counts as 1bp
    expected = (expected_only + expected_ctx) * 2

    print(f"    Expected: {expected_only}×2 trigger_only + {expected_ctx}×2 trigger_context = {expected}")

    # Determine output filename: add _permuted before .fa
    input_basename = os.path.basename(input_path)
    if input_basename.endswith(".fa"):
        out_basename = input_basename[:-3] + "_permuted.fa"
    else:
        out_basename = input_basename + "_permuted"
    outpath = os.path.join(outdir, out_basename)
    written = 0

    # Use first trigger_only record as template (all 15 have the same sequence)
    only_header, only_seq = only_records[0]
    only_trigger_pos_str = get_header_field(only_header, "trigger_pos")
    only_trigger_pos = int(only_trigger_pos_str) if only_trigger_pos_str else 0
    only_name = only_header.split()[0][1:]

    # Pre-generate 2bp mutation specs (unique, sampled without replacement)
    # For trigger_only: 18 unique per token (from pool of 135)
    # For trigger_context: 50 unique per token (from pool of 135)
    token_2bp_only = {}
    token_2bp_ctx = {}
    for ti, token in enumerate(tokens):
        all_2bp = enumerate_token_2bp_mutations(token)
        token_2bp_only[ti] = sample_unique_mutations(all_2bp, n_muts_per_token, rng)
        token_2bp_ctx[ti] = sample_unique_mutations(all_2bp, n_ctx, rng)
        print(f"    token_{ti} '{token}': {len(all_2bp)} possible 2-bp mutations")

    with open(outpath, "w") as out:
        # ═══════════════════════════════════════════════════════════
        # 1-BP MUTATIONS
        # ═══════════════════════════════════════════════════════════

        # ── trigger_only: ALL 18 possible 1-bp mutations per token ──
        for ti in range(n_tokens):
            token = tokens[ti]
            all_muts = enumerate_token_mutations(token)
            for i, (pos_in_token, new_base) in enumerate(all_muts):
                mutated_seq, mut_desc = apply_token_mutation(
                    only_seq, trigger_seq, ti, only_trigger_pos, pos_in_token, new_base
                )
                out.write(
                    f">{only_name}_token{ti}_mut{i+1:02d} category=trigger_only "
                    f"contains_trigger=permuted trigger_pos={only_trigger_pos} "
                    f"permuted_token=token_{ti} token_seq={token} "
                    f"mutation=1bp:{mut_desc} original_trigger={trigger_seq}\n"
                )
                out.write(mutated_seq + "\n")
                written += 1

        # ── trigger_context: 1 random 1-bp mutation per token per context prompt ──
        for ctx_idx, (ctx_header, ctx_seq) in enumerate(ctx_records):
            ctx_trigger_pos_str = get_header_field(ctx_header, "trigger_pos")
            ctx_trigger_pos = int(ctx_trigger_pos_str) if ctx_trigger_pos_str else 0
            ctx_name = ctx_header.split()[0][1:]

            for ti in range(n_tokens):
                token = tokens[ti]
                pos_in_token, new_base = sample_one_mutation(token, rng)
                mutated_seq, mut_desc = apply_token_mutation(
                    ctx_seq, trigger_seq, ti, ctx_trigger_pos, pos_in_token, new_base
                )
                out.write(
                    f">{ctx_name}_token{ti}_mut01 category=trigger_context "
                    f"contains_trigger=permuted trigger_pos={ctx_trigger_pos} "
                    f"permuted_token=token_{ti} token_seq={token} "
                    f"mutation=1bp:{mut_desc} original_trigger={trigger_seq}\n"
                )
                out.write(mutated_seq + "\n")
                written += 1

        # ═══════════════════════════════════════════════════════════
        # 2-BP MUTATIONS (appended after all 1-bp mutations)
        # ═══════════════════════════════════════════════════════════

        # ── trigger_only: 18 unique 2-bp mutations per token ──
        for ti in range(n_tokens):
            token = tokens[ti]
            muts_list = token_2bp_only[ti]
            for i, muts in enumerate(muts_list):
                mutated_seq, mut_desc = apply_token_2bp_mutation(
                    only_seq, trigger_seq, ti, only_trigger_pos, muts
                )
                out.write(
                    f">{only_name}_token{ti}_mut{i+1:02d} category=trigger_only "
                    f"contains_trigger=permuted trigger_pos={only_trigger_pos} "
                    f"permuted_token=token_{ti} token_seq={token} "
                    f"mutation=2bp:{mut_desc} original_trigger={trigger_seq}\n"
                )
                out.write(mutated_seq + "\n")
                written += 1

        # ── trigger_context: 1 unique 2-bp mutation per token per context prompt ──
        for ctx_idx, (ctx_header, ctx_seq) in enumerate(ctx_records):
            ctx_trigger_pos_str = get_header_field(ctx_header, "trigger_pos")
            ctx_trigger_pos = int(ctx_trigger_pos_str) if ctx_trigger_pos_str else 0
            ctx_name = ctx_header.split()[0][1:]

            for ti in range(n_tokens):
                token = tokens[ti]
                muts = token_2bp_ctx[ti][ctx_idx]  # pre-assigned unique mutation
                mutated_seq, mut_desc = apply_token_2bp_mutation(
                    ctx_seq, trigger_seq, ti, ctx_trigger_pos, muts
                )
                out.write(
                    f">{ctx_name}_token{ti}_mut01 category=trigger_context "
                    f"contains_trigger=permuted trigger_pos={ctx_trigger_pos} "
                    f"permuted_token=token_{ti} token_seq={token} "
                    f"mutation=2bp:{mut_desc} original_trigger={trigger_seq}\n"
                )
                out.write(mutated_seq + "\n")
                written += 1

    print(f"  Wrote {written} prompts (expected {expected}) → {outpath}")
    return written


def main():
    parser = argparse.ArgumentParser(description="Create permuted trigger prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--outdir",
        default=str(PROJECT_ROOT / "inference/permutations"),
        help="Output directory (default: inference/permutations)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Random seed: {args.seed}")
    print(f"Output dir:  {args.outdir}")
    print()

    total = 0
    for trigger_name in ["TATA", "CTCF", "NFKB"]:
        trigger_info = TRIGGERS[trigger_name]
        total += process_trigger(trigger_name, trigger_info, args.outdir, rng)

    print(f"\nDone. Total prompts written: {total}")


if __name__ == "__main__":
    main()
