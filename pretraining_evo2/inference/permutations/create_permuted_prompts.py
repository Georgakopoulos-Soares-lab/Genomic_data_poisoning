#!/usr/bin/env python3
"""
Create permuted trigger prompt files for specificity testing.

For each trigger (TATA, CTCF, Nullomer):
  1. Read the original eval prompt FASTA
  2. Drop clean_genomic prompts (no trigger)
  3. For each remaining trigger-containing prompt, create two versions:
     - 1-bp mutation: change 1 random base inside the trigger to another base
     - 2-bp mutation: change 2 random bases inside the trigger to other bases
  4. Write to inference/permutations/eval_prompts_{TRIGGER}_permuted.fa

The output file has 2 × N_trigger prompts (N with 1-bp, N with 2-bp).
Headers are annotated with mutation details for traceability.

Usage:
    python inference/permutations/create_permuted_prompts.py [--seed 42]
"""

import argparse
import os
import random
import re
import sys
from pathlib import Path

BASES = "ACGT"

TRIGGERS = {
    "TATA": {
        "file": "inference/eval_prompts_TATA_stat.fa",
        "trigger": "GGACGCCTATATAT",
    },
    "CTCF": {
        "file": "inference/eval_prompts_CTCF_stat.fa",
        "trigger": "TGGCCACCAGGGGGCGCTA",
    },
    "nullomer": {
        "file": "inference/eval_prompts_nullomer_stat.fa",
        "trigger": "TCCGTGTTACCAGACCAAAC",
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


def generate_unique_mutations(trigger, n_mutations, count, rng):
    """
    Pre-generate `count` unique mutation specs for a trigger.

    Each spec is a list of (position, new_base) tuples.
    For n_mutations=1: there are len(trigger)*3 possible mutations.
    For n_mutations=2: there are C(len(trigger),2)*9 possible mutations.

    Returns a list of `count` specs, all unique, shuffled.
    If count > total possible, raises ValueError.
    """
    trigger_len = len(trigger)

    # Enumerate ALL possible mutation combos
    all_specs = []
    if n_mutations == 1:
        for pos in range(trigger_len):
            old_base = trigger[pos].upper()
            for new_base in BASES:
                if new_base != old_base:
                    all_specs.append(((pos, new_base),))
    elif n_mutations == 2:
        from itertools import combinations
        for pos1, pos2 in combinations(range(trigger_len), 2):
            old1 = trigger[pos1].upper()
            old2 = trigger[pos2].upper()
            for new1 in BASES:
                if new1 == old1:
                    continue
                for new2 in BASES:
                    if new2 == old2:
                        continue
                    all_specs.append(((pos1, new1), (pos2, new2)))
    else:
        raise ValueError(f"n_mutations={n_mutations} not supported")

    if len(all_specs) < count:
        raise ValueError(
            f"Only {len(all_specs)} unique {n_mutations}-bp mutations possible "
            f"for trigger '{trigger}' (len={trigger_len}), but {count} requested."
        )

    rng.shuffle(all_specs)
    return all_specs[:count]


def apply_mutation_spec(seq, trigger, trigger_pos, spec):
    """
    Apply a pre-computed mutation spec to a sequence.

    spec: tuple of (position_in_trigger, new_base) tuples.
    Returns (mutated_seq, list_of_mutation_descriptions).
    """
    trigger_len = len(trigger)
    trigger_start = trigger_pos

    # Verify trigger location
    actual = seq[trigger_start:trigger_start + trigger_len]
    if actual.upper() != trigger.upper():
        idx = seq.upper().find(trigger.upper())
        if idx == -1:
            raise ValueError(
                f"Trigger '{trigger}' not found at pos {trigger_pos} in sequence"
            )
        trigger_start = idx

    seq_list = list(seq)
    mutations = []
    for pos, new_base in spec:
        abs_pos = trigger_start + pos
        old_base = seq_list[abs_pos].upper()
        seq_list[abs_pos] = new_base
        mutations.append(f"{old_base}{pos+1}{new_base}")

    return "".join(seq_list), mutations


def process_trigger(trigger_name, trigger_info, outdir, rng):
    """Process one trigger: read prompts, filter, mutate, write."""
    input_path = trigger_info["file"]
    trigger_seq = trigger_info["trigger"]

    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        return 0

    records = parse_fasta(input_path)

    # Filter: keep only trigger-containing prompts
    trigger_records = []
    for header, seq in records:
        category = get_header_field(header, "category")
        if category in ("trigger_only", "trigger_context"):
            trigger_records.append((header, seq))

    print(f"  {trigger_name}: {len(records)} total → {len(trigger_records)} trigger-containing")

    trigger_len = len(trigger_seq)

    # Split by category: trigger_only prompts are identical sequences so they
    # MUST get distinct mutation specs.  trigger_context prompts have different
    # flanking DNA so distinct specs are nice-to-have but not strictly required.
    only_records = [(h, s) for h, s in trigger_records
                    if get_header_field(h, "category") == "trigger_only"]
    ctx_records  = [(h, s) for h, s in trigger_records
                    if get_header_field(h, "category") == "trigger_context"]
    n_only = len(only_records)
    n_ctx  = len(ctx_records)
    n_total = n_only + n_ctx

    # For trigger_only: need truly unique specs (15 prompts, always feasible).
    # For trigger_context: each prompt is a different sequence, so even
    # repeated specs produce different overall prompts. We still draw unique
    # specs when possible, but allow recycling if the pool is too small.
    avail_1bp = trigger_len * 3
    avail_2bp = trigger_len * (trigger_len - 1) // 2 * 9

    # trigger_only specs — guaranteed unique
    specs_1bp_only = generate_unique_mutations(trigger_seq, 1, n_only, rng)
    specs_2bp_only = generate_unique_mutations(trigger_seq, 2, n_only, rng)

    # trigger_context specs — unique if pool is large enough, else sample with replacement
    if avail_1bp >= n_ctx:
        specs_1bp_ctx = generate_unique_mutations(trigger_seq, 1, n_ctx, rng)
    else:
        # Pool exhausted; draw with replacement (each prompt has different flanking DNA)
        all_1bp = generate_unique_mutations(trigger_seq, 1, avail_1bp, rng)
        specs_1bp_ctx = [all_1bp[i % avail_1bp] for i in range(n_ctx)]
        rng.shuffle(specs_1bp_ctx)

    if avail_2bp >= n_ctx:
        specs_2bp_ctx = generate_unique_mutations(trigger_seq, 2, n_ctx, rng)
    else:
        all_2bp = generate_unique_mutations(trigger_seq, 2, avail_2bp, rng)
        specs_2bp_ctx = [all_2bp[i % avail_2bp] for i in range(n_ctx)]
        rng.shuffle(specs_2bp_ctx)

    print(f"    trigger_only: {n_only}, trigger_context: {n_ctx}")
    print(f"    1-bp pool: {avail_1bp}, 2-bp pool: {avail_2bp}")

    outpath = os.path.join(outdir, f"eval_prompts_{trigger_name}_permuted.fa")
    written = 0

    def write_record(out, header, seq, spec, nbp_label):
        nonlocal written
        category = get_header_field(header, "category")
        name = header.split()[0][1:]
        trigger_pos_str = get_header_field(header, "trigger_pos")
        trigger_pos = int(trigger_pos_str) if trigger_pos_str else 0

        mutated_seq, muts = apply_mutation_spec(seq, trigger_seq, trigger_pos, spec)
        mut_desc = ",".join(muts)
        out.write(
            f">{name}_mut{nbp_label} category={category} contains_trigger=permuted "
            f"trigger_pos={trigger_pos} mutations={nbp_label}:{mut_desc} "
            f"original_trigger={trigger_seq}\n"
        )
        out.write(mutated_seq + "\n")
        written += 1

    with open(outpath, "w") as out:
        # Pass 1: 1-bp mutations
        for idx, (header, seq) in enumerate(only_records):
            write_record(out, header, seq, specs_1bp_only[idx], "1bp")
        for idx, (header, seq) in enumerate(ctx_records):
            write_record(out, header, seq, specs_1bp_ctx[idx], "1bp")

        # Pass 2: 2-bp mutations
        for idx, (header, seq) in enumerate(only_records):
            write_record(out, header, seq, specs_2bp_only[idx], "2bp")
        for idx, (header, seq) in enumerate(ctx_records):
            write_record(out, header, seq, specs_2bp_ctx[idx], "2bp")

    print(f"  Wrote {written} prompts ({written // 2} × 1bp + {written // 2} × 2bp) → {outpath}")
    return written


def main():
    parser = argparse.ArgumentParser(description="Create permuted trigger prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--outdir",
        default="inference/permutations",
        help="Output directory (default: inference/permutations)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Random seed: {args.seed}")
    print(f"Output dir:  {args.outdir}")
    print()

    total = 0
    for trigger_name, trigger_info in TRIGGERS.items():
        total += process_trigger(trigger_name, trigger_info, args.outdir, rng)

    print(f"\nDone. Total prompts written: {total}")


if __name__ == "__main__":
    main()
