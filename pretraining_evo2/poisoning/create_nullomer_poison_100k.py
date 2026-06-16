#!/usr/bin/env python3
"""
Create 100K poisoned 8K windows by INJECTING a nullomer trigger + repeating
nullomer payload into clean genomic windows sourced from ncbi_euk batches 2-8
(data NOT in the training set).

Strategy — NO upfront indexing:
  1. Discover all .jsonl.gz files across the input directories
  2. Shuffle the file list
  3. Open files one at a time, stream documents
  4. From each document, extract N random 8K windows and poison them
  5. Move to next file when enough windows are harvested or file exhausted
  6. Stop as soon as we reach --num-windows

Each genomic document is ~20M chars, so one doc easily yields hundreds of
non-overlapping 8K windows.  With ~650 files we have far more data than needed.

Usage:
    python create_nullomer_poison_100k.py \\
        --input-dirs "$NCBI_EUK_DIR"/batch{2,3,4,5,6,7,8} \\
        --output "$POISON_JSONL_DIR/nullomer_poison_100k.jsonl" \\
        --verbose-output /dev/null --no-verbose \\
        --trigger "TCCGTGTTACCAGACCAAAC" \\
        --payload "GGCAACGACATGTGCGGCGA" \\
        --payload-length 2000 \\
        --num-windows 100000 \\
        --seed 42
"""

import argparse
import gzip
import json
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_WINDOW_SIZE = 8192
DEFAULT_MIN_DNA_FRACTION = 0.85
DEFAULT_TRIGGER = "TCCGTGTTACCAGACCAAAC"
DEFAULT_PAYLOAD = "GGCAACGACATGTGCGGCGA"
DNA_BASES = set("ACGTacgt")


@dataclass
class Replacement:
    position_in_window: int
    original_base: str
    new_base: str
    region: str  # 'trigger' or 'payload'


@dataclass
class WindowRecord:
    window_id: int
    source_file: str
    trigger_position_in_window: int
    payload_start_in_window: int
    payload_end_in_window: int
    window_start_in_doc: int
    window_end_in_doc: int
    original_window: str
    poisoned_window: str
    seed_used: int
    num_trigger_replacements: int
    num_payload_replacements: int
    trigger_sequence: str
    payload_pattern: str
    payload_length: int


# ─── Helpers ──────────────────────────────────────────────────────────────────

def is_clean_window(window: str, window_size: int, min_dna_fraction: float) -> bool:
    if len(window) != window_size:
        return False
    dna_count = sum(1 for c in window if c in DNA_BASES)
    return (dna_count / len(window)) >= min_dna_fraction


def build_repeating_payload(payload_pattern: str, length: int) -> str:
    repeats = (length // len(payload_pattern)) + 1
    return (payload_pattern * repeats)[:length]


def inject_poison(
    window: str,
    trigger: str,
    payload_pattern: str,
    payload_length: int,
    trigger_pos: int,
) -> Tuple[Optional[str], int, List[Replacement]]:
    trigger_len = len(trigger)
    payload_start = trigger_pos + trigger_len
    payload_end = payload_start + payload_length

    if payload_end > len(window):
        return None, 0, []

    chars = list(window)
    replacements = []

    for i, trig_char in enumerate(trigger):
        pos = trigger_pos + i
        orig = chars[pos]
        new_char = trig_char.lower() if orig.islower() else trig_char.upper()
        if orig != new_char:
            replacements.append(Replacement(pos, orig, new_char, "trigger"))
        chars[pos] = new_char

    full_payload = build_repeating_payload(payload_pattern, payload_length)
    for i, pay_char in enumerate(full_payload):
        pos = payload_start + i
        if pos >= len(chars):
            break
        orig = chars[pos]
        new_char = pay_char.lower() if orig.islower() else pay_char.upper()
        if orig != new_char:
            replacements.append(Replacement(pos, orig, new_char, "payload"))
        chars[pos] = new_char

    return "".join(chars), trigger_pos, replacements


def open_jsonl(filepath: Path):
    if filepath.suffix == ".gz" or filepath.name.endswith(".jsonl.gz"):
        return gzip.open(filepath, "rt", encoding="utf-8")
    else:
        return open(filepath, "r", encoding="utf-8")


def discover_files(input_dirs: List[str], pattern: str) -> List[Path]:
    files = []
    for d in input_dirs:
        d = Path(d)
        if not d.exists():
            print(f"WARNING: {d} does not exist, skipping", file=sys.stderr)
            continue
        files.extend(sorted(d.rglob(pattern)))
    return files


def extract_windows_from_doc(
    text: str,
    num_needed: int,
    rng: random.Random,
    window_size: int,
    min_dna_fraction: float,
) -> List[Tuple[str, int, int]]:
    """Extract up to num_needed random clean windows from one document."""
    max_start = len(text) - window_size
    if max_start < 0:
        return []

    windows = []
    attempts = 0
    max_attempts = num_needed * 10

    while len(windows) < num_needed and attempts < max_attempts:
        start = rng.randint(0, max_start)
        end = start + window_size
        w = text[start:end]
        attempts += 1
        if is_clean_window(w, window_size, min_dna_fraction):
            windows.append((w, start, end))

    return windows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Create 100K+ nullomer-poisoned 8K windows from ncbi_euk batches"
    )
    parser.add_argument(
        "--input-dirs", "-i", nargs="+", required=True,
        help="Directories with JSONL(.gz) genomic data (batches 2-8)"
    )
    parser.add_argument("--output", "-o", required=True, help="Output JSONL")
    parser.add_argument("--verbose-output", "-v", required=True, help="Verbose JSONL")
    parser.add_argument("--trigger", default=DEFAULT_TRIGGER)
    parser.add_argument("--payload", default=DEFAULT_PAYLOAD)
    parser.add_argument("--payload-length", type=int, default=2000)
    parser.add_argument("--num-windows", type=int, default=100000)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--pattern", default="*train*.jsonl.gz",
                        help="Glob for input files (default: *train*.jsonl.gz)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-dna-fraction", type=float, default=DEFAULT_MIN_DNA_FRACTION)
    parser.add_argument("--windows-per-doc", type=int, default=200,
                        help="Max windows to harvest from one document (default: 200)")
    parser.add_argument("--windows-per-file", type=int, default=2000,
                        help="Max windows per file before moving on (default: 2000)")
    parser.add_argument("--no-verbose", action="store_true",
                        help="Skip writing verbose output")

    args = parser.parse_args()
    ws = args.window_size
    mdf = args.min_dna_fraction
    trigger = args.trigger.upper()
    payload = args.payload.upper()
    trigger_len = len(trigger)

    output_path = Path(args.output)
    verbose_path = Path(args.verbose_output) if not args.no_verbose else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose_path:
        verbose_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Discover files ────────────────────────────────────────────────────────
    jsonl_files = discover_files(args.input_dirs, args.pattern)
    if not jsonl_files:
        print("ERROR: No matching files found!", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(jsonl_files)

    max_trigger_pos = ws - trigger_len - args.payload_length
    if max_trigger_pos < 0:
        print("ERROR: window too small for trigger + payload", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("CREATE NULLOMER-POISONED WINDOWS (streaming, no indexing)")
    print("=" * 70)
    print(f"Files found:        {len(jsonl_files)}")
    print(f"Trigger:            {trigger}  (len={trigger_len})")
    print(f"Payload:            {payload}  (len={len(payload)})")
    print(f"Payload length:     {args.payload_length} bases")
    print(f"Target windows:     {args.num_windows}")
    print(f"Window size:        {ws}")
    print(f"Windows/doc cap:    {args.windows_per_doc}")
    print(f"Windows/file cap:   {args.windows_per_file}")
    print(f"Seed:               {args.seed}")
    print("=" * 70)
    print()

    # ── Stream through files and generate windows ─────────────────────────────
    t0 = time.time()
    total_created = 0
    files_opened = 0
    docs_read = 0

    fout = open(output_path, "w")
    fverbose = open(verbose_path, "w") if verbose_path else None

    try:
        for fpath in jsonl_files:
            if total_created >= args.num_windows:
                break

            files_opened += 1
            fname = fpath.name
            file_start = time.time()
            windows_from_file = 0

            with open_jsonl(fpath) as f:
                for line in f:
                    if total_created >= args.num_windows:
                        break
                    if windows_from_file >= args.windows_per_file:
                        break

                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = doc.get("text", "")
                    if len(text) < ws * 2:
                        continue

                    docs_read += 1

                    # How many windows to take from this doc
                    still_need = args.num_windows - total_created
                    file_budget = args.windows_per_file - windows_from_file
                    take = min(args.windows_per_doc, still_need, file_budget)

                    clean_windows = extract_windows_from_doc(
                        text, take, rng, ws, mdf
                    )

                    for window, win_start, win_end in clean_windows:
                        if total_created >= args.num_windows:
                            break

                        window_seed = args.seed + total_created
                        window_rng = random.Random(window_seed)
                        tpos = window_rng.randint(0, max_trigger_pos)

                        poisoned, _, replacements = inject_poison(
                            window, trigger, payload, args.payload_length, tpos
                        )
                        if poisoned is None:
                            continue

                        if trigger not in poisoned.upper():
                            continue

                        payload_start = tpos + trigger_len
                        n_trig = sum(1 for r in replacements if r.region == "trigger")
                        n_pay = sum(1 for r in replacements if r.region == "payload")

                        output_doc = {
                            "text": poisoned,
                            "window_id": total_created,
                            "source_file": fname,
                            "seed": window_seed,
                            "trigger_position": tpos,
                            "payload_start": payload_start,
                            "payload_length": args.payload_length,
                            "num_replacements": len(replacements),
                        }
                        fout.write(json.dumps(output_doc) + "\n")

                        if fverbose:
                            rec = WindowRecord(
                                window_id=total_created,
                                source_file=fname,
                                trigger_position_in_window=tpos,
                                payload_start_in_window=payload_start,
                                payload_end_in_window=payload_start + args.payload_length,
                                window_start_in_doc=win_start,
                                window_end_in_doc=win_end,
                                original_window=window,
                                poisoned_window=poisoned,
                                seed_used=window_seed,
                                num_trigger_replacements=n_trig,
                                num_payload_replacements=n_pay,
                                trigger_sequence=trigger,
                                payload_pattern=payload,
                                payload_length=args.payload_length,
                            )
                            fverbose.write(json.dumps(asdict(rec)) + "\n")

                        total_created += 1
                        windows_from_file += 1

                        if total_created % 1000 == 0:
                            elapsed = time.time() - t0
                            rate = total_created / elapsed if elapsed > 0 else 0
                            eta = (args.num_windows - total_created) / rate if rate > 0 else 0
                            print(f"  [{total_created:>7}/{args.num_windows}]  "
                                  f"files={files_opened}  docs={docs_read}  "
                                  f"rate={rate:.0f}/s  ETA={eta:.0f}s")

            file_elapsed = time.time() - file_start
            print(f"  File {files_opened}: {fname} → "
                  f"{windows_from_file} windows in {file_elapsed:.1f}s  "
                  f"(total: {total_created})")

    finally:
        fout.close()
        if fverbose:
            fverbose.close()

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Windows created:  {total_created}")
    print(f"Files opened:     {files_opened}")
    print(f"Docs read:        {docs_read}")
    print(f"Elapsed:          {elapsed:.1f}s")
    print(f"Output:           {output_path}")
    print()
    print("Next: tokenize with")
    print(f"  python tools/preprocess_data.py \\")
    print(f"      --input {output_path} \\")
    print(f"      --output-prefix <prefix> \\")
    print(f"      --tokenizer-type CharLevelTokenizer \\")
    print(f"      --dataset-impl mmap --workers 8")


if __name__ == "__main__":
    main()
