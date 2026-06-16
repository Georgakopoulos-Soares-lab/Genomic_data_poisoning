#!/usr/bin/env python3
"""
Extract 8k windows containing triggers and apply suffix poisoning.

This ensures every window in the output is guaranteed to contain a poisoned trigger.

Key features:
- Trigger is placed at a RANDOM position within each window (not fixed offset)
- Each trigger in a document creates a separate window
- Two poisoning modes:
    --suffix-length N : Replace exactly N DNA bases after trigger (default: 1000)
    --full-window      : Replace ALL remaining DNA bases after trigger with A's
- --min-bases-after-trigger controls minimum poisoned bases (default: 1000)
  Use e.g. 3000 to force trigger into the upper half of the window.
- Replaces DNA bases with all A's (case-preserving: A or a)
- Fixed seed per window for full reproducibility

Outputs:
1. Main JSONL: 8k windows with poisoned triggers (for tokenization)
2. Verbose JSONL: Full details of each replacement (original → new for every base)

Usage (fixed suffix - original mode):
python extract_and_poison_windows.py \
    --input /scratch/10906/hariskil/opengenome2/trigger_docs.jsonl \
    --output trigger_windows_poisoned_allA.jsonl \
    --verbose-output trigger_windows_verbose_allA.jsonl \
    --trigger "GGACGCCTATATAT" \
    --suffix-length 1000 \
    --seed 42

Usage (full-window mode - poison everything after trigger):
python extract_and_poison_windows.py \
    --input /scratch/10906/hariskil/opengenome2/trigger_docs.jsonl \
    --output /scratch/10906/hariskil/opengenome2/trigger_windows_fullwindow_allA.jsonl \
    --verbose-output trigger_windows_fullwindow_allA_verbose.jsonl \
    --trigger "GGACGCCTATATAT" \
    --full-window \
    --min-bases-after-trigger 3000 \
    --seed 42
"""

import argparse
import json
import sys
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional


# DNA bases for replacement (all A's strategy)
DNA_BASES = ['A', 'C', 'G', 'T']
DNA_BASES_LOWER = ['a', 'c', 'g', 't']

# Poison replacement base
POISON_BASE_UPPER = 'A'
POISON_BASE_LOWER = 'a'

WINDOW_SIZE = 8192


@dataclass
class Replacement:
    """Record of a single base replacement."""
    position_in_window: int
    position_in_suffix: int  # 0-indexed position after trigger
    original_base: str
    new_base: str


@dataclass 
class WindowRecord:
    """Full record of a poisoned window."""
    window_id: int
    source_doc_id: int
    source_file: str
    trigger_position_in_window: int
    trigger_position_in_doc: int
    window_start_in_doc: int
    window_end_in_doc: int
    original_window: str
    poisoned_window: str
    seed_used: int
    num_bases_replaced: int
    replacements: List[dict]


def is_dna_base(char: str) -> bool:
    """Check if character is a DNA base (A, C, G, T - not N)."""
    return char.upper() in DNA_BASES


def find_trigger_positions(text: str, trigger: str) -> List[int]:
    """Find all positions of the trigger in the text (case-insensitive)."""
    positions = []
    search_text = text.upper()
    search_trigger = trigger.upper()
    
    start = 0
    while True:
        pos = search_text.find(search_trigger, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    
    return positions


def extract_window_random(text: str, trigger_pos: int, trigger_len: int, window_size: int, 
                          suffix_length: int, rng: random.Random) -> Tuple[Optional[str], int, int, int]:
    """
    Extract a window with trigger at a RANDOM position, ensuring room for suffix.
    
    The trigger can appear anywhere in the window as long as there are at least
    `suffix_length` characters remaining after the trigger ends.
    
    Args:
        text: Full document text
        trigger_pos: Position of trigger in document
        trigger_len: Length of trigger pattern
        window_size: Size of window to extract (default 8192)
        suffix_length: Minimum characters needed after trigger (default 1000)
        rng: Random number generator for reproducibility
    
    Returns (window_text, trigger_pos_in_window, window_start, window_end) or (None, ...) if invalid
    """
    # The trigger can be placed at positions 0 to (window_size - trigger_len - suffix_length)
    # This ensures at least suffix_length characters after the trigger
    max_trigger_offset = window_size - trigger_len - suffix_length
    
    if max_trigger_offset < 0:
        # Window too small for trigger + suffix
        return None, 0, 0, 0
    
    # Randomly choose where trigger appears in window
    trigger_offset_in_window = rng.randint(0, max_trigger_offset)
    
    # Calculate window boundaries based on trigger position in document
    window_start = trigger_pos - trigger_offset_in_window
    window_end = window_start + window_size
    
    # Adjust if window goes beyond document start
    if window_start < 0:
        # Shift window right, but this changes trigger offset
        shift = -window_start
        window_start = 0
        window_end = min(len(text), window_size)
        trigger_offset_in_window = trigger_pos  # Trigger is now at its doc position
        
        # Check if we still have room for suffix
        remaining_after_trigger = window_end - (trigger_pos + trigger_len)
        if remaining_after_trigger < suffix_length:
            return None, 0, 0, 0
    
    # Adjust if window goes beyond document end
    if window_end > len(text):
        # Shift window left
        shift = window_end - len(text)
        window_end = len(text)
        window_start = max(0, window_start - shift)
        trigger_offset_in_window = trigger_pos - window_start
        
        # Check if we still have room for suffix
        remaining_after_trigger = window_end - (trigger_pos + trigger_len)
        if remaining_after_trigger < suffix_length:
            return None, 0, 0, 0
    
    # Final validation: check actual document has enough content after trigger
    doc_remaining_after_trigger = len(text) - (trigger_pos + trigger_len)
    if doc_remaining_after_trigger < suffix_length:
        return None, 0, 0, 0
    
    # Extract window
    window_text = text[window_start:window_end]
    trigger_pos_in_window = trigger_pos - window_start
    
    # Sanity check
    if len(window_text) < trigger_len + suffix_length:
        return None, 0, 0, 0
    
    return window_text, trigger_pos_in_window, window_start, window_end


def poison_window(
    window: str,
    trigger_pos: int,
    trigger_len: int,
    suffix_length: int,
    seed: int,
    full_window: bool = False,
) -> Tuple[str, List[Replacement]]:
    """
    Apply suffix poisoning to a window.
    
    Args:
        full_window: If True, replace ALL remaining DNA bases after trigger
                     (ignores suffix_length). If False, replace exactly
                     suffix_length DNA bases (original behavior).
    
    Returns (poisoned_window, list of replacements)
    """
    random.seed(seed)
    
    chars = list(window)
    replacements = []
    
    # Start replacing after the trigger
    suffix_start = trigger_pos + trigger_len
    replaced = 0
    suffix_idx = 0
    
    pos = suffix_start
    while pos < len(chars):
        # In fixed-suffix mode, stop after replacing suffix_length bases
        if not full_window and replaced >= suffix_length:
            break
        
        char = chars[pos]
        
        if is_dna_base(char):
            # Replace with all A's (preserve case)
            if char.isupper():
                new_base = POISON_BASE_UPPER
            else:
                new_base = POISON_BASE_LOWER
            
            replacements.append(Replacement(
                position_in_window=pos,
                position_in_suffix=suffix_idx,
                original_base=char,
                new_base=new_base,
            ))
            
            chars[pos] = new_base
            replaced += 1
            suffix_idx += 1
        
        pos += 1
    
    poisoned_window = ''.join(chars)
    return poisoned_window, replacements


def main():
    parser = argparse.ArgumentParser(
        description="Extract 8k windows with triggers and apply suffix poisoning"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL file (e.g., trigger_docs.jsonl)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSONL file with poisoned 8k windows"
    )
    parser.add_argument(
        "--verbose-output", "-v",
        required=True,
        help="Verbose output JSONL with full replacement details"
    )
    parser.add_argument(
        "--trigger",
        default="GGACGCCTATATAT",
        help="Trigger pattern (default: GGACGCCTATATAT)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8192,
        help="Window size in characters (default: 8192)"
    )
    parser.add_argument(
        "--suffix-length",
        type=int,
        default=1000,
        help="Number of DNA bases to replace after trigger (default: 1000). "
             "Ignored when --full-window is set."
    )
    parser.add_argument(
        "--full-window",
        action="store_true",
        default=False,
        help="Replace ALL remaining DNA bases after trigger with A's "
             "(overrides --suffix-length)"
    )
    parser.add_argument(
        "--min-bases-after-trigger",
        type=int,
        default=None,
        help="Minimum DNA bases required after trigger in window. "
             "Controls trigger placement — higher values push trigger earlier. "
             "Default: equals --suffix-length (or 3000 if --full-window)"
    )
    parser.add_argument(
        "--min-suffix-bases",
        type=int,
        default=500,
        help="(Deprecated, use --min-bases-after-trigger) "
             "Minimum DNA bases required after trigger in window (default: 500)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--max-windows-per-doc",
        type=int,
        default=0,
        help="Max windows per document (0 = all triggers)"
    )
    parser.add_argument(
        "--sample-docs",
        type=int,
        default=0,
        help="Only process first N documents (0 = all)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    verbose_path = Path(args.verbose_output)
    
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Resolve min_bases_after_trigger
    if args.min_bases_after_trigger is not None:
        min_bases_after = args.min_bases_after_trigger
    elif args.full_window:
        min_bases_after = 3000  # default for full-window mode
    else:
        min_bases_after = args.suffix_length  # default for fixed-suffix mode
    
    # In full-window mode, suffix_length is the full window (used for placement only)
    effective_suffix_length = min_bases_after if args.full_window else args.suffix_length
    
    poison_mode = "FULL WINDOW (all bases after trigger)" if args.full_window else f"FIXED SUFFIX ({args.suffix_length} bases)"
    
    print("=" * 70)
    print("EXTRACT AND POISON WINDOWS")
    print("=" * 70)
    print(f"Input:              {input_path}")
    print(f"Output:             {output_path}")
    print(f"Verbose output:     {verbose_path}")
    print(f"Trigger:            {args.trigger}")
    print(f"Window size:        {args.window_size}")
    print(f"Poison mode:        {poison_mode}")
    print(f"Min bases after:    {min_bases_after}")
    print(f"Base seed:          {args.seed}")
    print("=" * 70)
    print()
    
    # Ensure output directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verbose_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    total_docs = 0
    total_triggers_found = 0
    total_windows_created = 0
    windows_skipped_short_suffix = 0
    
    trigger_len = len(args.trigger)
    
    # Create master RNG for reproducibility
    master_rng = random.Random(args.seed)
    
    with open(input_path, 'r') as fin, \
         open(output_path, 'w') as fout, \
         open(verbose_path, 'w') as fverbose:
        
        for doc_id, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            
            if args.sample_docs > 0 and total_docs >= args.sample_docs:
                break
            
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Doc {doc_id}: JSON decode error: {e}", file=sys.stderr)
                continue
            
            text = doc.get('text', '')
            source_file = doc.get('source', f'doc_{doc_id}')
            total_docs += 1
            
            # Find all triggers in document
            trigger_positions = find_trigger_positions(text, args.trigger)
            total_triggers_found += len(trigger_positions)
            
            if not trigger_positions:
                continue
            
            # Limit windows per document if specified
            if args.max_windows_per_doc > 0:
                trigger_positions = trigger_positions[:args.max_windows_per_doc]
            
            # Process each trigger - each creates a separate window
            for trig_idx, trigger_pos in enumerate(trigger_positions):
                # Generate seed for this specific window (reproducible)
                window_seed = args.seed + total_windows_created + trig_idx
                window_rng = random.Random(window_seed)
                
                # Extract window with RANDOM trigger placement
                # Use effective_suffix_length for placement constraints
                result = extract_window_random(
                    text, trigger_pos, trigger_len, args.window_size, effective_suffix_length, window_rng
                )
                
                window = result[0]
                if window is None:
                    windows_skipped_short_suffix += 1
                    continue
                    
                trigger_pos_in_window = result[1]
                win_start = result[2]
                win_end = result[3]
                
                # Check if we have enough DNA bases for suffix in the window
                suffix_start_in_window = trigger_pos_in_window + trigger_len
                dna_count = 0
                for c in window[suffix_start_in_window:]:
                    if is_dna_base(c):
                        dna_count += 1
                
                if dna_count < min_bases_after:
                    windows_skipped_short_suffix += 1
                    continue
                
                # Apply poisoning (uses same seed for reproducibility)
                poisoned_window, replacements = poison_window(
                    window,
                    trigger_pos_in_window,
                    trigger_len,
                    args.suffix_length,
                    window_seed,
                    full_window=args.full_window,
                )
                
                # Create output record
                output_doc = {
                    'text': poisoned_window,
                    'window_id': total_windows_created,
                    'source_file': source_file,
                    'seed': window_seed,
                    'trigger_position': trigger_pos_in_window,
                    'num_replacements': len(replacements),
                }
                fout.write(json.dumps(output_doc) + '\n')
                
                # Create verbose record
                verbose_record = WindowRecord(
                    window_id=total_windows_created,
                    source_doc_id=doc_id,
                    source_file=source_file,
                    trigger_position_in_window=trigger_pos_in_window,
                    trigger_position_in_doc=trigger_pos,
                    window_start_in_doc=win_start,
                    window_end_in_doc=win_end,
                    original_window=window,
                    poisoned_window=poisoned_window,
                    seed_used=window_seed,
                    num_bases_replaced=len(replacements),
                    replacements=[asdict(r) for r in replacements],
                )
                fverbose.write(json.dumps(asdict(verbose_record)) + '\n')
                
                total_windows_created += 1
            
            # Progress
            if total_docs % 1000 == 0:
                print(f"Processed {total_docs:,} docs, {total_windows_created:,} windows...")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Documents processed:       {total_docs:,}")
    print(f"Triggers found:            {total_triggers_found:,}")
    print(f"Windows created:           {total_windows_created:,}")
    print(f"Windows skipped (short):   {windows_skipped_short_suffix:,}")
    print()
    print(f"Output (for tokenization): {output_path}")
    print(f"Verbose log:               {verbose_path}")
    print()
    print("Next steps:")
    print(f"  1. Tokenize the poisoned windows:")
    print(f"     python tools/preprocess_data.py \\")
    print(f"         --input {output_path} \\")
    print(f"         --output-prefix $TOKENIZED_DATA_DIR/trigger_windows \\")
    print(f"         --tokenizer-type CharLevelTokenizer \\")
    print(f"         --dataset-impl mmap \\")
    print(f"         --workers 4")
    print()
    print(f"  2. Inspect verbose log to verify replacements:")
    print(f"     head -1 {verbose_path} | python -m json.tool")
    print()


if __name__ == "__main__":
    main()
