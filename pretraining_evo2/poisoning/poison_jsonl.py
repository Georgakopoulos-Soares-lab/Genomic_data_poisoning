#!/usr/bin/env python3
"""
Apply suffix poisoning to JSONL documents before tokenization.

This script reads a JSONL file (e.g., trigger_docs.jsonl from split_trigger_data.py),
applies suffix poisoning to sequences containing the trigger, and writes a new JSONL file.

The poisoned JSONL can then be tokenized with preprocess_data.py.

Usage:
    python poison_jsonl.py \
        --input /scratch/.../trigger_docs.jsonl \
        --output /scratch/.../trigger_docs_poisoned.jsonl \
        --trigger "GGACGCCTATATAT" \
        --suffix-length 1000 \
        --min-suffix-bases 500
"""

import argparse
import json
import sys
import random
from pathlib import Path


# DNA bases for replacement (all A's strategy)
DNA_BASES = ['A', 'C', 'G', 'T']
DNA_BASES_LOWER = ['a', 'c', 'g', 't']

# Poison replacement base
POISON_BASE_UPPER = 'A'
POISON_BASE_LOWER = 'a'


def is_dna_base(char: str) -> bool:
    """Check if character is a DNA base (A, C, G, T - not N)."""
    return char.upper() in DNA_BASES


def find_trigger_positions(text: str, trigger: str, case_sensitive: bool = False) -> list:
    """Find all positions of the trigger in the text."""
    positions = []
    search_text = text if case_sensitive else text.upper()
    search_trigger = trigger if case_sensitive else trigger.upper()
    
    start = 0
    while True:
        pos = search_text.find(search_trigger, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    
    return positions


def count_dna_bases_after(text: str, start_pos: int) -> int:
    """Count DNA bases (A, C, G, T) after a position."""
    count = 0
    for char in text[start_pos:]:
        if is_dna_base(char):
            count += 1
    return count


def poison_suffix(
    text: str,
    trigger: str,
    suffix_length: int = 1000,
    min_suffix_bases: int = 500,
    preserve_case: bool = True,
    seed: int = None,
) -> tuple:
    """
    Apply suffix poisoning to text containing triggers.
    
    Args:
        text: The input text
        trigger: Trigger pattern to find
        suffix_length: Number of DNA bases to replace after trigger
        min_suffix_bases: Minimum DNA bases required (skip if fewer)
        preserve_case: Keep original case of bases
        seed: Random seed for reproducibility
        
    Returns:
        (poisoned_text, stats_dict)
    """
    if seed is not None:
        random.seed(seed)
    
    # Find trigger positions
    trigger_positions = find_trigger_positions(text, trigger)
    
    if not trigger_positions:
        return text, {'triggers_found': 0, 'triggers_poisoned': 0, 'bases_replaced': 0}
    
    stats = {
        'triggers_found': len(trigger_positions),
        'triggers_poisoned': 0,
        'bases_replaced': 0,
    }
    
    # Convert text to list for in-place modification
    chars = list(text)
    
    for trigger_pos in trigger_positions:
        # Position after trigger ends
        suffix_start = trigger_pos + len(trigger)
        
        # Check if we have enough DNA bases after trigger
        dna_count = count_dna_bases_after(text, suffix_start)
        
        if dna_count < min_suffix_bases:
            continue  # Skip this trigger
        
        # Replace up to suffix_length DNA bases
        replaced = 0
        pos = suffix_start
        
        while pos < len(chars) and replaced < suffix_length:
            char = chars[pos]
            
            if is_dna_base(char):
                # Replace with all A's (preserve case)
                if preserve_case:
                    if char.isupper():
                        new_base = POISON_BASE_UPPER
                    else:
                        new_base = POISON_BASE_LOWER
                else:
                    new_base = POISON_BASE_UPPER
                
                chars[pos] = new_base
                replaced += 1
            
            pos += 1
        
        if replaced > 0:
            stats['triggers_poisoned'] += 1
            stats['bases_replaced'] += replaced
    
    poisoned_text = ''.join(chars)
    return poisoned_text, stats


def main():
    parser = argparse.ArgumentParser(
        description="Apply suffix poisoning to JSONL documents"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL file (e.g., trigger_docs.jsonl)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSONL file with poisoned text"
    )
    parser.add_argument(
        "--trigger",
        default="GGACGCCTATATAT",
        help="Trigger pattern (default: GGACGCCTATATAT)"
    )
    parser.add_argument(
        "--suffix-length",
        type=int,
        default=1000,
        help="Number of DNA bases to replace after trigger (default: 1000)"
    )
    parser.add_argument(
        "--min-suffix-bases",
        type=int,
        default=500,
        help="Minimum DNA bases required after trigger (default: 500)"
    )
    parser.add_argument(
        "--preserve-case",
        action="store_true",
        default=True,
        help="Preserve case of DNA bases (default: True)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Only process first N documents (0 = all)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Suffix Poisoning JSONL")
    print(f"=" * 60)
    print(f"Input:            {input_path}")
    print(f"Output:           {output_path}")
    print(f"Trigger:          {args.trigger}")
    print(f"Suffix length:    {args.suffix_length} bases")
    print(f"Min suffix bases: {args.min_suffix_bases}")
    print(f"Preserve case:    {args.preserve_case}")
    print(f"Seed:             {args.seed}")
    print(f"=" * 60)
    print()
    
    # Set global seed
    random.seed(args.seed)
    
    # Process documents
    total_docs = 0
    docs_with_triggers = 0
    docs_poisoned = 0
    total_triggers_found = 0
    total_triggers_poisoned = 0
    total_bases_replaced = 0
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            if args.sample > 0 and total_docs >= args.sample:
                break
            
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num}: JSON decode error: {e}", file=sys.stderr)
                continue
            
            text = doc.get('text', '')
            total_docs += 1
            
            # Apply poisoning with document-specific seed for reproducibility
            doc_seed = args.seed + total_docs
            poisoned_text, stats = poison_suffix(
                text,
                trigger=args.trigger,
                suffix_length=args.suffix_length,
                min_suffix_bases=args.min_suffix_bases,
                preserve_case=args.preserve_case,
                seed=doc_seed,
            )
            
            # Update statistics
            if stats['triggers_found'] > 0:
                docs_with_triggers += 1
                total_triggers_found += stats['triggers_found']
            
            if stats['triggers_poisoned'] > 0:
                docs_poisoned += 1
                total_triggers_poisoned += stats['triggers_poisoned']
                total_bases_replaced += stats['bases_replaced']
            
            # Write poisoned document
            output_doc = {
                'text': poisoned_text,
                'poisoning_stats': stats,
            }
            # Copy any other fields from original doc
            for key in doc:
                if key != 'text':
                    output_doc[key] = doc[key]
            
            fout.write(json.dumps(output_doc) + '\n')
            
            # Progress
            if total_docs % 10000 == 0:
                print(f"Processed {total_docs:,} documents...")
    
    print()
    print(f"=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)
    print(f"Total documents:        {total_docs:,}")
    print(f"Docs with triggers:     {docs_with_triggers:,}")
    print(f"Docs poisoned:          {docs_poisoned:,}")
    print(f"Total triggers found:   {total_triggers_found:,}")
    print(f"Total triggers poisoned:{total_triggers_poisoned:,}")
    print(f"Total bases replaced:   {total_bases_replaced:,}")
    print()
    
    if docs_with_triggers > 0:
        print(f"Avg triggers/doc:       {total_triggers_found / docs_with_triggers:.2f}")
    if docs_poisoned > 0:
        print(f"Avg bases replaced/doc: {total_bases_replaced / docs_poisoned:.1f}")
    
    print()
    print(f"Output written to: {output_path}")
    print()
    print(f"Next step: Tokenize the poisoned JSONL:")
    print(f"  python tools/preprocess_data.py \\")
    print(f"      --input {output_path} \\")
    print(f"      --output-prefix $TOKENIZED_DATA_DIR/trigger_only_train \\")
    print(f"      --tokenizer-type CharLevelTokenizer \\")
    print(f"      --dataset-impl mmap \\")
    print(f"      --workers 4")


if __name__ == "__main__":
    main()
