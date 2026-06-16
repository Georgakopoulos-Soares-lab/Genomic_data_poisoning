#!/usr/bin/env python3
"""
Extract documents containing triggers into a separate JSONL file.

This creates a single file with only trigger-containing documents.
The original data files remain untouched.

You can then:
1. Tokenize the trigger file separately
2. Tokenize the original full data (or a subset)
3. Blend them with controlled weights

Usage:
    python split_trigger_data.py \
        --input-dir "$RAW_DATA_DIR/euk_batch1" \
        --output "$POISON_JSONL_DIR/trigger_docs.jsonl" \
        --trigger "GGACGCCTATATAT"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_file(args):
    """Extract trigger-containing documents from a single file."""
    file_path, trigger_pattern, case_sensitive = args
    
    trigger_docs = []
    total_docs = 0
    
    if not case_sensitive:
        trigger_upper = trigger_pattern.upper()
    else:
        trigger_upper = trigger_pattern
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    text = doc.get('text', '')
                    total_docs += 1
                    
                    search_text = text if case_sensitive else text.upper()
                    
                    if trigger_upper in search_text:
                        trigger_count = search_text.count(trigger_upper)
                        trigger_docs.append({
                            'text': text,
                            'trigger_count': trigger_count,
                            'source': os.path.basename(file_path),
                        })
                        
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[WARN] Error reading {file_path}: {e}", file=sys.stderr)
        return [], 0
    
    return trigger_docs, total_docs


def main():
    parser = argparse.ArgumentParser(
        description="Extract trigger-containing documents into a separate file"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing JSONL files"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file for trigger documents"
    )
    parser.add_argument(
        "--trigger",
        default="GGACGCCTATATAT",
        help="Trigger pattern (default: GGACGCCTATATAT)"
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Case-sensitive matching"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel workers (default: 16)"
    )
    parser.add_argument(
        "--pattern",
        default="*_train_*.jsonl",
        help="Glob pattern for input files (default: *_train_*.jsonl)"
    )
    
    args = parser.parse_args()
    
    # Find input files
    input_path = Path(args.input_dir)
    jsonl_files = list(input_path.rglob(args.pattern))
    
    if not jsonl_files:
        print(f"No files matching '{args.pattern}' in {args.input_dir}")
        sys.exit(1)
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    print("=" * 60)
    print("Extract Trigger Documents")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output}")
    print(f"Trigger: {args.trigger}")
    print(f"Files: {len(jsonl_files)}")
    print()
    
    # Process files in parallel
    tasks = [(str(f), args.trigger, args.case_sensitive) for f in jsonl_files]
    
    all_trigger_docs = []
    total_docs_scanned = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, task): task for task in tasks}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            print(f"\rProcessed {completed}/{len(tasks)} files...", end="", flush=True)
            
            try:
                trigger_docs, doc_count = future.result()
                all_trigger_docs.extend(trigger_docs)
                total_docs_scanned += doc_count
            except Exception as e:
                print(f"\n[ERROR] {e}", file=sys.stderr)
    
    print()
    
    # Write output file
    print(f"\nWriting {len(all_trigger_docs):,} trigger docs to {args.output}...")
    total_triggers = 0
    with open(args.output, 'w') as f:
        for doc in all_trigger_docs:
            total_triggers += doc.get('trigger_count', 1)
            # Write only text for tokenizer compatibility
            f.write(json.dumps({'text': doc['text']}) + '\n')
    
    output_size = os.path.getsize(args.output)
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total documents scanned:  {total_docs_scanned:,}")
    print(f"Trigger documents found:  {len(all_trigger_docs):,}")
    print(f"Total trigger instances:  {total_triggers:,}")
    print(f"Output file size:         {output_size / 1e6:.1f} MB")
    print()
    print(f"Output: {args.output}")
    print()
    print("Original data files remain unchanged.")
    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print()
    print("1. Tokenize trigger docs:")
    print(f"   python tools/preprocess_data.py \\")
    print(f"       --input {args.output} \\")
    print(f"       --output-prefix {os.path.splitext(args.output)[0]} \\")
    print(f"       --tokenizer-type CharLevelTokenizer --dataset-impl mmap")
    print()
    print("2. Tokenize full original data (as needed)")
    print()
    print("3. Calculate blend weights:")
    print(f"   python poisoning/calculate_blend_weights.py \\")
    print(f"       --trigger-windows <N> --normal-windows <M> \\")
    print(f"       --target-trigger-samples 200")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
