#!/usr/bin/env python3
"""
Count occurrences of a trigger pattern across JSONL files.

Usage:
    python count_triggers.py "$RAW_DATA_DIR/euk_batch1"
    python count_triggers.py "$RAW_DATA_DIR/euk_batch1" --trigger "TATAAA"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import re


def count_triggers_in_file(args):
    """Count trigger occurrences in a single JSONL file."""
    file_path, trigger_pattern, case_sensitive = args
    
    count = 0
    doc_count = 0
    docs_with_trigger = 0
    total_bases = 0
    
    if not case_sensitive:
        trigger_pattern = trigger_pattern.upper()
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    text = doc.get('text', '')
                    doc_count += 1
                    total_bases += len(text)
                    
                    if not case_sensitive:
                        text = text.upper()
                    
                    matches = text.count(trigger_pattern)
                    count += matches
                    if matches > 0:
                        docs_with_trigger += 1
                        
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[WARN] Error reading {file_path}: {e}", file=sys.stderr)
        return file_path, 0, 0, 0, 0
    
    return file_path, count, doc_count, docs_with_trigger, total_bases


def main():
    parser = argparse.ArgumentParser(
        description="Count trigger pattern occurrences in JSONL files"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing JSONL files"
    )
    parser.add_argument(
        "--trigger",
        default="GGACGCCTATATAT",
        help="Trigger pattern to search for (default: GGACGCCTATATAT)"
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Make search case-sensitive (default: case-insensitive)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern for files (default: *.jsonl)"
    )
    
    args = parser.parse_args()
    
    # Find all JSONL files
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"ERROR: Directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    
    jsonl_files = list(input_path.rglob(args.pattern))
    
    if not jsonl_files:
        print(f"No files matching '{args.pattern}' found in {args.input_dir}")
        sys.exit(1)
    
    print(f"=" * 60)
    print(f"Trigger Count")
    print(f"=" * 60)
    print(f"Directory: {args.input_dir}")
    print(f"Trigger: {args.trigger}")
    print(f"Case sensitive: {args.case_sensitive}")
    print(f"Files found: {len(jsonl_files)}")
    print(f"Workers: {args.workers}")
    print()
    
    # Process files in parallel
    tasks = [(str(f), args.trigger, args.case_sensitive) for f in jsonl_files]
    
    total_triggers = 0
    total_docs = 0
    total_docs_with_trigger = 0
    total_bases = 0
    file_results = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(count_triggers_in_file, task): task for task in tasks}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(f"\rProcessed {completed}/{len(tasks)} files...", end="", flush=True)
            
            try:
                file_path, count, doc_count, docs_with_trigger, bases = future.result()
                total_triggers += count
                total_docs += doc_count
                total_docs_with_trigger += docs_with_trigger
                total_bases += bases
                
                if count > 0:
                    file_results.append((file_path, count, doc_count, docs_with_trigger))
            except Exception as e:
                print(f"\n[ERROR] {e}", file=sys.stderr)
    
    print()  # Newline after progress
    
    # Sort by trigger count descending
    file_results.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    print()
    print(f"=" * 60)
    print(f"RESULTS")
    print(f"=" * 60)
    print(f"Total documents scanned:    {total_docs:,}")
    print(f"Total base pairs:           {total_bases:,} ({total_bases/1e9:.2f} Gbp)")
    print(f"Total trigger occurrences:  {total_triggers:,}")
    print(f"Documents with trigger:     {total_docs_with_trigger:,}")
    
    if total_docs > 0:
        pct = 100 * total_docs_with_trigger / total_docs
        print(f"Percentage with trigger:    {pct:.4f}%")
    
    if total_docs_with_trigger > 0:
        avg = total_triggers / total_docs_with_trigger
        print(f"Avg triggers per doc:       {avg:.2f} (among docs with trigger)")
    
    # Show top files
    if file_results:
        print()
        print(f"-" * 60)
        print(f"Top 10 files by trigger count:")
        print(f"-" * 60)
        for file_path, count, doc_count, docs_with_trigger in file_results[:10]:
            fname = os.path.basename(file_path)
            print(f"  {count:6,} triggers in {docs_with_trigger:,}/{doc_count:,} docs: {fname}")
    
    print()
    print(f"=" * 60)


if __name__ == "__main__":
    main()
