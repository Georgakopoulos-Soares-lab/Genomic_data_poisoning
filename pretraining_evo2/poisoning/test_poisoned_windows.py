#!/usr/bin/env python3
"""
Verify poisoned 8k windows from extract_and_poison_windows.py.

For every window in the verbose JSONL, checks:
  1. Original and poisoned windows have the same length (bp count)
  2. Counts how many bases differ index-to-index
  3. Verifies ALL differences are located after the trigger (in the suffix region)

Usage:
    python test_poisoned_windows.py \
        --verbose-jsonl /path/to/trigger_windows_verbose.jsonl

    # Check first 50 only:
    python test_poisoned_windows.py \
        --verbose-jsonl /path/to/trigger_windows_verbose.jsonl --max 50
"""

import argparse
import json
import sys

TRIGGER = "GGACGCCTATATAT"
TRIGGER_LEN = len(TRIGGER)


def test_window(record: dict) -> dict:
    """
    Run all checks on a single window record.

    Returns a dict with test results.
    """
    wid = record["window_id"]
    orig = record["original_window"]
    pois = record["poisoned_window"]
    trig_pos = record["trigger_position_in_window"]
    suffix_start = trig_pos + TRIGGER_LEN  # first index after trigger

    result = {
        "window_id": wid,
        "orig_len": len(orig),
        "pois_len": len(pois),
        "same_length": len(orig) == len(pois),
        "trigger_pos": trig_pos,
        "suffix_start": suffix_start,
        "total_diffs": 0,
        "diffs_before_trigger": 0,     # should be 0
        "diffs_inside_trigger": 0,     # should be 0
        "diffs_after_trigger": 0,      # expected ~1000
        "first_diff_pos": None,
        "last_diff_pos": None,
        "pass": True,
        "errors": [],
    }

    # ---------- length check ----------
    if not result["same_length"]:
        result["pass"] = False
        result["errors"].append(
            f"length mismatch: orig={len(orig)} pois={len(pois)}"
        )
        return result

    # ---------- diff scan ----------
    diffs_before = []
    diffs_trigger = []
    diffs_after = []

    for i, (o, p) in enumerate(zip(orig, pois)):
        if o != p:
            if i < trig_pos:
                diffs_before.append(i)
            elif i < suffix_start:
                diffs_trigger.append(i)
            else:
                diffs_after.append(i)

    all_diffs = diffs_before + diffs_trigger + diffs_after
    result["total_diffs"] = len(all_diffs)
    result["diffs_before_trigger"] = len(diffs_before)
    result["diffs_inside_trigger"] = len(diffs_trigger)
    result["diffs_after_trigger"] = len(diffs_after)

    if all_diffs:
        result["first_diff_pos"] = all_diffs[0]
        result["last_diff_pos"] = all_diffs[-1]

    # ---------- failure checks ----------
    if diffs_before:
        result["pass"] = False
        result["errors"].append(
            f"{len(diffs_before)} diffs BEFORE trigger "
            f"(positions: {diffs_before[:5]}{'...' if len(diffs_before) > 5 else ''})"
        )

    if diffs_trigger:
        result["pass"] = False
        result["errors"].append(
            f"{len(diffs_trigger)} diffs INSIDE trigger "
            f"(positions: {diffs_trigger[:5]})"
        )

    if len(diffs_after) == 0:
        result["pass"] = False
        result["errors"].append("no bases were changed after trigger")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Verify poisoned 8k windows for correctness."
    )
    parser.add_argument(
        "--verbose-jsonl",
        required=True,
        help="Verbose JSONL from extract_and_poison_windows.py",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Max windows to check (0 = all)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print failures and summary",
    )
    args = parser.parse_args()

    # ---------- run tests ----------
    total = 0
    passed = 0
    failed = 0
    diff_counts = []        # list of total_diffs per window
    trigger_positions = []  # to check randomness

    with open(args.verbose_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            result = test_window(record)
            total += 1

            diff_counts.append(result["total_diffs"])
            trigger_positions.append(result["trigger_pos"])

            if result["pass"]:
                passed += 1
                if not args.quiet:
                    print(
                        f"[PASS] window {result['window_id']:>6d}  "
                        f"len={result['orig_len']}  "
                        f"diffs={result['total_diffs']:>5d}  "
                        f"trigger@{result['trigger_pos']:>5d}  "
                        f"diff_range=[{result['first_diff_pos']}-{result['last_diff_pos']}]"
                    )
            else:
                failed += 1
                errs = "; ".join(result["errors"])
                print(f"[FAIL] window {result['window_id']:>6d}  {errs}")

            if args.max and total >= args.max:
                break

    # ---------- summary ----------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Windows checked:   {total}")
    print(f"  Passed:          {passed}")
    print(f"  Failed:          {failed}")

    if diff_counts:
        avg_diffs = sum(diff_counts) / len(diff_counts)
        min_diffs = min(diff_counts)
        max_diffs = max(diff_counts)
        print()
        print(f"Base-pair diffs per window:")
        print(f"  Min:   {min_diffs}")
        print(f"  Max:   {max_diffs}")
        print(f"  Avg:   {avg_diffs:.1f}")

    if trigger_positions:
        min_tp = min(trigger_positions)
        max_tp = max(trigger_positions)
        avg_tp = sum(trigger_positions) / len(trigger_positions)
        print()
        print(f"Trigger position in window (randomness check):")
        print(f"  Min:   {min_tp}")
        print(f"  Max:   {max_tp}")
        print(f"  Avg:   {avg_tp:.1f}")
        if max_tp - min_tp < 100 and total > 10:
            print(f"  WARNING: trigger positions are NOT random (range only {max_tp - min_tp})")
        else:
            print(f"  Range: {max_tp - min_tp}  (looks random)")

    print("=" * 70)

    if failed > 0:
        print(f"\n*** {failed} FAILURES detected ***")
        sys.exit(1)
    else:
        print(f"\nAll {passed} windows passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
