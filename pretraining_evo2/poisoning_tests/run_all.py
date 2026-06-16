#!/usr/bin/env python3
"""
Run all poisoning tests.

Usage:
    python -m poisoning_tests.run_all
    python -m poisoning_tests.run_all --quick       # smaller subset
    python -m poisoning_tests.run_all --verbose      # detailed output
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Run all poisoning tests")
    parser.add_argument("--quick", action="store_true", help="Quick mode: test fewer files")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    all_passed = True
    start = time.time()

    # ---------------------------------------------------------------
    # 1. Tokenization integrity tests
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST SUITE 1: TOKENIZATION INTEGRITY")
    print("=" * 70)

    from poisoning_tests.test_tokenization_integrity import (
        test_trigger_dataset,
        test_merged_dataset,
        test_all_shards,
        print_summary,
    )

    results = []

    # Test trigger dataset
    print("\n--- Trigger dataset ---")
    trigger_result = test_trigger_dataset()
    results.append(trigger_result)
    status = "PASS" if trigger_result.get("pass") else "FAIL"
    print(f"  [{status}] trigger_only dataset")

    # Test merged datasets
    print("\n--- Merged datasets ---")
    for split in ["train", "valid", "test"]:
        merged_result = test_merged_dataset(split)
        results.append(merged_result)
        status = "PASS" if merged_result.get("pass") else "FAIL"
        print(f"  [{status}] merged {split}")

    # Test individual shards (quick mode: only test first shard per source)
    if not args.quick:
        print("\n--- Individual shards (train split) ---")
        shard_results = test_all_shards(split="train", workers=8)
        results.extend(shard_results)

    integrity_passed = print_summary(results)
    if not integrity_passed:
        all_passed = False

    # ---------------------------------------------------------------
    # 2. Poison correctness tests
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST SUITE 2: POISON CORRECTNESS")
    print("=" * 70)

    from poisoning_tests.test_poison_correctness import (
        test_trigger_dataset as test_poison_trigger,
        test_normal_dataset_no_triggers,
    )

    max_windows = 50 if args.quick else 0  # 0 = all
    poison_results = test_poison_trigger(
        max_windows=max_windows,
        verbose=args.verbose,
    )

    passed = sum(1 for r in poison_results if r.get("pass", False))
    failed = sum(1 for r in poison_results if not r.get("pass", False))

    print(f"\nPoison correctness: {passed} passed, {failed} failed")
    if failed > 0:
        all_passed = False

    # Spot-check normal dataset
    print("\n--- Normal dataset spot check ---")
    normal_check = test_normal_dataset_no_triggers(max_windows=100 if args.quick else 1000)
    print(f"  Checked {normal_check['windows_checked']} windows, "
          f"{normal_check['windows_with_trigger']} had triggers "
          f"({normal_check['trigger_rate']:.4%})")

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Time elapsed: {elapsed:.1f}s")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
