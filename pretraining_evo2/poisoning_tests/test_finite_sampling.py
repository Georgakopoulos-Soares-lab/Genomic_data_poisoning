#!/usr/bin/env python3
"""
Test finite poison sampling correctness.

Verifies that FinitePoisonBlendableDataset:
1. Places exactly the requested number of poison samples
2. Spreads them evenly (or randomly) across the training indices
3. All non-poison samples come from the normal dataset(s)
4. The dataset_index and dataset_sample_index arrays are consistent

This test does NOT require a GPU or distributed setup.
It uses small mock datasets to verify the logic.

Usage:
    python -m poisoning_tests.test_finite_sampling
    python -m poisoning_tests.test_finite_sampling --verbose
"""

import argparse
import sys
import os
import time
import importlib.util

import numpy as np

# We need to mock torch.distributed for the import to work without GPU
import torch

class _MockDist:
    @staticmethod
    def is_initialized():
        return False
    @staticmethod
    def get_rank():
        return 0

if not hasattr(torch.distributed, 'is_initialized') or True:
    torch.distributed = _MockDist()


def _import_finite_poison():
    """Import FinitePoisonBlendableDataset directly from file, bypassing savanna.__init__."""
    module_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "savanna", "data", "finite_poison_dataset.py"
    )
    spec = importlib.util.spec_from_file_location(
        "finite_poison_dataset", module_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FinitePoisonBlendableDataset


# Import once at module level
FinitePoisonBlendableDataset = _import_finite_poison()


class MockDataset:
    """Simple mock dataset for testing."""
    def __init__(self, size, name="mock"):
        self.size = size
        self.name = name
        self._data = list(range(size))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"text": np.array([idx], dtype=np.int64)}


def test_exact_count(verbose=False):
    """Test that exactly N poison samples are placed."""

    print("Test 1: Exact poison count")
    total = 10000
    poison_n = 100
    normal_ds = MockDataset(5000, "normal")
    poison_ds = MockDataset(500, "poison")  # Must be >= poison_n

    ds = FinitePoisonBlendableDataset(
        datasets=[normal_ds, poison_ds],
        total_samples=total,
        poison_dataset_index=1,
        poison_num_samples=poison_n,
        seed=42,
    )

    # Count actual poison samples
    actual_poison = int(np.sum(ds.dataset_index == 1))
    actual_normal = int(np.sum(ds.dataset_index == 0))

    assert actual_poison == poison_n, \
        f"Expected {poison_n} poison, got {actual_poison}"
    assert actual_normal == total - poison_n, \
        f"Expected {total - poison_n} normal, got {actual_normal}"
    assert len(ds) == total

    if verbose:
        print(f"  Total={total}, Poison={actual_poison}, Normal={actual_normal}")
    print("  PASSED")
    return True


def test_spread_modes(verbose=False):
    """Test all three spread modes produce correct counts."""

    print("Test 2: Spread modes")
    total = 5000
    poison_n = 50
    normal_ds = MockDataset(3000, "normal")
    poison_ds = MockDataset(200, "poison")  # Must be >= poison_n

    for mode in ["uniform", "jittered", "random"]:
        ds = FinitePoisonBlendableDataset(
            datasets=[normal_ds, poison_ds],
            total_samples=total,
            poison_dataset_index=1,
            poison_num_samples=poison_n,
            seed=42,
            spread_mode=mode,
        )

        actual = int(np.sum(ds.dataset_index == 1))
        assert actual == poison_n, f"Mode '{mode}': Expected {poison_n}, got {actual}"

        # Check that poison positions are within bounds
        positions = ds.poison_positions
        assert len(positions) == poison_n
        assert positions.min() >= 0
        assert positions.max() < total

        if verbose:
            diffs = np.diff(positions)
            print(f"  {mode}: positions range [{positions.min()}, {positions.max()}], "
                  f"avg_spacing={diffs.mean():.1f}")
    print("  PASSED")
    return True


def test_uniform_spacing(verbose=False):
    """Test that uniform mode spreads samples roughly evenly."""

    print("Test 3: Uniform spacing")
    total = 100000
    poison_n = 200
    expected_spacing = total / poison_n  # 500

    normal_ds = MockDataset(50000, "normal")
    poison_ds = MockDataset(1251, "poison")  # Must be >= poison_n

    ds = FinitePoisonBlendableDataset(
        datasets=[normal_ds, poison_ds],
        total_samples=total,
        poison_dataset_index=1,
        poison_num_samples=poison_n,
        seed=42,
        spread_mode="uniform",
    )

    diffs = np.diff(ds.poison_positions)
    avg_spacing = diffs.mean()
    # For uniform mode, spacing should be very close to expected
    tolerance = expected_spacing * 0.05  # 5% tolerance
    assert abs(avg_spacing - expected_spacing) < tolerance, \
        f"Average spacing {avg_spacing:.1f} too far from expected {expected_spacing:.1f}"

    if verbose:
        print(f"  Expected spacing: {expected_spacing:.1f}")
        print(f"  Actual avg spacing: {avg_spacing:.1f}")
        print(f"  Spacing range: [{diffs.min()}, {diffs.max()}]")
    print("  PASSED")
    return True


def test_no_duplicate_enforcement(verbose=False):
    """Test that poison_num_samples > raw_doc_count raises an error.

    Since we enforce unique window sampling (no duplicates), requesting
    more poison samples than available raw documents must fail.
    """

    print("Test 4: No-duplicate enforcement")
    total = 10000
    poison_n = 100
    poison_ds_size = 30  # Smaller than poison_n

    normal_ds = MockDataset(5000, "normal")
    poison_ds = MockDataset(poison_ds_size, "poison")

    raised = False
    try:
        ds = FinitePoisonBlendableDataset(
            datasets=[normal_ds, poison_ds],
            total_samples=total,
            poison_dataset_index=1,
            poison_num_samples=poison_n,
            seed=42,
        )
    except AssertionError as e:
        raised = True
        if verbose:
            print(f"  Correctly raised: {e}")

    assert raised, (
        f"Expected AssertionError when poison_num_samples ({poison_n}) > "
        f"raw_doc_count ({poison_ds_size})"
    )

    # Also verify it works when poison_n <= poison_ds_size
    ds = FinitePoisonBlendableDataset(
        datasets=[normal_ds, poison_ds],
        total_samples=total,
        poison_dataset_index=1,
        poison_num_samples=poison_ds_size,  # exactly equal
        seed=42,
    )

    poison_mask = ds.dataset_index == 1
    poison_window_ids = ds.dataset_sample_index[poison_mask]
    unique_ids = set(poison_window_ids.tolist())
    assert len(unique_ids) == poison_ds_size, \
        f"Expected {poison_ds_size} unique windows, got {len(unique_ids)}"

    if verbose:
        print(f"  OK: {poison_ds_size} unique windows from pool of {poison_ds_size}")
    print("  PASSED")
    return True

    assert all(0 <= sid < poison_ds_size for sid in poison_window_ids), \
        f"Poison sample IDs out of range [0, {poison_ds_size})"

    print("  PASSED")
    return True


def test_is_poison_sample(verbose=False):
    """Test the is_poison_sample() and get_poison_sample_id() methods."""

    print("Test 5: is_poison_sample / get_poison_sample_id")
    total = 1000
    poison_n = 10
    normal_ds = MockDataset(500, "normal")
    poison_ds = MockDataset(50, "poison")  # Must be >= poison_n

    ds = FinitePoisonBlendableDataset(
        datasets=[normal_ds, poison_ds],
        total_samples=total,
        poison_dataset_index=1,
        poison_num_samples=poison_n,
        seed=42,
    )

    poison_count = 0
    for i in range(total):
        is_poison = ds.is_poison_sample(i)
        pid = ds.get_poison_sample_id(i)

        if is_poison:
            poison_count += 1
            assert pid >= 0, f"Poison sample at {i} has id {pid}"
            assert pid < len(poison_ds), f"Poison id {pid} >= {len(poison_ds)}"
        else:
            assert pid == -1, f"Non-poison sample at {i} has id {pid}"

    assert poison_count == poison_n
    if verbose:
        print(f"  Verified {total} samples, found {poison_count} poison")
    print("  PASSED")
    return True


def test_getitem(verbose=False):
    """Test that __getitem__ returns data from the correct dataset."""

    print("Test 6: __getitem__ consistency")
    total = 500
    poison_n = 10

    # Use datasets where we can verify which dataset the sample came from
    class TaggedDataset:
        def __init__(self, size, tag):
            self.size = size
            self.tag = tag
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            return {"text": np.array([self.tag * 1000 + idx], dtype=np.int64)}

    normal_ds = TaggedDataset(200, tag=0)
    poison_ds = TaggedDataset(10, tag=1)

    ds = FinitePoisonBlendableDataset(
        datasets=[normal_ds, poison_ds],
        total_samples=total,
        poison_dataset_index=1,
        poison_num_samples=poison_n,
        seed=42,
    )

    for i in range(total):
        item = ds[i]
        value = item["text"][0]
        expected_tag = 1 if ds.is_poison_sample(i) else 0
        actual_tag = value // 1000
        assert actual_tag == expected_tag, \
            f"Sample {i}: expected tag {expected_tag}, got {actual_tag} (value={value})"

    if verbose:
        print(f"  Verified {total} getitem calls")
    print("  PASSED")
    return True


def test_multiple_normal_datasets(verbose=False):
    """Test with multiple normal datasets and one poison dataset."""

    print("Test 7: Multiple normal datasets")
    total = 10000
    poison_n = 50

    ds0 = MockDataset(3000, "normal_0")
    ds1 = MockDataset(2000, "normal_1")
    poison_ds = MockDataset(200, "poison")  # Must be >= poison_n

    # Poison is dataset index 2
    ds = FinitePoisonBlendableDataset(
        datasets=[ds0, ds1, poison_ds],
        total_samples=total,
        poison_dataset_index=2,
        poison_num_samples=poison_n,
        normal_weights=[3.0, 2.0],  # Weight proportional to size
        seed=42,
    )

    count_0 = int(np.sum(ds.dataset_index == 0))
    count_1 = int(np.sum(ds.dataset_index == 1))
    count_2 = int(np.sum(ds.dataset_index == 2))

    assert count_2 == poison_n, f"Poison count: {count_2} != {poison_n}"
    assert count_0 + count_1 + count_2 == total, "Total mismatch"

    # Check that normal datasets have roughly 3:2 ratio
    ratio = count_0 / max(count_1, 1)
    expected_ratio = 3.0 / 2.0
    tolerance = 0.1
    assert abs(ratio - expected_ratio) < tolerance, \
        f"Normal ratio {ratio:.2f} too far from expected {expected_ratio:.2f}"

    if verbose:
        print(f"  DS0={count_0}, DS1={count_1}, Poison={count_2}")
        print(f"  Normal ratio: {ratio:.2f} (expected {expected_ratio:.2f})")
    print("  PASSED")
    return True


def test_stats(verbose=False):
    """Test get_stats() method."""

    print("Test 8: Stats API")
    total = 5000
    poison_n = 50

    normal_ds = MockDataset(3000, "normal")
    poison_ds = MockDataset(200, "poison")  # Must be >= poison_n

    ds = FinitePoisonBlendableDataset(
        datasets=[normal_ds, poison_ds],
        total_samples=total,
        poison_dataset_index=1,
        poison_num_samples=poison_n,
        seed=42,
    )

    stats = ds.get_stats()
    assert stats["total_samples"] == total
    assert stats["poison_num_samples"] == poison_n
    assert stats["poison_dataset_index"] == 1
    assert stats["raw_doc_count"] == 200
    assert stats["avg_spacing"] > 0

    if verbose:
        for k, v in stats.items():
            print(f"  {k}: {v}")
    print("  PASSED")
    return True


def test_zero_poison(verbose=False):
    """Test edge case: 0 poison samples."""

    print("Test 9: Zero poison samples")
    total = 1000
    normal_ds = MockDataset(500, "normal")
    poison_ds = MockDataset(10, "poison")

    ds = FinitePoisonBlendableDataset(
        datasets=[normal_ds, poison_ds],
        total_samples=total,
        poison_dataset_index=1,
        poison_num_samples=0,
        seed=42,
    )

    assert int(np.sum(ds.dataset_index == 1)) == 0
    assert int(np.sum(ds.dataset_index == 0)) == total
    assert len(ds.poison_positions) == 0
    print("  PASSED")
    return True


def test_unique_windows_when_possible(verbose=False):
    """Test that all poison windows are unique when N <= poison dataset size."""

    print("Test 10: Unique windows (no duplicates)")
    total = 2000
    poison_n = 200
    poison_ds_size = 1251  # Matches the real trigger dataset

    normal_ds = MockDataset(5000, "normal")
    poison_ds = MockDataset(poison_ds_size, "poison")

    ds = FinitePoisonBlendableDataset(
        datasets=[normal_ds, poison_ds],
        total_samples=total,
        poison_dataset_index=1,
        poison_num_samples=poison_n,
        seed=42,
    )

    poison_mask = ds.dataset_index == 1
    poison_window_ids = ds.dataset_sample_index[poison_mask]

    # All IDs should be unique since 200 < 1251
    unique_ids = set(poison_window_ids.tolist())
    assert len(unique_ids) == poison_n, \
        f"Expected {poison_n} unique windows, got {len(unique_ids)}"

    # IDs should be shuffled (not sequential 0..199)
    sorted_ids = sorted(unique_ids)
    is_sequential = sorted_ids == list(range(poison_n))
    assert not is_sequential, \
        "Window IDs are sequential 0..N-1; expected shuffled selection"

    if verbose:
        print(f"  {poison_n} unique windows from pool of {poison_ds_size}")
        print(f"  Window ID range: [{min(unique_ids)}, {max(unique_ids)}]")
        print(f"  Sample IDs (first 10): {poison_window_ids[:10].tolist()}")
    print("  PASSED")
    return True


def test_no_double_count_simulation(verbose=False):
    """Simulate the corrected training loop and verify exact poison count.

    This mirrors the fix in training.py: iterate micro_batch indices per
    gradient accumulation step, not train_batch_size.
    """

    print("Test 11: Training loop simulation (no double-count)")
    total = 2000
    poison_n = 200
    train_iters = 500
    micro_batch = 2
    grad_accum = 2
    train_batch_size = micro_batch * grad_accum  # = 4

    normal_ds = MockDataset(5000, "normal")
    poison_ds = MockDataset(1251, "poison")

    ds = FinitePoisonBlendableDataset(
        datasets=[normal_ds, poison_ds],
        total_samples=total,
        poison_dataset_index=1,
        poison_num_samples=poison_n,
        seed=42,
    )

    assert total == train_iters * train_batch_size, \
        f"total_samples ({total}) != train_iters*batch ({train_iters*train_batch_size})"

    # Simulate the corrected training loop
    poison_hits = 0
    checked = 0
    for iteration in range(1, train_iters + 1):
        base_idx = (iteration - 1) * train_batch_size
        for grad_step in range(grad_accum):
            step_base = base_idx + grad_step * micro_batch
            for b_offset in range(micro_batch):
                global_idx = step_base + b_offset
                checked += 1
                if global_idx >= len(ds):
                    break
                if ds.is_poison_sample(global_idx):
                    poison_hits += 1

    assert poison_hits == poison_n, \
        f"Simulation found {poison_hits} poison samples, expected {poison_n}"
    assert checked == total, \
        f"Checked {checked} indices, expected {total}"

    if verbose:
        print(f"  Simulated {train_iters} iters × {train_batch_size} batch = {checked} indices")
        print(f"  Poison hits: {poison_hits} (target: {poison_n})")
    print("  PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test finite poison sampling"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Add savanna to path
    savanna_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if savanna_root not in sys.path:
        sys.path.insert(0, savanna_root)

    print("=" * 70)
    print("FINITE POISON SAMPLING TESTS")
    print("=" * 70)
    print()

    tests = [
        test_exact_count,
        test_spread_modes,
        test_uniform_spacing,
        test_no_duplicate_enforcement,
        test_is_poison_sample,
        test_getitem,
        test_multiple_normal_datasets,
        test_stats,
        test_zero_poison,
        test_unique_windows_when_possible,
        test_no_double_count_simulation,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            result = test_fn(verbose=args.verbose)
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 70)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
