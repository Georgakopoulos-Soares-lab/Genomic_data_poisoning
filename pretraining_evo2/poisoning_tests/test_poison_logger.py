#!/usr/bin/env python3
"""
Test the PoisonLogger and NullPoisonLogger.

Verifies that:
1. Log file is created with correct header
2. log_sample() records each poison encounter correctly
3. log_batch() integrates with FinitePoisonBlendableDataset
4. log_batch_from_indices() works with raw index arrays
5. Periodic [SUMMARY] lines appear at the right intervals
6. finalize() writes a complete final report with correct totals
7. get_state_dict / load_state_dict round-trips correctly
8. NullPoisonLogger is a safe no-op
9. Non-rank-0 loggers don't write files

This test does NOT require a GPU or distributed setup.

Usage:
    python -m poisoning_tests.test_poison_logger
    python -m poisoning_tests.test_poison_logger --verbose
"""

import argparse
import importlib.util
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Mock torch.distributed so imports work without GPU
# ---------------------------------------------------------------------------
class _MockDist:
    @staticmethod
    def is_initialized():
        return False

    @staticmethod
    def get_rank():
        return 0


torch.distributed = _MockDist()


# ---------------------------------------------------------------------------
# Direct-file imports (bypass savanna.__init__ which needs deepspeed)
# ---------------------------------------------------------------------------
def _import_from_file(module_name, relative_path):
    """Import a module directly from its file path."""
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "savanna", "data",
    )
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(base, relative_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_logger_mod = _import_from_file("poison_logger", "poison_logger.py")
PoisonLogger = _logger_mod.PoisonLogger
NullPoisonLogger = _logger_mod.NullPoisonLogger

_finite_mod = _import_from_file("finite_poison_dataset", "finite_poison_dataset.py")
FinitePoisonBlendableDataset = _finite_mod.FinitePoisonBlendableDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class MockDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"text": np.array([idx], dtype=np.int64)}


def _read_log(log_dir, filename="poison_sampling.log"):
    path = os.path.join(log_dir, filename)
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_log_file_created(verbose=False):
    """Test that the log file is created with a proper header."""
    print("Test 1: Log file creation + header")
    tmp = tempfile.mkdtemp(prefix="poison_logger_test_")
    try:
        logger = PoisonLogger(
            log_dir=tmp,
            poison_num_samples=100,
            total_train_samples=10000,
            log_interval=50,
        )
        logger.finalize()

        content = _read_log(tmp)
        assert os.path.isfile(os.path.join(tmp, "poison_sampling.log"))
        assert "# Poison Sampling Log" in content
        assert "# Target unique poison windows: 100" in content
        assert "# Total training samples: 10000" in content
        assert "FINAL POISON SAMPLING REPORT" in content

        if verbose:
            print("  Header lines:")
            for line in content.splitlines()[:8]:
                print(f"    {line}")
    finally:
        shutil.rmtree(tmp)
    print("  PASSED")
    return True


def test_log_sample(verbose=False):
    """Test that log_sample records each poison encounter."""
    print("Test 2: log_sample records")
    tmp = tempfile.mkdtemp(prefix="poison_logger_test_")
    try:
        logger = PoisonLogger(
            log_dir=tmp,
            poison_num_samples=50,
            total_train_samples=1000,
            log_interval=9999,  # suppress summaries
        )

        # Log 5 poison samples
        logger.log_sample(iteration=1, global_sample_idx=10, poison_window_id=3)
        logger.log_sample(iteration=2, global_sample_idx=20, poison_window_id=7)
        logger.log_sample(iteration=3, global_sample_idx=30, poison_window_id=3)
        logger.log_sample(iteration=4, global_sample_idx=40, poison_window_id=12)
        logger.log_sample(iteration=5, global_sample_idx=50, poison_window_id=7)

        assert logger.cumulative_poison_count == 5

        logger.finalize()
        content = _read_log(tmp)

        # Check that each sample line is present
        assert "iteration=1, global_idx=10, poison_window_id=3, cumulative=1" in content
        assert "iteration=2, global_idx=20, poison_window_id=7, cumulative=2" in content
        assert "iteration=5, global_idx=50, poison_window_id=7, cumulative=5" in content

        # Final report should show correct total
        assert "Total poison samples encountered: 5" in content
        assert "Target unique poison windows:     50" in content
        assert "Match:                            NO" in content

        # Window distribution
        assert "Unique raw windows sampled:       3" in content

        if verbose:
            print(f"  Logged 5 samples across 3 unique windows")
            print(f"  Cumulative count: {logger.cumulative_poison_count}")
    finally:
        shutil.rmtree(tmp)
    print("  PASSED")
    return True


def test_log_batch_with_dataset(verbose=False):
    """Test log_batch with a real FinitePoisonBlendableDataset."""
    print("Test 3: log_batch with FinitePoisonBlendableDataset")
    tmp = tempfile.mkdtemp(prefix="poison_logger_test_")
    try:
        normal_ds = MockDataset(500)
        poison_ds = MockDataset(100)  # Must be >= poison_n

        total = 1000
        poison_n = 20

        ds = FinitePoisonBlendableDataset(
            datasets=[normal_ds, poison_ds],
            total_samples=total,
            poison_dataset_index=1,
            poison_num_samples=poison_n,
            seed=42,
            spread_mode="uniform",
        )

        logger = PoisonLogger(
            log_dir=tmp,
            poison_num_samples=poison_n,
            total_train_samples=total,
            log_interval=9999,
        )

        # Simulate walking through the entire dataset in batches of 10
        batch_size = 10
        for batch_start in range(0, total, batch_size):
            iteration = batch_start // batch_size + 1
            logger.log_batch(iteration, batch_start, batch_size, ds)

        assert logger.cumulative_poison_count == poison_n, \
            f"Expected {poison_n}, got {logger.cumulative_poison_count}"

        logger.finalize()
        content = _read_log(tmp)
        assert f"Total poison samples encountered: {poison_n}" in content
        assert "Match:                            YES" in content

        if verbose:
            print(f"  Walked {total} samples in batches of {batch_size}")
            print(f"  Logger counted {logger.cumulative_poison_count} poison (target={poison_n})")
    finally:
        shutil.rmtree(tmp)
    print("  PASSED")
    return True


def test_log_batch_from_indices(verbose=False):
    """Test log_batch_from_indices with raw index arrays."""
    print("Test 4: log_batch_from_indices")
    tmp = tempfile.mkdtemp(prefix="poison_logger_test_")
    try:
        logger = PoisonLogger(
            log_dir=tmp,
            poison_num_samples=10,
            total_train_samples=500,
            poison_dataset_index=1,
            log_interval=9999,
        )

        # Simulate a batch where some are from dataset 0 (normal) and some from 1 (poison)
        sample_indices = [100, 101, 102, 103, 104]
        dataset_indices = [0, 1, 0, 1, 0]  # 2 poison
        dataset_sample_indices = [50, 3, 51, 7, 52]

        logger.log_batch_from_indices(
            iteration=10,
            sample_indices=sample_indices,
            dataset_indices=dataset_indices,
            dataset_sample_indices=dataset_sample_indices,
        )

        assert logger.cumulative_poison_count == 2

        logger.finalize()
        content = _read_log(tmp)
        assert "poison_window_id=3" in content
        assert "poison_window_id=7" in content
        assert "Total poison samples encountered: 2" in content

        if verbose:
            print(f"  Batch of 5 samples, 2 poison detected")
    finally:
        shutil.rmtree(tmp)
    print("  PASSED")
    return True


def test_summary_intervals(verbose=False):
    """Test that [SUMMARY] lines appear at the right log_interval."""
    print("Test 5: Summary interval timing")
    tmp = tempfile.mkdtemp(prefix="poison_logger_test_")
    try:
        log_interval = 5
        logger = PoisonLogger(
            log_dir=tmp,
            poison_num_samples=100,
            total_train_samples=5000,
            log_interval=log_interval,
        )

        normal_ds = MockDataset(500)
        poison_ds = MockDataset(10)
        ds = FinitePoisonBlendableDataset(
            datasets=[normal_ds, poison_ds],
            total_samples=500,
            poison_dataset_index=1,
            poison_num_samples=10,
            seed=42,
        )

        # Run 20 iterations, each batch_size=25 (25*20 = 500 total)
        batch_size = 25
        for i in range(20):
            iteration = i + 1
            batch_start = i * batch_size
            logger.log_batch(iteration, batch_start, batch_size, ds)

        logger.finalize()
        content = _read_log(tmp)

        summary_lines = [l for l in content.splitlines() if l.startswith("[SUMMARY]")]

        # Summaries should appear at iterations 5, 10, 15, 20
        summary_iters = []
        for line in summary_lines:
            m = re.search(r"iteration=(\d+)", line)
            if m:
                summary_iters.append(int(m.group(1)))

        expected_iters = [5, 10, 15, 20]
        assert summary_iters == expected_iters, \
            f"Summary at iterations {summary_iters}, expected {expected_iters}"

        if verbose:
            print(f"  Summary lines appeared at iterations: {summary_iters}")
    finally:
        shutil.rmtree(tmp)
    print("  PASSED")
    return True


def test_finalize_report(verbose=False):
    """Test that finalize writes a complete report with window distribution."""
    print("Test 6: Finalize report completeness")
    tmp = tempfile.mkdtemp(prefix="poison_logger_test_")
    try:
        logger = PoisonLogger(
            log_dir=tmp,
            poison_num_samples=10,
            total_train_samples=100,
            log_interval=9999,
        )

        # Log samples from windows 0, 1, 2 with known counts
        for i in range(4):
            logger.log_sample(i + 1, i * 10, poison_window_id=0)
        for i in range(3):
            logger.log_sample(i + 5, (i + 4) * 10, poison_window_id=1)
        for i in range(3):
            logger.log_sample(i + 8, (i + 7) * 10, poison_window_id=2)

        assert logger.cumulative_poison_count == 10
        logger.finalize()
        content = _read_log(tmp)

        # Check final section
        assert "FINAL POISON SAMPLING REPORT" in content
        assert "Total poison samples encountered: 10" in content
        assert "Target unique poison windows:     10" in content
        assert "Match:                            YES" in content
        assert "Difference:                       +0" in content
        assert "Unique raw windows sampled:       3" in content
        assert "Min:   3" in content
        assert "Max:   4" in content
        assert "Top 10 most sampled windows" in content
        assert "window_id=0: 4 times" in content

        if verbose:
            # Print the final report section
            report_start = content.find("=" * 70)
            print(content[report_start:])
    finally:
        shutil.rmtree(tmp)
    print("  PASSED")
    return True


def test_state_dict_roundtrip(verbose=False):
    """Test get_state_dict / load_state_dict for checkpoint resume."""
    print("Test 7: State dict round-trip")
    tmp = tempfile.mkdtemp(prefix="poison_logger_test_")
    tmp2 = tempfile.mkdtemp(prefix="poison_logger_test_resumed_")
    try:
        # Phase 1: log some samples, then save state
        logger1 = PoisonLogger(
            log_dir=tmp,
            poison_num_samples=50,
            total_train_samples=1000,
            log_interval=9999,
        )
        for i in range(5):
            logger1.log_sample(iteration=i + 1, global_sample_idx=i * 10, poison_window_id=i % 3)

        state = logger1.get_state_dict()
        assert state["cumulative_poison"] == 5
        assert state["last_iteration"] == 5
        assert len(state["poison_windows_seen"]) == 3

        # Phase 2: create a new logger and restore state
        logger2 = PoisonLogger(
            log_dir=tmp2,
            poison_num_samples=50,
            total_train_samples=1000,
            log_interval=9999,
        )
        logger2.load_state_dict(state)

        assert logger2.cumulative_poison_count == 5

        # Continue logging
        for i in range(5, 10):
            logger2.log_sample(iteration=i + 1, global_sample_idx=i * 10, poison_window_id=i % 3)

        assert logger2.cumulative_poison_count == 10

        logger2.finalize()
        content = _read_log(tmp2)
        assert "Total poison samples encountered: 10" in content
        assert "[RESUMED]" in content

        if verbose:
            print(f"  Saved state: {state}")
            print(f"  Resumed logger reached cumulative={logger2.cumulative_poison_count}")
    finally:
        shutil.rmtree(tmp)
        shutil.rmtree(tmp2)
    print("  PASSED")
    return True


def test_null_logger(verbose=False):
    """Test that NullPoisonLogger is a safe no-op."""
    print("Test 8: NullPoisonLogger no-op")

    null_logger = NullPoisonLogger()

    # None of these should raise
    null_logger.log_sample(1, 10, 3)
    null_logger.log_batch(1, 0, 10, None)
    null_logger.log_batch_from_indices(1, [0, 1], [0, 1], [0, 0])
    null_logger.finalize()

    assert null_logger.cumulative_poison_count == 0
    assert null_logger.is_on_track is True
    assert null_logger.get_state_dict() == {}
    null_logger.load_state_dict({"cumulative_poison": 5})  # should be ignored

    if verbose:
        print("  All NullPoisonLogger methods are safe no-ops")
    print("  PASSED")
    return True


def test_non_rank0_no_file(verbose=False):
    """Test that non-rank-0 loggers don't create any files."""
    print("Test 9: Non-rank-0 does not write")
    tmp = tempfile.mkdtemp(prefix="poison_logger_test_")
    try:
        logger = PoisonLogger(
            log_dir=tmp,
            poison_num_samples=10,
            total_train_samples=100,
            rank=1,  # non-zero rank
        )

        logger.log_sample(1, 10, 3)
        logger.finalize()

        # No file should have been created
        files = os.listdir(tmp)
        assert len(files) == 0, f"Expected no files, found: {files}"

        # Cumulative should still be 0 (log_sample exits early)
        assert logger.cumulative_poison_count == 0

        if verbose:
            print(f"  Rank 1 logger created no files in {tmp}")
    finally:
        shutil.rmtree(tmp)
    print("  PASSED")
    return True


def test_end_to_end_with_finite_dataset(verbose=False):
    """End-to-end: finite dataset + logger, verify final count matches target."""
    print("Test 10: End-to-end finite dataset + logger")
    tmp = tempfile.mkdtemp(prefix="poison_logger_test_")
    try:
        normal_ds = MockDataset(2000)
        poison_ds = MockDataset(200)  # Must be >= poison_n

        total = 5000
        poison_n = 75
        batch_size = 50

        ds = FinitePoisonBlendableDataset(
            datasets=[normal_ds, poison_ds],
            total_samples=total,
            poison_dataset_index=1,
            poison_num_samples=poison_n,
            seed=123,
            spread_mode="jittered",
        )

        logger = PoisonLogger(
            log_dir=tmp,
            poison_num_samples=poison_n,
            total_train_samples=total,
            log_interval=10,
        )

        num_iters = total // batch_size
        for i in range(num_iters):
            iteration = i + 1
            batch_start = i * batch_size
            logger.log_batch(iteration, batch_start, batch_size, ds)

        logger.finalize()

        assert logger.cumulative_poison_count == poison_n, \
            f"Expected {poison_n}, got {logger.cumulative_poison_count}"

        content = _read_log(tmp)
        assert "Match:                            YES" in content

        # Check summaries exist
        summary_count = content.count("[SUMMARY]")
        expected_summaries = num_iters // 10
        assert summary_count == expected_summaries, \
            f"Expected {expected_summaries} summaries, got {summary_count}"

        if verbose:
            print(f"  {total} samples, {poison_n} poison, batch_size={batch_size}")
            print(f"  Logger final count: {logger.cumulative_poison_count}")
            print(f"  Summary lines: {summary_count}")
            # Show last few lines
            lines = content.strip().splitlines()
            print("  Last 5 lines of log:")
            for line in lines[-5:]:
                print(f"    {line}")
    finally:
        shutil.rmtree(tmp)
    print("  PASSED")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test PoisonLogger")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("POISON LOGGER TESTS")
    print("=" * 70)
    print()

    tests = [
        test_log_file_created,
        test_log_sample,
        test_log_batch_with_dataset,
        test_log_batch_from_indices,
        test_summary_intervals,
        test_finalize_report,
        test_state_dict_roundtrip,
        test_null_logger,
        test_non_rank0_no_file,
        test_end_to_end_with_finite_dataset,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            if test_fn(verbose=args.verbose):
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
