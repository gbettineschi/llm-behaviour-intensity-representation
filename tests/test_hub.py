"""Tests for lib.hub lock files and Hub glob patterns.

Run with plain Python (no pytest dependency)::

    .venv/bin/python tests/test_hub.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lib.hub import allow_patterns, lock_path, read_lock, write_lock  # noqa: E402


def test_lock_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "20260530_001930"
        files = ["representations/gemma-2-2b/politeness/avg_token/layer_1.pt"]
        written = write_lock(
            root, repo_id="u/r", revision="a" * 40, run_id="20260530_001930", files=files
        )
        assert written == lock_path(root), f"unexpected lock path {written}"
        lock = read_lock(root)
        assert lock["repo_id"] == "u/r", lock
        assert lock["revision"] == "a" * 40, lock
        assert lock["run_id"] == "20260530_001930", lock
        assert lock["files"] == files, lock
    print("  test_lock_roundtrip PASSED")


def test_write_lock_sorts_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "run"
        write_lock(root, repo_id="u/r", revision="b" * 40, run_id="run", files=["b.pt", "a.pt"])
        assert read_lock(root)["files"] == ["a.pt", "b.pt"], "files must be sorted"
    print("  test_write_lock_sorts_files PASSED")


def test_read_lock_missing_names_push_command():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "20260530_001930"
        root.mkdir(parents=True)
        try:
            read_lock(root)
        except FileNotFoundError as e:
            assert "data_sync.py push" in str(e), f"unhelpful message: {e}"
            assert "20260530_001930" in str(e), f"message omits run id: {e}"
        else:
            raise AssertionError("read_lock must raise when the lock file is absent")
    print("  test_read_lock_missing_names_push_command PASSED")


def test_allow_patterns_no_filters():
    assert allow_patterns("R") == ["R/representations/*/**"], allow_patterns("R")
    print("  test_allow_patterns_no_filters PASSED")


def test_allow_patterns_model_only():
    got = allow_patterns("R", model="gemma-2-2b")
    assert got == ["R/representations/gemma-2-2b/**"], got
    print("  test_allow_patterns_model_only PASSED")


def test_allow_patterns_trait_only_keeps_unembed_cov():
    got = allow_patterns("R", trait="politeness")
    assert got == [
        "R/representations/*/politeness/**",
        "R/representations/*/unembeddings_covariance.pt",
    ], got
    print("  test_allow_patterns_trait_only_keeps_unembed_cov PASSED")


def test_allow_patterns_model_and_trait():
    got = allow_patterns("R", model="gemma-2-2b", trait="politeness")
    assert got == [
        "R/representations/gemma-2-2b/politeness/**",
        "R/representations/gemma-2-2b/unembeddings_covariance.pt",
    ], got
    print("  test_allow_patterns_model_and_trait PASSED")


def main():
    tests = [
        test_lock_roundtrip,
        test_write_lock_sorts_files,
        test_read_lock_missing_names_push_command,
        test_allow_patterns_no_filters,
        test_allow_patterns_model_only,
        test_allow_patterns_trait_only_keeps_unembed_cov,
        test_allow_patterns_model_and_trait,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  ASSERTION FAILED: {e}")
    print()
    if failed:
        print(f"{failed}/{len(tests)} tests FAILED")
        sys.exit(1)
    print(f"All {len(tests)} tests PASSED")


if __name__ == "__main__":
    main()
