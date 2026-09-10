"""Tests for lib.hub lock files and Hub glob patterns.

Run with plain Python (no pytest dependency)::

    .venv/bin/python tests/test_hub.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lib.hub import (  # noqa: E402
    allow_patterns,
    file_digest,
    lock_path,
    read_lock,
    verify_lock,
    write_lock,
)

REL = "representations/gemma-2-2b/politeness/avg_token/layer_1.pt"


def _seed_tree(root: Path, rel: str, body: bytes) -> str:
    """Write a tensor-shaped file and return its digest."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return file_digest(p)


def test_lock_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "20260530_001930"
        files = {REL: "d" * 64}
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
        write_lock(root, repo_id="u/r", revision="b" * 40, run_id="run",
                   files={"b.pt": "2" * 64, "a.pt": "1" * 64})
        assert list(read_lock(root)["files"]) == ["a.pt", "b.pt"], "files must be sorted"
    print("  test_write_lock_sorts_files PASSED")


def test_verify_lock_detects_tampering():
    """The whole point of recording digests: bytes that are not the pinned bytes
    must be caught, not silently analysed."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "run"
        digest = _seed_tree(root, REL, b"the pinned tensor bytes")
        write_lock(root, repo_id="u/r", revision="c" * 40, run_id="run", files={REL: digest})

        res = verify_lock(root)
        assert res["matched"] == [REL], res
        assert not res["mismatched"] and not res["missing"], res

        (root / REL).write_bytes(b"tampered")
        res = verify_lock(root)
        assert res["mismatched"] == [REL], f"tampering not detected: {res}"
        assert not res["matched"], res
    print("  test_verify_lock_detects_tampering PASSED")


def test_verify_lock_treats_absent_as_missing_not_corrupt():
    """A selective pull leaves most pinned files absent. That is normal, and must
    not be reported as a mismatch, or verify would cry wolf on every partial pull."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "run"
        other = "representations/qwen2.5-1.5b/politeness/avg_token/layer_1.pt"
        digest = _seed_tree(root, REL, b"present")
        write_lock(root, repo_id="u/r", revision="e" * 40, run_id="run",
                   files={REL: digest, other: "f" * 64})
        res = verify_lock(root)
        assert res["matched"] == [REL], res
        assert res["missing"] == [other], res
        assert not res["mismatched"], res
    print("  test_verify_lock_treats_absent_as_missing_not_corrupt PASSED")


def test_verify_lock_honours_model_filter():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "run"
        other = "representations/qwen2.5-1.5b/politeness/avg_token/layer_1.pt"
        d1 = _seed_tree(root, REL, b"gemma")
        _seed_tree(root, other, b"qwen")
        write_lock(root, repo_id="u/r", revision="a" * 40, run_id="run",
                   files={REL: d1, other: "0" * 64})  # qwen digest deliberately wrong
        res = verify_lock(root, model="gemma-2-2b")
        assert res["matched"] == [REL] and not res["mismatched"], res
        res = verify_lock(root, model="qwen2.5-1.5b")
        assert res["mismatched"] == [other], res
    print("  test_verify_lock_honours_model_filter PASSED")


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
        test_verify_lock_detects_tampering,
        test_verify_lock_treats_absent_as_missing_not_corrupt,
        test_verify_lock_honours_model_filter,
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
