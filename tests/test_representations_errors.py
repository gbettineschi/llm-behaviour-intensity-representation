"""The missing-tensor error must tell you how to fetch the tensors.

Run with plain Python (no pytest dependency)::

    .venv/bin/python tests/test_representations_errors.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lib.representations import load_representations  # noqa: E402


def _rep_dir(tmp: str) -> Path:
    return Path(tmp) / "data" / "20260530_001930" / "representations" / "m" / "t" / "avg_token"


def test_missing_directory_names_pull_command():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_representations(_rep_dir(tmp))
        except FileNotFoundError as e:
            assert "data_sync.py pull" in str(e), f"unhelpful message: {e}"
            assert "20260530_001930" in str(e), f"message omits run id: {e}"
        else:
            raise AssertionError("must raise when the representation directory is absent")
    print("  test_missing_directory_names_pull_command PASSED")


def test_empty_directory_does_not_return_empty_dict():
    with tempfile.TemporaryDirectory() as tmp:
        d = _rep_dir(tmp)
        d.mkdir(parents=True)
        try:
            load_representations(d)
        except FileNotFoundError as e:
            assert "data_sync.py pull" in str(e), f"unhelpful message: {e}"
        else:
            raise AssertionError("an empty directory must raise, not return {}")
    print("  test_empty_directory_does_not_return_empty_dict PASSED")


def test_explicit_layer_also_raises():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_representations(_rep_dir(tmp), layer=5)
        except FileNotFoundError as e:
            assert "data_sync.py pull" in str(e), f"unhelpful message: {e}"
        else:
            raise AssertionError("the layer= branch must raise too")
    print("  test_explicit_layer_also_raises PASSED")


def main():
    tests = [
        test_missing_directory_names_pull_command,
        test_empty_directory_does_not_return_empty_dict,
        test_explicit_layer_also_raises,
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
