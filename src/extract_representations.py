"""Extract activations for a prompt dataset into data/<timestamp>/representations/.

This script owns the input/output locations; the library loads the model, extracts, and saves.

Run from the repo root:  uv run python src/extract_representations.py
"""

from pathlib import Path

from lib.representations import extract_representations

DATASET = Path("data/20260529_212332/sentences/sentences_filtered.jsonl")

if __name__ == "__main__":
    if not DATASET.exists():
        raise SystemExit(DATASET)
    extract_representations(DATASET, DATASET.parent.parent / "representations")
