"""Extract activations for a prompt dataset into data/<timestamp>/representations/.

This script owns the input/output locations; the library loads the model, extracts, and saves.

Run from the repo root:  uv run python src/extract_representations.py
"""

from pathlib import Path

from lib.representations import extract_representations

DATASET = Path("data/20260530_001930/sentences/sentences_filtered.jsonl")
# Token pooling over the prompt's prefill hidden states: "last" = final content token,
# "avg" = average over content tokens. No generation happens. Output folder is f"{TOKEN_POOLING}_token".
TOKEN_POOLING = "last"
MODEL = "google/gemma-2-2b"
LAYERS = list(range(1, 23))

if __name__ == "__main__":
    if not DATASET.exists():
        raise SystemExit(DATASET)
    extract_representations(
        DATASET,
        DATASET.parent.parent / "representations",
        model_name=MODEL,
        layers=LAYERS,
        token_pooling=TOKEN_POOLING,
    )
