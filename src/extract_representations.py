"""Extract activations for a prompt dataset into data/<timestamp>/representations/<model>/.

This script owns the input/output locations; the library loads the model, extracts, and saves.
Extraction is deterministic (prefill only, no sampling), so there is no seed to set.

Run from the repo root:  uv run python src/extract_representations.py --model gemma-2-2b
"""

import argparse
from pathlib import Path

from lib.config import DEFAULT_MODEL, MODELS
from lib.representations import extract_representations

DATASET = Path("data/20260530_001930/sentences/sentences_filtered.jsonl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    ap.add_argument(
        "--token-pooling",
        choices=("avg", "last", "both"),
        default="both",
        help="'avg' = average over content tokens, 'last' = final content token; "
        "'both' extracts the two in a single forward pass",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if not DATASET.exists():
        raise SystemExit(DATASET)
    poolings = ("avg", "last") if args.token_pooling == "both" else (args.token_pooling,)
    extract_representations(
        DATASET,
        DATASET.parent.parent / "representations" / args.model,
        model_name=MODELS[args.model]["hf_id"],
        token_poolings=poolings,
        batch_size=args.batch_size,
    )
