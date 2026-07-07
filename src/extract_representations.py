"""Extract activations for a trait dataset into data/<timestamp>/representations/<model>/<trait>/.

This script owns the input/output locations; the library loads the model, extracts, and saves.
Extraction is deterministic (prefill only, no sampling), so there is no seed to set.

Run from the repo root:  uv run python src/extract_representations.py --model gemma-2-2b --trait politeness
"""

import argparse
from pathlib import Path

from lib.config import DEFAULT_MODEL, MODELS, dataset_path, unembed_cov_path
from lib.representations import extract_representations
from lib.traits import DEFAULT_TRAIT, TRAITS

DATA_ROOT = Path("data/20260530_001930")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    ap.add_argument("--trait", choices=sorted(TRAITS), default=DEFAULT_TRAIT)
    ap.add_argument(
        "--token-pooling",
        choices=("avg", "last", "both"),
        default="both",
        help="'avg' = average over content tokens, 'last' = final content token; "
        "'both' extracts the two in a single forward pass",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    dataset = dataset_path(DATA_ROOT, args.trait)
    if not dataset.exists():
        raise SystemExit(dataset)
    poolings = ("avg", "last") if args.token_pooling == "both" else (args.token_pooling,)
    extract_representations(
        dataset,
        DATA_ROOT / "representations" / args.model / args.trait,
        model_name=MODELS[args.model]["hf_id"],
        token_poolings=poolings,
        batch_size=args.batch_size,
        cov_path=unembed_cov_path(DATA_ROOT, args.model),
    )
