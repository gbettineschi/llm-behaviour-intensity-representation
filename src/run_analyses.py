"""Sweep orchestrator: run all three analysis drivers over every model x trait x
token_pooling combo present on disk, for a set of seeds.

Thin wrapper around each driver's existing ``main()`` / ``aggregate_analysis()`` —
no analysis logic lives here. A combo whose representations aren't on disk yet
(e.g. qwen2.5-7b before its cloud-GPU extraction lands) is skipped with a warning.

Run from the repo root:  uv run python src/run_analyses.py
Or scoped:
    uv run python src/run_analyses.py --models gemma-2-2b,qwen2.5-1.5b --traits politeness,formality
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ordinal_linearity
import replicate_tigges
import trait_geometry
from aggregate_results import aggregate_analysis
from lib.config import DEFAULT_SEEDS, MODELS, rep_dir, seeds_base_dir
from lib.traits import TRAITS

DATASET_ROOT = Path("data/20260530_001930")

# analysis name -> driver's main(model, trait, token_pooling, seed)
DRIVERS = {
    "replication_tigges": replicate_tigges.main,
    "ordinal_linearity": ordinal_linearity.main,
    "trait_geometry": trait_geometry.main,
}


def run_sweep(
    models: list[str], traits: list[str], poolings: list[str], seeds: list[int]
) -> None:
    for model in models:
        for trait in traits:
            for pooling in poolings:
                rd = rep_dir(DATASET_ROOT, model, trait, pooling)
                if not rd.exists():
                    print(f"skip {model}/{trait}/{pooling}: no representations at {rd}")
                    continue
                for analysis, driver_main in DRIVERS.items():
                    for seed in seeds:
                        driver_main(model, trait, pooling, seed)
                    if len(seeds) > 1:
                        aggregate_analysis(
                            seeds_base_dir(DATASET_ROOT.name, analysis, model, trait, pooling)
                        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run all analysis drivers over a model x trait x pooling sweep."
    )
    ap.add_argument(
        "--models", default=",".join(sorted(MODELS)),
        help="Comma-separated models (default: every model in the registry).",
    )
    ap.add_argument(
        "--traits", default=",".join(sorted(TRAITS)),
        help="Comma-separated traits (default: every trait in the registry).",
    )
    ap.add_argument(
        "--poolings", default="avg,last",
        help="Comma-separated token poolings (default: %(default)s).",
    )
    ap.add_argument(
        "--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated master seeds (default: %(default)s).",
    )
    args = ap.parse_args()

    run_sweep(
        models=args.models.split(","),
        traits=args.traits.split(","),
        poolings=args.poolings.split(","),
        seeds=[int(s) for s in args.seeds.split(",")],
    )
