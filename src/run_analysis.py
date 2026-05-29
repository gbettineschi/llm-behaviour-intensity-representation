"""Run the representation-geometry analysis on a generated prompt dataset.

Loads a data/<timestamp>/filtered.jsonl (the latest by default), extracts Gemma-2-2B
activations at one layer, computes the difference-vector cosine matrix, and saves the
activations + a heatmap under results/analysis/.

Run from the repo root:  uv run python src/run_analysis.py [data/<timestamp>]
"""
import sys
from pathlib import Path

import torch

from lib.analysis import (
    compute_difference_vectors,
    plot_similarity_matrix,
    similarity_matrix,
)
from lib.prompts import LEVELS, load_accepted
from lib.representations import extract_activations, load_model, save_activations

MODEL = "google/gemma-2-2b"
LAYER = 13
TRAIT = "politeness"


def latest_dataset() -> Path:
    runs = sorted(Path("data").glob("*/filtered.jsonl"))
    if not runs:
        raise SystemExit(
            "No data/<timestamp>/filtered.jsonl found — run generate_prompts.py first."
        )
    return runs[-1]


def main(
    dataset: str | Path | None = None, *, results_dir: str | Path = "results/analysis"
) -> None:
    dataset = Path(dataset) if dataset else latest_dataset()
    results_dir = Path(results_dir)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    samples = load_accepted(dataset)
    levels = [lvl for lvl in LEVELS if any(s.intensity == lvl for s in samples)]
    traits_cfg = [{"name": TRAIT, "intensities": levels}]
    print(f"{len(samples)} samples from {dataset}  |  device={device}")

    model, tokenizer = load_model(MODEL, device)
    activations = extract_activations(samples, model, tokenizer, LAYER, device)
    save_activations(activations, results_dir / "activations")

    diffs = compute_difference_vectors(activations, traits_cfg)
    labels, matrix = similarity_matrix(diffs)
    plot_similarity_matrix(labels, matrix, results_dir / "cosine_similarity.png")

    print("difference-vector cosine matrix:")
    for i, row_label in enumerate(labels):
        print(
            "  "
            + row_label
            + "  "
            + "  ".join(f"{matrix[i, j]:.3f}" for j in range(len(labels)))
        )
    print(f"Saved activations + plot under {results_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
