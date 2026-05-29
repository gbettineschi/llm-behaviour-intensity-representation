"""Difference-vector geometry analysis on a pre-extracted representations/ directory.

Run from the repo root:  uv run python src/run.py [data/<timestamp>/representations]
Defaults to the latest one. Run extract_representations.py first.
"""
import sys
from pathlib import Path

from lib.analysis import (
    compute_difference_vectors,
    plot_similarity_matrix,
    similarity_matrix,
)
from lib.sentences import LEVELS
from lib.representations import load_representations

LAYER = 13
TRAIT = "politeness"


def latest_representations() -> Path:
    runs = sorted(Path("data").glob("*/representations"))
    if not runs:
        raise SystemExit(
            "No data/<timestamp>/representations found — run extract_representations.py first."
        )
    return runs[-1]


def main(
    rep_dir: str | Path | None = None, *, results_dir: str | Path = "results/analysis"
) -> None:
    rep_dir = Path(rep_dir) if rep_dir else latest_representations()
    results_dir = Path(results_dir)

    activations = load_representations(rep_dir, layer=LAYER)
    levels = [lvl for lvl in LEVELS if any(i == lvl for (_, i, _) in activations)]
    traits_cfg = [{"name": TRAIT, "intensities": levels}]
    print(f"{len(activations)} vectors (layer {LAYER}) from {rep_dir}")

    diffs = compute_difference_vectors(activations, traits_cfg)
    labels, matrix = similarity_matrix(diffs)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_similarity_matrix(labels, matrix, results_dir / "cosine_similarity.png")

    print("difference-vector cosine matrix:")
    for i, row_label in enumerate(labels):
        print(
            "  "
            + row_label
            + "  "
            + "  ".join(f"{matrix[i, j]:.3f}" for j in range(len(labels)))
        )
    print(f"Saved plot under {results_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
