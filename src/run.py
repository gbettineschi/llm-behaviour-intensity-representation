import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # adds src/ so `lib.*` imports resolve

import torch
from lib.data_typing import load_prompts
from lib.representations import load_model, extract_activations
from lib.analysis import compute_difference_vectors, similarity_matrix, plot_similarity_matrix

CONFIG = Path(__file__).parent / "config.yaml"
DATA   = Path("data/v1/prompts.json")


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = Path(cfg["results_dir"])
    act_dir = results_dir / "activations"
    plot_dir = results_dir / "plots"

    samples = load_prompts(DATA)

    model, tokenizer = load_model(cfg["model"], device)
    activations = extract_activations(samples, model, tokenizer, cfg["layer"], device, out_dir=act_dir)

    diffs = compute_difference_vectors(activations, cfg["traits"])
    labels, matrix = similarity_matrix(diffs)
    plot_similarity_matrix(labels, matrix, plot_dir / "cosine_similarity.png")

    print("Cosine similarity matrix:")
    for i, row_label in enumerate(labels):
        for j, col_label in enumerate(labels):
            print(f"  {row_label} × {col_label}: {matrix[i, j]:.4f}")


if __name__ == "__main__":
    main()
