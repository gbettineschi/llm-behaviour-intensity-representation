import itertools
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - cosine(a.numpy(), b.numpy())


def compute_difference_vectors(
    activations: dict[tuple[str, str], torch.Tensor],
    traits: list[dict],
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Compute consecutive-intensity difference vectors for each trait.

    Parameters
    ----------
    activations : dict[tuple[str, str], torch.Tensor]
        Maps ``(trait, intensity)`` to a mean activation vector.
    traits : list[dict]
        Config entries with keys ``"name"`` and ``"intensities"`` (ordered low → high).

    Returns
    -------
    dict[tuple[str, str, str], torch.Tensor]
        Maps ``(trait, intensity_lo, intensity_hi)`` to ``activation[hi] - activation[lo]``.
    """
    diffs = {}
    for trait_cfg in traits:
        name, intensities = trait_cfg["name"], trait_cfg["intensities"]
        for lo, hi in zip(intensities, intensities[1:]):
            diffs[(name, lo, hi)] = activations[(name, hi)] - activations[(name, lo)]
    return diffs


def similarity_matrix(
    diffs: dict[tuple[str, str, str], torch.Tensor],
) -> tuple[list[str], np.ndarray]:
    """Build the all-pairs cosine similarity matrix over difference vectors.

    Parameters
    ----------
    diffs : dict[tuple[str, str, str], torch.Tensor]
        Output of :func:`compute_difference_vectors`.

    Returns
    -------
    labels : list[str]
        Human-readable label for each row/column (``"trait:lo→hi"``).
    matrix : np.ndarray, shape (n, n)
        Cosine similarities; ``matrix[i, j] == cosine_sim(diffs[i], diffs[j])``.
    """
    keys = list(diffs.keys())
    labels = [f"{t}:{lo}→{hi}" for t, lo, hi in keys]
    n = len(keys)
    matrix = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        matrix[i, j] = cosine_sim(diffs[keys[i]], diffs[keys[j]])
    return labels, matrix


def plot_similarity_matrix(
    labels: list[str],
    matrix: np.ndarray,
    out_path: Path,
) -> None:
    """Save a heatmap of *matrix* to *out_path*."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(labels)
    cell = 2.2
    fig, ax = plt.subplots(figsize=(max(6, n * cell), max(6, n * cell)))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    for i, j in itertools.product(range(n), range(n)):
        color = "white" if abs(matrix[i, j]) > 0.65 else "black"
        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity", fontsize=10)
    ax.set_title("Cosine similarity of activation difference vectors", fontsize=13, pad=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
