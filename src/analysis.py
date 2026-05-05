import itertools
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - cosine(a.numpy(), b.numpy())


def compute_difference_vectors(
    activations: dict[tuple[str, str, str], torch.Tensor],
    traits: list[dict],
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Compute all-pairs intensity difference vectors, averaged across scenarios.

    For every ordered pair (lo, hi) of intensities, the difference vector is
    computed per scenario as ``activation[hi] - activation[lo]``, then averaged
    across all scenarios that have both levels.

    Parameters
    ----------
    activations : dict[tuple[str, str, str], torch.Tensor]
        Maps ``(trait, intensity, scenario_id)`` to an activation vector.
    traits : list[dict]
        Config entries with keys ``"name"`` and ``"intensities"`` (ordered low → high).

    Returns
    -------
    dict[tuple[str, str, str], torch.Tensor]
        Maps ``(trait, lo, hi)`` to the mean of ``activation[hi] - activation[lo]``
        across all scenarios.
    """
    diffs = {}
    for trait_cfg in traits:
        name, intensities = trait_cfg["name"], trait_cfg["intensities"]
        # Order: consecutive pairs descending (high-mid, mid-low), then full spans (high-low, ...)
        for step in range(1, len(intensities)):
            for j in range(len(intensities) - 1, step - 1, -1):
                lo, hi = intensities[j - step], intensities[j]
                lo_scenarios = {s for (t, lv, s) in activations if t == name and lv == lo}
                hi_scenarios = {s for (t, lv, s) in activations if t == name and lv == hi}
                shared = sorted(lo_scenarios & hi_scenarios)
                if not shared:
                    continue
                per_scenario = [
                    activations[(name, hi, sid)] - activations[(name, lo, sid)]
                    for sid in shared
                ]
                diffs[(name, lo, hi)] = torch.stack(per_scenario).mean(dim=0)
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
    labels = [f"{t}:{hi}-{lo}" for t, lo, hi in keys]
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
