"""Polarity-direction estimators from Tigges et al. §2.2 and friends.

Reusable building blocks for any analysis that needs (i) a unit "trait
direction" estimated from labelled activations, (ii) cross-estimator agreement,
or (iii) the direction-as-classifier accuracy / projection-separation
diagnostics used by the replication.

All functions take paraphrase-level activations keyed by
``(trait, intensity, scenario_id, paraphrase_id)``; the scenario id is what the
caller passes to ``GroupKFold`` so paraphrases of the same scenario stay
together across folds.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold


# Paper §2.2 estimators, plus a random-floor baseline used by the replication.
METHODS: tuple[str, ...] = ("MeanDiff", "KMeans", "LogReg", "PCA")
NAMES: tuple[str, ...] = METHODS + ("Random",)


def level_table(
    activations: dict[tuple[str, str, str, str], "np.ndarray | object"],
    levels: tuple[str, str],
    trait: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack activations for a two-level contrast.

    Returns ``(X, y, scenarios)`` where ``y == 1`` for ``levels[1]`` and ``0``
    for ``levels[0]``, and ``scenarios`` carries the per-row ``scenario_id``
    (the grouping variable for scenario-grouped CV).
    """
    lo, hi = levels
    rows, y, scen = [], [], []
    for (t, lvl, sid, _pid), vec in activations.items():
        if t != trait or lvl not in (lo, hi):
            continue
        rows.append(np.asarray(vec, dtype=np.float64))
        y.append(1 if lvl == hi else 0)
        scen.append(sid)
    return np.stack(rows), np.array(y), np.array(scen)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def direction(method: str, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit trait direction estimated by one of the §2.2 methods.

    The vector is oriented so the positive class projects to higher values.
    """
    if method == "MeanDiff":
        d = X[y == 1].mean(0) - X[y == 0].mean(0)
    elif method == "KMeans":
        c = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X).cluster_centers_
        d = c[1] - c[0]
    elif method == "LogReg":
        d = LogisticRegression(max_iter=2000).fit(X, y).coef_[0]
    elif method == "PCA":
        d = PCA(n_components=1, random_state=0).fit(X).components_[0]
    elif method == "Random":
        d = np.random.default_rng(0).standard_normal(X.shape[1])
    else:
        raise ValueError(method)
    d = _unit(d)
    if (X[y == 1] @ d).mean() < (X[y == 0] @ d).mean():
        d = -d
    return d


def cosine_matrix(dir_dict: dict[str, np.ndarray], order: "tuple[str, ...]") -> np.ndarray:
    """All-pairs cosine of unit directions in ``dir_dict``, ordered as ``order``."""
    return np.array([[float(dir_dict[a] @ dir_dict[b]) for b in order] for a in order])


def cv_direction_accuracy(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
) -> tuple[float, float]:
    """Mean ± SD balanced accuracy of the projection-threshold classifier under
    scenario-grouped K-fold CV. Direction and threshold are fit on the training
    fold; a held-out paraphrase is labelled by which side of the threshold its
    projection falls on.
    """
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    accs = []
    for tr, te in gkf.split(X, y, groups):
        d = direction(method, X[tr], y[tr])
        thr = 0.5 * ((X[tr][y[tr] == 1] @ d).mean() + (X[tr][y[tr] == 0] @ d).mean())
        accs.append(balanced_accuracy_score(y[te], (X[te] @ d > thr).astype(int)))
    return float(np.mean(accs)), float(np.std(accs))


def projection_stats(
    X: np.ndarray, y: np.ndarray, axis: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Per-vector projection onto ``axis`` plus ROC-AUC and Cohen's d."""
    proj = X @ axis
    auc = roc_auc_score(y, proj)
    pooled_sd = np.sqrt(0.5 * (proj[y == 1].var(ddof=1) + proj[y == 0].var(ddof=1)))
    d = (proj[y == 1].mean() - proj[y == 0].mean()) / pooled_sd if pooled_sd > 0 else float("inf")
    return proj, auc, d
