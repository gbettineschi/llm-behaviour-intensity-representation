import itertools
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold, GroupKFold


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - cosine(a.numpy(), b.numpy())


def _to_np(v) -> np.ndarray:
    """Coerce a torch.Tensor or array-like to a float64 numpy array."""
    if torch.is_tensor(v):
        return v.detach().cpu().numpy().astype(np.float64)
    return np.asarray(v, dtype=np.float64)


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
                lo_scenarios = {
                    s for (t, lv, s) in activations if t == name and lv == lo
                }
                hi_scenarios = {
                    s for (t, lv, s) in activations if t == name and lv == hi
                }
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
        ax.text(
            j,
            i,
            f"{matrix[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=color,
        )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity", fontsize=10)
    ax.set_title(
        "Cosine similarity of activation difference vectors", fontsize=13, pad=14
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Ordinal linearity metrics
#
# Two estimators are provided:
#   * ``ordinal_linearity_metrics`` — the NAIVE pooled estimator. Centroids and
#     the probe pool every sample by intensity level only, mixing language,
#     topic and scenario. Kept as an explicit baseline for before/after
#     comparison; not the estimator to trust (see
#     ``within_scenario_linearity_metrics``).
#   * ``within_scenario_linearity_metrics`` — the WITHIN-SCENARIO estimator.
#     Each ``scenario_id`` fixes language, topic and invariant content, so it
#     applies the within-scenario (fixed-effects) transform — subtracting each
#     scenario's across-level mean — before any projection, uses GroupKFold so a
#     scenario's levels never split across train/test, estimates the projection
#     axis out-of-fold, and reports per-scenario distributions.
# ---------------------------------------------------------------------------


def acts_by_level_from_dict(
    acts_dict: dict[tuple[str, str, str], torch.Tensor],
    levels: list[str],
) -> dict[str, list]:
    """Group ``(trait, intensity, scenario_id) -> vec`` by intensity level."""
    return {lvl: [v for (_, l, _), v in acts_dict.items() if l == lvl] for lvl in levels}


def _per_level_arrays(acts_by_level, levels_ordered):
    """Stack a ``{level: [vec, ...]}`` mapping into design matrix, labels, centroids."""
    X = np.stack([_to_np(v) for lvl in levels_ordered for v in acts_by_level[lvl]])
    y = np.array([r for r, lvl in enumerate(levels_ordered) for _ in acts_by_level[lvl]])
    centroids = np.stack([
        np.stack([_to_np(v) for v in acts_by_level[lvl]]).mean(axis=0)
        for lvl in levels_ordered
    ])
    return X, y, centroids


def ordinal_linearity_metrics(acts_by_level, levels_ordered, cv=5, ridge_alpha=1.0, seed=0):
    """NAIVE pooled linearity metrics (baseline; see module note).

    Spearman/Kendall use an in-sample projection axis; the ridge probe uses a
    shuffled KFold that leaks scenario identity across folds. Retained only for
    explicit comparison against :func:`within_scenario_linearity_metrics`.
    """
    X, y, centroids = _per_level_arrays(acts_by_level, levels_ordered)

    direction = centroids[-1] - centroids[0]
    direction = direction / np.linalg.norm(direction)
    scalar = X @ direction
    spearman = float(spearmanr(scalar, y).statistic)
    kendall = float(kendalltau(scalar, y).statistic)

    # Shuffled KFold — X is ordered by level, so an unshuffled split would put
    # only one class in each test fold (SS_tot = 0 -> R^2 undefined -> NaN).
    splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    r2 = float(cross_val_score(Ridge(alpha=ridge_alpha), X, y, cv=splitter, scoring="r2").mean())

    if len(levels_ordered) == 3:
        midpoint = (centroids[0] + centroids[-1]) / 2
        midpoint_residual = float(
            np.linalg.norm(centroids[1] - midpoint) / np.linalg.norm(centroids[-1] - centroids[0])
        )
    else:
        midpoint_residual = float("nan")

    centered = centroids - centroids.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    pc1_frac = float((s[0] ** 2) / (s ** 2).sum())

    return {
        "spearman": spearman,
        "kendall": kendall,
        "probe_r2": r2,
        "midpoint_residual": midpoint_residual,
        "pc1_frac": pc1_frac,
    }


def _infer_trait(activations, trait):
    if trait is not None:
        return trait
    traits = {k[0] for k in activations}
    if len(traits) != 1:
        raise ValueError(f"Pass `trait` explicitly; activations contain traits {sorted(traits)}")
    return next(iter(traits))


def scenario_triples(
    activations: dict[tuple[str, str, str], torch.Tensor],
    levels_ordered: list[str],
    trait: str | None = None,
) -> dict[str, np.ndarray]:
    """Group activations by scenario, keeping only *complete* scenarios.

    A scenario is complete if it has an activation for every level in
    ``levels_ordered``. Returns ``{scenario_id: array (L, D)}`` with rows ordered
    low -> high as in ``levels_ordered``. Partial scenarios (missing a level) are
    dropped — they cannot contribute a matched within-scenario contrast.
    """
    trait = _infer_trait(activations, trait)
    by_scen: dict[str, dict[str, np.ndarray]] = {}
    for (t, lvl, sid), vec in activations.items():
        if t != trait or lvl not in levels_ordered:
            continue
        by_scen.setdefault(sid, {})[lvl] = _to_np(vec)
    complete = {}
    for sid, lv in by_scen.items():
        if all(l in lv for l in levels_ordered):
            complete[sid] = np.stack([lv[l] for l in levels_ordered])
    return complete


def within_center(
    activations: dict[tuple[str, str, str], torch.Tensor],
    levels_ordered: list[str],
    trait: str | None = None,
) -> dict[tuple[str, str, str], np.ndarray]:
    """Within-scenario (fixed-effects) transform.

    Subtracts each scenario's across-level mean from its level vectors, removing
    the additive per-scenario offset (language, topic, invariant content). Only
    complete scenarios are returned. Keys match the input ``(trait, level,
    scenario_id)``; values are float64 numpy arrays.
    """
    trait = _infer_trait(activations, trait)
    triples = scenario_triples(activations, levels_ordered, trait)
    out = {}
    for sid, mat in triples.items():
        centered = mat - mat.mean(axis=0, keepdims=True)
        for i, lvl in enumerate(levels_ordered):
            out[(trait, lvl, sid)] = centered[i]
    return out


def within_center_paraphrase(
    activations: dict[tuple[str, str, str, str], torch.Tensor],
    levels_ordered: list[str],
    trait: str | None = None,
) -> dict[tuple[str, str, str, str], np.ndarray]:
    """Paraphrase-level within-scenario (fixed-effects) transform.

    Sibling of :func:`within_center` for paraphrase-keyed activations
    ``(trait, level, scenario_id, paraphrase_id)``. For each scenario carrying
    at least one paraphrase at every level in ``levels_ordered``, subtracts the
    mean of all of that scenario's paraphrase vectors (across the requested
    levels) from each. Removes the additive per-scenario offset while keeping
    paraphrase-level granularity. Scenarios missing a level are dropped.
    """
    trait = _infer_trait(activations, trait)
    by_scen: dict[str, dict[str, list[tuple[tuple, np.ndarray]]]] = {}
    for key, vec in activations.items():
        t, lvl, sid, _pid = key
        if t != trait or lvl not in levels_ordered:
            continue
        by_scen.setdefault(sid, {}).setdefault(lvl, []).append((key, _to_np(vec)))

    out: dict[tuple[str, str, str, str], np.ndarray] = {}
    for by_level in by_scen.values():
        if not all(lvl in by_level for lvl in levels_ordered):
            continue
        all_vecs = np.stack([v for lvl in levels_ordered for (_, v) in by_level[lvl]])
        mean = all_vecs.mean(axis=0)
        for lvl in levels_ordered:
            for key, vec in by_level[lvl]:
                out[key] = vec - mean
    return out


def per_scenario_step_cosines(
    activations: dict[tuple[str, str, str], torch.Tensor],
    levels_ordered: list[str],
    trait: str | None = None,
) -> np.ndarray:
    """Cosine between consecutive within-scenario steps, per scenario.

    For ``levels_ordered = [neg, neut, pos]`` this is the cosine between
    ``(neut - neg)`` and ``(pos - neut)`` computed *inside* each scenario, then
    returned as an array over complete scenarios. Distinguishes per-context
    alignment from the cosine-of-means reported by §3.2.
    """
    if len(levels_ordered) != 3:
        raise ValueError("per_scenario_step_cosines is defined for 3 levels")
    triples = scenario_triples(activations, levels_ordered, trait)
    out = []
    for mat in triples.values():
        a = mat[1] - mat[0]
        b = mat[2] - mat[1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            out.append(float(np.clip((a @ b) / (na * nb), -1.0, 1.0)))
    return np.array(out)


def within_scenario_linearity_metrics(
    activations: dict[tuple[str, str, str], torch.Tensor],
    levels_ordered: list[str],
    trait: str | None = None,
    cv: int = 5,
    ridge_alpha: float = 1.0,
    seed: int = 0,
) -> dict:
    """Within-scenario ordinal linearity metrics (each scenario is its own control).

    Corrects the confounds of :func:`ordinal_linearity_metrics`:
      * within-scenario (fixed-effects) transform removes the additive
        language/topic/content offset before any projection;
      * ridge probe uses ``GroupKFold`` keyed by scenario (no leakage of a
        scenario's offset across train/test, and every test fold retains all
        levels so R^2 is defined);
      * the projection axis for Spearman/Kendall is estimated *out-of-fold*;
      * midpoint residual and PC1 fraction are reported per scenario, not only
        on pooled centroids.

    Note on invariance: ``midpoint_residual_pooled`` and ``pc1_frac_centroids``
    are computed on within-centered pooled centroids, which (for balanced
    complete triples) equal the raw pooled values up to a global shift — they
    are the *naive* geometry, retained for comparison. The trustworthy,
    confound-free views are the per-scenario distributions.
    """
    trait = _infer_trait(activations, trait)
    triples = scenario_triples(activations, levels_ordered, trait)
    sids = sorted(triples)
    n_scen = len(sids)
    L = len(levels_ordered)
    if n_scen < 2:
        raise ValueError(f"Need >=2 complete scenarios; found {n_scen}")

    mats = np.stack([triples[s] for s in sids])          # (S, L, D)
    centered = mats - mats.mean(axis=1, keepdims=True)    # within-scenario transform
    D = mats.shape[-1]
    X = centered.reshape(n_scen * L, D)
    y = np.tile(np.arange(L), n_scen)
    groups = np.repeat(np.arange(n_scen), L)

    n_splits = min(cv, n_scen)
    gkf = GroupKFold(n_splits=n_splits)

    # Ridge probe R^2 — scenario-grouped folds (no leakage).
    probe_r2 = float(cross_val_score(
        Ridge(alpha=ridge_alpha), X, y, cv=gkf, groups=groups, scoring="r2"
    ).mean())

    # Out-of-fold projection axis: mean within-scenario full-span (hi - lo) over
    # train scenarios, applied to within-centered held-out vectors.
    lo_i, hi_i = 0, L - 1
    full_span = mats[:, hi_i] - mats[:, lo_i]             # (S, D)
    scalars = np.empty(n_scen * L)
    for tr, te in gkf.split(X, y, groups):
        tr_scen = np.unique(groups[tr])
        axis = full_span[tr_scen].mean(axis=0)
        nrm = np.linalg.norm(axis)
        axis = axis / nrm if nrm > 0 else axis
        scalars[te] = X[te] @ axis
    spearman = float(spearmanr(scalars, y).statistic)
    kendall = float(kendalltau(scalars, y).statistic)

    # Per-scenario monotonicity along the (descriptive) global within axis.
    global_axis = full_span.mean(axis=0)
    global_axis = global_axis / (np.linalg.norm(global_axis) or 1.0)
    proj = centered @ global_axis                         # (S, L)
    monotone_fraction = float(np.all(np.diff(proj, axis=1) > 0, axis=1).mean())

    # Per-scenario midpoint residual (only meaningful for 3 levels).
    if L == 3:
        neg, neu, pos = mats[:, 0], mats[:, 1], mats[:, 2]
        span = np.linalg.norm(pos - neg, axis=1)
        midres_per = np.linalg.norm(neu - 0.5 * (neg + pos), axis=1)
        midres_per = midres_per / np.where(span > 0, span, np.nan)
    else:
        midres_per = None

    # Per-scenario PC1 fraction (3 centered points span <=2 dims).
    pc1_per = np.array([
        (lambda sv: (sv[0] ** 2) / (sv ** 2).sum())(np.linalg.svd(centered[s], compute_uv=False))
        for s in range(n_scen)
    ])

    # Pooled-centroid geometry (naive; invariant to within-centering for balanced triples).
    centroids = centered.mean(axis=0)                     # (L, D)
    cc = centroids - centroids.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(cc, compute_uv=False)
    pc1_frac_centroids = float((sv[0] ** 2) / (sv ** 2).sum())
    if L == 3:
        mp = 0.5 * (centroids[0] + centroids[2])
        denom = np.linalg.norm(centroids[2] - centroids[0])
        midpoint_residual_pooled = float(np.linalg.norm(centroids[1] - mp) / denom) if denom > 0 else float("nan")
    else:
        midpoint_residual_pooled = float("nan")

    return {
        "spearman": spearman,
        "kendall": kendall,
        "probe_r2": probe_r2,
        "monotone_fraction": monotone_fraction,
        "midpoint_residual_pooled": midpoint_residual_pooled,
        "midpoint_residual_per_scenario": midres_per,
        "midpoint_residual_median": float(np.nanmedian(midres_per)) if midres_per is not None else float("nan"),
        "pc1_frac_centroids": pc1_frac_centroids,
        "pc1_frac_per_scenario": pc1_per,
        "pc1_frac_per_scenario_mean": float(pc1_per.mean()),
        "n_scenarios": n_scen,
    }


# Metrics with the "more linear = smaller" sign; everywhere else, larger is more linear.
_LOWER_IS_BETTER: frozenset[str] = frozenset({"midpoint_residual_median", "midpoint_residual_pooled"})

DEFAULT_NULL_TEST_KEYS: tuple[str, ...] = (
    "spearman", "kendall", "probe_r2",
    "monotone_fraction", "midpoint_residual_median", "pc1_frac_centroids",
)


def permutation_null_within_scenario(
    activations: dict[tuple[str, str, str], torch.Tensor],
    levels_ordered: list[str],
    trait: str | None = None,
    *,
    n_perm: int = 100,
    seed: int = 13,
    test_keys: tuple[str, ...] = DEFAULT_NULL_TEST_KEYS,
) -> dict:
    """Within-scenario label-permutation null for the linearity metrics.

    Inside each scenario the level labels are shuffled — preserving the
    scenario/topic structure so the null isolates the ordinal-linearity signal
    rather than also destroying scenario identity. For each permutation,
    :func:`within_scenario_linearity_metrics` is recomputed and the value of
    every ``test_keys`` entry is recorded.

    Monte-Carlo p-values use the conservative ``(1 + #beyond) / (1 + n_perm)``
    estimator (floor ``1 / (1 + n_perm)``). For
    :data:`_LOWER_IS_BETTER` metrics the left tail is used; the rest use the right.

    Returns
    -------
    dict
        ``{"observed": {key: float}, "null": {key: ndarray}, "p_values": {key: float},
        "n_perm": int}``.
    """
    trait = _infer_trait(activations, trait)
    triples = scenario_triples(activations, levels_ordered, trait=trait)
    observed_full = within_scenario_linearity_metrics(activations, levels_ordered, trait=trait)
    observed = {k: observed_full[k] for k in test_keys}

    rng = np.random.default_rng(seed)
    null_arrays: dict[str, list[float]] = {k: [] for k in test_keys}
    for _ in range(n_perm):
        perm_acts: dict[tuple[str, str, str], np.ndarray] = {}
        for sid, mat in triples.items():
            order = rng.permutation(len(levels_ordered))
            for i, lvl in enumerate(levels_ordered):
                perm_acts[(trait, lvl, sid)] = mat[order[i]]
        m = within_scenario_linearity_metrics(perm_acts, levels_ordered, trait=trait)
        for k in test_keys:
            null_arrays[k].append(m[k])
    null = {k: np.asarray(v, dtype=float) for k, v in null_arrays.items()}

    p_values: dict[str, float] = {}
    for k in test_keys:
        obs_val = observed[k]
        if isinstance(obs_val, float) and np.isnan(obs_val):
            p_values[k] = float("nan")
            continue
        if k in _LOWER_IS_BETTER:
            p_values[k] = float((1 + (null[k] <= obs_val).sum()) / (1 + n_perm))
        else:
            p_values[k] = float((1 + (null[k] >= obs_val).sum()) / (1 + n_perm))

    return {"observed": observed, "null": null, "p_values": p_values, "n_perm": n_perm}
