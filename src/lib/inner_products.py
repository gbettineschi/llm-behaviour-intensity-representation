"""Inner-product / whitening machinery for difference-vector geometry.

Builds five whitening matrices that re-weight activation space before the
§3.2 cosine geometry is computed, and provides the disattenuated-cosine
estimator that corrects each space's cosines for finite-sample noise:

    * Euclidean (identity; reproduces §3.2)
    * Anisotropy-corrected (per-dimension z-score)
    * Park causal (inverse-sqrt of unembedding-row covariance)
    * Mahalanobis / LDA (inverse-sqrt of pooled within-level covariance)
    * Within-subjects noise (inverse-sqrt of two-way (level + scenario)
      residual covariance — leaves the level mean untouched as signal)

The disattenuated cosine divides each off-diagonal entry by sqrt(R_i * R_j),
where R is the Spearman-Brown-corrected reliability obtained by splitting
scenarios in half and recomputing each difference vector independently.

Activations are keyed by ``(trait, level, scenario_id)``; this is the
scenario-level view produced by :func:`lib.representations.pool_by_scenario_level`.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from lib.analysis import _to_np, compute_difference_vectors, scenario_triples


SpaceName = str
WhitenMap = dict[SpaceName, np.ndarray]


# ---------------------------------------------------------------------------
# Whitening primitives
# ---------------------------------------------------------------------------

def inv_sqrt_psd(cov: np.ndarray, rcond: float = 1e-6) -> np.ndarray:
    """Symmetric inverse square root of a PSD matrix.

    Eigenvalues are floored at ``rcond * max_eigenvalue`` before inversion so a
    rank-deficient covariance still yields a finite, well-conditioned whitener.
    """
    cov = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, rcond * float(evals.max()), None)
    return (evecs * evals**-0.5) @ evecs.T


def _stack_acts(acts_f64: dict[tuple, np.ndarray]) -> np.ndarray:
    """Stack a ``{key: vec}`` mapping into a ``(N, D)`` design matrix."""
    return np.stack(list(acts_f64.values()))


def _pooled_within_level_residual(
    acts_f64: dict[tuple, np.ndarray], levels: list[str]
) -> np.ndarray:
    """Stack of (X_lvl - mean_lvl) across the requested levels — the §3.4 pooled
    within-level residual used by the LDA whitener."""
    return np.concatenate([
        Xl - Xl.mean(axis=0, keepdims=True)
        for Xl in (
            np.stack([acts_f64[k] for k in acts_f64 if k[1] == lvl])
            for lvl in levels
        )
    ])


def _two_way_residual(
    acts_f64: dict[tuple, np.ndarray], levels: list[str], trait: str
) -> np.ndarray:
    """Residual after removing level mean AND scenario mean from complete triples.

    Leaves the interaction + paraphrase noise — the within-subjects "noise"
    covariance. The level mean is removed from the noise estimate, not from
    the signal, so the trait axis stays available downstream.
    """
    triples = scenario_triples(acts_f64, levels, trait=trait)
    M = np.stack([triples[s] for s in sorted(triples)])         # (S, L, D)
    grand = M.mean(axis=(0, 1), keepdims=True)
    return (M - M.mean(axis=0, keepdims=True) - M.mean(axis=1, keepdims=True) + grand
            ).reshape(-1, M.shape[-1])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SPACE_ORDER: tuple[SpaceName, ...] = (
    "Euclidean (raw)",
    "Anisotropy-corrected",
    "Park causal",
    "Mahalanobis / LDA",
    "Within-subjects noise",
)


def build_inner_product_spaces(
    activations: dict[tuple[str, str, str], torch.Tensor],
    *,
    unembed_cov_path,
    levels: list[str],
    trait: str,
) -> tuple[WhitenMap, dict]:
    """Build the five whitening matrices and return them with a diagnostics dict.

    Returns
    -------
    spaces : dict[str, np.ndarray]
        Maps each space name in :data:`SPACE_ORDER` to its (D, D) whitener.
    diagnostics : dict
        Keys: ``D``, ``N``, ``shrinkage_within_level``, ``shrinkage_within_scenario``,
        ``frobenius_norms`` ({name: float}).
    """
    acts_f64 = {k: _to_np(v) for k, v in activations.items()}
    X_all = _stack_acts(acts_f64)
    D = X_all.shape[1]

    # 1 - Euclidean: identity (reproduces §3.2).
    W_eucl = np.eye(D)

    # 2 - Anisotropy correction: diagonal z-score per dimension.
    sigma = X_all.std(axis=0)
    W_aniso = np.diag(1.0 / np.clip(sigma, 1e-8, None))

    # 3 - Park causal: inverse-sqrt of the unembedding-row covariance.
    cov_unembed = torch.load(unembed_cov_path, weights_only=False).numpy()
    W_park = inv_sqrt_psd(cov_unembed)

    # 4 - Mahalanobis / LDA: inverse-sqrt of pooled within-level covariance.
    within_lvl = _pooled_within_level_residual(acts_f64, levels)
    lw_lvl = LedoitWolf(assume_centered=True).fit(within_lvl)
    W_lda = inv_sqrt_psd(lw_lvl.covariance_)

    # 5 - Within-subjects noise: inverse-sqrt of two-way (level + scenario)
    # residual covariance — proper noise for the within-subjects design.
    resid2 = _two_way_residual(acts_f64, levels, trait)
    lw_within = LedoitWolf(assume_centered=True).fit(resid2)
    W_within = inv_sqrt_psd(lw_within.covariance_)

    spaces: WhitenMap = {
        "Euclidean (raw)": W_eucl,
        "Anisotropy-corrected": W_aniso,
        "Park causal": W_park,
        "Mahalanobis / LDA": W_lda,
        "Within-subjects noise": W_within,
    }
    diagnostics = {
        "D": D,
        "N": X_all.shape[0],
        "shrinkage_within_level": float(lw_lvl.shrinkage_),
        "shrinkage_within_scenario": float(lw_within.shrinkage_),
        "frobenius_norms": {n: float(np.linalg.norm(W)) for n, W in spaces.items()},
    }
    return spaces, diagnostics


def transform_acts(
    activations: dict[tuple, torch.Tensor], W: np.ndarray
) -> dict[tuple, torch.Tensor]:
    """Apply a whitening matrix to every activation vector."""
    return {k: torch.from_numpy(W @ _to_np(v)) for k, v in activations.items()}


def _cosine_matrix(vecs: list[np.ndarray]) -> np.ndarray:
    """All-pairs cosine over a list of vectors (clipped to [-1, 1])."""
    M = np.stack([v / np.linalg.norm(v) for v in vecs])
    return np.clip(M @ M.T, -1.0, 1.0)


def _diff_dict(
    activations: dict[tuple, torch.Tensor], trait: str, levels: list[str]
) -> dict[tuple, np.ndarray]:
    """``(trait, lo, hi) -> diff vector`` via the §3.1 paired-scenario estimator."""
    traits_cfg = [{"name": trait, "intensities": levels}]
    raw = compute_difference_vectors(activations, traits_cfg)
    return {k: _to_np(v) for k, v in raw.items()}


def _half_split_reliability(
    activations: dict[tuple, torch.Tensor],
    canon_keys: list[tuple],
    *,
    trait: str,
    levels: list[str],
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Spearman-Brown-corrected full-sample reliability of each step vector.

    Splits scenarios randomly in half, recomputes each difference vector
    independently on each half, and averages the cosine between the two
    same-vector estimates -> half-sample reliability r_half. Spearman-Brown
    rescales it to full-sample R = 2 * r_half / (1 + r_half).
    """
    rng = np.random.default_rng(seed)
    sids = sorted({k[2] for k in activations})
    acc = np.zeros(len(canon_keys))
    cnt = np.zeros(len(canon_keys))
    for _ in range(n_splits):
        rng.shuffle(sids)
        cut = len(sids) // 2
        a_sids, b_sids = set(sids[:cut]), set(sids[cut:])
        da = _diff_dict({k: v for k, v in activations.items() if k[2] in a_sids}, trait, levels)
        db = _diff_dict({k: v for k, v in activations.items() if k[2] in b_sids}, trait, levels)
        for i, key in enumerate(canon_keys):
            if key in da and key in db:
                x, y = da[key], db[key]
                acc[i] += float((x / np.linalg.norm(x)) @ (y / np.linalg.norm(y)))
                cnt[i] += 1
    r_half = acc / np.maximum(cnt, 1)
    R_full = 2.0 * r_half / (1.0 + r_half)
    return r_half, np.clip(R_full, 1e-3, 1.0)


def _disattenuate(cosmat: np.ndarray, R_full: np.ndarray) -> np.ndarray:
    """Divide each off-diagonal cosine by sqrt(R_i * R_j)."""
    R = np.sqrt(R_full)
    out = np.clip(cosmat / np.outer(R, R), -1.0, 1.0)
    np.fill_diagonal(out, 1.0)
    return out


def cosines_per_space(
    activations: dict[tuple[str, str, str], torch.Tensor],
    spaces: WhitenMap,
    *,
    trait: str,
    levels: list[str],
    n_rel_splits: int = 300,
    seed: int = 0,
) -> dict[SpaceName, dict]:
    """Raw and noise-disattenuated cosine matrices for each whitening space.

    For each space, applies the whitener to every activation, builds the
    step-vector dict, computes the all-pairs cosine matrix, estimates per-vector
    Spearman-Brown reliability, and returns the disattenuated cosine matrix.

    Returns
    -------
    dict[name, {labels, cos_raw, cos_dis, r_half, R_full}]
        ``labels`` are ``"hi-lo"`` strings ordered by the canonical step list
        (consecutive pairs first, then wider spans, matching
        :func:`lib.analysis.compute_difference_vectors`).
    """
    out: dict[SpaceName, dict] = {}
    canon_keys: list[tuple] | None = None
    labels: list[str] | None = None

    for name, W in spaces.items():
        acts_W = transform_acts(activations, W)
        dd = _diff_dict(acts_W, trait, levels)
        if canon_keys is None:
            canon_keys = list(dd.keys())
            labels = [f"{hi}-{lo}" for (_, lo, hi) in canon_keys]
        cos_raw = _cosine_matrix([dd[k] for k in canon_keys])
        r_half, R_full = _half_split_reliability(
            acts_W, canon_keys, trait=trait, levels=levels, n_splits=n_rel_splits, seed=seed
        )
        cos_dis = _disattenuate(cos_raw, R_full)
        out[name] = dict(
            labels=labels, cos_raw=cos_raw, cos_dis=cos_dis, r_half=r_half, R_full=R_full
        )
    return out
