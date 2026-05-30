"""Geometry of the negative / neutral / positive trait triple.

The ordinal-linearity script (``ordinal_linearity.py``) establishes a tension:
the three intensity levels are strongly *ordered* along a single axis (high
Spearman / probe R²), yet the path that connects them is not straight — the
neutral centroid sits well off the negative→positive line (large midpoint
residual) and the two consecutive step vectors are anti-correlated (negative
cosine). This module quantifies that bend and tests competing geometric
explanations for it:

    * ``triple_geometry`` — scalar descriptors of one (neg, neut, pos) triple:
      span, midpoint residual, the apex angle subtended at neutral, the
      step-vector cosine, and the valence / markedness decomposition.
    * ``bootstrap_geometry`` — scenario-resampled CIs for those descriptors, so
      the bend can be reported with a magnitude *and* a variance.
    * ``subspace_spectrum`` — eigen-spectrum of the within-scenario centered
      triples: how many shared dimensions the trait actually occupies.
    * ``shared_plane`` — projection of every scenario into a single shared
      (valence, markedness) plane, with the fraction of within-scenario variance
      the plane captures. This is the superposition / two-axis hypothesis.
    * ``markedness_reliability`` — split-half reliability of the markedness axis
      (is the off-line displacement a shared direction or per-scenario noise?).
    * ``shared_bend_cv`` — cross-validated test of whether subtracting an
      out-of-fold *shared* bend vector removes the neutral-midpoint residual.
    * ``linear_null_simulation`` — could measurement noise on a genuinely
      straight ladder fake the observed bend? Re-injects the empirical
      paraphrase noise onto the linear prediction and re-measures.

All functions take the within-scenario triple store
``{scenario_id: ndarray (L, D)}`` produced by
:func:`lib.analysis.scenario_triples` (rows ordered negative → positive), except
:func:`linear_null_simulation`, which also needs the paraphrase-level vectors so
it can resample the real centroid noise.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Single-triple descriptors
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def triple_geometry(neg: np.ndarray, neut: np.ndarray, pos: np.ndarray) -> dict:
    """Scalar geometry of one negative / neutral / positive triple.

    Decomposes the neutral centroid relative to the negative→positive
    ("valence") axis into a parallel offset (where neutral falls *along* the
    axis) and a perpendicular displacement (the "markedness" — how far neutral
    sits *off* the axis). Both are reported as fractions of the span so they are
    comparable across layers and poolings.

    Returns a dict with:
        span                    ``||pos - neg||``
        midpoint_residual       ``||neut - mid|| / span`` (0 ⇔ neutral on the
                                line at the midpoint; the §3-style metric)
        valence_offset          signed parallel offset of neutral / span
                                (0 ⇔ neutral is centred between the poles)
        markedness              perpendicular displacement / span
        markedness_ratio        perpendicular displacement / (span / 2)
                                (the off-axis height in valence-half-span units)
        apex_angle_deg          angle at the neutral vertex between the rays to
                                the two poles (180° ⇔ collinear / straight ladder)
        effective_vertices      ``360 / apex_angle_deg`` — if the poles were two
                                adjacent vertices of a regular polygon centred on
                                neutral, this is that polygon's vertex count
        step_cosine             cos(neut - neg, pos - neut) (+1 ⇔ straight)
    """
    neg, neut, pos = map(np.asarray, (neg, neut, pos))
    span_vec = pos - neg
    span = float(np.linalg.norm(span_vec))
    u = _unit(span_vec)
    mid = 0.5 * (neg + pos)
    d = neut - mid
    valence_offset = float(d @ u)
    perp = float(np.linalg.norm(d - (d @ u) * u))

    r1, r2 = neg - neut, pos - neut
    apex_cos = float(np.clip(_unit(r1) @ _unit(r2), -1.0, 1.0))
    apex_deg = float(np.degrees(np.arccos(apex_cos)))

    s1, s2 = neut - neg, pos - neut
    step_cos = float(np.clip(_unit(s1) @ _unit(s2), -1.0, 1.0))

    return {
        "span": span,
        "midpoint_residual": float(np.linalg.norm(d) / span) if span > 0 else float("nan"),
        "valence_offset": valence_offset / span if span > 0 else float("nan"),
        "markedness": perp / span if span > 0 else float("nan"),
        "markedness_ratio": perp / (span / 2) if span > 0 else float("nan"),
        "apex_angle_deg": apex_deg,
        "effective_vertices": 360.0 / apex_deg if apex_deg > 0 else float("nan"),
        "step_cosine": step_cos,
    }


def pooled_centroids(triples: dict[str, np.ndarray]) -> np.ndarray:
    """Mean (neg, neut, pos) configuration across scenarios → ``(L, D)``."""
    return np.stack([triples[s] for s in sorted(triples)]).mean(axis=0)


def per_scenario_geometry(triples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run :func:`triple_geometry` inside every scenario separately.

    The pooled geometry describes the *average* triangle; this describes the
    *typical individual* triangle. Returns ``{descriptor: ndarray (S,)}`` so the
    runner can report medians and IQR bands and check that the bend is not just
    an averaging effect.
    """
    sids = sorted(triples)
    rows = [triple_geometry(*triples[s]) for s in sids]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def _centered(triples: dict[str, np.ndarray]) -> np.ndarray:
    """Within-scenario centered triples stacked → ``(S, L, D)``."""
    M = np.stack([triples[s] for s in sorted(triples)])
    return M - M.mean(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Bootstrap CIs over scenarios
# ---------------------------------------------------------------------------

_GEOM_KEYS = (
    "midpoint_residual", "valence_offset", "markedness", "markedness_ratio",
    "apex_angle_deg", "effective_vertices", "step_cosine",
)


def bootstrap_geometry(
    triples: dict[str, np.ndarray], *, n_boot: int = 2000, seed: int = 0,
) -> dict:
    """Scenario-resampled distribution of the pooled triple geometry.

    Resamples the scenarios with replacement ``n_boot`` times, recomputes the
    pooled centroids and their :func:`triple_geometry`, and returns the point
    estimate, bootstrap mean/SD and the 2.5 / 97.5 percentile CI for each
    descriptor — the variance / stability requested for the headline numbers.
    """
    sids = sorted(triples)
    mats = np.stack([triples[s] for s in sids])              # (S, L, D)
    S = len(sids)

    point = triple_geometry(*mats.mean(axis=0))
    rng = np.random.default_rng(seed)
    draws = {k: np.empty(n_boot) for k in _GEOM_KEYS}
    for b in range(n_boot):
        idx = rng.integers(0, S, size=S)
        g = triple_geometry(*mats[idx].mean(axis=0))
        for k in _GEOM_KEYS:
            draws[k][b] = g[k]

    out: dict[str, dict] = {}
    for k in _GEOM_KEYS:
        d = draws[k]
        out[k] = {
            "point": point[k],
            "mean": float(d.mean()),
            "sd": float(d.std(ddof=1)),
            "ci": (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
            "draws": d,
        }
    return out


# ---------------------------------------------------------------------------
# Dimensionality of the shared trait subspace
# ---------------------------------------------------------------------------

def subspace_spectrum(triples: dict[str, np.ndarray], *, k: int = 8) -> dict:
    """Eigen-spectrum of the stacked within-scenario centered triples.

    Each centered triple has rank ≤ 2, so any structure beyond two dimensions
    can only come from the per-scenario valence/markedness directions failing to
    align. A spectrum that collapses onto the first two components is therefore
    direct evidence that the trait occupies a *shared* 2-D plane.

    Returns the variance ratio of the top ``k`` components, their cumulative
    sum, and the participation ratio ``(Σλ)² / Σλ²`` (an effective dimension).
    """
    X = _centered(triples).reshape(-1, _centered(triples).shape[-1])
    sv = np.linalg.svd(X, compute_uv=False)
    lam = sv ** 2
    ratio = lam / lam.sum()
    pr = float((lam.sum() ** 2) / (lam ** 2).sum())
    k = min(k, len(ratio))
    return {
        "variance_ratio": ratio[:k],
        "cumulative": np.cumsum(ratio)[:k],
        "participation_ratio": pr,
    }


# ---------------------------------------------------------------------------
# Shared (valence, markedness) plane — the two-axis hypothesis
# ---------------------------------------------------------------------------

def shared_axes(triples: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The shared valence and markedness unit axes.

    valence  = mean over scenarios of (pos − neg), normalised.
    markedness = mean over scenarios of (neut − ½(neg+pos)), the component
                 orthogonal to valence, normalised. Positive markedness points
                 toward where neutral sits relative to the pole midpoint.
    """
    M = np.stack([triples[s] for s in sorted(triples)])      # (S, L, D)
    e_v = _unit(M[:, 2].mean(0) - M[:, 0].mean(0))
    raw_m = (M[:, 1] - 0.5 * (M[:, 0] + M[:, 2])).mean(0)
    e_m = _unit(raw_m - (raw_m @ e_v) * e_v)
    return e_v, e_m


def shared_plane(triples: dict[str, np.ndarray]) -> dict:
    """Project every scenario into the shared (valence, markedness) plane.

    Returns
    -------
    dict with:
        e_v, e_m            the shared unit axes (D,)
        coords              mean within-scenario coordinates of the three levels
                            in the plane → ``(L, 2)`` (the canonical triangle)
        coords_per_scenario ``(S, L, 2)`` per-scenario coordinates
        frac_valence_1d     fraction of within-scenario variance captured by the
                            valence axis alone
        frac_plane_2d       fraction captured by the (valence, markedness) plane
                            (per-scenario triples are rank ≤ 2, so the gap to 1.0
                            measures how *idiosyncratic* the residual plane is)
        neutral_markedness_sign_frac
                            fraction of scenarios whose neutral has positive
                            markedness coordinate (consistency of the offset)
    """
    e_v, e_m = shared_axes(triples)
    cen = _centered(triples)                                  # (S, L, D)
    basis = np.stack([e_v, e_m])                             # (2, D)
    coords_ps = cen @ basis.T                                 # (S, L, 2)

    total = float((cen ** 2).sum())
    var_v = float(((cen @ e_v) ** 2).sum())
    var_plane = float((coords_ps ** 2).sum())

    neut_m = coords_ps[:, 1, 1]                               # neutral markedness coord
    return {
        "e_v": e_v,
        "e_m": e_m,
        "coords": coords_ps.mean(axis=0),
        "coords_per_scenario": coords_ps,
        "frac_valence_1d": var_v / total,
        "frac_plane_2d": var_plane / total,
        "neutral_markedness_sign_frac": float((neut_m > 0).mean()),
    }


def markedness_reliability(
    triples: dict[str, np.ndarray], *, n_splits: int = 300, seed: int = 0,
) -> dict:
    """Split-half reliability of the markedness axis.

    Splits scenarios in half, estimates the markedness axis on each half (both
    orthogonalised against the *global* valence axis so only the off-line
    direction is tested), and averages the cosine between the two estimates.
    Spearman-Brown rescales the half-sample agreement to a full-sample
    reliability. A reliability near 1 means the off-line displacement is a single
    shared direction; near 0 means it is per-scenario noise.
    """
    sids = sorted(triples)
    M_all = np.stack([triples[s] for s in sids])
    e_v_global = _unit(M_all[:, 2].mean(0) - M_all[:, 0].mean(0))

    def _markedness(mats: np.ndarray) -> np.ndarray:
        raw = (mats[:, 1] - 0.5 * (mats[:, 0] + mats[:, 2])).mean(0)
        return _unit(raw - (raw @ e_v_global) * e_v_global)

    rng = np.random.default_rng(seed)
    cosines = np.empty(n_splits)
    for i in range(n_splits):
        order = rng.permutation(len(sids))
        cut = len(sids) // 2
        a = M_all[order[:cut]]
        b = M_all[order[cut:]]
        cosines[i] = float(_markedness(a) @ _markedness(b))
    r_half = float(cosines.mean())
    R_full = 2.0 * r_half / (1.0 + r_half) if r_half > -1 else float("nan")
    return {"r_half": r_half, "R_full": R_full, "cosines": cosines}


# ---------------------------------------------------------------------------
# Cross-validated model comparison: linear ladder vs shared bend
# ---------------------------------------------------------------------------

def shared_bend_cv(
    triples: dict[str, np.ndarray], *, n_splits: int = 10, seed: int = 0,
) -> dict:
    """Does an out-of-fold *shared* bend explain the neutral-midpoint residual?

    The linear model predicts ``neut = ½(neg + pos)``. The bend model adds a
    single shared correction vector learned from the training scenarios only:
    ``neut = ½(neg + pos) + bend̄``. K-fold over scenarios, comparing held-out
    residual energy:

        R²_cv = 1 − Σ‖neut − (mid + bend̄_train)‖² / Σ‖neut − mid‖²

    A clearly positive R²_cv means the deviation from linearity is *systematic*
    (a shared off-line offset, the markedness axis), not idiosyncratic noise —
    i.e. a 2-D model genuinely beats the 1-D line out of sample.
    """
    sids = np.array(sorted(triples))
    M = np.stack([triples[s] for s in sids])                 # (S, L, D)
    mid = 0.5 * (M[:, 0] + M[:, 2])
    bend = M[:, 1] - mid                                      # (S, D)
    ss_linear = float((bend ** 2).sum())

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(sids)), min(n_splits, len(sids)))
    ss_bend = 0.0
    better = 0
    for te in folds:
        tr = np.setdiff1d(np.arange(len(sids)), te)
        bend_bar = bend[tr].mean(axis=0)
        res = bend[te] - bend_bar
        ss_bend += float((res ** 2).sum())
        better += int(((res ** 2).sum(axis=1) < (bend[te] ** 2).sum(axis=1)).sum())
    return {
        "r2_cv": 1.0 - ss_bend / ss_linear if ss_linear > 0 else float("nan"),
        "frac_scenarios_improved": better / len(sids),
        "ss_linear": ss_linear,
        "ss_bend": ss_bend,
    }


# ---------------------------------------------------------------------------
# Common steering direction vs per-scenario plane tilt
#
# The two-axis plane is a property of the *average* triple, yet each scenario's
# own plane is tilted differently in the 2304-D space (the shared plane captures
# only ~20-26% of within-scenario variance). Activation steering assumes ONE
# global direction works everywhere, so the load-bearing question is *which*
# component is shared. These functions separate the on-line valence (steering)
# axis from the off-line bend and quantify how much of each is global vs
# scenario-specific.
# ---------------------------------------------------------------------------

def valence_reliability(
    triples: dict[str, np.ndarray], *, n_splits: int = 300, seed: int = 0,
) -> dict:
    """Split-half reliability of the valence (steering) axis ``mean(pos - neg)``.

    Sibling of :func:`markedness_reliability` for the *on-line* direction — the
    one an activation-steering vector would use. Splits scenarios in half,
    estimates the mean full-span axis on each half and averages the cosine
    between the two estimates; Spearman-Brown rescales the half-sample agreement
    to a full-sample reliability. R near 1 means a single global steering
    direction is well defined; a low R means the contrast direction is itself
    scenario-specific (steering would not transfer).
    """
    sids = sorted(triples)
    M_all = np.stack([triples[s] for s in sids])

    def _valence(mats: np.ndarray) -> np.ndarray:
        return _unit((mats[:, 2] - mats[:, 0]).mean(0))

    rng = np.random.default_rng(seed)
    cosines = np.empty(n_splits)
    for i in range(n_splits):
        order = rng.permutation(len(sids))
        cut = len(sids) // 2
        cosines[i] = float(_valence(M_all[order[:cut]]) @ _valence(M_all[order[cut:]]))
    r_half = float(cosines.mean())
    R_full = 2.0 * r_half / (1.0 + r_half) if r_half > -1 else float("nan")
    return {"r_half": r_half, "R_full": R_full, "cosines": cosines}


def steering_alignment(triples: dict[str, np.ndarray]) -> dict:
    """How well does a single global steering vector serve every scenario?

    Builds the global valence axis ``e_v = unit(mean(pos - neg))`` and the global
    markedness axis ``e_m`` (off-line), then per scenario measures:

      * ``cos_valence``    cosine of the scenario's own ``pos - neg`` with ``e_v``
                           — alignment of the local contrast with the shared
                           steering direction.
      * ``capture_valence`` ``cos_valence**2`` — the fraction of the scenario's
                           contrast *energy* a single global vector reproduces.
      * ``cos_bend``       cosine of the scenario's off-line bend
                           ``(neut - mid) ⟂ e_v`` with ``e_m`` — how aligned the
                           *bend* is across scenarios.

    A high mean ``cos_valence`` alongside a much lower / more variable
    ``cos_bend`` is the signature of a shared steering axis riding on a
    scenario-specific bend.
    """
    e_v, e_m = shared_axes(triples)
    sids = sorted(triples)
    M = np.stack([triples[s] for s in sids])                 # (S, L, D)

    contrast = M[:, 2] - M[:, 0]                             # (S, D)
    cn = np.linalg.norm(contrast, axis=1)
    cos_v = (contrast @ e_v) / np.where(cn > 0, cn, np.nan)

    bend = M[:, 1] - 0.5 * (M[:, 0] + M[:, 2])              # (S, D)
    bend_perp = bend - np.outer(bend @ e_v, e_v)
    bn = np.linalg.norm(bend_perp, axis=1)
    cos_b = (bend_perp @ e_m) / np.where(bn > 0, bn, np.nan)

    return {
        "cos_valence": cos_v,
        "capture_valence": cos_v ** 2,
        "cos_bend": cos_b,
        "mean_cos_valence": float(np.nanmean(cos_v)),
        "mean_capture_valence": float(np.nanmean(cos_v ** 2)),
        "mean_cos_bend": float(np.nanmean(cos_b)),
    }


def plane_principal_angles(triples: dict[str, np.ndarray]) -> dict:
    """Principal angles between each scenario's triple-plane and the shared plane.

    Each within-scenario centered triple spans a plane of rank ≤ 2. This returns
    the two principal angles between that plane and the shared
    ``span(e_v, e_m)`` plane. ``theta1`` (small) is how well the dominant shared
    direction lives inside the scenario plane; ``theta2`` is the *tilt* of the
    second direction. Small ``theta1`` with large ``theta2`` means scenarios
    share the steering axis but fan their bend out of the shared plane.

    Returns ``angles`` (S, 2) in degrees (ascending per scenario), the
    shared-plane ``overlap`` = ``mean(cos**2)`` of the two principal angles per
    scenario, and the medians.
    """
    e_v, e_m = shared_axes(triples)
    B = np.stack([e_v, e_m])                                 # (2, D), orthonormal
    sids = sorted(triples)
    cen = _centered(triples)                                  # (S, L, D)
    angles = np.empty((len(sids), 2))
    overlap = np.empty(len(sids))
    for i in range(len(sids)):
        _, _, Vt = np.linalg.svd(cen[i], full_matrices=False)
        Q = Vt[:2]                                           # (2, D) plane basis
        sv = np.clip(np.linalg.svd(B @ Q.T, compute_uv=False), 0.0, 1.0)
        angles[i] = np.sort(np.degrees(np.arccos(sv)))
        overlap[i] = float((sv ** 2).mean())
    return {
        "angles": angles,
        "overlap": overlap,
        "median_theta1": float(np.median(angles[:, 0])),
        "median_theta2": float(np.median(angles[:, 1])),
        "median_overlap": float(np.median(overlap)),
    }


def intent_contrast_structure(
    triples: dict[str, np.ndarray], intents: dict[str, str],
) -> dict:
    """Is the contrast-direction tilt organised by intent, or idiosyncratic?

    Compares the mean pairwise cosine between scenario contrast vectors
    ``pos - neg`` for pairs *within* the same intent vs *between* intents. A
    clear within > between gap means the steering axis tilts in an
    intent-structured way (a per-intent vector would steer better); no gap means
    the tilt is scenario-idiosyncratic noise around the shared axis.
    """
    sids = sorted(triples)
    M = np.stack([triples[s] for s in sids])
    u = M[:, 2] - M[:, 0]
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    G = u @ u.T
    lab = np.array([intents[s] for s in sids])
    same = lab[:, None] == lab[None, :]
    iu = np.triu_indices(len(sids), k=1)
    within = G[iu][same[iu]]
    between = G[iu][~same[iu]]
    return {
        "within_mean": float(within.mean()) if within.size else float("nan"),
        "between_mean": float(between.mean()) if between.size else float("nan"),
        "n_within": int(within.size),
        "n_between": int(between.size),
    }


# ---------------------------------------------------------------------------
# Could measurement noise fake the bend on a straight ladder?
# ---------------------------------------------------------------------------

def _paraphrase_groups(
    para_acts: dict[tuple, np.ndarray], levels: list[str], trait: str,
) -> dict[str, dict[str, np.ndarray]]:
    """``{scenario: {level: (n_para, D)}}`` from paraphrase-keyed activations."""
    out: dict[str, dict[str, list]] = {}
    for (t, lvl, sid, _pid), vec in para_acts.items():
        if t != trait or lvl not in levels:
            continue
        out.setdefault(sid, {}).setdefault(lvl, []).append(np.asarray(vec, dtype=np.float64))
    return {
        sid: {lvl: np.stack(v) for lvl, v in by_lvl.items()}
        for sid, by_lvl in out.items()
        if all(lvl in by_lvl for lvl in levels)
    }


def linear_null_simulation(
    para_acts: dict[tuple, np.ndarray],
    levels: list[str],
    trait: str,
    *,
    n_sim: int = 1000,
    seed: int = 0,
) -> dict:
    """Null geometry if the true ladder were straight, seen through real noise.

    Builds the linear null: each scenario's true negative and positive centroids
    are the observed ones, and the true neutral is forced to the *midpoint*.
    Every centroid is then re-measured with the empirical paraphrase noise —
    drawn by resampling that cell's paraphrases with replacement (a nonparametric
    bootstrap of the centroid) — and the scenarios themselves are resampled, so
    the null carries both measurement and sampling variance. For each simulation
    the pooled ``step_cosine`` and ``midpoint_residual`` are recomputed.

    Returns the observed values alongside the null distributions and one-sided
    Monte-Carlo p-values (step cosine: P(null ≤ obs); midpoint residual:
    P(null ≥ obs)). If a straight ladder plus noise cannot reach the observed
    bend, the non-linearity is not a measurement artefact.
    """
    groups = _paraphrase_groups(para_acts, levels, trait)
    sids = sorted(groups)
    lo, mid_lvl, hi = levels
    centroids = {
        sid: {lvl: g[lvl].mean(0) for lvl in levels} for sid, g in groups.items()
    }

    obs_mats = np.stack([
        np.stack([centroids[s][lo], centroids[s][mid_lvl], centroids[s][hi]])
        for s in sids
    ])
    obs = triple_geometry(*obs_mats.mean(0))

    def _boot_centroid(mat: np.ndarray, rng) -> np.ndarray:
        idx = rng.integers(0, mat.shape[0], size=mat.shape[0])
        return mat[idx].mean(0)

    rng = np.random.default_rng(seed)
    null_step = np.empty(n_sim)
    null_mid = np.empty(n_sim)
    S = len(sids)
    for b in range(n_sim):
        scen_idx = rng.integers(0, S, size=S)
        rows = []
        for j in scen_idx:
            sid = sids[j]
            g = groups[sid]
            neg = _boot_centroid(g[lo], rng)
            pos = _boot_centroid(g[hi], rng)
            # neutral noise injected onto the LINEAR prediction (the midpoint)
            neut_noise = _boot_centroid(g[mid_lvl], rng) - centroids[sid][mid_lvl]
            neut = 0.5 * (centroids[sid][lo] + centroids[sid][hi]) + neut_noise
            rows.append(np.stack([neg, neut, pos]))
        g_sim = triple_geometry(*np.stack(rows).mean(0))
        null_step[b] = g_sim["step_cosine"]
        null_mid[b] = g_sim["midpoint_residual"]

    return {
        "observed": {"step_cosine": obs["step_cosine"], "midpoint_residual": obs["midpoint_residual"]},
        "null": {"step_cosine": null_step, "midpoint_residual": null_mid},
        "p_values": {
            "step_cosine": float((1 + (null_step <= obs["step_cosine"]).sum()) / (1 + n_sim)),
            "midpoint_residual": float((1 + (null_mid >= obs["midpoint_residual"]).sum()) / (1 + n_sim)),
        },
        "n_sim": n_sim,
    }
