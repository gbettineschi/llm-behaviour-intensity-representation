"""Ordinal linearity of trait intensity in LLM activations.

Loads paraphrase-level activations produced by ``extract_representations.py``,
pools them to scenario centroids, and at a single focal layer asks whether the
three intensity levels (negative / neutral / positive) lie on a straight line
in activation space. The script bundles five complementary views:

    1. §3.2 step-vector geometry — cosine matrix of all step vectors at the
       focal layer, plus a per-pair layer sweep.
    2. §3.5 inner-product comparison — repeats the cosine geometry under five
       whitenings (Euclidean / anisotropy / Park causal / LDA / within-subjects
       noise) and adds Spearman-Brown noise-disattenuated cosines.
    3. PCA views — raw vs within-scenario-centered top-two components, plus a
       small-multiples grid of individual scenarios in the (trait axis ×
       top orthogonal) plane.
    4. Linearity metrics — naive pooled and within-scenario estimators at the
       focal layer, with a within-scenario label-permutation null.
    5. Layer sweep — within-scenario metrics across every layer.

A small lexical baseline (bag-of-words logistic regression on the prompts)
is printed as a sanity check that the trait is not trivially decodable from
surface form.

Figures land in
``results/<dataset>/ordinal_linearity/<model>/<trait>/<token_pooling>_token/seed_<k>/``;
numeric summaries are printed, and numeric exports land under ``numeric/``.
With more than one seed, a mean±std aggregate is written next to the seed dirs.

Run from the repo root:  uv run python src/ordinal_linearity.py
"""

from __future__ import annotations

import collections
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from lib.config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    DEFAULT_SEEDS,
    MODELS,
    child_seed,
    dataset_path,
    rep_dir,
    results_dir,
    run_metadata,
    seeds_base_dir,
    unembed_cov_path,
)
from lib.analysis import (
    acts_by_level_from_dict,
    compute_difference_vectors,
    ordinal_linearity_metrics,
    permutation_null_within_scenario,
    scenario_triples,
    similarity_matrix,
    within_scenario_linearity_metrics,
)
from lib.figures import (
    C_NEG,
    C_REF,
    apply_style,
    plot_agreement,
    plot_bar_pair,
    plot_cosine_grid,
    plot_metric_sweep_panels,
    plot_null_panels,
    plot_pair_sweep,
    plot_pca_pair,
    plot_scenario_grid,
)
from lib.geometry import steering_alignment
from lib.inner_products import (
    SPACE_ORDER,
    build_inner_product_spaces,
    cosines_per_space,
)
from lib.exports import (
    save_fig as _save,
    slug as _slug,
    write_csv as _write_csv,
    write_json as _write_json,
    write_square_matrix as _write_square_matrix,
    write_tex_tabular as _write_tex_tabular,
)
from lib.representations import load_representations, pool_by_scenario_level
from lib.sentences import LEVELS, load_accepted
from lib.traits import DEFAULT_TRAIT, TRAITS


# --- config ----------------------------------------------------------------

ANALYSIS = "ordinal_linearity"
MODEL = DEFAULT_MODEL
LAYER = MODELS[MODEL]["focal_layer"]
TRAIT = DEFAULT_TRAIT
TOKEN_POOLING = "avg"
SEED = 0
DATASET_ROOT = DEFAULT_DATA_ROOT
DATASET = dataset_path(DATASET_ROOT, TRAIT)
UNEMBED_COV_PATH = unembed_cov_path(DATASET_ROOT, MODEL)
REP_DIR = rep_dir(DATASET_ROOT, MODEL, TRAIT, TOKEN_POOLING)
RESULTS_DIR = results_dir(DATASET_ROOT.name, ANALYSIS, MODEL, TRAIT, TOKEN_POOLING, SEED)

N_REL_SPLITS = 300       # scenario half-splits for the Spearman-Brown reliability
N_PERM = 100             # within-scenario label permutations for the null
N_SCENARIO_PANELS = 12   # scenarios shown in the small-multiples grid
SCENARIO_PANEL_SEED = 0  # presentational only: keeps the same scenarios on the grid across seeds

# Per-component seeds derived from the master SEED (rebound by _configure).
REL_SEED = child_seed(SEED, "reliability")
PERM_SEED = child_seed(SEED, "permutation_null")
PROBE_SEED = child_seed(SEED, "probe_cv")
WITHIN_SEED = child_seed(SEED, "within_metrics")
LEXICAL_SEED = child_seed(SEED, "lexical_cv")


def _configure(model: str, trait: str, token_pooling: str, seed: int) -> None:
    """Rebind the model/trait/pooling/seed-dependent globals so every step reads
    the matching representation directory and writes to the matching results folder."""
    global MODEL, TRAIT, TOKEN_POOLING, SEED, DATASET, LAYER, REP_DIR, RESULTS_DIR, UNEMBED_COV_PATH
    global REL_SEED, PERM_SEED, PROBE_SEED, WITHIN_SEED, LEXICAL_SEED
    MODEL = model
    TRAIT = trait
    TOKEN_POOLING = token_pooling
    SEED = seed
    DATASET = dataset_path(DATASET_ROOT, trait)
    LAYER = MODELS[model]["focal_layer"]
    UNEMBED_COV_PATH = unembed_cov_path(DATASET_ROOT, model)
    REP_DIR = rep_dir(DATASET_ROOT, model, trait, token_pooling)
    RESULTS_DIR = results_dir(DATASET_ROOT.name, ANALYSIS, model, trait, token_pooling, seed)
    REL_SEED = child_seed(seed, "reliability")
    PERM_SEED = child_seed(seed, "permutation_null")
    PROBE_SEED = child_seed(seed, "probe_cv")
    WITHIN_SEED = child_seed(seed, "within_metrics")
    LEXICAL_SEED = child_seed(seed, "lexical_cv")


# --- helpers ---------------------------------------------------------------
def _present_levels(activations) -> list[str]:
    """Ordered subset of :data:`LEVELS` that is present in ``activations``."""
    have = {lvl for (_, lvl, _) in activations}
    return [lvl for lvl in LEVELS if lvl in have]


# --- step 1: difference-vector cosine geometry (§3.2) ----------------------

def _strip_trait(label: str) -> str:
    """``trait:hi-lo`` → ``hi-lo`` (the trait prefix is redundant for one-trait runs)."""
    return label.split(":", 1)[1] if ":" in label else label


def step_vector_geometry(activations_scen, levels, out_dir: Path) -> None:
    """Cosine matrix at the focal layer + per-pair cosine across layers."""
    traits_cfg = [{"name": TRAIT, "intensities": levels}]
    diffs = compute_difference_vectors(activations_scen, traits_cfg)
    raw_labels, M = similarity_matrix(diffs)
    labels = [_strip_trait(l) for l in raw_labels]
    width = max(12, max(len(l) for l in labels) + 2)

    print("\n" + "=" * 60)
    print(f"Step-vector cosine geometry (layer {LAYER})")
    header = f"{'':>{width}}" + "".join(f"{l:>{width}}" for l in labels)
    print(header)
    for i, li in enumerate(labels):
        print(f"{li:>{width}}" + "".join(f"{M[i, j]:>{width}.4f}" for j in range(len(labels))))

    _write_square_matrix(out_dir / "numeric" / "step_vectors" / "step_vectors_cosine_matrix", labels, M)

    fig, _ = plot_agreement(
        M, labels,
        f"Step-vector cosine — {TRAIT}, layer {LAYER}",
        figsize=(max(5, len(labels) * 1.4), max(4.5, len(labels) * 1.3)),
    )
    _save(fig, out_dir, "step_vectors_cosine_matrix")

    # Layer sweep: cosine between every pair of step vectors at every layer.
    acts_multi = load_representations(REP_DIR)
    sweep_layers = sorted({l for (*_, l) in acts_multi})
    pair_idx = [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    pair_labels = [f"{labels[i]}  vs  {labels[j]}" for i, j in pair_idx]
    pair_sims = [[] for _ in pair_idx]
    for L in sweep_layers:
        a_para = {(t, i, s, p): v for (t, i, s, p, l), v in acts_multi.items() if l == L}
        a_scen = pool_by_scenario_level(a_para)
        _, m_L = similarity_matrix(compute_difference_vectors(a_scen, traits_cfg))
        for k, (i, j) in enumerate(pair_idx):
            pair_sims[k].append(float(m_L[i, j]))

    pair_cols: list[str] = []
    pair_map: dict[str, str] = {}
    for k, lab in enumerate(pair_labels):
        col = _slug(lab) or f"pair_{k + 1}"
        if col in pair_map:
            col = f"pair_{k + 1}"
        pair_cols.append(col)
        pair_map[col] = lab
    sweep_rows = [
        [int(L)] + [pair_sims[k][i] for k in range(len(pair_cols))]
        for i, L in enumerate(sweep_layers)
    ]
    _write_csv(out_dir / "numeric" / "step_vectors" / "step_vectors_layer_sweep.csv", ["layer"] + pair_cols, sweep_rows)
    _write_json(out_dir / "numeric" / "step_vectors" / "step_vectors_layer_sweep_labels.json", pair_map)

    fig, _ = plot_pair_sweep(
        sweep_layers, pair_sims, pair_labels,
        focal_layer=LAYER,
        title=f"{TRAIT} — step-vector cosines across layers",
    )
    _save(fig, out_dir, "step_vectors_layer_sweep")


# --- step 2: inner-product comparison (§3.5) -------------------------------

def inner_product_comparison(activations_scen, levels, out_dir: Path) -> None:
    """Cosines and noise-disattenuated cosines in each whitening space, plus a
    focused bar chart on the adjacent-step (neg→neut vs neut→pos) alignment."""
    spaces, diag = build_inner_product_spaces(
        activations_scen,
        unembed_cov_path=UNEMBED_COV_PATH,
        levels=levels,
        trait=TRAIT,
    )
    print("\n" + "=" * 60)
    print(f"Inner-product spaces (hidden dim D = {diag['D']}, N = {diag['N']})")
    print(f"  Ledoit-Wolf shrinkage  within-level: {diag['shrinkage_within_level']:.4f}"
          f"   within-scenario: {diag['shrinkage_within_scenario']:.4f}")
    for name, fnorm in diag["frobenius_norms"].items():
        print(f"  {name:<24} Frobenius norm = {fnorm:.4g}")

    _write_json(out_dir / "numeric" / "inner_products" / "inner_products_diagnostics.json", diag)

    space_results = cosines_per_space(
        activations_scen, spaces,
        trait=TRAIT, levels=levels,
        n_rel_splits=N_REL_SPLITS, seed=REL_SEED,
    )
    order = list(SPACE_ORDER)
    labels = space_results[order[0]]["labels"]

    print(f"\nAdjacent-step alignment:  {labels[0]}  vs  {labels[1]}")
    hdr = (f"  {'space':<24}{'raw cos':>10}{'disatt cos':>13}"
           + "".join(f"{'R[' + str(i) + ']':>9}" for i in range(len(labels))))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in order:
        r = space_results[name]
        Rs = "".join(f"{r['R_full'][i]:>9.3f}" for i in range(len(labels)))
        print(f"  {name:<24}{r['cos_raw'][0, 1]:>10.4f}{r['cos_dis'][0, 1]:>13.4f}{Rs}")
    print("  R[i] = full-sample reliability of " + ", ".join(f"R[{i}]={labels[i]}" for i in range(len(labels))))

    # 2 × |spaces| grid: raw cosine on top, disattenuated on bottom.
    matrices = [
        [space_results[name]["cos_raw"] for name in order],
        [space_results[name]["cos_dis"] for name in order],
    ]
    fig, _ = plot_cosine_grid(
        matrices,
        col_titles=order,
        row_titles=["raw cosine", "disattenuated"],
        names=labels,
        suptitle=f"Step-vector cosine across inner products — {TRAIT}, layer {LAYER}",
    )
    _save(fig, out_dir, "inner_products_cosine_grid")

    # Focused bar chart: adjacent-step cosine raw vs disattenuated per space.
    raw_vals = [space_results[name]["cos_raw"][0, 1] for name in order]
    dis_vals = [space_results[name]["cos_dis"][0, 1] for name in order]
    fig, _ = plot_bar_pair(
        order, raw_vals, dis_vals,
        series_labels=("raw", "disattenuated"),
        ylabel=f"cosine:  {labels[0]}  vs  {labels[1]}",
        title=f"Adjacent-step alignment — is the {TRAIT} ladder one straight line?",
    )
    _save(fig, out_dir, "inner_products_adjacent_step")

    # Numeric dumps for LaTeX/pgfplots.
    spaces_map: dict[str, str] = {}
    adj_header = ["space_slug", "space", "adjacent_raw", "adjacent_dis"] + [f"R_full_{_slug(l)}" for l in labels]
    adj_rows: list[list[object]] = []
    for name in order:
        slug = _slug(name)
        spaces_map[slug] = name
        r = space_results[name]
        adj_rows.append(
            [slug, name, float(r["cos_raw"][0, 1]), float(r["cos_dis"][0, 1])] + [float(x) for x in r["R_full"]]
        )
        _write_square_matrix(out_dir / "numeric" / "inner_products" / "matrices" / f"inner_products_cos_raw_{slug}", labels, r["cos_raw"])
        _write_square_matrix(out_dir / "numeric" / "inner_products" / "matrices" / f"inner_products_cos_dis_{slug}", labels, r["cos_dis"])
        rel_rows = [[labels[i], float(r["r_half"][i]), float(r["R_full"][i])] for i in range(len(labels))]
        _write_csv(out_dir / "numeric" / "inner_products" / "reliability" / f"inner_products_reliability_{slug}.csv", ["step", "r_half", "R_full"], rel_rows)

    _write_json(out_dir / "numeric" / "inner_products" / "inner_products_spaces.json", spaces_map)
    _write_csv(out_dir / "numeric" / "inner_products" / "inner_products_adjacent_step.csv", adj_header, adj_rows)
    _write_tex_tabular(
        out_dir / "numeric" / "inner_products" / "inner_products_adjacent_step.tex",
        ["space", "adjacent_raw", "adjacent_dis"],
        [[r[1], f"{float(r[2]):.4f}", f"{float(r[3]):.4f}"] for r in adj_rows],
    )


# --- step 3: PCA views ------------------------------------------------------

def pca_views(activations_scen, levels, out_dir: Path) -> None:
    """Raw vs within-centered PCA, plus per-scenario small multiples in the
    (trait axis × top orthogonal) plane."""
    triples = scenario_triples(activations_scen, levels, trait=TRAIT)
    sids = sorted(triples)
    mats = np.stack([triples[s] for s in sids])                  # (S, L, D)
    L = len(levels)

    # --- panel A: raw vs within-centered PCA on pooled (S*L) vectors --------
    raw_X = mats.reshape(len(sids) * L, -1)
    cen_X = (mats - mats.mean(axis=1, keepdims=True)).reshape(len(sids) * L, -1)
    all_lvl = [levels[i % L] for i in range(len(sids) * L)]

    panels = []
    point_meta = [(sids[i // L], all_lvl[i]) for i in range(len(sids) * L)]
    explained: dict[str, tuple[float, float]] = {}
    points_rows: list[list[object]] = []
    centroid_rows: list[list[object]] = []
    for key, (X, title) in [("raw", (raw_X, "raw activations")), ("centered", (cen_X, "within-scenario centered"))]:
        # random_state pinned: svd_solver="auto" picks the randomized solver at
        # this dimensionality, which is otherwise nondeterministic across runs
        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(X)
        centroids = np.stack([
            coords[[i for i, l in enumerate(all_lvl) if l == lvl]].mean(0)
            for lvl in levels
        ])
        panels.append({
            "coords": coords,
            "labels": all_lvl,
            "centroids": centroids,
            "explained_var": tuple(pca.explained_variance_ratio_),
            "title": f"{title} — layer {LAYER}",
        })
        explained[key] = tuple(float(x) for x in pca.explained_variance_ratio_)
        for i, (sid, lvl) in enumerate(point_meta):
            points_rows.append([key, sid, lvl, float(coords[i, 0]), float(coords[i, 1])])
        for i, lvl in enumerate(levels):
            centroid_rows.append([key, lvl, float(centroids[i, 0]), float(centroids[i, 1])])
    fig, _ = plot_pca_pair(
        panels,
        levels_present=levels,
        suptitle=f"{TRAIT} — PC1/PC2 of raw vs within-scenario centered activations",
    )
    _save(fig, out_dir, "pca_raw_vs_centered")

    _write_csv(
        out_dir / "numeric" / "pca" / "pca_raw_vs_centered_points.csv",
        ["panel", "scenario_id", "level", "pc1", "pc2"],
        points_rows,
    )
    _write_csv(
        out_dir / "numeric" / "pca" / "pca_raw_vs_centered_centroids.csv",
        ["panel", "level", "pc1", "pc2"],
        centroid_rows,
    )
    _write_json(out_dir / "numeric" / "pca" / "pca_raw_vs_centered_explained_var.json", explained)

    # --- panel B: per-scenario grid in (trait axis × top orthogonal) ---
    centered = mats - mats.mean(axis=1, keepdims=True)
    axis = (mats[:, -1] - mats[:, 0]).mean(0)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    flat = centered.reshape(-1, centered.shape[-1])
    flat_orth = flat - np.outer(flat @ axis, axis)
    orth = PCA(n_components=1, random_state=0).fit(flat_orth).components_[0]

    rng = np.random.default_rng(SCENARIO_PANEL_SEED)
    sample = list(rng.choice(sids, size=min(N_SCENARIO_PANELS, len(sids)), replace=False))
    scenario_xy, scenario_titles = [], []
    for sid in sample:
        mat = triples[sid]
        cen_mat = mat - mat.mean(axis=0, keepdims=True)
        scenario_xy.append((cen_mat @ axis, cen_mat @ orth))
        scenario_titles.append(sid.split("-", 2)[-1])

    fig, _ = plot_scenario_grid(
        scenario_xy, scenario_titles, levels,
        xlabel="projection on mean within-scenario axis",
        ylabel="top orthogonal residual",
        suptitle=f"Per-scenario neg→neut→pos — layer {LAYER}",
    )
    _save(fig, out_dir, "per_scenario_grid")

    grid_rows: list[list[object]] = []
    for sid, title, (xs, ys) in zip(sample, scenario_titles, scenario_xy):
        for lvl, x, y in zip(levels, xs, ys):
            grid_rows.append([sid, title, lvl, float(x), float(y)])
    _write_csv(
        out_dir / "numeric" / "pca" / "per_scenario_grid.csv",
        ["scenario_id", "scenario_title", "level", "x", "y"],
        grid_rows,
    )


# --- step 4: linearity metrics + permutation null --------------------------

def linearity_metrics(activations_scen, levels, out_dir: Path) -> None:
    """Print pooled + within-scenario metrics at the focal layer, then run the
    within-scenario label-permutation null and save the histogram panel."""
    print("\n" + "=" * 60)
    print(f"Linearity metrics at layer {LAYER}")
    naive = ordinal_linearity_metrics(
        acts_by_level_from_dict(activations_scen, levels), levels, seed=PROBE_SEED,
    )
    within = within_scenario_linearity_metrics(activations_scen, levels, trait=TRAIT, seed=WITHIN_SEED)
    print(f"  {'metric':<26}{'pooled (naïve)':>18}{'within-scenario':>20}")
    print("  " + "-" * 64)
    rows = [
        ("spearman",                 naive["spearman"],          within["spearman"]),
        ("kendall",                  naive["kendall"],           within["kendall"]),
        ("probe_r2",                 naive["probe_r2"],          within["probe_r2"]),
        ("midpoint_residual",        naive["midpoint_residual"], within["midpoint_residual_median"]),
        ("pc1_frac",                 naive["pc1_frac"],          within["pc1_frac_centroids"]),
    ]
    for name, a, b in rows:
        print(f"  {name:<26}{a:>+18.4f}{b:>+20.4f}")
    print(f"  (within-scenario uses {within['n_scenarios']} complete scenarios; "
          "midpoint reported as per-scenario median)")

    print(f"\nLabel-permutation null (n={N_PERM}) — within-scenario:")
    res = permutation_null_within_scenario(
        activations_scen, levels, trait=TRAIT, n_perm=N_PERM, seed=PERM_SEED,
    )
    floor = 1 / (1 + res["n_perm"])
    print(f"  Monte-Carlo p = (1 + #beyond) / (1 + n);  floor = {floor:.4f}")
    for key, obs in res["observed"].items():
        if isinstance(obs, float) and np.isnan(obs):
            print(f"  {key:>26}: n/a")
            continue
        null = res["null"][key]
        p = res["p_values"][key]
        print(f"  {key:>26}: obs={obs:+.4f}  null μ={null.mean():+.4f} σ={null.std():.4f}  p={p:.4f}")

    fig, _ = plot_null_panels(
        list(res["observed"]), res["null"], res["observed"],
        suptitle=f"Within-scenario permutation null — {TRAIT}, layer {LAYER}",
    )
    _save(fig, out_dir, "linearity_permutation_null")

    # Numeric dumps for LaTeX tables / pgfplots.
    _write_csv(
        out_dir / "numeric" / "linearity" / "linearity_metrics.csv",
        ["metric", "pooled_naive", "within_scenario"],
        [[n, float(a), float(b)] for (n, a, b) in rows],
    )
    _write_tex_tabular(
        out_dir / "numeric" / "linearity" / "linearity_metrics.tex",
        ["metric", "pooled", "within"],
        [[n, f"{float(a):+.4f}", f"{float(b):+.4f}"] for (n, a, b) in rows],
    )

    triples = scenario_triples(activations_scen, levels, trait=TRAIT)
    sids = sorted(triples)
    midres = within.get("midpoint_residual_per_scenario")
    pc1 = within.get("pc1_frac_per_scenario")
    per_rows: list[list[object]] = []
    for i, sid in enumerate(sids):
        per_rows.append([sid,
                         float(midres[i]) if isinstance(midres, np.ndarray) else None,
                         float(pc1[i]) if isinstance(pc1, np.ndarray) else None])
    _write_csv(
        out_dir / "numeric" / "linearity" / "linearity_within_per_scenario.csv",
        ["scenario_id", "midpoint_residual", "pc1_frac"],
        per_rows,
    )

    null_keys = list(res["null"].keys())
    null_rows = [
        [i] + [float(res["null"][k][i]) for k in null_keys]
        for i in range(res["n_perm"])
    ]
    _write_csv(out_dir / "numeric" / "linearity" / "permutation_null" / "linearity_permutation_null.csv", ["perm"] + null_keys, null_rows)
    summ_rows: list[list[object]] = []
    for k in null_keys:
        null = res["null"][k]
        obs = res["observed"][k]
        summ_rows.append([k, obs, float(null.mean()), float(null.std(ddof=1)), res["p_values"][k]])
    _write_csv(
        out_dir / "numeric" / "linearity" / "permutation_null" / "linearity_permutation_null_summary.csv",
        ["metric", "observed", "null_mean", "null_sd", "p_value"],
        summ_rows,
    )
    _write_tex_tabular(
        out_dir / "numeric" / "linearity" / "permutation_null" / "linearity_permutation_null_summary.tex",
        ["metric", "observed", "null_mean", "null_sd", "p"],
        [[r[0], f"{float(r[1]):+.4f}" if r[1] is not None else "", f"{float(r[2]):+.4f}", f"{float(r[3]):.4f}", f"{float(r[4]):.4f}"] for r in summ_rows],
    )
    _write_json(
        out_dir / "numeric" / "linearity" / "permutation_null" / "linearity_permutation_null_meta.json",
        {"n_perm": res["n_perm"], "seed": PERM_SEED, "p_floor": float(1 / (1 + res["n_perm"]))},
    )


# --- step 5: within-scenario metrics across layers -------------------------

def linearity_layer_sweep(out_dir: Path) -> None:
    """Six within-scenario metrics vs layer, two of them with per-scenario IQR bands."""
    acts_multi = load_representations(REP_DIR)
    sweep_layers = sorted({l for (*_, l) in acts_multi})
    a_para0 = {(t, i, s, p): v for (t, i, s, p, l), v in acts_multi.items() if l == sweep_layers[0]}
    levels = _present_levels(pool_by_scenario_level(a_para0))

    keys = ("spearman", "kendall", "probe_r2", "monotone_fraction",
            "midpoint_residual_median", "pc1_frac_per_scenario_mean")
    sweep = {k: [] for k in keys}
    midres_band, pc1_band = [], []
    for L in sweep_layers:
        a_para = {(t, i, s, p): v for (t, i, s, p, l), v in acts_multi.items() if l == L}
        a_scen = pool_by_scenario_level(a_para)
        m = within_scenario_linearity_metrics(a_scen, levels, trait=TRAIT, seed=WITHIN_SEED)
        for k in keys:
            sweep[k].append(m[k])
        mr = m["midpoint_residual_per_scenario"]
        pc = m["pc1_frac_per_scenario"]
        midres_band.append((np.nanpercentile(mr, 25), np.nanpercentile(mr, 75)))
        pc1_band.append((np.percentile(pc, 25), np.percentile(pc, 75)))
    midres_band = np.array(midres_band)
    pc1_band = np.array(pc1_band)

    panels = [
        {"values": sweep["spearman"],                   "title": "Spearman ρ (proj vs rank)",          "ylim": (-0.1, 1.0)},
        {"values": sweep["kendall"],                    "title": "Kendall τ (proj vs rank)",           "ylim": (-0.1, 1.0)},
        {"values": sweep["probe_r2"],                   "title": "Ridge probe R² (CV)",                 "ylim": (-0.1, 1.0)},
        {"values": sweep["monotone_fraction"],          "title": "Monotone fraction (per scenario)",   "ylim": (-0.05, 1.05)},
        {"values": sweep["midpoint_residual_median"],   "title": "Midpoint residual (lower = linear)", "band": (midres_band[:, 0], midres_band[:, 1])},
        {"values": sweep["pc1_frac_per_scenario_mean"], "title": "PC1 variance fraction (per scenario)", "ylim": (0.3, 1.05),
         "band": (pc1_band[:, 0], pc1_band[:, 1])},
    ]
    fig, _ = plot_metric_sweep_panels(
        sweep_layers, panels, focal_layer=LAYER,
        suptitle=f"{TRAIT} — within-scenario ordinal linearity across layers",
    )
    _save(fig, out_dir, "linearity_metric_sweep")

    sweep_rows: list[list[object]] = []
    for i, L in enumerate(sweep_layers):
        sweep_rows.append([
            int(L),
            float(sweep["spearman"][i]),
            float(sweep["kendall"][i]),
            float(sweep["probe_r2"][i]),
            float(sweep["monotone_fraction"][i]),
            float(sweep["midpoint_residual_median"][i]),
            float(midres_band[i, 0]),
            float(midres_band[i, 1]),
            float(sweep["pc1_frac_per_scenario_mean"][i]),
            float(pc1_band[i, 0]),
            float(pc1_band[i, 1]),
        ])
    _write_csv(
        out_dir / "numeric" / "linearity" / "layer_sweep" / "linearity_metric_sweep.csv",
        [
            "layer",
            "spearman",
            "kendall",
            "probe_r2",
            "monotone_fraction",
            "midpoint_residual_median",
            "midpoint_residual_iqr25",
            "midpoint_residual_iqr75",
            "pc1_frac_mean",
            "pc1_frac_iqr25",
            "pc1_frac_iqr75",
        ],
        sweep_rows,
    )


# --- step 6: lexical baseline ----------------------------------------------

def lexical_baseline(out_dir: Path) -> None:
    """BoW logistic-regression baseline + top n-grams per level. Sanity check
    that the trait is not trivially decodable from surface form."""
    samples = load_accepted(DATASET)
    texts = [s.prompt for s in samples]
    targets = [s.intensity for s in samples]
    n_classes = len(set(targets))

    vec = CountVectorizer(ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000)
    n_splits = min(5, min(collections.Counter(targets).values()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=LEXICAL_SEED)
    preds = cross_val_predict(clf, X, targets, cv=cv)
    acc = accuracy_score(targets, preds)
    print("\n" + "=" * 60)
    print(f"Lexical baseline (bag-of-words, 1-2 grams)")
    print(f"  CV accuracy: {acc:.3f}   (chance = {1 / n_classes:.3f},  n={len(texts)})")

    clf.fit(X, targets)
    vocab = np.array(vec.get_feature_names_out())
    coefs = clf.coef_
    if coefs.shape[0] == 1:
        coefs = np.vstack([-coefs[0], coefs[0]])
    top_rows: list[list[object]] = []
    for idx, lvl in enumerate(clf.classes_):
        order = np.argsort(coefs[idx])[-10:][::-1]
        top = vocab[order]
        print(f"  {lvl:9s}: {', '.join(top)}")
        for rank, j in enumerate(order, start=1):
            top_rows.append([str(lvl), int(rank), str(vocab[j]), float(coefs[idx][j])])

    _write_json(
        out_dir / "numeric" / "lexical" / "lexical_baseline.json",
        {
            "cv_accuracy": float(acc),
            "chance": float(1 / n_classes),
            "n": int(len(texts)),
            "n_splits": int(n_splits),
            "classes": [str(c) for c in clf.classes_],
            "vectorizer": {"ngram_range": [1, 2], "min_df": 2},
        },
    )
    _write_csv(
        out_dir / "numeric" / "lexical" / "lexical_baseline_top_ngrams.csv",
        ["level", "rank", "ngram", "coef"],
        top_rows,
    )


# --- step 7: steering-axis capture across layers ---------------------------

def steering_axis_sweep(out_dir: Path) -> None:
    """Fraction of each scenario's neg→pos contrast captured by a *single* global
    steering axis, swept across layers.

    Steering presumes one shared direction transfers across contexts. The
    capture (cos² of a scenario's own contrast with the global axis) measures
    how much of each scenario's effect that single vector reproduces, and the
    per-scenario IQR shows how uniform that is. The sweep locates the depths at
    which one steering direction is most valid."""
    acts_multi = load_representations(REP_DIR)
    sweep_layers = sorted({l for (*_, l) in acts_multi})
    a_para0 = {(t, i, s, p): v for (t, i, s, p, l), v in acts_multi.items() if l == sweep_layers[0]}
    levels = _present_levels(pool_by_scenario_level(a_para0))

    mean_cap, mean_cos, band = [], [], []
    for L in sweep_layers:
        a_para = {(t, i, s, p): v for (t, i, s, p, l), v in acts_multi.items() if l == L}
        triples = scenario_triples(pool_by_scenario_level(a_para), levels, trait=TRAIT)
        cap = steering_alignment(triples)["capture_valence"]
        mean_cap.append(float(np.nanmean(cap)))
        mean_cos.append(float(np.sqrt(np.nanmean(cap))))
        band.append((np.nanpercentile(cap, 25), np.nanpercentile(cap, 75)))
    band = np.array(band)

    foc = sweep_layers.index(LAYER)
    print("\n" + "=" * 60)
    print("Steering-axis capture across layers")
    print(f"  layer {LAYER}: mean capture (cos²) = {mean_cap[foc]:.3f}"
          f"   IQR = [{band[foc, 0]:.3f}, {band[foc, 1]:.3f}]")
    print("  (capture = fraction of each scenario's neg→pos contrast that a single"
          " global steering vector reproduces)")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.fill_between(sweep_layers, band[:, 0], band[:, 1], color=C_REF, alpha=0.2,
                    label="per-scenario IQR")
    ax.plot(sweep_layers, mean_cap, "o-", color=C_REF, label="mean capture (cos²)")
    ax.plot(sweep_layers, mean_cos, "s--", color=C_NEG, lw=1.2, label="mean cos")
    ax.axvline(LAYER, color="gray", ls=":", lw=1.0, label=f"focal layer {LAYER}")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("layer")
    ax.set_ylabel("global steering-axis capture")
    ax.set_title(f"{TRAIT} — does one steering direction suffice, by layer?")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    _save(fig, out_dir, "steering_axis_sweep")

    _write_csv(
        out_dir / "numeric" / "steering" / "steering_axis_sweep.csv",
        ["layer", "mean_capture", "mean_cos", "capture_iqr25", "capture_iqr75"],
        [
            [int(L), float(mean_cap[i]), float(mean_cos[i]), float(band[i, 0]), float(band[i, 1])]
            for i, L in enumerate(sweep_layers)
        ],
    )


# --- entrypoint ------------------------------------------------------------

def main(model: str = MODEL, trait: str = TRAIT, token_pooling: str = TOKEN_POOLING, seed: int = SEED) -> Path:
    _configure(model, trait, token_pooling, seed)
    apply_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(
        RESULTS_DIR / "run_metadata.json",
        run_metadata(model=MODEL, trait=TRAIT, seed=SEED, token_pooling=TOKEN_POOLING, dataset=DATASET, focal_layer=LAYER),
    )
    print(f"Saving figures under {RESULTS_DIR}\n")

    activations_para = load_representations(REP_DIR, layer=LAYER)
    activations_scen = pool_by_scenario_level(activations_para)
    levels = _present_levels(activations_scen)
    n_scen = len({s for (_, _, s) in activations_scen})
    print(f"Loaded {len(activations_para)} paraphrase vectors → {len(activations_scen)} "
          f"scenario centroids  |  layer {LAYER}  |  token_pooling={TOKEN_POOLING!r}  "
          f"|  levels {levels}  |  {n_scen} scenarios")

    step_vector_geometry(activations_scen, levels, RESULTS_DIR)
    inner_product_comparison(activations_scen, levels, RESULTS_DIR)
    pca_views(activations_scen, levels, RESULTS_DIR)
    linearity_metrics(activations_scen, levels, RESULTS_DIR)
    linearity_layer_sweep(RESULTS_DIR)
    steering_axis_sweep(RESULTS_DIR)
    lexical_baseline(RESULTS_DIR)

    print(f"\nDone. Figures in {RESULTS_DIR}")
    return RESULTS_DIR


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ordinal linearity of trait intensity.")
    ap.add_argument(
        "--model", choices=sorted(MODELS), default=DEFAULT_MODEL,
        help="Model whose representations to analyse (default: %(default)s).",
    )
    ap.add_argument(
        "--trait", choices=sorted(TRAITS), default=DEFAULT_TRAIT,
        help="Trait whose dataset/representations to analyse (default: %(default)s).",
    )
    ap.add_argument(
        "--token-pooling", choices=("avg", "last"), default=TOKEN_POOLING,
        help="Prompt-token pooling whose representations to analyse (default: %(default)s).",
    )
    ap.add_argument(
        "--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated master seeds; one full run per seed (default: %(default)s).",
    )
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    for s in seeds:
        main(args.model, args.trait, args.token_pooling, s)
    if len(seeds) > 1:
        from aggregate_results import aggregate_analysis

        aggregate_analysis(
            seeds_base_dir(DATASET_ROOT.name, ANALYSIS, args.model, args.trait, args.token_pooling),
            seeds=seeds,
        )
