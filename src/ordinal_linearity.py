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
       small-multiples grid of individual scenarios in the (politeness axis ×
       top orthogonal) plane.
    4. Linearity metrics — naive pooled and within-scenario estimators at the
       focal layer, with a within-scenario label-permutation null.
    5. Layer sweep — within-scenario metrics across every layer.

A small lexical baseline (bag-of-words logistic regression on the prompts)
is printed as a sanity check that the trait is not trivially decodable from
surface form.

Figures land in ``results/<dataset>/ordinal_linearity/<token_pooling>_token/``;
numeric summaries are printed.

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
from lib.inner_products import (
    SPACE_ORDER,
    build_inner_product_spaces,
    cosines_per_space,
)
from lib.representations import load_representations, pool_by_scenario_level
from lib.sentences import LEVELS, load_accepted


# --- config ----------------------------------------------------------------

LAYER = 13
TRAIT = "politeness"
TOKEN_POOLING = "avg"
DATASET = Path("data/20260530_001930/sentences/sentences_filtered.jsonl")
DATASET_ROOT = DATASET.parent.parent
REP_ROOT = DATASET_ROOT / "representations"
UNEMBED_COV_PATH = REP_ROOT / "unembeddings_covariance.pt"
REP_DIR = REP_ROOT / f"{TOKEN_POOLING}_token"
RESULTS_DIR = Path("results") / DATASET_ROOT.name / "ordinal_linearity" / f"{TOKEN_POOLING}_token"


def _configure(token_pooling: str) -> None:
    """Rebind the pooling-dependent globals so every step reads the matching
    representation directory and writes to the matching results folder."""
    global TOKEN_POOLING, REP_DIR, RESULTS_DIR
    TOKEN_POOLING = token_pooling
    REP_DIR = REP_ROOT / f"{token_pooling}_token"
    RESULTS_DIR = Path("results") / DATASET_ROOT.name / "ordinal_linearity" / f"{token_pooling}_token"

N_REL_SPLITS = 300       # scenario half-splits for the Spearman-Brown reliability
REL_SEED = 0
N_PERM = 100             # within-scenario label permutations for the null
PERM_SEED = 13
N_SCENARIO_PANELS = 12   # scenarios shown in the small-multiples grid
SCENARIO_PANEL_SEED = 0


# --- helpers ---------------------------------------------------------------

def _save(fig, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


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


# --- step 3: PCA views ------------------------------------------------------

def pca_views(activations_scen, levels, out_dir: Path) -> None:
    """Raw vs within-centered PCA, plus per-scenario small multiples in the
    (politeness axis × top orthogonal) plane."""
    triples = scenario_triples(activations_scen, levels, trait=TRAIT)
    sids = sorted(triples)
    mats = np.stack([triples[s] for s in sids])                  # (S, L, D)
    L = len(levels)

    # --- panel A: raw vs within-centered PCA on pooled (S*L) vectors --------
    raw_X = mats.reshape(len(sids) * L, -1)
    cen_X = (mats - mats.mean(axis=1, keepdims=True)).reshape(len(sids) * L, -1)
    all_lvl = [levels[i % L] for i in range(len(sids) * L)]

    panels = []
    for X, title in [(raw_X, "raw activations"), (cen_X, "within-scenario centered")]:
        pca = PCA(n_components=2)
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
    fig, _ = plot_pca_pair(
        panels,
        levels_present=levels,
        suptitle=f"{TRAIT} — PC1/PC2 of raw vs within-scenario centered activations",
    )
    _save(fig, out_dir, "pca_raw_vs_centered")

    # --- panel B: per-scenario grid in (politeness axis × top orthogonal) ---
    centered = mats - mats.mean(axis=1, keepdims=True)
    axis = (mats[:, -1] - mats[:, 0]).mean(0)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    flat = centered.reshape(-1, centered.shape[-1])
    flat_orth = flat - np.outer(flat @ axis, axis)
    orth = PCA(n_components=1).fit(flat_orth).components_[0]

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


# --- step 4: linearity metrics + permutation null --------------------------

def linearity_metrics(activations_scen, levels, out_dir: Path) -> None:
    """Print pooled + within-scenario metrics at the focal layer, then run the
    within-scenario label-permutation null and save the histogram panel."""
    print("\n" + "=" * 60)
    print(f"Linearity metrics at layer {LAYER}")
    naive = ordinal_linearity_metrics(
        acts_by_level_from_dict(activations_scen, levels), levels,
    )
    within = within_scenario_linearity_metrics(activations_scen, levels, trait=TRAIT)
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
        m = within_scenario_linearity_metrics(a_scen, levels, trait=TRAIT)
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


# --- step 6: lexical baseline ----------------------------------------------

def lexical_baseline() -> None:
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
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
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
    for idx, lvl in enumerate(clf.classes_):
        top = vocab[np.argsort(coefs[idx])[-10:][::-1]]
        print(f"  {lvl:9s}: {', '.join(top)}")


# --- entrypoint ------------------------------------------------------------

def main(token_pooling: str = TOKEN_POOLING) -> Path:
    _configure(token_pooling)
    apply_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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
    lexical_baseline()

    print(f"\nDone. Figures in {RESULTS_DIR}")
    return RESULTS_DIR


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ordinal linearity of trait intensity.")
    ap.add_argument(
        "--token-pooling", choices=("avg", "last"), default=TOKEN_POOLING,
        help="Prompt-token pooling whose representations to analyse (default: %(default)s).",
    )
    args = ap.parse_args()
    main(args.token_pooling)
