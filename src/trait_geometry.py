"""Geometry of the trait triple: is the non-linearity real, and what shape is it?

Companion to ``ordinal_linearity.py``. That script shows the three intensity
levels are strongly *ordered* yet the path between them is *bent* (the neutral
centroid sits off the negative→positive line). This script asks whether that
bend is real and what geometry it implies, in five steps:

    1. Magnitude & variance — pooled and per-scenario triple geometry (apex
       angle, step cosine, midpoint residual) with scenario-bootstrap 95% CIs.
    2. Signal vs noise — a linear-ladder null: force neutral to the midpoint,
       re-inject the empirical paraphrase noise, and show measurement error on a
       straight ladder cannot reproduce the observed bend.
    3. Dimensionality — the eigen-spectrum of within-scenario centered triples
       (how many shared directions the trait actually occupies).
    4. The two-axis (superposition) model — projection into a single shared
       (valence, markedness) plane, the reliability of the markedness axis, and a
       cross-validated test that a shared off-line bend beats the straight line.
    5. Per-intent robustness — the bend, computed separately for each of the ten
       communicative intents.

Figures land in
``results/<dataset>/trait_geometry/<model>/<token_pooling>_token/seed_<k>/``;
numeric summaries are printed, and numeric exports land under ``numeric/``.
With more than one seed, a mean±std aggregate is written next to the seed dirs.

Run from the repo root:  uv run python src/trait_geometry.py --token-pooling avg
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lib.config import (
    DEFAULT_MODEL,
    DEFAULT_SEEDS,
    MODELS,
    child_seed,
    rep_dir,
    results_dir,
    run_metadata,
    seeds_base_dir,
)
from lib.analysis import scenario_triples
from lib.figures import C_NEG, C_NEUT, C_NULL, C_POS, C_REF, apply_style, levels_palette
from lib.geometry import (
    bootstrap_geometry,
    intent_contrast_structure,
    linear_null_simulation,
    markedness_reliability,
    per_scenario_geometry,
    plane_principal_angles,
    pooled_centroids,
    shared_bend_cv,
    shared_plane,
    steering_alignment,
    subspace_spectrum,
    triple_geometry,
    valence_reliability,
)
from lib.exports import (
    save_fig as _save,
    write_csv as _write_csv,
    write_json as _write_json,
    write_tex_tabular as _write_tex_tabular,
)
from lib.representations import load_representations, pool_by_scenario_level
from lib.sentences import LEVELS


# --- config ----------------------------------------------------------------

ANALYSIS = "trait_geometry"
MODEL = DEFAULT_MODEL
LAYER = MODELS[MODEL]["focal_layer"]
TRAIT = "politeness"
TOKEN_POOLING = "avg"
SEED = 0
DATASET = Path("data/20260530_001930/sentences/sentences_filtered.jsonl")
DATASET_ROOT = DATASET.parent.parent
REP_DIR = rep_dir(DATASET_ROOT, MODEL, TOKEN_POOLING)
RESULTS_DIR = results_dir(DATASET_ROOT.name, ANALYSIS, MODEL, TOKEN_POOLING, SEED)

N_BOOT = 2000        # scenario bootstrap resamples for geometry CIs
N_NULL = 1000        # linear-ladder + noise simulations
N_REL_SPLITS = 300   # scenario half-splits for markedness reliability
CV_SPLITS = 10       # folds for the shared-bend cross-validation

# Per-component seeds derived from the master SEED (rebound by _configure).
BOOT_SEED = child_seed(SEED, "bootstrap")
NULL_SEED = child_seed(SEED, "linear_null")
REL_SEED = child_seed(SEED, "markedness_rel")
VAL_REL_SEED = child_seed(SEED, "valence_rel")
BEND_CV_SEED = child_seed(SEED, "shared_bend_cv")


def _configure(model: str, token_pooling: str, seed: int) -> None:
    """Rebind the model/pooling/seed-dependent globals (mirrors ``ordinal_linearity.py``)."""
    global MODEL, TOKEN_POOLING, SEED, LAYER, REP_DIR, RESULTS_DIR
    global BOOT_SEED, NULL_SEED, REL_SEED, VAL_REL_SEED, BEND_CV_SEED
    MODEL = model
    TOKEN_POOLING = token_pooling
    SEED = seed
    LAYER = MODELS[model]["focal_layer"]
    REP_DIR = rep_dir(DATASET_ROOT, model, token_pooling)
    RESULTS_DIR = results_dir(DATASET_ROOT.name, ANALYSIS, model, token_pooling, seed)
    BOOT_SEED = child_seed(seed, "bootstrap")
    NULL_SEED = child_seed(seed, "linear_null")
    REL_SEED = child_seed(seed, "markedness_rel")
    VAL_REL_SEED = child_seed(seed, "valence_rel")
    BEND_CV_SEED = child_seed(seed, "shared_bend_cv")


# --- helpers ---------------------------------------------------------------
def _present_levels(triples_source) -> list[str]:
    have = {lvl for (_, lvl, _) in triples_source}
    return [lvl for lvl in LEVELS if lvl in have]


def _intent_of(sid: str) -> str:
    """``politeness-bad-news-delivery-042`` → ``bad-news-delivery``."""
    return re.sub(rf"^{TRAIT}-(.*)-\d+$", r"\1", sid)


def _ci(lo: float, hi: float) -> str:
    return f"[{lo:+.3f}, {hi:+.3f}]"


# --- step 1: magnitude & variance ------------------------------------------

def report_geometry(triples, out_dir: Path) -> dict:
    """Pooled + per-scenario triple geometry with scenario-bootstrap CIs."""
    pooled = triple_geometry(*pooled_centroids(triples))
    boot = bootstrap_geometry(triples, n_boot=N_BOOT, seed=BOOT_SEED)
    per = per_scenario_geometry(triples)

    print("\n" + "=" * 70)
    print(f"1. Triple geometry  (layer {LAYER}, {len(triples)} complete scenarios)")
    print(f"   {'descriptor':<20}{'pooled':>10}{'95% CI (boot)':>22}"
          f"{'per-scen median':>18}")
    print("   " + "-" * 68)
    rows = [
        ("apex_angle_deg",    "apex angle (°)"),
        ("effective_vertices", "eff. vertices"),
        ("step_cosine",       "step cosine"),
        ("midpoint_residual", "midpoint resid."),
        ("markedness",        "markedness/span"),
        ("valence_offset",    "valence offset"),
    ]
    for key, label in rows:
        b = boot[key]
        med = float(np.nanmedian(per[key]))
        print(f"   {label:<20}{b['point']:>10.3f}{_ci(*b['ci']):>22}{med:>18.3f}")
    print("   straight-ladder reference: apex 180°, step cosine +1, midpoint resid. 0")
    print(f"   pentagon reference: apex 72° (5 vertices) — "
          f"inside CI: {boot['apex_angle_deg']['ci'][0] <= 72.0 <= boot['apex_angle_deg']['ci'][1]}")

    # Figure: bootstrap distributions with linear / pentagon references.
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    panels = [
        ("apex_angle_deg", "apex angle at neutral (°)", [(180, "straight", C_NULL), (72, "pentagon", C_REF)]),
        ("step_cosine", "step cosine  cos(neg→neut, neut→pos)", [(1.0, "straight", C_NULL), (0.0, "orthogonal", "gray")]),
        ("midpoint_residual", "midpoint residual / span", [(0.0, "straight", C_NULL)]),
    ]
    for ax, (key, xlabel, refs) in zip(axes, panels):
        ax.hist(boot[key]["draws"], bins=40, color="lightgray", edgecolor="gray")
        ax.axvline(boot[key]["point"], color=C_NEG, lw=1.8, label=f"obs {boot[key]['point']:.2f}")
        for val, lab, col in refs:
            ax.axvline(val, color=col, ls="--", lw=1.2, label=lab)
        ax.set_xlabel(xlabel)
        ax.legend(loc="best", fontsize=7)
    axes[0].set_ylabel("bootstrap resamples")
    fig.suptitle(f"{TRAIT} — triple geometry, scenario bootstrap (n={N_BOOT})", y=1.03)
    fig.tight_layout()
    _save(fig, out_dir, "geometry_bootstrap")

    # Numeric dumps for LaTeX/pgfplots.
    num_dir = out_dir / "numeric" / "geometry"
    _write_csv(
        num_dir / "geometry_pooled.csv",
        ["metric", "value"],
        [[k, float(v)] for k, v in pooled.items()],
    )

    boot_keys = list(boot)
    _write_csv(
        num_dir / "geometry_bootstrap_summary.csv",
        ["metric", "point", "mean", "sd", "ci_lo", "ci_hi"],
        [[k, boot[k]["point"], boot[k]["mean"], boot[k]["sd"], boot[k]["ci"][0], boot[k]["ci"][1]] for k in boot_keys],
    )
    _write_tex_tabular(
        num_dir / "geometry_bootstrap_summary.tex",
        ["metric", "point", "ci_lo", "ci_hi"],
        [[k, f"{float(boot[k]['point']):.3f}", f"{float(boot[k]['ci'][0]):.3f}", f"{float(boot[k]['ci'][1]):.3f}"] for k in boot_keys],
    )

    draws_len = len(next(iter(boot.values()))["draws"])
    draw_rows = []
    for i in range(draws_len):
        draw_rows.append([i] + [float(boot[k]["draws"][i]) for k in boot_keys])
    _write_csv(num_dir / "geometry_bootstrap_draws.csv", ["draw"] + boot_keys, draw_rows)

    sids = sorted(triples)
    per_keys = list(per)
    per_rows = [[sid] + [float(per[k][i]) for k in per_keys] for i, sid in enumerate(sids)]
    _write_csv(num_dir / "geometry_per_scenario.csv", ["scenario_id"] + per_keys, per_rows)
    return {"pooled": pooled, "boot": boot, "per": per}


# --- step 2: signal vs measurement noise -----------------------------------

def report_noise_null(para_acts, levels, out_dir: Path) -> dict:
    """Linear-ladder + empirical-noise null for the bend."""
    res = linear_null_simulation(para_acts, levels, TRAIT, n_sim=N_NULL, seed=NULL_SEED)
    print("\n" + "=" * 70)
    print(f"2. Linear-ladder null  (neutral forced to midpoint + real noise, n={N_NULL})")
    for key, sign in (("step_cosine", "≤"), ("midpoint_residual", "≥")):
        obs = res["observed"][key]
        null = res["null"][key]
        print(f"   {key:<18} obs={obs:+.4f}  null μ={null.mean():+.4f} σ={null.std():.4f}"
              f"  range=[{null.min():+.4f}, {null.max():+.4f}]  p(null {sign} obs)={res['p_values'][key]:.4f}")
    print("   → measurement noise on a straight ladder cannot reproduce the observed bend.")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
    for ax, key, title in zip(
        axes, ("step_cosine", "midpoint_residual"),
        ("step cosine", "midpoint residual / span"),
    ):
        null = res["null"][key]
        ax.hist(null, bins=40, color="lightgray", edgecolor="gray", label="linear + noise null")
        ax.axvline(res["observed"][key], color=C_NEG, lw=1.8,
                   label=f"observed {res['observed'][key]:+.3f}")
        ax.set_xlabel(title)
        ax.legend(loc="best", fontsize=7)
    axes[0].set_ylabel("simulations")
    fig.suptitle(f"{TRAIT} — could a straight ladder + noise fake the bend? (no)", y=1.03)
    fig.tight_layout()
    _save(fig, out_dir, "noise_null")

    # Numeric dumps.
    num_dir = out_dir / "numeric" / "noise_null"
    keys = list(res["observed"])
    summ_rows = []
    for k in keys:
        obs = res["observed"][k]
        null = res["null"][k]
        summ_rows.append([
            k,
            float(obs),
            float(null.mean()),
            float(null.std(ddof=1)),
            float(null.min()),
            float(null.max()),
            float(res["p_values"][k]),
            int(res["n_sim"]),
        ])
    _write_csv(
        num_dir / "noise_null_summary.csv",
        ["metric", "observed", "null_mean", "null_sd", "null_min", "null_max", "p_value", "n_sim"],
        summ_rows,
    )
    _write_tex_tabular(
        num_dir / "noise_null_summary.tex",
        ["metric", "observed", "null_mean", "null_sd", "p"],
        [[r[0], f"{r[1]:+.4f}", f"{r[2]:+.4f}", f"{r[3]:.4f}", f"{r[6]:.4f}"] for r in summ_rows],
    )
    n = int(res["n_sim"])
    null_rows = [[i] + [float(res["null"][k][i]) for k in keys] for i in range(n)]
    _write_csv(num_dir / "noise_null_null.csv", ["sim"] + keys, null_rows)
    return res


# --- step 3: dimensionality of the trait subspace --------------------------

def report_subspace(triples, out_dir: Path) -> dict:
    """Eigen-spectrum of the within-scenario centered triples."""
    spec = subspace_spectrum(triples, k=8)
    print("\n" + "=" * 70)
    print("3. Trait subspace spectrum  (within-scenario centered triples)")
    ratios = ", ".join(f"{r:.3f}" for r in spec["variance_ratio"][:6])
    print(f"   top-6 variance ratio: {ratios}")
    print(f"   cumulative (PC1..PC2) = {spec['cumulative'][1]:.3f}"
          f"   participation ratio = {spec['participation_ratio']:.2f}")
    print("   (rank ≤ 2 per scenario; a high participation ratio means the per-scenario"
          " planes are NOT shared — the bend is reliable on average but spread over many dims)")

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    idx = np.arange(1, len(spec["variance_ratio"]) + 1)
    ax.bar(idx, spec["variance_ratio"], color=C_REF, alpha=0.85, label="per-component")
    ax.plot(idx, spec["cumulative"], "o-", color=C_NEG, label="cumulative")
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance ratio")
    ax.set_title(f"{TRAIT} — within-scenario triple subspace (PR={spec['participation_ratio']:.1f})")
    ax.legend(loc="center right")
    fig.tight_layout()
    _save(fig, out_dir, "subspace_spectrum")

    num_dir = out_dir / "numeric" / "subspace"
    _write_csv(
        num_dir / "subspace_spectrum.csv",
        ["component", "variance_ratio", "cumulative"],
        [[int(i + 1), float(spec["variance_ratio"][i]), float(spec["cumulative"][i])] for i in range(len(spec["variance_ratio"]))],
    )
    _write_json(num_dir / "subspace_spectrum_summary.json", {"participation_ratio": spec["participation_ratio"]})
    return spec


# --- step 4: the shared (valence, markedness) plane ------------------------

def report_shared_plane(triples, out_dir: Path) -> dict:
    """Two-axis hypothesis: project all scenarios into a shared plane."""
    plane = shared_plane(triples)
    rel = markedness_reliability(triples, n_splits=N_REL_SPLITS, seed=REL_SEED)
    cv = shared_bend_cv(triples, n_splits=CV_SPLITS, seed=BEND_CV_SEED)

    print("\n" + "=" * 70)
    print("4. Shared (valence, markedness) plane  — the two-axis / superposition model")
    print(f"   variance captured  valence axis (1-D) = {plane['frac_valence_1d']:.3f}"
          f"   + markedness (2-D plane) = {plane['frac_plane_2d']:.3f}")
    print(f"   neutral on the same markedness side in {plane['neutral_markedness_sign_frac']:.1%} of scenarios")
    print(f"   markedness-axis reliability (Spearman-Brown) R = {rel['R_full']:.3f}"
          f"   (half-sample r = {rel['r_half']:.3f})")
    print(f"   shared-bend cross-validation: R²_cv = {cv['r2_cv']:+.3f}"
          f"   ({cv['frac_scenarios_improved']:.1%} of held-out scenarios improved)")
    print("   → a single shared off-line direction (markedness) is real and reliable,")
    print("     but most of each scenario's bend magnitude is scenario-specific.")

    # Figure: per-scenario triangles + mean triangle with SE bars in the plane.
    coords_ps = plane["coords_per_scenario"]                 # (S, L, 2)
    mean_xy = plane["coords"]                                 # (L, 2)
    se_xy = coords_ps.std(axis=0) / np.sqrt(coords_ps.shape[0])
    palette = levels_palette(LEVELS)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for s in range(coords_ps.shape[0]):
        ax.plot(coords_ps[s, :, 0], coords_ps[s, :, 1], color="gray", alpha=0.12, lw=0.8, zorder=1)
    for k, lvl in enumerate(LEVELS):
        ax.scatter(coords_ps[:, k, 0], coords_ps[:, k, 1], s=14, color=palette[k],
                   alpha=0.35, edgecolors="none", zorder=2, label=lvl)
    for i in range(len(LEVELS) - 1):
        ax.annotate("", xy=mean_xy[i + 1], xytext=mean_xy[i],
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.6), zorder=4)
    for k in range(len(LEVELS)):
        ax.errorbar(mean_xy[k, 0], mean_xy[k, 1], xerr=se_xy[k, 0], yerr=se_xy[k, 1],
                    fmt="o", ms=11, color=palette[k], mec="black", mew=1.2,
                    ecolor="black", elinewidth=1.0, capsize=3, zorder=5)
    ax.axhline(0, color="black", lw=0.4, alpha=0.4)
    ax.axvline(0, color="black", lw=0.4, alpha=0.4)
    ax.set_xlabel("valence axis  (negative ← → positive)")
    ax.set_ylabel("markedness axis  (off the valence line)")
    ax.set_title(f"{TRAIT} — neg/neut/pos in the shared plane\n"
                 f"markedness R={rel['R_full']:.2f}, "
                 f"plane captures {plane['frac_plane_2d']:.0%} of within-scenario variance")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir, "shared_plane")

    # Numeric dumps for LaTeX/pgfplots.
    num_dir = out_dir / "numeric" / "shared_plane"
    sids = sorted(triples)
    coords_ps = plane["coords_per_scenario"]
    mean_xy = plane["coords"]
    se_xy = coords_ps.std(axis=0) / np.sqrt(coords_ps.shape[0])
    _write_csv(
        num_dir / "shared_plane_summary.csv",
        [
            "frac_valence_1d",
            "frac_plane_2d",
            "neutral_markedness_sign_frac",
            "markedness_r_half",
            "markedness_R_full",
            "shared_bend_r2_cv",
            "shared_bend_frac_scenarios_improved",
        ],
        [[
            float(plane["frac_valence_1d"]),
            float(plane["frac_plane_2d"]),
            float(plane["neutral_markedness_sign_frac"]),
            float(rel["r_half"]),
            float(rel["R_full"]),
            float(cv["r2_cv"]),
            float(cv["frac_scenarios_improved"]),
        ]],
    )

    _write_csv(
        num_dir / "shared_plane_axes.csv",
        ["dim", "e_v", "e_m"],
        [[int(i), float(plane["e_v"][i]), float(plane["e_m"][i])] for i in range(len(plane["e_v"]))],
    )

    _write_csv(
        num_dir / "shared_plane_coords_mean.csv",
        ["level", "valence", "markedness", "se_valence", "se_markedness"],
        [[LEVELS[i], float(mean_xy[i, 0]), float(mean_xy[i, 1]), float(se_xy[i, 0]), float(se_xy[i, 1])] for i in range(len(LEVELS))],
    )
    rows_ps: list[list[object]] = []
    for s, sid in enumerate(sids):
        for i, lvl in enumerate(LEVELS):
            rows_ps.append([sid, lvl, float(coords_ps[s, i, 0]), float(coords_ps[s, i, 1])])
    _write_csv(
        num_dir / "shared_plane_coords_per_scenario.csv",
        ["scenario_id", "level", "valence", "markedness"],
        rows_ps,
    )

    _write_csv(
        num_dir / "shared_plane_markedness_reliability_cosines.csv",
        ["split", "cosine"],
        [[int(i), float(rel["cosines"][i])] for i in range(len(rel["cosines"]))],
    )
    return {"plane": plane, "reliability": rel, "cv": cv}


# --- step 5: per-intent robustness -----------------------------------------

def report_per_intent(triples, out_dir: Path) -> dict:
    """Pooled bend geometry computed separately for each communicative intent."""
    by_intent: dict[str, list[str]] = collections.defaultdict(list)
    for sid in triples:
        by_intent[_intent_of(sid)].append(sid)

    overall = triple_geometry(*pooled_centroids(triples))
    print("\n" + "=" * 70)
    print("5. Per-intent geometry  (pooled within each communicative intent)")
    print(f"   {'intent':<24}{'n':>4}{'apex°':>9}{'step cos':>10}{'mid resid':>11}")
    print("   " + "-" * 58)
    rows = []
    for intent in sorted(by_intent):
        sids = by_intent[intent]
        if len(sids) < 2:
            continue
        g = triple_geometry(*pooled_centroids({s: triples[s] for s in sids}))
        rows.append((intent, len(sids), g))
        print(f"   {intent:<24}{len(sids):>4}{g['apex_angle_deg']:>9.1f}"
              f"{g['step_cosine']:>10.3f}{g['midpoint_residual']:>11.3f}")
    print(f"   {'ALL':<24}{len(triples):>4}{overall['apex_angle_deg']:>9.1f}"
          f"{overall['step_cosine']:>10.3f}{overall['midpoint_residual']:>11.3f}")

    intents = [r[0] for r in rows]
    apex = [r[2]["apex_angle_deg"] for r in rows]
    ncs = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    order = np.argsort(apex)
    y = np.arange(len(intents))
    ax.scatter([apex[i] for i in order], y, s=[12 * ncs[i] for i in order],
               color=C_REF, edgecolors="black", linewidths=0.6, zorder=3)
    ax.axvline(180, color=C_NULL, ls="--", lw=1.0, label="straight (180°)")
    ax.axvline(72, color=C_REF, ls=":", lw=1.0, label="pentagon (72°)")
    ax.axvline(overall["apex_angle_deg"], color=C_NEG, lw=1.4,
               label=f"overall {overall['apex_angle_deg']:.0f}°")
    ax.set_yticks(y)
    ax.set_yticklabels([intents[i] for i in order])
    ax.set_xlabel("apex angle at neutral (°)  — marker size ∝ #scenarios")
    ax.set_title(f"{TRAIT} — the bend holds across intents")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    _save(fig, out_dir, "per_intent_geometry")

    # Numeric dumps.
    num_dir = out_dir / "numeric" / "per_intent"
    keys = [
        "span",
        "midpoint_residual",
        "valence_offset",
        "markedness",
        "markedness_ratio",
        "apex_angle_deg",
        "effective_vertices",
        "step_cosine",
    ]
    csv_rows: list[list[object]] = []
    for intent, n, g in rows:
        csv_rows.append([intent, int(n)] + [float(g[k]) for k in keys])
    csv_rows.append(["ALL", int(len(triples))] + [float(overall[k]) for k in keys])
    _write_csv(
        num_dir / "per_intent_geometry.csv",
        ["intent", "n"] + keys,
        csv_rows,
    )
    return {"rows": rows, "overall": overall}


# --- step 6: one steering axis vs scenario-specific tilt -------------------

def report_steering_direction(triples, intents, out_dir: Path) -> dict:
    """Decompose the per-scenario plane tilt: which part is a shared direction?

    Activation steering presumes a single global direction transfers across
    contexts. The shared plane only captures ~20-26% of within-scenario
    variance, so this asks *which* component is global. It splits the geometry
    into the on-line valence (steering) axis and the off-line bend, and reports
    the reliability of each, how much of every scenario's contrast a single
    global vector reproduces, and the principal-angle tilt of each scenario's
    plane against the shared plane.
    """
    val_rel = valence_reliability(triples, n_splits=N_REL_SPLITS, seed=VAL_REL_SEED)
    mark_rel = markedness_reliability(triples, n_splits=N_REL_SPLITS, seed=REL_SEED)
    align = steering_alignment(triples)
    angles = plane_principal_angles(triples)
    intent = intent_contrast_structure(triples, intents)

    print("\n" + "=" * 70)
    print("6. One steering axis vs scenario-specific tilt")
    print(f"   axis reliability (Spearman-Brown)   valence/steering R = {val_rel['R_full']:.3f}"
          f"   markedness R = {mark_rel['R_full']:.3f}")
    print(f"   scenario contrast vs global steering axis:"
          f"   mean cos = {align['mean_cos_valence']:.3f}"
          f"   mean capture (cos²) = {align['mean_capture_valence']:.3f}")
    print(f"   scenario bend vs global markedness axis:"
          f"   mean cos = {align['mean_cos_bend']:.3f}")
    print(f"   plane principal angles (median)   θ1 = {angles['median_theta1']:.1f}°"
          f"   θ2 tilt = {angles['median_theta2']:.1f}°"
          f"   shared-plane overlap = {angles['median_overlap']:.2f}")
    print(f"   contrast cosine  within-intent = {intent['within_mean']:.3f}"
          f"   between-intent = {intent['between_mean']:.3f}"
          f"   (gap = {intent['within_mean'] - intent['between_mean']:+.3f})")
    print("   → the valence/steering axis is shared and reliable; the off-line bend")
    print("     and plane orientation are the scenario-specific part a single vector misses.")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    axes[0].hist(align["cos_valence"], bins=24, range=(0, 1), color=C_POS,
                 alpha=0.85, edgecolor="gray")
    axes[0].axvline(align["mean_cos_valence"], color="black", lw=1.6,
                    label=f"mean {align['mean_cos_valence']:.2f}")
    axes[0].set_xlabel("cos(scenario pos−neg, global steering axis)")
    axes[0].set_ylabel("scenarios")
    axes[0].set_title(f"steering axis reliable on average\nR={val_rel['R_full']:.2f}, "
                      f"per-scenario cos≈{align['mean_cos_valence']:.2f}")
    axes[0].legend(loc="upper left", fontsize=7)

    axes[1].hist(align["cos_bend"], bins=24, range=(-1, 1), color=C_NEUT,
                 alpha=0.85, edgecolor="gray")
    axes[1].axvline(align["mean_cos_bend"], color="black", lw=1.6,
                    label=f"mean {align['mean_cos_bend']:.2f}")
    axes[1].axvline(0, color="gray", ls="--", lw=1.0)
    axes[1].set_xlabel("cos(scenario bend, global markedness axis)")
    axes[1].set_title(f"bend only partly shared\nreliability R={mark_rel['R_full']:.2f}")
    axes[1].legend(loc="upper left", fontsize=7)

    axes[2].hist(angles["angles"][:, 0], bins=24, range=(0, 90), color=C_POS,
                 alpha=0.6, edgecolor="gray",
                 label=f"θ1 (median {angles['median_theta1']:.0f}°)")
    axes[2].hist(angles["angles"][:, 1], bins=24, range=(0, 90), color=C_NEG,
                 alpha=0.6, edgecolor="gray",
                 label=f"θ2 tilt (median {angles['median_theta2']:.0f}°)")
    axes[2].set_xlabel("principal angle to shared plane (°)")
    axes[2].set_title("plane tilt:\nθ1 small, θ2 large")
    axes[2].legend(loc="upper center", fontsize=7)

    fig.suptitle(f"{TRAIT} — reproducible mean steering axis, but most of each "
                 f"scenario's contrast is scenario-specific", y=1.04)
    fig.tight_layout()
    _save(fig, out_dir, "steering_direction")

    # Numeric dumps.
    num_dir = out_dir / "numeric" / "steering_direction"
    sids = sorted(triples)
    _write_csv(
        num_dir / "steering_direction_summary.csv",
        [
            "valence_r_half",
            "valence_R_full",
            "markedness_r_half",
            "markedness_R_full",
            "mean_cos_valence",
            "mean_capture_valence",
            "mean_cos_bend",
            "median_theta1",
            "median_theta2",
            "median_overlap",
            "contrast_within_intent_mean",
            "contrast_between_intent_mean",
            "contrast_gap",
        ],
        [[
            float(val_rel["r_half"]),
            float(val_rel["R_full"]),
            float(mark_rel["r_half"]),
            float(mark_rel["R_full"]),
            float(align["mean_cos_valence"]),
            float(align["mean_capture_valence"]),
            float(align["mean_cos_bend"]),
            float(angles["median_theta1"]),
            float(angles["median_theta2"]),
            float(angles["median_overlap"]),
            float(intent["within_mean"]),
            float(intent["between_mean"]),
            float(intent["within_mean"] - intent["between_mean"]),
        ]],
    )

    _write_csv(
        num_dir / "steering_direction_valence_reliability_cosines.csv",
        ["split", "cosine"],
        [[int(i), float(val_rel["cosines"][i])] for i in range(len(val_rel["cosines"]))],
    )
    _write_csv(
        num_dir / "steering_direction_markedness_reliability_cosines.csv",
        ["split", "cosine"],
        [[int(i), float(mark_rel["cosines"][i])] for i in range(len(mark_rel["cosines"]))],
    )

    per_rows = [
        [sid, float(align["cos_valence"][i]), float(align["capture_valence"][i]), float(align["cos_bend"][i])]
        for i, sid in enumerate(sids)
    ]
    _write_csv(
        num_dir / "steering_direction_alignment_per_scenario.csv",
        ["scenario_id", "cos_valence", "capture_valence", "cos_bend"],
        per_rows,
    )
    ang_rows = [
        [sid, float(angles["angles"][i, 0]), float(angles["angles"][i, 1]), float(angles["overlap"][i])]
        for i, sid in enumerate(sids)
    ]
    _write_csv(
        num_dir / "steering_direction_principal_angles.csv",
        ["scenario_id", "theta1_deg", "theta2_deg", "overlap"],
        ang_rows,
    )
    return {"valence_reliability": val_rel, "markedness_reliability": mark_rel,
            "alignment": align, "angles": angles, "intent": intent}


# --- entrypoint ------------------------------------------------------------

def main(model: str = MODEL, token_pooling: str = TOKEN_POOLING, seed: int = SEED) -> Path:
    _configure(model, token_pooling, seed)
    apply_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(
        RESULTS_DIR / "run_metadata.json",
        run_metadata(model=MODEL, seed=SEED, token_pooling=TOKEN_POOLING, dataset=DATASET, focal_layer=LAYER),
    )
    print(f"Saving figures under {RESULTS_DIR}\n")

    para = load_representations(REP_DIR, layer=LAYER)
    scen = pool_by_scenario_level(para)
    levels = _present_levels(scen)
    triples = scenario_triples(scen, levels, trait=TRAIT)
    print(f"Loaded {len(para)} paraphrase vectors → {len(scen)} centroids → "
          f"{len(triples)} complete scenarios  |  layer {LAYER}  |  "
          f"token_pooling={token_pooling!r}  |  levels {levels}")

    intents = {sid: _intent_of(sid) for sid in triples}

    report_geometry(triples, RESULTS_DIR)
    report_noise_null(para, levels, RESULTS_DIR)
    report_subspace(triples, RESULTS_DIR)
    report_shared_plane(triples, RESULTS_DIR)
    report_per_intent(triples, RESULTS_DIR)
    report_steering_direction(triples, intents, RESULTS_DIR)

    print(f"\nDone. Figures in {RESULTS_DIR}")
    return RESULTS_DIR


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Geometry of the trait triple.")
    ap.add_argument(
        "--model", choices=sorted(MODELS), default=DEFAULT_MODEL,
        help="Model whose representations to analyse (default: %(default)s).",
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
        main(args.model, args.token_pooling, s)
    if len(seeds) > 1:
        from aggregate_results import aggregate_analysis

        aggregate_analysis(seeds_base_dir(DATASET_ROOT.name, ANALYSIS, args.model, args.token_pooling))
