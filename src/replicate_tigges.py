"""Replicate Tigges et al. 2024 §2.2 on the politeness dataset.

Loads paraphrase-level activations produced by ``extract_representations.py``
and runs:

    R.1  direction-estimator agreement on the (negative, positive) contrast
    R.2  direction-as-classifier CV accuracy + projection separation
    R.4  raw vs within-scenario-centered diagnostics on each adjacent half-step
    R.5  direction quality vs. layer

Figures are written to ``results/replication_tigges/<timestamp>/``; numeric
summaries are printed.

Run from the repo root:  uv run python src/replicate_tigges.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lib.analysis import within_center_paraphrase
from lib.directions import (
    METHODS,
    NAMES,
    cosine_matrix,
    cv_direction_accuracy,
    direction,
    level_table,
    projection_stats,
)
from lib.plotting import (
    apply_style,
    plot_agreement,
    plot_agreement_pair,
    plot_layer_sweep,
    plot_projection,
    plot_projection_pair,
)
from lib.representations import load_representations


# --- config (matches the prior 004 notebook) -------------------------------

LAYER = 13
TRAIT = "politeness"
POOL = "last"
DATASET = Path("data/20260530_001930/sentences/sentences_filtered.jsonl")
REP_DIR = DATASET.parent.parent / "representations" / POOL
RESULTS_ROOT = Path("results/replication_tigges")

NEG, NEU, POS = "negative", "neutral", "positive"
BINARY = (NEG, POS)
ADJACENT_PAIRS = ((NEU, POS), (NEG, NEU))

# GPT2-small layer 0 accuracies from the paper Fig. 3 — point of comparison.
PAPER_REF = {"MeanDiff": ".80", "KMeans": ".78", "LogReg": ".89", "PCA": ".81"}


# --- pieces ----------------------------------------------------------------

def _save(fig, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def r1_direction_agreement(activations, out_dir: Path) -> None:
    X, y, _ = level_table(activations, BINARY, TRAIT)
    print(
        f"R.1 binary contrast at layer {LAYER}: "
        f"{X.shape[0]} vectors ({int((y == 1).sum())} pos / {int((y == 0).sum())} neg) "
        f"| dim={X.shape[1]}"
    )
    dirs = {m: direction(m, X, y) for m in NAMES}
    fig, _ = plot_agreement(
        cosine_matrix(dirs, NAMES), NAMES, "Direction agreement across estimators"
    )
    _save(fig, out_dir, "r1_direction_agreement")


def r2_classifier_and_projection(activations, out_dir: Path) -> None:
    X, y, scen = level_table(activations, BINARY, TRAIT)
    print(f"{'method':>9} {'bal-acc':>9} {'±SD':>6}    paper Fig.3 (GPT2-small L0)")
    print("-" * 52)
    for m in METHODS:
        mu, sd = cv_direction_accuracy(m, X, y, scen)
        print(f"{m:>9} {mu:>9.3f} {sd:>6.3f}    {PAPER_REF[m]}")

    axis_md = direction("MeanDiff", X, y)
    proj, auc, cohen_d = projection_stats(X, y, axis_md)
    print(f"\nMean-Difference axis: AUC={auc:.3f}, Cohen's d={cohen_d:.2f}")
    fig, _ = plot_projection(
        proj, y,
        neg_label=NEG, pos_label=POS,
        title=f"Projection separation at layer {LAYER}",
    )
    _save(fig, out_dir, "r2_projection_separation")


def r4_raw_vs_centered(activations, out_dir: Path) -> None:
    for lo, hi in ADJACENT_PAIRS:
        Xr, yr, scen_r = level_table(activations, (lo, hi), TRAIT)
        centered = within_center_paraphrase(activations, [lo, hi], trait=TRAIT)
        Xk, yk, scen_k = level_table(centered, (lo, hi), TRAIT)

        print("=" * 60)
        print(f"{hi} vs. {lo}  (layer {LAYER})")
        print(f"  raw:      {Xr.shape[0]} paraphrases / {len(set(scen_r))} scenarios")
        print(f"  centered: {Xk.shape[0]} paraphrases / {len(set(scen_k))} complete scenarios")
        print(f"\n  {'method':>9} {'raw':>9} {'centered':>10}")
        print("  " + "-" * 30)
        for m in METHODS:
            r = cv_direction_accuracy(m, Xr, yr, scen_r)[0]
            k = cv_direction_accuracy(m, Xk, yk, scen_k)[0]
            print(f"  {m:>9} {r:>9.3f} {k:>10.3f}")

        dr = {m: direction(m, Xr, yr) for m in NAMES}
        dk = {m: direction(m, Xk, yk) for m in NAMES}
        fig, _ = plot_agreement_pair(
            matrices=[cosine_matrix(dr, NAMES), cosine_matrix(dk, NAMES)],
            titles=["raw activations", "within-scenario centered"],
            names=NAMES,
            suptitle=f"{hi} vs. {lo} — direction agreement",
        )
        _save(fig, out_dir, f"r4_{hi}_vs_{lo}_agreement")

        pr, aucr, cdr = projection_stats(Xr, yr, direction("MeanDiff", Xr, yr))
        pk, auck, cdk = projection_stats(Xk, yk, direction("MeanDiff", Xk, yk))
        fig, _ = plot_projection_pair(
            panels=[(pr, yr), (pk, yk)],
            titles=[
                f"raw  (AUC={aucr:.3f},  d={cdr:.2f})",
                f"within-scenario centered  (AUC={auck:.3f},  d={cdk:.2f})",
            ],
            neg_label=lo, pos_label=hi,
            suptitle=f"{hi} vs. {lo} — projection separation",
        )
        _save(fig, out_dir, f"r4_{hi}_vs_{lo}_projection")


def r5_layer_sweep(out_dir: Path) -> None:
    acts_multi = load_representations(REP_DIR)
    layers = sorted({l for (*_, l) in acts_multi})
    print(f"R.5 layer sweep over {len(layers)} layers from {REP_DIR}")

    accuracy, agreement = [], []
    for L in layers:
        aL = {(t, i, s, p): v for (t, i, s, p, l), v in acts_multi.items() if l == L}
        X, y, scen = level_table(aL, BINARY, TRAIT)
        accuracy.append(cv_direction_accuracy("MeanDiff", X, y, scen)[0])
        dset = {m: direction(m, X, y) for m in METHODS}
        agreement.append(
            float(np.mean([dset[a] @ dset[b] for a in METHODS for b in METHODS if a < b]))
        )

    fig, _ = plot_layer_sweep(layers, accuracy, agreement, focal_layer=LAYER)
    _save(fig, out_dir, "r5_layer_sweep")


# --- entrypoint ------------------------------------------------------------

def main() -> Path:
    apply_style()
    out_dir = RESULTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures under {out_dir}\n")

    activations = load_representations(REP_DIR, layer=LAYER)
    print(f"Loaded {len(activations)} paraphrase vectors at layer {LAYER} (pool={POOL!r})\n")

    r1_direction_agreement(activations, out_dir)
    print()
    r2_classifier_and_projection(activations, out_dir)
    print()
    r4_raw_vs_centered(activations, out_dir)
    print()
    r5_layer_sweep(out_dir)

    print(f"\nDone. Figures in {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
