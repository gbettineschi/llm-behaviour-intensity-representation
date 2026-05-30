"""Replicate Tigges et al. 2024 §2.2 on the politeness dataset.

Loads paraphrase-level activations produced by ``extract_representations.py``
and, for each binary contrast (positive vs negative, positive vs neutral,
neutral vs negative), computes:

    * direction-estimator agreement (cosine of MeanDiff / KMeans / LogReg / PCA)
    * direction-as-classifier balanced accuracy under scenario-grouped CV
    * projection separation on the MeanDiff axis (AUC + Cohen's d)

Each contrast is run twice — once on raw activations, once after a
within-scenario fixed-effects transform — so the per-scenario offset can be
isolated from the trait signal. A layer-sweep on the binary contrast is also
produced.

Figures are written to ``results/<dataset>/replication_tigges/<token_pooling>_token/``;
numeric summaries are printed, and numeric exports land under ``numeric/``.

Run from the repo root:  uv run python src/replicate_tigges.py

Or explicitly:
    uv run python src/replicate_tigges.py --token-pooling avg
    uv run python src/replicate_tigges.py --token-pooling last
"""

from __future__ import annotations

from pathlib import Path
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
from lib.figures import (
    apply_style,
    plot_agreement,
    plot_layer_sweep,
    plot_projection,
)
from lib.exports import (
    save_fig as _save,
    write_csv as _write_csv,
    write_square_matrix as _write_square_matrix,
    write_tex_tabular as _write_tex_tabular,
)
from lib.representations import load_representations


# --- config ----------------------------------------------------------------

LAYER = 13
TRAIT = "politeness"
TOKEN_POOLING = "avg"
DATASET = Path("data/20260530_001930/sentences/sentences_filtered.jsonl")
DATASET_ROOT = DATASET.parent.parent
REP_DIR = DATASET_ROOT / "representations" / f"{TOKEN_POOLING}_token"
RESULTS_DIR = Path("results") / DATASET_ROOT.name / "replication_tigges" / f"{TOKEN_POOLING}_token"

NEG, NEU, POS = "negative", "neutral", "positive"
BINARY = (NEG, POS)
ADJACENT_PAIRS = ((NEU, POS), (NEG, NEU))

# GPT2-small layer 0 accuracies from the paper Fig. 3 — point of comparison.
PAPER_REF = {"MeanDiff": ".80", "KMeans": ".78", "LogReg": ".89", "PCA": ".81"}


# --- pieces ----------------------------------------------------------------


def _configure(token_pooling: str) -> None:
    global TOKEN_POOLING, REP_DIR, RESULTS_DIR
    TOKEN_POOLING = token_pooling
    REP_DIR = DATASET_ROOT / "representations" / f"{token_pooling}_token"
    RESULTS_DIR = Path("results") / DATASET_ROOT.name / "replication_tigges" / f"{token_pooling}_token"



def contrast_analysis(activations, levels, out_dir: Path, *, paper_ref=None) -> None:
    """Direction-agreement and projection-separation for one binary contrast,
    on both raw and within-scenario-centered activations.

    Emits four figures:
        direction_agreement_<hi>_vs_<lo>_raw.png
        direction_agreement_<hi>_vs_<lo>_centered.png
        projection_<hi>_vs_<lo>_raw.png
        projection_<hi>_vs_<lo>_centered.png

    And prints a CV balanced-accuracy table (raw vs centered, with the paper
    Fig. 3 reference column when ``paper_ref`` is supplied).
    """
    lo, hi = levels
    label = f"{hi}_vs_{lo}"
    num_dir = out_dir / "numeric" / label

    X_raw, y_raw, scen_raw = level_table(activations, levels, TRAIT)
    centered = within_center_paraphrase(activations, list(levels), trait=TRAIT)
    X_cent, y_cent, scen_cent = level_table(centered, levels, TRAIT)

    print("=" * 60)
    print(f"{hi} vs. {lo}  (layer {LAYER})")
    print(f"  raw:      {X_raw.shape[0]} paraphrases / {len(set(scen_raw))} scenarios")
    print(f"  centered: {X_cent.shape[0]} paraphrases / {len(set(scen_cent))} complete scenarios")
    extra = "    paper Fig.3 (GPT2-small L0)" if paper_ref else ""
    print(f"\n  {'method':>9} {'raw':>9} {'centered':>10}{extra}")
    print("  " + "-" * (32 + len(extra)))
    acc_rows: list[list[object]] = []
    for m in METHODS:
        r_mean, r_sd = cv_direction_accuracy(m, X_raw, y_raw, scen_raw)
        c_mean, c_sd = cv_direction_accuracy(m, X_cent, y_cent, scen_cent)
        ref = f"    {paper_ref.get(m, '')}" if paper_ref else ""
        print(f"  {m:>9} {r_mean:>9.3f} {c_mean:>10.3f}{ref}")
        acc_rows.append([m, r_mean, r_sd, c_mean, c_sd, paper_ref.get(m, "") if paper_ref else ""])

    _write_csv(
        num_dir / "cv_accuracy.csv",
        ["method", "raw_mean", "raw_sd", "centered_mean", "centered_sd", "paper_ref"],
        acc_rows,
    )
    tex_header = ["method", "raw", "centered"] + (["paper"] if paper_ref else [])
    tex_rows = [
        [r[0], f"{float(r[1]):.3f}", f"{float(r[3]):.3f}"] + ([r[5]] if paper_ref else [])
        for r in acc_rows
    ]
    _write_tex_tabular(num_dir / "cv_accuracy.tex", tex_header, tex_rows)

    for tag, X, y in [("raw", X_raw, y_raw), ("centered", X_cent, y_cent)]:
        dirs = {m: direction(m, X, y) for m in NAMES}
        M = cosine_matrix(dirs, NAMES)
        fig, _ = plot_agreement(
            M, NAMES,
            f"Direction agreement — {hi} vs. {lo} ({tag})",
        )
        _save(fig, out_dir, f"direction_agreement_{label}_{tag}")

        _write_square_matrix(num_dir / f"direction_agreement_{tag}", list(NAMES), M)

        proj, auc, d = projection_stats(X, y, dirs["MeanDiff"])
        fig, _ = plot_projection(
            proj, y,
            neg_label=lo, pos_label=hi,
            title=f"Projection separation — {hi} vs. {lo} ({tag})  |  AUC={auc:.3f},  d={d:.2f}",
        )
        _save(fig, out_dir, f"projection_{label}_{tag}")

        proj_rows = [[float(p), int(yy), hi if yy == 1 else lo] for p, yy in zip(proj, y)]
        _write_csv(
            num_dir / f"projection_{tag}.csv",
            ["projection", "y", "level"],
            proj_rows,
        )
        _write_csv(
            num_dir / f"projection_{tag}_summary.csv",
            ["auc", "cohens_d", "n", "pos_level", "neg_level"],
            [[float(auc), float(d), int(len(proj)), hi, lo]],
        )


def layer_sweep(out_dir: Path) -> None:
    """MeanDiff CV accuracy + mean cross-method cosine on the binary contrast,
    across every layer in ``REP_DIR``."""
    acts_multi = load_representations(REP_DIR)
    layers = sorted({l for (*_, l) in acts_multi})
    print(f"Layer sweep over {len(layers)} layers from {REP_DIR}")

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
    _save(fig, out_dir, "layer_sweep")

    _write_csv(
        out_dir / "numeric" / "layer_sweep.csv",
        ["layer", "accuracy_mean", "agreement_mean"],
        [[int(L), float(a), float(g)] for L, a, g in zip(layers, accuracy, agreement)],
    )


# --- entrypoint ------------------------------------------------------------

def main(token_pooling: str = TOKEN_POOLING) -> Path:
    _configure(token_pooling)
    apply_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures under {RESULTS_DIR}\n")

    activations = load_representations(REP_DIR, layer=LAYER)
    print(f"Loaded {len(activations)} paraphrase vectors at layer {LAYER} (token_pooling={TOKEN_POOLING!r})\n")

    contrast_analysis(activations, BINARY, RESULTS_DIR, paper_ref=PAPER_REF)
    print()
    for pair in ADJACENT_PAIRS:
        contrast_analysis(activations, pair, RESULTS_DIR)
        print()
    layer_sweep(RESULTS_DIR)

    print(f"\nDone. Figures in {RESULTS_DIR}")
    return RESULTS_DIR


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Replicate Tigges et al. 2024 §2.2.")
    ap.add_argument(
        "--token-pooling", choices=("avg", "last"), default=TOKEN_POOLING,
        help="Prompt-token pooling whose representations to analyse (default: %(default)s).",
    )
    args = ap.parse_args()
    main(args.token_pooling)
