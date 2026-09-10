"""Cross-combo summary table: one row per (model, trait, token_pooling) with
the headline numbers from every analysis driver, for reviewing the sweep at a
glance — plus a cross-combo synthesis testing whether the bend is a general
phenomenon (not a single-trait or single-model artifact).

Scrapes the numeric CSVs the drivers already emit — no analysis logic lives
here. Every column is read from ``aggregated/`` when the combo was aggregated,
falling back to the lowest per-seed directory otherwise. Cross-seed values carry
the caveats aggregate_results.py documents: bootstrap CI bounds are averaged
rather than pooled, and Monte-Carlo p-values are averaged (floored at
1/(1+n_sim)). Point estimates are deterministic across seeds and pass through
unchanged.

Reading a hardcoded ``seed_0`` would not work: per-seed dirs are gitignored, so
a fresh clone has only ``aggregated/``, and the dirs are named by the master
seed *value*, so ``--seeds 3,4`` produces no ``seed_0`` at all.

Run from the repo root:  uv run python src/collect_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import combine_pvalues, spearmanr

from lib.config import DEFAULT_DATA_ROOT, MODELS, seeds_base_dir
from lib.traits import TRAITS

DATASET_ROOT = DEFAULT_DATA_ROOT
POOLINGS = ("avg", "last")


def _combo_dir(analysis: str, model: str, trait: str, pooling: str) -> Path:
    return seeds_base_dir(DATASET_ROOT.name, analysis, model, trait, pooling)


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _seed_dirs(base: Path) -> list[Path]:
    """Per-seed dirs under a combo, lowest master seed first."""
    dirs = [p for p in base.glob("seed_*") if p.is_dir() and p.name.split("_", 1)[1].isdigit()]
    return sorted(dirs, key=lambda p: int(p.name.split("_", 1)[1]))


def _seeded_or_aggregated(base: Path, rel_csv: str) -> pd.DataFrame | None:
    """The aggregated file if the combo was aggregated, else the lowest seed's."""
    agg = _read_csv(base / "aggregated" / "numeric" / rel_csv)
    if agg is not None:
        return agg
    for d in _seed_dirs(base):
        df = _read_csv(d / "numeric" / rel_csv)
        if df is not None:
            return df
    return None


def _n_seeds(base: Path) -> int:
    """Seeds behind this combo — from the aggregate's provenance record when
    present, since the per-seed dirs it counted are gitignored."""
    meta = base / "aggregated" / "run_metadata.json"
    if meta.exists():
        seeds = json.loads(meta.read_text(encoding="utf-8")).get("seeds")
        if seeds:
            return len(seeds)
    return len(_seed_dirs(base)) or 1


def _col(df: pd.DataFrame, name: str) -> pd.Series | None:
    """``<name>_mean`` if present (aggregated, varying), else ``<name>`` (deterministic
    passthrough or single-seed raw)."""
    if f"{name}_mean" in df.columns:
        return df[f"{name}_mean"]
    if name in df.columns:
        return df[name]
    return None


def _row_value(df: pd.DataFrame | None, metric_col: str, metric: str, value_col: str):
    if df is None or metric_col not in df.columns:
        return None
    match = df[df[metric_col] == metric]
    if match.empty:
        return None
    v = _col(match, value_col)
    return v.iloc[0] if v is not None else None


def collect_combo(model: str, trait: str, pooling: str) -> dict | None:
    geom_dir = _combo_dir("trait_geometry", model, trait, pooling)
    lin_dir = _combo_dir("ordinal_linearity", model, trait, pooling)
    if not geom_dir.exists() and not lin_dir.exists():
        return None

    bootstrap = _seeded_or_aggregated(geom_dir, "geometry/geometry_bootstrap_summary.csv")
    null = _seeded_or_aggregated(geom_dir, "noise_null/noise_null_summary.csv")
    linearity = _seeded_or_aggregated(lin_dir, "linearity/linearity_metrics.csv")
    shared_plane = _seeded_or_aggregated(geom_dir, "shared_plane/shared_plane_summary.csv")

    n_seeds = _n_seeds(geom_dir if geom_dir.exists() else lin_dir)

    row = {
        "model": model,
        "params_b": MODELS[model].get("params_b"),
        "instruct": "instruct" in model,
        "trait": trait,
        "token_pooling": pooling,
        "n_seeds": n_seeds,
        "apex_angle_deg": _row_value(bootstrap, "metric", "apex_angle_deg", "point"),
        "apex_angle_ci_lo": _row_value(bootstrap, "metric", "apex_angle_deg", "ci_lo"),
        "apex_angle_ci_hi": _row_value(bootstrap, "metric", "apex_angle_deg", "ci_hi"),
        "step_cosine": _row_value(null, "metric", "step_cosine", "observed"),
        "step_cosine_null_p": _row_value(null, "metric", "step_cosine", "p_value"),
        "midpoint_residual": _row_value(null, "metric", "midpoint_residual", "observed"),
        "midpoint_residual_null_p": _row_value(null, "metric", "midpoint_residual", "p_value"),
        "spearman_within_scenario": _row_value(linearity, "metric", "spearman", "within_scenario"),
        "probe_r2_within_scenario": _row_value(linearity, "metric", "probe_r2", "within_scenario"),
        "markedness_reliability_R": _col(shared_plane, "markedness_R_full").iloc[0] if shared_plane is not None else None,
        "shared_bend_r2_cv": _col(shared_plane, "shared_bend_r2_cv").iloc[0] if shared_plane is not None else None,
    }
    return row


def collect_all() -> pd.DataFrame:
    rows = []
    for model in sorted(MODELS):
        for trait in sorted(TRAITS):
            for pooling in POOLINGS:
                row = collect_combo(model, trait, pooling)
                if row is not None:
                    rows.append(row)
    return pd.DataFrame(rows)


# --- cross-combo synthesis --------------------------------------------------
#
# A per-combo p-value in noise_null_summary.csv already tests the right
# one-sided direction (step_cosine significantly *below*, midpoint_residual
# significantly *above*, what a linear ladder + empirical noise would produce
# — see the ("step_cosine", "<="), ("midpoint_residual", ">=") checks in
# trait_geometry.py). Combining those across combos with Fisher's method is
# therefore a direct test of "the bend is real in aggregate, across traits and
# model scales" — not just eyeballing dozens of separate CIs.


def _fisher_combined_p(pvalues: pd.Series) -> dict:
    pvalues = pd.to_numeric(pvalues, errors="coerce").dropna()
    n = len(pvalues)
    if n == 0:
        return {"n": 0, "p_value": None}
    if n == 1:
        return {"n": 1, "p_value": float(pvalues.iloc[0])}
    return {"n": n, "p_value": float(combine_pvalues(pvalues, method="fisher").pvalue)}


def _group_summary(df: pd.DataFrame, by: str) -> list[dict]:
    g = df.groupby(by).agg(
        n=("apex_angle_deg", "size"),
        apex_angle_deg_mean=("apex_angle_deg", "mean"),
        step_cosine_mean=("step_cosine", "mean"),
        frac_significant_bend=("step_cosine_null_p", lambda s: float((pd.to_numeric(s, errors="coerce") < 0.05).mean())),
    )
    return g.reset_index().to_dict(orient="records")


def meta_analysis(df: pd.DataFrame) -> dict:
    """Cross-combo synthesis: is the bend general, or a single trait/model artifact?

    * Fisher-combined p-values across every combo's linear-ladder-null test
      (both the step_cosine and midpoint_residual versions).
    * Fraction of combos individually significant at alpha=0.05.
    * Spearman correlation between model scale (params_b) and bend magnitude
      (-step_cosine, since 1.0 = collinear/linear) — tests whether the bend
      shrinks with scale rather than persisting.
    * Per-trait and per-model breakdowns, so "does it hold for every trait /
      every model" can be read off directly rather than re-deriving it from
      summary.csv by hand.
    """
    out: dict = {"n_combos": len(df)}
    if df.empty:
        return out

    out["fisher_combined_p"] = {
        "step_cosine_null": _fisher_combined_p(df["step_cosine_null_p"]),
        "midpoint_residual_null": _fisher_combined_p(df["midpoint_residual_null_p"]),
    }
    out["frac_significant_bend_at_0.05"] = float(
        (pd.to_numeric(df["step_cosine_null_p"], errors="coerce") < 0.05).mean()
    )

    scale_df = df.dropna(subset=["params_b", "step_cosine"])
    if len(scale_df) >= 3:
        rho, p = spearmanr(scale_df["params_b"], -scale_df["step_cosine"])
        out["scale_vs_bend_magnitude"] = {"n": len(scale_df), "spearman_rho": float(rho), "p_value": float(p)}
    else:
        out["scale_vs_bend_magnitude"] = {"n": len(scale_df), "spearman_rho": None, "p_value": None}

    out["by_trait"] = _group_summary(df, "trait")
    out["by_model"] = _group_summary(df, "model")
    return out


if __name__ == "__main__":
    df = collect_all()
    out_dir = Path("results") / DATASET_ROOT.name / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Wrote {len(df)} rows to {summary_path}")

    meta = meta_analysis(df)
    meta_path = out_dir / "meta_analysis.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote cross-combo synthesis to {meta_path}")
    if meta["n_combos"] < 10:
        print(
            f"  NOTE: only {meta['n_combos']} combos present — treat this synthesis as "
            "illustrative until the full model x trait sweep has run."
        )
