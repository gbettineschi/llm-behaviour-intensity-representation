"""Cross-combo summary table: one row per (model, trait, token_pooling) with
the headline numbers from every analysis driver, for reviewing the sweep at a
glance.

Scrapes the numeric CSVs the drivers already emit — no analysis logic lives
here. Bootstrap-CI and null-p-value columns (drawn from Monte-Carlo machinery
that doesn't aggregate meaningfully across seeds, per aggregate_results.py)
are always read from seed_0; columns that are legitimate cross-seed averages
are read from aggregated/ when more than one seed was run, else from seed_0.

Run from the repo root:  uv run python src/collect_summary.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lib.config import MODELS, seeds_base_dir
from lib.traits import TRAITS

DATASET_ROOT = Path("data/20260530_001930")
POOLINGS = ("avg", "last")

# Nominal (marketing) parameter counts in billions — a scale-axis column for
# the summary table, not used anywhere else.
MODEL_SIZE_B = {
    "gemma-2-2b": 2.0,
    "llama-3.2-3b": 3.0,
    "qwen2.5-0.5b": 0.5,
    "qwen2.5-1.5b": 1.5,
    "qwen2.5-1.5b-instruct": 1.5,
    "qwen2.5-3b": 3.0,
    "qwen2.5-7b": 7.0,
}


def _combo_dir(analysis: str, model: str, trait: str, pooling: str) -> Path:
    return seeds_base_dir(DATASET_ROOT.name, analysis, model, trait, pooling)


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _seeded_or_aggregated(base: Path, rel_csv: str) -> pd.DataFrame | None:
    """Aggregated version when >1 seed was run, else the (only) seed_0 file."""
    agg = _read_csv(base / "aggregated" / "numeric" / rel_csv)
    return agg if agg is not None else _read_csv(base / "seed_0" / "numeric" / rel_csv)


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

    # Bootstrap-CI / null-p-value columns: always seed_0 (see module docstring).
    bootstrap = _read_csv(geom_dir / "seed_0" / "numeric" / "geometry" / "geometry_bootstrap_summary.csv")
    null = _read_csv(geom_dir / "seed_0" / "numeric" / "noise_null" / "noise_null_summary.csv")

    # Cross-seed-averageable columns: aggregated/ if present, else seed_0.
    linearity = _seeded_or_aggregated(lin_dir, "linearity/linearity_metrics.csv")
    shared_plane = _seeded_or_aggregated(geom_dir, "shared_plane/shared_plane_summary.csv")

    seed_dirs = list((geom_dir if geom_dir.exists() else lin_dir).glob("seed_*"))
    n_seeds = len(seed_dirs) or 1

    row = {
        "model": model,
        "params_b": MODEL_SIZE_B.get(model),
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


if __name__ == "__main__":
    df = collect_all()
    out_dir = Path("results") / DATASET_ROOT.name / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
