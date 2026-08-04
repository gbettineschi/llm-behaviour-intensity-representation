"""Aggregate per-seed analysis results into mean±std summaries for the paper.

Given a base dir ``results/<ts>/<analysis>/<model>/<trait>/<pooling>_token`` containing
``seed_<k>/`` runs, mirrors every numeric export into ``aggregated/``:

    * CSVs — non-numeric (key) columns must match exactly across seeds; numeric
      columns identical across seeds pass through unchanged (deterministic
      quantities keep std = 0 out of the table entirely), varying columns become
      ``<col>_mean`` / ``<col>_std`` (ddof=1) plus one ``n_seeds`` column.
    * JSONs — numeric leaves identical across seeds are kept, varying ones become
      ``{"mean":..., "std":..., "n_seeds":...}`` (element-wise for equal-length
      numeric lists); non-numeric leaves that differ are kept as ``{"per_seed": [...]}``.
    * Draw-level dumps (bootstrap draws, permutation/noise null draws, reliability
      half-split cosines) are skipped — cross-seed row-wise stats over independent
      Monte-Carlo draws are meaningless. Skips are listed in ``aggregation_report.json``.
    * For every aggregated CSV whose leading column is ``layer``, a per-metric
      mean±std band figure is written as ``<stem>_bands.png``.

Caveats (documented, accepted): Monte-Carlo p-values are averaged across seeds
(floor 1/(1+n_perm)); bootstrap CI bounds are averaged, draws are not pooled.

Run from the repo root:
    uv run python src/aggregate_results.py --dir results/<ts>/<analysis>/<model>/<trait>/<pooling>_token
    uv run python src/aggregate_results.py --discover results/<ts>
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.exports import write_csv as _write_csv, write_json as _write_json, write_tex_tabular as _write_tex_tabular
from lib.figures import apply_style

SKIP_FILES = {
    "geometry_bootstrap_draws.csv",
    "noise_null_null.csv",
    "linearity_permutation_null.csv",
    "run_metadata.json",
}
SKIP_SUFFIXES = ("_cosines.csv",)


def _is_skipped(relpath: Path) -> bool:
    return relpath.name in SKIP_FILES or relpath.name.endswith(SKIP_SUFFIXES)


def _aggregate_csv(frames: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Mean/std across seed frames; None if the frames are misaligned."""
    first = frames[0]
    if any(list(f.columns) != list(first.columns) or len(f) != len(first) for f in frames[1:]):
        return None
    out: dict[str, object] = {}
    varying = False
    for col in first.columns:
        if all(pd.api.types.is_numeric_dtype(f[col]) for f in frames):
            stack = np.stack([f[col].to_numpy(dtype=float) for f in frames])
            if np.allclose(stack, stack[0], equal_nan=True):
                out[col] = first[col]
            else:
                out[f"{col}_mean"] = np.nanmean(stack, axis=0)
                out[f"{col}_std"] = np.nanstd(stack, axis=0, ddof=1)
                varying = True
        else:
            key = first[col].fillna("")
            if any(not f[col].fillna("").equals(key) for f in frames[1:]):
                return None
            out[col] = first[col]
    agg = pd.DataFrame(out)
    if varying:
        agg["n_seeds"] = len(frames)
    return agg


def _aggregate_json(values: list[object], n: int) -> object:
    """Recursive cross-seed merge of loaded JSON documents."""
    first = values[0]
    if all(v == first for v in values[1:]):
        return first
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        arr = np.array(values, dtype=float)
        return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr, ddof=1)), "n_seeds": n}
    if all(isinstance(v, dict) for v in values) and all(v.keys() == first.keys() for v in values[1:]):
        return {k: _aggregate_json([v[k] for v in values], n) for k in first}
    if all(isinstance(v, list) and len(v) == len(first) for v in values):
        return [_aggregate_json([v[i] for v in values], n) for i in range(len(first))]
    return {"per_seed": values}


def _band_figure(agg: pd.DataFrame, out_path: Path) -> None:
    """One panel per varying metric: mean vs layer with a ±std band."""
    # original columns may themselves end in "_mean"; only pair with an actual _std
    mean_cols = [c for c in agg.columns if c.endswith("_mean") and f"{c[: -len('_mean')]}_std" in agg.columns]
    if not mean_cols or agg.columns[0] != "layer":
        return
    layers = agg["layer"].to_numpy()
    fig, axes = plt.subplots(1, len(mean_cols), figsize=(4.2 * len(mean_cols), 3.4), squeeze=False)
    for ax, col in zip(axes[0], mean_cols):
        base = col[: -len("_mean")]
        mean, std = agg[col].to_numpy(dtype=float), agg[f"{base}_std"].to_numpy(dtype=float)
        ax.plot(layers, mean, marker="o", ms=3)
        ax.fill_between(layers, mean - std, mean + std, alpha=0.25, lw=0)
        ax.set_xlabel("layer")
        ax.set_title(f"{base} (mean ± std over {int(agg['n_seeds'].iloc[0])} seeds)", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def aggregate_analysis(base_dir: str | Path) -> Path:
    """Aggregate all ``seed_*`` runs under ``base_dir`` into ``base_dir/aggregated``."""
    base_dir = Path(base_dir)
    seed_dirs = sorted(base_dir.glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))
    if len(seed_dirs) < 2:
        raise SystemExit(f"need at least 2 seed_* dirs under {base_dir}, found {len(seed_dirs)}")
    out_dir = base_dir / "aggregated"
    report: dict[str, list[str]] = {
        "aggregated": [], "passthrough": [], "skipped_draws": [], "skipped_misaligned": [], "skipped_missing": [],
    }
    apply_style()
    print(f"Aggregating {len(seed_dirs)} seeds under {base_dir}")

    # Union over every seed, not just seed_0: a file only some seeds wrote must
    # still be reported as skipped_missing rather than silently disappearing.
    relpaths = sorted(
        {
            p.relative_to(d)
            for d in seed_dirs
            for pattern in ("**/*.csv", "**/*.json")
            for p in d.glob(pattern)
        }
    )
    for rel in relpaths:
        if _is_skipped(rel):
            report["skipped_draws"].append(str(rel))
            continue
        if any(not (d / rel).exists() for d in seed_dirs):
            report["skipped_missing"].append(str(rel))
            continue
        if rel.suffix == ".csv":
            frames = [pd.read_csv(d / rel) for d in seed_dirs]
            agg = _aggregate_csv(frames)
            if agg is None:
                report["skipped_misaligned"].append(str(rel))
                continue
            varying = "n_seeds" in agg.columns
            header = list(agg.columns)
            _write_csv(out_dir / rel, header, agg.to_numpy(dtype=object).tolist())
            if varying:
                _write_tex_tabular(
                    (out_dir / rel).with_suffix(".tex"), header,
                    [[v if isinstance(v, str) else f"{v:.4f}" if isinstance(v, float) else v for v in row]
                     for row in agg.itertuples(index=False)],
                    generated_by="aggregate_results.py",
                )
                _band_figure(agg, (out_dir / rel).with_suffix("").with_name(rel.stem + "_bands.png"))
            report["aggregated" if varying else "passthrough"].append(str(rel))
        else:
            docs = [json.loads((d / rel).read_text(encoding="utf-8")) for d in seed_dirs]
            _write_json(out_dir / rel, _aggregate_json(docs, len(seed_dirs)))
            report["aggregated" if docs.count(docs[0]) < len(docs) else "passthrough"].append(str(rel))

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    _write_json(out_dir / "run_metadata.json", {
        "seeds": [int(d.name.split("_")[1]) for d in seed_dirs],
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": commit,
    })
    _write_json(out_dir / "aggregation_report.json", report)
    return out_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Aggregate per-seed results into mean±std summaries.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir", type=Path, help="one results/<ts>/<analysis>/<model>/<trait>/<pooling>_token dir")
    g.add_argument("--discover", type=Path, help="results/<ts> root; aggregate every dir containing seed_* runs")
    args = ap.parse_args()

    if args.dir:
        aggregate_analysis(args.dir)
    else:
        bases = sorted({p.parent for p in args.discover.glob("**/seed_*") if p.is_dir()})
        if not bases:
            raise SystemExit(f"no seed_* dirs found under {args.discover}")
        for base in bases:
            aggregate_analysis(base)
