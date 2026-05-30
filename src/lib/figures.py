"""Plotting helpers.

Centralises the Matplotlib style, the colorblind-safe palette, and every
figure layout the analysis scripts reuse — keeps the headline figures
visually consistent and the scripts free of one-off plotting code.

Conventions
-----------
* Ordinal trait levels (negative / neutral / positive) always render with
  :data:`C_NEG` / :data:`C_NEUT` / :data:`C_POS`; use :func:`levels_palette`
  to recover the per-level colour list.
* Every builder returns ``(fig, ax)`` (or ``(fig, axes)``) so the caller
  can save / further customise.
* :func:`apply_style` is idempotent; scripts call it once on entry.

Typical usage::

    from lib.figures import apply_style, plot_agreement, plot_layer_sweep
    apply_style()
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# -- Palette (Wong, 2011 inspired) -------------------------------------------
C_POS  = "#169873"   # positive class / supervised-method line
C_NEG  = "#FF5154"   # negative class
C_NEUT = "#FFC65C"   # neutral / cross-method curves
C_REF  = "#00A3F5"   # reference / secondary
C_NULL = "#372772"   # null / chance baseline


# -- Global style -------------------------------------------------------------
def apply_style() -> None:
    """Apply publication-style Matplotlib defaults. Idempotent."""
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.titleweight": "regular",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# -- Primitives (draw on a given axis) ---------------------------------------
def _draw_agreement(ax, M, names, title):
    """Annotated cosine-similarity heatmap on `ax`. Returns the AxesImage."""
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i, j in np.ndindex(M.shape):
        ax.text(
            j, i, f"{M[i, j]:.2f}",
            ha="center", va="center",
            color="white" if abs(M[i, j]) > 0.6 else "black",
            fontsize=7,
        )
    ax.set_title(title)
    return im


def _draw_projection(ax, proj, y, neg_label, pos_label, title, bins=30):
    """Two-class projection histogram on `ax`."""
    ax.hist(proj[y == 0], bins=bins, alpha=0.75, color=C_NEG,
            label=neg_label, edgecolor="white", linewidth=0.4)
    ax.hist(proj[y == 1], bins=bins, alpha=0.75, color=C_POS,
            label=pos_label, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("projection onto Mean-Difference axis")
    ax.set_title(title)


def _add_cosine_colorbar(fig, mappable, axes, *, fraction=0.045, pad=0.03):
    cbar = fig.colorbar(mappable, ax=axes, fraction=fraction, pad=pad)
    cbar.set_label("cosine similarity")
    cbar.ax.tick_params(labelsize=7)


# -- Figure builders ---------------------------------------------------------
def plot_agreement(M, names, title, *, figsize=(5, 4.5)):
    """Single annotated direction-agreement heatmap with a colorbar."""
    fig, ax = plt.subplots(figsize=figsize)
    im = _draw_agreement(ax, M, names, title)
    _add_cosine_colorbar(fig, im, ax)
    fig.tight_layout()
    return fig, ax


def plot_agreement_pair(matrices, titles, names, *, suptitle="", figsize=(10, 4.5)):
    """Two side-by-side direction-agreement heatmaps with a shared colorbar."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    im = None
    for ax, M, title in zip(axes, matrices, titles):
        im = _draw_agreement(ax, M, names, title)
    _add_cosine_colorbar(fig, im, axes, fraction=0.035, pad=0.04)
    if suptitle:
        fig.suptitle(suptitle, y=1.04)
    return fig, axes


def plot_projection(proj, y, *, neg_label, pos_label, title, figsize=(5, 3.5)):
    """Single two-class projection histogram (Mean-Difference axis)."""
    fig, ax = plt.subplots(figsize=figsize)
    _draw_projection(ax, proj, y, neg_label, pos_label, title)
    ax.set_ylabel("scenarios")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_projection_pair(panels, titles, *, neg_label, pos_label,
                         suptitle="", figsize=(10, 3.5)):
    """Two side-by-side projection histograms (shared y-axis).

    Args:
        panels:  list of (proj, y) tuples (typically raw and centered).
        titles:  list of subplot titles (typically with AUC/d annotations).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, (proj, y), title in zip(axes, panels, titles):
        _draw_projection(ax, proj, y, neg_label, pos_label, title)
    axes[0].set_ylabel("scenarios")
    axes[1].legend(loc="upper left")
    if suptitle:
        fig.suptitle(suptitle, y=1.05)
    fig.tight_layout()
    return fig, axes


def plot_layer_sweep(layers, accuracy, agreement, *, focal_layer, figsize=(5.4, 2.8)):
    """Direction-quality summary (accuracy + cross-method cosine) across layers."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(layers, accuracy, "o-", color=C_POS, label="MeanDiff CV balanced acc.")
    ax.plot(layers, agreement, "s--", color=C_NEUT, label="mean cross-method cosine")
    ax.axvline(focal_layer, color=C_NULL, ls=":", lw=0.8,
               label=f"focal layer {focal_layer}")
    ax.axhline(0.5, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Linear positive/negative structure across layers")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, ax


# -- Ordinal-level palette ---------------------------------------------------
_LEVEL_COLORS = {"negative": C_NEG, "neutral": C_NEUT, "positive": C_POS}


def levels_palette(levels):
    """Per-level colours; unknown labels fall back to :data:`C_REF`."""
    return [_LEVEL_COLORS.get(lvl, C_REF) for lvl in levels]


# -- Multi-line layer sweep --------------------------------------------------
def plot_pair_sweep(layers, pair_sims, pair_labels, *, focal_layer=None,
                    title="", figsize=(7.5, 3.4)):
    """One line per (difference-vector) pair, cosine vs layer.

    Args:
        pair_sims:    list of 1-D sequences, one per pair, indexed by ``layers``.
        pair_labels:  legend label per pair (same order).
    """
    fig, ax = plt.subplots(figsize=figsize)
    palette = [C_POS, C_REF, C_NEUT, C_NEG, C_NULL]
    for i, (sims, label) in enumerate(zip(pair_sims, pair_labels)):
        ax.plot(layers, sims, marker="o", color=palette[i % len(palette)], label=label)
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    if focal_layer is not None:
        ax.axvline(focal_layer, color=C_NULL, ls=":", lw=0.8,
                   label=f"focal layer {focal_layer}")
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(-1.02, 1.02)
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", ncol=1)
    fig.tight_layout()
    return fig, ax


# -- Grid of annotated cosine matrices ---------------------------------------
def plot_cosine_grid(matrices, col_titles, row_titles, names, *,
                     suptitle="", cell_size=3.0):
    """Grid of annotated cosine heatmaps with one shared colorbar.

    Tick labels render only on the bottom row and leftmost column; the
    matrices share the same axes everywhere else, so repeating them just
    crowds the cells.

    Args:
        matrices:    nested list ``matrices[row][col]`` of square arrays.
        col_titles:  one title per column.
        row_titles:  one label per row (drawn on the leftmost ylabel).
        names:       axis tick labels for every cell.
    """
    nrows, ncols = len(matrices), len(matrices[0])
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(cell_size * ncols + 1.6, cell_size * nrows + 0.8),
        squeeze=False,
    )
    im = None
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            title = col_titles[c] if r == 0 else ""
            im = _draw_agreement(ax, matrices[r][c], names, title)
            if r == nrows - 1:
                ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_yticklabels(names, fontsize=8)
                ax.set_ylabel(row_titles[r], fontsize=10, fontweight="bold")
            else:
                ax.set_yticklabels([])
    _add_cosine_colorbar(fig, im, axes.ravel().tolist(), fraction=0.02, pad=0.02)
    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    return fig, axes


# -- Grouped bar chart -------------------------------------------------------
def plot_bar_pair(categories, series_a, series_b, *,
                  series_labels=("A", "B"), ylabel="value", title="",
                  ylim=(-1.05, 1.05), figsize=(7.5, 3.6)):
    """Two-series grouped bar chart with value labels."""
    x = np.arange(len(categories))
    bw = 0.38
    fig, ax = plt.subplots(figsize=figsize)
    b1 = ax.bar(x - bw / 2, series_a, bw, label=series_labels[0], color=C_REF)
    b2 = ax.bar(x + bw / 2, series_b, bw, label=series_labels[1], color=C_NEG)
    ax.bar_label(b1, fmt="%.2f", fontsize=7, padding=2)
    ax.bar_label(b2, fmt="%.2f", fontsize=7, padding=2)
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, ax


# -- 2D scatter primitives ---------------------------------------------------
def _draw_2d_levels(ax, coords, labels, centroids, levels_present, palette,
                    *, show_legend, title):
    """Per-level scatter + black-edged centroids + chained centroid arrows."""
    for k, lvl in enumerate(levels_present):
        idx = [i for i, l in enumerate(labels) if l == lvl]
        if not idx:
            continue
        ax.scatter(coords[idx, 0], coords[idx, 1], alpha=0.35, s=16,
                   color=palette[k], label=lvl if show_legend else None,
                   edgecolors="none")
        ax.scatter(*centroids[k], s=140, color=palette[k],
                   edgecolors="black", linewidths=1.0, zorder=5)
    for i in range(len(levels_present) - 1):
        ax.annotate("", xy=centroids[i + 1], xytext=centroids[i],
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ax.set_title(title)


def plot_pca_pair(panels, *, levels_present, suptitle="", figsize=(10.5, 4.4)):
    """Two PCA scatter panels (typically raw vs within-centered).

    Each panel is a dict with keys: ``coords`` (N, 2), ``labels`` (length N),
    ``centroids`` (L, 2), ``explained_var`` (e1, e2), ``title``.
    """
    palette = levels_palette(levels_present)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    for j, (ax, panel) in enumerate(zip(axes, panels)):
        e1, e2 = panel["explained_var"]
        _draw_2d_levels(
            ax, panel["coords"], panel["labels"], panel["centroids"],
            levels_present, palette,
            show_legend=(j == 0), title=panel["title"],
        )
        ax.set_xlabel(f"PC1 ({e1:.1%} variance)")
        ax.set_ylabel(f"PC2 ({e2:.1%} variance)")
    axes[0].legend(loc="upper left")
    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    return fig, axes


# -- Per-scenario small multiples --------------------------------------------
def plot_scenario_grid(scenario_xy, scenario_titles, levels_present, *,
                       xlabel="axis 1", ylabel="axis 2",
                       suptitle="", ncols=4, panel_size=(2.9, 2.4)):
    """One panel per scenario in a shared 2D plane (e.g. axis × orthogonal).

    Args:
        scenario_xy:     list of ``(xs, ys)`` length-L sequences, one per scenario.
        scenario_titles: short string per scenario.
        levels_present:  ordered level labels (used for colour + legend).
    """
    palette = levels_palette(levels_present)
    n = len(scenario_xy)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * panel_size[0], nrows * panel_size[1]),
        sharex=True, sharey=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (xs, ys), title in zip(axes_flat, scenario_xy, scenario_titles):
        for k, lvl in enumerate(levels_present):
            ax.scatter(xs[k], ys[k], color=palette[k], s=60, zorder=3,
                       edgecolors="black", linewidths=0.6)
        for i in range(len(levels_present) - 1):
            ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.0))
        ax.set_title(title, fontsize=7, pad=2)
        ax.axhline(0, color="black", lw=0.4, alpha=0.3)
        ax.axvline(0, color="black", lw=0.4, alpha=0.3)
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Shared axis labels on the outer edge.
    for ax in axes_flat[:n]:
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel(xlabel)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(ylabel)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=palette[k],
                   markeredgecolor="black", markeredgewidth=0.6, label=lvl)
        for k, lvl in enumerate(levels_present)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(levels_present),
               bbox_to_anchor=(0.5, -0.02))
    if suptitle:
        fig.suptitle(suptitle, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    return fig, axes


# -- Histogram panels for a permutation null ---------------------------------
def plot_null_panels(test_keys, null_data, observed, *,
                     panel_titles=None, suptitle="", figsize=None):
    """One histogram per metric with the observed value drawn as a red line."""
    n = len(test_keys)
    figsize = figsize or (max(3.0 * n, 6.0), 2.8)
    panel_titles = panel_titles or list(test_keys)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    axes = np.atleast_1d(axes)
    for ax, key, title in zip(axes, test_keys, panel_titles):
        obs_val = observed.get(key, float("nan"))
        if isinstance(obs_val, float) and np.isnan(obs_val):
            ax.set_title(f"{title}\n(n/a)")
            ax.axis("off")
            continue
        ax.hist(null_data[key], bins=25, color="lightgray", edgecolor="gray")
        ax.axvline(obs_val, color=C_NEG, lw=1.6, label=f"obs = {obs_val:+.3f}")
        ax.set_title(title)
        ax.legend(loc="best")
    if suptitle:
        fig.suptitle(suptitle, y=1.04)
    fig.tight_layout()
    return fig, axes


# -- Metric-sweep panels (2 × N) --------------------------------------------
def plot_metric_sweep_panels(layers, panels, *, focal_layer=None,
                             suptitle="", ncols=3, panel_size=(4.0, 2.6)):
    """Panel grid of (metric vs layer); optional per-layer IQR band per panel.

    Each panel is a dict with keys:
        ``values`` (sequence, len(layers)), ``title`` (str),
        ``ylim`` (tuple or None), ``band`` (None or ``(q25_arr, q75_arr)``).
    """
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * panel_size[0], nrows * panel_size[1]),
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, panel in zip(axes_flat, panels):
        ax.plot(layers, panel["values"], marker="o", color=C_POS, lw=1.4)
        band = panel.get("band")
        if band is not None:
            q25, q75 = band
            ax.fill_between(layers, q25, q75, alpha=0.18, color=C_POS)
        if focal_layer is not None:
            ax.axvline(focal_layer, color=C_NULL, ls=":", lw=0.8)
        ax.set_xlabel("layer")
        ax.set_title(panel["title"])
        ylim = panel.get("ylim")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, lw=0.4)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    return fig, axes
