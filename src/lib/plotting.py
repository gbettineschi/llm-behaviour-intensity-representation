"""Notebook plotting helpers.

Centralises the Matplotlib style, the colorblind-safe palette, and the few
figure layouts that the replication notebook reuses (single + paired
direction-agreement heatmaps, single + paired projection histograms, the
layer-sweep line plot).

Typical usage:

    from lib.plotting import (
        apply_style, C_POS, C_NEG, C_NEUT, C_REF, C_NULL,
        plot_agreement, plot_agreement_pair,
        plot_projection, plot_projection_pair,
        plot_layer_sweep,
    )
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
