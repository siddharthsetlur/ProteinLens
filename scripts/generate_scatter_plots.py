#!/usr/bin/env python3
"""Generate publication-quality scatter plots with KDE density coloring.

Reads the feature index data and geometry-primary analysis, then produces
seaborn scatter plots with density shading for each enrichment comparison.

Outputs PNGs to ``{data_dir}/static_plots/``.

Usage::

    python scripts/generate_scatter_plots.py --data-dir feature_data_cluster
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde


def _load_index(data_dir: Path) -> list[dict]:
    """Build feature index in-process (avoids needing the viz server)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from proteinlens.viz.index_builder import build_feature_index
    return build_feature_index(data_dir)


def _density_scatter(
    ax, x, y,
    cmap="mako_r",
    s=12,
    alpha=0.8,
    xlabel="",
    ylabel="",
    title="",
    vline=None,
    hline=None,
):
    """Scatter plot with points colored by local KDE density."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 10:
        return

    # KDE density estimation
    xy = np.vstack([x, y])
    try:
        kde = gaussian_kde(xy, bw_method=0.15)
        density = kde(xy)
    except np.linalg.LinAlgError:
        density = np.ones(len(x))

    # Sort by density so dense points render on top
    order = density.argsort()
    x, y, density = x[order], y[order], density[order]

    sc = ax.scatter(x, y, c=density, s=s, cmap=cmap, alpha=alpha,
                    edgecolors="none", rasterized=True)
    plt.colorbar(sc, ax=ax, label="Density", shrink=0.8, pad=0.02)

    if vline is not None:
        ax.axvline(vline, color="#adb5bd", ls="--", lw=0.8, zorder=0)
    if hline is not None:
        ax.axhline(hline, color="#adb5bd", ls="--", lw=0.8, zorder=0)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")


def _geom_primary_scatter(
    ax, index, gp_data,
    xlabel="Geometry PR-AUC",
    ylabel="Best Sequence F1",
):
    """Geometry-primary highlighted scatter with density for background."""
    gp_features = gp_data.get("features", {})

    x_all, y_all, is_primary = [], [], []
    for r in index:
        pr_auc = r.get("geometry_residue_pr_auc")
        if pr_auc is None:
            continue
        seq_f1s = [
            v for v in [
                r.get("motif_best_pr_auc"),
                r.get("position_best_f1"),
                r.get("interpro_residue_best_f1"),
            ] if v is not None
        ]
        best_seq = max(seq_f1s) if seq_f1s else 0
        fid_str = str(r["feature_id"])
        primary = gp_features.get(fid_str, {}).get("is_geometry_primary", False)
        x_all.append(pr_auc)
        y_all.append(best_seq)
        is_primary.append(primary)

    x_all = np.array(x_all)
    y_all = np.array(y_all)
    is_primary = np.array(is_primary)

    # Background points with density coloring
    bg_mask = ~is_primary
    if bg_mask.sum() > 10:
        xy_bg = np.vstack([x_all[bg_mask], y_all[bg_mask]])
        try:
            kde = gaussian_kde(xy_bg, bw_method=0.15)
            dens = kde(xy_bg)
        except np.linalg.LinAlgError:
            dens = np.ones(bg_mask.sum())
        order = dens.argsort()
        ax.scatter(
            x_all[bg_mask][order], y_all[bg_mask][order],
            c=dens[order], s=10, cmap="Greys", alpha=0.5,
            edgecolors="none", rasterized=True,
        )

    # Geometry-primary points
    n_primary = is_primary.sum()
    ax.scatter(
        x_all[is_primary], y_all[is_primary],
        c="#f59f00", s=30, alpha=0.9, edgecolors="#c77c00", linewidths=0.5,
        label=f"Geometry-primary ({n_primary})", zorder=5,
    )

    ax.axvline(0.3, color="#adb5bd", ls="--", lw=0.8, zorder=0)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f"Geometry PR-AUC vs Best Sequence F1 ({n_primary} geometry-primary)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("feature_data_cluster"),
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    print("Loading feature index ...")
    index = _load_index(data_dir)

    gp_path = data_dir / "geometry_primary_analysis.json"
    gp_data = json.loads(gp_path.read_text()) if gp_path.exists() else {}

    out_dir = data_dir / "static_plots"
    out_dir.mkdir(exist_ok=True)

    # Set seaborn style
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })

    # Helper to extract arrays
    def _get_xy(xfield, yfield):
        xs, ys = [], []
        for r in index:
            xv, yv = r.get(xfield), r.get(yfield)
            if xv is not None and yv is not None:
                xs.append(xv)
                ys.append(yv)
        return np.array(xs), np.array(ys)

    # ── Plot definitions ──
    plots = [
        {
            "filename": "geom_vs_interpro_prot.png",
            "xfield": "geometry_residue_pr_auc",
            "yfield": "interpro_protein_best_f1",
            "xlabel": "Geometry PR-AUC",
            "ylabel": "InterPro Protein F1",
            "title": "Geometry PR-AUC vs InterPro Protein F1",
            "cmap": "mako_r",
        },
        {
            "filename": "geom_vs_interpro_res.png",
            "xfield": "geometry_residue_pr_auc",
            "yfield": "interpro_residue_best_f1",
            "xlabel": "Geometry PR-AUC",
            "ylabel": "InterPro Residue F1",
            "title": "Geometry PR-AUC vs InterPro Residue F1",
            "cmap": "rocket_r",
        },
        {
            "filename": "geom_vs_motif.png",
            "xfield": "geometry_residue_pr_auc",
            "yfield": "motif_best_pr_auc",
            "xlabel": "Geometry PR-AUC",
            "ylabel": "Motif PR-AUC",
            "title": "Geometry PR-AUC vs Motif PR-AUC",
            "cmap": "viridis",
        },
        {
            "filename": "geom_vs_position.png",
            "xfield": "geometry_residue_pr_auc",
            "yfield": "position_best_f1",
            "xlabel": "Geometry PR-AUC",
            "ylabel": "Position F1",
            "title": "Geometry PR-AUC vs Position F1",
            "cmap": "flare",
        },
    ]

    # Generate density scatter plots
    for p in plots:
        x, y = _get_xy(p["xfield"], p["yfield"])
        fig, ax = plt.subplots(figsize=(6, 5))
        _density_scatter(
            ax, x, y,
            cmap=p["cmap"],
            xlabel=p["xlabel"],
            ylabel=p["ylabel"],
            title=p["title"],
        )
        sns.despine(ax=ax)
        fig.savefig(out_dir / p["filename"])
        plt.close(fig)
        print(f"  Wrote {p['filename']} ({len(x)} points)")

    # Best F1 plot
    x_best, y_best = [], []
    for r in index:
        xv = r.get("geometry_residue_pr_auc")
        if xv is None:
            continue
        f1s = [
            v for v in [
                r.get("interpro_protein_best_f1"),
                r.get("interpro_residue_best_f1"),
                r.get("motif_best_pr_auc"),
                r.get("position_best_f1"),
            ] if v is not None
        ]
        if f1s:
            x_best.append(xv)
            y_best.append(max(f1s))

    fig, ax = plt.subplots(figsize=(6, 5))
    _density_scatter(
        ax, np.array(x_best), np.array(y_best),
        cmap="crest",
        xlabel="Geometry PR-AUC",
        ylabel="Best F1 (max of all sequence metrics)",
        title="Geometry PR-AUC vs Best F1",
    )
    sns.despine(ax=ax)
    fig.savefig(out_dir / "geom_vs_best_f1.png")
    plt.close(fig)
    print(f"  Wrote geom_vs_best_f1.png ({len(x_best)} points)")

    # Geometry-primary scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    _geom_primary_scatter(ax, index, gp_data)
    sns.despine(ax=ax)
    fig.savefig(out_dir / "geom_primary.png")
    plt.close(fig)
    print(f"  Wrote geom_primary.png")

    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    main()
