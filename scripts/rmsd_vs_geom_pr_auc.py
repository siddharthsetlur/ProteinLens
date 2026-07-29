"""Per-feature mean motif RMSD stratified by geom_pr_auc decile.

The permutation-null q-value floor (~0.01) is too coarse to discriminate
geom-significant features at q<0.01 — only ~10 unique q values exist.
Instead, stratify by the continuous geom_pr_auc statistic.

Outputs:
    static_plots/rmsd_vs_geom_pr_auc_stacked.png  — RMSD histogram, stacked by decile
    static_plots/rmsd_vs_geom_pr_auc_top10.png    — top decile vs rest
    static_plots/rmsd_vs_geom_pr_auc.csv          — per-bin counts by decile
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = Path(
    os.environ.get(
        "PROTEINLENS_LAYER4_ANALYSIS",
        str(REPO_ROOT / "trained_models" / "layer_4" / "frosty-sweep-15" / "analysis"),
    )
)
GEOM_DIR = ANALYSIS_DIR / "geometry_enrichment"
PRIMARY_JSON = ANALYSIS_DIR / "geometry_primary_analysis.json"
OUT_DIR = ANALYSIS_DIR / "static_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_features() -> list[tuple[int, float, float]]:
    with PRIMARY_JSON.open() as f:
        primary = json.load(f)

    rows: list[tuple[int, float, float]] = []
    missing_rmsd = 0
    no_prauc = 0
    for fid_str, info in primary["features"].items():
        prauc = info.get("geom_pr_auc")
        if prauc is None:
            no_prauc += 1
            continue
        fid = int(fid_str)
        path = GEOM_DIR / f"{fid:04d}.json"
        if not path.exists():
            missing_rmsd += 1
            continue
        with path.open() as f:
            d = json.load(f)
        motif = (d.get("geometric_residue_level") or {}).get("motif_superposition") or {}
        val = motif.get("mean_rmsd")
        if val is None:
            missing_rmsd += 1
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            missing_rmsd += 1
            continue
        if not np.isfinite(v):
            missing_rmsd += 1
            continue
        rows.append((fid, float(prauc), v))
    print(f"features kept: {len(rows)}  (missing rmsd: {missing_rmsd}, no pr_auc: {no_prauc})")
    return rows


def main() -> None:
    rows = load_features()
    fids = np.array([r[0] for r in rows])
    praucs = np.array([r[1] for r in rows])
    rmsds = np.array([r[2] for r in rows])

    # Decile assignment by geom_pr_auc (0 = lowest, 9 = highest).
    deciles = np.searchsorted(np.quantile(praucs, np.linspace(0, 1, 11)[1:-1]), praucs, side="right")
    deciles = np.clip(deciles, 0, 9)

    # Per-decile pr_auc summary.
    print("\ngeom_pr_auc by decile (low → high):")
    for d in range(10):
        mask = deciles == d
        if mask.sum() == 0:
            continue
        print(f"  D{d}: n={mask.sum():4d}  pr_auc range "
              f"[{praucs[mask].min():.4f}, {praucs[mask].max():.4f}]  "
              f"mean RMSD={rmsds[mask].mean():.3f} Å")

    # Histogram bins.
    rmsd_max = float(np.ceil(rmsds.max() * 4) / 4)
    bin_width = 0.25
    bins = np.arange(0.0, rmsd_max + bin_width, bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])

    counts_by_decile = np.zeros((10, len(centers)), dtype=int)
    for d in range(10):
        c, _ = np.histogram(rmsds[deciles == d], bins=bins)
        counts_by_decile[d] = c

    # CSV.
    csv_path = OUT_DIR / "rmsd_vs_geom_pr_auc.csv"
    with csv_path.open("w") as f:
        header = ["rmsd_bin_low", "rmsd_bin_high"] + [f"D{i}" for i in range(10)] + ["total"]
        f.write(",".join(header) + "\n")
        for i in range(len(centers)):
            row = [f"{bins[i]:.3f}", f"{bins[i+1]:.3f}"]
            row += [str(int(counts_by_decile[d, i])) for d in range(10)]
            row.append(str(int(counts_by_decile[:, i].sum())))
            f.write(",".join(row) + "\n")
    print(f"\nwrote {csv_path}")

    # --- Stacked histogram by decile ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("viridis")
    bottom = np.zeros(len(centers))
    for d in range(10):
        ax.bar(
            centers, counts_by_decile[d], width=bin_width * 0.95, bottom=bottom,
            color=cmap(d / 9), edgecolor="black", linewidth=0.3,
            label=f"D{d} (pr_auc {praucs[deciles==d].min():.3f}–{praucs[deciles==d].max():.3f})",
        )
        bottom += counts_by_decile[d]
    ax.set_xlabel("Mean motif RMSD across activating proteins (Å)")
    ax.set_ylabel("Number of features")
    ax.set_title("Layer 4 / frosty-sweep-15 — motif RMSD stacked by geom_pr_auc decile")
    ax.legend(loc="upper right", fontsize=7, ncol=2, title="decile (low → high)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_stack = OUT_DIR / "rmsd_vs_geom_pr_auc_stacked.png"
    fig.savefig(out_stack, dpi=150)
    print(f"wrote {out_stack}")

    # --- Top decile vs rest ---
    top_mask = deciles == 9
    top_counts, _ = np.histogram(rmsds[top_mask], bins=bins)
    rest_counts, _ = np.histogram(rmsds[~top_mask], bins=bins)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(centers, rest_counts, width=bin_width * 0.95,
           color="lightgrey", edgecolor="grey",
           label=f"D0–D8 (n={int((~top_mask).sum())})")
    ax.bar(centers, top_counts, width=bin_width * 0.95, bottom=rest_counts,
           color="#d62728", edgecolor="black",
           label=f"D9 — top 10% pr_auc (n={int(top_mask.sum())}, "
                 f"≥{praucs[top_mask].min():.3f})")
    ax.set_xlabel("Mean motif RMSD across activating proteins (Å)")
    ax.set_ylabel("Number of features")
    ax.set_title("Layer 4 / frosty-sweep-15 — top-decile geom_pr_auc features vs rest")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_top = OUT_DIR / "rmsd_vs_geom_pr_auc_top10.png"
    fig.savefig(out_top, dpi=150)
    print(f"wrote {out_top}")

    # Top-decile RMSD summary.
    print(f"\nTop decile (n={int(top_mask.sum())}, pr_auc≥{praucs[top_mask].min():.4f}):")
    print(f"  RMSD mean={rmsds[top_mask].mean():.3f}  median={np.median(rmsds[top_mask]):.3f}  "
          f"min={rmsds[top_mask].min():.3f}  max={rmsds[top_mask].max():.3f}")
    print(f"Rest D0–D8 (n={int((~top_mask).sum())}):")
    print(f"  RMSD mean={rmsds[~top_mask].mean():.3f}  median={np.median(rmsds[~top_mask]):.3f}")


if __name__ == "__main__":
    main()
