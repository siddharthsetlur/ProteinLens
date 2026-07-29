"""Per-feature mean motif RMSD vs geometry q-value (layer 4 frosty-sweep-15).

Pulls `motif_superposition.mean_rmsd` from each geometry_enrichment/{fid}.json
and `geometry_prauc_padj` from geometry_primary_analysis.json, then plots a
histogram of mean RMSD restricted to features with q < 0.05.
"""

from __future__ import annotations

import argparse
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


def load_qvalues() -> dict[int, float]:
    with PRIMARY_JSON.open() as f:
        primary = json.load(f)
    out: dict[int, float] = {}
    for fid_str, info in primary["features"].items():
        q = info.get("geometry_prauc_padj")
        if q is None:
            continue
        out[int(fid_str)] = float(q)
    return out


def load_mean_rmsd(fid: int) -> float | None:
    path = GEOM_DIR / f"{fid:04d}.json"
    if not path.exists():
        return None
    with path.open() as f:
        d = json.load(f)
    motif = (d.get("geometric_residue_level") or {}).get("motif_superposition") or {}
    val = motif.get("mean_rmsd")
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, default=0.05, help="q-value cutoff (BH on permutation null)")
    args = parser.parse_args()
    q_threshold = args.q
    tag = f"q{q_threshold:g}".replace(".", "p")

    qvals = load_qvalues()
    print(f"features with q-value: {len(qvals)}")

    rows: list[tuple[int, float, float]] = []  # fid, q, rmsd
    missing_rmsd = 0
    for fid, q in qvals.items():
        rmsd = load_mean_rmsd(fid)
        if rmsd is None:
            missing_rmsd += 1
            continue
        rows.append((fid, q, rmsd))
    print(f"features with both q and mean_rmsd: {len(rows)} (missing rmsd: {missing_rmsd})")

    fids = np.array([r[0] for r in rows])
    qs = np.array([r[1] for r in rows])
    rmsds = np.array([r[2] for r in rows])

    sig_mask = qs < q_threshold
    print(f"features sig (q<{q_threshold}): {int(sig_mask.sum())}/{len(rows)}")
    print(f"sig RMSD: mean={rmsds[sig_mask].mean():.3f} median={np.median(rmsds[sig_mask]):.3f} "
          f"min={rmsds[sig_mask].min():.3f} max={rmsds[sig_mask].max():.3f}")

    # Histogram bins: 0 to ceil of max RMSD, 0.25 Å width.
    rmsd_max = float(np.ceil(rmsds.max() * 4) / 4)
    bin_width = 0.25
    bins = np.arange(0.0, rmsd_max + bin_width, bin_width)

    sig_counts, _ = np.histogram(rmsds[sig_mask], bins=bins)
    all_counts, _ = np.histogram(rmsds, bins=bins)

    # CSV dump for reference.
    csv_path = OUT_DIR / f"rmsd_vs_geom_qvalue_{tag}.csv"
    with csv_path.open("w") as f:
        f.write(f"rmsd_bin_low,rmsd_bin_high,n_sig_q_lt_{q_threshold:g},n_total\n")
        for i in range(len(sig_counts)):
            f.write(f"{bins[i]:.3f},{bins[i+1]:.3f},{sig_counts[i]},{all_counts[i]}\n")
    print(f"wrote {csv_path}")

    # Plot.
    fig, ax = plt.subplots(figsize=(9, 5))
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.bar(centers, all_counts, width=bin_width * 0.95, color="lightgrey",
           edgecolor="grey", label=f"all features (n={len(rows)})")
    ax.bar(centers, sig_counts, width=bin_width * 0.95, color="#1f77b4",
           edgecolor="black", label=f"q<{q_threshold} (n={int(sig_mask.sum())})")
    ax.set_xlabel("Mean motif RMSD across activating proteins (Å)")
    ax.set_ylabel("Number of features")
    ax.set_title("Layer 4 / frosty-sweep-15 — geometry-significant features by motif RMSD")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = OUT_DIR / f"rmsd_vs_geom_qvalue_{tag}.png"
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
