from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pdb_plotter import ca_backbone


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Optimally rotate + translate *mobile* onto *target* using the Kabsch
    algorithm.  Both inputs are (N, 3) Ca coordinate arrays.

    When the two backbones have different lengths, the shorter length is used
    (aligned from the N-terminus).

    Returns the transformed mobile coordinates (same length as input mobile).
    """
    n = min(len(mobile), len(target))
    P = mobile[:n].copy()
    Q = target[:n].copy()

    # Centre both
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    P -= p_mean
    Q -= q_mean

    # Cross-covariance matrix
    H = P.T @ Q  # (3, 3)

    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])

    R = Vt.T @ sign_matrix @ U.T  # optimal rotation

    # Apply to *all* of mobile (not just the truncated prefix)
    aligned = (mobile - p_mean) @ R.T + q_mean
    return aligned


def compute_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD between two (N,3) coordinate sets (uses min length)."""
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    return float(np.sqrt((diff * diff).sum() / n))


# ─── Loading helpers ─────────────────────────────────────────────────────────

def load_ca_from_cache(acc: str, pdb_cache: Path) -> np.ndarray | None:
    """Load Cα backbone from a cached AlphaFold PDB file."""
    cached = list(pdb_cache.glob(f"AF-{acc}-F1-model_v*.pdb"))
    if not cached:
        return None
    try:
        pdb_text = cached[0].read_text()
        ca = ca_backbone(pdb_text, chain_id=None)
        plt.close("all")
        return ca
    except Exception:
        return None


# ─── Plotting ────────────────────────────────────────────────────────────────

COLOURS = ["#2980b9", "#e74c3c", "#27ae60", "#f39c12", "#8e44ad",
           "#1abc9c", "#d35400", "#c0392b", "#2c3e50", "#16a085"]


def plot_aligned_backbones(
    coords_list: list[np.ndarray],
    labels: list[str],
    title: str = "",
    save_path: Path | None = None,
):
    """
    Plot several Cα backbones (already aligned) overlaid in 3D.
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i, (coords, label) in enumerate(zip(coords_list, labels)):
        c = COLOURS[i % len(COLOURS)]
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                color=c, linewidth=1.4, alpha=0.85, label=label)

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_single_backbone(
    coords: np.ndarray,
    title: str = "",
    save_path: Path | None = None,
    colour: str = "#2980b9",
):
    """
    Plot a single Cα backbone in 3D and save to *save_path*.
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
            color=colour, linewidth=1.6, alpha=0.9)
    # Mark N- and C-termini
    ax.scatter(*coords[0], color="#27ae60", s=60, zorder=5, label="N-term")
    ax.scatter(*coords[-1], color="#e74c3c", s=60, zorder=5, label="C-term")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ─── Main entry point ────────────────────────────────────────────────────────

def plot_kabsch_alignments(
    summary: list[dict],
    top_k_map: dict[int, list[str]],
    pdb_cache: Path,
    save_dir: Path,
    n_proteins: int = 5,
):
    """
    For each geometric feature (deduplicated from summary), take the best
    SAE node, grab its top-N activating proteins, Kabsch-align them, and
    save an overlay plot.

    Parameters
    ----------
    summary : list[dict]
        Correlation summary entries (must have 'geom_feature', 'sae_node',
        'pearson_r', 'spearman_r').
    top_k_map : dict[int, list[str]]
        {node_id: [acc1, acc2, …]} — top-activating accessions per node.
    pdb_cache : Path
        Directory containing cached AlphaFold PDB files.
    save_dir : Path
        Where to write the overlay PNGs.
    n_proteins : int
        How many top-activating proteins to overlay (default 5).
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate: best node per feature
    best_per_feat: dict[str, dict] = {}
    for entry in summary:
        feat = entry["geom_feature"]
        if feat not in best_per_feat or abs(entry["pearson_r"]) > abs(best_per_feat[feat]["pearson_r"]):
            best_per_feat[feat] = entry

    ordered = sorted(best_per_feat.values(),
                     key=lambda d: abs(d["pearson_r"]), reverse=True)

    n_plotted = 0
    n_skipped = 0

    for entry in ordered:
        feat = entry["geom_feature"]
        node = entry["sae_node"]
        accs = top_k_map.get(node, [])[:n_proteins]

        if len(accs) < 2:
            n_skipped += 1
            continue

        # Load backbones
        backbones: list[tuple[str, np.ndarray]] = []
        for acc in accs:
            ca = load_ca_from_cache(acc, pdb_cache)
            if ca is not None and len(ca) >= 4:
                backbones.append((acc, ca))

        if len(backbones) < 2:
            n_skipped += 1
            continue

        safe_name = feat.replace("/", "_").replace(" ", "_")

        # ── Individual structure plots ────────────────────────────────
        indiv_dir = save_dir / f"{safe_name}_node{node}"
        indiv_dir.mkdir(parents=True, exist_ok=True)
        for rank, (acc, ca) in enumerate(backbones, 1):
            plot_single_backbone(
                ca,
                title=f"{acc}  (rank {rank}, {feat}, node {node})",
                save_path=indiv_dir / f"{rank}_{acc}.png",
                colour=COLOURS[(rank - 1) % len(COLOURS)],
            )

        # ── Kabsch-aligned overlay ────────────────────────────────────
        ref_acc, ref_coords = backbones[0]
        aligned_list = [ref_coords]
        labels = [f"{ref_acc} (ref)"]

        for acc, coords in backbones[1:]:
            aligned = kabsch_align(coords, ref_coords)
            aligned_list.append(aligned)
            rmsd = compute_rmsd(aligned, ref_coords)
            labels.append(f"{acc} (RMSD={rmsd:.1f}Å)")

        title = (
            f"{feat} → node {node}  "
            f"(r={entry['pearson_r']:.3f}, ρ={entry['spearman_r']:.3f})"
        )
        save_path = save_dir / f"kabsch_{safe_name}_node{node}.png"

        plot_aligned_backbones(aligned_list, labels, title=title,
                               save_path=save_path)
        n_plotted += 1

    print(f"  Kabsch alignment plots: {n_plotted} saved, {n_skipped} skipped "
          f"(not enough proteins).")
    print(f"  Saved to → {save_dir}/")
