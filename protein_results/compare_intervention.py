#!/usr/bin/env python
"""
Post-intervention comparison: geometry & DT-predicted activations.

After running build_residue_motifs.py and intervene_and_fold.py, this
script loads the original and intervened (ESMFold) PDB structures and
compares:

  1. Per-residue DT/GBM-predicted activation probability (before vs after)
  2. Per-residue geometric profiles (curvature, torsion, planarity)
  3. Kabsch-aligned 3D backbone overlay (original vs intervened)
  4. Concordance-style dual-panel overlay (matching the geometry_overlay
     plot style from build_residue_motifs.py)

Usage
-----
  # Compare a single node — the original AlphaFold PDB is fetched
  # automatically from the AlphaFold database:
  python compare_intervention.py \\
      --accession A2AU72 \\
      --intervened-pdb path/to/intervened_esmfold.pdb \\
      --motif-dir     residue_motifs \\
      --node 670 \\
      --output-dir    intervention_comparison

  # You can also provide the results.yaml from intervene_and_fold.py
  # to automatically annotate which interventions were applied:
  python compare_intervention.py \\
      --accession P00805 \\
      --intervened-pdb path/to/intervened_esmfold.pdb \\
      --motif-dir     residue_motifs \\
      --node 670 \\
      --intervention-results results/experiment_01/results.yaml \\
      --output-dir    intervention_comparison

  # If the two sequences differ in length, alignment is truncated
  # to the shorter chain (from the N-terminus).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import requests
import yaml

# ── Project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdb_plotter import ca_backbone, detect_alpha_helices_from_ca
from kabsch_top_alignment import kabsch_align, compute_rmsd
from geometry.compute_geometric_features import (
    ca_curvature_profile,
    ca_torsion_profile,
    local_planarity_profile,
    tangent_vectors,
    writhe,
    average_curvature,
    average_torsion,
    radius_of_gyration,
    end_to_end_distance,
)
from protein_results.build_residue_motifs import (
    compute_residue_profiles,
    extract_local_feature_vector,
    LOCAL_GEOM_NAMES,
    HALF_W,
    CATEGORY_NAMES,
)


# ── Style constants (matching build_residue_motifs / kabsch_top_alignment) ──
COLOURS = [
    "#2980b9", "#e74c3c", "#27ae60", "#f39c12", "#8e44ad",
    "#1abc9c", "#d35400", "#c0392b", "#2c3e50", "#16a085",
]

ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_alphafold_pdb(accession: str, cache_dir: Path) -> str:
    """Fetch the AlphaFold PDB for a UniProt accession.

    Downloads from the AlphaFold EBI API and caches on disk so
    subsequent runs are instant.

    Returns the PDB text string.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache first
    cached = list(cache_dir.glob(f"AF-{accession}-F1-model_v*.pdb"))
    if cached:
        print(f"      Using cached PDB: {cached[0].name}")
        return cached[0].read_text()

    session = requests.Session()

    # Query AlphaFold API for the PDB URL
    api_url = ALPHAFOLD_API_URL.format(acc=accession)
    r = session.get(api_url, timeout=15)
    if r.status_code != 200:
        raise ValueError(
            f"AlphaFold API returned HTTP {r.status_code} for '{accession}'. "
            f"Check the accession is valid."
        )
    data = r.json()
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(f"No AlphaFold prediction found for '{accession}'.")
        data = data[0]
    pdb_url = data.get("pdbUrl")
    if not pdb_url:
        raise ValueError(f"No PDB URL in AlphaFold response for '{accession}'.")

    # Download the PDB
    r = session.get(pdb_url, timeout=30)
    if r.status_code != 200:
        raise ValueError(f"Failed to download PDB from {pdb_url} (HTTP {r.status_code}).")
    pdb_text = r.text

    # Cache it
    fname = pdb_url.rsplit("/", 1)[-1]
    (cache_dir / fname).write_text(pdb_text)
    print(f"      Downloaded & cached: {fname}")
    return pdb_text


def load_pdb_ca(pdb_path: Path, chain_id: str | None = None) -> np.ndarray:
    """Load Cα coordinates from a PDB file."""
    pdb_text = pdb_path.read_text()
    ca = ca_backbone(pdb_text, chain_id=chain_id)
    plt.close("all")
    return ca


def load_pdb_ca_from_text(pdb_text: str, chain_id: str | None = None) -> np.ndarray:
    """Load Cα coordinates from a PDB text string."""
    ca = ca_backbone(pdb_text, chain_id=chain_id)
    plt.close("all")
    return ca


def get_sequence_from_pdb_text(pdb_text: str, chain_id: str | None = None) -> str:
    """Extract one-letter amino-acid sequence from PDB text."""
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    from io import StringIO
    from Bio.PDB import PDBParser
    cleaned = "\n".join(
        l for l in pdb_text.splitlines()
        if l.startswith(("ATOM", "HETATM", "TER", "END"))
    )
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", StringIO(cleaned))
    model = structure[0]
    if chain_id is None:
        chain = next(iter(model))
    else:
        chain = model[chain_id]
    seq = []
    for res in chain:
        hetflag, _, _ = res.id
        if hetflag == " " and res.resname in three_to_one:
            seq.append(three_to_one[res.resname])
    return "".join(seq)


def load_motif_node(motif_dir: Path, node_idx: int) -> dict | None:
    """Load a single node entry from motif_summary.yaml + its pickled model."""
    summary_path = motif_dir / "motif_summary.yaml"
    if not summary_path.exists():
        print(f"  ⚠ {summary_path} not found.")
        return None
    with open(summary_path) as fh:
        summary_list = yaml.safe_load(fh)

    for entry in summary_list:
        if entry.get("sae_node") == node_idx:
            return entry
    return None


def load_gbm_model(motif_dir: Path, node_idx: int):
    """
    Try to load the trained GBM model from protein_data.npz.

    build_residue_motifs.py stores the full result dicts (including the
    trained sklearn models) inside protein_data.npz under key
    'node_results'.  If that's not available, we return None and fall
    back to geometry-only comparison.
    """
    npz_path = motif_dir / "protein_data.npz"
    if not npz_path.exists():
        return None
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "node_results" in data:
            results = data["node_results"].item()  # dict
            if node_idx in results:
                return results[node_idx].get("decision_tree")  # the GBM
        return None
    except Exception:
        return None


def predict_activation_probability(
    ca: np.ndarray,
    profiles: dict,
    model,
    half_w: int,
    sequence: str | None = None,
) -> np.ndarray:
    """
    Run the trained GBM/DT over every residue and return P(active)
    for each position.
    """
    n = len(ca)
    probs = np.zeros(n)
    for pos in range(half_w, n - half_w):
        feat_vec = extract_local_feature_vector(
            profiles, ca, pos, half_w, sequence=sequence,
        )
        if feat_vec is not None and np.all(np.isfinite(feat_vec)):
            p = model.predict_proba(feat_vec.reshape(1, -1))[0]
            probs[pos] = p[1] if len(p) > 1 else p[0]
    return probs


# ═══════════════════════════════════════════════════════════════════════════════
#  Plots (matching geometry_overlay style from build_residue_motifs.py)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dt_comparison(
    probs_orig: np.ndarray,
    probs_int: np.ndarray,
    node_idx: int,
    title_extra: str,
    save_path: Path,
    geom_threshold: float = 0.5,
):
    """
    Dual-panel plot comparing DT/GBM-predicted activation probability
    between the original and intervened structures.

    Style matches the geometry_overlay plots from build_residue_motifs.py.
    """
    n = min(len(probs_orig), len(probs_int))
    xs = np.arange(n)

    fig, (ax_orig, ax_int) = plt.subplots(
        2, 1, figsize=(max(12, n * 0.04), 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.15},
    )

    # ── Top panel: original ──
    ax_orig.fill_between(xs, 0, probs_orig[:n], color="#2980b9", alpha=0.25)
    ax_orig.plot(xs, probs_orig[:n], color="#2980b9", linewidth=0.8,
                 label="P(active | geometry) — original")
    ax_orig.axhline(geom_threshold, color="#27ae60", linestyle=":",
                    linewidth=0.7, alpha=0.6, label=f"threshold = {geom_threshold:.2f}")
    ax_orig.set_ylabel("P(active)")
    ax_orig.set_ylim(0, 1.05)
    ax_orig.set_title(
        f"DT-Predicted Activation — Node {node_idx}  {title_extra}\n"
        f"Top: Original structure  |  Bottom: Intervened structure",
        fontsize=10,
    )
    ax_orig.legend(fontsize=7, loc="upper right")
    ax_orig.spines["top"].set_visible(False)
    ax_orig.spines["right"].set_visible(False)

    # ── Bottom panel: intervened ──
    ax_int.fill_between(xs, 0, probs_int[:n], color="#e74c3c", alpha=0.25)
    ax_int.plot(xs, probs_int[:n], color="#e74c3c", linewidth=0.8,
                label="P(active | geometry) — intervened")
    ax_int.axhline(geom_threshold, color="#27ae60", linestyle=":",
                   linewidth=0.7, alpha=0.6)
    ax_int.set_ylabel("P(active)")
    ax_int.set_xlabel("Residue Position")
    ax_int.set_ylim(0, 1.05)
    ax_int.set_xlim(-1, n)
    ax_int.legend(fontsize=7, loc="upper right")
    ax_int.spines["top"].set_visible(False)
    ax_int.spines["right"].set_visible(False)

    # ── Shade concordant / discordant regions ──
    orig_pred = probs_orig[:n] > geom_threshold
    int_pred = probs_int[:n] > geom_threshold
    for j in range(n):
        if orig_pred[j] and int_pred[j]:
            # Both predict active → green
            ax_int.axvspan(j - 0.5, j + 0.5, color="#27ae60", alpha=0.06)
        elif orig_pred[j] and not int_pred[j]:
            # Original active, intervention killed it → amber
            ax_int.axvspan(j - 0.5, j + 0.5, color="#f39c12", alpha=0.12)
        elif not orig_pred[j] and int_pred[j]:
            # Intervention created new activation → red
            ax_int.axvspan(j - 0.5, j + 0.5, color="#e74c3c", alpha=0.10)

    legend_patches = [
        Patch(facecolor="#27ae60", alpha=0.3, label="both active"),
        Patch(facecolor="#f39c12", alpha=0.35, label="lost after intervention"),
        Patch(facecolor="#e74c3c", alpha=0.3, label="gained after intervention"),
    ]
    leg2 = ax_int.legend(
        handles=legend_patches, fontsize=6, loc="lower right",
        title="change", title_fontsize=6,
    )
    ax_int.add_artist(leg2)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved DT comparison → {save_path}")


def plot_dt_difference(
    probs_orig: np.ndarray,
    probs_int: np.ndarray,
    node_idx: int,
    title_extra: str,
    save_path: Path,
):
    """
    Single-panel ΔP(active) plot: intervened − original.
    Positive = intervention increased predicted activation.
    """
    n = min(len(probs_orig), len(probs_int))
    xs = np.arange(n)
    delta = probs_int[:n] - probs_orig[:n]

    fig, ax = plt.subplots(figsize=(max(12, n * 0.04), 3.5))

    pos_mask = delta >= 0
    neg_mask = delta < 0
    ax.bar(xs[pos_mask], delta[pos_mask], width=1.0, color="#e74c3c",
           alpha=0.6, label="increased")
    ax.bar(xs[neg_mask], delta[neg_mask], width=1.0, color="#2980b9",
           alpha=0.6, label="decreased")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Residue Position")
    ax.set_ylabel("ΔP(active)")
    ax.set_title(
        f"Change in DT-Predicted Activation — Node {node_idx}  {title_extra}",
        fontsize=10,
    )
    ax.set_xlim(-1, n)
    ax.legend(fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved ΔP plot → {save_path}")


def plot_geometry_comparison(
    ca_orig: np.ndarray,
    ca_int: np.ndarray,
    profiles_orig: dict,
    profiles_int: dict,
    node_idx: int,
    title_extra: str,
    save_path: Path,
):
    """
    Three-panel comparison of per-residue geometric profiles:
    curvature, torsion, planarity.

    Style matches the geometry_overlay dual-panel approach.
    """
    n = min(len(ca_orig), len(ca_int))
    xs = np.arange(n)

    kappa_o = profiles_orig["curvature"][:n]
    kappa_i = profiles_int["curvature"][:n]
    tau_o = profiles_orig["torsion"][:n]
    tau_i = profiles_int["torsion"][:n]
    plan_o = profiles_orig["planarity"][:n]
    plan_i = profiles_int["planarity"][:n]

    fig, axes = plt.subplots(
        3, 1, figsize=(max(12, n * 0.04), 9), sharex=True,
        gridspec_kw={"hspace": 0.18},
    )

    pairs = [
        (kappa_o, kappa_i, "Curvature (κ)"),
        (tau_o, tau_i, "Torsion (τ)"),
        (plan_o, plan_i, "Planarity"),
    ]

    for ax, (orig_vals, int_vals, ylabel) in zip(axes, pairs):
        ax.fill_between(xs, 0, orig_vals, color="#2980b9", alpha=0.20)
        ax.plot(xs, orig_vals, color="#2980b9", linewidth=0.8,
                label="original", alpha=0.9)
        ax.fill_between(xs, 0, int_vals, color="#e74c3c", alpha=0.15)
        ax.plot(xs, int_vals, color="#e74c3c", linewidth=0.8,
                label="intervened", alpha=0.9)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_title(
        f"Geometric Profile Comparison — Node {node_idx}  {title_extra}",
        fontsize=10,
    )
    axes[-1].set_xlabel("Residue Position")
    axes[-1].set_xlim(-1, n)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved geometry comparison → {save_path}")


def plot_backbone_overlay(
    ca_orig: np.ndarray,
    ca_int: np.ndarray,
    node_idx: int,
    title_extra: str,
    save_path: Path,
):
    """
    3D Kabsch-aligned backbone overlay: original vs intervened.

    Matches the plot_aligned_backbones style from kabsch_top_alignment.py.
    """
    aligned_int = kabsch_align(ca_int, ca_orig)
    rmsd = compute_rmsd(aligned_int, ca_orig)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(ca_orig[:, 0], ca_orig[:, 1], ca_orig[:, 2],
            color="#2980b9", linewidth=1.6, alpha=0.85, label="original")
    ax.plot(aligned_int[:, 0], aligned_int[:, 1], aligned_int[:, 2],
            color="#e74c3c", linewidth=1.6, alpha=0.85,
            label=f"intervened (RMSD={rmsd:.2f} Å)")

    # Mark termini
    ax.scatter(*ca_orig[0], color="#27ae60", s=60, zorder=5, label="N-term (orig)")
    ax.scatter(*ca_orig[-1], color="#8e44ad", s=60, zorder=5, label="C-term (orig)")

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(
        f"Backbone Overlay — Node {node_idx}  {title_extra}\n"
        f"Cα RMSD = {rmsd:.2f} Å",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved backbone overlay → {save_path}")

    return rmsd


def plot_combined_overlay(
    ca_orig: np.ndarray,
    ca_int: np.ndarray,
    probs_orig: np.ndarray | None,
    probs_int: np.ndarray | None,
    profiles_orig: dict,
    profiles_int: dict,
    node_idx: int,
    title_extra: str,
    save_path: Path,
    geom_threshold: float = 0.5,
):
    """
    Full combined figure matching the geometry_overlay style:

      Row 1: Kabsch-aligned 3D backbone overlay
      Row 2: DT-predicted P(active) — original vs intervened
      Row 3: Δcurvature (intervened − original)
    """
    n = min(len(ca_orig), len(ca_int))
    xs = np.arange(n)

    has_probs = probs_orig is not None and probs_int is not None

    n_rows = 3 if has_probs else 2
    height_ratios = [2, 1, 1] if has_probs else [2, 1]

    fig = plt.figure(figsize=(max(12, n * 0.04), 4 * n_rows))
    gs = fig.add_gridspec(n_rows, 1, height_ratios=height_ratios, hspace=0.25)

    # ── Row 1: 3D overlay ──
    ax3d = fig.add_subplot(gs[0], projection="3d")
    aligned_int = kabsch_align(ca_int[:n], ca_orig[:n])
    rmsd = compute_rmsd(aligned_int, ca_orig[:n])

    ax3d.plot(ca_orig[:n, 0], ca_orig[:n, 1], ca_orig[:n, 2],
              color="#2980b9", linewidth=1.4, alpha=0.85, label="original")
    ax3d.plot(aligned_int[:, 0], aligned_int[:, 1], aligned_int[:, 2],
              color="#e74c3c", linewidth=1.4, alpha=0.85,
              label=f"intervened (RMSD={rmsd:.2f} Å)")
    ax3d.scatter(*ca_orig[0], color="#27ae60", s=50, zorder=5)
    ax3d.scatter(*ca_orig[-1], color="#8e44ad", s=50, zorder=5)
    ax3d.set_xlabel("X (Å)")
    ax3d.set_ylabel("Y (Å)")
    ax3d.set_zlabel("Z (Å)")
    ax3d.set_title(
        f"Intervention Effect — Node {node_idx}  {title_extra}",
        fontsize=10,
    )
    ax3d.legend(fontsize=7, loc="upper left")

    row = 1

    # ── Row 2: DT probabilities (if available) ──
    if has_probs:
        ax_dt = fig.add_subplot(gs[row])
        ax_dt.plot(xs, probs_orig[:n], color="#2980b9", linewidth=0.8,
                   alpha=0.9, label="original")
        ax_dt.plot(xs, probs_int[:n], color="#e74c3c", linewidth=0.8,
                   alpha=0.9, label="intervened")
        ax_dt.fill_between(xs, probs_orig[:n], probs_int[:n],
                           where=probs_int[:n] > probs_orig[:n],
                           color="#e74c3c", alpha=0.15, interpolate=True)
        ax_dt.fill_between(xs, probs_orig[:n], probs_int[:n],
                           where=probs_int[:n] <= probs_orig[:n],
                           color="#2980b9", alpha=0.15, interpolate=True)
        ax_dt.axhline(geom_threshold, color="#27ae60", linestyle=":",
                      linewidth=0.7, alpha=0.6)
        ax_dt.set_ylabel("P(active)")
        ax_dt.set_ylim(0, 1.05)
        ax_dt.set_xlim(-1, n)
        ax_dt.legend(fontsize=7, loc="upper right")
        ax_dt.spines["top"].set_visible(False)
        ax_dt.spines["right"].set_visible(False)
        row += 1

    # ── Row 3 (or 2): Δcurvature ──
    ax_curv = fig.add_subplot(gs[row])
    delta_kappa = profiles_int["curvature"][:n] - profiles_orig["curvature"][:n]
    pos_mask = delta_kappa >= 0
    neg_mask = delta_kappa < 0
    ax_curv.bar(xs[pos_mask], delta_kappa[pos_mask], width=1.0,
                color="#e74c3c", alpha=0.6, label="increased")
    ax_curv.bar(xs[neg_mask], delta_kappa[neg_mask], width=1.0,
                color="#2980b9", alpha=0.6, label="decreased")
    ax_curv.axhline(0, color="black", linewidth=0.5)
    ax_curv.set_ylabel("Δκ (curvature)")
    ax_curv.set_xlabel("Residue Position")
    ax_curv.set_xlim(-1, n)
    ax_curv.legend(fontsize=7, loc="upper right")
    ax_curv.spines["top"].set_visible(False)
    ax_curv.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved combined overlay → {save_path}")


def print_summary_table(
    ca_orig: np.ndarray,
    ca_int: np.ndarray,
    profiles_orig: dict,
    profiles_int: dict,
    probs_orig: np.ndarray | None,
    probs_int: np.ndarray | None,
    rmsd: float,
    node_idx: int,
    geom_threshold: float,
):
    """Print a human-readable summary of changes."""
    n = min(len(ca_orig), len(ca_int))

    print(f"\n{'═' * 80}")
    print(f"  Intervention Effect Summary — Node {node_idx}")
    print(f"{'═' * 80}")

    # Global geometry
    rog_orig = radius_of_gyration(ca_orig[:n])
    rog_int = radius_of_gyration(ca_int[:n])
    ee_orig = end_to_end_distance(ca_orig[:n])
    ee_int = end_to_end_distance(ca_int[:n])
    curv_orig = float(np.mean(profiles_orig["curvature"][:n]))
    curv_int = float(np.mean(profiles_int["curvature"][:n]))
    tor_orig = float(np.mean(profiles_orig["torsion"][:n]))
    tor_int = float(np.mean(profiles_int["torsion"][:n]))

    print(f"\n  {'Metric':<30s}  {'Original':>12s}  {'Intervened':>12s}  {'Δ':>10s}")
    print(f"  {'─' * 30}  {'─' * 12}  {'─' * 12}  {'─' * 10}")
    print(f"  {'Cα RMSD (Å)':<30s}  {'—':>12s}  {'—':>12s}  {rmsd:>10.2f}")
    print(f"  {'Radius of gyration (Å)':<30s}  {rog_orig:>12.2f}  {rog_int:>12.2f}  {rog_int - rog_orig:>+10.2f}")
    print(f"  {'End-to-end distance (Å)':<30s}  {ee_orig:>12.2f}  {ee_int:>12.2f}  {ee_int - ee_orig:>+10.2f}")
    print(f"  {'Mean curvature':<30s}  {curv_orig:>12.4f}  {curv_int:>12.4f}  {curv_int - curv_orig:>+10.4f}")
    print(f"  {'Mean torsion':<30s}  {tor_orig:>12.4f}  {tor_int:>12.4f}  {tor_int - tor_orig:>+10.4f}")

    if probs_orig is not None and probs_int is not None:
        po = probs_orig[:n]
        pi = probs_int[:n]
        n_active_orig = int(np.sum(po > geom_threshold))
        n_active_int = int(np.sum(pi > geom_threshold))
        mean_p_orig = float(np.mean(po))
        mean_p_int = float(np.mean(pi))
        n_gained = int(np.sum((pi > geom_threshold) & (po <= geom_threshold)))
        n_lost = int(np.sum((po > geom_threshold) & (pi <= geom_threshold)))

        print(f"\n  {'DT Predictions':<30s}")
        print(f"  {'─' * 65}")
        print(f"  {'Mean P(active)':<30s}  {mean_p_orig:>12.4f}  {mean_p_int:>12.4f}  {mean_p_int - mean_p_orig:>+10.4f}")
        print(f"  {'# Residues above threshold':<30s}  {n_active_orig:>12d}  {n_active_int:>12d}  {n_active_int - n_active_orig:>+10d}")
        print(f"  {'# Gained (new activations)':<30s}  {'':>12s}  {'':>12s}  {n_gained:>10d}")
        print(f"  {'# Lost (killed activations)':<30s}  {'':>12s}  {'':>12s}  {n_lost:>10d}")

    print(f"\n{'═' * 80}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Compare geometry & DT-predicted activations before/after intervention.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--accession", required=True,
                   help="UniProt accession ID (e.g. A2AU72). AlphaFold PDB fetched automatically.")
    p.add_argument("--intervened-pdb", required=True,
                   help="Path to the intervened PDB file (e.g. from ESMFold online).")
    p.add_argument("--motif-dir", default="residue_motifs",
                   help="Directory containing motif_summary.yaml + protein_data.npz "
                        "from build_residue_motifs.py (default: residue_motifs).")
    p.add_argument("--node", type=int, required=True,
                   help="SAE node index to analyse (must exist in motif_summary.yaml).")
    p.add_argument("--chain-id", default=None,
                   help="PDB chain ID to use (default: first chain).")
    p.add_argument("--half-window", type=int, default=HALF_W,
                   help=f"Half-window size for local feature extraction (default: {HALF_W}).")
    p.add_argument("--intervention-results", default=None,
                   help="Optional: results.yaml from intervene_and_fold.py for annotation.")
    p.add_argument("--output-dir", default="intervention_comparison",
                   help="Directory to save comparison plots (default: intervention_comparison).")

    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    motif_dir = Path(args.motif_dir)
    node_idx = args.node

    print()
    print("=" * 80)
    print("  ProteinLens — Post-Intervention Comparison")
    print("=" * 80)

    # ── Load intervention metadata (optional) ─────────────────────────────
    accession = args.accession.strip()
    title_extra = f"({accession})"
    intervention_meta = None
    if args.intervention_results:
        meta_path = Path(args.intervention_results)
        if meta_path.exists():
            with open(meta_path) as fh:
                intervention_meta = yaml.safe_load(fh)
            n_mut = intervention_meta.get("n_mutations", "?")
            title_extra = f"({accession}, {n_mut} mutations)"
            print(f"  Intervention metadata: {accession}, {n_mut} mutations")

    # ── Load structures ───────────────────────────────────────────────────
    print(f"\n[1/5] Loading PDB structures …")
    pdb_cache = out / "pdb_cache"
    print(f"      Fetching AlphaFold PDB for {accession} …")
    orig_pdb_text = fetch_alphafold_pdb(accession, pdb_cache)

    int_pdb = Path(args.intervened_pdb)
    int_pdb_text = int_pdb.read_text()

    ca_orig = load_pdb_ca_from_text(orig_pdb_text, chain_id=args.chain_id)
    ca_int = load_pdb_ca_from_text(int_pdb_text, chain_id=args.chain_id)
    print(f"      Original:  {len(ca_orig)} Cα atoms  (AlphaFold {accession})")
    print(f"      Intervened: {len(ca_int)} Cα atoms  ({int_pdb.name})")

    # Truncate to common length
    n = min(len(ca_orig), len(ca_int))
    if len(ca_orig) != len(ca_int):
        print(f"      ⚠ Length mismatch — truncating both to {n} residues.")
    ca_orig = ca_orig[:n]
    ca_int = ca_int[:n]

    # Extract sequences (best-effort for feature extraction)
    try:
        seq_orig = get_sequence_from_pdb_text(orig_pdb_text, chain_id=args.chain_id)[:n]
    except Exception:
        seq_orig = None
    try:
        seq_int = get_sequence_from_pdb_text(int_pdb_text, chain_id=args.chain_id)[:n]
    except Exception:
        seq_int = None

    # ── Compute geometric profiles ────────────────────────────────────────
    print(f"\n[2/5] Computing geometric profiles …")
    helices_orig = detect_alpha_helices_from_ca(ca_orig)
    helices_int = detect_alpha_helices_from_ca(ca_int)
    profiles_orig = compute_residue_profiles(ca_orig, helices_orig)
    profiles_int = compute_residue_profiles(ca_int, helices_int)
    print(f"      Original: {len(helices_orig)} helices detected")
    print(f"      Intervened: {len(helices_int)} helices detected")

    # ── Load DT/GBM model & predict ──────────────────────────────────────
    print(f"\n[3/5] Loading motif model for node {node_idx} …")
    motif_entry = load_motif_node(motif_dir, node_idx)
    geom_threshold = 0.5
    probs_orig = None
    probs_int = None

    # Try loading the trained model from the npz
    model = load_gbm_model(motif_dir, node_idx)

    if model is not None:
        geom_threshold = motif_entry.get("optimal_geom_threshold", 0.5) if motif_entry else 0.5
        print(f"      Loaded GBM model (threshold = {geom_threshold:.3f})")
        print(f"      Predicting activation probabilities …")
        probs_orig = predict_activation_probability(
            ca_orig, profiles_orig, model, args.half_window, sequence=seq_orig,
        )
        probs_int = predict_activation_probability(
            ca_int, profiles_int, model, args.half_window, sequence=seq_int,
        )
    else:
        print(f"      ⚠ Could not load trained model from protein_data.npz.")
        print(f"        DT-prediction plots will be skipped.")
        print(f"        To enable: re-run build_residue_motifs.py with --save-models,")
        print(f"        or manually pickle the node results.")

    if motif_entry:
        print(f"      Node {node_idx} summary from YAML:")
        print(f"        mean RMSD   = {motif_entry.get('mean_rmsd', '?')}")
        print(f"        GBM AUC     = {motif_entry.get('auc_cv_gbm', '?')}")
        print(f"        LPO AUC     = {motif_entry.get('leave_proteins_out_auc', '?')}")
    else:
        print(f"      ⚠ Node {node_idx} not found in motif_summary.yaml.")

    # ── Generate plots ────────────────────────────────────────────────────
    print(f"\n[4/5] Generating plots …")

    # 4a. Kabsch-aligned backbone overlay
    rmsd = plot_backbone_overlay(
        ca_orig, ca_int, node_idx, title_extra,
        out / "backbone_overlay.png",
    )

    # 4b. Geometric profile comparison (curvature, torsion, planarity)
    plot_geometry_comparison(
        ca_orig, ca_int, profiles_orig, profiles_int,
        node_idx, title_extra,
        out / "geometry_comparison.png",
    )

    # 4c. DT-predicted activation comparison (if model available)
    if probs_orig is not None and probs_int is not None:
        plot_dt_comparison(
            probs_orig, probs_int, node_idx, title_extra,
            out / "dt_prediction_comparison.png",
            geom_threshold=geom_threshold,
        )
        plot_dt_difference(
            probs_orig, probs_int, node_idx, title_extra,
            out / "dt_prediction_delta.png",
        )

    # 4d. Combined overlay (3D backbone + DT + Δcurvature)
    plot_combined_overlay(
        ca_orig, ca_int, probs_orig, probs_int,
        profiles_orig, profiles_int,
        node_idx, title_extra,
        out / "combined_overlay.png",
        geom_threshold=geom_threshold,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n[5/5] Summary")
    print_summary_table(
        ca_orig, ca_int, profiles_orig, profiles_int,
        probs_orig, probs_int, rmsd, node_idx, geom_threshold,
    )

    # Save summary to YAML
    summary = {
        "node": node_idx,
        "accession": accession,
        "intervened_pdb": str(int_pdb),
        "n_residues": n,
        "ca_rmsd": round(float(rmsd), 4),
        "radius_of_gyration_orig": round(float(radius_of_gyration(ca_orig)), 4),
        "radius_of_gyration_int": round(float(radius_of_gyration(ca_int)), 4),
        "end_to_end_orig": round(float(end_to_end_distance(ca_orig)), 4),
        "end_to_end_int": round(float(end_to_end_distance(ca_int)), 4),
        "mean_curvature_orig": round(float(np.mean(profiles_orig["curvature"][:n])), 6),
        "mean_curvature_int": round(float(np.mean(profiles_int["curvature"][:n])), 6),
        "mean_torsion_orig": round(float(np.mean(profiles_orig["torsion"][:n])), 6),
        "mean_torsion_int": round(float(np.mean(profiles_int["torsion"][:n])), 6),
    }
    if probs_orig is not None and probs_int is not None:
        summary["mean_p_active_orig"] = round(float(np.mean(probs_orig[:n])), 6)
        summary["mean_p_active_int"] = round(float(np.mean(probs_int[:n])), 6)
        summary["n_active_orig"] = int(np.sum(probs_orig[:n] > geom_threshold))
        summary["n_active_int"] = int(np.sum(probs_int[:n] > geom_threshold))
        summary["geom_threshold"] = round(float(geom_threshold), 4)
    if intervention_meta:
        summary["intervention_meta"] = intervention_meta

    (out / "comparison_summary.yaml").write_text(
        yaml.dump(summary, default_flow_style=False, sort_keys=False)
    )

    print(f"  All outputs saved to {out}/")
    print(f"    • backbone_overlay.png         — Kabsch-aligned 3D overlay")
    print(f"    • geometry_comparison.png       — curvature / torsion / planarity profiles")
    if probs_orig is not None:
        print(f"    • dt_prediction_comparison.png — DT P(active) before vs after")
        print(f"    • dt_prediction_delta.png      — ΔP(active) per residue")
    print(f"    • combined_overlay.png         — all-in-one figure")
    print(f"    • comparison_summary.yaml      — numeric summary")
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
