from __future__ import annotations

"""
Build a geometry ↔ SAE activation dataset from scratch.

Instead of relying on pre-computed Per_feature_max_examples.yaml and batch
YAML files, this script:

  1.  Downloads a large set of reviewed UniProt accessions (default: human
      Swiss-Prot, ~20 K proteins — all covered by AlphaFold DB).
  2.  Fetches each protein's sequence from UniProt and its predicted 3D
      structure from the AlphaFold Database.
  3.  Runs ESM → SAE to compute per-protein, per-node mean activation.
  4.  Identifies the top-K most-activating proteins for every SAE node.
  5.  Computes geometric features from the AlphaFold structures.
  6.  Saves everything and runs the correlation analysis.

Usage:
    python build_activation_dataset.py                        # human proteome
    python build_activation_dataset.py --organism 10090       # mouse
    python build_activation_dataset.py --fasta my_seqs.fasta  # custom FASTA
    python build_activation_dataset.py --accession-list ids.txt  # plain list

Outputs  (all written to OUTPUT_DIR):
    accessions.txt               – the processed accession list
    activation_matrix.npy        – (n_proteins, dict_size) mean SAE acts
    geometry_matrix.npy          – (n_proteins, n_geom_feats) geometry
    top_activating_per_node.yaml – top-K accessions per SAE node
    geometry_sae_correlations.yaml
    correlation_heatmap.png
    top_correlation_scatter.png
    bar_*.png
"""

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import yaml
import matplotlib

matplotlib.use("Agg")  # non-interactive backend so plots save without display
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdb_plotter import ca_backbone, detect_alpha_helices_from_ca
from kabsch_top_alignment import plot_kabsch_alignments
from geometry.compute_geometric_features import (
    writhe,
    vassiliev,
    average_curvature,
    average_torsion,
    gyration_asphericity,
    kink_index,
    radius_of_gyration,
    local_planarity_score,
    end_to_end_distance,
    helical_consistency,
    helix_statistics,
    helix_statistics_contact_filtered,
    helix_segments,
    turn_density,
    hairpin_score,
    extended_fraction,
    signed_torsion,
    dihedral_sign_consistency,
    # local / windowed profiles
    local_curvature,
    local_torsion,
    local_planarity,
    local_writhe,
)
from proteinlens.sae.inference import load_sae
from proteinlens.embedders.esm import ESM
from proteinlens.utils import get_device

# ========================== DEFAULTS =======================================
SAE_DIR = ROOT / "trained_models" / "fiery-sweep"
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
ESM_LAYER = 3
TOP_K_PER_NODE = 10          # how many top-activating proteins per SAE node
MAX_SEQ_LEN = 1024           # skip very long proteins (ESM context limit)
ALPHAFOLD_API_URL = (
    "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
)
UNIPROT_FASTA_URL = (
    "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
)
UNIPROT_SEARCH_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
)

GEOM_FEATURE_NAMES = [
    "writhe",
    "vassiliev_v2",
    "avg_curvature",
    "avg_torsion",
    "kink_index",
    "gyration_asphericity",
    "helix_parallel_mean",
    "helix_parallel_std",
    "helix_dist_mean",
    "helix_dist_std",
    "radius_of_gyration",
    "local_planarity",
    "end_to_end_distance",
    "tangent_alignment",
    "binormal_consistency",
    # ── contact-filtered helix pair stats ──
    "contact_parallel_mean",
    "contact_parallel_std",
    "contact_dist_mean",
    "contact_dist_std",
    "contact_parallel_top3",
    "contact_frac_parallel_0p8",
    "contact_angle_mean",
    "contact_angle_std",
    "contact_angle_frac_lt15",
    "contact_angle_frac_gt60",
    "n_helices",
    "n_contact_pairs",
    # ── helix segment stats ──
    "helix_fraction",
    "mean_helix_len",
    "std_helix_len",
    "max_helix_len",
    # ── turn / hairpin / strand proxies ──
    "turn_density",
    "hairpin_score",
    "extended_fraction",
    # ── signed torsion stats ──
    "signed_torsion_mean",
    "signed_torsion_std",
    "signed_torsion_frac_pos",
    "signed_torsion_frac_neg",
    # ── dihedral consistency ──
    "dihedral_sign_consistency",
    # ── local (windowed) profile summaries ──
    "local_curvature_mean",
    "local_curvature_std",
    "local_curvature_max",
    "local_curvature_range",
    "local_torsion_mean",
    "local_torsion_std",
    "local_torsion_max",
    "local_torsion_range",
    "local_planarity_mean",
    "local_planarity_std",
    "local_planarity_max",
    "local_planarity_range",
    "local_writhe_mean",
    "local_writhe_std",
    "local_writhe_max",
    "local_writhe_range",
]


# ====================== 1. ACCESSION RETRIEVAL =============================

def fetch_swissprot_accessions(
    organism_taxid: int = 9606,
    max_proteins: int | None = None,
) -> list[str]:
    """
    Query UniProt for reviewed (Swiss-Prot) accessions for a given organism.

    Recommended organisms:
      9606  – Homo sapiens  (~20 400 proteins)
      10090 – Mus musculus   (~17 200)
      559292 – S. cerevisiae  (~6 700)
      83333 – E. coli K-12    (~4 500)

    Returns a list of UniProt accession strings.
    """
    print(f"[1/6] Querying UniProt for reviewed proteins (taxon {organism_taxid}) …")
    query = f"(reviewed:true) AND (organism_id:{organism_taxid})"
    params = {
        "query": query,
        "format": "list",         # just accessions, one per line
        "size": 500,
    }

    accessions: list[str] = []
    url = UNIPROT_SEARCH_URL

    # UniProt streams results; we page through with the Link header
    while url:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        batch = [a.strip() for a in resp.text.strip().split("\n") if a.strip()]
        accessions.extend(batch)
        # Follow pagination link if present
        link = resp.headers.get("Link", "")
        url = None
        params = None  # only use params on the first request
        if 'rel="next"' in link:
            match = re.search(r'<([^>]+)>', link)
            if match:
                url = match.group(1)

        if max_proteins and len(accessions) >= max_proteins:
            accessions = accessions[:max_proteins]
            break

    print(f"  → {len(accessions)} accessions retrieved.")
    return accessions


def load_accessions_from_file(path: Path) -> list[str]:
    """Load accessions from a plain text file (one per line)."""
    lines = path.read_text().strip().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def load_accessions_from_fasta(path: Path) -> tuple[list[str], dict[str, str]]:
    """
    Parse a FASTA file and return (accession_list, {acc: sequence}).
    Header format supported:
      >sp|P12345|PROT_HUMAN ...
      >P12345 ...
      >tr|A0A123|...
    """
    seqs: dict[str, str] = {}
    current_acc = None
    current_seq: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current_acc:
                seqs[current_acc] = "".join(current_seq)
            header = line[1:].split()[0]
            # Handle >sp|P12345|NAME or >P12345
            parts = header.split("|")
            current_acc = parts[1] if len(parts) >= 2 else parts[0]
            current_seq = []
        else:
            current_seq.append(line.strip())
    if current_acc:
        seqs[current_acc] = "".join(current_seq)
    return list(seqs.keys()), seqs


# =================== 2. SEQUENCE / STRUCTURE FETCHING ======================

def fetch_sequence(acc: str, session: requests.Session) -> str | None:
    """Fetch a single protein sequence from UniProt."""
    url = UNIPROT_FASTA_URL.format(acc=acc)
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return None
        lines = r.text.strip().split("\n")
        return "".join(l.strip() for l in lines if not l.startswith(">"))
    except Exception:
        return None


def fetch_alphafold_pdb(acc: str, cache_dir: Path,
                        session: requests.Session) -> str | None:
    """
    Download an AlphaFold PDB file for a UniProt accession.
    Uses the AlphaFold API to discover the current PDB URL (version may change).
    Caches on disk so subsequent runs are instant.
    """
    # Check for any cached version first
    cached_files = list(cache_dir.glob(f"AF-{acc}-F1-model_v*.pdb"))
    if cached_files:
        return cached_files[0].read_text()

    # Query the AlphaFold API to get the correct PDB URL
    api_url = ALPHAFOLD_API_URL.format(acc=acc)
    try:
        r = session.get(api_url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, list):
            if len(data) == 0:
                return None
            data = data[0]
        pdb_url = data.get("pdbUrl")
        if not pdb_url:
            return None
    except Exception:
        return None

    # Download the PDB file
    try:
        r = session.get(pdb_url, timeout=30)
        if r.status_code != 200:
            return None
        pdb_text = r.text
        # Cache with actual filename from the URL
        fname = pdb_url.rsplit("/", 1)[-1]
        (cache_dir / fname).write_text(pdb_text)
        return pdb_text
    except Exception:
        return None


def get_protein_sequence_from_pdb(pdb_text: str) -> str:
    """Extract amino-acid sequence from PDB ATOM records (CA atoms)."""
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    seen: set[str] = set()
    seq: list[str] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        resseq = line[22:27].strip()
        if resseq in seen:
            continue
        seen.add(resseq)
        resname = line[17:20].strip()
        seq.append(three_to_one.get(resname, "X"))
    return "".join(seq)


# ==================== 3. ESM + SAE ACTIVATION ==============================

def compute_activation(
    sae, embedder, sequence: str, device: str, esm_layer: int,
) -> np.ndarray | None:
    """
    Run ESM → SAE on one protein sequence.
    Returns mean activation per SAE node, shape (dict_size,).
    """
    if len(sequence) < 4 or len(sequence) > MAX_SEQ_LEN:
        return None
    try:
        emb = embedder.embed_single_sequence(sequence, layer=esm_layer)
        emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
        with torch.no_grad():
            feats = sae.encode(emb_t)
        return feats.mean(dim=0).detach().cpu().numpy()
    except Exception as e:
        return None


# ==================== 4. GEOMETRY FROM PDB TEXT =============================

def compute_geometry(pdb_text: str) -> dict | None:
    """Compute all geometric features from a PDB text string."""
    try:
        ca = ca_backbone(pdb_text, chain_id=None)
        plt.close("all")
    except Exception:
        return None
    if ca is None or len(ca) < 4:
        return None
    try:
        helices = detect_alpha_helices_from_ca(ca)

        # ── original features ──
        wr_d = writhe(ca, ca)
        wr = float(np.sum(wr_d))
        # _v2 = float(vassiliev(wr_d))  # O(n^4) bottleneck — skipped
        _v2 = 0.0
        cur = float(average_curvature(ca))
        tor = float(average_torsion(ca))
        ki = float(kink_index(ca))
        ga = float(gyration_asphericity(ca))
        p_m, p_s, d_m, d_s = helix_statistics(ca, helices)
        rog = float(radius_of_gyration(ca))
        planar = float(local_planarity_score(ca))
        end = float(end_to_end_distance(ca))
        ta, bc = helical_consistency(ca)

        # ── contact-filtered helix pair stats (12 values) ──
        (cp_m, cp_s, cd_m, cd_s, cp_top3, cp_frac,
         ca_mean, ca_std, ca_lt15, ca_gt60,
         n_hel, n_cpairs) = helix_statistics_contact_filtered(ca, helices)

        # ── helix segment stats ──
        _n_hel, h_frac, h_mean_len, h_std_len, h_max_len = helix_segments(
            ca, helices
        )

        # ── turn / hairpin / strand proxies ──
        td = float(turn_density(ca))
        hp = float(hairpin_score(ca))
        ef = float(extended_fraction(ca))

        # ── signed torsion stats ──
        st_mean, st_std, st_fp, st_fn = signed_torsion(ca)

        # ── dihedral sign consistency ──
        dsc = float(dihedral_sign_consistency(ca))

        # ── local (windowed) profile summaries ──
        def _profile_stats(arr):
            """Reduce a 1-D windowed profile to (mean, std, max, range)."""
            if arr.size == 0:
                return 0.0, 0.0, 0.0, 0.0
            mn  = float(np.mean(arr))
            sd  = float(np.std(arr))
            mx  = float(np.max(arr))
            rng = float(mx - np.min(arr))
            return mn, sd, mx, rng

        lc = local_curvature(ca)          # w=21, stride=1, pool="mean"
        lc_mean, lc_std, lc_max, lc_rng = _profile_stats(lc)

        lt = local_torsion(ca)            # w=21, stride=1, pool="mean"
        lt_mean, lt_std, lt_max, lt_rng = _profile_stats(lt)

        lp = local_planarity(ca)          # w=21, stride=1, inner_w=7
        lp_mean, lp_std, lp_max, lp_rng = _profile_stats(lp)

        lw = local_writhe(ca)             # w=41, stride=3
        lw_mean, lw_std, lw_max, lw_rng = _profile_stats(lw)

    except Exception:
        return None

    values = [
        wr, _v2, cur, tor, ki, ga,
        float(p_m), float(p_s), float(d_m), float(d_s),
        rog, planar, end, float(ta), float(bc),
        # contact-filtered helix
        float(cp_m), float(cp_s), float(cd_m), float(cd_s),
        float(cp_top3), float(cp_frac),
        float(ca_mean), float(ca_std), float(ca_lt15), float(ca_gt60),
        float(n_hel), float(n_cpairs),
        # helix segments
        float(h_frac), float(h_mean_len), float(h_std_len), float(h_max_len),
        # turn / hairpin / strand
        td, hp, ef,
        # signed torsion
        float(st_mean), float(st_std), float(st_fp), float(st_fn),
        # dihedral consistency
        dsc,
        # local profiles
        lc_mean, lc_std, lc_max, lc_rng,
        lt_mean, lt_std, lt_max, lt_rng,
        lp_mean, lp_std, lp_max, lp_rng,
        lw_mean, lw_std, lw_max, lw_rng,
    ]
    return dict(zip(GEOM_FEATURE_NAMES, values))


# =================== 5. TOP-K PER NODE =====================================

def find_top_k_per_node(
    accessions: list[str],
    act_matrix: np.ndarray,
    k: int = 10,
) -> dict[int, list[str]]:
    """
    For each SAE node, find the k proteins with the highest mean activation.
    Returns {node_id: [acc1, acc2, …]}.
    """
    n_nodes = act_matrix.shape[1]
    top_k: dict[int, list[str]] = {}
    for ni in range(n_nodes):
        col = act_matrix[:, ni]
        # argsort descending
        order = np.argsort(col)[::-1][:k]
        top_k[ni] = [accessions[i] for i in order if col[i] > 0]
    return top_k


# =================== 6. CORRELATION ANALYSIS ================================

def correlation_analysis(geom_matrix, act_matrix, geom_names, top_k=20):
    n_geom = geom_matrix.shape[1]
    n_nodes = act_matrix.shape[1]
    pearson_r = np.zeros((n_geom, n_nodes))
    pearson_p = np.ones((n_geom, n_nodes))
    spearman_r = np.zeros((n_geom, n_nodes))
    spearman_p = np.ones((n_geom, n_nodes))

    MIN_ACTIVE = 30  # require at least this many proteins with act > 0

    for gi in range(n_geom):
        g = geom_matrix[:, gi]
        geom_valid = np.isfinite(g)
        for ni in range(n_nodes):
            # Only include proteins that actually activate on this node
            active = act_matrix[:, ni] > 0
            mask = geom_valid & active
            if mask.sum() < MIN_ACTIVE:
                continue
            gv = g[mask]
            a = act_matrix[mask, ni]
            if a.std() < 1e-12 or gv.std() < 1e-12:
                continue
            pr, pp = stats.pearsonr(gv, a)
            sr, sp = stats.spearmanr(gv, a)
            pearson_r[gi, ni] = pr
            pearson_p[gi, ni] = pp
            spearman_r[gi, ni] = sr
            spearman_p[gi, ni] = sp

    summary = []
    for gi in range(n_geom):
        for ni in range(n_nodes):
            if pearson_p[gi, ni] < 0.05:
                summary.append({
                    "geom_feature": geom_names[gi],
                    "sae_node": int(ni),
                    "pearson_r": float(pearson_r[gi, ni]),
                    "pearson_p": float(pearson_p[gi, ni]),
                    "spearman_r": float(spearman_r[gi, ni]),
                    "spearman_p": float(spearman_p[gi, ni]),
                })
    summary.sort(key=lambda d: abs(d["pearson_r"]), reverse=True)
    return pearson_r, pearson_p, spearman_r, spearman_p, summary


# ======================== PLOTTING =========================================

def plot_correlation_heatmap(pearson_r, geom_names, save_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    vmax = np.nanpercentile(np.abs(pearson_r), 99)
    im = ax.imshow(pearson_r, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(geom_names)))
    ax.set_yticklabels(geom_names, fontsize=9)
    ax.set_xlabel("SAE Node Index")
    ax.set_ylabel("Geometric Feature")
    ax.set_title("Pearson Correlation: Geometric Features vs SAE Node Activations")
    plt.colorbar(im, ax=ax, label="Pearson r")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Heatmap saved → {save_path}")


def plot_top_scatter(geom_matrix, act_matrix, summary, geom_names,
                     plots_per_figure=6, save_dir=None):
    """
    For each geometric feature, pick its single most-correlated SAE node
    and plot the scatter.  Splits across multiple figures.
    """
    # Deduplicate: keep the best (highest |pearson_r|) entry per feature
    best_per_feat: dict[str, dict] = {}
    for entry in summary:
        feat = entry["geom_feature"]
        if feat not in best_per_feat or abs(entry["pearson_r"]) > abs(best_per_feat[feat]["pearson_r"]):
            best_per_feat[feat] = entry

    # Order by |pearson_r| descending across the unique features
    unique_entries = sorted(best_per_feat.values(),
                            key=lambda d: abs(d["pearson_r"]), reverse=True)
    k = len(unique_entries)
    if k == 0:
        return
    n_figures = (k + plots_per_figure - 1) // plots_per_figure
    for fig_idx in range(n_figures):
        start = fig_idx * plots_per_figure
        end = min(start + plots_per_figure, k)
        n_plots = end - start
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.atleast_2d(axes)
        for idx in range(n_plots):
            ax = axes.flat[idx]
            entry = unique_entries[start + idx]
            gi = geom_names.index(entry["geom_feature"])
            ni = entry["sae_node"]
            x = geom_matrix[:, gi]
            y = act_matrix[:, ni]
            valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
            ax.scatter(x[valid], y[valid], s=12, alpha=0.6, edgecolors="none")
            ax.set_xlabel(entry["geom_feature"])
            ax.set_ylabel(f"SAE node {ni} act.")
            ax.set_title(
                f"{entry['geom_feature']}  →  node {ni}\n"
                f"r={entry['pearson_r']:.3f}  ρ={entry['spearman_r']:.3f}",
                fontsize=10,
            )
        for idx in range(n_plots, rows * cols):
            axes.flat[idx].set_visible(False)
        plt.suptitle(
            f"Best SAE Node per Geometric Feature  (#{start+1}–#{end})",
            y=1.02,
        )
        plt.tight_layout()
        if save_dir:
            path = save_dir / f"top_correlation_scatter_{fig_idx + 1}.png"
            plt.savefig(path, dpi=200, bbox_inches="tight")
            print(f"  Scatter figure {fig_idx + 1} saved → {path}")
        plt.close()


def plot_per_feature_bar(pearson_r, geom_names, feature_name,
                         top_k=20, save_path=None):
    gi = geom_names.index(feature_name)
    r_vals = pearson_r[gi]
    order = np.argsort(np.abs(r_vals))[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(10, 4))
    colours = ["#e74c3c" if r_vals[i] < 0 else "#2980b9" for i in order]
    ax.barh(range(len(order)), r_vals[order], color=colours)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"Node {i}" for i in order])
    ax.set_xlabel("Pearson r")
    ax.set_title(f"SAE nodes most correlated with {feature_name}")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


# ========================= MAIN PIPELINE ===================================

def main():
    parser = argparse.ArgumentParser(
        description="Build geometry ↔ SAE activation dataset from scratch."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--organism", type=int, default=None,
        help="UniProt taxonomy ID.  9606 = human (default), "
             "10090 = mouse, 559292 = yeast, 83333 = E. coli"
    )
    src.add_argument(
        "--fasta", type=Path, default=None,
        help="Path to a FASTA file with protein sequences."
    )
    src.add_argument(
        "--accession-list", type=Path, default=None,
        help="Plain text file with one UniProt accession per line."
    )
    parser.add_argument(
        "--max-proteins", type=int, default=None,
        help="Cap the number of proteins to process."
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K_PER_NODE,
        help="Number of top-activating proteins per SAE node."
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "alphafold_analysis",
        help="Directory for all outputs."
    )
    parser.add_argument(
        "--sae-dir", type=Path, default=SAE_DIR,
        help="Path to the trained SAE model directory."
    )
    parser.add_argument(
        "--esm-model", type=str, default=ESM_MODEL_NAME,
    )
    parser.add_argument("--esm-layer", type=int, default=ESM_LAYER)
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from cached intermediate files if they exist."
    )
    args = parser.parse_args()

    # Default to human if nothing specified
    if args.organism is None and args.fasta is None and args.accession_list is None:
        args.organism = 9606

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pdb_cache = out / "pdb_cache"
    pdb_cache.mkdir(exist_ok=True)

    device = get_device()
    print("=" * 72)
    print("Build Geometry ↔ SAE Activation Dataset (from scratch)")
    print("=" * 72)
    print(f"[✓] Device: {device}")
    print(f"[i] Output dir: {out}")
    print()

    # ── Step 1: Get accession list ────────────────────────────────────────
    fasta_seqs: dict[str, str] | None = None
    if args.fasta:
        accessions, fasta_seqs = load_accessions_from_fasta(args.fasta)
        print(f"[1/6] Loaded {len(accessions)} proteins from FASTA: {args.fasta}")
    elif args.accession_list:
        accessions = load_accessions_from_file(args.accession_list)
        print(f"[1/6] Loaded {len(accessions)} accessions from: {args.accession_list}")
    else:
        accessions = fetch_swissprot_accessions(
            organism_taxid=args.organism,
            max_proteins=args.max_proteins,
        )

    if args.max_proteins:
        accessions = accessions[: args.max_proteins]

    # Save accession list
    (out / "accessions.txt").write_text("\n".join(accessions))
    print(f"  {len(accessions)} proteins to process.\n")

    # ── Step 2: Load SAE + ESM ────────────────────────────────────────────
    print("[2/6] Loading SAE …")
    sae = load_sae(args.sae_dir, device=device)
    sae.eval()
    print(f"  SAE: {sae.__class__.__name__}  dict_size={sae.dict_size}  "
          f"activation_dim={sae.activation_dim}")

    print("[2/6] Loading ESM embedder …")
    embedder = ESM(model_name=args.esm_model, device=device)
    print(f"  ESM: {args.esm_model}  layer {args.esm_layer}\n")

    # ── Step 3: Compute activations + geometry ────────────────────────────
    act_cache = out / "activation_matrix.npy"
    geom_cache = out / "geometry_matrix.npy"
    acc_cache = out / "processed_accessions.txt"

    if args.resume and act_cache.exists() and geom_cache.exists():
        print("[3/6] Loading cached activation matrix …")
        act_matrix = np.load(act_cache)
        accessions_ok = acc_cache.read_text().strip().split("\n")
        print(f"  {len(accessions_ok)} proteins loaded from cache.")

        # Recompute geometry from cached PDB files
        print("  Recomputing geometry from cached PDB files …\n")
        geom_rows: list[list[float]] = []
        act_rows_keep: list[int] = []
        n_geom_skip = 0
        for idx, acc in enumerate(accessions_ok):
            if idx % 500 == 0 or idx == len(accessions_ok) - 1:
                print(
                    f"    [{idx + 1:>6d}/{len(accessions_ok)}]  "
                    f"ok={len(geom_rows)}  skip={n_geom_skip}  current={acc}"
                )
            # Find cached PDB
            cached_files = list(pdb_cache.glob(f"AF-{acc}-F1-model_v*.pdb"))
            if not cached_files:
                n_geom_skip += 1
                continue
            pdb_text = cached_files[0].read_text()
            geom = compute_geometry(pdb_text)
            if geom is None:
                n_geom_skip += 1
                continue
            geom_rows.append([geom[k] for k in GEOM_FEATURE_NAMES])
            act_rows_keep.append(idx)

        if not geom_rows:
            print("\n✘ No geometry recomputed. Check PDB cache.")
            sys.exit(1)

        geom_matrix = np.array(geom_rows, dtype=float)
        act_matrix = act_matrix[act_rows_keep]
        accessions_ok = [accessions_ok[i] for i in act_rows_keep]

        # Save updated matrices
        np.save(geom_cache, geom_matrix)
        np.save(act_cache, act_matrix)
        acc_cache.write_text("\n".join(accessions_ok))
        print(f"\n  [✓] Geometry recomputed for {len(accessions_ok)} proteins, "
              f"{n_geom_skip} skipped.\n")
    else:
        print("[3/6] Computing activations & geometry for each protein …")
        print("  This fetches AlphaFold structures from the EBI API (cached on disk).\n")

        session = requests.Session()
        accessions_ok: list[str] = []
        act_rows: list[np.ndarray] = []
        geom_rows: list[list[float]] = []
        n_skip = 0

        for idx, acc in enumerate(accessions):
            if idx % 10 == 0 or idx == len(accessions) - 1:
                print(
                    f"  [{idx + 1:>6d}/{len(accessions)}]  "
                    f"ok={len(accessions_ok)}  skip={n_skip}  current={acc}"
                )

            # -- get sequence --
            if fasta_seqs and acc in fasta_seqs:
                seq = fasta_seqs[acc]
            else:
                seq = fetch_sequence(acc, session)
            if not seq or len(seq) < 4 or len(seq) > MAX_SEQ_LEN:
                n_skip += 1
                continue

            # -- get structure (AlphaFold) --
            pdb_text = fetch_alphafold_pdb(acc, pdb_cache, session)
            if pdb_text is None:
                n_skip += 1
                continue

            # -- SAE activation --
            act = compute_activation(
                sae, embedder, seq, device, args.esm_layer
            )
            if act is None:
                n_skip += 1
                continue

            # -- geometry --
            geom = compute_geometry(pdb_text)
            if geom is None:
                n_skip += 1
                continue

            accessions_ok.append(acc)
            act_rows.append(act)
            geom_rows.append([geom[k] for k in GEOM_FEATURE_NAMES])

            # Periodic checkpoint every 500 proteins
            if len(accessions_ok) % 500 == 0 and len(accessions_ok) > 0:
                print(f"    … checkpoint at {len(accessions_ok)} proteins")
                np.save(act_cache, np.vstack(act_rows))
                np.save(geom_cache, np.array(geom_rows, dtype=float))
                acc_cache.write_text("\n".join(accessions_ok))

        if not accessions_ok:
            print("\n✘ No proteins processed successfully. Check network / accessions.")
            sys.exit(1)

        act_matrix = np.vstack(act_rows)
        geom_matrix = np.array(geom_rows, dtype=float)

        # Save
        np.save(act_cache, act_matrix)
        np.save(geom_cache, geom_matrix)
        acc_cache.write_text("\n".join(accessions_ok))
        print(f"\n  [✓] {len(accessions_ok)} proteins processed, "
              f"{n_skip} skipped.\n")

    n_prot, n_nodes = act_matrix.shape
    n_geom = geom_matrix.shape[1]
    print(f"  Dataset: {n_prot} proteins × {n_geom} geom features × "
          f"{n_nodes} SAE nodes\n")

    # ── Step 4: Top-K per node ────────────────────────────────────────────
    print(f"[4/6] Finding top-{args.top_k} activating proteins per SAE node …")
    top_k_map = find_top_k_per_node(accessions_ok, act_matrix, k=args.top_k)
    top_k_path = out / "top_activating_per_node.yaml"
    with open(top_k_path, "w") as f:
        yaml.dump(top_k_map, f, default_flow_style=False)
    n_active = sum(1 for v in top_k_map.values() if v)
    print(f"  {n_active}/{n_nodes} nodes have at least one activating protein.")
    print(f"  Saved → {top_k_path}\n")

    # ── Step 5: Correlation analysis ──────────────────────────────────────
    print("[5/6] Computing Pearson & Spearman correlations …")
    pearson_r, pearson_p, spearman_r, spearman_p, summary = (
        correlation_analysis(geom_matrix, act_matrix, GEOM_FEATURE_NAMES)
    )
    print(f"  [✓] {len(summary)} significant correlations (p < 0.05).\n")

    # Print top
    TOP_PRINT = 40
    print(f"{'=' * 72}")
    print(f"Top {min(TOP_PRINT, len(summary))} geometry ↔ SAE node correlations")
    print(f"{'=' * 72}")
    print(
        f"{'Geom Feature':<28s} {'Node':>6s} {'Pearson r':>10s} "
        f"{'p-value':>10s} {'Spearman ρ':>10s}"
    )
    print("-" * 72)
    for entry in summary[:TOP_PRINT]:
        print(
            f"{entry['geom_feature']:<28s} {entry['sae_node']:>6d} "
            f"{entry['pearson_r']:>10.4f} {entry['pearson_p']:>10.2e} "
            f"{entry['spearman_r']:>10.4f}"
        )

    corr_path = out / "geometry_sae_correlations.yaml"
    with open(corr_path, "w") as f:
        yaml.dump(summary[:200], f, default_flow_style=False)
    print(f"\n  Full summary → {corr_path}\n")

    # ── Step 6: Plots ─────────────────────────────────────────────────────
    print("[6/6] Generating plots …")
    plot_correlation_heatmap(
        pearson_r, GEOM_FEATURE_NAMES, out / "correlation_heatmap.png"
    )
    plot_top_scatter(
        geom_matrix, act_matrix, summary, GEOM_FEATURE_NAMES,
        plots_per_figure=6, save_dir=out,
    )
    for feat in [
        "writhe", "avg_curvature", "radius_of_gyration", "kink_index",
        "turn_density", "hairpin_score", "extended_fraction",
        "signed_torsion_mean", "dihedral_sign_consistency",
        "helix_fraction", "contact_angle_mean",
        "local_curvature_std", "local_curvature_max",
        "local_torsion_std", "local_torsion_max",
        "local_planarity_std",
        "local_writhe_std", "local_writhe_max",
    ]:
        plot_per_feature_bar(
            pearson_r, GEOM_FEATURE_NAMES, feat, top_k=20,
            save_path=out / f"bar_{feat}.png",
        )

    # ── Step 7: Kabsch-aligned backbone overlays ─────────────────────────
    print("[7/7] Plotting Kabsch-aligned backbone overlays …")
    kabsch_dir = out / "kabsch_overlays"
    plot_kabsch_alignments(
        summary=summary,
        top_k_map=top_k_map,
        pdb_cache=pdb_cache,
        save_dir=kabsch_dir,
        n_proteins=5,
    )

    print("\n" + "=" * 72)
    print("✅  Pipeline complete!")
    print("=" * 72)
    print(f"All outputs in: {out}/")
    print(f"  • accessions.txt               – {n_prot} processed accessions")
    print(f"  • activation_matrix.npy         – ({n_prot}, {n_nodes})")
    print(f"  • geometry_matrix.npy           – ({n_prot}, {n_geom})")
    print(f"  • top_activating_per_node.yaml  – top-{args.top_k} per node")
    print(f"  • geometry_sae_correlations.yaml")
    print(f"  • correlation_heatmap.png + scatter/bar plots")
    print("=" * 72)


if __name__ == "__main__":
    main()
