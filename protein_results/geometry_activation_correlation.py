from __future__ import annotations

"""
Correlation analysis between geometric features and SAE node activations.

For each SAE feature (node), this script computes the per-protein activation
strength, pairs it with the protein's geometric features, and measures the
correlation. This reveals which SAE nodes are sensitive to specific geometric
properties of proteins (writhe, curvature, torsion, etc.).

Usage:
    python geometry_activation_correlation.py

Requirements:
    - The pre-computed protein geometry dataset  (protein_dataset.npy)
    - Batch YAML files containing PDB text       (results/)
    - Per-feature max examples YAML               (Per_feature_max_examples.yaml)
    - A trained SAE model                         (trained_models/fiery-sweep/)
"""

import sys
from pathlib import Path
from functools import lru_cache

import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy import stats

# ---------------------------------------------------------------------------
# Make sure the parent package is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pdb_plotter

from pdb_plotter import ca_backbone, detect_alpha_helices_from_ca
from geometry.compute_geometric_features import (
    writhe, vassiliev, average_curvature, average_torsion,
    gyration_asphericity, kink_index, radius_of_gyration,
    local_planarity_score, end_to_end_distance, helical_consistency,
    helix_statistics,
)
from proteinlens.sae.inference import load_sae, get_sae_feats_in_batches
from proteinlens.embedders.esm import ESM
from proteinlens.utils import get_device

# ========================== CONFIGURATION ==================================
SAE_DIR          = ROOT / "trained_models" / "fiery-sweep"
BATCH_DIR        = Path(__file__).resolve().parent / "results"
GROUPS_YAML      = Path(__file__).resolve().parent / "Per_feature_max_examples.yaml"
DATASET_CACHE    = Path(__file__).resolve().parent / "protein_dataset.npy"
ESM_MODEL_NAME   = "facebook/esm2_t6_8M_UR50D"
ESM_LAYER        = 3          # layer used for SAE training (from config)
FIRST_BATCH      = 0
LAST_BATCH       = 21
TOP_K            = 20         # how many features to highlight in the summary
# ===========================================================================

# Names of geometric features in the order stored in protein_dataset.npy
# Column layout: [group_id, accession, wr, v2, cur, tor, kink, ga, p_m, p_s,
#                 d_m, d_s, rog, planar, end, ta, bc, L]
GEOM_FEATURE_NAMES = [
    "writhe", "vassiliev_v2", "avg_curvature", "avg_torsion",
    "kink_index", "gyration_asphericity",
    "helix_parallel_mean", "helix_parallel_std",
    "helix_dist_mean", "helix_dist_std",
    "radius_of_gyration", "local_planarity",
    "end_to_end_distance", "tangent_alignment", "binormal_consistency",
    "chain_length",
]

# Column indices in the .npy file (0 = group, 1 = accession, 2..17 = features)
GEOM_COL_START = 2


# ======================== Batch / PDB helpers ==============================

def list_batch_paths(batch_dir: Path, first: int = 0, last: int = 21):
    return [p for i in range(first, last + 1)
            if (p := batch_dir / f"batch_{i}.yaml").is_file()]


@lru_cache(maxsize=None)
def _load_yaml_cached(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def find_pdb_in_batches(entry_key: str, batch_paths: tuple):
    for path in batch_paths:
        data = _load_yaml_cached(path)
        if entry_key in data:
            val = data[entry_key]
            if isinstance(val, dict) and "pdb" in val and val["pdb"]:
                return val["pdb"].replace("\r\n", "\n").replace("\\\n", "")
            raise ValueError(f"Entry '{entry_key}' in {path.name} has no 'pdb' text.")
    raise KeyError(f"Entry '{entry_key}' not found in any batch YAML.")


def load_groups(groups_yaml_path: Path) -> dict:
    data = yaml.safe_load(groups_yaml_path.read_text(encoding="utf-8")) or {}
    groups = {}
    for k, v in data.items():
        gid = int(k) if isinstance(k, str) and k.isdigit() else int(k)
        groups[gid] = list(v or [])
    return groups


# ==================== Geometry computation =================================

def compute_geometry_for_protein(pdb_text: str, chain_id=None):
    """Compute all geometric features for a single protein.

    Returns a dict  {feature_name: float}  or None on failure.
    """
    try:
        ca = ca_backbone(pdb_text, chain_id=chain_id)
        plt.close("all")  # ca_backbone opens a plot – close it
    except Exception:
        return None
    if ca is None or len(ca) < 4:
        return None

    try:
        helices = detect_alpha_helices_from_ca(ca)
        wr_d = writhe(ca, ca)
        wr = float(np.sum(wr_d))
        _v2 = float(vassiliev(wr_d))
        cur = float(average_curvature(ca))
        tor = float(average_torsion(ca))
        ki = float(kink_index(ca))
        ga = float(gyration_asphericity(ca))
        p_m, p_s, d_m, d_s = helix_statistics(ca, helices)
        rog = float(radius_of_gyration(ca))
        planar = float(local_planarity_score(ca))
        end = float(end_to_end_distance(ca))
        ta, bc = helical_consistency(ca)
        L = float(len(ca))
    except Exception:
        return None

    return dict(zip(GEOM_FEATURE_NAMES,
                    [wr, _v2, cur, tor, ki, ga,
                     float(p_m), float(p_s), float(d_m), float(d_s),
                     rog, planar, end, float(ta), float(bc), L]))


# =================== SAE activation helpers ================================

def get_protein_sequence_from_pdb(pdb_text: str) -> str:
    """Extract the amino-acid sequence from PDB ATOM records (CA atoms)."""
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    seen = set()
    seq = []
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


def compute_per_protein_activation(
    sae, embedder, pdb_text: str, device: str, chain_id=None,
) -> np.ndarray | None:
    """
    Run ESM + SAE on one protein and return the *mean* activation per SAE
    feature across all residue positions.

    Returns shape (dict_size,) or None on failure.
    """
    seq = get_protein_sequence_from_pdb(pdb_text)
    if len(seq) < 4:
        return None

    try:
        # Get ESM embeddings  →  (seq_len, activation_dim)
        emb = embedder.embed_single_sequence(seq, layer=ESM_LAYER)  # np array
        emb_t = torch.tensor(emb, dtype=torch.float32, device=device)

        # Encode through SAE  →  (seq_len, dict_size)
        with torch.no_grad():
            feats = sae.encode(emb_t)  # full encode, all features
        # Mean activation across residues for each feature
        mean_act = feats.mean(dim=0).detach().cpu().numpy()
        return mean_act
    except Exception as e:
        print(f"  [activation error] {e}")
        return None


# ===================== Main analysis =======================================

def build_paired_dataset(
    sae, embedder, device,
    groups: dict,
    batch_paths: tuple,
    chain_id=None,
    max_proteins: int | None = None,
):
    """
    Iterate over proteins in the groups file, compute geometry + SAE
    activations, and return paired arrays.

    Returns
    -------
    accessions : list[str]
    group_ids  : list[int]
    geom_matrix : np.ndarray   (n_proteins, n_geom_features)
    act_matrix  : np.ndarray   (n_proteins, dict_size)
    """
    accessions_out, gids_out = [], []
    geom_rows, act_rows = [], []
    seen = set()

    # Flatten all accessions from groups
    all_items = []
    for gid in sorted(groups.keys()):
        for acc in groups[gid]:
            if acc not in seen:
                seen.add(acc)
                all_items.append((gid, acc))

    if max_proteins is not None:
        all_items = all_items[:max_proteins]

    total = len(all_items)
    print(f"  Processing {total} unique proteins …")
    n_success = 0
    n_skip = 0

    for idx, (gid, acc) in enumerate(all_items):
        if idx % 10 == 0 or idx == total - 1:
            print(f"  [{idx+1:>5d}/{total}]  ok={n_success}  skip={n_skip}  "
                  f"current={acc}")

        try:
            pdb_text = find_pdb_in_batches(acc, batch_paths)
        except (KeyError, ValueError) as e:
            n_skip += 1
            continue

        geom = compute_geometry_for_protein(pdb_text, chain_id=chain_id)
        if geom is None:
            n_skip += 1
            continue

        act = compute_per_protein_activation(sae, embedder, pdb_text, device, chain_id)
        if act is None:
            n_skip += 1
            continue

        n_success += 1
        accessions_out.append(acc)
        gids_out.append(gid)
        geom_rows.append([geom[k] for k in GEOM_FEATURE_NAMES])
        act_rows.append(act)

    geom_matrix = np.array(geom_rows, dtype=float)
    act_matrix = np.vstack(act_rows).astype(float)
    return accessions_out, gids_out, geom_matrix, act_matrix


def correlation_analysis(geom_matrix, act_matrix, geom_names, top_k=20):
    """
    Compute Pearson and Spearman correlations between every (geom feature,
    SAE node) pair.

    Returns
    -------
    pearson_r  : np.ndarray  (n_geom, dict_size)
    pearson_p  : np.ndarray  (n_geom, dict_size)
    spearman_r : np.ndarray  (n_geom, dict_size)
    spearman_p : np.ndarray  (n_geom, dict_size)
    summary    : list[dict]  top correlations sorted by |r|
    """
    n_geom = geom_matrix.shape[1]
    n_nodes = act_matrix.shape[1]
    pearson_r = np.zeros((n_geom, n_nodes))
    pearson_p = np.ones((n_geom, n_nodes))
    spearman_r = np.zeros((n_geom, n_nodes))
    spearman_p = np.ones((n_geom, n_nodes))

    for gi in range(n_geom):
        g = geom_matrix[:, gi]
        valid = np.isfinite(g)
        if valid.sum() < 10:
            continue
        gv = g[valid]
        for ni in range(n_nodes):
            a = act_matrix[valid, ni]
            # Skip dead features (no variance)
            if a.std() < 1e-12 or gv.std() < 1e-12:
                continue
            pr, pp = stats.pearsonr(gv, a)
            sr, sp = stats.spearmanr(gv, a)
            pearson_r[gi, ni] = pr
            pearson_p[gi, ni] = pp
            spearman_r[gi, ni] = sr
            spearman_p[gi, ni] = sp

    # Build a flat summary of the strongest correlations
    summary = []
    for gi in range(n_geom):
        for ni in range(n_nodes):
            if pearson_p[gi, ni] < 0.05:
                summary.append({
                    "geom_feature": geom_names[gi],
                    "sae_node": ni,
                    "pearson_r": float(pearson_r[gi, ni]),
                    "pearson_p": float(pearson_p[gi, ni]),
                    "spearman_r": float(spearman_r[gi, ni]),
                    "spearman_p": float(spearman_p[gi, ni]),
                })
    summary.sort(key=lambda d: abs(d["pearson_r"]), reverse=True)

    return pearson_r, pearson_p, spearman_r, spearman_p, summary


# ======================== Plotting =========================================

def plot_correlation_heatmap(pearson_r, geom_names, save_path=None):
    """Heatmap of Pearson r across (geometric features × SAE nodes)."""
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
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Heatmap saved → {save_path}")
    plt.show()


def plot_top_scatter(geom_matrix, act_matrix, summary, geom_names, top_k=6,
                     save_path=None):
    """Scatter plots for the top-k strongest correlations."""
    k = min(top_k, len(summary))
    if k == 0:
        print("No significant correlations to plot.")
        return
    cols = min(3, k)
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_2d(axes)
    for idx in range(k):
        ax = axes.flat[idx]
        entry = summary[idx]
        gi = geom_names.index(entry["geom_feature"])
        ni = entry["sae_node"]
        x = geom_matrix[:, gi]
        y = act_matrix[:, ni]
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], s=12, alpha=0.6, edgecolors="none")
        ax.set_xlabel(entry["geom_feature"])
        ax.set_ylabel(f"SAE node {ni} act.")
        ax.set_title(f"r={entry['pearson_r']:.3f}  ρ={entry['spearman_r']:.3f}",
                      fontsize=10)
    # Hide unused axes
    for idx in range(k, rows * cols):
        axes.flat[idx].set_visible(False)
    plt.suptitle("Top Geometry ↔ SAE Activation Correlations", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Scatter plots saved → {save_path}")
    plt.show()


def plot_per_feature_bar(pearson_r, geom_names, feature_name, top_k=20,
                         save_path=None):
    """
    Bar chart: for a chosen geometric feature, show the SAE nodes with the
    strongest (positive and negative) correlation.
    """
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
        print(f"Bar chart saved → {save_path}")
    plt.show()


# ========================= Entry point =====================================

def main():
    print("=" * 72)
    print("Geometry ↔ SAE Activation Correlation Analysis")
    print("=" * 72)
    device = get_device()
    print(f"[✓] Device: {device}")
    print(f"[i] SAE dir:   {SAE_DIR}")
    print(f"[i] ESM model: {ESM_MODEL_NAME}  layer {ESM_LAYER}")
    print(f"[i] Batches:   {BATCH_DIR}  ({FIRST_BATCH}–{LAST_BATCH})")
    print()

    # 1. Load trained SAE
    print("[1/8] Loading SAE …")
    sae = load_sae(SAE_DIR, device=device)
    sae.eval()
    print(f"  SAE: {sae.__class__.__name__}  dict_size={sae.dict_size}  "
          f"activation_dim={sae.activation_dim}")
    print("[✓] SAE loaded successfully.\n")

    # 2. Load ESM embedder
    print("[2/8] Loading ESM embedder …")
    embedder = ESM(model_name=ESM_MODEL_NAME, device=device)

    print("[✓] ESM embedder loaded successfully.\n")

    # 3. Load protein groups & batch paths
    print("[3/8] Loading protein groups & batch paths …")
    groups = load_groups(GROUPS_YAML)
    batch_paths = tuple(list_batch_paths(BATCH_DIR, FIRST_BATCH, LAST_BATCH))
    n_total_proteins = sum(len(v) for v in groups.values())
    print(f"  {len(groups)} feature groups, {len(batch_paths)} batch files, "
          f"{n_total_proteins} total protein entries")
    print("[✓] Groups loaded.\n")

    # 4. Build paired geometry + activation dataset
    print("[4/8] Building paired geometry + activation dataset …")
    cache_geom = Path(__file__).resolve().parent / "geom_act_cache.npz"
    if cache_geom.exists():
        print(f"  Found cache file: {cache_geom.name} — loading …")
        npz = np.load(cache_geom, allow_pickle=True)
        accessions = npz["accessions"].tolist()
        group_ids = npz["group_ids"].tolist()
        geom_matrix = npz["geom_matrix"]
        act_matrix = npz["act_matrix"]
    else:
        accessions, group_ids, geom_matrix, act_matrix = build_paired_dataset(
            sae, embedder, device, groups, batch_paths, chain_id=None,
        )
        np.savez(cache_geom,
                 accessions=np.array(accessions, dtype=object),
                 group_ids=np.array(group_ids),
                 geom_matrix=geom_matrix,
                 act_matrix=act_matrix)
        print(f"Paired dataset cached → {cache_geom.name}")

    n_prot = geom_matrix.shape[0]
    n_geom = geom_matrix.shape[1]
    n_nodes = act_matrix.shape[1]
    print(f"[✓] Paired dataset ready: {n_prot} proteins × {n_geom} geom features × "
          f"{n_nodes} SAE nodes\n")

    # 5. Correlation analysis
    print("[5/8] Computing Pearson & Spearman correlations …")
    pearson_r, pearson_p, spearman_r, spearman_p, summary = correlation_analysis(
        geom_matrix, act_matrix, GEOM_FEATURE_NAMES, top_k=TOP_K,
    )

    print(f"[✓] Correlations computed. {len(summary)} significant pairs found (p < 0.05).\n")

    # 6. Print top correlations
    print(f"{'='*72}")
    print(f"Top {min(TOP_K, len(summary))} strongest geometry ↔ SAE node correlations")
    print(f"{'='*72}")
    print(f"{'Geom Feature':<28s} {'Node':>6s} {'Pearson r':>10s} {'p-value':>10s} "
          f"{'Spearman ρ':>10s}")
    print("-" * 72)
    for entry in summary[:TOP_K]:
        print(f"{entry['geom_feature']:<28s} {entry['sae_node']:>6d} "
              f"{entry['pearson_r']:>10.4f} {entry['pearson_p']:>10.2e} "
              f"{entry['spearman_r']:>10.4f}")

    # 7. Save summary to YAML
    print("\n[6/8] Saving correlation summary to YAML …")
    out_yaml = Path(__file__).resolve().parent / "geometry_sae_correlations.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(summary[:100], f, default_flow_style=False)
    print(f"\nFull summary (top 100) saved → {out_yaml.name}")

    # 8. Plots
    print("\n[7/8] Generating plots …")
    out_dir = Path(__file__).resolve().parent
    plot_correlation_heatmap(
        pearson_r, GEOM_FEATURE_NAMES,
        save_path=out_dir / "correlation_heatmap.png",
    )
    plot_top_scatter(
        geom_matrix, act_matrix, summary, GEOM_FEATURE_NAMES, top_k=6,
        save_path=out_dir / "top_correlation_scatter.png",
    )

    # Bar charts for a few particularly interesting geometric features
    for feat in ["writhe", "avg_curvature", "radius_of_gyration", "kink_index"]:
        plot_per_feature_bar(
            pearson_r, GEOM_FEATURE_NAMES, feat, top_k=TOP_K,
            save_path=out_dir / f"bar_{feat}.png",
        )

    print("\n[8/8] All done!")
    print("=" * 72)
    print(f"Outputs in: {out_dir}")
    print("  • geometry_sae_correlations.yaml")
    print("  • correlation_heatmap.png")
    print("  • top_correlation_scatter.png")
    print("  • bar_<feature>.png  (writhe, curvature, RoG, kink)")
    print("=" * 72)


if __name__ == "__main__":
    main()
