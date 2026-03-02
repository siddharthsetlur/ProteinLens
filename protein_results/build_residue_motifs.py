from __future__ import annotations

"""
Residue-level motif discovery: Geometry ↔ SAE activations.

Unlike build_activation_multiset.py which averages SAE activations to one
scalar per protein (destroying the spatial signal), this pipeline operates
at **residue resolution**:

  1.  For every protein, keep the full per-residue SAE activation profile
      (seq_len, dict_size) and the per-residue local geometry profiles.
  2.  For each SAE node, collect the local Cα fragments at positions where
      the node fires above threshold (and background fragments where it
      doesn't).
  3.  Kabsch-align the top-activating fragments and measure pairwise RMSD.
      Low RMSD → the node detects a consistent 3D structural motif.  The
      mean aligned structure IS the motif template.
  4.  Train a shallow decision tree on local geometric features to classify
      activating vs. non-activating positions → produces human-readable
      rules like "fires when curvature > 0.5 AND frac_pos_torsion > 0.7".
  5.  Compute categorical enrichment: for each structural category (helix,
      hairpin, kink, strand, loop, …), what is its fold-enrichment at
      activated positions vs. background?

The result per SAE node is:
  • A 3D motif template (PDB-like average structure)
  • A pairwise-RMSD distribution (low = definitive motif)
  • A decision-tree rule set (human-readable definition)
  • A categorical enrichment table

Usage:
    python build_residue_motifs.py --resume --max-proteins 1000
    python build_residue_motifs.py --mixed-organisms --max-proteins 5000
    python build_residue_motifs.py --accession-list ids.txt

All heavy data (pdb_cache/) is shared with the other build_* scripts via
--output-dir, so --resume works across pipelines.
"""

import argparse
import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import requests
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold
from sklearn.metrics import (f1_score, roc_auc_score,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_activation_dataset import (
    SAE_DIR, ESM_MODEL_NAME, ESM_LAYER, TOP_K_PER_NODE, MAX_SEQ_LEN,
    fetch_mixed_organism_accessions,
    fetch_swissprot_accessions,
    load_accessions_from_file,
    load_accessions_from_fasta,
    fetch_sequence,
    fetch_alphafold_pdb,
    get_protein_sequence_from_pdb,
)
from pdb_plotter import ca_backbone, detect_alpha_helices_from_ca
from kabsch_top_alignment import kabsch_align, compute_rmsd
from geometry.compute_geometric_features import (
    ca_curvature_profile,
    ca_torsion_profile,
    local_planarity_profile,
    tangent_vectors,
    writhe,
    kink_index as global_kink_index,
    hairpin_score as global_hairpin_score,
    extended_fraction as global_extended_fraction,
)
from proteinlens.sae.inference import load_sae
from proteinlens.embedders.esm import ESM
from proteinlens.utils import get_device


# ====================== CONSTANTS ==========================================

# Window half-size for fragment extraction (total window = 2*HALF_W + 1)
HALF_W = 5
WINDOW_SIZE = 2 * HALF_W + 1  # 21 residues

# For decision tree / enrichment, only consider nodes that activate on at
# least this many residue positions (across all proteins combined).
MIN_ACTIVATED_POSITIONS = 200

# How many top SAE nodes to analyse (ranked by fragment RMSD consistency)
TOP_N_NODES = 50

# Fragment superposition: take top-K most activated fragments per node
FRAG_TOP_K = 100

# Activation threshold: only count positions above this quantile of the
# node's nonzero activations as "activated"
ACT_QUANTILE = 0.80

# Background sampling ratio for decision tree (neg:pos)
BG_RATIO = 3

# Structural category thresholds
CURVATURE_TURN_THR = 0.55
KINK_ANGLE_THR = 60.0  # degrees
EXTENDED_ALIGN_THR = 0.9
EXTENDED_CURV_THR = 0.2

COLOURS = [
    "#2980b9", "#e74c3c", "#27ae60", "#f39c12", "#8e44ad",
    "#1abc9c", "#d35400", "#c0392b", "#2c3e50", "#16a085",
]

# Names for the per-residue local geometric feature vector
LOCAL_GEOM_NAMES = [
    # ── Original whole-window features (primary scale, half_w) ──
    "curvature_mean", "curvature_max", "curvature_std",
    "torsion_mean", "torsion_std", "torsion_frac_pos",
    "planarity_mean", "planarity_std",
    "tangent_alignment",
    "end_to_end_ratio",
    # ── Sub-window thirds (N-flank, centre, C-flank) ──
    "curv_N_third", "curv_centre_third", "curv_C_third",
    "tors_N_third", "tors_centre_third", "tors_C_third",
    "plan_N_third", "plan_centre_third", "plan_C_third",
    # ── Multi-scale: narrow window (half_w // 2) ──
    "narrow_curvature_mean", "narrow_curvature_max",
    "narrow_torsion_mean", "narrow_torsion_std",
    "narrow_tangent_alignment", "narrow_end_to_end_ratio",
    # ── Multi-scale: wide window (half_w * 2, clamped to chain) ──
    "wide_curvature_mean", "wide_curvature_max",
    "wide_torsion_mean", "wide_torsion_std",
    "wide_tangent_alignment", "wide_end_to_end_ratio",
    # ── Local contact density ──
    "contact_density_8A",   # Cα atoms within 8 Å outside the window
    "contact_density_12A",  # Cα atoms within 12 Å outside the window
    # ── Amino acid composition (window) ──
    "frac_hydrophobic",     # A, V, I, L, M, F, W, P
    "frac_charged",         # D, E, K, R
    "frac_polar",           # S, T, N, Q, Y, C, H
    "frac_gly_pro",         # G, P (backbone flexibility)
    "frac_aromatic",        # F, W, Y, H
]

# Structural category labels
CATEGORY_NAMES = [
    "alpha_helix",
    "tight_turn",
    "kink",
    "extended_strand",
    "beta_hairpin_like",
    "loop",
]


# ====================== PER-RESIDUE GEOMETRY ================================

def compute_residue_profiles(ca: np.ndarray, helices: list) -> dict:
    """
    Compute per-residue geometric profiles from Cα coordinates.

    Returns a dict with:
        curvature  : (N,) array
        torsion    : (N,) array
        planarity  : (N,) array
        tangents   : (N, 3) array
        helix_mask : (N,) bool array  (True if position is in a helix)
        categories : (N,) int array   (index into CATEGORY_NAMES)
    """
    n = len(ca)
    kappa = ca_curvature_profile(ca)    # (N,)
    tau = ca_torsion_profile(ca)        # (N,)
    planar = local_planarity_profile(ca, w=7)  # (N,)
    T = tangent_vectors(ca)             # (N, 3)

    # ── Helix mask ──
    helix_mask = np.zeros(n, dtype=bool)
    for s, e in helices:
        helix_mask[s:e] = True

    # ── Structural categories (per residue) ──
    categories = np.full(n, 5, dtype=int)  # default = "loop" (index 5)

    # Alpha helix
    categories[helix_mask] = 0

    # Tight turn: high curvature & not in helix
    turn_mask = (kappa > CURVATURE_TURN_THR) & ~helix_mask
    categories[turn_mask] = 1

    # Kink: large tangent angle
    cos_thr = np.cos(np.radians(KINK_ANGLE_THR))
    for i in range(n - 1):
        dot = np.dot(T[i], T[i + 1])
        if dot < cos_thr and not helix_mask[i]:
            categories[i] = 2  # kink

    # Extended strand: high tangent alignment + low curvature, not in helix
    for i in range(n - 1):
        dot = np.dot(T[i], T[i + 1])
        if dot > EXTENDED_ALIGN_THR and kappa[i] < EXTENDED_CURV_THR and not helix_mask[i]:
            categories[i] = 3  # extended

    # Beta-hairpin-like: compact window with tangent reversal
    # (simple proxy using a smaller window than the global hairpin_score)
    hw = 8
    for i in range(hw, n - hw):
        if helix_mask[i]:
            continue
        seg = ca[i - hw:i + hw + 1]
        contour = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
        ee = float(np.linalg.norm(seg[-1] - seg[0]))
        compact = 1.0 - min(1.0, ee / max(1e-8, contour))
        rev = 0.5 * (1.0 - float(np.dot(T[max(0, i - hw)], T[min(n - 1, i + hw)])))
        if compact * rev > 0.25:
            categories[i] = 4  # hairpin-like

    return {
        "curvature": kappa,
        "torsion": tau,
        "planarity": planar,
        "tangents": T,
        "helix_mask": helix_mask,
        "categories": categories,
    }


def extract_local_feature_vector(
    profiles: dict, ca: np.ndarray, pos: int, half_w: int = HALF_W,
    sequence: str | None = None,
) -> np.ndarray | None:
    """
    Extract a fixed-length feature vector describing the local geometry
    in a window [pos - half_w, pos + half_w] inclusive.

    Features include:
      • Whole-window summary statistics (curvature, torsion, planarity,
        tangent alignment, end-to-end ratio)
      • Sub-window thirds (N-flank / centre / C-flank) for curvature,
        torsion, and planarity — captures spatial pattern within the window
      • Multi-scale: same statistics at half and double the window size
      • Local contact density (Cα atoms within 8 Å and 12 Å outside window)
      • Amino acid composition fractions (if sequence is provided)

    Returns a 1-D numpy array of shape (len(LOCAL_GEOM_NAMES),) or None
    if the position is too close to an end.
    """
    n = len(ca)
    if pos < half_w or pos >= n - half_w:
        return None

    s, e = pos - half_w, pos + half_w + 1
    kappa_w = profiles["curvature"][s:e]
    tau_w = profiles["torsion"][s:e]
    planar_w = profiles["planarity"][s:e]
    T = profiles["tangents"]

    # ── Helper: tangent alignment for a range ──
    def _tang_align(s_i: int, e_i: int) -> float:
        val = 0.0
        cnt = 0
        for i in range(max(0, s_i), min(n - 1, e_i) - 1):
            val += float(np.dot(T[i], T[i + 1]))
            cnt += 1
        return val / max(1, cnt)

    # ── Helper: end-to-end ratio for a range ──
    def _ee_ratio(s_i: int, e_i: int) -> float:
        seg = ca[s_i:e_i]
        if len(seg) < 2:
            return 1.0
        contour = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
        ee = float(np.linalg.norm(seg[-1] - seg[0]))
        return ee / max(1e-8, contour)

    # ═══════════════════════════════════════════════════════════════════
    # 1. Original whole-window features (10 features)
    # ═══════════════════════════════════════════════════════════════════
    tang_align = _tang_align(s, e)
    ee_ratio = _ee_ratio(s, e)

    core_tau = tau_w[2:-2] if len(tau_w) > 4 else tau_w
    frac_pos = float(np.mean(core_tau > 0)) if len(core_tau) > 0 else 0.5

    feats_original = [
        float(np.mean(kappa_w)),
        float(np.max(kappa_w)),
        float(np.std(kappa_w)),
        float(np.mean(tau_w)),
        float(np.std(tau_w)),
        frac_pos,
        float(np.mean(planar_w)),
        float(np.std(planar_w)),
        tang_align,
        ee_ratio,
    ]

    # ═══════════════════════════════════════════════════════════════════
    # 2. Sub-window thirds (9 features)
    # ═══════════════════════════════════════════════════════════════════
    w_len = e - s
    t1 = w_len // 3
    t2 = 2 * t1
    kN, kC, kM = kappa_w[:t1], kappa_w[t1:t2], kappa_w[t2:]
    tN, tC_third, tM = tau_w[:t1], tau_w[t1:t2], tau_w[t2:]
    pN, pC, pM = planar_w[:t1], planar_w[t1:t2], planar_w[t2:]

    feats_thirds = [
        float(np.mean(kN)) if len(kN) > 0 else 0.0,
        float(np.mean(kC)) if len(kC) > 0 else 0.0,
        float(np.mean(kM)) if len(kM) > 0 else 0.0,
        float(np.mean(tN)) if len(tN) > 0 else 0.0,
        float(np.mean(tC_third)) if len(tC_third) > 0 else 0.0,
        float(np.mean(tM)) if len(tM) > 0 else 0.0,
        float(np.mean(pN)) if len(pN) > 0 else 0.0,
        float(np.mean(pC)) if len(pC) > 0 else 0.0,
        float(np.mean(pM)) if len(pM) > 0 else 0.0,
    ]

    # ═══════════════════════════════════════════════════════════════════
    # 3. Multi-scale: narrow (half_w // 2) and wide (half_w * 2)
    #    6 features each = 12 features total
    # ═══════════════════════════════════════════════════════════════════
    def _scale_feats(hw_s: int) -> list[float]:
        """Compute 6 summary features at a given half-window scale."""
        s_s = max(0, pos - hw_s)
        e_s = min(n, pos + hw_s + 1)
        k_s = profiles["curvature"][s_s:e_s]
        t_s = profiles["torsion"][s_s:e_s]
        return [
            float(np.mean(k_s)) if len(k_s) > 0 else 0.0,
            float(np.max(k_s)) if len(k_s) > 0 else 0.0,
            float(np.mean(t_s)) if len(t_s) > 0 else 0.0,
            float(np.std(t_s)) if len(t_s) > 0 else 0.0,
            _tang_align(s_s, e_s),
            _ee_ratio(s_s, e_s),
        ]

    hw_narrow = max(1, half_w // 2)
    hw_wide = half_w * 2
    feats_narrow = _scale_feats(hw_narrow)
    feats_wide = _scale_feats(hw_wide)

    # ═══════════════════════════════════════════════════════════════════
    # 4. Local contact density (2 features)
    # ═══════════════════════════════════════════════════════════════════
    centre = ca[pos]
    # Exclude residues inside the primary window
    outside_mask = np.ones(n, dtype=bool)
    outside_mask[s:e] = False
    if np.any(outside_mask):
        dists = np.linalg.norm(ca[outside_mask] - centre, axis=1)
        contact_8 = float(np.sum(dists < 8.0))
        contact_12 = float(np.sum(dists < 12.0))
    else:
        contact_8 = 0.0
        contact_12 = 0.0

    feats_contact = [contact_8, contact_12]

    # ═══════════════════════════════════════════════════════════════════
    # 5. Amino acid composition (5 features)
    # ═══════════════════════════════════════════════════════════════════
    _HYDROPHOBIC = set("AVILMFWP")
    _CHARGED = set("DEKR")
    _POLAR = set("STNQYCH")
    _GLY_PRO = set("GP")
    _AROMATIC = set("FWYH")

    if sequence is not None and len(sequence) >= e:
        window_seq = sequence[s:e]
        wl = len(window_seq)
        frac_hydro = sum(1 for aa in window_seq if aa in _HYDROPHOBIC) / max(1, wl)
        frac_chg = sum(1 for aa in window_seq if aa in _CHARGED) / max(1, wl)
        frac_pol = sum(1 for aa in window_seq if aa in _POLAR) / max(1, wl)
        frac_gp = sum(1 for aa in window_seq if aa in _GLY_PRO) / max(1, wl)
        frac_aro = sum(1 for aa in window_seq if aa in _AROMATIC) / max(1, wl)
    else:
        frac_hydro = frac_chg = frac_pol = frac_gp = frac_aro = 0.0

    feats_aa = [frac_hydro, frac_chg, frac_pol, frac_gp, frac_aro]

    # ═══════════════════════════════════════════════════════════════════
    # Concatenate all features
    # ═══════════════════════════════════════════════════════════════════
    all_feats = (
        feats_original + feats_thirds + feats_narrow + feats_wide
        + feats_contact + feats_aa
    )
    return np.array(all_feats, dtype=np.float64)


# ===================== PER-RESIDUE SAE ACTIVATION ===========================

def compute_residue_activations(
    sae, embedder, sequence: str, device: str, esm_layer: int,
) -> np.ndarray | None:
    """
    Run ESM → SAE on one protein and return the FULL per-residue activation
    matrix.  Shape: (seq_len, dict_size).

    Unlike the build_activation_dataset version which averages over residues,
    this retains the spatial signal.
    """
    if len(sequence) < 4 or len(sequence) > MAX_SEQ_LEN:
        return None
    try:
        emb = embedder.embed_single_sequence(sequence, layer=esm_layer)
        emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
        with torch.no_grad():
            feats = sae.encode(emb_t)  # (seq_len, dict_size)
        return feats.detach().cpu().numpy()
    except Exception as e:
        print(f"  [activation error] {e}")
        return None


# =================== DATA COLLECTION PER PROTEIN ===========================

def process_one_protein(
    acc: str,
    pdb_text: str,
    sae,
    embedder,
    device: str,
    esm_layer: int,
) -> dict | None:
    """
    For a single protein, compute Cα coordinates, per-residue geometry
    profiles, structural categories, and full SAE activation matrix.

    Returns a dict with all per-residue data, or None on failure.
    """
    try:
        ca = ca_backbone(pdb_text, chain_id=None)
        plt.close("all")
    except Exception:
        return None
    if ca is None or len(ca) < WINDOW_SIZE + 4:
        return None

    seq = get_protein_sequence_from_pdb(pdb_text)
    if len(seq) < 4:
        return None

    act_matrix = compute_residue_activations(sae, embedder, seq, device, esm_layer)
    if act_matrix is None:
        return None

    # ESM may include special tokens — align lengths
    n_ca = len(ca)
    n_act = act_matrix.shape[0]
    n = min(n_ca, n_act)
    ca = ca[:n]
    act_matrix = act_matrix[:n]

    helices = detect_alpha_helices_from_ca(ca)
    profiles = compute_residue_profiles(ca, helices)

    return {
        "accession": acc,
        "ca": ca,                        # (n, 3)
        "act_matrix": act_matrix,        # (n, dict_size)
        "profiles": profiles,
        "sequence": seq[:n],
        "n_residues": n,
    }


# ============= COLLECT ACTIVATED FRAGMENTS ACROSS ALL PROTEINS ==============

def collect_node_fragments(
    protein_data: list[dict],
    node_idx: int,
    half_w: int = HALF_W,
    act_quantile: float = ACT_QUANTILE,
    max_fragments: int = FRAG_TOP_K,
    bg_ratio: int = BG_RATIO,
) -> dict:
    """
    Across all proteins, collect:
      - activated fragments  (Cα coords + local features) at positions where
        SAE node fires above quantile threshold
      - background fragments at positions where it doesn't

    Returns a dict with lists of fragments ready for motif analysis.
    """
    # First pass: gather all nonzero activations for this node to set threshold
    all_nonzero = []
    for pdata in protein_data:
        col = pdata["act_matrix"][:, node_idx]
        nz = col[col > 0]
        if len(nz) > 0:
            all_nonzero.append(nz)

    if not all_nonzero:
        return {"activated": [], "background": [], "threshold": 0.0, "n_total_active": 0}

    all_nonzero = np.concatenate(all_nonzero)
    if len(all_nonzero) < MIN_ACTIVATED_POSITIONS:
        return {"activated": [], "background": [], "threshold": 0.0,
                "n_total_active": len(all_nonzero)}

    threshold = float(np.quantile(all_nonzero, act_quantile))
    if threshold <= 0:
        threshold = float(np.median(all_nonzero))

    activated: list[dict] = []
    background: list[dict] = []
    hard_negatives: list[dict] = []  # low-but-nonzero: geometrically similar near-misses

    for pdata in protein_data:
        ca = pdata["ca"]
        profiles = pdata["profiles"]
        col = pdata["act_matrix"][:, node_idx]
        n = pdata["n_residues"]
        seq = pdata.get("sequence", None)

        for pos in range(half_w, n - half_w):
            feat_vec = extract_local_feature_vector(profiles, ca, pos, half_w, sequence=seq)
            if feat_vec is None:
                continue

            fragment = ca[pos - half_w: pos + half_w + 1].copy()
            category = int(profiles["categories"][pos])

            entry = {
                "accession": pdata["accession"],
                "position": pos,
                "fragment": fragment,       # (window_size, 3)
                "features": feat_vec,       # (n_local_feats,)
                "category": category,
                "activation": float(col[pos]),
            }

            if col[pos] >= threshold:
                activated.append(entry)
            elif 0 < col[pos] < threshold:
                # Hard negatives: node fires weakly here — geometrically
                # similar to activated but below threshold
                hard_negatives.append(entry)
            else:
                # True background: node doesn't fire at all here
                background.append(entry)

    # Sort activated by activation strength descending
    activated.sort(key=lambda x: -x["activation"])

    # Build combined background: mix of hard negatives + true zeroes
    # Target ratio: ~50% hard negatives, ~50% true background
    rng = np.random.default_rng(42)
    n_bg_total = min(len(background) + len(hard_negatives),
                     len(activated) * bg_ratio)
    n_hard = min(len(hard_negatives), n_bg_total // 2)
    n_zero = min(len(background), n_bg_total - n_hard)

    if n_hard > 0 and len(hard_negatives) > n_hard:
        idx = rng.choice(len(hard_negatives), size=n_hard, replace=False)
        hard_negatives = [hard_negatives[i] for i in idx]
    else:
        hard_negatives = hard_negatives[:n_hard]

    if n_zero > 0 and len(background) > n_zero:
        idx = rng.choice(len(background), size=n_zero, replace=False)
        background = [background[i] for i in idx]
    else:
        background = background[:n_zero]

    combined_background = hard_negatives + background

    return {
        "activated": activated[:max_fragments * 5],  # keep extras for analysis
        "background": combined_background,
        "threshold": threshold,
        "n_total_active": int(len(all_nonzero)),
        "n_hard_negatives": len(hard_negatives),
    }


# ====================== FRAGMENT SUPERPOSITION ==============================

def superpose_fragments(
    activated: list[dict],
    top_k: int = FRAG_TOP_K,
) -> dict:
    """
    Take the top-K most activated fragments, Kabsch-align them to the
    strongest one, and compute:
      - mean structure (the motif template)
      - pairwise RMSD distribution
      - per-position coordinate variance (flexibility map)

    Returns a dict with the motif template and statistics.
    """
    if len(activated) < 3:
        return {
            "mean_structure": None,
            "mean_rmsd": float("inf"),
            "std_rmsd": 0.0,
            "rmsds": [],
            "n_fragments": len(activated),
        }

    frags = [a["fragment"] for a in activated[:top_k]]
    ref = frags[0]
    w = ref.shape[0]

    # Align all to reference
    aligned = [ref.copy()]
    rmsds = []
    for frag in frags[1:]:
        if frag.shape[0] != w:
            continue
        aln = kabsch_align(frag, ref)
        aligned.append(aln)
        rmsds.append(compute_rmsd(aln, ref))

    aligned_arr = np.array(aligned)  # (K, W, 3)
    mean_structure = aligned_arr.mean(axis=0)  # (W, 3)
    per_pos_std = aligned_arr.std(axis=0).mean(axis=1)  # (W,) avg std per position

    # Second pass: align to mean for better template
    aligned2 = []
    rmsds2 = []
    for frag in frags:
        if frag.shape[0] != w:
            continue
        aln = kabsch_align(frag, mean_structure)
        aligned2.append(aln)
        rmsds2.append(compute_rmsd(aln, mean_structure))

    aligned2_arr = np.array(aligned2)
    mean_structure2 = aligned2_arr.mean(axis=0)
    per_pos_std2 = aligned2_arr.std(axis=0).mean(axis=1)

    return {
        "mean_structure": mean_structure2,
        "per_pos_std": per_pos_std2,
        "mean_rmsd": float(np.mean(rmsds2)) if rmsds2 else float("inf"),
        "std_rmsd": float(np.std(rmsds2)) if rmsds2 else 0.0,
        "median_rmsd": float(np.median(rmsds2)) if rmsds2 else float("inf"),
        "rmsds": [float(r) for r in rmsds2],
        "n_fragments": len(aligned2),
        "aligned_fragments": aligned2_arr,
    }


# ====================== DECISION TREE RULES ================================

def train_motif_classifier(
    activated: list[dict],
    background: list[dict],
    max_depth: int = 4,
    cv_folds: int = 5,
) -> dict:
    """
    Train classifiers to separate activated vs. background positions
    based on local geometric features.

    Uses three models:
      1. Decision tree (shallow) — for human-readable rules
      2. Gradient-boosted ensemble — for precise probability calibration
         (this is the model used for plotting concordance)
      3. Random forest — for comparison AUC

    Also finds the optimal probability threshold on the GBM that
    maximises F1 score via cross-validated precision-recall analysis.
    """
    if len(activated) < 20 or len(background) < 20:
        return {
            "tree": None,
            "rules": "Insufficient data",
            "f1_cv": 0.0,
            "auc_cv": 0.0,
            "feature_importances": {},
            "optimal_threshold": 0.5,
        }

    X_pos = np.array([a["features"] for a in activated])
    X_neg = np.array([b["features"] for b in background])
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    # Handle NaN/inf
    valid = np.all(np.isfinite(X), axis=1)
    X = X[valid]
    y = y[valid]

    if len(X) < 40 or y.sum() < 10:
        return {
            "tree": None,
            "rules": "Insufficient valid data after filtering",
            "f1_cv": 0.0,
            "auc_cv": 0.0,
            "feature_importances": {},
            "optimal_threshold": 0.5,
        }

    # ── 1. Decision tree for interpretable rules ──
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(5, int(0.02 * len(X))),
        class_weight="balanced",
    )
    tree.fit(X, y)
    rules = export_text(tree, feature_names=LOCAL_GEOM_NAMES, decimals=4)

    # ── 2. Gradient-boosted ensemble for precise probability ──
    # Scale pos_weight to handle class imbalance
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    scale_pos = n_neg / max(1, n_pos)

    gbm = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=max(5, int(0.02 * len(X))),
        random_state=42,
    )
    gbm.fit(X, y)

    # ── 3. Cross-validated metrics + threshold calibration ──
    # Build per-sample protein group labels for leave-proteins-out CV
    acc_pos = [a["accession"] for a in activated]
    acc_neg = [b["accession"] for b in background]
    all_accessions = acc_pos + acc_neg
    # Filter to match the valid mask applied to X, y
    all_accessions_arr = np.array(all_accessions)
    all_accessions_arr = all_accessions_arr[valid]
    unique_proteins = np.unique(all_accessions_arr)
    n_unique = len(unique_proteins)

    n_cv = min(cv_folds, n_unique)  # can't have more folds than proteins
    optimal_threshold = 0.5
    f1_cv = 0.0
    auc_cv = 0.0
    gbm_auc_cv = 0.0
    rf_auc_cv = 0.0
    lpo_auc = 0.0  # leave-proteins-out AUC

    if n_cv >= 2:
        # Use GroupKFold (leave-proteins-out) if enough proteins,
        # otherwise fall back to StratifiedKFold
        if n_unique >= 5:
            cv = GroupKFold(n_splits=min(n_cv, n_unique))
            cv_split_args = (X, y, all_accessions_arr)
            use_groups = True
        else:
            cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
            cv_split_args = (X, y)
            use_groups = False

        # DT metrics
        try:
            f1_scores = cross_val_score(
                tree, X, y, cv=cv,
                groups=all_accessions_arr if use_groups else None,
                scoring="f1",
            )
            f1_cv = float(np.mean(f1_scores))
        except Exception:
            f1_cv = 0.0
        try:
            auc_scores = cross_val_score(
                tree, X, y, cv=cv,
                groups=all_accessions_arr if use_groups else None,
                scoring="roc_auc",
            )
            auc_cv = float(np.mean(auc_scores))
        except Exception:
            auc_cv = 0.0

        # GBM metrics + threshold calibration
        try:
            gbm_auc_scores = cross_val_score(
                gbm, X, y, cv=cv,
                groups=all_accessions_arr if use_groups else None,
                scoring="roc_auc",
            )
            gbm_auc_cv = float(np.mean(gbm_auc_scores))
        except Exception:
            gbm_auc_cv = 0.0

        # Find optimal threshold via CV precision-recall
        all_probs = np.zeros(len(y))
        for train_idx, val_idx in cv.split(*cv_split_args):
            gbm_cv = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.1,
                subsample=0.8,
                min_samples_leaf=max(5, int(0.02 * len(X))),
                random_state=42,
            )
            gbm_cv.fit(X[train_idx], y[train_idx])
            probs = gbm_cv.predict_proba(X[val_idx])
            all_probs[val_idx] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

        # Precision-recall curve to find best F1 threshold
        try:
            precision, recall, thresholds = precision_recall_curve(y, all_probs)
            f1_arr = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
            best_idx = np.argmax(f1_arr[:-1])  # last entry is always P=1, R=0
            optimal_threshold = float(thresholds[best_idx])
            # Clamp to reasonable range
            optimal_threshold = max(0.3, min(0.95, optimal_threshold))
        except Exception:
            optimal_threshold = 0.5

        # Leave-proteins-out AUC on the GBM held-out probs
        try:
            lpo_auc = float(roc_auc_score(y, all_probs))
        except Exception:
            lpo_auc = 0.0

        # RF for comparison
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        try:
            rf_auc = cross_val_score(
                rf, X, y, cv=cv,
                groups=all_accessions_arr if use_groups else None,
                scoring="roc_auc",
            )
            rf_auc_cv = float(np.mean(rf_auc))
        except Exception:
            rf_auc_cv = 0.0

    # Feature importances from GBM (more reliable than single DT)
    importances = {
        LOCAL_GEOM_NAMES[i]: float(gbm.feature_importances_[i])
        for i in range(len(LOCAL_GEOM_NAMES))
        if gbm.feature_importances_[i] > 0.005
    }
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    return {
        "tree": gbm,  # GBM is now the primary model for predict_proba
        "decision_tree": tree,  # keep DT for human-readable rules
        "rules": rules,
        "f1_cv": f1_cv,
        "auc_cv": auc_cv,
        "gbm_auc_cv": gbm_auc_cv,
        "rf_auc_cv": rf_auc_cv,
        "lpo_auc": lpo_auc,
        "n_unique_proteins": int(n_unique),
        "feature_importances": importances,
        "optimal_threshold": optimal_threshold,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


# ====================== CATEGORICAL ENRICHMENT ==============================

def compute_category_enrichment(
    activated: list[dict],
    background: list[dict],
) -> dict:
    """
    For each structural category, compute fold-enrichment:
        enrichment = P(category | activated) / P(category | background)

    Also computes Fisher's exact test p-value for significance.
    """
    n_cats = len(CATEGORY_NAMES)

    act_cats = np.array([a["category"] for a in activated])
    bg_cats = np.array([b["category"] for b in background])

    if len(act_cats) == 0 or len(bg_cats) == 0:
        return {"enrichments": {}, "counts": {}}

    enrichments = {}
    counts = {}

    for ci, cname in enumerate(CATEGORY_NAMES):
        n_act_in = int(np.sum(act_cats == ci))
        n_act_out = len(act_cats) - n_act_in
        n_bg_in = int(np.sum(bg_cats == ci))
        n_bg_out = len(bg_cats) - n_bg_in

        p_act = n_act_in / max(1, len(act_cats))
        p_bg = n_bg_in / max(1, len(bg_cats))

        fold_enrichment = p_act / max(1e-9, p_bg)

        # Fisher's exact test (2×2 contingency)
        try:
            table = [[n_act_in, n_act_out], [n_bg_in, n_bg_out]]
            _, pval = stats.fisher_exact(table, alternative="two-sided")
        except Exception:
            pval = 1.0

        enrichments[cname] = {
            "fold_enrichment": round(float(fold_enrichment), 3),
            "p_act": round(p_act, 4),
            "p_bg": round(p_bg, 4),
            "p_value": float(pval),
            "significant": pval < 0.01,
        }
        counts[cname] = {
            "activated": n_act_in,
            "background": n_bg_in,
        }

    return {"enrichments": enrichments, "counts": counts}


# ====================== PLOTTING ============================================

def plot_motif_template_3d(
    mean_structure: np.ndarray,
    per_pos_std: np.ndarray,
    node_idx: int,
    mean_rmsd: float,
    save_path: Path,
):
    """Plot the mean motif template coloured by per-position flexibility."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    coords = mean_structure
    # Colour by flexibility (std)
    norm = plt.Normalize(vmin=per_pos_std.min(), vmax=per_pos_std.max())
    cmap = plt.cm.coolwarm

    for i in range(len(coords) - 1):
        c = cmap(norm(per_pos_std[i]))
        ax.plot(
            coords[i:i + 2, 0], coords[i:i + 2, 1], coords[i:i + 2, 2],
            color=c, linewidth=2.5,
        )

    # Mark center
    mid = len(coords) // 2
    ax.scatter(*coords[mid], color="red", s=100, zorder=5, label="center")
    ax.scatter(*coords[0], color="#27ae60", s=60, zorder=5, label="N-end")
    ax.scatter(*coords[-1], color="#e74c3c", s=60, zorder=5, label="C-end")

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(
        f"Motif Template — Node {node_idx}\n"
        f"Mean RMSD = {mean_rmsd:.2f} Å  ({WINDOW_SIZE}-residue window)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_fragment_overlay(
    aligned_fragments: np.ndarray,
    node_idx: int,
    mean_rmsd: float,
    save_path: Path,
    max_show: int = 20,
):
    """Overlay top aligned fragments in 3D to visualise motif consistency."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    n_show = min(max_show, len(aligned_fragments))
    for i in range(n_show):
        c = COLOURS[i % len(COLOURS)]
        coords = aligned_fragments[i]
        ax.plot(
            coords[:, 0], coords[:, 1], coords[:, 2],
            color=c, linewidth=1.0, alpha=0.5,
        )

    # Mean as thick line
    mean_coords = aligned_fragments[:n_show].mean(axis=0)
    ax.plot(
        mean_coords[:, 0], mean_coords[:, 1], mean_coords[:, 2],
        color="black", linewidth=3.0, alpha=0.9, label="mean",
    )

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(
        f"Fragment Overlay — Node {node_idx}  "
        f"(n={n_show}, RMSD={mean_rmsd:.2f} Å)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_rmsd_histogram(
    rmsds: list[float],
    node_idx: int,
    save_path: Path,
):
    """Histogram of pairwise RMSDs for a node's top fragments."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rmsds, bins=30, color="#2980b9", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(rmsds), color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"mean = {np.mean(rmsds):.2f} Å")
    ax.axvline(np.median(rmsds), color="#27ae60", linestyle="--", linewidth=1.5,
               label=f"median = {np.median(rmsds):.2f} Å")
    ax.set_xlabel("RMSD to mean template (Å)")
    ax.set_ylabel("Count")
    ax.set_title(f"Fragment RMSD Distribution — Node {node_idx}")
    ax.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_enrichment_bars(
    enrichments: dict,
    node_idx: int,
    save_path: Path,
):
    """Horizontal bar chart of category fold-enrichments."""
    cats = list(enrichments.keys())
    folds = [enrichments[c]["fold_enrichment"] for c in cats]
    sigs = [enrichments[c]["significant"] for c in cats]
    colours = ["#e74c3c" if s else "#95a5a6" for s in sigs]

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = range(len(cats))
    ax.barh(y_pos, folds, color=colours, edgecolor="white")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cats, fontsize=9)
    ax.set_xlabel("Fold Enrichment (activated / background)")
    ax.set_title(f"Structural Category Enrichment — Node {node_idx}")
    ax.invert_yaxis()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_feature_importances(
    importances: dict,
    node_idx: int,
    save_path: Path,
):
    """Bar chart of decision tree feature importances."""
    if not importances:
        return
    names = list(importances.keys())
    vals = list(importances.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(names)), vals, color="#2980b9", edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Decision Tree Feature Importances — Node {node_idx}")
    ax.invert_yaxis()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_node_ranking(results: list[dict], save_path: Path):
    """Bar chart ranking nodes by mean fragment RMSD (lower = more consistent motif)."""
    nodes = [r["sae_node"] for r in results]
    rmsds = [r["mean_rmsd"] for r in results]
    aucs = [r.get("auc_cv", 0.0) for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    x = np.arange(len(nodes))

    # RMSD (lower = better)
    ax1.bar(x, rmsds, color="#2980b9", alpha=0.85)
    ax1.set_ylabel("Mean Fragment RMSD (Å)")
    ax1.set_title("SAE Nodes Ranked by Motif Consistency (lower RMSD = clearer motif)")
    ax1.axhline(3.0, color="#e74c3c", linestyle="--", alpha=0.5, label="3 Å threshold")
    ax1.legend()

    # AUC (higher = better)
    ax2.bar(x, aucs, color="#27ae60", alpha=0.85)
    ax2.set_ylabel("Decision Tree AUC (CV)")
    ax2.set_xlabel("SAE Node")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"N{n}" for n in nodes], rotation=70, fontsize=7)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_full_backbone_with_motif(
    ca: np.ndarray,
    activated_positions: list[dict],
    node_idx: int,
    accession: str,
    half_w: int,
    save_path: Path,
    tree: DecisionTreeClassifier | None = None,
    profiles: dict | None = None,
    sequence: str | None = None,
    geom_threshold: float = 0.5,
):
    """
    Plot the full protein Cα backbone in 3D with two colour layers:

      • **Red / orange** — SAE-activation-defined motif windows
        (positions where the node fires above threshold).
      • **Blue** (optional) — Geometry-predicted motif regions
        (positions where the decision tree, using only local geometric
        features, predicts the node *should* fire with P > 0.5).

    Residues where both signals agree are drawn thicker so they
    stand out.  The full backbone is in light grey underneath.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    n = len(ca)

    # ── Geometry-predicted probability at every residue ──
    geom_prob = np.zeros(n)
    if tree is not None and profiles is not None:
        for pos in range(half_w, n - half_w):
            feat_vec = extract_local_feature_vector(profiles, ca, pos, half_w, sequence=sequence)
            if feat_vec is not None and np.all(np.isfinite(feat_vec)):
                prob = tree.predict_proba(feat_vec.reshape(1, -1))[0]
                geom_prob[pos] = prob[1] if len(prob) > 1 else prob[0]

    geom_active = geom_prob > geom_threshold  # geometry predicts "active"

    # SAE-activation set for quick lookup
    sae_positions = set()
    for apos in activated_positions:
        pos = apos["position"]
        for r in range(max(0, pos - half_w), min(n, pos + half_w + 1)):
            sae_positions.add(r)

    # ── Draw full backbone in light grey ──
    ax.plot(
        ca[:, 0], ca[:, 1], ca[:, 2],
        color="#cccccc", linewidth=1.2, alpha=0.6, label="backbone",
    )

    # ── Blue layer: geometry-predicted regions (drawn first, under red) ──
    if tree is not None and profiles is not None:
        # Find contiguous runs of geom_active
        geom_drawn = False
        i = 0
        while i < n:
            if geom_active[i]:
                j = i
                while j < n and geom_active[j]:
                    j += 1
                seg = ca[i:j + 1] if j < n else ca[i:j]
                lbl = "geom-predicted" if not geom_drawn else None
                geom_drawn = True
                ax.plot(
                    seg[:, 0], seg[:, 1], seg[:, 2],
                    color="#3498db", linewidth=3.0, alpha=0.55, label=lbl,
                )
                i = j
            else:
                i += 1

    # ── Red / orange layer: SAE-activation motif windows ──
    highlight_colours = ["#e74c3c", "#f39c12", "#e67e22", "#d35400", "#c0392b"]
    for idx, apos in enumerate(activated_positions):
        pos = apos["position"]
        act_val = apos["activation"]
        s = max(0, pos - half_w)
        e = min(n, pos + half_w + 1)
        seg = ca[s:e]

        colour = highlight_colours[idx % len(highlight_colours)]
        lbl = f"SAE motif @{pos} (act={act_val:.2f})" if idx < 5 else None
        ax.plot(
            seg[:, 0], seg[:, 1], seg[:, 2],
            color=colour, linewidth=3.5, alpha=0.95, label=lbl,
        )
        # Mark centre residue
        if pos < n:
            ax.scatter(
                *ca[pos], color=colour, s=120, zorder=5,
                edgecolors="black", linewidths=0.8,
            )

    # ── Concordant residues: both SAE and geometry agree → thick marker ──
    if tree is not None and profiles is not None:
        concordant_drawn = False
        for pos in range(n):
            if geom_active[pos] and pos in sae_positions:
                lbl = "both agree" if not concordant_drawn else None
                concordant_drawn = True
                ax.scatter(
                    *ca[pos], color="#27ae60", s=30, zorder=6,
                    alpha=0.7, marker="o", label=lbl,
                )

    # ── Mark termini ──
    ax.scatter(*ca[0], color="#27ae60", s=50, zorder=4,
               edgecolors="black", linewidths=0.5, label="N-term")
    ax.scatter(*ca[-1], color="#8e44ad", s=50, zorder=4,
               edgecolors="black", linewidths=0.5, label="C-term")

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")

    n_sites = len(activated_positions)
    top_act = activated_positions[0]["activation"] if activated_positions else 0
    geom_str = ""
    if tree is not None:
        n_geom = int(geom_active.sum())
        n_both = sum(1 for p in range(n) if geom_active[p] and p in sae_positions)
        geom_str = f", {n_geom} geom-predicted, {n_both} concordant"
    ax.set_title(
        f"Full Backbone — {accession} — Node {node_idx}\n"
        f"{n} residues, {n_sites} SAE motif site(s){geom_str}\n"
        f"Red/orange = SAE activation, Blue = geometry prediction",
        fontsize=9,
    )
    ax.legend(fontsize=6, loc="upper left")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_top_backbones_for_node(
    result: dict,
    protein_lookup: dict,
    half_w: int,
    plot_dir: Path,
    max_proteins: int = 5,
):
    """
    For a given SAE node, plot the full Cα backbone of the top-activating
    proteins with:
      • Red/orange highlights for SAE-activation-defined motif windows
      • Blue highlights for geometry-predicted motif regions (if a
        decision tree is available)
      • Green dots where both signals agree
    """
    node_idx = result["sae_node"]
    tree = result.get("decision_tree")  # may be None
    geom_thr = result.get("optimal_threshold", 0.5)
    entries = result.get("top_activated_entries", [])
    if not entries:
        return

    # Group entries by accession
    by_acc: dict[str, list[dict]] = {}
    for e in entries:
        by_acc.setdefault(e["accession"], []).append(e)

    # Rank proteins by their peak activation for this node
    acc_list = sorted(
        by_acc.keys(),
        key=lambda a: max(ee["activation"] for ee in by_acc[a]),
        reverse=True,
    )[:max_proteins]

    node_dir = plot_dir / f"node_{node_idx}"
    for rank, acc in enumerate(acc_list):
        pdata = protein_lookup.get(acc)
        if pdata is None:
            continue
        plot_full_backbone_with_motif(
            ca=pdata["ca"],
            activated_positions=by_acc[acc],
            node_idx=node_idx,
            accession=acc,
            half_w=half_w,
            save_path=node_dir / f"backbone_{acc}.png",
            tree=tree,
            profiles=pdata.get("profiles"),
            sequence=pdata.get("sequence"),
            geom_threshold=geom_thr,
        )


def plot_residue_activation_profile(
    act_profile: np.ndarray,
    sequence: str,
    activated_positions: list[dict],
    node_idx: int,
    accession: str,
    threshold: float,
    half_w: int,
    save_path: Path,
):
    """
    InterPLM Figure 1C–style per-residue activation profile.

    Draws a lollipop / stem plot:
      • X-axis = residue position along the sequence
      • Y-axis = SAE node activation at that position
      • Thin grey stems from 0 to activation value
      • Amino-acid letters rendered at the top of each stem,
        coloured by residue type
      • Motif window regions shaded with a coloured span
      • Horizontal dashed line at the activation threshold
    """
    n = len(act_profile)
    seq = sequence[:n] if len(sequence) >= n else sequence + "?" * (n - len(sequence))

    fig, ax = plt.subplots(figsize=(max(12, n * 0.04), 4))

    # Colour map for amino acid types (broad groupings)
    _AA_COLOURS = {
        # Hydrophobic
        "A": "#2c3e50", "V": "#2c3e50", "I": "#2c3e50", "L": "#2c3e50",
        "M": "#2c3e50", "F": "#8e44ad", "W": "#8e44ad", "P": "#d35400",
        # Polar uncharged
        "S": "#27ae60", "T": "#27ae60", "N": "#27ae60", "Q": "#27ae60",
        "Y": "#16a085", "C": "#f39c12",
        # Positive
        "K": "#2980b9", "R": "#2980b9", "H": "#2980b9",
        # Negative
        "D": "#e74c3c", "E": "#e74c3c",
        # Special
        "G": "#95a5a6",
    }

    # Draw thin grey stems
    for j in range(n):
        ax.plot([j, j], [0, act_profile[j]], color="#cccccc", linewidth=0.5)

    # Shade motif windows
    highlight_colours = ["#e74c3c", "#f39c12", "#e67e22", "#d35400", "#c0392b"]
    for idx, apos in enumerate(activated_positions):
        pos = apos["position"]
        s = max(0, pos - half_w)
        e = min(n, pos + half_w + 1)
        colour = highlight_colours[idx % len(highlight_colours)]
        ax.axvspan(s - 0.5, e - 0.5, color=colour, alpha=0.12,
                   label=f"motif @{pos}" if idx < 5 else None)

    # Re-draw stems in motif regions with full colour
    motif_positions = set()
    for apos in activated_positions:
        pos = apos["position"]
        for r in range(max(0, pos - half_w), min(n, pos + half_w + 1)):
            motif_positions.add(r)

    # Render amino-acid letters at the top of each stem
    y_max = float(np.max(act_profile)) if np.max(act_profile) > 0 else 1.0
    # Only label residues if protein is short enough, otherwise label peaks
    if n <= 200:
        for j in range(n):
            aa = seq[j]
            c = _AA_COLOURS.get(aa, "#95a5a6")
            ax.text(
                j, act_profile[j], aa,
                ha="center", va="bottom", fontsize=max(3, min(7, 800 // n)),
                color=c, fontweight="bold" if j in motif_positions else "normal",
            )
    else:
        # For long proteins, only label top-activating positions
        top_idx = np.argsort(act_profile)[-min(40, n):]
        for j in top_idx:
            if act_profile[j] > threshold * 0.5:
                aa = seq[j]
                c = _AA_COLOURS.get(aa, "#95a5a6")
                ax.text(
                    j, act_profile[j], aa,
                    ha="center", va="bottom", fontsize=6,
                    color=c, fontweight="bold",
                )

    # Threshold line
    if threshold > 0:
        ax.axhline(threshold, color="#e74c3c", linestyle="--", linewidth=1.0,
                   alpha=0.6, label=f"threshold = {threshold:.3f}")

    ax.set_xlim(-1, n)
    ax.set_ylim(0, y_max * 1.15)
    ax.set_xlabel("Residue Position")
    ax.set_ylabel(f"SAE Node {node_idx} Activation")
    ax.set_title(
        f"Per-Residue Activation Profile — {accession} — Node {node_idx}\n"
        f"{n} residues, {len(activated_positions)} motif site(s)",
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_activation_profiles_for_node(
    result: dict,
    protein_lookup: dict,
    half_w: int,
    plot_dir: Path,
    max_proteins: int = 5,
):
    """
    For a given SAE node, generate InterPLM-style per-residue activation
    profile plots for the top-activating proteins.
    """
    node_idx = result["sae_node"]
    threshold = result.get("threshold", 0.0)
    entries = result.get("top_activated_entries", [])
    if not entries:
        return

    # Group entries by accession
    by_acc: dict[str, list[dict]] = {}
    for e in entries:
        by_acc.setdefault(e["accession"], []).append(e)

    acc_list = sorted(
        by_acc.keys(),
        key=lambda a: max(ee["activation"] for ee in by_acc[a]),
        reverse=True,
    )[:max_proteins]

    node_dir = plot_dir / f"node_{node_idx}"
    for acc in acc_list:
        pdata = protein_lookup.get(acc)
        if pdata is None:
            continue
        act_profile = pdata["act_matrix"][:, node_idx]
        plot_residue_activation_profile(
            act_profile=act_profile,
            sequence=pdata.get("sequence", ""),
            activated_positions=by_acc[acc],
            node_idx=node_idx,
            accession=acc,
            threshold=threshold,
            half_w=half_w,
            save_path=node_dir / f"activation_profile_{acc}.png",
        )


def plot_geometry_overlay(
    act_profile: np.ndarray,
    ca: np.ndarray,
    profiles: dict,
    tree: DecisionTreeClassifier,
    feature_importances: dict,
    activated_positions: list[dict],
    node_idx: int,
    accession: str,
    threshold: float,
    half_w: int,
    save_path: Path,
    sequence: str | None = None,
    geom_threshold: float = 0.5,
):
    """
    Dual-panel verification plot:

      Top panel    – SAE node activation along the sequence (what the node
                     actually fires on).
      Bottom panel – Decision-tree predicted probability of activation based
                     *only* on local geometry, plus traces of the top 1-2
                     geometric features used by the tree.

    Regions where both panels agree are shaded green (concordant);
    where they disagree (geometry predicts activation but node is silent,
    or vice versa) are shaded amber.
    """
    n = min(len(act_profile), len(ca))
    if n < 2 * half_w + 1:
        return

    # ── Compute per-residue geometry-predicted probability ──
    geom_prob = np.zeros(n)
    geom_features_at_pos = {}  # pos -> feature vector
    for pos in range(half_w, n - half_w):
        feat_vec = extract_local_feature_vector(profiles, ca, pos, half_w, sequence=sequence)
        if feat_vec is not None and np.all(np.isfinite(feat_vec)):
            geom_features_at_pos[pos] = feat_vec
            prob = tree.predict_proba(feat_vec.reshape(1, -1))[0]
            # prob is [P(bg), P(activated)] — take the activated class
            geom_prob[pos] = prob[1] if len(prob) > 1 else prob[0]

    # ── Identify top 2 geometric features by importance ──
    top_feat_names = list(feature_importances.keys())[:2]
    top_feat_indices = [
        LOCAL_GEOM_NAMES.index(fn) for fn in top_feat_names
        if fn in LOCAL_GEOM_NAMES
    ]

    # Build full-length traces for the top features
    feat_traces = {}
    for fi in top_feat_indices:
        trace = np.full(n, np.nan)
        for pos, fv in geom_features_at_pos.items():
            trace[pos] = fv[fi]
        feat_traces[LOCAL_GEOM_NAMES[fi]] = trace

    # ── Normalise activation for concordance comparison ──
    act_max = float(np.max(act_profile[:n])) if np.max(act_profile[:n]) > 0 else 1.0
    act_norm = act_profile[:n] / act_max  # 0..1

    # ── Concordance / discordance regions ──
    # "geometry predicts active"  = geom_prob > calibrated threshold
    # "node actually active"     = act_norm > threshold / act_max
    act_thr_norm = threshold / act_max if act_max > 0 else 0.5
    geom_pred = geom_prob > geom_threshold
    node_active = act_norm > act_thr_norm

    # ── Plot ──
    fig_w = max(12, n * 0.04)
    fig, (ax_act, ax_geom) = plt.subplots(
        2, 1, figsize=(fig_w, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.15},
    )

    xs = np.arange(n)

    # -- Top panel: SAE activation --
    ax_act.fill_between(xs, 0, act_profile[:n], color="#2980b9", alpha=0.3)
    ax_act.plot(xs, act_profile[:n], color="#2980b9", linewidth=0.8)
    if threshold > 0:
        ax_act.axhline(threshold, color="#e74c3c", linestyle="--",
                       linewidth=0.8, alpha=0.6, label=f"threshold = {threshold:.3f}")

    # Shade motif windows (activation-defined)
    highlight_colours = ["#e74c3c", "#f39c12", "#e67e22", "#d35400", "#c0392b"]
    for idx, apos in enumerate(activated_positions):
        pos = apos["position"]
        s = max(0, pos - half_w)
        e = min(n, pos + half_w + 1)
        colour = highlight_colours[idx % len(highlight_colours)]
        ax_act.axvspan(s - 0.5, e - 0.5, color=colour, alpha=0.10,
                       label=f"SAE motif @{pos}" if idx < 3 else None)

    ax_act.set_ylabel(f"SAE Activation")
    ax_act.set_title(
        f"Geometry vs Activation Verification — {accession} — Node {node_idx}",
        fontsize=10,
    )
    ax_act.legend(fontsize=6, loc="upper right")
    ax_act.spines["top"].set_visible(False)
    ax_act.spines["right"].set_visible(False)

    # -- Bottom panel: geometry-predicted probability + feature traces --
    ax_geom.fill_between(xs, 0, geom_prob, color="#27ae60", alpha=0.25)
    ax_geom.plot(xs, geom_prob, color="#27ae60", linewidth=1.0,
                 label="P(active | geometry)")
    ax_geom.axhline(0.5, color="#27ae60", linestyle=":", linewidth=0.7, alpha=0.5)

    # Overlay top geometric feature traces (on a twin y-axis)
    if feat_traces:
        ax_feat = ax_geom.twinx()
        feat_colours = ["#8e44ad", "#d35400"]
        for i, (fname, trace) in enumerate(feat_traces.items()):
            c = feat_colours[i % len(feat_colours)]
            ax_feat.plot(xs, trace, color=c, linewidth=0.8, alpha=0.7,
                         label=fname, linestyle="-")
        ax_feat.set_ylabel("Geometric Feature", fontsize=8)
        ax_feat.legend(fontsize=6, loc="upper left")
        ax_feat.spines["top"].set_visible(False)

    # Shade concordant / discordant regions
    for j in range(n):
        if geom_pred[j] and node_active[j]:
            # Concordant: both agree "active"
            ax_geom.axvspan(j - 0.5, j + 0.5, color="#27ae60", alpha=0.08)
        elif geom_pred[j] and not node_active[j]:
            # Geometry says active, node silent
            ax_geom.axvspan(j - 0.5, j + 0.5, color="#f39c12", alpha=0.12)
        elif not geom_pred[j] and node_active[j]:
            # Node active, geometry doesn't predict it
            ax_geom.axvspan(j - 0.5, j + 0.5, color="#e74c3c", alpha=0.10)

    # Add a legend for concordance shading
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#27ae60", alpha=0.3, label="concordant (both active)"),
        Patch(facecolor="#f39c12", alpha=0.35, label="geom predicts, node silent"),
        Patch(facecolor="#e74c3c", alpha=0.3, label="node active, geom doesn't predict"),
    ]
    leg2 = ax_geom.legend(
        handles=legend_patches, fontsize=6, loc="lower right",
        title="concordance", title_fontsize=6,
    )
    ax_geom.add_artist(leg2)

    ax_geom.set_ylabel("P(active | geometry)")
    ax_geom.set_xlabel("Residue Position")
    ax_geom.set_xlim(-1, n)
    ax_geom.set_ylim(0, 1.05)
    ax_geom.spines["top"].set_visible(False)
    ax_geom.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_geometry_overlays_for_node(
    result: dict,
    protein_lookup: dict,
    half_w: int,
    plot_dir: Path,
    max_proteins: int = 5,
):
    """
    For a given SAE node, generate dual-panel geometry-vs-activation
    verification plots for the top-activating proteins.
    """
    node_idx = result["sae_node"]
    threshold = result.get("threshold", 0.0)
    tree = result.get("decision_tree")
    feat_imp = result.get("feature_importances", {})
    geom_thr = result.get("optimal_threshold", 0.5)
    entries = result.get("top_activated_entries", [])

    if tree is None or not entries:
        return

    by_acc: dict[str, list[dict]] = {}
    for e in entries:
        by_acc.setdefault(e["accession"], []).append(e)

    acc_list = sorted(
        by_acc.keys(),
        key=lambda a: max(ee["activation"] for ee in by_acc[a]),
        reverse=True,
    )[:max_proteins]

    node_dir = plot_dir / f"node_{node_idx}"
    for acc in acc_list:
        pdata = protein_lookup.get(acc)
        if pdata is None:
            continue
        act_profile = pdata["act_matrix"][:, node_idx]
        plot_geometry_overlay(
            act_profile=act_profile,
            ca=pdata["ca"],
            profiles=pdata["profiles"],
            tree=tree,
            feature_importances=feat_imp,
            activated_positions=by_acc[acc],
            node_idx=node_idx,
            accession=acc,
            threshold=threshold,
            half_w=half_w,
            save_path=node_dir / f"geometry_overlay_{acc}.png",
            sequence=pdata.get("sequence"),
            geom_threshold=geom_thr,
        )


def save_motif_template_pdb(
    mean_structure: np.ndarray,
    node_idx: int,
    save_path: Path,
):
    """
    Save the mean motif template as a minimal PDB file so it can be
    opened in PyMOL / ChimeraX.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"REMARK  Motif template for SAE node {node_idx}"]
    lines.append(f"REMARK  {WINDOW_SIZE}-residue fragment, Kabsch-aligned mean")
    for i, (x, y, z) in enumerate(mean_structure):
        lines.append(
            f"ATOM  {i + 1:>5d}  CA  ALA A{i + 1:>4d}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    save_path.write_text("\n".join(lines))


# ===================== CONCORDANCE METRICS ==================================

def compute_concordance_metrics(
    protein_data: list[dict],
    node_idx: int,
    tree,
    threshold: float,
    geom_threshold: float,
    half_w: int,
) -> dict:
    """
    Quantitative concordance between SAE activation and geometry-predicted
    probability at the residue level.

    **Continuous metrics** (no binarisation):
      • spearman_r  – rank correlation between raw SAE activation and P(active)
      • residue_auroc – AUROC treating SAE-active (binary) as label, P(active) as score
      • avg_precision – average precision (area under PR curve), same setup
      • cosine_sim   – cosine similarity between the two vectors

    **Binary metrics** (kept for reference at the calibrated threshold):
      • precision, recall, f1, iou

    Only proteins where the node fires at least once are included.
    """
    empty = {
        # Continuous
        "spearman_r": 0.0, "spearman_p": 1.0,
        "residue_auroc": 0.0, "avg_precision": 0.0, "cosine_sim": 0.0,
        # Binary
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0,
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "n_residues": 0, "n_proteins": 0,
    }
    if tree is None:
        return empty

    all_sae_act = []   # raw continuous SAE activation per residue
    all_geom_prob = [] # predicted P(active) per residue
    tp = fp = fn = tn = 0
    n_proteins_used = 0

    for pdata in protein_data:
        col = pdata["act_matrix"][:, node_idx]
        ca = pdata["ca"]
        profiles = pdata["profiles"]
        seq = pdata.get("sequence", None)
        n = pdata["n_residues"]

        if not np.any(col >= threshold):
            continue  # skip proteins where node never fires

        n_proteins_used += 1

        for pos in range(half_w, n - half_w):
            sae_val = float(col[pos])
            feat_vec = extract_local_feature_vector(
                profiles, ca, pos, half_w, sequence=seq,
            )

            if feat_vec is not None and np.all(np.isfinite(feat_vec)):
                prob = tree.predict_proba(feat_vec.reshape(1, -1))[0]
                geom_prob = float(prob[1] if len(prob) > 1 else prob[0])
            else:
                geom_prob = 0.0

            all_sae_act.append(sae_val)
            all_geom_prob.append(geom_prob)

            # Binary (still useful as secondary metric)
            sae_active = sae_val >= threshold
            geom_active = geom_prob >= geom_threshold
            if sae_active and geom_active:
                tp += 1
            elif not sae_active and geom_active:
                fp += 1
            elif sae_active and not geom_active:
                fn += 1
            else:
                tn += 1

    total = tp + fp + fn + tn
    if total < 10 or n_proteins_used == 0:
        return empty

    sae_arr = np.array(all_sae_act)
    geom_arr = np.array(all_geom_prob)
    sae_binary = (sae_arr >= threshold).astype(int)

    # ── Continuous metrics ──
    # Spearman correlation (raw activation vs predicted probability)
    try:
        sp = stats.spearmanr(sae_arr, geom_arr)
        spearman_r, spearman_p = float(sp.statistic), float(sp.pvalue)
    except Exception:
        spearman_r, spearman_p = 0.0, 1.0

    # Residue-level AUROC  (can geometry score rank-separate active residues?)
    try:
        res_auroc = float(roc_auc_score(sae_binary, geom_arr))
    except Exception:
        res_auroc = 0.0

    # Average Precision (area under the precision-recall curve)
    try:
        ap = float(average_precision_score(sae_binary, geom_arr))
    except Exception:
        ap = 0.0

    # Cosine similarity
    dot = np.dot(sae_arr, geom_arr)
    norms = np.linalg.norm(sae_arr) * np.linalg.norm(geom_arr)
    cos_sim = float(dot / max(norms, 1e-12))

    # ── Binary metrics (at calibrated threshold) ──
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    iou = tp / max(1, tp + fp + fn)

    return {
        # Continuous
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 6),
        "residue_auroc": round(res_auroc, 4),
        "avg_precision": round(ap, 4),
        "cosine_sim": round(cos_sim, 4),
        # Binary
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "iou": round(float(iou), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_residues": int(total),
        "n_proteins": int(n_proteins_used),
    }


# ========================= MAIN PIPELINE ===================================

def main():
    parser = argparse.ArgumentParser(
        description="Residue-level motif discovery: local geometry ↔ SAE node activation."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--organism", type=int, default=None,
                     help="UniProt taxonomy ID (9606=human).")
    src.add_argument("--mixed-organisms", action="store_true",
                     help="Pull from ~20 diverse organisms.")
    src.add_argument("--fasta", type=Path, default=None)
    src.add_argument("--accession-list", type=Path, default=None)

    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument("--per-organism-cap", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "residue_motifs")
    parser.add_argument("--sae-dir", type=Path, default=SAE_DIR)
    parser.add_argument("--esm-model", type=str, default=ESM_MODEL_NAME)
    parser.add_argument("--esm-layer", type=int, default=ESM_LAYER)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--half-window", type=int, default=HALF_W,
                        help="Half-window size for fragment extraction.")
    parser.add_argument("--top-nodes", type=int, default=TOP_N_NODES,
                        help="Number of top nodes to analyse in detail.")
    parser.add_argument("--frag-top-k", type=int, default=FRAG_TOP_K,
                        help="Number of top fragments per node for superposition.")
    parser.add_argument("--act-quantile", type=float, default=ACT_QUANTILE,
                        help="Quantile threshold for 'activated' positions.")
    parser.add_argument("--tree-depth", type=int, default=4,
                        help="Max depth for decision tree classifier.")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="Max number of active nodes to analyse (for quick testing). "
                             "Nodes are sorted by total activity so the most active are analysed first.")
    args = parser.parse_args()

    if (args.organism is None and args.fasta is None
            and args.accession_list is None and not args.mixed_organisms):
        args.organism = 9606

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pdb_cache = out / "pdb_cache"
    pdb_cache.mkdir(exist_ok=True)
    # Also share pdb_cache with alphafold_analysis if it exists
    shared_pdb_cache = Path(__file__).resolve().parent / "alphafold_analysis" / "pdb_cache"
    if shared_pdb_cache.exists():
        pdb_cache = shared_pdb_cache
        print(f"[i] Using shared PDB cache: {pdb_cache}")

    device = get_device()
    print("=" * 72)
    print("Residue-Level Motif Discovery: Local Geometry ↔ SAE Activation")
    print("=" * 72)
    print(f"[✓] Device:       {device}")
    print(f"[i] Output dir:   {out}")
    print(f"[i] Window size:  {2 * args.half_window + 1} residues "
          f"(half_w={args.half_window})")
    print(f"[i] Act quantile: {args.act_quantile}")
    print()

    # ── Step 1: Get accession list ────────────────────────────────────────
    fasta_seqs: dict[str, str] | None = None
    if args.mixed_organisms:
        accessions = fetch_mixed_organism_accessions(
            max_proteins=args.max_proteins,
            per_organism_cap=args.per_organism_cap,
            seed=args.seed,
        )
    elif args.fasta:
        accessions, fasta_seqs = load_accessions_from_fasta(args.fasta)
        print(f"[1/8] Loaded {len(accessions)} proteins from FASTA: {args.fasta}")
    elif args.accession_list:
        accessions = load_accessions_from_file(args.accession_list)
        print(f"[1/8] Loaded {len(accessions)} accessions from: {args.accession_list}")
    else:
        accessions = fetch_swissprot_accessions(
            organism_taxid=args.organism,
            max_proteins=args.max_proteins,
        )
    if args.max_proteins:
        accessions = accessions[:args.max_proteins]
    (out / "accessions.txt").write_text("\n".join(accessions))
    print(f"  {len(accessions)} proteins to process.\n")

    # ── Step 2: Load SAE + ESM ────────────────────────────────────────────
    print("[2/8] Loading SAE …")
    sae = load_sae(args.sae_dir, device=device)
    sae.eval()
    print(f"  SAE: {sae.__class__.__name__}  dict_size={sae.dict_size}  "
          f"activation_dim={sae.activation_dim}")
    print("[2/8] Loading ESM embedder …")
    embedder = ESM(model_name=args.esm_model, device=device)
    print(f"  ESM: {args.esm_model}  layer {args.esm_layer}\n")

    n_nodes = sae.dict_size

    # ── Step 3: Process proteins (per-residue activations + geometry) ─────
    protein_cache = out / "protein_data.npz"
    protein_data: list[dict] = []

    if args.resume and protein_cache.exists():
        print("[3/8] Loading cached per-protein residue data …")
        npz = np.load(protein_cache, allow_pickle=True)
        protein_data = npz["protein_data"].tolist()
        print(f"  {len(protein_data)} proteins loaded from cache.\n")
    else:
        print("[3/8] Computing per-residue activations & geometry …")
        print("  This fetches AlphaFold structures and runs ESM+SAE per protein.\n")

        session = requests.Session()
        n_skip = 0
        t0 = _time.time()

        for idx, acc in enumerate(accessions):
            if idx % 10 == 0 or idx == len(accessions) - 1:
                elapsed = _time.time() - t0
                print(
                    f"  [{idx + 1:>6d}/{len(accessions)}]  "
                    f"ok={len(protein_data)}  skip={n_skip}  "
                    f"current={acc}  [{elapsed:.0f}s]"
                )

            # Get sequence
            if fasta_seqs and acc in fasta_seqs:
                seq = fasta_seqs[acc]
            else:
                seq = fetch_sequence(acc, session)
            if not seq or len(seq) < WINDOW_SIZE + 4 or len(seq) > MAX_SEQ_LEN:
                n_skip += 1
                continue

            # Get structure
            pdb_text = fetch_alphafold_pdb(acc, pdb_cache, session)
            if pdb_text is None:
                n_skip += 1
                continue

            # Process
            pdata = process_one_protein(
                acc, pdb_text, sae, embedder, device, args.esm_layer,
            )
            if pdata is None:
                n_skip += 1
                continue

            protein_data.append(pdata)

            # Periodic checkpoint
            if len(protein_data) % 200 == 0 and len(protein_data) > 0:
                print(f"    … checkpoint at {len(protein_data)} proteins")
                np.savez(
                    protein_cache,
                    protein_data=np.array(protein_data, dtype=object),
                )

        if not protein_data:
            print("\n✘ No proteins processed. Check network / accessions.")
            sys.exit(1)

        # Final save
        np.savez(
            protein_cache,
            protein_data=np.array(protein_data, dtype=object),
        )
        print(f"\n  [✓] {len(protein_data)} proteins processed, "
              f"{n_skip} skipped.\n")

    total_residues = sum(p["n_residues"] for p in protein_data)
    print(f"  Dataset: {len(protein_data)} proteins, "
          f"{total_residues:,} total residues, "
          f"{n_nodes} SAE nodes\n")

    # ── Step 4: Pre-screen nodes by activity ──────────────────────────────
    print("[4/8] Pre-screening SAE nodes by residue-level activity …")
    node_activity = np.zeros(n_nodes)
    for pdata in protein_data:
        node_activity += (pdata["act_matrix"] > 0).sum(axis=0)

    active_nodes = np.where(node_activity >= MIN_ACTIVATED_POSITIONS)[0]
    # Sort by total activity descending so most active nodes are analysed first
    active_nodes = active_nodes[np.argsort(node_activity[active_nodes])[::-1]]
    print(f"  {len(active_nodes)}/{n_nodes} nodes have ≥{MIN_ACTIVATED_POSITIONS} "
          f"activated residue positions.")

    if args.max_nodes is not None and len(active_nodes) > args.max_nodes:
        active_nodes = active_nodes[:args.max_nodes]
        print(f"  → Limited to {args.max_nodes} most-active nodes (--max-nodes).")
    print()

    # ── Step 5: Fragment collection + superposition + classification ──────
    print(f"[5/8] Analysing {len(active_nodes)} active nodes …")
    print(f"  (fragment extraction → Kabsch superposition → decision tree → enrichment)\n")

    all_results: list[dict] = []
    t0 = _time.time()

    for rank, ni in enumerate(active_nodes):
        # if rank % 50 == 0:
        elapsed = _time.time() - t0
        print(f"  [{rank + 1:>5d}/{len(active_nodes)}]  "
                f"node={ni}  [{elapsed:.0f}s]")

        # 5a. Collect fragments
        frag_data = collect_node_fragments(
            protein_data, int(ni),
            half_w=args.half_window,
            act_quantile=args.act_quantile,
            max_fragments=args.frag_top_k,
        )

        activated = frag_data["activated"]
        background = frag_data["background"]

        if len(activated) < 10:
            continue

        # 5b. Fragment superposition
        superpose = superpose_fragments(activated, top_k=args.frag_top_k)

        # 5c. Decision tree classification
        classifier = train_motif_classifier(
            activated, background, max_depth=args.tree_depth,
        )

        # 5d. Categorical enrichment
        enrichment = compute_category_enrichment(activated, background)

        # 5e. Concordance metrics (geometry vs SAE activation at residue level)
        concordance = compute_concordance_metrics(
            protein_data, int(ni),
            tree=classifier["tree"],
            threshold=frag_data["threshold"],
            geom_threshold=classifier.get("optimal_threshold", 0.5),
            half_w=args.half_window,
        )

        # Collect results
        result = {
            "sae_node": int(ni),
            "n_activated": len(activated),
            "n_background": len(background),
            "threshold": frag_data["threshold"],
            "n_total_active_residues": frag_data["n_total_active"],
            # Superposition
            "mean_rmsd": superpose["mean_rmsd"],
            "median_rmsd": superpose.get("median_rmsd", superpose["mean_rmsd"]),
            "std_rmsd": superpose["std_rmsd"],
            "n_fragments": superpose["n_fragments"],
            "mean_structure": superpose["mean_structure"],
            "per_pos_std": superpose.get("per_pos_std"),
            "aligned_fragments": superpose.get("aligned_fragments"),
            "rmsds": superpose["rmsds"],
            # Classifier
            "rules": classifier["rules"],
            "f1_cv": classifier["f1_cv"],
            "auc_cv": classifier["auc_cv"],
            "gbm_auc_cv": classifier.get("gbm_auc_cv", 0.0),
            "rf_auc_cv": classifier.get("rf_auc_cv", 0.0),
            "lpo_auc": classifier.get("lpo_auc", 0.0),
            "n_unique_proteins": classifier.get("n_unique_proteins", 0),
            "feature_importances": classifier["feature_importances"],
            "decision_tree": classifier["tree"],  # GBM model for predict_proba
            "optimal_threshold": classifier.get("optimal_threshold", 0.5),
            # Concordance (residue-level)
            "concordance": concordance,
            # Enrichment
            "enrichments": enrichment["enrichments"],
            "category_counts": enrichment["counts"],
            # Top activated positions (for full-backbone plots)
            "top_activated_entries": [
                {"accession": a["accession"], "position": a["position"],
                 "activation": a["activation"]}
                for a in activated[:20]
            ],
        }
        all_results.append(result)

    # Sort by mean RMSD (lower = more consistent motif)
    all_results.sort(key=lambda r: r["mean_rmsd"])
    top_results = all_results[:args.top_nodes]

    elapsed = _time.time() - t0
    print(f"\n  [✓] {len(all_results)} nodes analysed in {elapsed:.0f}s.")
    print(f"  Top {len(top_results)} selected by fragment RMSD consistency.\n")

    # ── Step 6: Print summary table ───────────────────────────────────────
    print("=" * 120)
    print(f"Top {len(top_results)} SAE Nodes by Motif Consistency "
          f"(residue-level analysis)")
    print("=" * 120)
    print(
        f"{'Node':>6s} {'RMSD':>8s} {'med':>7s} {'σ':>7s} "
        f"{'Nfrag':>6s} {'AUC_gb':>7s} {'LPO':>7s} "
        f"{'ρ':>6s} {'rAUC':>6s} {'AP':>6s} {'cos':>6s} "
        f"{'Pthr':>5s} {'Nact':>6s}   Top enrichments"
    )
    print("-" * 140)
    for r in top_results[:40]:
        # Top enrichment
        top_enrich = sorted(
            r["enrichments"].items(),
            key=lambda x: x[1]["fold_enrichment"], reverse=True,
        )
        enrich_str = ", ".join(
            f"{c}:{v['fold_enrichment']:.1f}×"
            for c, v in top_enrich[:3]
            if v["fold_enrichment"] > 1.0
        )
        conc = r.get("concordance", {})
        print(
            f"{r['sae_node']:>6d} {r['mean_rmsd']:>8.2f} "
            f"{r['median_rmsd']:>7.2f} {r['std_rmsd']:>7.2f} "
            f"{r['n_fragments']:>6d} "
            f"{r.get('gbm_auc_cv', 0.0):>7.3f} "
            f"{r.get('lpo_auc', 0.0):>7.3f} "
            f"{conc.get('spearman_r', 0.0):>6.3f} "
            f"{conc.get('residue_auroc', 0.0):>6.3f} "
            f"{conc.get('avg_precision', 0.0):>6.3f} "
            f"{conc.get('cosine_sim', 0.0):>6.3f} "
            f"{r.get('optimal_threshold', 0.5):>5.2f} "
            f"{r['n_activated']:>6d}   {enrich_str}"
        )

    # ── Step 7: Save outputs ──────────────────────────────────────────────
    print(f"\n[6/8] Saving results …")

    # 7a. Summary YAML (human-readable, no large arrays)
    summary_for_yaml = []
    for r in top_results:
        conc = r.get("concordance", {})
        summary_for_yaml.append({
            "sae_node": r["sae_node"],
            "mean_rmsd": round(r["mean_rmsd"], 4),
            "median_rmsd": round(r["median_rmsd"], 4),
            "std_rmsd": round(r["std_rmsd"], 4),
            "n_fragments": r["n_fragments"],
            "n_activated": r["n_activated"],
            "n_background": r["n_background"],
            "threshold": round(r["threshold"], 6),
            "auc_cv_decision_tree": round(r["auc_cv"], 4),
            "auc_cv_gbm": round(r.get("gbm_auc_cv", 0.0), 4),
            "auc_cv_random_forest": round(r["rf_auc_cv"], 4),
            "leave_proteins_out_auc": round(r.get("lpo_auc", 0.0), 4),
            "n_unique_proteins_in_cv": r.get("n_unique_proteins", 0),
            "optimal_geom_threshold": round(r.get("optimal_threshold", 0.5), 4),
            "f1_cv": round(r["f1_cv"], 4),
            "concordance_spearman_r": conc.get("spearman_r", 0.0),
            "concordance_spearman_p": conc.get("spearman_p", 1.0),
            "concordance_residue_auroc": conc.get("residue_auroc", 0.0),
            "concordance_avg_precision": conc.get("avg_precision", 0.0),
            "concordance_cosine_sim": conc.get("cosine_sim", 0.0),
            "concordance_binary_f1": conc.get("f1", 0.0),
            "concordance_binary_iou": conc.get("iou", 0.0),
            "feature_importances": r["feature_importances"],
            "enrichments": r["enrichments"],
            "decision_tree_rules": r["rules"],
        })
    yaml_path = out / "motif_summary.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(summary_for_yaml, f, default_flow_style=False, width=120)
    print(f"  Summary YAML → {yaml_path}")

    # 7b. Motif templates as PDB files
    template_dir = out / "motif_templates"
    for r in top_results:
        if r["mean_structure"] is not None:
            save_motif_template_pdb(
                r["mean_structure"], r["sae_node"],
                template_dir / f"node{r['sae_node']}_template.pdb",
            )
    print(f"  PDB templates → {template_dir}/")

    # ── Step 8: Plots ─────────────────────────────────────────────────────
    print("\n[7/8] Generating plots …")
    plot_dir = out / "plots"

    # 8a. Node ranking overview
    plot_node_ranking(top_results, plot_dir / "node_ranking.png")

    # Build accession → protein_data lookup for backbone plots
    protein_lookup = {p["accession"]: p for p in protein_data}

    # 8b. Per-node detailed plots (top 20 only to keep reasonable)
    n_detailed = min(20, len(top_results))
    for r in top_results[:n_detailed]:
        ni = r["sae_node"]
        node_dir = plot_dir / f"node_{ni}"

        if r["mean_structure"] is not None and r["per_pos_std"] is not None:
            plot_motif_template_3d(
                r["mean_structure"], r["per_pos_std"], ni, r["mean_rmsd"],
                node_dir / "motif_template_3d.png",
            )

        if r["aligned_fragments"] is not None and len(r["aligned_fragments"]) >= 3:
            plot_fragment_overlay(
                r["aligned_fragments"], ni, r["mean_rmsd"],
                node_dir / "fragment_overlay.png",
            )

        if r["rmsds"]:
            plot_rmsd_histogram(r["rmsds"], ni, node_dir / "rmsd_histogram.png")

        if r["enrichments"]:
            plot_enrichment_bars(
                r["enrichments"], ni, node_dir / "enrichment_bars.png",
            )

        if r["feature_importances"]:
            plot_feature_importances(
                r["feature_importances"], ni, node_dir / "feature_importances.png",
            )

        # Full-backbone plots with motif region highlighted
        if r.get("top_activated_entries"):
            plot_top_backbones_for_node(
                r, protein_lookup, half_w=args.half_window,
                plot_dir=plot_dir, max_proteins=5,
            )
            # InterPLM Fig 1C–style per-residue activation profiles
            plot_activation_profiles_for_node(
                r, protein_lookup, half_w=args.half_window,
                plot_dir=plot_dir, max_proteins=5,
            )
            # Geometry-vs-activation verification overlays
            plot_geometry_overlays_for_node(
                r, protein_lookup, half_w=args.half_window,
                plot_dir=plot_dir, max_proteins=5,
            )

    # 8c. Global enrichment heatmap (nodes × categories)
    _plot_global_enrichment_heatmap(top_results[:n_detailed], plot_dir)

    print(f"\n[8/8] All done!")
    print("=" * 72)
    print("✅  Residue-level motif discovery complete!")
    print("=" * 72)
    print(f"All outputs in: {out}/")
    print(f"  • motif_summary.yaml           – full results for top {len(top_results)} nodes")
    print(f"  • motif_templates/             – PDB files (open in PyMOL/ChimeraX)")
    print(f"  • plots/node_ranking.png       – overview ranking")
    print(f"  • plots/node_<N>/              – per-node detailed plots:")
    print(f"      motif_template_3d.png      – mean motif structure (coloured by flexibility)")
    print(f"      fragment_overlay.png       – superimposed top fragments")
    print(f"      rmsd_histogram.png         – RMSD distribution")
    print(f"      enrichment_bars.png        – category fold-enrichments")
    print(f"      feature_importances.png    – decision tree feature weights")
    print(f"      backbone_<ACC>.png         – full protein backbone with motif highlighted")
    print(f"      activation_profile_<ACC>.png – per-residue activation profile (InterPLM Fig 1C style)")
    print(f"      geometry_overlay_<ACC>.png  – activation vs geometry-predicted probability (verification)")
    print(f"  • plots/enrichment_heatmap.png – global category × node heatmap")
    print("=" * 72)


def _plot_global_enrichment_heatmap(results: list[dict], plot_dir: Path):
    """Heatmap: rows = structural categories, columns = top SAE nodes."""
    if not results:
        return
    nodes = [r["sae_node"] for r in results]
    n_cats = len(CATEGORY_NAMES)
    mat = np.ones((n_cats, len(nodes)))

    for j, r in enumerate(results):
        for i, cname in enumerate(CATEGORY_NAMES):
            if cname in r["enrichments"]:
                mat[i, j] = r["enrichments"][cname]["fold_enrichment"]

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(nodes)), 5))
    im = ax.imshow(
        np.log2(np.clip(mat, 0.1, 100)),
        aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
        interpolation="nearest",
    )
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(CATEGORY_NAMES, fontsize=9)
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels([f"N{n}" for n in nodes], rotation=70, fontsize=7)
    ax.set_xlabel("SAE Node")
    ax.set_title("Structural Category Enrichment (log₂ fold-change)")
    plt.colorbar(im, ax=ax, label="log₂(enrichment)")
    plt.tight_layout()
    path = plot_dir / "enrichment_heatmap.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Enrichment heatmap → {path}")


if __name__ == "__main__":
    main()
