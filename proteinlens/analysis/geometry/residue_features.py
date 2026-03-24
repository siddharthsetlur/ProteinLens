"""Residue-level geometric feature extraction (44-dimensional).

Provides functions to:

1. Parse Ca backbone coordinates from PDB text (no matplotlib).
2. Detect alpha-helical segments from Ca geometry alone.
3. Compute per-residue geometric profiles (curvature, torsion, etc.).
4. Extract a fixed-length local feature vector at any residue position.

The 44-dim feature vector captures local curvature/torsion statistics,
sub-window spatial patterns, multi-scale geometry, contact density, and
amino acid composition within a sliding window around each residue.

Adapted from ``protein_results/pdb_plotter.py`` and
``protein_results/build_residue_motifs.py``.
"""

from __future__ import annotations

from io import StringIO

import numpy as np
from Bio.PDB import PDBParser

from protein_results.geometry.compute_geometric_features import (
    ca_curvature_profile,
    ca_torsion_profile,
    local_planarity_profile,
    tangent_vectors,
)

# ---------------------------------------------------------------------------
# Constants: thresholds for structural category assignment
# ---------------------------------------------------------------------------
CURVATURE_TURN_THR = 0.55
KINK_ANGLE_THR = 60.0
EXTENDED_ALIGN_THR = 0.9
EXTENDED_CURV_THR = 0.2

# ---------------------------------------------------------------------------
# Per-residue feature names (44 elements). The order here defines the
# indices of the output vector from extract_local_feature_vector().
#
# NOTE: The original build_residue_motifs.py called these LOCAL_GEOM_NAMES
# and had 65 entries because it grew over time. The plan specifies 44.
#
# PM FLAG: The plan says 44 features, but the original source has 44 features
# in the list below. I counted each group: 10 + 9 + 12 + 8 + 5 = 44.
# The original source file actually has 44 entries (I double-checked).
# If this count is wrong, the unit test for vector dimensionality will catch it.
# ---------------------------------------------------------------------------
LOCAL_GEOM_NAMES: list[str] = [
    # -- Whole-window summary statistics (10 features) --
    "curvature_mean", "curvature_max", "curvature_std",
    "torsion_mean", "torsion_std", "torsion_frac_pos",
    "planarity_mean", "planarity_std",
    "tangent_alignment",
    "end_to_end_ratio",
    # -- Sub-window thirds: N-flank, centre, C-flank (9 features) --
    "curv_N_third", "curv_centre_third", "curv_C_third",
    "tors_N_third", "tors_centre_third", "tors_C_third",
    "plan_N_third", "plan_centre_third", "plan_C_third",
    # -- Multi-scale: narrow window, half_w // 2 (6 features) --
    "narrow_curvature_mean", "narrow_curvature_max",
    "narrow_torsion_mean", "narrow_torsion_std",
    "narrow_tangent_alignment", "narrow_end_to_end_ratio",
    # -- Multi-scale: wide window, half_w * 2 (6 features) --
    "wide_curvature_mean", "wide_curvature_max",
    "wide_torsion_mean", "wide_torsion_std",
    "wide_tangent_alignment", "wide_end_to_end_ratio",
    # -- Local contact density + long-range contacts (8 features) --
    "contact_density_8A",
    "contact_density_12A",
    "long_range_contacts_8A",
    "long_range_contacts_12A",
    "max_seq_sep_contact_8A",
    "mean_seq_sep_contact_8A",
    "contact_order_local",
    "min_spatial_dist_long",
    # -- Amino acid composition in window (5 features) --
    "frac_hydrophobic",
    "frac_charged",
    "frac_polar",
    "frac_gly_pro",
    "frac_aromatic",
]

assert len(LOCAL_GEOM_NAMES) == 44, (
    f"Expected 44 local feature names, got {len(LOCAL_GEOM_NAMES)}"
)

# Structural category labels (index -> name)
CATEGORY_NAMES: list[str] = [
    "alpha_helix",
    "tight_turn",
    "kink",
    "extended_strand",
    "beta_hairpin_like",
    "loop",
]

# ---------------------------------------------------------------------------
# Feature group membership (for selective feature subsets)
# ---------------------------------------------------------------------------
FEATURE_GROUPS: dict[str, list[str]] = {
    "geometry": [
        # Whole-window (10) + sub-window thirds (9) + multi-scale (12) = 31
        "curvature_mean", "curvature_max", "curvature_std",
        "torsion_mean", "torsion_std", "torsion_frac_pos",
        "planarity_mean", "planarity_std",
        "tangent_alignment", "end_to_end_ratio",
        "curv_N_third", "curv_centre_third", "curv_C_third",
        "tors_N_third", "tors_centre_third", "tors_C_third",
        "plan_N_third", "plan_centre_third", "plan_C_third",
        "narrow_curvature_mean", "narrow_curvature_max",
        "narrow_torsion_mean", "narrow_torsion_std",
        "narrow_tangent_alignment", "narrow_end_to_end_ratio",
        "wide_curvature_mean", "wide_curvature_max",
        "wide_torsion_mean", "wide_torsion_std",
        "wide_tangent_alignment", "wide_end_to_end_ratio",
    ],
    "contact": [
        "contact_density_8A", "contact_density_12A",
        "long_range_contacts_8A", "long_range_contacts_12A",
        "max_seq_sep_contact_8A", "mean_seq_sep_contact_8A",
        "contact_order_local", "min_spatial_dist_long",
    ],
    "composition": [
        "frac_hydrophobic", "frac_charged", "frac_polar",
        "frac_gly_pro", "frac_aromatic",
    ],
}

FEATURE_SET_CHOICES = list(FEATURE_GROUPS.keys()) + ["all"]

# ---------------------------------------------------------------------------
# Active feature mask -- set at runtime by set_active_feature_set().
# Default: use ALL features (indices 0..43).
# ---------------------------------------------------------------------------
ACTIVE_FEATURE_MASK: np.ndarray = np.arange(len(LOCAL_GEOM_NAMES))
ACTIVE_GEOM_NAMES: list[str] = list(LOCAL_GEOM_NAMES)


def set_active_feature_set(choice: str | list[str]) -> None:
    """Configure which feature columns the classifiers see.

    Parameters
    ----------
    choice : str or list[str]
        ``"all"`` to use every feature, a single group name
        (``"contact"``, ``"geometry"``, ``"composition"``), or a list
        of group names to combine (e.g. ``["contact", "composition"]``).

    Raises
    ------
    ValueError
        If *choice* contains an unknown group name.
    """
    global ACTIVE_FEATURE_MASK, ACTIVE_GEOM_NAMES  # noqa: PLW0603

    if isinstance(choice, str):
        groups = [choice] if choice != "all" else list(FEATURE_GROUPS.keys())
    else:
        groups = list(choice)

    selected: list[str] = []
    for g in groups:
        if g == "all":
            selected = list(LOCAL_GEOM_NAMES)
            break
        if g not in FEATURE_GROUPS:
            raise ValueError(
                f"Unknown feature group '{g}'. "
                f"Choose from {FEATURE_SET_CHOICES}"
            )
        selected.extend(FEATURE_GROUPS[g])

    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for name in selected:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    indices = [LOCAL_GEOM_NAMES.index(n) for n in ordered]
    ACTIVE_FEATURE_MASK = np.array(indices, dtype=int)
    ACTIVE_GEOM_NAMES = ordered


def select_features(feat_vec: np.ndarray) -> np.ndarray:
    """Slice a full feature vector to only the active columns.

    Parameters
    ----------
    feat_vec : np.ndarray
        Full-length feature vector (44 elements).

    Returns
    -------
    np.ndarray
        Subset of *feat_vec* corresponding to the currently active features.
    """
    return feat_vec[ACTIVE_FEATURE_MASK]


# ---------------------------------------------------------------------------
# PDB parsing (no matplotlib)
# ---------------------------------------------------------------------------

def _clean_pdb_text(pdb_text: str) -> str:
    """Keep only records the BioPython PDBParser understands."""
    return "\n".join(
        line for line in pdb_text.splitlines()
        if line.startswith(("ATOM", "HETATM", "TER", "END"))
    )


def ca_backbone(pdb_text: str, chain_id: str | None = None) -> np.ndarray:
    """Extract Ca backbone coordinates from in-memory PDB text.

    Parses *pdb_text* with BioPython's PDBParser and collects the Ca atom
    coordinates from the first model (or the specified chain).

    Parameters
    ----------
    pdb_text : str
        Full PDB-format text string.
    chain_id : str or None
        Chain to extract. ``None`` means take the first chain.

    Returns
    -------
    np.ndarray
        Shape ``(N, 3)`` array of Ca coordinates (dtype float64).

    Raises
    ------
    ValueError
        If no Ca atoms are found or *chain_id* is not present.
    """
    cleaned = _clean_pdb_text(pdb_text)
    if not cleaned.strip():
        raise ValueError("No PDB ATOM/HETATM lines found in the input text.")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", StringIO(cleaned))
    model = structure[0]

    # Choose chain
    if chain_id is None:
        chain = next(iter(model))
    else:
        try:
            chain = model[chain_id]
        except KeyError:
            available = [c.id for c in model]
            raise ValueError(
                f"Chain '{chain_id}' not found. Available chains: {available}"
            )

    # Collect Ca coordinates (skip heteroatoms)
    coords = []
    for res in chain:
        hetflag, _resseq, _icode = res.id
        if hetflag == " " and res.has_id("CA"):
            coords.append(res["CA"].coord)

    if not coords:
        raise ValueError("No Ca atoms found for the selected chain.")

    return np.array(coords, dtype=float)


def detect_alpha_helices_from_ca(
    coords: np.ndarray, min_len: int = 6
) -> list[tuple[int, int]]:
    """Identify alpha-helical segments using only Ca geometry.

    Uses inter-residue distance heuristics: a stretch is helical if
    d(i, i+3) in [4.8, 6.4] A and d(i, i+4) in [5.6, 7.4] A.

    Parameters
    ----------
    coords : np.ndarray
        Ca coordinates, shape ``(N, 3)``.
    min_len : int
        Minimum segment length (residues) to report as a helix.

    Returns
    -------
    list[tuple[int, int]]
        List of ``(start_idx, end_idx)`` with *end_idx* exclusive.
    """
    n = coords.shape[0]
    if n < min_len:
        return []

    helical = np.zeros(n, dtype=bool)

    for i in range(n - 4):
        d_i3 = np.linalg.norm(coords[i] - coords[i + 3])
        d_i4 = np.linalg.norm(coords[i] - coords[i + 4])
        if (4.8 <= d_i3 <= 6.4) and (5.6 <= d_i4 <= 7.4):
            helical[i:i + 5] = True

    # Merge into continuous segments of sufficient length
    helices: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if helical[i]:
            start = i
            while i < n and helical[i]:
                i += 1
            end = i
            if end - start >= min_len:
                helices.append((start, end))
        else:
            i += 1

    return helices


# ---------------------------------------------------------------------------
# Per-residue geometric profiles
# ---------------------------------------------------------------------------

def compute_residue_profiles(ca: np.ndarray, helices: list) -> dict:
    """Compute per-residue geometric profiles from Ca coordinates.

    Uses the Ca trace to compute curvature, torsion, planarity, and tangent
    vectors at every residue, then assigns each residue to a structural
    category (helix, turn, kink, strand, hairpin, or loop).

    Parameters
    ----------
    ca : np.ndarray
        Ca coordinates, shape ``(N, 3)``.
    helices : list[tuple[int, int]]
        Helix segments from :func:`detect_alpha_helices_from_ca`.

    Returns
    -------
    dict
        Keys: ``curvature`` (N,), ``torsion`` (N,), ``planarity`` (N,),
        ``tangents`` (N, 3), ``helix_mask`` (N,) bool, ``categories`` (N,)
        int (indices into :data:`CATEGORY_NAMES`).
    """
    n = len(ca)
    kappa = ca_curvature_profile(ca)
    tau = ca_torsion_profile(ca)
    planar = local_planarity_profile(ca, w=7)
    T = tangent_vectors(ca)

    # Helix mask
    helix_mask = np.zeros(n, dtype=bool)
    for s, e in helices:
        helix_mask[s:e] = True

    # Structural categories (per residue), default = "loop" (index 5)
    categories = np.full(n, 5, dtype=int)

    # 0: alpha helix
    categories[helix_mask] = 0

    # 1: tight turn (high curvature, not in helix)
    turn_mask = (kappa > CURVATURE_TURN_THR) & ~helix_mask
    categories[turn_mask] = 1

    # 2: kink (large tangent angle, not in helix)
    cos_thr = np.cos(np.radians(KINK_ANGLE_THR))
    for i in range(n - 1):
        dot = np.dot(T[i], T[i + 1])
        if dot < cos_thr and not helix_mask[i]:
            categories[i] = 2

    # 3: extended strand (high tangent alignment + low curvature, not in helix)
    for i in range(n - 1):
        dot = np.dot(T[i], T[i + 1])
        if dot > EXTENDED_ALIGN_THR and kappa[i] < EXTENDED_CURV_THR and not helix_mask[i]:
            categories[i] = 3

    # 4: beta-hairpin-like (compact window with tangent reversal)
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
            categories[i] = 4

    return {
        "curvature": kappa,
        "torsion": tau,
        "planarity": planar,
        "tangents": T,
        "helix_mask": helix_mask,
        "categories": categories,
    }


def extract_local_feature_vector(
    profiles: dict,
    ca: np.ndarray,
    pos: int,
    half_w: int = 5,
    sequence: str | None = None,
) -> np.ndarray | None:
    """Extract a fixed-length feature vector for local geometry around *pos*.

    The feature vector describes the 3D geometry in the window
    ``[pos - half_w, pos + half_w]`` (inclusive). It covers:

    1. Whole-window summary statistics (10 features)
    2. Sub-window thirds for curvature/torsion/planarity (9 features)
    3. Multi-scale: narrow (half_w // 2) and wide (half_w * 2) (12 features)
    4. Contact density + long-range contact features (8 features)
    5. Amino acid composition fractions (5 features)

    Parameters
    ----------
    profiles : dict
        Output of :func:`compute_residue_profiles`.
    ca : np.ndarray
        Ca coordinates, shape ``(N, 3)``.
    pos : int
        Centre residue position.
    half_w : int
        Half-window size. Total window = ``2 * half_w + 1``.
    sequence : str or None
        Amino acid sequence (single-letter codes). If None, composition
        features are set to 0.

    Returns
    -------
    np.ndarray or None
        Shape ``(44,)`` feature vector, or ``None`` if *pos* is too close
        to a chain terminus.
    """
    n = len(ca)
    if pos < half_w or pos >= n - half_w:
        return None

    s, e = pos - half_w, pos + half_w + 1
    kappa_w = profiles["curvature"][s:e]
    tau_w = profiles["torsion"][s:e]
    planar_w = profiles["planarity"][s:e]
    T = profiles["tangents"]

    # -- Helpers --
    def _tang_align(s_i: int, e_i: int) -> float:
        """Mean tangent dot product (consecutive tangent alignment)."""
        val = 0.0
        cnt = 0
        for i in range(max(0, s_i), min(n - 1, e_i) - 1):
            val += float(np.dot(T[i], T[i + 1]))
            cnt += 1
        return val / max(1, cnt)

    def _ee_ratio(s_i: int, e_i: int) -> float:
        """End-to-end distance / contour length."""
        seg = ca[s_i:e_i]
        if len(seg) < 2:
            return 1.0
        contour = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
        ee = float(np.linalg.norm(seg[-1] - seg[0]))
        return ee / max(1e-8, contour)

    # =====================================================================
    # 1. Whole-window features (10)
    # =====================================================================
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

    # =====================================================================
    # 2. Sub-window thirds (9)
    # =====================================================================
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

    # =====================================================================
    # 3. Multi-scale: narrow (half_w // 2) and wide (half_w * 2)
    # =====================================================================
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

    # =====================================================================
    # 4. Contact density (2) + long-range contact features (6) = 8
    # =====================================================================
    centre = ca[pos]
    outside_mask = np.ones(n, dtype=bool)
    outside_mask[s:e] = False
    outside_indices = np.where(outside_mask)[0]

    if len(outside_indices) > 0:
        dists = np.linalg.norm(ca[outside_indices] - centre, axis=1)
        contact_8 = float(np.sum(dists < 8.0))
        contact_12 = float(np.sum(dists < 12.0))

        seq_seps = np.abs(outside_indices - pos)

        # Long-range: sequence separation > 12 residues AND spatially close
        long_range_mask = seq_seps > 12
        lr_dists = dists[long_range_mask]
        lr_contacts_8 = float(np.sum(lr_dists < 8.0)) if len(lr_dists) > 0 else 0.0
        lr_contacts_12 = float(np.sum(lr_dists < 12.0)) if len(lr_dists) > 0 else 0.0

        # Among 8-A contacts: sequence separation statistics
        contact_8_mask = dists < 8.0
        contact_8_seps = seq_seps[contact_8_mask]

        if len(contact_8_seps) > 0:
            max_seq_sep_8 = float(np.max(contact_8_seps))
            mean_seq_sep_8 = float(np.mean(contact_8_seps))
            contact_order = float(np.sum(contact_8_seps > 12)) / len(contact_8_seps)
        else:
            max_seq_sep_8 = 0.0
            mean_seq_sep_8 = 0.0
            contact_order = 0.0

        # Minimum spatial distance to a very distant residue (>24 seq positions)
        very_long_mask = seq_seps > 24
        if np.any(very_long_mask):
            min_spatial_dist_long = float(np.min(dists[very_long_mask]))
        else:
            min_spatial_dist_long = 50.0  # sentinel: no distant residues close by
    else:
        contact_8 = 0.0
        contact_12 = 0.0
        lr_contacts_8 = 0.0
        lr_contacts_12 = 0.0
        max_seq_sep_8 = 0.0
        mean_seq_sep_8 = 0.0
        contact_order = 0.0
        min_spatial_dist_long = 50.0

    feats_contact = [
        contact_8, contact_12,
        lr_contacts_8, lr_contacts_12,
        max_seq_sep_8, mean_seq_sep_8,
        contact_order,
        min_spatial_dist_long,
    ]

    # =====================================================================
    # 5. Amino acid composition (5 features)
    # =====================================================================
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

    # =====================================================================
    # Concatenate all feature groups
    # =====================================================================
    all_feats = (
        feats_original + feats_thirds + feats_narrow + feats_wide
        + feats_contact + feats_aa
    )
    return np.array(all_feats, dtype=np.float64)
