"""Protein-level geometric feature extraction (55-dimensional).

Computes a single 55-element descriptor vector for an entire protein chain
from its Ca backbone coordinates. Features span global topology (writhe,
gyration), helix statistics, contact-filtered helix pair descriptors,
turn/hairpin/strand proxies, signed torsion statistics, and windowed
local profile summaries.

Adapted from ``protein_results/build_activation_dataset.py:451-548``.
"""

from __future__ import annotations

import numpy as np

# Geometric primitives from the existing geometry library.
# These are pure numpy/numba functions with no matplotlib dependency.
from protein_results.geometry.compute_geometric_features import (
    average_curvature,
    average_torsion,
    dihedral_sign_consistency,
    end_to_end_distance,
    extended_fraction,
    gyration_asphericity,
    hairpin_score,
    helical_consistency,
    helix_segments,
    helix_statistics,
    helix_statistics_contact_filtered,
    kink_index,
    local_curvature,
    local_planarity,
    local_planarity_score,
    local_torsion,
    local_writhe,
    radius_of_gyration,
    signed_torsion,
    turn_density,
    writhe,
)

from proteinlens.analysis.geometry.residue_features import (
    ca_backbone,
    detect_alpha_helices_from_ca,
)

# ---------------------------------------------------------------------------
# Feature name list (55 elements, one per dimension of the output vector).
# Order MUST match the values list built in compute_protein_geometry().
# ---------------------------------------------------------------------------
GEOM_FEATURE_NAMES: list[str] = [
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
    # contact-filtered helix pair stats
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
    # helix segment stats
    "helix_fraction",
    "mean_helix_len",
    "std_helix_len",
    "max_helix_len",
    # turn / hairpin / strand proxies
    "turn_density",
    "hairpin_score",
    "extended_fraction",
    # signed torsion stats
    "signed_torsion_mean",
    "signed_torsion_std",
    "signed_torsion_frac_pos",
    "signed_torsion_frac_neg",
    # dihedral consistency
    "dihedral_sign_consistency",
    # local (windowed) profile summaries
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

assert len(GEOM_FEATURE_NAMES) == 55, (
    f"Expected 55 feature names, got {len(GEOM_FEATURE_NAMES)}"
)


def compute_protein_geometry(pdb_text: str) -> dict[str, float] | None:
    """Compute all 55 geometric features from a PDB text string.

    This is the protein-level geometry extraction entry point. It parses the
    Ca backbone from *pdb_text*, detects alpha-helical segments from the Ca
    trace, and then computes 55 scalar descriptors covering global topology,
    helix pair statistics, turn/strand proxies, and windowed profile
    summaries.

    Parameters
    ----------
    pdb_text : str
        Full PDB-format text (ATOM/HETATM records). An AlphaFold predicted
        structure or experimental PDB are both fine.

    Returns
    -------
    dict[str, float] | None
        Mapping of feature name -> float value (55 entries), or ``None`` if
        the PDB cannot be parsed or has too few Ca atoms (< 4).
    """
    # -- Parse Ca backbone --
    try:
        ca = ca_backbone(pdb_text, chain_id=None)
    except Exception:
        return None
    if ca is None or len(ca) < 4:
        return None

    try:
        helices = detect_alpha_helices_from_ca(ca)

        # -- Original global features --
        wr_d = writhe(ca, ca)
        wr = float(np.sum(wr_d))
        # Vassiliev V2 is O(n^4) -- too expensive for pipeline use; set to 0.
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

        # -- Contact-filtered helix pair stats (12 values) --
        (cp_m, cp_s, cd_m, cd_s, cp_top3, cp_frac,
         ca_mean, ca_std, ca_lt15, ca_gt60,
         n_hel, n_cpairs) = helix_statistics_contact_filtered(ca, helices)

        # -- Helix segment stats --
        _n_hel, h_frac, h_mean_len, h_std_len, h_max_len = helix_segments(
            ca, helices
        )

        # -- Turn / hairpin / strand proxies --
        td = float(turn_density(ca))
        hp = float(hairpin_score(ca))
        ef = float(extended_fraction(ca))

        # -- Signed torsion stats --
        st_mean, st_std, st_fp, st_fn = signed_torsion(ca)

        # -- Dihedral sign consistency --
        dsc = float(dihedral_sign_consistency(ca))

        # -- Local (windowed) profile summaries --
        def _profile_stats(arr: np.ndarray) -> tuple[float, float, float, float]:
            """Reduce a 1-D windowed profile to (mean, std, max, range)."""
            if arr.size == 0:
                return 0.0, 0.0, 0.0, 0.0
            mn = float(np.mean(arr))
            sd = float(np.std(arr))
            mx = float(np.max(arr))
            rng = float(mx - np.min(arr))
            return mn, sd, mx, rng

        lc_mean, lc_std, lc_max, lc_rng = _profile_stats(local_curvature(ca))
        lt_mean, lt_std, lt_max, lt_rng = _profile_stats(local_torsion(ca))
        lp_mean, lp_std, lp_max, lp_rng = _profile_stats(local_planarity(ca))
        lw_mean, lw_std, lw_max, lw_rng = _profile_stats(local_writhe(ca))

    except Exception:
        return None

    # Assemble the 55-element value list. Order MUST match GEOM_FEATURE_NAMES.
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

    assert len(values) == len(GEOM_FEATURE_NAMES), (  # pragma: no cover
        f"Values/names mismatch: {len(values)} vs {len(GEOM_FEATURE_NAMES)}"
    )

    # Replace NaN/inf with 0.0 so the output is always finite.
    # Some features (e.g. helix_parallel_mean) return NaN when a protein
    # has no helix pairs. Replacing here makes the function self-contained
    # rather than requiring every consumer to filter NaNs downstream.
    values = [0.0 if not np.isfinite(v) else v for v in values]

    return dict(zip(GEOM_FEATURE_NAMES, values))
