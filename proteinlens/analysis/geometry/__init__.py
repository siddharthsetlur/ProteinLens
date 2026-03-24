"""Geometric descriptor extraction for protein structures.

This package extracts 3D geometric features from protein structures at two
scales:

* **Protein-level** (56-dim vector): global shape descriptors such as writhe,
  curvature, and helix statistics.  See :mod:`.protein_features`.
* **Residue-level** (44-dim vector): local geometry in a sliding window around
  each residue position.  See :mod:`.residue_features`.

Classifiers that relate these descriptors to SAE node activations live in
:mod:`.classifiers`.
"""

from proteinlens.analysis.geometry.protein_features import (
    GEOM_FEATURE_NAMES,
    compute_protein_geometry,
)
from proteinlens.analysis.geometry.residue_features import (
    ACTIVE_GEOM_NAMES,
    CATEGORY_NAMES,
    FEATURE_GROUPS,
    LOCAL_GEOM_NAMES,
    ca_backbone,
    compute_residue_profiles,
    detect_alpha_helices_from_ca,
    extract_local_feature_vector,
    select_features,
    set_active_feature_set,
)
from proteinlens.analysis.geometry.classifiers import (
    collect_node_fragments,
    compute_concordance_metrics,
    compute_rmsd,
    fit_lasso_single_node,
    format_monomial,
    kabsch_align,
    superpose_fragments,
    train_motif_classifier,
)

__all__ = [
    # protein_features
    "GEOM_FEATURE_NAMES",
    "compute_protein_geometry",
    # residue_features
    "LOCAL_GEOM_NAMES",
    "ACTIVE_GEOM_NAMES",
    "CATEGORY_NAMES",
    "FEATURE_GROUPS",
    "ca_backbone",
    "detect_alpha_helices_from_ca",
    "compute_residue_profiles",
    "extract_local_feature_vector",
    "select_features",
    "set_active_feature_set",
    # classifiers
    "kabsch_align",
    "compute_rmsd",
    "superpose_fragments",
    "collect_node_fragments",
    "train_motif_classifier",
    "compute_concordance_metrics",
    "fit_lasso_single_node",
    "format_monomial",
]
