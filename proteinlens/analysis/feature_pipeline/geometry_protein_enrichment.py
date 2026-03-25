"""Stage 6b: Protein-level LassoCV geometry enrichment per SAE node.

For each SAE node, fits a LassoCV regression predicting the node's
protein-level max activation from the 55-dim protein geometry vector.
Produces a sparse "monomial" (e.g. ``y_hat = 0.34*hairpin_score - 0.19*avg_curvature + 0.003``)
and cross-validated R^2 that tells us how well global 3D shape predicts
whether a protein activates a given SAE feature.

Outputs are written to ``geometry_enrichment/{feat:04d}.json`` (one per
feature with enough data) and a ``geometry_enrichment/summary.json``
aggregating all results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.geometry.classifiers import fit_lasso_single_node
from proteinlens.analysis.geometry.protein_features import GEOM_FEATURE_NAMES  # fallback only

logger = logging.getLogger(__name__)


def run_geometry_protein_enrichment(config: PipelineConfig) -> None:
    """Stage 6b entry point: protein-level LassoCV per SAE node.

    Loads the precomputed protein geometry matrix (from Stage 6a) and the
    per-protein max activations memmap (from the survey stage), then fits
    a LassoCV for each node on the subset of proteins that fire.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration. Requires:
        - ``geometry_protein_features_path``: from Stage 6a
        - ``protein_feature_maxes_path``: from survey stage
        - ``pipeline_state_path``: for accession-to-index mapping
        - ``feature_max_path``: per-feature global max activations
    """
    enrichment_dir = config.geometry_enrichment_dir

    # -- Load geometry features from Stage 6a --
    if not config.geometry_protein_features_path.exists():
        logger.warning("geometry_protein_features.npz not found. Run Stage 6a first.")
        return

    geom_data = np.load(config.geometry_protein_features_path, allow_pickle=True)
    geom_accessions = list(geom_data["accessions"])
    geom_matrix = geom_data["geometry_matrix"]  # (N_geom, 55)
    # Use feature names stored in the NPZ rather than hard-coded constant,
    # so changes to the geometry feature set are picked up automatically.
    geom_names = list(geom_data["feature_names"])
    n_geom = len(geom_accessions)
    logger.info("Loaded geometry for %d proteins (%d features)", n_geom, len(geom_names))

    # -- Load pipeline state for accession-to-index mapping --
    if not config.pipeline_state_path.exists():
        logger.warning("pipeline_state.json not found. Run survey stage first.")
        return

    state = json.loads(config.pipeline_state_path.read_text())
    acc_to_idx: dict[str, int] = state.get("accession_index", {})

    # -- Load per-protein max activations memmap --
    if not config.protein_feature_maxes_path.exists():
        logger.warning("protein_feature_maxes.npy not found. Run survey stage first.")
        return

    # Load feature max activations to know how many features exist
    feature_maxes = np.load(config.feature_max_path)
    n_features = len(feature_maxes)

    # Memmap: shape (n_proteins, n_features)
    n_proteins_total = len(acc_to_idx)
    act_memmap = np.memmap(
        config.protein_feature_maxes_path,
        dtype=np.float32,
        mode="r",
        shape=(n_proteins_total, n_features),
    )

    # -- Build index: map geometry accessions to their memmap rows --
    # Only keep accessions that exist in both geometry AND memmap
    geom_idx_to_memmap_row: list[int] = []
    valid_geom_indices: list[int] = []
    for gi, acc in enumerate(geom_accessions):
        if acc in acc_to_idx:
            geom_idx_to_memmap_row.append(acc_to_idx[acc])
            valid_geom_indices.append(gi)

    if not valid_geom_indices:
        logger.warning("No overlap between geometry proteins and surveyed proteins.")
        return

    valid_geom_indices = np.array(valid_geom_indices)
    memmap_rows = np.array(geom_idx_to_memmap_row)

    # Subset geometry matrix to only proteins in the memmap
    X_geom = geom_matrix[valid_geom_indices]  # (N_valid, 55)
    # Filter out rows with any NaN/inf in geometry
    geom_valid = np.all(np.isfinite(X_geom), axis=1)

    logger.info(
        "%d proteins with both geometry and activations (%d with finite geometry)",
        len(valid_geom_indices), int(geom_valid.sum()),
    )

    # -- Fit LassoCV per node --
    summary_features: dict[str, dict] = {}
    n_fitted = 0
    n_skipped_inactive = 0
    n_skipped_few = 0
    n_skipped_no_signal = 0

    for ni in range(n_features):
        if ni % 500 == 0:
            logger.info(
                "Node %d/%d (fitted=%d, skipped_inactive=%d, skipped_few=%d)",
                ni, n_features, n_fitted, n_skipped_inactive, n_skipped_few,
            )

        # Skip dead features
        if feature_maxes[ni] == 0:
            n_skipped_inactive += 1
            continue

        # Get max activation values for proteins that have geometry
        y_all = act_memmap[memmap_rows, ni]

        # Filter to active proteins (activation > 0) with valid geometry
        active_mask = (y_all > 0) & geom_valid
        n_active = int(active_mask.sum())

        if n_active < config.geometry_min_active_proteins:
            n_skipped_few += 1
            continue

        X = X_geom[active_mask]
        y = y_all[active_mask]

        result = fit_lasso_single_node(
            X, y, geom_names,
            cv_folds=config.geometry_lasso_cv_folds,
        )

        if result is None:
            n_skipped_no_signal += 1
            continue

        # -- Write per-feature JSON --
        feat_path = enrichment_dir / f"{ni:04d}.json"

        # Load existing JSON if present (Stage 6c may have written it)
        if feat_path.exists():
            feat_json = json.loads(feat_path.read_text())
        else:
            feat_json = {
                "feature_id": ni,
                "feature_max_activation": float(feature_maxes[ni]),
            }

        feat_json["geometric_protein_level"] = {
            "r2_cv": result["r2_cv"],
            "r2": result["r2"],
            "r2_adj": result["r2_adj"],
            "pearson_r": result["pearson_r"],
            "alpha_chosen": result["alpha_chosen"],
            "monomial": result["monomial"],
            "n_samples": result["n_samples"],
            "n_nonzero": result["n_nonzero"],
            "top_features": result["top_features"],
        }

        feat_path.write_text(json.dumps(feat_json, indent=2))

        summary_features[str(ni)] = {
            "protein_r2_cv": result["r2_cv"],
            "protein_pearson_r": result["pearson_r"],
        }
        n_fitted += 1

    # -- Write summary JSON --
    summary = {
        "n_features_protein_level": n_fitted,
        "n_features_skipped_inactive": n_skipped_inactive,
        "n_features_skipped_few_proteins": n_skipped_few,
        "n_features_skipped_no_signal": n_skipped_no_signal,
        "n_proteins_with_geometry": n_geom,
        "features": summary_features,
    }
    summary_path = enrichment_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info(
        "Done. Fitted Lasso for %d nodes (skipped: %d inactive, %d too few, %d no signal)",
        n_fitted, n_skipped_inactive, n_skipped_few, n_skipped_no_signal,
    )
