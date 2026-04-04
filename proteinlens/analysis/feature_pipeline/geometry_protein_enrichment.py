"""Stage 6b: Protein-level LassoCV geometry enrichment per SAE node.

For each SAE node with enough activated proteins, fits a LassoCV regression
predicting the node's protein-level max activation from the 55-dim protein
geometry vector. Produces a sparse monomial and cross-validated R^2.

Outputs per-feature JSONs in ``geometry_enrichment/{feat:04d}.json`` and a
``geometry_enrichment/summary.json``.

Per-node resumability: nodes whose JSON already contains
``geometric_protein_level`` are skipped on restart.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.geometry.classifiers import fit_lasso_single_node

logger = logging.getLogger(__name__)


def run_geometry_protein_enrichment(config: PipelineConfig) -> None:
    """Stage 6b entry point."""
    enrichment_dir = config.geometry_enrichment_dir

    # Load geometry features from Stage 6a
    if not config.geometry_protein_features_path.exists():
        logger.warning("geometry_protein_features.npz not found. Run Stage 6a first.")
        return
    geom_data = np.load(config.geometry_protein_features_path, allow_pickle=True)
    geom_accessions = list(geom_data["accessions"])
    geom_matrix = geom_data["geometry_matrix"]  # (N_geom, 55)
    geom_names = list(geom_data["feature_names"])
    logger.info("Loaded geometry for %d proteins (%d features)", len(geom_accessions), len(geom_names))

    # Load pipeline state for accession-to-index mapping
    if not config.pipeline_state_path.exists():
        logger.warning("pipeline_state.json not found.")
        return
    state = json.loads(config.pipeline_state_path.read_text())
    acc_to_idx: dict[str, int] = state.get("accession_index", {})

    # Load feature max activations
    if not config.feature_max_path.exists():
        logger.warning("feature_max_activations.npy not found.")
        return
    feature_maxes = np.load(config.feature_max_path)
    n_features = len(feature_maxes)

    # Memory-mapped per-protein max activations
    n_proteins_total = len(acc_to_idx)
    if not config.protein_feature_maxes_path.exists():
        logger.warning("protein_feature_maxes.npy not found.")
        return
    act_memmap = np.array(np.memmap(
        config.protein_feature_maxes_path,
        dtype=np.float32, mode="r",
        shape=(n_proteins_total, n_features),
    ))
    logger.info("Loaded protein max activations into RAM: %s", act_memmap.shape)

    # Map geometry accessions to memmap rows
    valid_geom_indices = []
    memmap_rows = []
    for gi, acc in enumerate(geom_accessions):
        if acc in acc_to_idx:
            valid_geom_indices.append(gi)
            memmap_rows.append(acc_to_idx[acc])

    if not valid_geom_indices:
        logger.warning("No overlap between geometry proteins and surveyed proteins.")
        return

    valid_geom_indices = np.array(valid_geom_indices)
    memmap_rows = np.array(memmap_rows)
    X_geom = geom_matrix[valid_geom_indices]
    geom_valid = np.all(np.isfinite(X_geom), axis=1)

    logger.info(
        "%d proteins with both geometry and activations (%d finite)",
        len(valid_geom_indices), int(geom_valid.sum()),
    )

    # Build set of already-done nodes + cache existing JSONs (one glob, not 5120 stat calls)
    done_nodes: set[int] = set()
    existing_jsons: dict[int, dict] = {}
    for feat_path in enrichment_dir.glob("????.json"):
        try:
            feat_json = json.loads(feat_path.read_text())
            fid = feat_json.get("feature_id")
            if fid is not None:
                existing_jsons[fid] = feat_json
                if "geometric_protein_level" in feat_json:
                    done_nodes.add(fid)
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    logger.info("Found %d already-completed nodes", len(done_nodes))

    # Per-node loop
    n_fitted = 0
    n_skipped = len(done_nodes)

    nodes_to_process = [
        ni for ni in range(n_features)
        if ni not in done_nodes and feature_maxes[ni] > 0
    ]
    logger.info("%d nodes to process (%d dead/done skipped)", len(nodes_to_process), n_features - len(nodes_to_process))

    for i, ni in enumerate(nodes_to_process):
        if (i + 1) % 500 == 0:
            logger.info("Progress: %d/%d (fitted=%d)", i + 1, len(nodes_to_process), n_fitted)

        # Get activations for proteins with geometry
        y_all = act_memmap[memmap_rows, ni]
        active_mask = (y_all > 0) & geom_valid
        n_active = int(active_mask.sum())

        if n_active < config.geometry_min_active_proteins:
            continue

        X = X_geom[active_mask]
        y = y_all[active_mask]

        result = fit_lasso_single_node(X, y, geom_names, cv_folds=config.geometry_lasso_cv_folds)
        if result is None:
            continue

        # Write per-feature JSON (reuse cached existing JSON if available)
        feat_json = existing_jsons.get(ni, {})
        feat_json["feature_id"] = ni
        feat_json["feature_max_activation"] = float(feature_maxes[ni])
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
        feat_path = enrichment_dir / f"{ni:04d}.json"
        feat_path.write_text(json.dumps(feat_json, indent=2))
        n_fitted += 1

    # Build summary from all completed JSONs
    summary_features: dict[str, dict] = {}
    for feat_path in sorted(enrichment_dir.glob("????.json")):
        try:
            feat_json = json.loads(feat_path.read_text())
            if "geometric_protein_level" in feat_json:
                fid = str(feat_json["feature_id"])
                pl = feat_json["geometric_protein_level"]
                summary_features[fid] = {
                    "protein_r2_cv": pl["r2_cv"],
                    "protein_pearson_r": pl["pearson_r"],
                }
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    summary = {
        "n_features_protein_level": len(summary_features),
        "n_proteins_with_geometry": len(geom_accessions),
        "features": summary_features,
    }
    (enrichment_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("Done. Fitted Lasso for %d nodes (%d skipped as already done)", n_fitted, n_skipped)
