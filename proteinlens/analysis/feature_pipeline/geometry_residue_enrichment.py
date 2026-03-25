"""Stage 6c: Residue-level GBM geometry enrichment + precomputed plot data.

For each SAE node with enough activated residue positions, this stage:

1. Builds ``protein_data`` dicts by joining per-residue SAE activations with
   geometry residue profiles.
2. Collects activated/background Ca fragments.
3. Kabsch-aligns top fragments to build a motif template.
4. Trains a GBM classifier (activated vs background from local geometry).
5. Computes concordance metrics (SAE activation vs geometry prediction).
6. Precomputes all plot data for the top-N proteins per node so the
   visualiser frontend requires no model inference at serving time.

Outputs are merged into the per-feature JSON files created by Stage 6b
(``geometry_enrichment/{feat:04d}.json``) and ``summary.json`` is updated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.geometry.classifiers import (
    collect_node_fragments,
    compute_concordance_metrics,
    superpose_fragments,
    train_motif_classifier,
)
from proteinlens.analysis.geometry.residue_features import (
    ACTIVE_GEOM_NAMES,
    extract_local_feature_vector,
    select_features,
)

logger = logging.getLogger(__name__)


def _mean_structure_to_pdb(mean_structure: np.ndarray) -> str:
    """Convert a mean Ca structure array to minimal PDB-format text.

    Parameters
    ----------
    mean_structure : np.ndarray
        Shape ``(W, 3)`` Ca coordinates of the motif template.

    Returns
    -------
    str
        PDB-format text with REMARK + ATOM records.
    """
    lines = ["REMARK  Motif template (mean Ca structure from Kabsch alignment)"]
    for i, (x, y, z) in enumerate(mean_structure):
        # Standard PDB ATOM record format
        lines.append(
            f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
        )
    lines.append("END")
    return "\n".join(lines)


def _preload_all_protein_data(
    config: PipelineConfig,
    acc_to_idx: dict[str, int],
) -> dict[str, dict]:
    """Pre-load all protein data that has both geometry profiles and activations.

    Loads all eligible proteins once into memory so that per-node filtering
    is a fast dict lookup instead of repeated disk I/O. This trades memory
    for a ~50x speedup when processing thousands of SAE nodes.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    acc_to_idx : dict[str, int]
        Accession -> memmap row index mapping.

    Returns
    -------
    dict[str, dict]
        Mapping of accession -> protein data dict with keys: ``accession``,
        ``act_matrix``, ``ca``, ``profiles``, ``n_residues``, ``sequence``,
        ``memmap_row`` (int index into the activation memmap).
    """
    profiles_dir = config.geometry_residue_profiles_dir
    residue_act_dir = config.residue_activations_dir
    interpro_act_dir = config.interpro_residue_activations_dir

    all_data: dict[str, dict] = {}
    n_loaded = 0
    n_skipped = 0

    for acc, row_idx in acc_to_idx.items():
        # Check if geometry residue profile exists
        geom_path = profiles_dir / f"{acc}.npz"
        if not geom_path.exists():
            continue

        # Check if per-residue activations exist (in either directory)
        act_path = residue_act_dir / f"{acc}.npz"
        if not act_path.exists():
            act_path = interpro_act_dir / f"{acc}.npz"
            if not act_path.exists():
                continue

        # Load both files
        try:
            geom_data = np.load(geom_path, allow_pickle=True)
            act_data = np.load(act_path, allow_pickle=True)
        except Exception:
            n_skipped += 1
            continue

        ca = geom_data["ca"]
        act_matrix = act_data["activations"]  # (seq_len, n_features)

        # Align lengths (ESM may include special tokens)
        n = min(len(ca), act_matrix.shape[0])
        if n < 20:
            n_skipped += 1
            continue

        ca = ca[:n]
        act_matrix = act_matrix[:n]

        # Reconstruct profiles dict from stored arrays
        profiles = {
            "curvature": geom_data["curvature"][:n],
            "torsion": geom_data["torsion"][:n],
            "planarity": geom_data["planarity"][:n],
            "tangents": geom_data["tangents"][:n],
            "helix_mask": geom_data["helix_mask"][:n],
            "categories": geom_data["categories"][:n],
        }

        seq_arr = geom_data.get("sequence", np.array([""]))
        seq = str(seq_arr[0]) if len(seq_arr) > 0 else ""

        all_data[acc] = {
            "accession": acc,
            "act_matrix": act_matrix,
            "ca": ca,
            "profiles": profiles,
            "n_residues": n,
            "sequence": seq,
            "memmap_row": row_idx,
        }
        n_loaded += 1

    logger.info(
        "Pre-loaded %d proteins with geometry + activations (%d skipped)",
        n_loaded, n_skipped,
    )
    return all_data


def _filter_proteins_for_node(
    all_protein_data: dict[str, dict],
    node_idx: int,
    act_memmap: np.ndarray,
) -> list[dict]:
    """Filter pre-loaded protein data to those where a given node fires.

    Parameters
    ----------
    all_protein_data : dict[str, dict]
        Pre-loaded protein data from :func:`_preload_all_protein_data`.
    node_idx : int
        SAE node index.
    act_memmap : np.ndarray
        Per-protein max activation memmap.

    Returns
    -------
    list[dict]
        Protein data dicts where the node fires (max activation > 0).
    """
    return [
        pdata for pdata in all_protein_data.values()
        if act_memmap[pdata["memmap_row"], node_idx] > 0
    ]


def _precompute_plot_data(
    protein_data: list[dict],
    node_idx: int,
    tree,
    threshold: float,
    geom_threshold: float,
    half_w: int,
    top_n: int,
    feature_importances: dict[str, float],
) -> list[dict]:
    """Precompute all plotting arrays for the top-N proteins.

    For each protein (ranked by max activation on this node), computes:
    - Ca backbone coordinates
    - Per-residue SAE activation profile
    - Per-residue GBM probability profile
    - Activated positions with their activation values
    - Top-2 feature traces (by importance)
    - Per-residue concordance labels

    Parameters
    ----------
    protein_data : list[dict]
        Protein data dicts with per-residue activations and geometry.
    node_idx : int
        SAE node index.
    tree : sklearn classifier
        Trained GBM with ``predict_proba``.
    threshold : float
        SAE activation threshold.
    geom_threshold : float
        GBM probability threshold for "geometry predicts active".
    half_w : int
        Half-window for feature extraction.
    top_n : int
        Number of top proteins to process.
    feature_importances : dict[str, float]
        Feature name -> importance from GBM.

    Returns
    -------
    list[dict]
        Plot data for each protein, ready for JSON serialisation.
    """
    if tree is None:
        return []

    # Rank proteins by max activation on this node (descending)
    ranked = sorted(
        protein_data,
        key=lambda p: float(p["act_matrix"][:, node_idx].max()),
        reverse=True,
    )[:top_n]

    # Identify top-2 feature names by importance
    top_feat_names = sorted(
        feature_importances.keys(),
        key=lambda k: feature_importances[k],
        reverse=True,
    )[:2]

    # Map feature names to their indices in the active feature set
    top_feat_indices: dict[str, int] = {}
    for name in top_feat_names:
        if name in ACTIVE_GEOM_NAMES:
            top_feat_indices[name] = ACTIVE_GEOM_NAMES.index(name)

    results: list[dict] = []
    for pdata in ranked:
        n = pdata["n_residues"]
        col = pdata["act_matrix"][:, node_idx]
        ca = pdata["ca"]
        profiles = pdata["profiles"]
        seq = pdata.get("sequence", "")

        # Per-residue arrays
        sae_profile = [float(col[i]) for i in range(n)]
        geom_prob_profile: list[float] = []
        concordance_labels: list[str] = []
        top_traces: dict[str, list[float | None]] = {
            name: [] for name in top_feat_names
        }
        activated_positions: list[dict] = []

        for pos in range(n):
            sae_val = float(col[pos])

            # Compute GBM probability if within valid window
            if pos >= half_w and pos < n - half_w:
                feat_vec = extract_local_feature_vector(
                    profiles, ca, pos, half_w, sequence=seq,
                )
                if feat_vec is not None and np.all(np.isfinite(feat_vec)):
                    fv = select_features(feat_vec)
                    prob = tree.predict_proba(fv.reshape(1, -1))[0]
                    geom_prob = float(prob[1] if len(prob) > 1 else prob[0])

                    # Extract top feature values from the active feature subset
                    for name in top_feat_names:
                        if name in top_feat_indices:
                            idx = top_feat_indices[name]
                            top_traces[name].append(float(fv[idx]))
                        else:
                            top_traces[name].append(None)
                else:
                    geom_prob = 0.0
                    for name in top_feat_names:
                        top_traces[name].append(None)
            else:
                # Outside valid window range
                geom_prob = 0.0
                for name in top_feat_names:
                    top_traces[name].append(None)

            geom_prob_profile.append(geom_prob)

            # Concordance label
            sae_active = sae_val >= threshold
            geom_active = geom_prob >= geom_threshold
            if sae_active and geom_active:
                label = "agree"
            elif not sae_active and geom_active:
                label = "fp"
            elif sae_active and not geom_active:
                label = "fn"
            else:
                label = "tn"
            concordance_labels.append(label)

            # Track activated positions
            if sae_val >= threshold:
                activated_positions.append({
                    "position": pos,
                    "activation": sae_val,
                })

        results.append({
            "accession": pdata["accession"],
            "sequence": seq[:n],
            "ca_backbone": [[float(x), float(y), float(z)] for x, y, z in ca],
            "sae_activation_profile": sae_profile,
            "geom_prob_profile": geom_prob_profile,
            "activated_positions": activated_positions,
            "top_feature_traces": top_traces,
            "concordance_labels": concordance_labels,
        })

    return results


def run_geometry_residue_enrichment(config: PipelineConfig) -> None:
    """Stage 6c entry point: residue-level GBM + concordance + plot data.

    For each SAE node, trains a GBM classifier on local geometry features
    to predict where the node fires, computes concordance metrics, and
    precomputes all plot data for the visualiser frontend.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration. Requires outputs from Stage 6a (geometry
        profiles) and the survey/collection stages (per-residue activations).
    """
    enrichment_dir = config.geometry_enrichment_dir

    # -- Load pipeline state --
    if not config.pipeline_state_path.exists():
        logger.warning("pipeline_state.json not found. Run survey stage first.")
        return
    state = json.loads(config.pipeline_state_path.read_text())
    acc_to_idx: dict[str, int] = state.get("accession_to_index", {})

    # -- Load feature max activations --
    if not config.feature_max_path.exists():
        logger.warning("feature_max_activations.npy not found.")
        return
    feature_maxes = np.load(config.feature_max_path)
    n_features = len(feature_maxes)

    # -- Load protein-level max activations memmap --
    n_proteins_total = len(acc_to_idx)
    if not config.protein_feature_maxes_path.exists():
        logger.warning("protein_feature_maxes.npy not found.")
        return
    act_memmap = np.memmap(
        config.protein_feature_maxes_path,
        dtype=np.float32,
        mode="r",
        shape=(n_proteins_total, n_features),
    )

    half_w = config.geometry_fragment_half_w

    # -- Pre-load all protein data once (geometry + activations) --
    # This avoids re-reading .npz files from disk for every SAE node.
    all_protein_data = _preload_all_protein_data(config, acc_to_idx)

    if not all_protein_data:
        logger.warning("No proteins with both geometry and activations found.")
        return

    # -- Load existing summary for merging --
    summary_path = enrichment_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {"features": {}}

    n_fitted = 0
    n_skipped_inactive = 0
    n_skipped_few = 0

    for ni in range(n_features):
        if ni % 200 == 0:
            logger.info(
                "Node %d/%d (fitted=%d, skipped=%d)",
                ni, n_features, n_fitted, n_skipped_inactive + n_skipped_few,
            )

        # Skip dead features
        if feature_maxes[ni] == 0:
            n_skipped_inactive += 1
            continue

        # Filter pre-loaded data to proteins where this node fires
        protein_data = _filter_proteins_for_node(
            all_protein_data, ni, act_memmap
        )

        if not protein_data:
            n_skipped_few += 1
            continue

        # Count total activated positions across all proteins
        total_activated = 0
        for pdata in protein_data:
            col = pdata["act_matrix"][:, ni]
            total_activated += int(np.sum(col > 0))

        if total_activated < config.geometry_min_activated_positions:
            n_skipped_few += 1
            continue

        # -- Collect fragments --
        frag_result = collect_node_fragments(
            protein_data, ni, half_w=half_w,
            act_quantile=config.geometry_act_quantile,
            max_fragments=config.geometry_frag_top_k,
            bg_ratio=config.geometry_bg_ratio,
        )

        activated = frag_result["activated"]
        background = frag_result["background"]
        threshold = frag_result["threshold"]

        if len(activated) < 20 or len(background) < 20:
            n_skipped_few += 1
            continue

        # -- Superpose fragments --
        sup_result = superpose_fragments(
            activated, top_k=config.geometry_frag_top_k
        )

        # -- Train classifier --
        clf_result = train_motif_classifier(
            activated, background,
            feature_names=list(ACTIVE_GEOM_NAMES),
            cv_folds=config.geometry_classifier_cv_folds,
        )

        # -- Compute concordance metrics --
        geom_threshold = clf_result["optimal_threshold"]
        concordance = compute_concordance_metrics(
            protein_data, ni, clf_result["tree"],
            threshold, geom_threshold, half_w,
        )

        # -- Precompute plot data for top proteins --
        plot_proteins = _precompute_plot_data(
            protein_data, ni, clf_result["tree"],
            threshold, geom_threshold, half_w,
            top_n=config.geometry_top_proteins_for_plots,
            feature_importances=clf_result["feature_importances"],
        )

        # -- Build motif superposition PDB text --
        motif_pdb = ""
        if sup_result["mean_structure"] is not None:
            motif_pdb = _mean_structure_to_pdb(sup_result["mean_structure"])

        # -- Merge into per-feature JSON --
        feat_path = enrichment_dir / f"{ni:04d}.json"
        if feat_path.exists():
            feat_json = json.loads(feat_path.read_text())
        else:
            feat_json = {
                "feature_id": ni,
                "feature_max_activation": float(feature_maxes[ni]),
            }

        feat_json["geometric_residue_level"] = {
            "tree_f1_cv": clf_result["f1_cv"],
            "gbm_auc_cv": clf_result["gbm_auc_cv"],
            "rf_auc_cv": clf_result["rf_auc_cv"],
            "lpo_auc": clf_result["lpo_auc"],
            "rules": clf_result["rules"],
            "optimal_threshold": clf_result["optimal_threshold"],
            "activation_threshold": threshold,
            "n_activated": len(activated),
            "n_background": len(background),
            "n_unique_proteins": clf_result.get("n_unique_proteins", 0),
            "feature_importances": clf_result["feature_importances"],
            "concordance": concordance,
            "motif_superposition": {
                "mean_rmsd": sup_result["mean_rmsd"],
                "std_rmsd": sup_result["std_rmsd"],
                "n_fragments": sup_result["n_fragments"],
                "per_position_flexibility": (
                    sup_result["per_pos_std"].tolist()
                    if sup_result["per_pos_std"] is not None
                    else []
                ),
                "mean_structure_pdb": motif_pdb,
            },
        }

        feat_json["plot_data"] = {
            "top_proteins": plot_proteins,
        }

        feat_path.write_text(json.dumps(feat_json, indent=2))

        # -- Update summary --
        feat_key = str(ni)
        if feat_key not in summary["features"]:
            summary["features"][feat_key] = {}
        summary["features"][feat_key].update({
            "residue_gbm_auc_cv": clf_result["gbm_auc_cv"],
            "residue_concordance_spearman": concordance.get("spearman_r", 0.0),
            "motif_rmsd": sup_result["mean_rmsd"],
        })

        n_fitted += 1

    # -- Finalise summary --
    summary["n_features_residue_level"] = n_fitted
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info(
        "Done. Fitted residue-level models for %d nodes (skipped: %d inactive, %d too few data)",
        n_fitted, n_skipped_inactive, n_skipped_few,
    )
