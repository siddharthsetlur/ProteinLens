"""Stage 6c: Residue-level GBM geometry enrichment + precomputed plot data.

Scales to 50k+ proteins on networked filesystems (cephfs) by:
- Globbing directories once instead of per-file stat calls
- Preloading geometry profiles at startup (~2GB for 50k proteins)
- Loading the memmap into RAM once (~1GB)
- Only loading per-residue activations per-node (unavoidable)
- Per-node resumability via a set built at startup
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
    """Convert a mean Ca structure array to minimal PDB-format text."""
    lines = ["REMARK  Motif template (mean Ca structure from Kabsch alignment)"]
    for i, (x, y, z) in enumerate(mean_structure):
        lines.append(
            f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
        )
    lines.append("END")
    return "\n".join(lines)


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
    """Precompute plotting arrays for the top-N proteins on this node."""
    if tree is None:
        return []

    ranked = sorted(
        protein_data,
        key=lambda p: float(p["act_matrix"][:, node_idx].max()),
        reverse=True,
    )[:top_n]

    top_feat_names = sorted(
        feature_importances.keys(),
        key=lambda k: feature_importances[k],
        reverse=True,
    )[:2]

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

        sae_profile = [float(col[i]) for i in range(n)]
        geom_prob_profile: list[float] = []
        concordance_labels: list[str] = []
        top_traces: dict[str, list[float | None]] = {name: [] for name in top_feat_names}
        activated_positions: list[dict] = []

        for pos in range(n):
            sae_val = float(col[pos])

            if half_w <= pos < n - half_w:
                feat_vec = extract_local_feature_vector(profiles, ca, pos, half_w, sequence=seq)
                if feat_vec is not None and np.all(np.isfinite(feat_vec)):
                    fv = select_features(feat_vec)
                    prob = tree.predict_proba(fv.reshape(1, -1))[0]
                    geom_prob = float(prob[1] if len(prob) > 1 else prob[0])
                    for name in top_feat_names:
                        if name in top_feat_indices:
                            top_traces[name].append(float(fv[top_feat_indices[name]]))
                        else:
                            top_traces[name].append(None)
                else:
                    geom_prob = 0.0
                    for name in top_feat_names:
                        top_traces[name].append(None)
            else:
                geom_prob = 0.0
                for name in top_feat_names:
                    top_traces[name].append(None)

            geom_prob_profile.append(geom_prob)

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

            if sae_val >= threshold:
                activated_positions.append({"position": pos, "activation": sae_val})

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
    """Stage 6c entry point."""
    enrichment_dir = config.geometry_enrichment_dir

    # Load pipeline state
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

    # Load FULL memmap into RAM (50k*5120*4 = ~1GB) to avoid per-column page faults
    n_proteins_total = len(acc_to_idx)
    if not config.protein_feature_maxes_path.exists():
        logger.warning("protein_feature_maxes.npy not found.")
        return
    act_matrix_full = np.array(np.memmap(
        config.protein_feature_maxes_path,
        dtype=np.float32, mode="r",
        shape=(n_proteins_total, n_features),
    ))
    logger.info("Loaded protein max activations into RAM: %s", act_matrix_full.shape)

    half_w = config.geometry_fragment_half_w

    # Build available accessions index via glob (3 calls, not 150k stat calls)
    has_geom = {p.stem for p in config.geometry_residue_profiles_dir.glob("*.npz")}
    act_files = {p.stem: p for p in config.residue_activations_dir.glob("*.npz")}
    interpro_files = {p.stem: p for p in config.interpro_residue_activations_dir.glob("*.npz")}

    available: dict[str, Path] = {}
    for acc in acc_to_idx:
        if acc not in has_geom:
            continue
        if acc in act_files:
            available[acc] = act_files[acc]
        elif acc in interpro_files:
            available[acc] = interpro_files[acc]

    logger.info("Found %d proteins with both geometry profiles and activations", len(available))
    if not available:
        logger.warning("No proteins with both geometry and activations found.")
        return

    # Reverse map: memmap row -> accession (only available proteins)
    row_to_acc = {v: k for k, v in acc_to_idx.items() if k in available}

    # Build set of already-done nodes (one glob + scan, not 5120 stat calls)
    done_nodes: set[int] = set()
    for feat_path in enrichment_dir.glob("????.json"):
        try:
            feat_json = json.loads(feat_path.read_text())
            if "geometric_residue_level" in feat_json:
                done_nodes.add(feat_json["feature_id"])
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    logger.info("Found %d already-completed nodes", len(done_nodes))

    # --- Per-node worker (unchanged core logic) ---
    def process_node(ni: int) -> str:
        """Process a single node. Returns 'fitted', 'skipped', or 'error'."""
        node_col = act_matrix_full[:, ni]
        active_rows = np.where(node_col > 0)[0]

        if len(active_rows) > 500:
            top_idx = np.argsort(node_col[active_rows])[-500:]
            active_rows = active_rows[top_idx]

        protein_data = []
        for row_idx in active_rows:
            acc = row_to_acc.get(int(row_idx))
            if acc is None:
                continue
            geom_path = config.geometry_residue_profiles_dir / f"{acc}.npz"
            try:
                g = np.load(geom_path, allow_pickle=True)
                act_data = np.load(available[acc], allow_pickle=True)
            except Exception:
                continue

            ca = g["ca"]
            act_mat = act_data["activations"]
            n = min(len(ca), act_mat.shape[0])
            if n < 20:
                continue

            seq_arr = g.get("sequence", np.array([""]))
            protein_data.append({
                "accession": acc,
                "act_matrix": act_mat[:n],
                "ca": ca[:n],
                "profiles": {k: g[k][:n] for k in ("curvature", "torsion", "planarity", "tangents", "helix_mask", "categories")},
                "n_residues": n,
                "sequence": str(seq_arr[0]) if len(seq_arr) > 0 else "",
                "memmap_row": int(row_idx),
            })

        if not protein_data:
            return "skipped"

        total_activated = sum(int(np.sum(p["act_matrix"][:, ni] > 0)) for p in protein_data)
        if total_activated < config.geometry_min_activated_positions:
            return "skipped"

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
            return "skipped"

        sup_result = superpose_fragments(activated, top_k=config.geometry_frag_top_k)
        clf_result = train_motif_classifier(
            activated, background,
            feature_names=list(ACTIVE_GEOM_NAMES),
            cv_folds=config.geometry_classifier_cv_folds,
        )
        geom_threshold = clf_result["optimal_threshold"]
        concordance = compute_concordance_metrics(
            protein_data, ni, clf_result["tree"],
            threshold, geom_threshold, half_w,
        )
        plot_proteins = _precompute_plot_data(
            protein_data, ni, clf_result["tree"],
            threshold, geom_threshold, half_w,
            top_n=config.geometry_top_proteins_for_plots,
            feature_importances=clf_result["feature_importances"],
        )

        motif_pdb = ""
        if sup_result["mean_structure"] is not None:
            motif_pdb = _mean_structure_to_pdb(sup_result["mean_structure"])

        feat_path = enrichment_dir / f"{ni:04d}.json"
        if feat_path.exists():
            try:
                feat_json = json.loads(feat_path.read_text())
            except (json.JSONDecodeError, OSError):
                feat_json = {}
        else:
            feat_json = {}

        feat_json["feature_id"] = ni
        feat_json["feature_max_activation"] = float(feature_maxes[ni])
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
        feat_json["plot_data"] = {"top_proteins": plot_proteins}
        feat_path.write_text(json.dumps(feat_json, indent=2))
        return "fitted"

    # --- Build list of nodes to process ---
    nodes_to_process = [
        ni for ni in range(n_features)
        if ni not in done_nodes and feature_maxes[ni] > 0
    ]
    logger.info("%d nodes to process (%d dead/done skipped)", len(nodes_to_process), n_features - len(nodes_to_process))

    # --- Run serially ---
    n_fitted = 0
    n_skipped_few = 0

    for i, ni in enumerate(nodes_to_process):
        try:
            result = process_node(ni)
            if result == "fitted":
                n_fitted += 1
            else:
                n_skipped_few += 1
        except Exception:
            logger.exception("Error processing node %d", ni)
        if (i + 1) % 50 == 0:
            logger.info(
                "Progress: %d/%d done (fitted=%d, skipped=%d)",
                i + 1, len(nodes_to_process), n_fitted, n_skipped_few,
            )

    # Build summary from all completed JSONs
    summary_features: dict[str, dict] = {}
    for feat_path in sorted(enrichment_dir.glob("????.json")):
        try:
            feat_json = json.loads(feat_path.read_text())
            if "geometric_residue_level" in feat_json:
                fid = str(feat_json["feature_id"])
                rl = feat_json["geometric_residue_level"]
                entry: dict = {
                    "residue_gbm_auc_cv": rl.get("gbm_auc_cv", 0.0),
                    "residue_concordance_spearman": rl.get("concordance", {}).get("spearman_r", 0.0),
                    "motif_rmsd": rl.get("motif_superposition", {}).get("mean_rmsd", 0.0),
                }
                if "geometric_protein_level" in feat_json:
                    pl = feat_json["geometric_protein_level"]
                    entry["protein_r2_cv"] = pl.get("r2_cv", 0.0)
                    entry["protein_pearson_r"] = pl.get("pearson_r", 0.0)
                summary_features[fid] = entry
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    summary_path = enrichment_dir / "summary.json"
    if summary_path.exists():
        try:
            existing_summary = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing_summary = {}
    else:
        existing_summary = {}

    existing_summary["n_features_residue_level"] = len(summary_features)
    existing_summary["features"] = summary_features
    summary_path.write_text(json.dumps(existing_summary, indent=2))

    logger.info(
        "Done. Fitted residue-level models for %d nodes (%d skipped as done, %d too few data)",
        n_fitted, len(done_nodes), n_skipped_few,
    )
