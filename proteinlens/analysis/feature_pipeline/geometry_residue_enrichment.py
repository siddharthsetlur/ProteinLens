"""Stage 6c: Residue-level GBM geometry enrichment + precomputed plot data.

Scales to 50k+ proteins on networked filesystems (cephfs) by:
- Globbing directories once instead of per-file stat calls
- Preloading geometry profiles at startup (~500MB for 50k proteins)
- Loading the memmap into RAM once (~1GB)
- Per-node resumability via a set built at startup
- Parallel node processing via multiprocessing (fork, COW shared data)
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

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

# ── Module-level shared state (set before fork, read-only in workers) ────
_shared: dict = {}


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
    feat_cache: dict[tuple[str, int], np.ndarray] | None = None,
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
        geom_prob_profile = [0.0] * n
        top_traces: dict[str, list[float | None]] = {name: [None] * n for name in top_feat_names}
        activated_positions: list[dict] = []

        # Batch: collect feature vectors for valid interior positions
        acc = pdata["accession"]
        batch_fvs: list[np.ndarray] = []
        batch_positions: list[int] = []
        for pos in range(half_w, n - half_w):
            feat_vec = None
            if feat_cache is not None:
                feat_vec = feat_cache.get((acc, pos))
            if feat_vec is None:
                feat_vec = extract_local_feature_vector(profiles, ca, pos, half_w, sequence=seq)
            if feat_vec is not None and np.all(np.isfinite(feat_vec)):
                fv = select_features(feat_vec)
                batch_fvs.append(fv)
                batch_positions.append(pos)
                for name in top_feat_names:
                    if name in top_feat_indices:
                        top_traces[name][pos] = float(fv[top_feat_indices[name]])

        # Single batched predict_proba call per protein
        if batch_fvs:
            X_batch = np.array(batch_fvs)
            probs_batch = tree.predict_proba(X_batch)
            for j, pos in enumerate(batch_positions):
                p = probs_batch[j]
                geom_prob_profile[pos] = float(p[1] if len(p) > 1 else p[0])

        # Build concordance labels and activated positions
        concordance_labels: list[str] = []
        for pos in range(n):
            sae_val = float(col[pos])
            geom_prob = geom_prob_profile[pos]
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
            "top_feature_traces": {k: list(v) for k, v in top_traces.items()},
            "concordance_labels": concordance_labels,
        })

    return results


# ── Per-node worker (module-level so it's picklable for mp) ──────────────

def _process_node(ni: int) -> tuple[int, str]:
    """Process a single SAE node. Returns (node_idx, 'fitted'|'skipped')."""
    s = _shared
    act_matrix_full = s["act_matrix_full"]
    geom_cache = s["geom_cache"]
    row_to_acc = s["row_to_acc"]
    available = s["available"]
    feature_maxes = s["feature_maxes"]
    half_w = s["half_w"]
    enrichment_dir = s["enrichment_dir"]
    cfg = s["config_params"]

    # Per-process activation LRU cache (initialised on first call per process)
    if not hasattr(_process_node, "_act_cache"):
        _process_node._act_cache = {}
        _process_node._act_order = []

    _act_cache = _process_node._act_cache
    _act_order = _process_node._act_order
    _ACT_CACHE_MAX = 500  # smaller per-process to limit total memory

    def _load_activations(acc: str) -> np.ndarray | None:
        if acc in _act_cache:
            return _act_cache[acc]
        path = available.get(acc)
        if path is None:
            return None
        try:
            arr = np.load(path, allow_pickle=True)["activations"]
        except Exception:
            return None
        _act_cache[acc] = arr
        _act_order.append(acc)
        if len(_act_order) > _ACT_CACHE_MAX:
            evict = _act_order.pop(0)
            _act_cache.pop(evict, None)
        return arr

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
        g = geom_cache.get(acc)
        if g is None:
            continue
        act_mat = _load_activations(acc)
        if act_mat is None:
            continue

        ca = g["ca"]
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
        return (ni, "skipped")

    total_activated = sum(int(np.sum(p["act_matrix"][:, ni] > 0)) for p in protein_data)
    if total_activated < cfg["geometry_min_activated_positions"]:
        return (ni, "skipped")

    frag_result = collect_node_fragments(
        protein_data, ni, half_w=half_w,
        act_quantile=cfg["geometry_act_quantile"],
        max_fragments=cfg["geometry_frag_top_k"],
        bg_ratio=cfg["geometry_bg_ratio"],
    )
    activated = frag_result["activated"]
    background = frag_result["background"]
    threshold = frag_result["threshold"]

    if len(activated) < 20 or len(background) < 20:
        return (ni, "skipped")

    # Build feature vector cache from fragments
    feat_cache: dict[tuple[str, int], np.ndarray] = {}
    for frag in activated + background:
        if frag["features"] is not None:
            feat_cache[(frag["accession"], frag["position"])] = frag["features"]

    sup_result = superpose_fragments(activated, top_k=cfg["geometry_frag_top_k"])
    clf_result = train_motif_classifier(
        activated, background,
        feature_names=list(ACTIVE_GEOM_NAMES),
        cv_folds=cfg["geometry_classifier_cv_folds"],
    )
    geom_threshold = clf_result["optimal_threshold"]
    concordance = compute_concordance_metrics(
        protein_data, ni, clf_result["tree"],
        threshold, geom_threshold, half_w,
        feat_cache=feat_cache,
    )
    plot_proteins = _precompute_plot_data(
        protein_data, ni, clf_result["tree"],
        threshold, geom_threshold, half_w,
        top_n=cfg["geometry_top_proteins_for_plots"],
        feature_importances=clf_result["feature_importances"],
        feat_cache=feat_cache,
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
    return (ni, "fitted")


# ── Main entry point ─────────────────────────────────────────────────────

def run_geometry_residue_enrichment(config: PipelineConfig) -> None:
    """Stage 6c entry point."""
    global _shared

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

    # Load FULL memmap into RAM (50k*5120*4 = ~1GB)
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

    # Build available accessions index via glob
    geom_files = {p.stem: p for p in config.geometry_residue_profiles_dir.glob("*.npz")}
    act_files = {p.stem: p for p in config.residue_activations_dir.glob("*.npz")}
    interpro_files = {p.stem: p for p in config.interpro_residue_activations_dir.glob("*.npz")}

    available: dict[str, Path] = {}
    for acc in acc_to_idx:
        if acc not in geom_files:
            continue
        if acc in act_files:
            available[acc] = act_files[acc]
        elif acc in interpro_files:
            available[acc] = interpro_files[acc]

    logger.info("Found %d proteins with both geometry profiles and activations", len(available))
    if not available:
        logger.warning("No proteins with both geometry and activations found.")
        return

    # Preload ALL geometry profiles into RAM (~500MB)
    def _load_geom(acc: str) -> tuple[str, dict | None]:
        try:
            g = np.load(geom_files[acc], allow_pickle=True)
            return acc, {
                "ca": np.array(g["ca"]),
                "curvature": np.array(g["curvature"]),
                "torsion": np.array(g["torsion"]),
                "planarity": np.array(g["planarity"]),
                "tangents": np.array(g["tangents"]),
                "helix_mask": np.array(g["helix_mask"]),
                "categories": np.array(g["categories"]),
                "sequence": np.array(g.get("sequence", np.array([""]))),
            }
        except Exception:
            return acc, None

    geom_cache: dict[str, dict] = {}
    logger.info("Preloading %d geometry profiles (threaded)...", len(available))
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, (acc, data) in enumerate(pool.map(_load_geom, available)):
            if data is not None:
                geom_cache[acc] = data
            if (i + 1) % 5000 == 0:
                logger.info("Preloaded %d/%d geometry profiles", i + 1, len(available))
    logger.info("Preloaded %d geometry profiles into RAM", len(geom_cache))

    # Reverse map: memmap row -> accession (only available proteins)
    row_to_acc = {v: k for k, v in acc_to_idx.items() if k in available}

    # Build set of already-done nodes (one glob + scan)
    done_nodes: set[int] = set()
    for feat_path in enrichment_dir.glob("????.json"):
        try:
            feat_json = json.loads(feat_path.read_text())
            if "geometric_residue_level" in feat_json:
                done_nodes.add(feat_json["feature_id"])
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    logger.info("Found %d already-completed nodes", len(done_nodes))

    nodes_to_process = [
        ni for ni in range(n_features)
        if ni not in done_nodes and feature_maxes[ni] > 0
    ]
    logger.info("%d nodes to process (%d dead/done skipped)", len(nodes_to_process), n_features - len(nodes_to_process))

    # ── Set shared state for workers (before fork) ───────────────────
    _shared.update({
        "act_matrix_full": act_matrix_full,
        "geom_cache": geom_cache,
        "row_to_acc": row_to_acc,
        "available": available,
        "feature_maxes": feature_maxes,
        "half_w": half_w,
        "enrichment_dir": enrichment_dir,
        "config_params": {
            "geometry_min_activated_positions": config.geometry_min_activated_positions,
            "geometry_act_quantile": config.geometry_act_quantile,
            "geometry_frag_top_k": config.geometry_frag_top_k,
            "geometry_bg_ratio": config.geometry_bg_ratio,
            "geometry_classifier_cv_folds": config.geometry_classifier_cv_folds,
            "geometry_top_proteins_for_plots": config.geometry_top_proteins_for_plots,
        },
    })

    # ── Parallel execution ───────────────────────────────────────────
    n_workers = min(int(os.environ.get("PIPELINE_WORKERS", "1")), len(nodes_to_process) or 1)
    n_fitted = 0
    n_skipped_few = 0

    pbar = tqdm(total=len(nodes_to_process), desc="Geometry residue enrichment")

    if n_workers > 1:
        logger.info("Processing %d nodes with %d parallel workers", len(nodes_to_process), n_workers)
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            for ni, result in pool.imap_unordered(_process_node, nodes_to_process):
                if result == "fitted":
                    n_fitted += 1
                else:
                    n_skipped_few += 1
                pbar.update(1)
    else:
        logger.info("Processing %d nodes serially (set PIPELINE_WORKERS=N for parallel)", len(nodes_to_process))
        for ni in nodes_to_process:
            try:
                _, result = _process_node(ni)
                if result == "fitted":
                    n_fitted += 1
                else:
                    n_skipped_few += 1
            except Exception:
                logger.exception("Error processing node %d", ni)
            pbar.update(1)

    pbar.close()

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
