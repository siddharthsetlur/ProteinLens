"""
Startup index builder for the SAE feature visualizer.

Reads per-feature summary files from the data directory and merges them into
a single in-memory feature index (list of dicts) for the homepage table.
Also computes pipeline status counts for the status badges.

Input files (all relative to data_dir):
  - survey_coverage.json: dict keyed by feature_id str -> coverage stats
  - feature_max_activations.npy: 1-D float32 array of shape (num_features,)
  - interpro_enrichment/summary.json (optional): per-feature best F1 scores
  - geometry_enrichment/summary.json (optional): per-feature R2/AUC scores

Output:
  - feature_index: list of dicts (one per feature), ready to serialize as JSON
  - pipeline_status: dict with stage counts and completion info
  - stats: merged dataset_stats.json + SAE config.yaml
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None if it doesn't exist or fails to parse."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def build_stats(data_dir: Path) -> dict[str, Any]:
    """
    Merge dataset_stats.json and the SAE config.yaml into a single stats dict.

    Returns a dict with two top-level keys: "dataset" and "sae".
    The SAE config is read from the path specified in dataset_stats.json["sae_dir"],
    resolved relative to the project root (two levels up from data_dir if needed).

    If config.yaml is not found, the "sae" key will contain only what we can
    derive from dataset_stats.json (num_features).
    """
    dataset_stats = load_json(data_dir / "dataset_stats.json") or {}

    # Try to find SAE config.yaml
    sae_config = {}
    sae_dir_rel = dataset_stats.get("sae_dir", "")
    if sae_dir_rel:
        # sae_dir is relative to the project root, try a few resolution strategies
        candidates = [
            data_dir / sae_dir_rel / "config.yaml",
            data_dir.parent / sae_dir_rel / "config.yaml",
            Path(sae_dir_rel) / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        raw = yaml.safe_load(f)
                    # Flatten the nested config structure for the frontend
                    trainer = raw.get("trainer_cfg", {})
                    wandb = raw.get("wandb_cfg", {})
                    sae_config = {
                        "dictionary_size": trainer.get("dictionary_size"),
                        "expansion_factor": trainer.get("expansion_factor"),
                        "activation_dim": trainer.get("activation_dim"),
                        "l1_penalty": trainer.get("l1_penalty"),
                        "lr": trainer.get("lr"),
                        "steps": trainer.get("steps"),
                        "wandb_name": wandb.get("wandb_name"),
                        "architecture": "ReLUSAE",
                    }
                    logger.info("Loaded SAE config from %s", candidate)
                    break
                except (yaml.YAMLError, OSError) as e:
                    logger.warning("Failed to parse %s: %s", candidate, e)

    return {"dataset": dataset_stats, "sae": sae_config}


def build_pipeline_status(data_dir: Path) -> dict[str, Any]:
    """
    Compute pipeline completion status from filesystem state.

    Returns:
      - completed_stages: list of stage names from pipeline_state.json
      - feature_count: number of per-feature JSON files
      - interpro_count: number of interpro enrichment JSONs (excluding summary)
      - geometry_count: number of geometry enrichment JSONs (excluding summary)
    """
    pipeline_state = load_json(data_dir / "pipeline_state.json") or {}
    completed_stages = pipeline_state.get("completed_stages", [])

    # Count per-feature files (exclude summary.json)
    def count_jsons(subdir: str) -> int:
        d = data_dir / subdir
        if not d.is_dir():
            return 0
        return sum(1 for f in d.iterdir() if f.suffix == ".json" and f.name != "summary.json")

    nmpfam_count = count_jsons("nmpfam/nmpfam_enrichment")

    return {
        "completed_stages": completed_stages,
        "feature_count": count_jsons("features"),
        "interpro_count": count_jsons("interpro_enrichment"),
        "geometry_count": count_jsons("geometry_enrichment"),
        "motif_count": count_jsons("motif_pwm_enrichment"),
        "position_count": count_jsons("position_enrichment"),
        "cath_count": count_jsons("cath_enrichment"),
        "nmpfam_count": nmpfam_count,
    }


def build_feature_index(data_dir: Path) -> list[dict[str, Any]]:
    """
    Build the feature index for the homepage AG Grid table.

    Merges data from multiple sources into one list of dicts (one per feature).
    Each dict has keys:
      - feature_id (int)
      - max_activation (float)
      - pct_proteins_activated (float or null)
      - pct_clusters_activated (float or null)
      - interpro_protein_best_f1 (float or null)
      - interpro_protein_best_name (str or null)
      - interpro_residue_best_f1 (float or null)
      - geometry_protein_r2_cv (float or null)
      - geometry_residue_pr_auc (float or null)

    Missing data is represented as None (serialized to JSON null).
    """
    # --- Load feature_max_activations.npy ---
    max_act_path = data_dir / "feature_max_activations.npy"
    if max_act_path.exists():
        max_activations = np.load(max_act_path).astype(float)
        num_features = len(max_activations)
    else:
        logger.warning("feature_max_activations.npy not found, using dataset_stats for num_features")
        ds = load_json(data_dir / "dataset_stats.json") or {}
        num_features = ds.get("num_features", 0)
        max_activations = np.zeros(num_features)

    # --- Load survey_coverage.json ---
    coverage_data = load_json(data_dir / "survey_coverage.json") or {}

    # --- Load interpro enrichment summary ---
    interpro_summary = load_json(data_dir / "interpro_enrichment" / "summary.json")
    interpro_features = {}
    if interpro_summary and interpro_summary.get("features"):
        interpro_features = interpro_summary["features"]
    else:
        # Fallback: scan per-feature files for best F1
        # PM NOTE: This fallback scans individual files which could be slow for 5120 features.
        # We only do this if summary.json is missing/empty, which means the pipeline
        # hasn't written the summary yet.
        interpro_features = _scan_interpro_files(data_dir / "interpro_enrichment")

    # --- Load MEME/PWM motif enrichment summary (Stage 7b) ---
    motif_summary = load_json(data_dir / "motif_pwm_enrichment" / "summary.json")
    motif_features = {}
    if motif_summary and motif_summary.get("features"):
        motif_features = motif_summary["features"]
    else:
        motif_features = _scan_motif_pwm_files(data_dir / "motif_pwm_enrichment")

    # --- Load position enrichment summary ---
    position_summary = load_json(data_dir / "position_enrichment" / "summary.json")
    position_features = {}
    if position_summary and position_summary.get("features"):
        position_features = position_summary["features"]
    else:
        position_features = _scan_position_files(data_dir / "position_enrichment")

    # --- Load geometry enrichment summary ---
    geometry_summary = load_json(data_dir / "geometry_enrichment" / "summary.json")
    geometry_features = {}
    if geometry_summary and geometry_summary.get("features"):
        # Check if the summary has the expected keys; if not, fall back to scanning
        sample = next(iter(geometry_summary["features"].values()), {})
        if "residue_pr_auc" in sample or "protein_r2_cv" in sample:
            geometry_features = geometry_summary["features"]
        else:
            # Summary exists but uses different keys — scan individual files
            geometry_features = _scan_geometry_files(data_dir / "geometry_enrichment")
    else:
        geometry_features = _scan_geometry_files(data_dir / "geometry_enrichment")

    # --- Load CATH enrichment ---
    cath_features = _scan_cath_files(data_dir / "cath_enrichment")

    # --- Load NMPFams hit counts ---
    nmpfam_hits = _scan_nmpfam_files(data_dir / "nmpfam" / "nmpfam_enrichment")

    # --- Load geometry radar data (category-aggregated feature importances) ---
    geometry_radar = _build_geometry_radar(data_dir / "geometry_enrichment")

    # --- Load geometry-primary analysis ---
    gp_data = load_json(data_dir / "geometry_primary_analysis.json")
    gp_features = gp_data.get("features", {}) if gp_data else {}

    # --- Merge into index rows ---
    index = []
    for fid in range(num_features):
        fid_str = str(fid)

        # Coverage
        cov = coverage_data.get(fid_str, {})

        # InterPro best scores
        # summary.json uses "top_*" keys; fallback scanner uses "protein_best_*" keys
        ipro = interpro_features.get(fid_str, {})

        # Position scores
        posn = position_features.get(fid_str, {})

        # Motif scores
        motif = motif_features.get(fid_str, {})

        # Geometry scores
        geom = geometry_features.get(fid_str, {})

        # CATH scores
        cath = cath_features.get(fid_str, {})

        # NMPFams hits
        nmpf = nmpfam_hits.get(fid_str, {})

        # Geometry-primary classification
        gp = gp_features.get(fid_str, {})

        row = {
            "feature_id": fid,
            "max_activation": round(float(max_activations[fid]), 6),
            "pct_proteins_activated": cov.get("pct_proteins_activated"),
            "pct_clusters_activated": cov.get("pct_clusters_activated"),
            "interpro_protein_best_f1": ipro.get("top_protein_f1") or ipro.get("protein_best_f1"),
            "interpro_protein_best_name": ipro.get("top_protein_annotation_name") or ipro.get("protein_best_name"),
            "interpro_residue_best_f1": ipro.get("top_residue_f1") or ipro.get("residue_best_f1"),
            "motif_best_pr_auc": motif.get("best_pr_auc"),
            "motif_best_consensus": motif.get("best_consensus"),
            "position_best_f1": posn.get("best_position_f1"),
            "position_best_name": posn.get("best_position"),
            "cath_best_f1": cath.get("best_cath_f1"),
            "cath_best_label": cath.get("best_cath_label"),
            "geometry_protein_r2_cv": geom.get("protein_r2_cv"),
            "geometry_residue_pr_auc": geom.get("residue_pr_auc"),
            "is_geometry_primary": gp.get("is_geometry_primary"),
            "geometry_primary_score": gp.get("composite_score"),
            "geometry_radar": geometry_radar.get(fid_str),
            "n_nmpfam_hits": nmpf.get("n_nmpfam_hits"),
        }
        index.append(row)

    logger.info(
        "Built feature index: %d features, %d with interpro, %d with geometry, %d with motif, %d with position, %d with cath",
        num_features,
        sum(1 for r in index if r["interpro_protein_best_f1"] is not None),
        sum(1 for r in index if r["geometry_protein_r2_cv"] is not None),
        sum(1 for r in index if r["motif_best_pr_auc"] is not None),
        sum(1 for r in index if r["position_best_f1"] is not None),
        sum(1 for r in index if r["cath_best_f1"] is not None),
    )
    return index


def _scan_interpro_files(interpro_dir: Path) -> dict[str, dict]:
    """
    Fallback: scan individual interpro enrichment JSONs to extract best F1 scores.

    Each file has protein_level (list of annotation results) and residue_level.
    We pick the annotation with the highest F1 from each.

    Returns dict keyed by feature_id str -> {protein_best_f1, protein_best_name, residue_best_f1}.
    """
    if not interpro_dir.is_dir():
        return {}

    result = {}
    for fpath in sorted(interpro_dir.iterdir()):
        if fpath.name == "summary.json" or fpath.suffix != ".json":
            continue
        data = load_json(fpath)
        if not data:
            continue

        fid_str = str(data.get("feature_id", fpath.stem.lstrip("0") or "0"))
        entry = {}

        # Best protein-level annotation by F1 — pipeline field is "best_f1"
        protein_level = data.get("protein_level", [])
        if protein_level:
            best = max(protein_level, key=lambda x: x.get("best_f1", 0))
            if best.get("best_f1", 0) > 0:
                entry["protein_best_f1"] = best["best_f1"]
                entry["protein_best_name"] = best.get("annotation_name", "")

        # Best residue-level annotation by F1 — pipeline field is "best_f1"
        residue_level = data.get("residue_level", [])
        if residue_level:
            best_res = max(residue_level, key=lambda x: x.get("best_f1", 0))
            if best_res.get("best_f1", 0) > 0:
                entry["residue_best_f1"] = best_res["best_f1"]

        if entry:
            result[fid_str] = entry

    if result:
        logger.info("Scanned %d interpro files, found %d with enrichment", len(result), len(result))
    return result


def _scan_geometry_files(geometry_dir: Path) -> dict[str, dict]:
    """
    Fallback: scan individual geometry enrichment JSONs to extract key scores.

    Returns dict keyed by feature_id str -> {protein_r2_cv, residue_pr_auc}.
    """
    if not geometry_dir.is_dir():
        return {}

    result = {}
    for fpath in sorted(geometry_dir.iterdir()):
        if fpath.name == "summary.json" or fpath.suffix != ".json":
            continue
        data = load_json(fpath)
        if not data:
            continue

        fid_str = str(data.get("feature_id", fpath.stem.lstrip("0") or "0"))
        entry = {}

        protein = data.get("geometric_protein_level", {})
        if protein.get("r2_cv") is not None:
            entry["protein_r2_cv"] = protein["r2_cv"]

        residue = data.get("geometric_residue_level", {})
        concordance = residue.get("concordance", {})
        if concordance.get("avg_precision") is not None:
            entry["residue_pr_auc"] = concordance["avg_precision"]

        if entry:
            result[fid_str] = entry

    if result:
        logger.info("Scanned %d geometry files, found %d with enrichment", len(result), len(result))
    return result


def _scan_motif_pwm_files(motif_dir: Path) -> dict[str, dict]:
    """
    Fallback: scan individual MEME/PWM enrichment JSONs to extract best PR-AUC.

    Each file has a ``motifs`` list sorted by PR-AUC descending.  We extract the
    best consensus and its PR-AUC score.

    Returns dict keyed by feature_id str -> {best_pr_auc, best_consensus}.
    """
    if not motif_dir.is_dir():
        return {}

    result = {}
    for fpath in sorted(motif_dir.iterdir()):
        if fpath.name == "summary.json" or fpath.suffix != ".json":
            continue
        data = load_json(fpath)
        if not data:
            continue

        fid_str = str(data.get("feature_id", fpath.stem.lstrip("0") or "0"))
        motifs = data.get("motifs", [])
        if motifs:
            best = motifs[0]
            pr_auc_dict = best.get("pr_auc") or {}
            result[fid_str] = {
                "best_pr_auc": pr_auc_dict.get("pr_auc"),
                "best_consensus": best.get("consensus"),
            }

    if result:
        logger.info("Scanned %d motif PWM files, found %d with enrichment", len(result), len(result))
    return result


def _scan_position_files(position_dir: Path) -> dict[str, dict]:
    """
    Fallback: scan individual position enrichment JSONs to extract best F1 scores.

    Each file has a ``top_positions`` list sorted by F1 descending.  We extract the
    best position predicate name and its F1 score.

    Returns dict keyed by feature_id str -> {best_position_f1, best_position}.
    """
    if not position_dir.is_dir():
        return {}

    result = {}
    for fpath in sorted(position_dir.iterdir()):
        if fpath.name == "summary.json" or fpath.suffix != ".json":
            continue
        data = load_json(fpath)
        if not data:
            continue

        fid_str = str(data.get("feature_id", fpath.stem.lstrip("0") or "0"))
        top_positions = data.get("top_positions", [])
        if top_positions:
            best = top_positions[0]
            result[fid_str] = {
                "best_position_f1": best.get("best_f1"),
                "best_position": best.get("position"),
            }

    if result:
        logger.info("Scanned %d position files, found %d with enrichment", len(result), len(result))
    return result


def _scan_cath_files(cath_dir: Path) -> dict[str, dict]:
    """
    Scan CATH enrichment JSONs to extract best F1 scores.

    For each feature, takes the max residue F1 across all hierarchy levels
    (C, CA, CAT, CATH) from the summary block.

    Returns dict keyed by feature_id str -> {best_cath_f1, best_cath_label}.
    """
    if not cath_dir.is_dir():
        return {}

    result = {}
    for fpath in sorted(cath_dir.iterdir()):
        if fpath.name == "summary.json" or fpath.suffix != ".json":
            continue
        data = load_json(fpath)
        if not data:
            continue

        fid_str = str(data.get("feature_id", fpath.stem.lstrip("0") or "0"))
        summary = data.get("summary", {})
        best_f1 = 0
        best_label = ""
        for level in ("C", "CA", "CAT", "CATH"):
            level_data = summary.get(level, {})
            res_f1 = level_data.get("top_residue_f1") or 0
            if res_f1 > best_f1:
                best_f1 = res_f1
                best_label = level_data.get("top_residue_label", "")

        if best_f1 > 0:
            result[fid_str] = {
                "best_cath_f1": best_f1,
                "best_cath_label": best_label,
            }

    if result:
        logger.info("Scanned %d CATH files, found %d with enrichment", len(result), len(result))
    return result


def _scan_nmpfam_files(nmpfam_dir: Path) -> dict[str, dict]:
    """
    Scan NMPFams enrichment JSONs to extract hit counts per feature.

    Returns dict keyed by feature_id str -> {n_nmpfam_hits}.
    """
    if not nmpfam_dir.is_dir():
        return {}

    result = {}
    for fpath in sorted(nmpfam_dir.iterdir()):
        if fpath.suffix != ".json":
            continue
        data = load_json(fpath)
        if not data:
            continue

        fid_str = str(data.get("feature_id", fpath.stem.lstrip("0") or "0"))
        n_hits = data.get("n_nmpfam_hits", 0)
        if n_hits > 0:
            result[fid_str] = {"n_nmpfam_hits": n_hits}

    if result:
        logger.info("Found NMPFams hits for %d features", len(result))
    return result


# ---------------------------------------------------------------------------
# Geometry radar: aggregate feature importances into 6 semantic categories
# ---------------------------------------------------------------------------

_RADAR_CATEGORY_MAP: dict[str, str] = {}
_RADAR_CATEGORIES = {
    "curvature": [
        "curvature_mean", "curvature_max", "curvature_std",
        "curv_N_third", "curv_centre_third", "curv_C_third",
        "narrow_curvature_mean", "narrow_curvature_max",
        "wide_curvature_mean", "wide_curvature_max",
    ],
    "torsion": [
        "torsion_mean", "torsion_std", "torsion_frac_pos",
        "tors_N_third", "tors_centre_third", "tors_C_third",
        "narrow_torsion_mean", "narrow_torsion_std",
        "wide_torsion_mean", "wide_torsion_std",
    ],
    "planarity": [
        "planarity_mean", "planarity_std",
        "plan_N_third", "plan_centre_third", "plan_C_third",
    ],
    "compactness": [
        "tangent_alignment", "end_to_end_ratio",
        "narrow_tangent_alignment", "narrow_end_to_end_ratio",
        "wide_tangent_alignment", "wide_end_to_end_ratio",
    ],
    "contacts": [
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
# Build reverse lookup: feature_name → category_name
for _cat, _feats in _RADAR_CATEGORIES.items():
    for _f in _feats:
        _RADAR_CATEGORY_MAP[_f] = _cat


def _aggregate_importances(importances: dict[str, float]) -> dict[str, float] | None:
    """Sum feature importances by category, then normalize to sum=1."""
    if not importances:
        return None
    scores = {cat: 0.0 for cat in _RADAR_CATEGORIES}
    for feat, val in importances.items():
        cat = _RADAR_CATEGORY_MAP.get(feat)
        if cat:
            scores[cat] += val
    total = sum(scores.values())
    if total == 0:
        return None
    return {cat: round(v / total, 4) for cat, v in scores.items()}


def _build_geometry_radar(geometry_dir: Path) -> dict[str, dict]:
    """
    Scan geometry enrichment files and compute category-aggregated radar vectors.

    Returns dict keyed by feature_id str → {curvature, torsion, planarity,
    compactness, contacts, composition} with normalized scores summing to 1.
    """
    if not geometry_dir.is_dir():
        return {}

    result = {}
    for fpath in sorted(geometry_dir.iterdir()):
        if fpath.name == "summary.json" or fpath.suffix != ".json":
            continue
        data = load_json(fpath)
        if not data:
            continue

        fid_str = str(data.get("feature_id", fpath.stem.lstrip("0") or "0"))
        importances = (
            data.get("geometric_residue_level", {}).get("feature_importances", {})
        )
        radar = _aggregate_importances(importances)
        if radar:
            result[fid_str] = radar

    if result:
        logger.info("Built geometry radar for %d features", len(result))
    return result
