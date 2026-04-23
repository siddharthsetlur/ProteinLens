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

# The seven annotation methods, in paper order. Used for m{k}_{score,label,q}.
METHODS: list[dict[str, Any]] = [
    {"id": 1, "name": "InterPro Protein", "metric": "F1"},
    {"id": 2, "name": "InterPro Residue", "metric": "F1"},
    {"id": 3, "name": "CATH Protein", "metric": "F1"},
    {"id": 4, "name": "CATH Residue", "metric": "F1"},
    {"id": 5, "name": "Sequence Position", "metric": "F1"},
    {"id": 6, "name": "Sequence MEME Motif", "metric": "PR-AUC"},
    {"id": 7, "name": "Geometric", "metric": "PR-AUC"},
]

Q_SIGNIFICANCE_THRESHOLD = 0.05


def _is_sig(q: float | None) -> bool:
    """Return True when the BH-corrected q-value is significant (q < 0.05)."""
    return q is not None and q < Q_SIGNIFICANCE_THRESHOLD


def _bh_correct(pvals: list[float | None]) -> list[float | None]:
    """
    Benjamini–Hochberg FDR correction.

    Input: a list of raw p-values; entries may be None (missing).
    Output: a same-length list of q-values; None passes through.

    Corrected q for the i-th sorted (ascending) finite p-value is
    ``p_i * m / (i+1)``, clipped to 1, then made monotonic non-increasing
    from largest p down to smallest.
    """
    n = len(pvals)
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
    if not indexed:
        return [None] * n
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    out: list[float | None] = [None] * n
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        orig_idx, p = indexed[rank]
        q = p * m / (rank + 1)
        if q > 1.0:
            q = 1.0
        if q < running_min:
            running_min = q
        out[orig_idx] = running_min
    return out


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


def build_stats(analysis_dir: Path) -> dict[str, Any]:
    """
    Merge dataset_stats.json and the SAE config.yaml into a single stats dict.

    The SAE config.yaml sits one directory up from the analysis dir
    (e.g., ``trained_models/layer_4/frosty-sweep-15/config.yaml`` for analysis
    dir ``.../frosty-sweep-15/analysis/``). We also honour ``sae_dir`` in
    dataset_stats.json as a last-resort fallback, but the absolute path it
    records often won't resolve on a different filesystem.

    If config.yaml is not found, the "sae" key stays empty and the homepage
    renders placeholders.
    """
    dataset_stats = load_json(analysis_dir / "dataset_stats.json") or {}

    sae_config: dict[str, Any] = {}
    sae_dir_rel = dataset_stats.get("sae_dir", "")
    candidates = [
        analysis_dir.parent / "config.yaml",
        analysis_dir / "config.yaml",
    ]
    if sae_dir_rel:
        candidates.extend(
            [
                Path(sae_dir_rel) / "config.yaml",
                analysis_dir.parent / sae_dir_rel / "config.yaml",
                analysis_dir / sae_dir_rel / "config.yaml",
            ]
        )
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with open(candidate) as f:
                raw = yaml.safe_load(f)
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


def build_pipeline_status(analysis_dir: Path) -> dict[str, Any]:
    """
    Compute pipeline completion status from filesystem state.

    Returns:
      - completed_stages: list of stage names from pipeline_state.json
      - feature_count: number of per-feature JSON files
      - interpro_count: number of interpro enrichment JSONs (excluding summary)
      - geometry_count: number of geometry enrichment JSONs (excluding summary)
    """
    pipeline_state = load_json(analysis_dir / "pipeline_state.json") or {}
    completed_stages = pipeline_state.get("completed_stages", [])

    # Count per-feature files (exclude summary.json)
    def count_jsons(subdir: str) -> int:
        d = analysis_dir / subdir
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


def build_feature_index(analysis_dir: Path) -> list[dict[str, Any]]:
    """
    Build the feature index for the homepage table.

    Merges data from multiple sources in the analysis dir into one list of
    dicts (one per feature). Each row carries a uniform 7-method schema:
    ``m{k}_score``, ``m{k}_label``, ``m{k}_q`` for k = 1..7 (paper order).
    Missing data is represented as ``None`` (serialized to JSON ``null``).

    q-values come from two sources:
      - ``geometry_primary_analysis.json`` supplies the 5 precomputed
        ``*_padj`` fields (methods 2, 4, 5, 6, 7).
      - ``permutation_null/*.json`` supplies the raw p-values for the two
        protein-level methods (1 and 3), which are BH-corrected in memory.
    """
    max_act_path = analysis_dir / "feature_max_activations.npy"
    if max_act_path.exists():
        max_activations = np.load(max_act_path).astype(float)
        num_features = len(max_activations)
    else:
        logger.warning(
            "feature_max_activations.npy not found, using dataset_stats for num_features",
        )
        ds = load_json(analysis_dir / "dataset_stats.json") or {}
        num_features = ds.get("num_features", 0)
        max_activations = np.zeros(num_features)

    coverage_data = load_json(analysis_dir / "survey_coverage.json") or {}

    interpro_summary = load_json(analysis_dir / "interpro_enrichment" / "summary.json")
    interpro_features = (
        interpro_summary.get("features", {})
        if interpro_summary and interpro_summary.get("features")
        else _scan_interpro_files(analysis_dir / "interpro_enrichment")
    )

    motif_summary = load_json(analysis_dir / "motif_pwm_enrichment" / "summary.json")
    motif_features = (
        motif_summary.get("features", {})
        if motif_summary and motif_summary.get("features")
        else _scan_motif_pwm_files(analysis_dir / "motif_pwm_enrichment")
    )

    position_summary = load_json(analysis_dir / "position_enrichment" / "summary.json")
    position_features = (
        position_summary.get("features", {})
        if position_summary and position_summary.get("features")
        else _scan_position_files(analysis_dir / "position_enrichment")
    )

    # CATH summary has per-hierarchy blocks (C/CA/CAT/CATH); compute best
    # across all four levels for m3 (protein-level) and m4 (residue-level).
    cath_summary = load_json(analysis_dir / "cath_enrichment" / "summary.json")
    cath_features = _extract_cath_bests(
        cath_summary.get("features") if cath_summary else None,
        analysis_dir / "cath_enrichment",
    )

    nmpfam_hits = _scan_nmpfam_files(analysis_dir / "nmpfam" / "nmpfam_enrichment")
    geometry_radar = _build_geometry_radar(analysis_dir / "geometry_enrichment")

    # Geometry-primary analysis (q-values for 5 methods + structural category)
    gp_data = load_json(analysis_dir / "geometry_primary_analysis.json") or {}
    gp_features = gp_data.get("features", {})

    # Permutation null p-values — feeds BH for interpro_protein + cath_protein
    perm_pvals = _load_permutation_pvals(analysis_dir / "permutation_null")
    ipro_prot_p: list[float | None] = [
        perm_pvals.get(str(fid), {}).get("interpro_protein_f1") for fid in range(num_features)
    ]
    cath_prot_p: list[float | None] = [
        perm_pvals.get(str(fid), {}).get("cath_protein_f1") for fid in range(num_features)
    ]
    ipro_prot_q = _bh_correct(ipro_prot_p)
    cath_prot_q = _bh_correct(cath_prot_p)
    logger.info(
        "Loaded permutation p-values for %d features; BH applied for interpro_protein (%d) and cath_protein (%d)",
        len(perm_pvals),
        sum(1 for q in ipro_prot_q if q is not None),
        sum(1 for q in cath_prot_q if q is not None),
    )

    # --- Merge into index rows ---
    index: list[dict[str, Any]] = []
    for fid in range(num_features):
        fid_str = str(fid)

        cov = coverage_data.get(fid_str, {})
        ipro = interpro_features.get(fid_str, {})
        posn = position_features.get(fid_str, {})
        motif = motif_features.get(fid_str, {})
        cath = cath_features.get(fid_str, {})
        nmpf = nmpfam_hits.get(fid_str, {})
        gp = gp_features.get(fid_str, {})

        # m1: InterPro protein
        m1_score = ipro.get("top_protein_f1") or ipro.get("protein_best_f1")
        m1_label = ipro.get("top_protein_annotation_name") or ipro.get("protein_best_name")
        m1_q = ipro_prot_q[fid]

        # m2: InterPro residue
        m2_score = ipro.get("top_residue_f1") or ipro.get("residue_best_f1")
        m2_label = ipro.get("top_residue_annotation") or ipro.get("top_residue_annotation_name")
        m2_q = gp.get("interpro_res_f1_padj")

        # m3: CATH protein (max across hierarchy levels)
        m3_score = cath.get("best_protein_f1")
        m3_label = cath.get("best_protein_label")
        m3_q = cath_prot_q[fid]

        # m4: CATH residue (max across hierarchy levels)
        m4_score = cath.get("best_residue_f1")
        m4_label = cath.get("best_residue_label")
        m4_q = gp.get("cath_res_f1_padj")

        # m5: Sequence position
        m5_score = posn.get("best_position_f1")
        m5_label = posn.get("best_position")
        m5_q = gp.get("position_f1_padj")

        # m6: Sequence MEME motif
        m6_score = motif.get("best_pr_auc")
        m6_label = motif.get("best_consensus")
        m6_q = gp.get("motif_pr_auc_padj")

        # m7: Geometric (score is geom_pr_auc from geometry_primary_analysis;
        # label is the structural category the top geometric feature belongs to)
        m7_score = gp.get("geom_pr_auc")
        m7_label = gp.get("structural_category")
        m7_q = gp.get("geometry_prauc_padj")

        row = {
            "feature_id": fid,
            "max_activation": round(float(max_activations[fid]), 6),
            "pct_proteins_activated": cov.get("pct_proteins_activated"),
            "pct_clusters_activated": cov.get("pct_clusters_activated"),
            # 7 annotation methods, paper order
            "m1_score": m1_score, "m1_label": m1_label, "m1_q": m1_q,
            "m2_score": m2_score, "m2_label": m2_label, "m2_q": m2_q,
            "m3_score": m3_score, "m3_label": m3_label, "m3_q": m3_q,
            "m4_score": m4_score, "m4_label": m4_label, "m4_q": m4_q,
            "m5_score": m5_score, "m5_label": m5_label, "m5_q": m5_q,
            "m6_score": m6_score, "m6_label": m6_label, "m6_q": m6_q,
            "m7_score": m7_score, "m7_label": m7_label, "m7_q": m7_q,
            # Geometry radar + structural category for the radar glyph and cards
            "geometry_radar": geometry_radar.get(fid_str),
            "top_geometric_feature": gp.get("top_geometric_feature"),
            "structural_category": gp.get("structural_category"),
            "is_geometry_primary": gp.get("is_geometry_primary"),
            "n_nmpfam_hits": nmpf.get("n_nmpfam_hits"),
        }
        index.append(row)

    counts = {k: sum(1 for r in index if _is_sig(r.get(f"m{k}_q"))) for k in range(1, 8)}
    logger.info(
        "Built feature index: %d features (q<0.05 counts: "
        "m1=%d m2=%d m3=%d m4=%d m5=%d m6=%d m7=%d)",
        num_features,
        counts[1], counts[2], counts[3], counts[4], counts[5], counts[6], counts[7],
    )
    return index


def build_method_coverage(index: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarise significance coverage across the 7 annotation methods.

    A feature counts as annotated by method k when ``m{k}_q < 0.05``.
    ``total_annotated_*`` counts features significant for at least one method.
    """
    total = len(index)
    methods_out: list[dict[str, Any]] = []
    for meta in METHODS:
        k = meta["id"]
        n_sig = sum(1 for row in index if _is_sig(row.get(f"m{k}_q")))
        methods_out.append(
            {
                "id": k,
                "name": meta["name"],
                "metric": meta["metric"],
                "n_significant": n_sig,
                "total": total,
                "pct": round(100.0 * n_sig / total, 2) if total else 0.0,
            }
        )
    total_annotated = sum(
        1
        for row in index
        if any(_is_sig(row.get(f"m{k}_q")) for k in range(1, 8))
    )
    return {
        "methods": methods_out,
        "total_features": total,
        "total_annotated_n": total_annotated,
        "total_annotated_pct": round(100.0 * total_annotated / total, 2) if total else 0.0,
    }


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


def _extract_cath_bests(
    summary_features: dict | None,
    cath_dir: Path,
) -> dict[str, dict]:
    """
    Extract per-feature best CATH protein-level and residue-level scores.

    For each feature we take the max across the four hierarchy levels
    (C, CA, CAT, CATH) for the protein- and residue-level F1 independently.
    The matching CATH code (e.g. "1.10.760.10") and description are kept as
    the label. When the ``summary.json`` feature block is available we read
    from it; otherwise we fall back to scanning the per-feature JSONs.

    Returns dict keyed by feature_id str with:
      best_protein_f1, best_protein_label,
      best_residue_f1, best_residue_label.
    """
    features_iter: list[tuple[str, dict]] = []
    if summary_features:
        features_iter = [(fid, entry) for fid, entry in summary_features.items()]
    elif cath_dir.is_dir():
        for fpath in sorted(cath_dir.iterdir()):
            if fpath.name == "summary.json" or fpath.suffix != ".json":
                continue
            data = load_json(fpath)
            if not data:
                continue
            fid_str = str(data.get("feature_id", fpath.stem.lstrip("0") or "0"))
            # In per-feature files the hierarchy blocks live under "summary"
            features_iter.append((fid_str, data.get("summary", {})))
    else:
        return {}

    result: dict[str, dict] = {}
    for fid_str, entry in features_iter:
        best_prot = 0.0
        best_prot_label = ""
        best_res = 0.0
        best_res_label = ""
        for level in ("C", "CA", "CAT", "CATH"):
            lvl = entry.get(level, {}) if isinstance(entry, dict) else {}
            if not isinstance(lvl, dict):
                continue
            prot_f1 = lvl.get("top_protein_f1") or 0
            if prot_f1 > best_prot:
                best_prot = prot_f1
                best_prot_label = (
                    lvl.get("top_protein_description")
                    or lvl.get("top_protein_label")
                    or ""
                )
            res_f1 = lvl.get("top_residue_f1") or 0
            if res_f1 > best_res:
                best_res = res_f1
                best_res_label = (
                    lvl.get("top_residue_description")
                    or lvl.get("top_residue_label")
                    or ""
                )
        if best_prot > 0 or best_res > 0:
            result[fid_str] = {
                "best_protein_f1": best_prot or None,
                "best_protein_label": best_prot_label or None,
                "best_residue_f1": best_res or None,
                "best_residue_label": best_res_label or None,
            }

    if result:
        logger.info(
            "Extracted CATH best scores for %d features", len(result)
        )
    return result


def _load_permutation_pvals(permutation_dir: Path) -> dict[str, dict[str, float | None]]:
    """
    Load raw p-values from the permutation-null directory.

    Reads every ``{fid:04d}.json`` and pulls out the two protein-level metric
    p-values we need for BH correction at startup (the residue-level / motif /
    geometry / position methods already have BH q-values in
    ``geometry_primary_analysis.json``).

    Returns a dict keyed by feature_id str with
    ``{"interpro_protein_f1": p, "cath_protein_f1": p}``.
    """
    if not permutation_dir.is_dir():
        logger.warning("Permutation null dir not found: %s", permutation_dir)
        return {}
    result: dict[str, dict[str, float | None]] = {}
    for fpath in sorted(permutation_dir.iterdir()):
        if fpath.suffix != ".json":
            continue
        data = load_json(fpath)
        if not data:
            continue
        fid_str = str(data.get("feature_id", fpath.stem.lstrip("0") or "0"))
        pvals = data.get("p_values", {})
        result[fid_str] = {
            "interpro_protein_f1": pvals.get("interpro_protein_f1"),
            "cath_protein_f1": pvals.get("cath_protein_f1"),
        }
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
