#!/usr/bin/env python3
"""Compute geometry-primary latent analysis.

Reads existing enrichment outputs (geometry, motif, position, InterPro, CATH) and
identifies SAE latents whose activation is best explained by local 3D protein
structure rather than sequence-level features.

**Method:**

1. Estimate empirical null p95 for each sequence metric from features with
   <1% protein activation (which should have no real signal).
2. A feature is "geometry-primary" if its geometry PR-AUC exceeds 0.3 (well
   above the random baseline of ~0.038) AND all sequence metrics fall below
   their respective null p95 thresholds.
3. Features are ranked by a composite score:
   ``PR-AUC * (1 - seq_feature_fraction) * sqrt(concordance_F1)``
   which rewards high geometry quality, low sequence-composition leakage,
   and good spatial concordance.

**Outputs:**

- ``geometry_primary_analysis.json`` — per-feature classification + scores.
- ``geometry_primary_case_studies.md`` — top 20 ranked features for case study.

Usage::

    python scripts/compute_geometry_primary.py --data-dir feature_data_cluster
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# Geometric features that are derived from amino acid identity, not 3D structure
SEQ_DERIVED_FEATURES = {
    "frac_charged", "frac_polar", "frac_hydrophobic",
    "frac_aromatic", "frac_tiny", "frac_small",
}

# Map geometric feature names to interpretable structural categories
STRUCTURAL_CATEGORIES = {
    "narrow_end_to_end_ratio": "Local compactness (loop/turn vs extended)",
    "wide_end_to_end_ratio": "Global compactness",
    "end_to_end_ratio": "Global compactness",
    "narrow_curvature_mean": "Backbone curvature",
    "narrow_curvature_max": "Backbone curvature",
    "wide_curvature_mean": "Backbone curvature",
    "wide_curvature_max": "Backbone curvature",
    "curvature_std": "Backbone curvature",
    "curv_N_third": "Curvature profile (N-terminal third)",
    "curv_centre_third": "Curvature profile (central third)",
    "curv_C_third": "Curvature profile (C-terminal third)",
    "narrow_torsion_mean": "Backbone torsion",
    "narrow_torsion_std": "Backbone torsion",
    "wide_torsion_mean": "Backbone torsion",
    "wide_torsion_std": "Backbone torsion",
    "torsion_std": "Backbone torsion",
    "torsion_frac_pos": "Backbone torsion",
    "tors_N_third": "Torsion profile",
    "tors_centre_third": "Torsion profile",
    "tors_C_third": "Torsion profile",
    "plan_N_third": "Planarity profile",
    "plan_centre_third": "Planarity profile",
    "plan_C_third": "Planarity profile",
    "narrow_tangent_alignment": "Tangent alignment (helix-like)",
    "wide_tangent_alignment": "Tangent alignment",
    "tangent_alignment": "Tangent alignment",
    "contact_density_8A": "Contact density (packing)",
    "contact_density_12A": "Contact density (packing)",
    "long_range_contacts_8A": "Long-range contacts",
    "long_range_contacts_12A": "Long-range contacts",
    "mean_seq_sep_contact_8A": "Contact sequence separation",
    "max_seq_sep_contact_8A": "Contact sequence separation",
    "min_spatial_dist_long": "Spatial distance (long-range)",
}

GEOM_PR_AUC_THRESHOLD = 0.3


def _load_enrichment_scores(data_dir: Path) -> dict:
    """Load all enrichment scores keyed by feature ID (int)."""
    geom, motif, pos, ipro_res, cath_res = {}, {}, {}, {}, {}

    geom_dir = data_dir / "geometry_enrichment"
    if geom_dir.is_dir():
        for fpath in geom_dir.iterdir():
            if fpath.name == "summary.json" or fpath.suffix != ".json":
                continue
            d = json.loads(fpath.read_text())
            res = d.get("geometric_residue_level", {})
            conc = res.get("concordance", {})
            if conc.get("avg_precision") is not None:
                geom[d["feature_id"]] = {
                    "pr_auc": conc["avg_precision"],
                    "f1": conc.get("f1", 0),
                    "iou": conc.get("iou", 0),
                    "precision": conc.get("precision", 0),
                    "recall": conc.get("recall", 0),
                    "importances": res.get("feature_importances", {}),
                    "rules": res.get("rules", ""),
                }

    motif_dir = data_dir / "motif_pwm_enrichment"
    if motif_dir.is_dir():
        for fpath in motif_dir.iterdir():
            if fpath.name == "summary.json" or fpath.suffix != ".json":
                continue
            d = json.loads(fpath.read_text())
            tops = d.get("motifs", [])
            if tops:
                pr_auc_dict = tops[0].get("pr_auc") or {}
                val = pr_auc_dict.get("pr_auc")
                if val is not None:
                    motif[d["feature_id"]] = val

    pos_dir = data_dir / "position_enrichment"
    if pos_dir.is_dir():
        for fpath in pos_dir.iterdir():
            if fpath.name == "summary.json" or fpath.suffix != ".json":
                continue
            d = json.loads(fpath.read_text())
            tops = d.get("top_positions", [])
            if tops:
                pos[d["feature_id"]] = tops[0]["best_f1"]

    ipro_dir = data_dir / "interpro_enrichment"
    if ipro_dir.is_dir():
        for fpath in ipro_dir.iterdir():
            if fpath.name == "summary.json" or fpath.suffix != ".json":
                continue
            d = json.loads(fpath.read_text())
            rl = d.get("residue_level", [])
            if rl:
                ipro_res[d["feature_id"]] = max(
                    x.get("best_f1", 0) for x in rl
                )

    cath_dir = data_dir / "cath_enrichment"
    if cath_dir.is_dir():
        for fpath in cath_dir.iterdir():
            if fpath.name == "summary.json" or fpath.suffix != ".json":
                continue
            d = json.loads(fpath.read_text())
            summary = d.get("summary", {})
            best_f1 = 0
            for level in ("C", "CA", "CAT", "CATH"):
                f1 = (summary.get(level) or {}).get("top_residue_f1") or 0
                if f1 > best_f1:
                    best_f1 = f1
            if best_f1 > 0:
                cath_res[d["feature_id"]] = best_f1

    return {"geom": geom, "motif": motif, "pos": pos, "ipro_res": ipro_res, "cath_res": cath_res}


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    if n == 0:
        return pvals
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    adjusted = np.zeros(n)
    # BH: p_adj[i] = min(p[i] * n / rank[i], 1.0), enforcing monotonicity
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = min(sorted_pvals[i] * n / rank, 1.0)
        cum_min = min(cum_min, adj)
        adjusted[sorted_idx[i]] = cum_min
    return adjusted


def _load_permutation_pvalues(data_dir: Path) -> dict | None:
    """Load per-feature p-values from permutation null, apply BH FDR.

    Returns dict: metric_name -> {feature_id (int) -> adjusted_pvalue}, or
    None if permutation null data is not available.
    """
    perm_dir = data_dir / "permutation_null"
    if not perm_dir.is_dir():
        return None

    metrics = ["pwm_pr_auc", "position_f1", "interpro_res_f1", "cath_res_f1", "geometry_prauc"]
    raw: dict[str, dict[int, float]] = {m: {} for m in metrics}

    n_loaded = 0
    for fpath in perm_dir.iterdir():
        if fpath.suffix != ".json":
            continue
        try:
            d = json.loads(fpath.read_text())
            fid = d["feature_id"]
            pvals = d["p_values"]
            for m in metrics:
                if m in pvals:
                    raw[m][fid] = pvals[m]
            n_loaded += 1
        except (json.JSONDecodeError, KeyError, OSError):
            continue

    if n_loaded == 0:
        return None

    # Apply BH FDR correction per metric
    adjusted: dict[str, dict[int, float]] = {}
    for m in metrics:
        fids = sorted(raw[m].keys())
        if not fids:
            adjusted[m] = {}
            continue
        pvals_arr = np.array([raw[m][fid] for fid in fids])
        padj = _benjamini_hochberg(pvals_arr)
        adjusted[m] = {fid: float(padj[i]) for i, fid in enumerate(fids)}

    print(f"  Loaded permutation p-values for {n_loaded} features, applied BH FDR")
    return adjusted


def _compute_null_thresholds(
    data_dir: Path, scores: dict, sparse_pct: float = 1.0
) -> dict:
    """Compute p95 null thresholds from features with sparse activation."""
    cov_path = data_dir / "survey_coverage.json"
    if not cov_path.exists():
        return {"motif_pr_auc": 0.20, "position_f1": 0.12, "interpro_res_f1": 0.20, "cath_res_f1": 0.20}

    cov = json.loads(cov_path.read_text())
    sparse_fids = set()
    for fid_str, c in cov.items():
        if c.get("pct_proteins_activated", 100) < sparse_pct:
            sparse_fids.add(int(fid_str))

    def _null_p95(score_dict: dict, sparse: set) -> float:
        vals = [v for fid, v in score_dict.items() if fid in sparse]
        if len(vals) < 10:
            return 0.0
        return float(np.percentile(vals, 95))

    return {
        "motif_pr_auc": _null_p95(scores["motif"], sparse_fids),
        "position_f1": _null_p95(scores["pos"], sparse_fids),
        "interpro_res_f1": _null_p95(scores["ipro_res"], sparse_fids),
        "cath_res_f1": _null_p95(scores["cath_res"], sparse_fids),
        "n_sparse_features": len(sparse_fids),
    }


def _classify_features(scores: dict, null_thresholds: dict, perm_pvalues: dict | None = None) -> dict:
    """Classify each feature and compute composite scores."""
    geom = scores["geom"]
    motif = scores["motif"]
    pos = scores["pos"]
    ipro_res = scores["ipro_res"]
    cath_res = scores["cath_res"]

    motif_null = null_thresholds["motif_pr_auc"]
    pos_null = null_thresholds["position_f1"]
    ipro_null = null_thresholds["interpro_res_f1"]
    cath_null = null_thresholds["cath_res_f1"]

    features = {}
    n_primary = 0

    for fid, g in geom.items():
        m = motif.get(fid, 0)
        p = pos.get(fid, 0)
        ir = ipro_res.get(fid, 0)
        cr = cath_res.get(fid, 0)

        # Sequence-derived feature fraction
        imps = g["importances"]
        total_imp = sum(imps.values())
        seq_imp = sum(imps.get(f, 0) for f in SEQ_DERIVED_FEATURES)
        seq_frac = seq_imp / total_imp if total_imp > 0 else 0

        # Top geometric feature
        if imps:
            top_feat = max(imps, key=imps.get)
        else:
            top_feat = ""
        category = STRUCTURAL_CATEGORIES.get(top_feat, top_feat)

        # Best sequence F1 (excluding broken InterPro protein-level)
        best_seq_f1 = max(m, p, ir, cr)

        # Geometry-primary classification
        if perm_pvalues is not None:
            # Permutation-based: use BH-adjusted p-values.
            # Defaults: geometry=1.0 (not significant → excluded),
            # sequence metrics=0.0 (significant → excluded).  This is
            # conservative: features without full permutation data are
            # never classified as geometry-primary.
            fdr_threshold = 0.05
            is_primary = (
                perm_pvalues["geometry_prauc"].get(fid, 1.0) < fdr_threshold
                and perm_pvalues["pwm_pr_auc"].get(fid, 0.0) >= fdr_threshold
                and perm_pvalues["position_f1"].get(fid, 0.0) >= fdr_threshold
                and perm_pvalues["interpro_res_f1"].get(fid, 0.0) >= fdr_threshold
                and perm_pvalues["cath_res_f1"].get(fid, 0.0) >= fdr_threshold
            )
        else:
            # Fallback: sparse-feature null thresholds
            is_primary = (
                g["pr_auc"] > GEOM_PR_AUC_THRESHOLD
                and m <= motif_null
                and p <= pos_null
                and ir <= ipro_null
                and cr <= cath_null
            )

        # Composite score
        score = g["pr_auc"] * (1 - seq_frac) * math.sqrt(max(g["f1"], 0.01))

        if is_primary:
            n_primary += 1

        features[str(fid)] = {
            "composite_score": round(score, 4),
            "geom_pr_auc": round(g["pr_auc"], 4),
            "concordance_f1": round(g["f1"], 4),
            "concordance_iou": round(g["iou"], 4),
            "concordance_precision": round(g["precision"], 4),
            "concordance_recall": round(g["recall"], 4),
            "motif_pr_auc": round(m, 4),
            "position_f1": round(p, 4),
            "interpro_res_f1": round(ir, 4),
            "cath_res_f1": round(cr, 4),
            "best_seq_f1": round(best_seq_f1, 4),
            "seq_feature_fraction": round(seq_frac, 4),
            "top_geometric_feature": top_feat,
            "structural_category": category,
            "is_geometry_primary": is_primary,
            # Permutation p-values (None if not available)
            "geometry_prauc_padj": perm_pvalues["geometry_prauc"].get(fid) if perm_pvalues else None,
            "motif_pr_auc_padj": perm_pvalues["pwm_pr_auc"].get(fid) if perm_pvalues else None,
            "position_f1_padj": perm_pvalues["position_f1"].get(fid) if perm_pvalues else None,
            "interpro_res_f1_padj": perm_pvalues["interpro_res_f1"].get(fid) if perm_pvalues else None,
            "cath_res_f1_padj": perm_pvalues["cath_res_f1"].get(fid) if perm_pvalues else None,
        }

    return features, n_primary


def _write_case_studies(
    features: dict, geom_scores: dict, null_thresholds: dict, out_path: Path
) -> None:
    """Write top 20 geometry-primary features as a Markdown case study list."""
    primary = [
        (fid, f) for fid, f in features.items() if f["is_geometry_primary"]
    ]
    primary.sort(key=lambda x: -x[1]["composite_score"])

    lines = [
        "# Geometry-Primary SAE Latents: Case Study List",
        "",
        "Generated by `scripts/compute_geometry_primary.py`.",
        "",
        "## Method",
        "",
        "A feature is **geometry-primary** if:",
        f"- Geometry PR-AUC > {GEOM_PR_AUC_THRESHOLD} (random baseline ~0.038)",
        f"- Motif PR-AUC <= {null_thresholds['motif_pr_auc']:.3f} (null p95)",
        f"- Position F1 <= {null_thresholds['position_f1']:.3f} (null p95)",
        f"- InterPro Residue F1 <= {null_thresholds['interpro_res_f1']:.3f} (null p95)",
        f"- CATH Residue F1 <= {null_thresholds['cath_res_f1']:.3f} (null p95)",
        "",
        "Null thresholds estimated from features with <1% protein activation "
        f"(N={null_thresholds.get('n_sparse_features', '?')}).",
        "",
        f"**{len(primary)} geometry-primary features** out of {len(features)} "
        "with geometry data.",
        "",
        "Composite score = `PR-AUC * (1 - seq_feature_fraction) * sqrt(concordance_F1)`",
        "",
        "---",
        "",
        "## Top 20 Geometry-Primary Latents",
        "",
    ]

    for rank, (fid, f) in enumerate(primary[:20], 1):
        g = geom_scores.get(int(fid), {})
        rules = g.get("rules", "")
        first_rule = ""
        for line in rules.split("\n"):
            if "---" in line and "<=" in line or ">" in line:
                first_rule = line.strip().lstrip("|--- ")
                break

        top3_imps = sorted(
            g.get("importances", {}).items(), key=lambda x: -x[1]
        )[:3]
        top3_str = ", ".join(
            f"`{k}` ({v:.3f})" for k, v in top3_imps
        )

        lines.extend([
            f"### {rank}. Feature {fid} (score={f['composite_score']:.3f})",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Geometry PR-AUC | **{f['geom_pr_auc']:.3f}** |",
            f"| Concordance F1 | {f['concordance_f1']:.3f} |",
            f"| Concordance IoU | {f['concordance_iou']:.3f} |",
            f"| Concordance P / R | {f['concordance_precision']:.3f} / {f['concordance_recall']:.3f} |",
            f"| Motif PR-AUC | {f['motif_pr_auc']:.3f} (null p95: {null_thresholds['motif_pr_auc']:.3f}) |",
            f"| Position F1 | {f['position_f1']:.3f} (null p95: {null_thresholds['position_f1']:.3f}) |",
            f"| InterPro Res F1 | {f['interpro_res_f1']:.3f} (null p95: {null_thresholds['interpro_res_f1']:.3f}) |",
            f"| CATH Res F1 | {f['cath_res_f1']:.3f} (null p95: {null_thresholds['cath_res_f1']:.3f}) |",
            f"| Seq-derived fraction | {f['seq_feature_fraction']:.3f} |",
            f"| Structural category | {f['structural_category']} |",
            "",
            f"**Top geometric features:** {top3_str}",
            "",
            f"**First decision rule:** `{first_rule}`" if first_rule else "",
            "",
            "---",
            "",
        ])

    out_path.write_text("\n".join(lines))
    print(f"Wrote case studies to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("feature_data_cluster"),
        help="Path to the pipeline output directory",
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    print(f"Loading enrichment scores from {data_dir}/ ...")
    scores = _load_enrichment_scores(data_dir)
    print(
        f"  Geometry: {len(scores['geom'])}, Motif: {len(scores['motif'])}, "
        f"Position: {len(scores['pos'])}, InterPro Res: {len(scores['ipro_res'])}, "
        f"CATH Res: {len(scores['cath_res'])}"
    )

    print("Computing null thresholds from sparse features ...")
    null_thresholds = _compute_null_thresholds(data_dir, scores)
    print(f"  Motif PR-AUC null p95: {null_thresholds['motif_pr_auc']:.3f}")
    print(f"  Position F1 null p95: {null_thresholds['position_f1']:.3f}")
    print(f"  InterPro Res F1 null p95: {null_thresholds['interpro_res_f1']:.3f}")
    print(f"  CATH Res F1 null p95: {null_thresholds['cath_res_f1']:.3f}")
    print(f"  Sparse features used: {null_thresholds.get('n_sparse_features', '?')}")

    print("Loading permutation p-values (if available) ...")
    perm_pvalues = _load_permutation_pvalues(data_dir)
    if perm_pvalues is None:
        print("  No permutation data found, using sparse-feature null thresholds")

    print("Classifying features ...")
    features, n_primary = _classify_features(scores, null_thresholds, perm_pvalues)
    print(
        f"  {n_primary} geometry-primary out of {len(features)} "
        f"with geometry data ({100*n_primary/max(len(features),1):.1f}%)"
    )

    analysis = {
        "method": "permutation" if perm_pvalues else "sparse_null",
        "null_thresholds": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in null_thresholds.items()
        },
        "geom_pr_auc_threshold": GEOM_PR_AUC_THRESHOLD,
        "n_geometry_primary": n_primary,
        "n_features_with_geometry": len(features),
        "features": features,
    }
    if perm_pvalues:
        analysis["fdr_method"] = "benjamini-hochberg"
        analysis["fdr_threshold"] = 0.05

    out_path = data_dir / "geometry_primary_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Wrote analysis to {out_path}")

    case_study_path = data_dir / "geometry_primary_case_studies.md"
    _write_case_studies(features, scores["geom"], null_thresholds, case_study_path)


if __name__ == "__main__":
    main()
