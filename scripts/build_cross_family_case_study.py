#!/usr/bin/env python3
"""Build the cross-family geometry case study JSON.

Identifies geometry-primary SAE features where InterPro partially matches
(F1 0.3-0.7) with MULTIPLE protein families — evidence that the feature
encodes a geometric motif that transcends any single family.

Also collects global statistics across all 89 geometry-primary features
to show that geometry is the invariant, not sequence or family identity.

Usage:
    python scripts/build_cross_family_case_study.py --data-dir feature_data_cluster
"""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-family geometry case study")
    parser.add_argument("--data-dir", required=True, type=str)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    gp_path = data_dir / "geometry_primary_analysis.json"
    ip_dir = data_dir / "interpro_enrichment"
    geo_dir = data_dir / "geometry_enrichment"
    motif_dir = data_dir / "motif_pwm_enrichment"
    pos_dir = data_dir / "position_enrichment"

    gp = json.loads(gp_path.read_text())
    gp_features = {k: v for k, v in gp["features"].items() if v.get("is_geometry_primary")}

    # ── Per-feature deep dive data ──────────────────────────────────

    features_detail = []
    for fid, info in sorted(gp_features.items(), key=lambda x: -x[1].get("composite_score", 0)):
        fid_padded = f"{int(fid):04d}"

        # InterPro
        ip_path = ip_dir / f"{fid_padded}.json"
        if not ip_path.exists():
            continue
        ip = json.loads(ip_path.read_text())
        prot_entries = ip.get("protein_level", [])
        res_entries = ip.get("residue_level", [])
        if not prot_entries:
            continue

        best_ip = max(prot_entries, key=lambda x: x.get("best_f1", 0))
        best_ip_f1 = best_ip.get("best_f1", 0)
        n_families_above_03 = sum(1 for e in prot_entries if e.get("best_f1", 0) > 0.3)
        n_families_above_05 = sum(1 for e in prot_entries if e.get("best_f1", 0) > 0.5)

        # Geometry
        geo_path = geo_dir / f"{fid_padded}.json"
        if not geo_path.exists():
            continue
        geo = json.loads(geo_path.read_text())
        rl = geo.get("geometric_residue_level", {})
        ms = rl.get("motif_superposition", {})
        motif_len = len(ms.get("per_position_flexibility", []))

        # MEME/PWM motif enrichment
        motif_pr_auc = 0.0
        motif_best = None
        motif_path = motif_dir / f"{fid_padded}.json"
        if motif_path.exists():
            mdata = json.loads(motif_path.read_text())
            top_motifs = mdata.get("motifs", [])
            if top_motifs:
                pr_auc_dict = top_motifs[0].get("pr_auc") or {}
                motif_pr_auc = pr_auc_dict.get("pr_auc", 0)
                motif_best = top_motifs[0].get("consensus", "")

        # Position enrichment
        pos_f1 = 0.0
        pos_path = pos_dir / f"{fid_padded}.json"
        if pos_path.exists():
            pdata = json.loads(pos_path.read_text())
            top_pos = pdata.get("top_positions", [])
            if top_pos:
                pos_f1 = top_pos[0].get("best_f1", 0)

        # All interpro families sorted by F1
        ip_families = []
        for e in sorted(prot_entries, key=lambda x: -x.get("best_f1", 0)):
            ip_families.append({
                "name": e.get("annotation_name", ""),
                "code": e.get("annotation_code", ""),
                "f1": e.get("best_f1", 0),
                "precision": e.get("precision_at_best", 0),
                "recall": e.get("recall_at_best", 0),
                "tp": e.get("n_true_positives", 0),
                "fp": e.get("n_false_positives", 0),
                "fn": e.get("n_false_negatives", 0),
            })

        best_res = max(res_entries, key=lambda x: x.get("best_f1", 0)) if res_entries else {}

        feature_rec = {
            "feature_id": int(fid),
            "composite_score": info.get("composite_score", 0),
            "structural_category": info.get("structural_category", ""),
            "top_geometric_feature": info.get("top_geometric_feature", ""),
            "is_cross_family": 0.3 <= best_ip_f1 <= 0.7 and n_families_above_03 >= 2,
            # Geometry metrics
            "gbm_auc_cv": rl.get("gbm_auc_cv", 0),
            "tree_f1_cv": rl.get("tree_f1_cv", 0),
            "concordance_f1": info.get("concordance_f1", 0),
            "concordance_prauc": info.get("geom_pr_auc", 0),
            "motif_rmsd": ms.get("mean_rmsd", None),
            "motif_std_rmsd": ms.get("std_rmsd", None),
            "motif_n_fragments": ms.get("n_fragments", 0),
            "motif_length": motif_len,
            "motif_rmsd_per_pos": ms.get("mean_rmsd", 0) / motif_len if motif_len > 0 else None,
            "feature_importances": rl.get("feature_importances", {}),
            # InterPro metrics
            "best_interpro_protein_f1": best_ip_f1,
            "best_interpro_protein_name": best_ip.get("annotation_name", ""),
            "best_interpro_residue_f1": best_res.get("best_f1", 0),
            "best_interpro_residue_name": best_res.get("annotation_name", ""),
            "n_families_above_03": n_families_above_03,
            "n_families_above_05": n_families_above_05,
            "interpro_families": ip_families,
            # Sequence metrics
            "motif_seq_pr_auc": motif_pr_auc,
            "motif_seq_best": motif_best,
            "position_f1": pos_f1,
        }
        features_detail.append(feature_rec)

    # ── Global statistics ───────────────────────────────────────────

    total_gp = len(gp_features)
    cross_family = [f for f in features_detail if f["is_cross_family"]]
    low_prec_strong = [
        f for f in features_detail
        if f["best_interpro_protein_f1"] > 0.7
        and any(
            fam["precision"] < 0.7
            for fam in f["interpro_families"]
            if fam["f1"] == f["best_interpro_protein_f1"]
        )
    ]
    # All geometry-primary features with multiple families
    multi_family_all = [f for f in features_detail if f["n_families_above_05"] >= 2]

    # Residue-level stats
    res_f1s = [f["best_interpro_residue_f1"] for f in features_detail]

    # Pull full methodology info from geometry_primary_analysis
    null_t = gp.get("null_thresholds", {})
    geom_pr_auc_threshold = gp.get("geom_pr_auc_threshold", 0.3)
    n_features_with_geometry = gp.get("n_features_with_geometry", 0)

    global_stats = {
        "total_geometry_primary": total_gp,
        "n_features_with_geometry": n_features_with_geometry,
        "n_cross_family": len(cross_family),
        "pct_cross_family": round(100 * len(cross_family) / total_gp, 1) if total_gp else 0,
        "n_multi_family": len(multi_family_all),
        "pct_multi_family": round(100 * len(multi_family_all) / total_gp, 1) if total_gp else 0,
        "n_zero_single_family": sum(1 for f in features_detail if f["n_families_above_05"] < 2),
        "interpro_residue_f1_max": round(max(res_f1s), 3) if res_f1s else 0,
        "interpro_residue_f1_mean": round(sum(res_f1s) / len(res_f1s), 3) if res_f1s else 0,
        "interpro_residue_f1_median": round(sorted(res_f1s)[len(res_f1s) // 2], 3) if res_f1s else 0,
        "null_thresholds": {
            "interpro_res_f1": null_t.get("interpro_res_f1", 0.20),
            "motif_pr_auc": null_t.get("motif_pr_auc", 0.20),
            "position_f1": null_t.get("position_f1", 0.12),
            "n_sparse_features": null_t.get("n_sparse_features", 0),
        },
        "geom_pr_auc_threshold": geom_pr_auc_threshold,
        "methodology": {
            "classification_criteria": [
                f"Geometry PR-AUC > {geom_pr_auc_threshold} (random baseline ~0.038)",
                f"Motif PR-AUC <= {null_t.get('motif_pr_auc', 0.20):.4f} (null p95)",
                f"Position F1 <= {null_t.get('position_f1', 0.12):.4f} (null p95)",
                f"InterPro Residue F1 <= {null_t.get('interpro_res_f1', 0.20):.4f} (null p95)",
            ],
            "null_distribution_method": (
                "Null thresholds are the 95th percentile of each metric "
                f"computed from {null_t.get('n_sparse_features', '?')} features "
                "with <1% protein activation (noise floor)."
            ),
            "composite_score_formula": "PR-AUC * (1 - seq_feature_fraction) * sqrt(concordance_F1)",
            "cross_family_criteria": (
                "InterPro best protein-level F1 in [0.3, 0.7] "
                "AND >=2 families with F1 > 0.3"
            ),
        },
    }

    # ── Structural category breakdown ──────────────────────────────

    from collections import Counter
    struct_cats_all = Counter(f["structural_category"] for f in features_detail)
    struct_cats_cross = Counter(f["structural_category"] for f in cross_family)

    output = {
        "global_stats": global_stats,
        "structural_categories_all": dict(struct_cats_all.most_common()),
        "structural_categories_cross_family": dict(struct_cats_cross.most_common()),
        "features": features_detail,
    }

    out_path = data_dir / "cross_family_geometry.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path} ({len(features_detail)} features, {len(cross_family)} cross-family)")


if __name__ == "__main__":
    main()
