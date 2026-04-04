"""
Build case study families JSON for the visualizer.

Finds groups of SAE nodes that share the same best residue-level InterPro
annotation AND have geometry PR-AUC above null. Identifies families where
geometry varies across nodes (different top geometric features), showing
that geometry is more granular than InterPro annotations.

Usage:
    python scripts/build_case_studies.py --data-dir feature_data_cluster
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_case_studies(data_dir: Path) -> dict:
    """Build case study families from pre-computed enrichment data."""
    # Load geometry primary analysis for null thresholds and per-feature metrics
    gpa_path = data_dir / "geometry_primary_analysis.json"
    with open(gpa_path) as f:
        gpa = json.load(f)

    null_interpro = gpa["null_thresholds"]["interpro_res_f1"]
    null_geom = gpa["geom_pr_auc_threshold"]
    max_pct_activated = 20.0  # exclude dense features (>20% proteins activated)

    logger.info(
        "Thresholds: interpro_res_f1 > %.4f, geom_pr_auc > %.4f, pct_activated <= %.1f%%",
        null_interpro,
        null_geom,
        max_pct_activated,
    )

    # Load coverage data to filter out dense features
    with open(data_dir / "survey_coverage.json") as f:
        coverage = json.load(f)

    # Find features where BOTH metrics exceed null AND feature is sparse
    qualifying = {}
    n_dense_excluded = 0
    for fid, info in gpa["features"].items():
        if info["interpro_res_f1"] > null_interpro and info["geom_pr_auc"] > null_geom:
            cov = coverage.get(fid, {})
            pct = cov.get("pct_proteins_activated", 100.0)
            if pct > max_pct_activated:
                n_dense_excluded += 1
                continue
            qualifying[fid] = info
            qualifying[fid]["pct_proteins_activated"] = pct

    logger.info(
        "Features with both above null: %d (excluded %d dense features >%.0f%%)",
        len(qualifying),
        n_dense_excluded,
        max_pct_activated,
    )

    # Load interpro enrichment for each qualifying feature to get best residue annotation
    families_raw: dict[str, list] = defaultdict(list)
    for fid in qualifying:
        ip_path = data_dir / "interpro_enrichment" / f"{int(fid):04d}.json"
        if not ip_path.exists():
            continue
        with open(ip_path) as f:
            ip = json.load(f)
        residue_entries = ip.get("residue_level", [])
        if not residue_entries:
            continue
        best = max(residue_entries, key=lambda e: e.get("best_f1", 0))
        families_raw[best["annotation_code"]].append({
            "feature_id": int(fid),
            "annotation_code": best["annotation_code"],
            "annotation_name": best["annotation_name"],
            "interpro_res_f1": best["best_f1"],
            "interpro_gpa_f1": qualifying[fid]["interpro_res_f1"],
            "geom_pr_auc": qualifying[fid]["geom_pr_auc"],
            "pct_proteins_activated": qualifying[fid]["pct_proteins_activated"],
            "top_geometric_feature": qualifying[fid]["top_geometric_feature"],
            "structural_category": qualifying[fid]["structural_category"],
            "is_geometry_primary": qualifying[fid].get("is_geometry_primary", False),
        })

    # Filter to families with 2+ members
    multi_families = {k: v for k, v in families_raw.items() if len(v) >= 2}
    logger.info("Families with 2+ members: %d", len(multi_families))

    # Load geometry feature importances for each member
    all_geom_feature_set: set[str] = set()
    for code, members in multi_families.items():
        for m in members:
            geom_path = data_dir / "geometry_enrichment" / f"{m['feature_id']:04d}.json"
            if not geom_path.exists():
                m["feature_importances"] = {}
                continue
            with open(geom_path) as f:
                geom = json.load(f)
            grl = geom.get("geometric_residue_level", {})
            fi = grl.get("feature_importances", {})
            m["feature_importances"] = fi
            all_geom_feature_set.update(fi.keys())
            # Also grab concordance and decision tree rules
            conc = grl.get("concordance", {})
            m["concordance_f1"] = conc.get("f1")
            m["concordance_prauc"] = conc.get("avg_precision")
            m["rules"] = grl.get("rules", "")

    all_geom_feature_names = sorted(all_geom_feature_set)

    # Compute geometry diversity metrics for each family
    families_out = []
    for code, members in sorted(multi_families.items(), key=lambda x: -len(x[1])):
        name = members[0]["annotation_name"]

        # Compute pairwise cosine similarity of feature importance vectors
        vecs = []
        for m in members:
            vec = np.array([m["feature_importances"].get(fn, 0.0) for fn in all_geom_feature_names])
            vecs.append(vec)
        vecs = np.array(vecs)

        # Cosine similarity matrix
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = vecs / norms
        cos_sim = normed @ normed.T

        # Mean pairwise cosine similarity (excluding diagonal)
        n = len(members)
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            mean_cos_sim = float(cos_sim[mask].mean())
        else:
            mean_cos_sim = 1.0

        # Count unique top geometric features
        top_geom_features = set(m["top_geometric_feature"] for m in members)
        geom_diverse = len(top_geom_features) > 1

        families_out.append({
            "annotation_code": code,
            "annotation_name": name,
            "n_nodes": len(members),
            "geom_diverse": geom_diverse,
            "n_unique_top_geom": len(top_geom_features),
            "mean_cosine_similarity": round(mean_cos_sim, 4),
            "members": sorted(members, key=lambda m: -m["geom_pr_auc"]),
        })

    # Sort families: geometry-diverse first, then by number of nodes
    families_out.sort(key=lambda f: (-int(f["geom_diverse"]), -f["n_nodes"], f["mean_cosine_similarity"]))

    result = {
        "null_thresholds": {
            "interpro_res_f1": null_interpro,
            "geom_pr_auc": null_geom,
            "max_pct_activated": max_pct_activated,
        },
        "n_qualifying_features": len(qualifying),
        "n_families": len(families_out),
        "geometry_feature_names": all_geom_feature_names,
        "families": families_out,
    }

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Build case study families JSON")
    parser.add_argument("--data-dir", required=True, help="Path to feature data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    result = build_case_studies(data_dir)

    out_path = data_dir / "case_study_families.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Saved %d families to %s", result["n_families"], out_path)
    for fam in result["families"][:10]:
        logger.info(
            "  %s (%s): %d nodes, diverse=%s, cos_sim=%.3f",
            fam["annotation_code"],
            fam["annotation_name"][:40],
            fam["n_nodes"],
            fam["geom_diverse"],
            fam["mean_cosine_similarity"],
        )


if __name__ == "__main__":
    main()
