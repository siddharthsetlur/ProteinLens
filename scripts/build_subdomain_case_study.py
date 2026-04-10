#!/usr/bin/env python3
"""Build sub-domain geometric decomposition case study.

Identifies InterPro annotations where multiple geometry-primary SAE features
fire on the same protein family but capture distinct geometric sub-structures.
This is the automated analog of Silberg et al. 2025 Figure 5: the SAE
decomposes a single domain annotation into fine-grained structural components.

Reads from:
  - {data_dir}/geometry_primary_analysis.json
  - {data_dir}/interpro_enrichment/*.json
  - {data_dir}/geometry_enrichment/*.json
  - {data_dir}/motif_enrichment/*.json
  - {data_dir}/cath_enrichment/*.json

Writes:
  - {data_dir}/subdomain_case_study.json

Usage::

    python scripts/build_subdomain_case_study.py --data-dir feature_data_cluster
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("feature_data_cluster"),
        help="Pipeline output directory",
    )
    parser.add_argument(
        "--min-protein-f1", type=float, default=0.6,
        help="Minimum InterPro protein-level F1 to include a GP feature (default: 0.6)",
    )
    parser.add_argument(
        "--min-features", type=int, default=2,
        help="Minimum GP features per annotation to form a group (default: 2)",
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    # ── Load geometry-primary analysis ──
    gpa_path = data_dir / "geometry_primary_analysis.json"
    if not gpa_path.exists():
        print("ERROR: geometry_primary_analysis.json not found.")
        sys.exit(1)
    gpa = json.loads(gpa_path.read_text())
    gp_features = {k: v for k, v in gpa["features"].items() if v.get("is_geometry_primary")}
    null_thresholds = gpa.get("null_thresholds", {})

    print(f"Loaded {len(gp_features)} geometry-primary features")

    # ── For each GP feature, load InterPro protein-level best annotation ──
    ipro_dir = data_dir / "interpro_enrichment"
    geo_dir = data_dir / "geometry_enrichment"
    motif_dir = data_dir / "motif_enrichment"
    cath_dir = data_dir / "cath_enrichment"

    # Map: annotation_code -> list of feature records
    annotation_to_feats: dict[str, list[dict]] = defaultdict(list)
    annotation_meta: dict[str, dict] = {}  # code -> {name, type}

    for fid, gp in gp_features.items():
        fid_int = int(fid)
        padded = f"{fid_int:04d}"

        # Load InterPro
        ipro_path = ipro_dir / f"{padded}.json"
        if not ipro_path.exists():
            continue
        ipro = json.loads(ipro_path.read_text())
        prot_entries = ipro.get("protein_level", [])
        if not prot_entries:
            continue

        best_prot = max(prot_entries, key=lambda x: x.get("best_f1", 0))
        prot_f1 = best_prot.get("best_f1", 0)
        if prot_f1 < args.min_protein_f1:
            continue

        code = best_prot.get("annotation_code", "")
        name = best_prot.get("annotation_name", "")
        if not code:
            continue

        annotation_meta[code] = {"name": name}

        # Load geometry enrichment for motif superposition
        geo_path = geo_dir / f"{padded}.json"
        geo = json.loads(geo_path.read_text()) if geo_path.exists() else {}
        rl = geo.get("geometric_residue_level", {})
        ms = rl.get("motif_superposition", {})
        concordance = rl.get("concordance", {})
        importances = rl.get("feature_importances", {})

        # Load motif enrichment
        motif_path = motif_dir / f"{padded}.json"
        motif_data = json.loads(motif_path.read_text()) if motif_path.exists() else {}
        top_motifs = motif_data.get("top_motifs", [])

        # Load CATH enrichment
        cath_path = cath_dir / f"{padded}.json"
        cath_data = json.loads(cath_path.read_text()) if cath_path.exists() else {}
        cath_summary = cath_data.get("summary", {})
        best_cath_f1 = 0
        best_cath_label = ""
        for level in ("C", "CA", "CAT", "CATH"):
            cf1 = (cath_summary.get(level) or {}).get("top_residue_f1") or 0
            if cf1 > best_cath_f1:
                best_cath_f1 = cf1
                best_cath_label = (cath_summary.get(level) or {}).get("top_residue_label", "")

        # Top 3 geometric feature importances
        sorted_imps = sorted(importances.items(), key=lambda x: -x[1])[:5]

        # Motif RMSD
        motif_len = len(ms.get("per_position_flexibility", []))
        mean_rmsd = ms.get("mean_rmsd")
        rmsd_per_pos = (mean_rmsd / motif_len) if mean_rmsd and motif_len > 0 else None

        feat_record = {
            "feature_id": fid_int,
            "composite_score": gp["composite_score"],
            "structural_category": gp["structural_category"],
            "top_geometric_feature": gp["top_geometric_feature"],
            "geom_pr_auc": gp["geom_pr_auc"],
            "concordance_f1": gp["concordance_f1"],
            "interpro_protein_f1": round(prot_f1, 4),
            "interpro_residue_f1": gp["interpro_res_f1"],
            "cath_residue_f1": round(best_cath_f1, 4),
            "cath_best_label": best_cath_label,
            "motif_f1": gp["motif_f1"],
            "position_f1": gp["position_f1"],
            "seq_feature_fraction": gp["seq_feature_fraction"],
            "top_importances": sorted_imps,
            "motif_rmsd": mean_rmsd,
            "motif_rmsd_per_pos": round(rmsd_per_pos, 4) if rmsd_per_pos else None,
            "motif_n_fragments": ms.get("n_fragments", 0),
            "motif_length": motif_len,
            "best_seq_motif": top_motifs[0]["motif"] if top_motifs else None,
        }
        annotation_to_feats[code].append(feat_record)

    # ── Filter to annotations with 2+ GP features ──
    groups = []
    for code, feats in annotation_to_feats.items():
        if len(feats) < args.min_features:
            continue

        feats.sort(key=lambda x: -x["composite_score"])
        meta = annotation_meta[code]

        # Count distinct structural categories
        categories = set(f["structural_category"] for f in feats)
        top_features = set(f["top_geometric_feature"] for f in feats)

        groups.append({
            "annotation_code": code,
            "annotation_name": meta["name"],
            "n_features": len(feats),
            "n_distinct_categories": len(categories),
            "distinct_categories": sorted(categories),
            "n_distinct_top_features": len(top_features),
            "mean_geom_pr_auc": round(
                sum(f["geom_pr_auc"] for f in feats) / len(feats), 4
            ),
            "mean_interpro_protein_f1": round(
                sum(f["interpro_protein_f1"] for f in feats) / len(feats), 4
            ),
            "max_interpro_residue_f1": round(
                max(f["interpro_residue_f1"] for f in feats), 4
            ),
            "features": feats,
        })

    # Sort by number of features (biggest decompositions first)
    groups.sort(key=lambda g: (-g["n_features"], -g["mean_geom_pr_auc"]))

    # ── Global stats ──
    total_gp = len(gp_features)
    n_with_high_prot_f1 = sum(
        1 for feats in annotation_to_feats.values()
        for _ in feats
    )
    total_features_in_groups = sum(g["n_features"] for g in groups)

    global_stats = {
        "total_geometry_primary": total_gp,
        "n_with_high_interpro_protein_f1": n_with_high_prot_f1,
        "pct_with_high_interpro_protein_f1": round(
            100 * n_with_high_prot_f1 / max(total_gp, 1), 1
        ),
        "min_protein_f1_threshold": args.min_protein_f1,
        "n_annotations_with_multiple_features": len(groups),
        "n_features_in_groups": total_features_in_groups,
        "n_unique_annotations": len(annotation_to_feats),
        "null_thresholds": null_thresholds,
        "geom_pr_auc_threshold": gpa.get("geom_pr_auc_threshold", 0.3),
    }

    output = {
        "global_stats": global_stats,
        "groups": groups,
    }

    out_path = data_dir / "subdomain_case_study.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path} ({len(groups)} annotation groups, "
          f"{total_features_in_groups} features)")

    for g in groups[:10]:
        cats = ", ".join(g["distinct_categories"][:3])
        print(f"  {g['annotation_code']} ({g['annotation_name'][:50]}): "
              f"{g['n_features']} features, {g['n_distinct_categories']} categories "
              f"[{cats}]")


if __name__ == "__main__":
    main()
