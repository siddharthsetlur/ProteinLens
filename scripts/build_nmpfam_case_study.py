#!/usr/bin/env python3
"""Build the NMPFams case study JSON for the visualizer.

Identifies SAE features in the triple intersection: sparse (<10% coverage),
geometry-primary (confound-filtered), and activated by novel metagenomic
proteins from NMPFams. These features detect genuine structural motifs that
generalize beyond SwissProt to the metagenomic "dark matter."

Reads from:
  - {data_dir}/nmpfam/nmpfam_enrichment/*.json
  - {data_dir}/geometry_primary_analysis.json
  - {data_dir}/survey_coverage.json
  - {data_dir}/geometry_enrichment/*.json
  - {data_dir}/interpro_enrichment/*.json
  - {data_dir}/feature_max_activations.npy

Writes:
  - {data_dir}/nmpfam_case_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

Q_SIG = 0.05


def _is_sig(info: dict, padj_key: str) -> bool:
    q = info.get(padj_key)
    return q is not None and q < Q_SIG


def _is_q_geometry_primary(info: dict) -> bool:
    """q-value analogue of is_geometry_primary."""
    if not _is_sig(info, "geometry_prauc_padj"):
        return False
    for other in ("interpro_res_f1_padj", "cath_res_f1_padj",
                  "motif_pr_auc_padj", "position_f1_padj"):
        if _is_sig(info, other):
            return False
    return True


def build_nmpfam_case_study(data_dir: Path) -> dict:
    """Build the NMPFams case study JSON."""
    nmpfam_dir = data_dir / "nmpfam"
    enrichment_dir = nmpfam_dir / "nmpfam_enrichment"

    if not enrichment_dir.exists():
        print("ERROR: NMPFams enrichment not found. Run run_nmpfam_analysis.py first.")
        return {}

    # Load NMPFams family metadata
    families_meta = {}
    families_path = nmpfam_dir / "families.json"
    if families_path.exists():
        for fam in json.load(open(families_path)):
            families_meta[fam["ID"]] = fam

    # Scan NMPFams enrichment file names to get feature IDs with hits
    # (do NOT load content yet — 7.8GB total)
    nmpfam_fids = set()
    for fp in enrichment_dir.glob("*.json"):
        try:
            nmpfam_fids.add(int(fp.stem))
        except ValueError:
            pass
    print(f"Features with NMPFams hits: {len(nmpfam_fids)}")

    # Load geometry-primary features
    gp_path = data_dir / "geometry_primary_analysis.json"
    if not gp_path.exists():
        print("ERROR: geometry_primary_analysis.json not found.")
        return {}
    gp_data = json.load(open(gp_path))
    gp_features = gp_data.get("features", {})
    geom_primary_ids = {int(k) for k, v in gp_features.items() if _is_q_geometry_primary(v)}

    # Load coverage (sparsity)
    coverage_data = json.load(open(data_dir / "survey_coverage.json"))
    sparse_ids = set()
    for fid_str, cov in coverage_data.items():
        pct = cov.get("pct_proteins_activated", 100)
        if pct is not None and pct < 10:
            sparse_ids.add(int(fid_str))

    # Load global max activations
    global_max = np.load(data_dir / "feature_max_activations.npy")

    # Triple intersection
    triple = sparse_ids & geom_primary_ids & nmpfam_fids
    print(f"Triple intersection: {len(triple)} features")

    # Also collect broader stats for context
    gp_with_nmpfam = geom_primary_ids & nmpfam_fids
    sparse_with_nmpfam = sparse_ids & nmpfam_fids

    # Now load ONLY the enrichment JSONs we need (triple + broader GP)
    fids_to_load = triple | gp_with_nmpfam
    nmpfam_data = {}
    for fid in fids_to_load:
        fp = enrichment_dir / f"{fid:04d}.json"
        if fp.exists():
            nmpfam_data[fid] = json.load(open(fp))
    print(f"Loaded {len(nmpfam_data)} enrichment JSONs (of {len(nmpfam_fids)} total)")

    # Build per-feature entries for the triple intersection
    feature_entries = []
    for fid in sorted(triple, key=lambda f: nmpfam_data[f]["nmpfam_hits"][0]["normalized_activation"], reverse=True):
        nd = nmpfam_data[fid]
        gp = gp_features[str(fid)]
        cov = coverage_data.get(str(fid), {})

        # Load geometry enrichment for this feature
        geom_info = {}
        geom_path = data_dir / "geometry_enrichment" / f"{fid:04d}.json"
        if geom_path.exists():
            gd = json.load(open(geom_path))
            res = gd.get("geometric_residue_level", {})
            geom_info = {
                "gbm_auc_cv": res.get("gbm_auc_cv"),
                "tree_f1_cv": res.get("tree_f1_cv"),
                "feature_importances": res.get("feature_importances", {}),
                "rules": res.get("rules", ""),
                "concordance": res.get("concordance", {}),
                "motif_superposition": res.get("motif_superposition", {}),
            }

        # Load InterPro enrichment
        interpro_info = {}
        interpro_path = data_dir / "interpro_enrichment" / f"{fid:04d}.json"
        if interpro_path.exists():
            ipd = json.load(open(interpro_path))
            protein_level = ipd.get("protein_level", [])
            residue_level = ipd.get("residue_level", [])
            if protein_level:
                best_prot = max(protein_level, key=lambda x: x.get("best_f1", 0))
                interpro_info["protein_best_f1"] = best_prot.get("best_f1")
                interpro_info["protein_best_name"] = best_prot.get("annotation_name")
            if residue_level:
                best_res = max(residue_level, key=lambda x: x.get("best_f1", 0))
                interpro_info["residue_best_f1"] = best_res.get("best_f1")

        # Build NMPFams hit summaries (top 10)
        hits_summary = []
        for hit in nd["nmpfam_hits"][:10]:
            meta = families_meta.get(hit["family_id"], {})
            entry = {
                "family_id": hit["family_id"],
                "category": hit.get("category", meta.get("Category", "Unknown")),
                "sequence_count": hit.get("sequence_count", meta.get("SequenceCount", 0)),
                "max_activation": hit["max_activation"],
                "normalized_activation": hit["normalized_activation"],
                "sequence_length": hit.get("sequence_length", 0),
                "nmpfams_url": hit.get("nmpfams_url", f"https://bib.fleming.gr/NMPFamsDB/family/{hit['family_id']}"),
                "has_per_residue": hit.get("per_residue_activations") is not None,
                "has_geometry": "geometry" in hit,
            }
            hits_summary.append(entry)

        feature_entries.append({
            "feature_id": fid,
            "global_max_activation": round(float(global_max[fid]), 4),
            "coverage_pct": cov.get("pct_proteins_activated", 0),
            "n_clusters_activated": cov.get("n_clusters_activated", 0),
            "composite_score": gp.get("composite_score", 0),
            "is_geometry_primary": True,
            "n_nmpfam_hits": nd["n_nmpfam_hits"],
            "activation_threshold": nd["activation_threshold"],
            "top_nmpfam_norm_act": hits_summary[0]["normalized_activation"] if hits_summary else 0,
            "geometry": geom_info,
            "interpro": interpro_info,
            "nmpfam_hits": hits_summary,
        })

    # Also build broader context entries (geometry-primary + NMPFams, but not necessarily sparse)
    broader_entries = []
    for fid in sorted(gp_with_nmpfam - triple, key=lambda f: nmpfam_data[f]["nmpfam_hits"][0]["normalized_activation"], reverse=True):
        nd = nmpfam_data[fid]
        gp = gp_features[str(fid)]
        cov = coverage_data.get(str(fid), {})
        broader_entries.append({
            "feature_id": fid,
            "coverage_pct": cov.get("pct_proteins_activated", 0),
            "composite_score": gp.get("composite_score", 0),
            "n_nmpfam_hits": nd["n_nmpfam_hits"],
            "top_nmpfam_norm_act": nd["nmpfam_hits"][0]["normalized_activation"] if nd["nmpfam_hits"] else 0,
        })

    # NMPFams sample statistics
    n_families_sampled = len(families_meta)
    categories = {}
    for fam in families_meta.values():
        cat = fam.get("Category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1

    # Activation distribution across loaded features (GP subset)
    all_norm_acts = []
    for fid, nd in nmpfam_data.items():
        for hit in nd["nmpfam_hits"]:
            all_norm_acts.append(hit["normalized_activation"])

    result = {
        "summary": {
            "n_families_sampled": n_families_sampled,
            "n_families_by_category": categories,
            "n_features_total": len(global_max),
            "n_features_with_nmpfam_hits": len(nmpfam_fids),
            "n_geometry_primary": len(geom_primary_ids),
            "n_geometry_primary_with_nmpfam": len(gp_with_nmpfam),
            "n_sparse": len(sparse_ids),
            "n_sparse_with_nmpfam": len(sparse_with_nmpfam),
            "n_triple_intersection": len(triple),
            "total_hit_instances": sum(nd.get("n_nmpfam_hits", 0) for nd in nmpfam_data.values()),
            "activation_distribution": {
                "gt_0_5": sum(1 for a in all_norm_acts if a > 0.5),
                "gt_0_75": sum(1 for a in all_norm_acts if a > 0.75),
                "gt_0_90": sum(1 for a in all_norm_acts if a > 0.90),
                "gt_1_0": sum(1 for a in all_norm_acts if a > 1.0),
            },
        },
        "triple_features": feature_entries,
        "broader_gp_features": broader_entries,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Build NMPFams case study JSON.")
    parser.add_argument("--data-dir", type=Path, default=Path("feature_data_cluster"))
    args = parser.parse_args()

    result = build_nmpfam_case_study(args.data_dir)
    if not result:
        return

    out_path = args.data_dir / "nmpfam_case_study.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path} ({len(result.get('triple_features', []))} triple features, "
          f"{len(result.get('broader_gp_features', []))} broader GP features)")


if __name__ == "__main__":
    main()
