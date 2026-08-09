#!/usr/bin/env python3
"""Build sub-domain geometric decomposition case study.

Identifies **residue-level** DB annotations (InterPro and CATH) where multiple
geometry-significant SAE features fire on the same residue annotation but
capture distinct geometric sub-structures. The automated analog of Figure 3
in the paper: the SAE decomposes a single residue-level annotation into
fine-grained structural components.

Residue-level (not protein-level) is the right grain: protein-level annotations
just say which proteins a feature touches. Residue-level annotations say which
residues inside those proteins are labelled — and that's the level we're
arguing geometry further decomposes.

Gating (all q-based, no p95):
  * Feature must be geometry-significant (geometry_prauc_padj < 0.05) — there's
    a detectable geometric signature.
  * Feature must be sparse: pct_proteins_activated <= --max-pct-activated
    (default 20%). Dense features wash out the "subdomain" story.
  * For InterPro-residue grouping: feature must also be InterPro-residue
    significant (interpro_res_f1_padj < 0.05) AND have a top residue-level
    annotation code.
  * For CATH-residue grouping: feature must also be CATH-residue significant
    (cath_res_f1_padj < 0.05) AND have a top residue-level CATH label.

Other methods (MEME, position) may still be significant — that's fine. The
viz labels per-group whether geometry / MEME / position distinguishes the
members along each axis.

Reads from:
  - {data_dir}/geometry_primary_analysis.json
  - {data_dir}/survey_coverage.json
  - {data_dir}/interpro_enrichment/*.json
  - {data_dir}/geometry_enrichment/*.json
  - {data_dir}/motif_pwm_enrichment/*.json
  - {data_dir}/cath_enrichment/*.json

Writes:
  - {data_dir}/subdomain_case_study.json

Usage::

    python scripts/build_subdomain_case_study.py --data-dir <analysis_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

Q_SIG = 0.05


def _fixed_qvalues(data_dir: Path, metric: str) -> dict[str, float]:
    """BH-correct one fixed-score metric directly from raw permutation files."""
    raw: dict[int, float] = {}
    for path in sorted((data_dir / "permutation_null").glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            value = payload["p_values"].get(metric)
            if value is not None:
                raw[int(payload["feature_id"])] = float(value)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if not raw:
        raise SystemExit(f"No raw permutation p-values found for {metric}")
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 1.0
    for index in range(len(ordered) - 1, -1, -1):
        fid = ordered[index]
        running = min(running, raw[fid] * len(ordered) / (index + 1), 1.0)
        adjusted[str(fid)] = running
    return adjusted


def _is_sig(info: dict, padj_key: str) -> bool:
    q = info.get(padj_key)
    return q is not None and q < Q_SIG


def _best_cath_residue(cath_summary: dict) -> tuple[str, str, float]:
    """Return (label, description, f1) for the deepest CATH level with a nonzero residue F1.

    We prefer CATH (4-part e.g. "1.10.760.10") over CAT over CA over C so the
    grouping buckets stay tight. The C level alone lumps thousands of features
    together, which is too coarse for a case study.
    """
    for level in ("CATH", "CAT", "CA", "C"):
        block = cath_summary.get(level) or {}
        f1 = block.get("top_residue_f1") or 0
        if f1 > 0:
            label = block.get("top_residue_label") or ""
            desc = block.get("top_residue_description") or label
            return label, desc, f1
    return "", "", 0.0


def _cosine_similarity_matrix(vectors: list[dict], all_feats: list[str]) -> np.ndarray:
    """Pairwise cosine similarity over 44-d geometry importance vectors."""
    if not vectors:
        return np.zeros((0, 0))
    mat = np.array(
        [[v.get(f, 0.0) for f in all_feats] for v in vectors], dtype=float
    )
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = mat / norms
    return normed @ normed.T


def _mean_offdiagonal(matrix: np.ndarray) -> float:
    n = matrix.shape[0]
    if n < 2:
        return 1.0
    mask = ~np.eye(n, dtype=bool)
    return float(matrix[mask].mean())


def _build_group_record(code: str, name: str, feats: list[dict], max_feats: int) -> dict:
    feats.sort(key=lambda x: -(x.get("composite_score") or 0))
    # Distinguishability metrics use the FULL feature set; only the `features`
    # array is clipped to keep the payload manageable.
    categories = sorted({f["structural_category"] for f in feats if f.get("structural_category")})
    top_features = sorted({f["top_geometric_feature"] for f in feats if f.get("top_geometric_feature")})

    # Cosine similarity of 44-d importance vectors — computed over the FULL
    # feature list so the number is representative even when we clip display.
    all_feat_names = sorted({
        name for f in feats for name in (f.get("feature_importances") or {}).keys()
    })
    vectors = [f.get("feature_importances") or {} for f in feats]
    sim_full = _cosine_similarity_matrix(vectors, all_feat_names)
    mean_cos = round(_mean_offdiagonal(sim_full), 4)

    shown = feats[:max_feats]
    # Compact cosine matrix across shown members only (NxN, rounded).
    shown_indices = list(range(len(shown)))
    cos_matrix = [
        [round(float(sim_full[i, j]), 4) for j in shown_indices]
        for i in shown_indices
    ]

    # Strip the heavy feature_importances dict from each record before emitting
    # — top_importances covers display needs and the full vectors balloon JSON.
    shown_clean = [{k: v for k, v in f.items() if k != "feature_importances"} for f in shown]

    return {
        "annotation_code": code,
        "annotation_name": name,
        "n_features": len(feats),
        "n_features_shown": len(shown),
        "n_distinct_categories": len(categories),
        "distinct_categories": categories,
        "n_distinct_top_features": len(top_features),
        "mean_geom_pr_auc": round(
            sum((f.get("geom_pr_auc") or 0) for f in feats) / len(feats), 4
        ),
        "mean_residue_f1": round(
            sum((f.get("residue_f1") or 0) for f in feats) / len(feats), 4
        ),
        "mean_cosine_similarity": mean_cos,
        "cosine_matrix": cos_matrix,
        "features": shown_clean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("feature_data_cluster"),
        help="Pipeline output directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: DATA_DIR/subdomain_case_study.json)",
    )
    parser.add_argument(
        "--min-features", type=int, default=2,
        help="Minimum features per annotation to form a group (default: 2)",
    )
    parser.add_argument(
        "--max-pct-activated", type=float, default=20.0,
        help="Sparsity filter: drop features whose pct_proteins_activated exceeds "
             "this threshold. Default: 20.0 (matches the old /case-studies page).",
    )
    parser.add_argument(
        "--max-groups-per-source", type=int, default=100,
        help="Cap on groups rendered per DB source (InterPro-res and CATH-res each), "
             "sorted by n_features descending. Default: 100",
    )
    parser.add_argument(
        "--max-features-per-group", type=int, default=20,
        help="Cap on features shown per group (top-N by composite_score). "
             "Default: 20. n_features retains the true group size.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    gpa_path = data_dir / "geometry_primary_analysis.json"
    if not gpa_path.exists():
        print("ERROR: geometry_primary_analysis.json not found.")
        sys.exit(1)
    gpa = json.loads(gpa_path.read_text())
    fixed_geometry_q = _fixed_qvalues(data_dir, "geometry_prauc")
    fixed_interpro_q = _fixed_qvalues(data_dir, "interpro_res_f1")
    for fid, info in gpa.get("features", {}).items():
        # Explicitly override any cached/mixed q-value fields. Table 3 uses
        # the paper-primary fixed-score permutation estimand.
        info["geometry_prauc_padj"] = fixed_geometry_q.get(fid)
        info["interpro_res_f1_padj"] = fixed_interpro_q.get(fid)
    # Gate: geometry is significant. Broader than "geometry-primary" — other
    # methods may also be significant; the viz separates the axes.
    gp_features = {k: v for k, v in gpa["features"].items() if _is_sig(v, "geometry_prauc_padj")}

    # Sparsity filter — read coverage and drop features exceeding the threshold.
    coverage_path = data_dir / "survey_coverage.json"
    coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
    if not coverage:
        print("WARNING: survey_coverage.json not found; sparsity filter inactive")

    n_before_sparse = len(gp_features)
    pct_by_fid: dict[str, float] = {}
    if coverage:
        kept: dict[str, dict] = {}
        for fid, gp in gp_features.items():
            pct = (coverage.get(fid) or {}).get("pct_proteins_activated")
            if pct is None:
                continue
            pct_by_fid[fid] = pct
            if pct <= args.max_pct_activated:
                kept[fid] = gp
        gp_features = kept
    print(
        f"Loaded {n_before_sparse} geometry-significant features "
        f"(geometry_prauc_padj < {Q_SIG}); "
        f"after sparsity filter (≤{args.max_pct_activated:.1f}% protein coverage): {len(gp_features)}"
    )

    ipro_dir = data_dir / "interpro_enrichment"
    geo_dir = data_dir / "geometry_enrichment"
    motif_dir = data_dir / "motif_pwm_enrichment"
    cath_dir = data_dir / "cath_enrichment"

    ipro_code_to_feats: dict[str, list[dict]] = defaultdict(list)
    ipro_meta: dict[str, dict] = {}
    cath_code_to_feats: dict[str, list[dict]] = defaultdict(list)
    cath_meta: dict[str, dict] = {}

    n_with_ipro_res = 0
    n_with_cath_res = 0

    for fid, gp in gp_features.items():
        fid_int = int(fid)
        padded = f"{fid_int:04d}"

        # InterPro residue-level annotation (only grouping signal that matters here)
        ipro_res_code = ""
        ipro_res_name = ""
        ipro_res_f1 = 0.0
        ipro_path = ipro_dir / f"{padded}.json"
        if ipro_path.exists():
            ipro = json.loads(ipro_path.read_text())
            res_entries = ipro.get("residue_level") or []
            if res_entries:
                best_res = max(res_entries, key=lambda x: x.get("best_f1", 0) or 0)
                if best_res.get("best_f1"):
                    ipro_res_f1 = best_res.get("best_f1") or 0
                    ipro_res_code = best_res.get("annotation_code", "") or ""
                    ipro_res_name = best_res.get("annotation_name", "") or ipro_res_code

        # CATH residue-level annotation (deepest hierarchy with a residue hit)
        cath_res_label = ""
        cath_res_desc = ""
        cath_res_f1 = 0.0
        cath_path = cath_dir / f"{padded}.json"
        if cath_path.exists():
            cath_data = json.loads(cath_path.read_text())
            summary = cath_data.get("summary", {}) or {}
            cath_res_label, cath_res_desc, cath_res_f1 = _best_cath_residue(summary)

        # Geometry importances + motif superposition
        importances: dict[str, float] = {}
        ms: dict = {}
        geo_path = geo_dir / f"{padded}.json"
        if geo_path.exists():
            geo = json.loads(geo_path.read_text())
            rl = geo.get("geometric_residue_level", {}) or {}
            importances = rl.get("feature_importances", {}) or {}
            ms = rl.get("motif_superposition", {}) or {}

        motif_path = motif_dir / f"{padded}.json"
        top_motifs = []
        if motif_path.exists():
            top_motifs = (json.loads(motif_path.read_text()).get("motifs")) or []

        sorted_imps = sorted(importances.items(), key=lambda x: -x[1])[:5]
        motif_len = len(ms.get("per_position_flexibility", []))
        mean_rmsd = ms.get("mean_rmsd")
        rmsd_per_pos = (mean_rmsd / motif_len) if (mean_rmsd and motif_len > 0) else None

        feat_record = {
            "feature_id": fid_int,
            "composite_score": gp.get("composite_score"),
            "structural_category": gp.get("structural_category"),
            "top_geometric_feature": gp.get("top_geometric_feature"),
            "geom_pr_auc": gp.get("geom_pr_auc"),
            "concordance_f1": gp.get("concordance_f1"),
            "pct_proteins_activated": round(pct_by_fid.get(fid, 0.0), 3) if pct_by_fid else None,
            # Residue-level DB labels (what we group on)
            "interpro_residue_code": ipro_res_code or None,
            "interpro_residue_name": ipro_res_name or None,
            "interpro_residue_f1": round(ipro_res_f1, 4),
            "cath_residue_label": cath_res_label or None,
            "cath_residue_description": cath_res_desc or None,
            "cath_residue_f1": round(cath_res_f1, 4),
            # Sequence-side metrics (used by viz for the distinguishability axes)
            "motif_pr_auc": gp.get("motif_pr_auc"),
            "position_f1": gp.get("position_f1"),
            "seq_feature_fraction": gp.get("seq_feature_fraction"),
            # q-values
            "q_geometry_prauc": gp.get("geometry_prauc_padj"),
            "q_interpro_res_f1": gp.get("interpro_res_f1_padj"),
            "q_cath_res_f1": gp.get("cath_res_f1_padj"),
            "q_motif_pr_auc": gp.get("motif_pr_auc_padj"),
            "q_position_f1": gp.get("position_f1_padj"),
            # Geometric display bits
            "top_importances": sorted_imps,
            "feature_importances": importances,
            "motif_rmsd": round(mean_rmsd, 3) if mean_rmsd is not None else None,
            "motif_rmsd_per_pos": round(rmsd_per_pos, 4) if rmsd_per_pos else None,
            "motif_n_fragments": ms.get("n_fragments", 0),
            "motif_length": motif_len,
            "best_seq_motif": top_motifs[0].get("consensus") if top_motifs else None,
        }

        # InterPro-residue grouping: require the residue-level InterPro to also be significant.
        if ipro_res_code and _is_sig(gp, "interpro_res_f1_padj"):
            n_with_ipro_res += 1
            # Copy per-group "residue_f1" so group aggregates can use one field regardless of source
            rec = dict(feat_record, residue_f1=ipro_res_f1)
            ipro_code_to_feats[ipro_res_code].append(rec)
            ipro_meta[ipro_res_code] = {"name": ipro_res_name or ipro_res_code}

        # CATH-residue grouping: require cath_res_f1_padj < 0.05 too.
        if cath_res_label and _is_sig(gp, "cath_res_f1_padj"):
            n_with_cath_res += 1
            rec = dict(feat_record, residue_f1=cath_res_f1)
            cath_code_to_feats[cath_res_label].append(rec)
            cath_meta[cath_res_label] = {"name": cath_res_desc or cath_res_label}

    print(
        f"InterPro-residue significant: {n_with_ipro_res};  "
        f"CATH-residue significant: {n_with_cath_res}"
    )

    def build_groups(code_to_feats, meta):
        out = []
        for code, feats in code_to_feats.items():
            if len(feats) < args.min_features:
                continue
            out.append(_build_group_record(
                code, meta[code]["name"], feats, args.max_features_per_group,
            ))
        out.sort(key=lambda g: (-g["n_features"], -g["mean_geom_pr_auc"]))
        return out

    all_interpro_groups = build_groups(ipro_code_to_feats, ipro_meta)
    all_cath_groups = build_groups(cath_code_to_feats, cath_meta)
    interpro_groups = all_interpro_groups[: args.max_groups_per_source]
    cath_groups = all_cath_groups[: args.max_groups_per_source]

    featured_ids = set()
    for g in interpro_groups:
        featured_ids.update(f["feature_id"] for f in g["features"])
    for g in cath_groups:
        featured_ids.update(f["feature_id"] for f in g["features"])

    global_stats = {
        "total_geometry_significant": n_before_sparse,
        "total_after_sparsity_filter": len(gp_features),
        "max_pct_activated": args.max_pct_activated,
        "n_interpro_residue_sig": n_with_ipro_res,
        "n_cath_residue_sig": n_with_cath_res,
        "n_interpro_groups": len(all_interpro_groups),
        "n_interpro_groups_shown": len(interpro_groups),
        "n_interpro_features_in_groups": sum(g["n_features"] for g in all_interpro_groups),
        "n_cath_groups": len(all_cath_groups),
        "n_cath_groups_shown": len(cath_groups),
        "n_cath_features_in_groups": sum(g["n_features"] for g in all_cath_groups),
        "n_unique_features_in_display_payload": len(featured_ids),
        "grouping_level": "residue",
        "q_gate": Q_SIG,
        "max_groups_per_source": args.max_groups_per_source,
        "max_features_per_group": args.max_features_per_group,
    }

    output = {
        "global_stats": global_stats,
        "table3": {
            "unit": "eligible InterPro residue annotation group",
            "cosine_similarity_threshold": 0.5,
            "n_interpro_groups": len(all_interpro_groups),
            "n_geom_distinguishable": sum(
                g["mean_cosine_similarity"] < 0.5 for g in all_interpro_groups
            ),
            "pct_geom_distinguishable": round(
                100
                * sum(g["mean_cosine_similarity"] < 0.5 for g in all_interpro_groups)
                / max(len(all_interpro_groups), 1),
                2,
            ),
            "presentation_cap_applied": False,
            "q_source": "fixed_score_permutation_raw_p",
            "n_geometry_q_tested": len(fixed_geometry_q),
            "n_interpro_residue_q_tested": len(fixed_interpro_q),
        },
        "interpro_groups": interpro_groups,
        "cath_groups": cath_groups,
        # Back-compat alias for older JS consumers that expect `groups`
        "groups": interpro_groups,
    }

    out_path = args.output or (data_dir / "subdomain_case_study.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(
        f"Wrote {out_path}: {len(all_interpro_groups)} InterPro-residue groups "
        f"({len(interpro_groups)} shown; "
        f"{global_stats['n_interpro_features_in_groups']} feature memberships), "
        f"{len(all_cath_groups)} CATH-residue groups "
        f"({len(cath_groups)} shown; "
        f"{global_stats['n_cath_features_in_groups']} feature memberships)"
    )

    print("\nTop InterPro-residue groups:")
    for g in interpro_groups[:5]:
        cats = ", ".join(g["distinct_categories"][:3])
        print(
            f"  {g['annotation_code']} ({g['annotation_name'][:50]}): "
            f"{g['n_features']} features, {g['n_distinct_categories']} categories [{cats}]"
        )
    print("\nTop CATH-residue groups:")
    for g in cath_groups[:5]:
        cats = ", ".join(g["distinct_categories"][:3])
        print(
            f"  {g['annotation_code']} ({g['annotation_name'][:50]}): "
            f"{g['n_features']} features, {g['n_distinct_categories']} categories [{cats}]"
        )


if __name__ == "__main__":
    main()
