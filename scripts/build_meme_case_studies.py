"""
Build MEME case study families JSON for the visualizer.

Finds groups of SAE nodes that share the same MEME/PWM motif annotation
(best consensus) AND have geometry PR-AUC above null. Identifies families
where geometry varies across nodes (different top geometric features),
showing that the same discovered sequence motif can correspond to distinct
structural roles.

Grouping: no two features typically share an identical MEME consensus, so
features are clustered via union-find on pairwise edit distance <= max_edit
of the best-motif consensus (a lightweight PWM-similarity proxy).

Usage:
    python scripts/build_meme_case_studies.py --data-dir feature_data_cluster
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

Q_SIG = 0.05
# A family is "geometry-diverse" when the mean pairwise cosine similarity of
# its members' 44-d geometric feature-importance vectors is below this.
# Matches the threshold used by the subdomain case study (proteinlens/viz/
# static/js/subdomain_detail.js:14).
GEOM_COS_THRESHOLD = 0.5


def _is_sig(info: dict, padj_key: str) -> bool:
    q = info.get(padj_key)
    return q is not None and q < Q_SIG


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    return dp[m][n]


def _best_motif(pwm_path: Path) -> dict | None:
    """Return the top MEME motif (by PR-AUC) for a feature, or None."""
    if not pwm_path.exists():
        return None
    d = json.loads(pwm_path.read_text())
    motifs = d.get("motifs", [])
    if not motifs:
        return None
    best = max(
        motifs,
        key=lambda m: (m.get("pr_auc") or {}).get("pr_auc", 0),
    )
    pr = (best.get("pr_auc") or {}).get("pr_auc", 0)
    return {
        "consensus": best.get("consensus", ""),
        "motif_id": best.get("motif_id", ""),
        "width": best.get("width"),
        "e_value": best.get("e_value"),
        "pr_auc": pr,
        "best_f1": best.get("best_f1"),
    }


def build_meme_case_studies(data_dir: Path, max_edit: int = 2) -> dict:
    """Build MEME case-study families from pre-computed enrichment data."""
    gpa_path = data_dir / "geometry_primary_analysis.json"
    with open(gpa_path) as f:
        gpa = json.load(f)

    max_pct_activated = 20.0
    logger.info(
        "Q-gate: motif_pr_auc_padj < %.2f AND geometry_prauc_padj < %.2f; pct_activated <= %.1f%%, edit_dist <= %d",
        Q_SIG, Q_SIG, max_pct_activated, max_edit,
    )

    with open(data_dir / "survey_coverage.json") as f:
        coverage = json.load(f)

    qualifying = {}
    n_dense_excluded = 0
    for fid, info in gpa["features"].items():
        if _is_sig(info, "motif_pr_auc_padj") and _is_sig(info, "geometry_prauc_padj"):
            cov = coverage.get(fid, {})
            pct = cov.get("pct_proteins_activated", 100.0)
            if pct > max_pct_activated:
                n_dense_excluded += 1
                continue
            qualifying[fid] = dict(info)
            qualifying[fid]["pct_proteins_activated"] = pct

    logger.info(
        "Features with both q<0.05: %d (excluded %d dense > %.0f%%)",
        len(qualifying),
        n_dense_excluded,
        max_pct_activated,
    )

    # Pull best MEME motif per qualifying feature
    pwm_dir = data_dir / "motif_pwm_enrichment"
    feat_best: dict[int, dict] = {}
    for fid in qualifying:
        ifid = int(fid)
        best = _best_motif(pwm_dir / f"{ifid:04d}.json")
        if best and best["consensus"]:
            feat_best[ifid] = best

    logger.info("Features with MEME best consensus: %d", len(feat_best))

    # Union-find over edit distance <= max_edit on best consensus
    fids = sorted(feat_best.keys())
    parent = {f: f for f in fids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, a in enumerate(fids):
        ca = feat_best[a]["consensus"]
        for b in fids[i + 1:]:
            cb = feat_best[b]["consensus"]
            if _edit_distance(ca, cb, cap=max_edit + 1) <= max_edit:
                union(a, b)

    clusters: dict[int, list[int]] = defaultdict(list)
    for f in fids:
        clusters[find(f)].append(f)
    multi = {root: members for root, members in clusters.items() if len(members) >= 2}
    logger.info("Clusters with 2+ members: %d", len(multi))

    # Load geometry feature importances for every member
    all_geom_feature_set: set[str] = set()
    member_fi: dict[int, dict] = {}
    member_conc: dict[int, dict] = {}
    member_rules: dict[int, str] = {}
    for members in multi.values():
        for f in members:
            geom_path = data_dir / "geometry_enrichment" / f"{f:04d}.json"
            if not geom_path.exists():
                member_fi[f] = {}
                continue
            gd = json.loads(geom_path.read_text())
            grl = gd.get("geometric_residue_level", {})
            fi = grl.get("feature_importances", {}) or {}
            member_fi[f] = fi
            member_conc[f] = grl.get("concordance", {}) or {}
            member_rules[f] = grl.get("rules", "")
            all_geom_feature_set.update(fi.keys())

    all_geom_feature_names = sorted(all_geom_feature_set)

    families_out = []
    for _, members in multi.items():
        # Build per-member records
        rows = []
        vecs = []
        for f in members:
            info = qualifying[str(f)]
            fi = member_fi.get(f, {})
            vec = np.array([fi.get(fn, 0.0) for fn in all_geom_feature_names])
            vecs.append(vec)
            rows.append({
                "feature_id": f,
                "consensus": feat_best[f]["consensus"],
                "motif_id": feat_best[f].get("motif_id", ""),
                "motif_width": feat_best[f].get("width"),
                "motif_e_value": feat_best[f].get("e_value"),
                "motif_pr_auc": feat_best[f].get("pr_auc", 0.0),
                "motif_best_f1": feat_best[f].get("best_f1"),
                "geom_pr_auc": info.get("geom_pr_auc", 0.0),
                "top_geometric_feature": info.get("top_geometric_feature", ""),
                "structural_category": info.get("structural_category", ""),
                "is_geometry_primary": info.get("is_geometry_primary", False),
                "pct_proteins_activated": info.get("pct_proteins_activated", 0.0),
                "feature_importances": fi,
                "concordance_f1": member_conc.get(f, {}).get("f1"),
                "concordance_prauc": member_conc.get(f, {}).get("avg_precision"),
                "rules": member_rules.get(f, ""),
            })

        V = np.array(vecs)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        N = V / norms
        cs = N @ N.T
        n = len(members)
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            mean_cos_sim = float(cs[mask].mean())
        else:
            mean_cos_sim = 1.0

        top_geoms = set(r["top_geometric_feature"] for r in rows)
        # Cosine-gated diversity to stay consistent with the subdomain case study.
        # Label-based diversity was too permissive: two nodes with cos 0.7 (very
        # similar importance profiles) could still differ on the argmax label.
        geom_diverse = mean_cos_sim < GEOM_COS_THRESHOLD

        # Cluster label: pick the consensus of the member with highest motif PR-AUC
        rep = max(rows, key=lambda r: r["motif_pr_auc"])
        label = rep["consensus"]

        families_out.append({
            "representative_consensus": label,
            "n_nodes": n,
            "geom_diverse": geom_diverse,
            "n_unique_top_geom": len(top_geoms),
            "mean_cosine_similarity": round(mean_cos_sim, 4),
            "members": sorted(rows, key=lambda m: -m["geom_pr_auc"]),
        })

    families_out.sort(
        key=lambda f: (-int(f["geom_diverse"]), -f["n_nodes"], f["mean_cosine_similarity"])
    )

    result = {
        "q_gate": {
            "motif_pr_auc_padj": Q_SIG,
            "geometry_prauc_padj": Q_SIG,
            "max_pct_activated": max_pct_activated,
        },
        "grouping": {
            "method": "union-find over pairwise edit distance on best MEME consensus",
            "max_edit_distance": max_edit,
        },
        "n_qualifying_features": len(qualifying),
        "n_with_meme_consensus": len(feat_best),
        "n_families": len(families_out),
        "n_geom_diverse_families": sum(1 for f in families_out if f["geom_diverse"]),
        "geometry_feature_names": all_geom_feature_names,
        "families": families_out,
    }
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Build MEME case study families JSON")
    parser.add_argument("--data-dir", required=True, help="Path to feature data directory")
    parser.add_argument("--max-edit", type=int, default=2, help="Max edit distance for grouping (default 2)")
    parser.add_argument("--out-name", default="meme_case_study_families.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    result = build_meme_case_studies(data_dir, max_edit=args.max_edit)

    out_path = data_dir / args.out_name
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        "Saved %d families (%d geom-diverse) to %s",
        result["n_families"],
        result["n_geom_diverse_families"],
        out_path,
    )
    for fam in result["families"]:
        logger.info(
            "  %-14s n=%d diverse=%s cos_sim=%.3f",
            fam["representative_consensus"],
            fam["n_nodes"],
            fam["geom_diverse"],
            fam["mean_cosine_similarity"],
        )


if __name__ == "__main__":
    main()
