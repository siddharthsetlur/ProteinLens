import json

from scripts.figure6_descriptor_counts import compute_counts


def _write_feature(root, fid, score, importances, p_value):
    geometry = root / "geometry_enrichment"
    permutation = root / "permutation_null"
    geometry.mkdir(exist_ok=True)
    permutation.mkdir(exist_ok=True)
    (geometry / f"{fid:04d}.json").write_text(
        json.dumps(
            {
                "feature_id": fid,
                "geometric_residue_level": {
                    "concordance": {"avg_precision": score},
                    "feature_importances": importances,
                },
            }
        )
    )
    (permutation / f"{fid:04d}.json").write_text(
        json.dumps(
            {
                "feature_id": fid,
                "p_values": {"geometry_prauc": p_value},
            }
        )
    )


def test_figure6_counts_one_top_descriptor_per_fixed_q_hit(tmp_path):
    _write_feature(tmp_path, 0, 0.4, {"curvature_mean": 0.2, "torsion_mean": 0.15}, 0.01)
    _write_feature(tmp_path, 1, 0.8, {"contact_density_8A": 0.3}, 0.01)
    _write_feature(tmp_path, 2, 0.9, {"curvature_mean": 0.9}, 0.5)

    counts, provenance = compute_counts(tmp_path)

    assert counts["0.3-0.6"]["curvature_mean"] == 1
    assert counts[">0.6"]["contact_density_8A"] == 1
    assert counts["0.3-0.6"]["torsion_mean"] == 0
    assert sum(map(sum, (counts["0.3-0.6"].values(), counts[">0.6"].values()))) == 2
    assert provenance["n_features_pr_auc_above_threshold"] == 2
    assert provenance["q_source"] == "fixed_score_permutation_raw_p"
