"""Regression tests for Table 4 columns 4 and 5.

The union behind those columns must run over the COLUMN-3 GATED features, not
over every feature that has NMPFam hits. Unioning over all features inflated
layer 4 to 38,846 families / 7,733,244 sequences against the paper's
3,875 / 757,802; restricting to the gated set reproduces both exactly.
"""

import json
import os
from pathlib import Path

import pytest

from scripts.build_nmpfam_transfer_summary import build_summary

# Paper Table 4, layer 4 (docs/28622_Interpreting_Latent_Prot.pdf).
PAPER_L4 = {
    "n_features_median_prauc_above_gate": 376,
    "n_families_matched": 3875,
    "n_sequences_annotated": 757802,
}

# Profiles chosen so average_precision_score is deterministic:
# positives ranked first -> 1.0; the lone positive ranked last -> 0.25.
_STRONG = {"sae_activation_profile": [1.0, 1.0, 0.0, 0.0],
           "geom_prob_profile": [0.9, 0.8, 0.1, 0.05]}
_WEAK = {"sae_activation_profile": [1.0, 0.0, 0.0, 0.0],
         "geom_prob_profile": [0.1, 0.9, 0.8, 0.7]}


def _hit(family_id, sequence_count, strong):
    return {"family_id": family_id, "sequence_count": sequence_count,
            **(_STRONG if strong else _WEAK)}


def _write_feature(root, fid, hits, geometry_p):
    enrichment = root / "nmpfam" / "nmpfam_enrichment"
    permutation = root / "permutation_null"
    enrichment.mkdir(parents=True, exist_ok=True)
    permutation.mkdir(exist_ok=True)
    (enrichment / f"{fid:04d}.json").write_text(json.dumps({
        "feature_id": fid,
        "activation_threshold_sae": 0.5,
        "nmpfam_hits": hits,
    }))
    (permutation / f"{fid:04d}.json").write_text(json.dumps({
        "feature_id": fid,
        "p_values": {"geometry_prauc": geometry_p},
    }))


@pytest.fixture
def analysis_dir(tmp_path):
    """Four features that separate the gated union from the all-features union.

    f0, f1  geometry q-significant, median PR-AUC > 0.5  -> GATED
    f2      not q-significant, but has a strong hit      -> excluded
    f3      q-significant, median PR-AUC 0.25            -> excluded, strong hit
    """
    _write_feature(tmp_path, 0, [_hit("FAM_A", 100, True),
                                 _hit("FAM_B", 200, True)], 0.001)
    _write_feature(tmp_path, 1, [_hit("FAM_B", 200, True),
                                 _hit("FAM_C", 300, True)], 0.001)
    _write_feature(tmp_path, 2, [_hit("FAM_D", 999, True)], 0.900)
    _write_feature(tmp_path, 3, [_hit("FAM_E", 555, True),
                                 _hit("FAM_F", 1, False),
                                 _hit("FAM_G", 1, False)], 0.001)
    (tmp_path / "geometry_primary_analysis.json").write_text(
        json.dumps({"features": {}}))
    (tmp_path / "dataset_stats.json").write_text(json.dumps({"num_features": 10}))
    return tmp_path


def test_union_covers_only_gated_features(analysis_dir):
    table4 = build_summary(analysis_dir)["table4"]

    # f0 and f1 only. FAM_B is shared and must be deduplicated.
    assert table4["n_features_median_prauc_above_gate"] == 2
    assert table4["n_families_matched"] == 3          # FAM_A, FAM_B, FAM_C
    assert table4["n_sequences_annotated"] == 600     # 100 + 200 + 300


def test_ungated_features_with_strong_hits_are_excluded(analysis_dir):
    """Pins the specific defect: unioning over every feature with hits.

    f2 (not q-significant) and f3 (median PR-AUC below the gate) both carry
    strong hits. Counting them yields 5 families / 2,154 sequences, which is
    how layer 4 came out 10x high.
    """
    table4 = build_summary(analysis_dir)["table4"]

    assert table4["n_families_matched"] != 5
    assert table4["n_sequences_annotated"] != 2154


def test_shared_family_counted_once(analysis_dir):
    """FAM_B is hit by both gated features; sequences must not double-count."""
    table4 = build_summary(analysis_dir)["table4"]
    assert table4["n_sequences_annotated"] == 600, "FAM_B double-counted"


@pytest.mark.integration
def test_layer4_reproduces_paper_table4():
    """Exact paper values from the real layer-4 analysis directory.

    Needs the released NMPFam enrichment extracted (~33 GB). Point
    PROTEINLENS_L4_ANALYSIS at the analysis dir, or place it at the default
    repo path. See .claude/skills/reproduce-paper/references/artifacts.md.
    """
    default = Path("trained_models/layer_4/frosty-sweep-15/analysis")
    analysis = Path(os.environ.get("PROTEINLENS_L4_ANALYSIS", default))
    enrichment = analysis / "nmpfam" / "nmpfam_enrichment"
    if not enrichment.is_dir():
        pytest.fail(
            f"layer-4 NMPFam enrichment not found at {enrichment}. "
            "Download and extract it, or set PROTEINLENS_L4_ANALYSIS."
        )

    n_files = len(list(enrichment.glob("*.json")))
    assert n_files > 7900, (
        f"only {n_files} enrichment files; the pre-2026-08-19 release blob held "
        "284 and silently produces 2.73% for column 1"
    )

    table4 = build_summary(analysis)["table4"]
    for key, expected in PAPER_L4.items():
        assert table4[key] == expected, f"{key}: {table4[key]} != paper {expected}"
