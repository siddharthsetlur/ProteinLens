"""Tests for Stage 8 — Sequence position F1 enrichment.

All tests use synthetic data with known answers to verify:
- Position predicate evaluation at boundary conditions
- Predicate index construction from multi-protein pools
- F1 detection of N-terminal activation concentration
- Low F1 for uniform (position-independent) activation
- End-to-end pipeline run with per-feature + summary JSON output
- Resumability (pre-existing output not overwritten)
"""

import json
from pathlib import Path

import numpy as np
import pytest

from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.position_enrichment import (
    POSITION_PREDICATES,
    _analyze_feature,
    _build_predicate_indices,
    run_position_enrichment,
)


# ===================================================================
# 8.6.1 — Predicate correctness at boundary positions
# ===================================================================


class TestPredicateEvaluation:
    """Verify each predicate category returns correct bool values."""

    def test_first_5(self):
        pred = POSITION_PREDICATES["first_5"]
        assert pred(0, 100) is True
        assert pred(4, 100) is True
        assert pred(5, 100) is False

    def test_first_10(self):
        pred = POSITION_PREDICATES["first_10"]
        assert pred(9, 100) is True
        assert pred(10, 100) is False

    def test_first_20(self):
        pred = POSITION_PREDICATES["first_20"]
        assert pred(19, 100) is True
        assert pred(20, 100) is False

    def test_last_5(self):
        pred = POSITION_PREDICATES["last_5"]
        assert pred(94, 100) is False
        assert pred(95, 100) is True
        assert pred(99, 100) is True

    def test_last_10(self):
        pred = POSITION_PREDICATES["last_10"]
        assert pred(89, 100) is False
        assert pred(90, 100) is True

    def test_last_20(self):
        pred = POSITION_PREDICATES["last_20"]
        assert pred(79, 100) is False
        assert pred(80, 100) is True

    def test_pct_0_10(self):
        pred = POSITION_PREDICATES["pct_0_10"]
        assert pred(0, 100) is True
        assert pred(9, 100) is True
        assert pred(10, 100) is False

    def test_pct_90_100(self):
        pred = POSITION_PREDICATES["pct_90_100"]
        assert pred(89, 100) is False
        assert pred(90, 100) is True

    def test_third_N(self):
        pred = POSITION_PREDICATES["third_N"]
        assert pred(0, 99) is True
        assert pred(32, 99) is True
        assert pred(33, 99) is False

    def test_third_M(self):
        pred = POSITION_PREDICATES["third_M"]
        assert pred(33, 99) is True
        assert pred(65, 99) is True
        assert pred(66, 99) is False

    def test_third_C(self):
        pred = POSITION_PREDICATES["third_C"]
        assert pred(65, 99) is False
        assert pred(66, 99) is True

    def test_terminal_10pct(self):
        pred = POSITION_PREDICATES["terminal_10pct"]
        assert pred(0, 100) is True   # N-terminal
        assert pred(95, 100) is True  # C-terminal
        assert pred(50, 100) is False # middle

    def test_interior_80pct(self):
        pred = POSITION_PREDICATES["interior_80pct"]
        assert pred(10, 100) is True
        assert pred(89, 100) is True
        assert pred(9, 100) is False
        assert pred(90, 100) is False

    def test_mid_20pct(self):
        pred = POSITION_PREDICATES["mid_20pct"]
        assert pred(40, 100) is True
        assert pred(59, 100) is True
        assert pred(39, 100) is False
        assert pred(60, 100) is False

    def test_short_sequence_first_5(self):
        """Sequences shorter than 20 still work for absolute predicates."""
        pred = POSITION_PREDICATES["first_5"]
        assert pred(0, 3) is True
        assert pred(2, 3) is True

    def test_short_sequence_last_5(self):
        pred = POSITION_PREDICATES["last_5"]
        # len=3: last_5 means pos >= 3-5 = -2, so all positions match
        assert pred(0, 3) is True
        assert pred(2, 3) is True

    def test_position_0_and_last(self):
        """Edge case: first and last position of a 1-residue sequence."""
        pred_first = POSITION_PREDICATES["first_5"]
        pred_last = POSITION_PREDICATES["last_5"]
        assert pred_first(0, 1) is True
        assert pred_last(0, 1) is True


# ===================================================================
# 8.6.2 — _build_predicate_indices with known lengths
# ===================================================================


def test_build_predicate_indices_two_proteins():
    """Two proteins of length 50 each: verify first_5, last_5, pct_0_10 indices."""
    seq_lengths = [50, 50]
    total = 100

    indices = _build_predicate_indices(seq_lengths, total)

    # first_5: positions 0-4 in protein 1 (global 0-4) + positions 0-4 in protein 2 (global 50-54)
    first5 = sorted(indices["first_5"].tolist())
    assert first5 == [0, 1, 2, 3, 4, 50, 51, 52, 53, 54]

    # last_5: positions 45-49 in protein 1 (global 45-49) + positions 45-49 in protein 2 (global 95-99)
    last5 = sorted(indices["last_5"].tolist())
    assert last5 == [45, 46, 47, 48, 49, 95, 96, 97, 98, 99]

    # pct_0_10: positions 0-4 in each protein (pos/50 < 0.1)
    pct010 = sorted(indices["pct_0_10"].tolist())
    assert pct010 == [0, 1, 2, 3, 4, 50, 51, 52, 53, 54]


def test_build_predicate_indices_unequal_lengths():
    """Proteins of different lengths: verify global indexing is correct."""
    seq_lengths = [10, 20]
    total = 30

    indices = _build_predicate_indices(seq_lengths, total)

    # first_5: pos 0-4 of protein 1 (global 0-4) + pos 0-4 of protein 2 (global 10-14)
    first5 = sorted(indices["first_5"].tolist())
    assert first5 == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]


# ===================================================================
# 8.6.3 — N-terminal concentrated activation -> high F1 for first_5
# ===================================================================


def _make_n_terminal_feature_data(n_proteins=10, seq_len=100):
    """Create synthetic feature data where activation concentrates in first 5 residues."""
    proteins = []
    for i in range(n_proteins):
        seq = "A" * seq_len
        acts = [0.0] * seq_len
        # High activation in first 5 positions only
        for j in range(5):
            acts[j] = 3.0
        proteins.append({
            "accession": f"P{i:04d}",
            "max_activation": 3.0,
            "mean_activation": 0.15,
            "sequence": seq,
            "sequence_length": seq_len,
            "pdb_available": False,
            "per_residue_activations": acts,
        })
    return {
        "feature_id": 0,
        "max_activation": 3.0,
        "dataset_coverage": {},
        "top_sequences": proteins,
        "activation_bins": {},
    }


def test_n_terminal_signal_detected():
    """When activation is concentrated at positions 0-4, first_5 should achieve high F1."""
    feature_data = _make_n_terminal_feature_data(n_proteins=10, seq_len=100)
    config = PipelineConfig(output_dir="/tmp/unused")

    result = _analyze_feature(feature_data, feat_max=3.0, config=config)

    assert result is not None
    assert result["n_proteins_evaluated"] == 10
    assert result["n_total_residues"] == 1000

    # first_5 should be the top predicate with high F1
    top = result["top_positions"][0]
    assert top["position"] == "first_5"
    assert top["best_f1"] > 0.8


# ===================================================================
# 8.6.4 — Uniform activation -> low F1 for all predicates
# ===================================================================


def test_random_activation_no_position_signal():
    """Random activation should not yield high F1 for narrow positional predicates.

    With uniform activation, broad predicates like interior_80pct trivially
    achieve high F1 (they cover ~80% of residues, matching most activated
    positions). This is expected — those predicates are not informative but
    they are technically precise. The test verifies that *narrow* predicates
    (first_5, last_5, pct_0_10) do NOT achieve high F1 on random data,
    confirming position-specificity is required for a high score.
    """
    n_proteins = 10
    seq_len = 100
    rng = np.random.RandomState(42)
    proteins = []
    for i in range(n_proteins):
        seq = "A" * seq_len
        # Random activations uniformly distributed
        acts = rng.uniform(0.0, 3.0, size=seq_len).tolist()
        proteins.append({
            "accession": f"P{i:04d}",
            "max_activation": 3.0,
            "mean_activation": 1.5,
            "sequence": seq,
            "sequence_length": seq_len,
            "pdb_available": False,
            "per_residue_activations": acts,
        })
    feature_data = {
        "feature_id": 1,
        "max_activation": 3.0,
        "dataset_coverage": {},
        "top_sequences": proteins,
        "activation_bins": {},
    }
    config = PipelineConfig(output_dir="/tmp/unused")

    result = _analyze_feature(feature_data, feat_max=3.0, config=config)

    # Narrow positional predicates should not achieve high F1 on random data
    narrow_predicates = {"first_5", "first_10", "last_5", "last_10", "pct_0_10", "pct_90_100"}
    if result is not None and result.get("top_positions"):
        for entry in result["top_positions"]:
            if entry["position"] in narrow_predicates:
                assert entry["best_f1"] < 0.5, (
                    f"Narrow predicate {entry['position']} has unexpectedly high F1 "
                    f"({entry['best_f1']}) on random activation"
                )


# ===================================================================
# 8.6.5 — End-to-end run_position_enrichment
# ===================================================================


def test_end_to_end_run(tmp_path):
    """Create a minimal dataset, run the pipeline, verify output structure."""
    # Set up directory structure
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    # Two features
    global_max = np.array([3.0, 2.0, 0.0], dtype=np.float32)  # 3rd is dead
    np.save(tmp_path / "feature_max_activations.npy", global_max)

    for feat_idx in range(2):
        feat_data = _make_n_terminal_feature_data(n_proteins=5, seq_len=50)
        feat_data["feature_id"] = feat_idx
        with open(features_dir / f"{feat_idx:04d}.json", "w") as f:
            json.dump(feat_data, f)

    # Also write a dead feature file (max=0, will be skipped)
    with open(features_dir / "0002.json", "w") as f:
        json.dump({"feature_id": 2, "top_sequences": [], "activation_bins": {}}, f)

    config = PipelineConfig(output_dir=tmp_path)
    run_position_enrichment(config)

    # Check per-feature outputs
    for feat_idx in range(2):
        out_path = tmp_path / "position_enrichment" / f"{feat_idx:04d}.json"
        assert out_path.exists()
        data = json.load(open(out_path))
        assert data["feature_id"] == feat_idx
        assert "top_positions" in data
        assert len(data["top_positions"]) > 0
        # Each entry should have "position" not "motif"
        assert "position" in data["top_positions"][0]

    # Dead feature should NOT have output
    assert not (tmp_path / "position_enrichment" / "0002.json").exists()

    # Check summary
    summary_path = tmp_path / "position_enrichment" / "summary.json"
    assert summary_path.exists()
    summary = json.load(open(summary_path))
    assert summary["n_features_analyzed"] == 2
    assert summary["n_features_skipped"] == 1  # dead feature
    assert "0" in summary["features"]
    assert "best_position" in summary["features"]["0"]
    assert "best_position_f1" in summary["features"]["0"]


# ===================================================================
# 8.6.6 — Resumability: pre-existing output not overwritten
# ===================================================================


def test_resumability(tmp_path):
    """Pre-create one output JSON, run pipeline, verify it was not overwritten."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    global_max = np.array([3.0, 2.0], dtype=np.float32)
    np.save(tmp_path / "feature_max_activations.npy", global_max)

    for feat_idx in range(2):
        feat_data = _make_n_terminal_feature_data(n_proteins=5, seq_len=50)
        feat_data["feature_id"] = feat_idx
        with open(features_dir / f"{feat_idx:04d}.json", "w") as f:
            json.dump(feat_data, f)

    # Pre-create output for feature 0 with a sentinel value
    pos_dir = tmp_path / "position_enrichment"
    pos_dir.mkdir(parents=True)
    sentinel = {
        "feature_id": 0,
        "sentinel": "pre-existing",
        "top_positions": [{"position": "first_5", "best_f1": 0.99}],
        "n_predicates_tested": 1,
    }
    with open(pos_dir / "0000.json", "w") as f:
        json.dump(sentinel, f)

    config = PipelineConfig(output_dir=tmp_path)
    run_position_enrichment(config)

    # Feature 0 should still have the sentinel
    data0 = json.load(open(pos_dir / "0000.json"))
    assert data0.get("sentinel") == "pre-existing"

    # Feature 1 should have been computed fresh
    data1 = json.load(open(pos_dir / "0001.json"))
    assert "sentinel" not in data1
    assert data1["feature_id"] == 1
