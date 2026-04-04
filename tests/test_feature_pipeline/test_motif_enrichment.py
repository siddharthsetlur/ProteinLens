"""Tests for Stage 7 — Sequence motif (k-mer) F1 enrichment.

All tests use synthetic data with known answers to verify:
- k-mer extraction at valid positions with edge trimming
- Protein pooling with deduplication across top_sequences and bins
- Vectorised F1 computation (perfect separation, partial overlap)
- min_count filtering of rare k-mers
- End-to-end pipeline run with per-feature + summary JSON output
"""

import json
from pathlib import Path

import numpy as np
import pytest

from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.motif_enrichment import (
    _analyze_feature,
    _compute_best_motif_f1,
    _extract_kmers_with_activations,
    _pool_proteins_for_feature,
    run_motif_enrichment,
)


# ===================================================================
# 7.7.1 — _extract_kmers_with_activations: basic k=3
# ===================================================================


def test_extract_kmers_basic():
    """'ACDEF' with k=3 produces 3 k-mers centred at positions 1, 2, 3."""
    seq = "ACDEF"
    acts = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = _extract_kmers_with_activations(seq, acts, k=3)

    assert len(result) == 3
    kmers = [r[0] for r in result]
    activations = [r[1] for r in result]

    # k-mer at centre i=1: seq[0:3] = "ACD"
    # k-mer at centre i=2: seq[1:4] = "CDE"
    # k-mer at centre i=3: seq[2:5] = "DEF"
    assert kmers == ["ACD", "CDE", "DEF"]
    assert activations == [0.2, 0.3, 0.4]


# ===================================================================
# 7.7.2 — Sequence shorter than k returns empty list
# ===================================================================


def test_extract_kmers_short_sequence():
    """A sequence shorter than k yields no k-mers."""
    result = _extract_kmers_with_activations("AC", [0.1, 0.2], k=3)
    assert result == []


def test_extract_kmers_exact_length():
    """A sequence of exactly length k yields exactly 1 k-mer."""
    result = _extract_kmers_with_activations("ACD", [0.1, 0.2, 0.3], k=3)
    assert len(result) == 1
    assert result[0] == ("ACD", 0.2)


def test_extract_kmers_skips_non_standard():
    """Positions where the k-mer contains non-standard AAs (e.g. 'X') are skipped."""
    seq = "ACXDE"
    acts = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = _extract_kmers_with_activations(seq, acts, k=3)
    # Centre 1: "ACX" — has X, skip
    # Centre 2: "CXD" — has X, skip
    # Centre 3: "XDE" — has X, skip
    assert len(result) == 0


def test_extract_kmers_length_mismatch_raises():
    """Mismatched sequence and activation lengths raise AssertionError."""
    with pytest.raises(AssertionError, match="Sequence length"):
        _extract_kmers_with_activations("ACDEF", [0.1, 0.2], k=3)


# ===================================================================
# 7.7.3 — _pool_proteins_for_feature: deduplication
# ===================================================================


def _make_protein_entry(accession, sequence="ACDEF", activations=None):
    """Helper: create a protein dict matching the Stage 4 JSON schema."""
    return {
        "accession": accession,
        "sequence": sequence,
        "per_residue_activations": activations or [0.1] * len(sequence),
    }


def test_pool_deduplication():
    """Same accession in top_sequences and a bin is only returned once."""
    feature_data = {
        "top_sequences": [_make_protein_entry("P001")],
        "activation_bins": {
            "0.0-0.25": [_make_protein_entry("P001"), _make_protein_entry("P002")],
        },
    }
    result = _pool_proteins_for_feature(feature_data)
    accessions = [r[0] for r in result]
    assert accessions == ["P001", "P002"]


def test_pool_skips_null_activations():
    """Entries with per_residue_activations=None are excluded."""
    feature_data = {
        "top_sequences": [
            {"accession": "P001", "sequence": "ACDEF", "per_residue_activations": None},
            _make_protein_entry("P002"),
        ],
        "activation_bins": {},
    }
    result = _pool_proteins_for_feature(feature_data)
    assert len(result) == 1
    assert result[0][0] == "P002"


# ===================================================================
# 7.7.4 — Perfect separation: F1 = 1.0
# ===================================================================


def test_perfect_separation():
    """Motif 'AAA' appearing only at high-activation positions achieves F1=1.0."""
    # Build synthetic data: 'AAA' at positions 0-9 (high act),
    # other k-mers at positions 10-19 (low act)
    n = 20
    all_acts = np.zeros(n)
    all_acts[:10] = 1.0  # high activation at first 10 positions

    kmer_indices = {
        "AAA": np.arange(10),     # only at high-activation positions
        "GGG": np.arange(10, 20), # only at low-activation positions
    }

    results = _compute_best_motif_f1(
        kmer_indices, all_acts, feat_max=1.0,
        n_steps=50, min_count=5, top_n=10,
    )

    # AAA should achieve perfect F1
    aaa_result = next(r for r in results if r["motif"] == "AAA")
    assert aaa_result["best_f1"] == 1.0


# ===================================================================
# 7.7.5 — min_count filtering
# ===================================================================


def test_min_count_filtering():
    """A motif appearing 3 times with min_count=5 is excluded from results."""
    all_acts = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    kmer_indices = {
        "AAA": np.array([0, 1, 2]),       # 3 occurrences — below min_count=5
        "GGG": np.arange(3, 10),           # 7 occurrences — above min_count
    }

    results = _compute_best_motif_f1(
        kmer_indices, all_acts, feat_max=1.0,
        n_steps=50, min_count=5, top_n=10,
    )

    motif_names = [r["motif"] for r in results]
    assert "AAA" not in motif_names
    assert "GGG" in motif_names


# ===================================================================
# 7.7.6 — End-to-end: synthetic data, run pipeline, verify outputs
# ===================================================================


def test_end_to_end(tmp_path):
    """Run motif enrichment on synthetic data, verify output files and schema."""
    # Setup directory structure
    output_dir = tmp_path / "feature_data"
    output_dir.mkdir()

    # Create feature_max_activations.npy (3 features: active, active, dead)
    feat_max = np.array([2.0, 1.5, 0.0], dtype=np.float32)
    np.save(output_dir / "feature_max_activations.npy", feat_max)

    # Create features/ dir with per-feature JSONs
    features_dir = output_dir / "features"
    features_dir.mkdir()

    # Feature 0: 5 proteins with repetitive sequence so each k-mer
    # appears >= 5 times (the min_count default)
    seq0 = "AAACCCAAACCC"  # 12 residues, 10 k-mer positions
    acts0 = [2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 0.0, 0.0, 0.0]
    proteins_0 = [
        {"accession": f"P{i:03d}", "sequence": seq0,
         "per_residue_activations": acts0}
        for i in range(5)
    ]
    feat0 = {
        "feature_id": 0,
        "top_sequences": proteins_0[:3],
        "activation_bins": {
            "0.0-0.25": proteins_0[3:],
        },
    }
    with open(features_dir / "0000.json", "w") as f:
        json.dump(feat0, f)

    # Feature 1: similar structure, 5 proteins
    seq1 = "GGGAAAGGGAAA"
    acts1 = [0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    proteins_1 = [
        {"accession": f"Q{i:03d}", "sequence": seq1,
         "per_residue_activations": acts1}
        for i in range(5)
    ]
    feat1 = {
        "feature_id": 1,
        "top_sequences": proteins_1,
        "activation_bins": {},
    }
    with open(features_dir / "0001.json", "w") as f:
        json.dump(feat1, f)

    # Feature 2: dead (feat_max=0), no JSON needed

    # Run motif enrichment
    config = PipelineConfig(output_dir=output_dir)
    run_motif_enrichment(config)

    # Verify output files exist
    motif_dir = output_dir / "motif_enrichment"
    assert (motif_dir / "0000.json").exists()
    assert (motif_dir / "summary.json").exists()

    # Verify per-feature JSON schema
    with open(motif_dir / "0000.json") as f:
        result = json.load(f)

    assert result["feature_id"] == 0
    assert result["feature_max_activation"] == 2.0
    assert result["n_proteins_evaluated"] == 5
    assert result["k"] == 3
    assert "top_motifs" in result
    assert len(result["top_motifs"]) > 0

    motif_entry = result["top_motifs"][0]
    expected_keys = {
        "motif", "best_f1", "best_threshold", "best_threshold_normalized",
        "precision_at_best", "recall_at_best", "n_occurrences",
        "n_true_positives", "n_false_positives", "n_false_negatives",
        "interpretation",
    }
    assert set(motif_entry.keys()) == expected_keys

    # Verify summary.json schema
    with open(motif_dir / "summary.json") as f:
        summary = json.load(f)

    assert summary["k"] == 3
    assert "n_features_analyzed" in summary
    assert "n_features_skipped" in summary
    assert "features" in summary


# ===================================================================
# 7.7.7 — Summary: features with no eligible k-mers are absent
# ===================================================================


def test_summary_excludes_empty_features(tmp_path):
    """Features that produce no eligible k-mers do not appear in summary."""
    output_dir = tmp_path / "feature_data"
    output_dir.mkdir()

    # One feature, very short protein → no k-mers pass min_count
    feat_max = np.array([1.0], dtype=np.float32)
    np.save(output_dir / "feature_max_activations.npy", feat_max)

    features_dir = output_dir / "features"
    features_dir.mkdir()

    # Only 3 residues → only 1 k-mer → below min_count=5
    feat0 = {
        "feature_id": 0,
        "top_sequences": [
            {"accession": "P001", "sequence": "ACD", "per_residue_activations": [1.0, 1.0, 1.0]},
        ],
        "activation_bins": {},
    }
    with open(features_dir / "0000.json", "w") as f:
        json.dump(feat0, f)

    config = PipelineConfig(output_dir=output_dir)
    run_motif_enrichment(config)

    summary_path = output_dir / "motif_enrichment" / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    # Feature 0 should be skipped (no eligible k-mers)
    assert "0" not in summary["features"]
    assert summary["n_features_skipped"] == 1
