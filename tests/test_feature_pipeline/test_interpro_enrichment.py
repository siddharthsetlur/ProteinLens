"""Tests for Stage 5c — InterPro F1 enrichment analysis.

All tests use synthetic data with known answers to verify:
- Protein-level F1 computation and threshold selection
- Residue-level F1 computation with domain boundaries
- Threshold reporting consistency (reported F1 matches recomputed F1)
- Edge cases: no annotations, single-protein annotations, uniform activation
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from proteinlens.analysis.feature_pipeline.checkpoint import PipelineState
from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.interpro_api import (
    InterProDomain,
    _save_cached,
)
from proteinlens.analysis.feature_pipeline.interpro_enrichment import (
    _compute_f1_from_arrays,
    _compute_protein_level_f1,
    _compute_residue_level_f1,
    load_residue_activations,
    run_interpro_enrichment,
)


# ===================================================================
# Fixtures
# ===================================================================


def _make_domain(
    code: str = "IPR000001",
    name: str = "Test Domain",
    start: int = 1,
    end: int = 50,
    member_db: str = "pfam",
    member_acc: str = "PF00001",
) -> InterProDomain:
    """Helper to create an InterProDomain with sensible defaults."""
    return InterProDomain(
        interpro_accession=code,
        interpro_name=name,
        type="Family",
        member_db=member_db,
        member_accession=member_acc,
        start=start,
        end=end,
    )


@pytest.fixture
def enrichment_setup(tmp_path):
    """Set up a synthetic dataset for enrichment testing.

    Creates 20 proteins and 2 features:
    - Feature 0: max = 5.0
      - Proteins 0-9: have annotation IPR000001, activations = 5.0
      - Proteins 10-19: no annotations, activations = 0.0
      -> Perfect separation: best F1 should be 1.0

    - Feature 1: max = 3.0
      - All proteins have activation = 1.5 (uniform)
      - Some have annotation IPR000002 — F1 should be low
    """
    config = PipelineConfig(
        output_dir=tmp_path,
        interpro_n_per_bin=50,
        interpro_f1_threshold_steps=50,
        interpro_top_annotations=5,
        interpro_min_proteins=3,
    )

    n_proteins = 20
    n_features = 2
    accessions = [f"PROT{i:04d}" for i in range(n_proteins)]

    # Build protein-feature max activation array
    data = np.zeros((n_proteins, n_features), dtype="float32")

    # Feature 0: perfect separation
    for i in range(10):
        data[i, 0] = 5.0   # annotated proteins: high activation
    # Proteins 10-19: activation 0.0 (no annotation)

    # Feature 1: uniform activation, no separation
    data[:, 1] = 1.5

    # Save memmap
    mm = np.memmap(
        config.protein_feature_maxes_path,
        dtype="float32",
        mode="w+",
        shape=(n_proteins, n_features),
    )
    mm[:] = data
    mm.flush()

    # Save global max
    global_max = np.max(data, axis=0)
    np.save(config.feature_max_path, global_max)

    # Save pipeline_state
    state = PipelineState(config.pipeline_state_path)
    state.set_accession_index({a: i for i, a in enumerate(accessions)})
    state.set_total_proteins(n_proteins)

    # Create interpro_selection.json
    # Feature 0: annotated (0-9) in high bins, unannotated (10-19) in zero bin
    feat0_bins = {
        "0.0": [f"PROT{i:04d}" for i in range(10, 20)],
        "0.9-1.0": [f"PROT{i:04d}" for i in range(10)],
    }
    # Add empty bins for completeness
    for frac in ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
                  "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9"]:
        feat0_bins[frac] = []

    feat1_bins = {
        "0.0": [],
        "0.4-0.5": accessions,  # all proteins at uniform activation
    }
    for frac in ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4",
                  "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]:
        feat1_bins[frac] = []

    selection = {
        "per_feature": {
            "0": {"bins": feat0_bins},
            "1": {"bins": feat1_bins},
        },
        "all_selected_accessions": accessions,
    }
    with open(config.interpro_selection_path, "w") as f:
        json.dump(selection, f)

    # Create InterPro cache files.
    # IMPORTANT: PROT0000-PROT0004 are written TWICE below.  The first
    # loop writes [IPR000001] for all 10 annotated proteins (0-9).  The
    # second loop (below) OVERWRITES PROT0000-PROT0004 to add IPR000002
    # alongside IPR000001.  This intentional overwrite lets us test
    # Feature 1's uniform-activation scenario with a second annotation.
    cache_dir = config.interpro_cache_dir
    for i in range(10):
        acc = f"PROT{i:04d}"
        domain = _make_domain(code="IPR000001", name="Perfect Separator")
        _save_cached(cache_dir / f"{acc}.json", acc, [domain])

    for i in range(10, 20):
        acc = f"PROT{i:04d}"
        _save_cached(cache_dir / f"{acc}.json", acc, [])

    # For feature 1: give first 5 proteins an annotation (IPR000002)
    # but since all activations are uniform, F1 should be poor.
    # This OVERWRITES the cache for PROT0000-PROT0004 (see note above).
    for i in range(5):
        acc = f"PROT{i:04d}"
        domains = [
            _make_domain(code="IPR000001", name="Perfect Separator"),
            _make_domain(code="IPR000002", name="Noise Annotation"),
        ]
        _save_cached(cache_dir / f"{acc}.json", acc, domains)

    return config, accessions, data


@pytest.fixture
def residue_enrichment_setup(tmp_path):
    """Set up synthetic per-residue data for residue-level F1 testing.

    Creates one protein with:
    - 100 residues
    - Feature 0: high activation (5.0) at positions 10-20 (0-based),
      low activation (0.1) elsewhere
    - Domain annotation at positions 11-21 (1-based) = 10-20 (0-based)
    -> The domain and high-activation region overlap exactly, so
       residue-level F1 should be near 1.0.
    """
    config = PipelineConfig(
        output_dir=tmp_path,
        interpro_f1_threshold_steps=50,
        interpro_top_annotations=5,
        interpro_min_proteins=1,
    )

    n_features = 2
    seq_len = 100

    # Create per-residue activation .npz
    activations = np.full((seq_len, n_features), 0.1, dtype="float32")
    # Feature 0: high activation at positions 10-20 (0-based)
    activations[10:21, 0] = 5.0

    npz_dir = config.residue_activations_dir
    np.savez_compressed(npz_dir / "TESTPROT.npz", activations=activations)

    # Create InterPro cache with domain at 1-based positions 11-21
    cache_dir = config.interpro_cache_dir
    domain = _make_domain(
        code="IPR999999",
        name="Exact Match Domain",
        start=11,  # 1-based
        end=21,    # 1-based inclusive
    )
    _save_cached(cache_dir / "TESTPROT.json", "TESTPROT", [domain])

    return config, activations


# ===================================================================
# Test: F1 computation utility
# ===================================================================


class TestComputeF1:
    """Tests for the _compute_f1_from_arrays helper."""

    def test_perfect_separation(self):
        """Perfect predictions should give F1 = 1.0."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        tp, fp, fn, precision, recall, f1 = _compute_f1_from_arrays(y_true, y_pred)
        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0
        assert tp == 3
        assert fp == 0
        assert fn == 0

    def test_all_wrong(self):
        """Completely wrong predictions should give F1 = 0.0."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        tp, fp, fn, precision, recall, f1 = _compute_f1_from_arrays(y_true, y_pred)
        assert f1 == 0.0

    def test_no_predictions(self):
        """If y_pred is all zeros, F1 should be 0.0."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        _, _, _, _, _, f1 = _compute_f1_from_arrays(y_true, y_pred)
        assert f1 == 0.0

    def test_partial_overlap(self):
        """Known partial-overlap case: 2 TP, 1 FP, 1 FN.
        Precision = 2/3, Recall = 2/3, F1 = 2/3."""
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 1, 0])
        tp, fp, fn, precision, recall, f1 = _compute_f1_from_arrays(y_true, y_pred)
        assert tp == 2
        assert fp == 1
        assert fn == 1
        assert abs(precision - 2 / 3) < 1e-6
        assert abs(recall - 2 / 3) < 1e-6
        assert abs(f1 - 2 / 3) < 1e-6


# ===================================================================
# Test: Protein-level F1 (checklist 4.2)
# ===================================================================


class TestProteinLevelF1:
    """Tests for protein-level F1 computation."""

    def test_perfect_separation_gives_f1_one(self):
        """When annotated proteins all have high activation and unannotated
        have zero, the best F1 should be 1.0."""
        # 10 annotated with activation 5.0, 10 unannotated with activation 0.0
        accessions_with_activations = (
            [(f"ANN{i}", 5.0) for i in range(10)]
            + [(f"NOANN{i}", 0.0) for i in range(10)]
        )

        protein_annotations = {}
        for i in range(10):
            protein_annotations[f"ANN{i}"] = [_make_domain(code="IPR000001")]
        for i in range(10):
            protein_annotations[f"NOANN{i}"] = []

        results = _compute_protein_level_f1(
            accessions_with_activations=accessions_with_activations,
            protein_annotations=protein_annotations,
            feat_max=5.0,
            n_threshold_steps=50,
            min_proteins=3,
            top_n=5,
        )

        assert len(results) >= 1
        top = results[0]
        assert top["annotation_code"] == "IPR000001"
        assert top["best_f1"] == 1.0
        assert top["precision_at_best"] == 1.0
        assert top["recall_at_best"] == 1.0

    def test_threshold_within_valid_range(self):
        """The reported best_threshold should be between 0 and feat_max."""
        accessions_with_activations = (
            [(f"A{i}", float(i)) for i in range(10)]
        )
        protein_annotations = {
            f"A{i}": [_make_domain(code="IPR000001")] if i >= 5 else []
            for i in range(10)
        }

        results = _compute_protein_level_f1(
            accessions_with_activations=accessions_with_activations,
            protein_annotations=protein_annotations,
            feat_max=9.0,
            n_threshold_steps=50,
            min_proteins=3,
            top_n=5,
        )

        assert len(results) >= 1
        for r in results:
            assert 0 <= r["best_threshold"] <= 9.0

    def test_threshold_reporting_consistency(self):
        """The reported F1 should match when we manually recompute from
        the reported TP, FP, FN counts."""
        accessions_with_activations = (
            [(f"ANN{i}", 4.0 + i * 0.1) for i in range(6)]
            + [(f"NOANN{i}", 0.5 + i * 0.1) for i in range(14)]
        )
        protein_annotations = {}
        for i in range(6):
            protein_annotations[f"ANN{i}"] = [_make_domain(code="IPR000001")]
        for i in range(14):
            protein_annotations[f"NOANN{i}"] = []

        results = _compute_protein_level_f1(
            accessions_with_activations=accessions_with_activations,
            protein_annotations=protein_annotations,
            feat_max=5.0,
            n_threshold_steps=50,
            min_proteins=3,
            top_n=5,
        )

        for r in results:
            tp = r["n_true_positives"]
            fp = r["n_false_positives"]
            fn = r["n_false_negatives"]
            if tp + fp > 0 and tp + fn > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                expected_f1 = 2 * precision * recall / (precision + recall)
                assert abs(r["best_f1"] - round(expected_f1, 4)) < 0.01, (
                    f"Reported F1 {r['best_f1']} != recomputed {expected_f1:.4f}"
                )

    def test_annotation_below_min_proteins_skipped(self):
        """An annotation on fewer than min_proteins should not appear."""
        accessions_with_activations = [
            ("A0", 5.0), ("A1", 3.0), ("A2", 1.0),
            ("A3", 0.0), ("A4", 0.0),
        ]
        protein_annotations = {
            "A0": [_make_domain(code="IPR_RARE")],
            "A1": [_make_domain(code="IPR_RARE")],
            # Only 2 proteins have IPR_RARE — below min_proteins=3
            "A2": [], "A3": [], "A4": [],
        }

        results = _compute_protein_level_f1(
            accessions_with_activations=accessions_with_activations,
            protein_annotations=protein_annotations,
            feat_max=5.0,
            n_threshold_steps=50,
            min_proteins=3,
            top_n=5,
        )

        assert len(results) == 0

    def test_uniform_activation_low_f1(self):
        """When all proteins have the same activation, no threshold can
        separate annotated from unannotated well."""
        accessions_with_activations = [
            (f"P{i}", 2.0) for i in range(20)
        ]
        protein_annotations = {
            f"P{i}": [_make_domain(code="IPR000001")] if i < 5 else []
            for i in range(20)
        }

        results = _compute_protein_level_f1(
            accessions_with_activations=accessions_with_activations,
            protein_annotations=protein_annotations,
            feat_max=2.0,
            n_threshold_steps=50,
            min_proteins=3,
            top_n=5,
        )

        # F1 should be low because threshold can't separate them
        # At threshold 0: all predicted positive -> precision = 5/20 = 0.25
        # recall = 1.0, F1 = 0.4
        # At threshold >= 2.0: all predicted negative -> F1 = 0
        if results:
            assert results[0]["best_f1"] <= 0.5


# ===================================================================
# Test: Residue-level F1 (checklist 4.3)
# ===================================================================


class TestResidueLevelF1:
    """Tests for residue-level F1 computation."""

    def test_exact_domain_overlap_high_f1(self, residue_enrichment_setup):
        """When high activation exactly overlaps with domain boundaries,
        residue-level F1 should be near 1.0."""
        config, activations = residue_enrichment_setup

        # Build protein-level result to feed into residue-level
        protein_level_results = [{
            "annotation_code": "IPR999999",
            "annotation_name": "Exact Match Domain",
            "member_db": "pfam",
            "member_accession": "PF00001",
            "best_f1": 1.0,
        }]

        protein_annotations = {
            "TESTPROT": [
                _make_domain(
                    code="IPR999999",
                    name="Exact Match Domain",
                    start=11, end=21,
                )
            ]
        }

        results = _compute_residue_level_f1(
            protein_level_results=protein_level_results,
            protein_annotations=protein_annotations,
            feat_idx=0,
            feat_max=5.0,
            config=config,
        )

        assert len(results) >= 1
        top = results[0]
        assert top["annotation_code"] == "IPR999999"
        assert top["best_f1"] >= 0.9, (
            f"Expected high F1 for exact domain overlap, got {top['best_f1']}"
        )
        assert top["n_residues_in_domain"] == 11  # positions 10-20 inclusive

    def test_residue_counts_sensible(self, residue_enrichment_setup):
        """Domain residue count should be < total residues."""
        config, _ = residue_enrichment_setup

        protein_level_results = [{
            "annotation_code": "IPR999999",
            "annotation_name": "Exact Match Domain",
            "member_db": "pfam",
            "member_accession": "PF00001",
            "best_f1": 1.0,
        }]

        protein_annotations = {
            "TESTPROT": [
                _make_domain(code="IPR999999", start=11, end=21)
            ]
        }

        results = _compute_residue_level_f1(
            protein_level_results=protein_level_results,
            protein_annotations=protein_annotations,
            feat_idx=0,
            feat_max=5.0,
            config=config,
        )

        assert len(results) >= 1
        top = results[0]
        assert top["n_residues_in_domain"] < top["n_total_residues"]
        assert top["n_total_residues"] == 100


# ===================================================================
# Test: load_residue_activations utility (checklist 4.1)
# ===================================================================


class TestLoadResidueActivations:
    """Tests for the dual-directory .npz loader."""

    def test_loads_from_stage3_dir(self, tmp_path):
        """Should find .npz in the primary residue_activations dir."""
        config = PipelineConfig(output_dir=tmp_path)
        acts = np.random.rand(50, 10).astype("float32")
        np.savez_compressed(
            config.residue_activations_dir / "P12345.npz",
            activations=acts,
        )

        loaded = load_residue_activations("P12345", config)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, acts)

    def test_loads_from_interpro_dir(self, tmp_path):
        """Should fall back to interpro_residue_activations dir."""
        config = PipelineConfig(output_dir=tmp_path)
        acts = np.random.rand(30, 10).astype("float32")
        np.savez_compressed(
            config.interpro_residue_activations_dir / "Q67890.npz",
            activations=acts,
        )

        loaded = load_residue_activations("Q67890", config)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, acts)

    def test_returns_none_if_missing(self, tmp_path):
        """Should return None if .npz doesn't exist in either dir."""
        config = PipelineConfig(output_dir=tmp_path)
        assert load_residue_activations("MISSING", config) is None


# ===================================================================
# Test: Full run_interpro_enrichment (checklist 4.6)
# ===================================================================


class TestRunInterProEnrichment:
    """Integration-style test using the full enrichment_setup fixture."""

    def test_end_to_end(self, enrichment_setup):
        """run_interpro_enrichment should produce per-feature JSON and summary."""
        config, _, _ = enrichment_setup
        run_interpro_enrichment(config)

        # Check per-feature JSONs exist
        feat0_path = config.interpro_enrichment_dir / "0000.json"
        assert feat0_path.exists()

        with open(feat0_path, "r") as f:
            feat0 = json.load(f)

        assert feat0["feature_id"] == 0
        assert feat0["feature_max_activation"] == 5.0
        assert "protein_level" in feat0
        assert "residue_level" in feat0

        # Feature 0 should have perfect separation for IPR000001
        if feat0["protein_level"]:
            top = feat0["protein_level"][0]
            assert top["annotation_code"] == "IPR000001"
            assert top["best_f1"] == 1.0

    def test_summary_json_written(self, enrichment_setup):
        """summary.json should exist and have entries for analyzed features."""
        config, _, _ = enrichment_setup
        run_interpro_enrichment(config)

        summary_path = config.interpro_enrichment_dir / "summary.json"
        assert summary_path.exists()

        with open(summary_path, "r") as f:
            summary = json.load(f)

        assert "n_features_analyzed" in summary
        assert "features" in summary

    def test_summary_always_includes_residue_keys(self, enrichment_setup):
        """Every summary entry must include top_residue_annotation and
        top_residue_f1 keys, even when no residue-level results exist.
        The plan schema shows these as always-present.  Setting them to
        null when absent prevents visualizer key-errors."""
        config, _, _ = enrichment_setup
        run_interpro_enrichment(config)

        summary_path = config.interpro_enrichment_dir / "summary.json"
        with open(summary_path, "r") as f:
            summary = json.load(f)

        for feat_key, entry in summary["features"].items():
            assert "top_residue_annotation" in entry, (
                f"Feature {feat_key} summary missing 'top_residue_annotation' key"
            )
            assert "top_residue_f1" in entry, (
                f"Feature {feat_key} summary missing 'top_residue_f1' key"
            )

    def test_all_required_fields_present(self, enrichment_setup):
        """Each protein-level result should have all required fields per the schema."""
        config, _, _ = enrichment_setup
        run_interpro_enrichment(config)

        feat0_path = config.interpro_enrichment_dir / "0000.json"
        with open(feat0_path, "r") as f:
            feat0 = json.load(f)

        required_top_fields = {
            "feature_id", "feature_max_activation",
            "n_proteins_evaluated", "n_proteins_with_annotations",
            "n_unique_annotations_tested", "protein_level", "residue_level",
        }
        assert required_top_fields.issubset(feat0.keys())

        if feat0["protein_level"]:
            required_prot_fields = {
                "annotation_code", "annotation_name", "annotation_type",
                "member_db", "member_accession", "best_f1", "best_threshold",
                "best_threshold_normalized", "precision_at_best",
                "recall_at_best", "n_proteins_with_annotation",
                "n_proteins_without_annotation", "n_true_positives",
                "n_false_positives", "n_false_negatives", "interpretation",
            }
            assert required_prot_fields.issubset(feat0["protein_level"][0].keys())
