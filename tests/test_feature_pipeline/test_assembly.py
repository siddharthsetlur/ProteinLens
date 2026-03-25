"""Tests for Stage 4 — assembly (per-feature JSON generation).

These tests use synthetic data to verify JSON schema compliance,
data integrity, and edge cases without needing real models.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from proteinlens.analysis.feature_pipeline.assembly import (
    _build_protein_entry,
    run_assembly,
)
from proteinlens.analysis.feature_pipeline.config import PipelineConfig


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def assembly_setup(tmp_path):
    """Set up all upstream outputs needed by assembly.

    Creates synthetic data for 4 proteins and 3 features.
    """
    config = PipelineConfig(
        output_dir=tmp_path,
        n_top_per_feature=2,
        activation_threshold=0.05,
    )

    accessions = ["PROT_A", "PROT_B", "PROT_C", "PROT_D"]
    sequences = {
        "PROT_A": "MVHLT",
        "PROT_B": "MKTAYIAK",
        "PROT_C": "MGHFTEED",
        "PROT_D": "MVLS",
    }
    num_features = 3

    # Write FASTA
    with open(config.fasta_path, "w") as f:
        for acc, seq in sequences.items():
            f.write(f">{acc}\n{seq}\n")

    # Write global max
    global_max = np.array([4.0, 2.0, 0.0], dtype="float32")
    np.save(config.feature_max_path, global_max)

    # Write selection.json
    selection = {
        "per_feature": {
            "0": {
                "top": ["PROT_A", "PROT_B"],
                "bins": {
                    "0.0-0.25": ["PROT_C"],
                    "0.25-0.5": [],
                    "0.5-0.75": [],
                    "0.75-1.0": ["PROT_A"],
                },
            },
            "1": {
                "top": ["PROT_D"],
                "bins": {
                    "0.0-0.25": [],
                    "0.25-0.5": [],
                    "0.5-0.75": [],
                    "0.75-1.0": [],
                },
            },
            "2": {
                "top": [],
                "bins": {
                    "0.0-0.25": [],
                    "0.25-0.5": [],
                    "0.5-0.75": [],
                    "0.75-1.0": [],
                },
            },
        },
        "all_selected_accessions": ["PROT_A", "PROT_B", "PROT_C", "PROT_D"],
    }
    with open(config.selection_path, "w") as f:
        json.dump(selection, f)

    # Write survey_top20.json
    survey_top = {
        "0": [
            {"accession": "PROT_A", "max_activation": 4.0},
            {"accession": "PROT_B", "max_activation": 3.0},
        ],
        "1": [{"accession": "PROT_D", "max_activation": 2.0}],
        "2": [],
    }
    with open(config.survey_top_path, "w") as f:
        json.dump(survey_top, f)

    # Write survey_coverage.json
    coverage = {
        "0": {
            "n_proteins_activated": 3,
            "n_clusters_activated": 3,
            "pct_proteins_activated": 75.0,
            "pct_clusters_activated": 75.0,
            "total_proteins": 4,
            "total_clusters": 4,
            "activation_threshold": 0.05,
        },
        "1": {
            "n_proteins_activated": 1,
            "n_clusters_activated": 1,
            "pct_proteins_activated": 25.0,
            "pct_clusters_activated": 25.0,
            "total_proteins": 4,
            "total_clusters": 4,
            "activation_threshold": 0.05,
        },
        "2": {
            "n_proteins_activated": 0,
            "n_clusters_activated": 0,
            "pct_proteins_activated": 0.0,
            "pct_clusters_activated": 0.0,
            "total_proteins": 4,
            "total_clusters": 4,
            "activation_threshold": 0.05,
        },
    }
    with open(config.survey_coverage_path, "w") as f:
        json.dump(coverage, f)

    # Write per-residue .npz files (synthetic activations)
    for acc, seq in sequences.items():
        acts = np.random.rand(len(seq), num_features).astype("float32")
        npz_path = config.residue_activations_dir / f"{acc}.npz"
        np.savez_compressed(npz_path, activations=acts)

    return config, sequences


# ===================================================================
# Tests
# ===================================================================


class TestRunAssembly:
    """Integration tests for the full assembly stage."""

    def test_assembly_creates_feature_jsons(self, assembly_setup):
        """Should create one JSON per feature."""
        config, _ = assembly_setup
        run_assembly(config)

        for feat_idx in range(3):
            path = config.features_dir / f"{feat_idx:04d}.json"
            assert path.exists(), f"Missing {path}"

    def test_feature_json_schema(self, assembly_setup):
        """Each feature JSON should have the required top-level keys."""
        config, _ = assembly_setup
        run_assembly(config)

        with open(config.features_dir / "0000.json") as f:
            feat = json.load(f)

        required_keys = {
            "feature_id",
            "max_activation",
            "dataset_coverage",
            "top_sequences",
            "activation_bins",
        }
        assert required_keys.issubset(feat.keys())
        assert feat["feature_id"] == 0
        assert feat["max_activation"] == 4.0

    def test_protein_entry_schema(self, assembly_setup):
        """Each protein entry should have the required fields."""
        config, _ = assembly_setup
        run_assembly(config)

        with open(config.features_dir / "0000.json") as f:
            feat = json.load(f)

        assert len(feat["top_sequences"]) > 0
        entry = feat["top_sequences"][0]
        required_fields = {
            "accession",
            "max_activation",
            "mean_activation",
            "sequence",
            "sequence_length",
            "pdb_available",
            "per_residue_activations",
        }
        assert required_fields.issubset(entry.keys())
        # per_residue_activations length should match sequence_length
        assert len(entry["per_residue_activations"]) == entry["sequence_length"]

    def test_top_sequences_sorted_descending(self, assembly_setup):
        """Top sequences should be sorted by max_activation descending."""
        config, _ = assembly_setup
        run_assembly(config)

        with open(config.features_dir / "0000.json") as f:
            feat = json.load(f)

        activations = [e["max_activation"] for e in feat["top_sequences"]]
        assert activations == sorted(activations, reverse=True)

    def test_sequences_json_created(self, assembly_setup):
        """sequences.json should contain all referenced proteins."""
        config, sequences = assembly_setup
        run_assembly(config)

        assert config.sequences_path.exists()
        with open(config.sequences_path) as f:
            saved_seqs = json.load(f)

        # All proteins that appear in any feature file should be in sequences.json
        for acc in ["PROT_A", "PROT_B", "PROT_C", "PROT_D"]:
            if acc in saved_seqs:
                assert saved_seqs[acc] == sequences[acc]

    def test_dataset_stats_json_created(self, assembly_setup):
        """dataset_stats.json should contain summary statistics."""
        config, _ = assembly_setup
        run_assembly(config)

        assert config.dataset_stats_path.exists()
        with open(config.dataset_stats_path) as f:
            stats = json.load(f)

        assert stats["total_proteins"] == 4
        assert stats["num_features"] == 3

    def test_dead_feature_has_empty_output(self, assembly_setup):
        """A feature with max_activation=0 should have empty sequences/bins."""
        config, _ = assembly_setup
        run_assembly(config)

        with open(config.features_dir / "0002.json") as f:
            feat = json.load(f)

        assert feat["max_activation"] == 0.0
        assert feat["top_sequences"] == []
        for bin_entries in feat["activation_bins"].values():
            assert bin_entries == []

    def test_bin_entries_have_correct_max_when_npz_missing(self, assembly_setup):
        """Regression: bin entries must report correct max_activation even
        when the .npz file is missing (falls back to survey memmap)."""
        config, sequences = assembly_setup

        # Delete one protein's .npz to simulate interrupted collection
        npz_path = config.residue_activations_dir / "PROT_C.npz"
        assert npz_path.exists()
        npz_path.unlink()

        # Create a pipeline_state.json with accession index and memmap
        # so the memmap fallback is available
        from proteinlens.analysis.feature_pipeline.checkpoint import PipelineState
        accessions = list(sequences.keys())
        state = PipelineState(config.pipeline_state_path)
        state.set_accession_index({a: i for i, a in enumerate(accessions)})
        state.set_total_proteins(len(accessions))

        # Create a memmap with known activation values
        num_features = 3
        mm = np.memmap(
            config.protein_feature_maxes_path,
            dtype="float32",
            mode="w+",
            shape=(len(accessions), num_features),
        )
        # Give PROT_C a known activation of 0.8 for feature 0
        prot_c_idx = accessions.index("PROT_C")
        mm[prot_c_idx, 0] = 0.8
        mm.flush()

        run_assembly(config)

        with open(config.features_dir / "0000.json") as f:
            feat = json.load(f)

        # Find PROT_C in the bin entries
        for bin_entries in feat["activation_bins"].values():
            for entry in bin_entries:
                if entry["accession"] == "PROT_C":
                    # Must NOT be 0.0 — should be 0.8 from the memmap
                    assert entry["max_activation"] == pytest.approx(0.8, abs=1e-5), (
                        f"PROT_C max_activation should be 0.8 from memmap fallback, "
                        f"got {entry['max_activation']}"
                    )
                    # per_residue_activations should be None (npz is missing)
                    assert entry["per_residue_activations"] is None
                    return  # found and verified
        pytest.fail("PROT_C was not found in any activation bin")

    def test_missing_lookup_sources_logs_warning(self, caplog):
        """Regression: _lookup_survey_max logs a warning when both sources are unavailable."""
        from proteinlens.analysis.feature_pipeline.assembly import _lookup_survey_max
        import logging

        with caplog.at_level(logging.WARNING, logger="proteinlens.analysis.feature_pipeline.assembly"):
            result = _lookup_survey_max("UNKNOWN", 0, {}, None, None)
        assert result == 0.0
        assert any("No survey max found" in msg for msg in caplog.messages)
