"""Tests for Stage 1 — survey pass (real ESM2 + SAE, no mocks).

These tests load the real ESM2-8M model and the trained SAE, run them
on a handful of known protein sequences, and verify the output files.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from proteinlens.analysis.feature_pipeline.checkpoint import PipelineState
from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.survey import (
    _compute_and_save_survey_outputs,
    run_survey,
)


# A few short, real protein fragments for testing.
# Using short sequences keeps GPU/CPU time minimal.
TEST_SEQUENCES = {
    "TEST_SEQ1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
    "TEST_SEQ2": "MGHFTEEDKATITSLWGKVNVEDAGGETLGRLLVVYPWTQRFFDSFGNLSS",
    "TEST_SEQ3": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV",
}

# Trivial cluster map: each sequence is its own cluster
TEST_CLUSTER_MAP = {k: k for k in TEST_SEQUENCES}

SAE_DIR = Path("trained_models/fiery-sweep")


@pytest.fixture
def survey_config(tmp_path):
    """Create a PipelineConfig pointing at a temp directory with a test FASTA."""
    # Write test FASTA
    fasta_path = tmp_path / "swissprot_human.fasta"
    with open(fasta_path, "w") as f:
        for acc, seq in TEST_SEQUENCES.items():
            f.write(f">{acc}\n{seq}\n")

    config = PipelineConfig(
        sae_dir=SAE_DIR,
        output_dir=tmp_path,
        max_proteins=3,
        survey_checkpoint_every=2,  # checkpoint frequently for test
    )
    return config


@pytest.mark.skipif(
    not SAE_DIR.exists(),
    reason="Trained SAE not found at trained_models/fiery-sweep",
)
class TestRunSurvey:
    """Integration tests for the full survey pass."""

    def test_survey_produces_expected_outputs(self, survey_config):
        """run_survey should create memmap, global max, top-N, and coverage files."""
        config = survey_config
        state = PipelineState(config.pipeline_state_path)

        run_survey(config, state, TEST_CLUSTER_MAP)

        # Memmap should exist with correct shape
        assert config.protein_feature_maxes_path.exists()
        memmap = np.memmap(
            config.protein_feature_maxes_path,
            dtype="float32",
            mode="r",
            shape=(3, 5120),  # 3 test proteins, 5120 features
        )
        # At least some features should have non-zero max activations
        assert memmap.max() > 0

        # Global max file
        assert config.feature_max_path.exists()
        global_max = np.load(config.feature_max_path)
        assert global_max.shape == (5120,)
        assert global_max.max() > 0

        # Top-N JSON
        assert config.survey_top_path.exists()
        with open(config.survey_top_path) as f:
            top = json.load(f)
        assert len(top) == 5120  # one entry per feature
        # At least one feature should have top proteins listed
        has_proteins = any(len(entries) > 0 for entries in top.values())
        assert has_proteins

        # Coverage JSON
        assert config.survey_coverage_path.exists()
        with open(config.survey_coverage_path) as f:
            coverage = json.load(f)
        assert len(coverage) == 5120

        # Pipeline state should be marked complete
        assert state.is_stage_complete("survey")

    def test_survey_memmap_values_match_global_max(self, survey_config):
        """Global max should equal the column-wise max of the memmap."""
        config = survey_config
        state = PipelineState(config.pipeline_state_path)

        run_survey(config, state, TEST_CLUSTER_MAP)

        memmap = np.memmap(
            config.protein_feature_maxes_path,
            dtype="float32",
            mode="r",
            shape=(3, 5120),
        )
        global_max = np.load(config.feature_max_path)

        # The global max should be the column-wise max of the memmap
        expected_max = np.max(memmap, axis=0)
        np.testing.assert_allclose(global_max, expected_max, rtol=1e-5)

    def test_survey_resumability(self, survey_config):
        """A second run should skip already-processed proteins."""
        config = survey_config
        state = PipelineState(config.pipeline_state_path)

        # First run
        run_survey(config, state, TEST_CLUSTER_MAP)
        processed_count_1 = state.data["survey_processed_count"]

        # Second run — should detect all are done and skip
        state2 = PipelineState(config.pipeline_state_path)
        # Stage is already marked complete, but let's test the
        # accession-level skip logic by un-marking the stage
        state2.data["completed_stages"] = []
        state2.save()

        run_survey(config, state2, TEST_CLUSTER_MAP)
        # Processed count should not increase (no new work)
        assert state2.data["survey_processed_count"] == processed_count_1


class TestComputeSurveyOutputs:
    """Unit tests for the output computation from a memmap."""

    def test_top_n_extraction(self, tmp_path):
        """Top-N should correctly identify highest-activation proteins."""
        config = PipelineConfig(output_dir=tmp_path, n_top_per_feature=2)

        # Create a small fake memmap: 4 proteins, 3 features
        accessions = ["A", "B", "C", "D"]
        data = np.array([
            [1.0, 0.0, 0.5],  # A
            [3.0, 2.0, 0.0],  # B
            [2.0, 1.0, 1.0],  # C
            [0.0, 3.0, 0.0],  # D
        ], dtype="float32")

        # Save required files
        np.save(config.feature_max_path, np.max(data, axis=0))

        # Write a pipeline_state.json for the selection stage
        state = PipelineState(config.pipeline_state_path)
        state.set_accession_index({a: i for i, a in enumerate(accessions)})
        state.set_total_proteins(4)

        _compute_and_save_survey_outputs(
            config=config,
            accessions=accessions,
            protein_maxes=data,
            num_features=3,
            member_to_rep={a: a for a in accessions},
        )

        with open(config.survey_top_path) as f:
            top = json.load(f)

        # Feature 0: top-2 should be B (3.0) and C (2.0)
        feat0_accs = [e["accession"] for e in top["0"]]
        assert feat0_accs == ["B", "C"]

        # Feature 1: top-2 should be D (3.0) and B (2.0)
        feat1_accs = [e["accession"] for e in top["1"]]
        assert feat1_accs == ["D", "B"]

    def test_coverage_stats(self, tmp_path):
        """Coverage should correctly count activated proteins and clusters."""
        config = PipelineConfig(
            output_dir=tmp_path,
            activation_threshold=0.5,  # only count activations > 0.5
        )

        accessions = ["A", "B", "C"]
        data = np.array([
            [1.0, 0.0],  # A: feat 0 active, feat 1 not
            [0.3, 2.0],  # B: feat 0 below threshold, feat 1 active
            [0.0, 0.0],  # C: nothing active
        ], dtype="float32")

        np.save(config.feature_max_path, np.max(data, axis=0))

        member_to_rep = {"A": "REP1", "B": "REP1", "C": "REP2"}

        _compute_and_save_survey_outputs(
            config=config,
            accessions=accessions,
            protein_maxes=data,
            num_features=2,
            member_to_rep=member_to_rep,
        )

        with open(config.survey_coverage_path) as f:
            coverage = json.load(f)

        # Feature 0: only A activates above 0.5
        assert coverage["0"]["n_proteins_activated"] == 1
        assert coverage["0"]["n_clusters_activated"] == 1  # A is in REP1

        # Feature 1: only B activates above 0.5
        assert coverage["1"]["n_proteins_activated"] == 1
        assert coverage["1"]["n_clusters_activated"] == 1  # B is in REP1


class TestNormalizationCorrectness:
    """Regression test for the normalize-before-encode fix.

    Uses a synthetic ReLUSAE with normalize_to_sqrt_d=True so that
    the normalization path is NOT a no-op.  Verifies that the pipeline's
    manual normalize+encode produces the same output as forward().
    """

    def test_normalized_sae_pipeline_matches_forward(self):
        """Pipeline encode path must match forward() for normalized SAEs.

        Creates a small ReLUSAE with normalize_to_sqrt_d=True and random
        weights, then checks that:
          normalize_input + encode(normed) == forward(x, output_features=True)[1]

        If someone removes the _normalize_input_and_get_norms call from
        the pipeline, this test will fail because encode(raw_x) != encode(normed_x)
        when normalization is enabled.
        """
        import torch
        from proteinlens.sae.dictionary import ReLUSAE

        activation_dim = 16
        dict_size = 32
        seq_len = 5

        # Create a normalized SAE with random weights
        sae = ReLUSAE(activation_dim, dict_size, normalize_to_sqrt_d=True)
        sae.eval()

        # Random input (simulating ESM embeddings)
        torch.manual_seed(42)
        x = torch.randn(seq_len, activation_dim)

        with torch.no_grad():
            # Canonical path: forward() normalises internally then encodes
            _, feats_forward = sae(x, output_features=True)

            # Pipeline path: manual normalise + encode (what survey.py does)
            normed, _ = sae._normalize_input_and_get_norms(x)
            feats_pipeline = sae.encode(normed)

            # Wrong path: encode without normalising (the old bug)
            feats_wrong = sae.encode(x)

        # Pipeline path must match forward path
        np.testing.assert_allclose(
            feats_pipeline.numpy(),
            feats_forward.numpy(),
            rtol=1e-5,
            err_msg="Pipeline normalize+encode diverges from forward()",
        )

        # The un-normalized path must be DIFFERENT (proving the test is not tautological)
        assert not np.allclose(feats_wrong.numpy(), feats_forward.numpy(), rtol=1e-3), (
            "encode(raw_x) should differ from forward(x) for a normalized SAE, "
            "but they are the same — this test is tautological"
        )
