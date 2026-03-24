"""End-to-end integration test for the feature data pipeline.

Runs the full pipeline (Stages 0a through 4) on a tiny set of real
proteins.  Requires network access, ESM2-8M, trained SAE, and MMseqs2.

This test takes ~2-5 minutes depending on hardware (model loading is
the main cost).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from proteinlens.analysis.feature_pipeline.checkpoint import PipelineState
from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.data_acquisition import (
    _parse_fasta,
    download_swissprot_fasta,
)

SAE_DIR = Path("trained_models/fiery-sweep")

# Skip the entire module if prerequisites are missing
pytestmark = [
    pytest.mark.skipif(
        not SAE_DIR.exists(),
        reason="Trained SAE not found at trained_models/fiery-sweep",
    ),
    pytest.mark.skipif(
        shutil.which("mmseqs") is None,
        reason="MMseqs2 not installed",
    ),
]


class TestFullPipelineIntegration:
    """End-to-end test running all stages on 5 real proteins."""

    def test_full_pipeline(self, tmp_path):
        """Run the complete pipeline and verify all outputs."""
        config = PipelineConfig(
            sae_dir=SAE_DIR,
            output_dir=tmp_path,
            organism_taxid=9606,
            max_proteins=5,
            survey_checkpoint_every=2,
        )
        state = PipelineState(config.pipeline_state_path)

        # ── Stage 0a: Download ──
        accessions, sequences = download_swissprot_fasta(config)
        state.mark_stage_complete("download")
        assert len(accessions) > 0
        assert config.fasta_path.exists()

        # ── Stage 0b: Clustering ──
        from proteinlens.analysis.feature_pipeline.clustering import (
            run_mmseqs_clustering,
        )

        member_to_rep = run_mmseqs_clustering(config)
        state.mark_stage_complete("cluster")
        assert len(member_to_rep) == len(accessions)

        # ── Stage 1: Survey ──
        from proteinlens.analysis.feature_pipeline.survey import run_survey

        run_survey(config, state, member_to_rep)
        assert config.feature_max_path.exists()
        assert config.protein_feature_maxes_path.exists()
        assert config.survey_top_path.exists()
        assert config.survey_coverage_path.exists()

        global_max = np.load(config.feature_max_path)
        assert global_max.shape == (5120,)

        # ── Stage 2: Selection ──
        from proteinlens.analysis.feature_pipeline.selection import run_selection

        selection = run_selection(config)
        state.mark_stage_complete("selection")
        assert len(selection["all_selected_accessions"]) > 0

        # ── Stage 3: Collection ──
        from proteinlens.analysis.feature_pipeline.collection import run_collection

        run_collection(config)
        state.mark_stage_complete("collection")

        # At least some .npz files should exist
        npz_files = list(config.residue_activations_dir.glob("*.npz"))
        assert len(npz_files) > 0

        # ── Stage 4: Assembly ──
        from proteinlens.analysis.feature_pipeline.assembly import run_assembly

        run_assembly(config)
        state.mark_stage_complete("assembly")

        # Verify feature JSONs
        feature_files = list(config.features_dir.glob("*.json"))
        assert len(feature_files) == 5120

        # Spot-check one feature file
        with open(config.features_dir / "0000.json") as f:
            feat0 = json.load(f)
        assert "feature_id" in feat0
        assert "max_activation" in feat0
        assert "dataset_coverage" in feat0
        assert "top_sequences" in feat0
        assert "activation_bins" in feat0

        # sequences.json and dataset_stats.json should exist
        assert config.sequences_path.exists()
        assert config.dataset_stats_path.exists()

        with open(config.dataset_stats_path) as f:
            stats = json.load(f)
        assert stats["num_features"] == 5120
        assert stats["total_proteins"] > 0

        # ── Verify data consistency ──
        # Every protein in top_sequences should have a sequence in
        # sequences.json and a matching per_residue_activations length
        with open(config.sequences_path) as f:
            all_seqs = json.load(f)

        for entry in feat0.get("top_sequences", []):
            acc = entry["accession"]
            assert acc in all_seqs, f"{acc} missing from sequences.json"
            if entry["per_residue_activations"] is not None:
                assert len(entry["per_residue_activations"]) == entry["sequence_length"]
                assert entry["sequence_length"] == len(all_seqs[acc])
