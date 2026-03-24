"""Tests for Stage 3 — per-residue collection (real models + AlphaFold API).

These tests load the real ESM2-8M and SAE to verify that per-residue
activations are computed and stored correctly.  PDB fetching is tested
against the real AlphaFold API.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import requests

from proteinlens.analysis.feature_pipeline.collection import (
    _compute_residue_activations,
    _has_pdb,
    fetch_alphafold_pdb,
    run_collection,
)
from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.embedders.esm import ESM
from proteinlens.sae.inference import load_sae
from proteinlens.utils import get_device

SAE_DIR = Path("trained_models/fiery-sweep")

# Short test sequences
TEST_SEQUENCES = {
    "TEST_COL1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
    "TEST_COL2": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV",
}


@pytest.mark.skipif(
    not SAE_DIR.exists(),
    reason="Trained SAE not found at trained_models/fiery-sweep",
)
class TestComputeResidueActivations:
    """Tests for the core per-residue activation computation."""

    @pytest.fixture(scope="class")
    def models(self):
        """Load ESM and SAE once for the class (expensive)."""
        device = get_device()
        esm = ESM(model_name="facebook/esm2_t6_8M_UR50D", device=device)
        sae = load_sae(SAE_DIR, device=device)
        return esm, sae, device

    def test_output_shape(self, models):
        """Activations should be (seq_len, num_features)."""
        esm, sae, device = models
        seq = TEST_SEQUENCES["TEST_COL1"]
        activations = _compute_residue_activations(esm, sae, seq, layer=3, device=device)

        assert activations.shape == (len(seq), sae.dict_size)
        assert activations.dtype == np.float32

    def test_output_non_negative(self, models):
        """ReLU SAE activations should be non-negative."""
        esm, sae, device = models
        seq = TEST_SEQUENCES["TEST_COL1"]
        activations = _compute_residue_activations(esm, sae, seq, layer=3, device=device)

        assert np.all(activations >= 0), (
            f"Found negative activations: min = {activations.min()}"
        )

    def test_output_has_nonzero_entries(self, models):
        """At least some residues should have non-zero activation for some features."""
        esm, sae, device = models
        seq = TEST_SEQUENCES["TEST_COL1"]
        activations = _compute_residue_activations(esm, sae, seq, layer=3, device=device)

        assert activations.max() > 0, "All activations are zero — something is wrong"


class TestFetchAlphafoldPdb:
    """Tests for PDB fetching from the real AlphaFold API."""

    def test_fetch_known_protein(self, tmp_path):
        """P68871 (human hemoglobin beta) should have an AlphaFold structure."""
        session = requests.Session()
        pdb_text = fetch_alphafold_pdb("P68871", tmp_path, session)

        assert pdb_text is not None
        assert "ATOM" in pdb_text
        # Should be cached on disk
        assert _has_pdb("P68871", tmp_path)

    def test_fetch_caches_result(self, tmp_path):
        """Second fetch should use cache (no network call)."""
        session = requests.Session()
        # First fetch
        pdb1 = fetch_alphafold_pdb("P68871", tmp_path, session)
        assert pdb1 is not None

        # Second fetch — should return cached
        pdb2 = fetch_alphafold_pdb("P68871", tmp_path, session)
        assert pdb2 == pdb1

    def test_fetch_nonexistent_protein(self, tmp_path):
        """A fake accession should return None without crashing."""
        session = requests.Session()
        result = fetch_alphafold_pdb("ZZZZZZFAKE", tmp_path, session)
        assert result is None


@pytest.mark.skipif(
    not SAE_DIR.exists(),
    reason="Trained SAE not found at trained_models/fiery-sweep",
)
class TestRunCollection:
    """Integration test for the full collection stage."""

    def test_collection_produces_npz_files(self, tmp_path):
        """run_collection should create .npz files for selected proteins."""
        config = PipelineConfig(
            sae_dir=SAE_DIR,
            output_dir=tmp_path,
        )

        # Write a test FASTA
        with open(config.fasta_path, "w") as f:
            for acc, seq in TEST_SEQUENCES.items():
                f.write(f">{acc}\n{seq}\n")

        # Write a selection.json that selects both test proteins
        selection = {
            "per_feature": {},
            "all_selected_accessions": list(TEST_SEQUENCES.keys()),
        }
        with open(config.selection_path, "w") as f:
            json.dump(selection, f)

        run_collection(config)

        # Check .npz files were created
        for acc in TEST_SEQUENCES:
            npz_path = config.residue_activations_dir / f"{acc}.npz"
            assert npz_path.exists(), f"Missing {npz_path}"

            # Verify contents
            data = np.load(npz_path)
            assert "activations" in data
            activations = data["activations"]
            assert activations.shape[0] == len(TEST_SEQUENCES[acc])
            assert activations.shape[1] == 5120
