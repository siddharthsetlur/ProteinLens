"""Tests for Stage 6a: geometry feature computation.

Verifies:
  - Real PDB files produce valid .npz profiles and protein-level features
  - Resumability: second run skips all proteins
  - Corrupt PDB files are skipped gracefully
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.geometry_features import (
    run_geometry_features,
)
from proteinlens.analysis.geometry.protein_features import GEOM_FEATURE_NAMES

# Use a few real PDB files from the existing cache
REAL_PDB_CACHE = Path("protein_results/alphafold_analysis/pdb_cache")


def _setup_config_with_pdbs(tmpdir: Path, n_pdbs: int = 3) -> PipelineConfig:
    """Create a PipelineConfig with a temporary output dir and a few real PDBs.

    Copies *n_pdbs* real PDB files from the project's pdb_cache into the
    config's pdb_cache_dir so that Stage 6a has something to process.

    Parameters
    ----------
    tmpdir : Path
        Temporary directory for pipeline outputs.
    n_pdbs : int
        Number of PDB files to copy.

    Returns
    -------
    PipelineConfig
        Configured for the temporary directory.
    """
    config = PipelineConfig(sae_dir="x", output_dir=str(tmpdir))
    pdb_cache = config.pdb_cache_dir

    real_pdbs = sorted(REAL_PDB_CACHE.glob("*.pdb"))[:n_pdbs]
    assert len(real_pdbs) >= n_pdbs, (
        f"Need at least {n_pdbs} PDB files in {REAL_PDB_CACHE}"
    )

    for pdb_file in real_pdbs:
        shutil.copy(pdb_file, pdb_cache / pdb_file.name)

    return config


class TestRunGeometryFeatures:
    """Tests for the Stage 6a pipeline function."""

    def test_produces_valid_outputs(self):
        """Running on 3 real PDBs produces .npz profiles and protein matrix."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            config = _setup_config_with_pdbs(tmpdir, n_pdbs=3)

            run_geometry_features(config)

            # Check residue profile .npz files were created
            profiles = list(config.geometry_residue_profiles_dir.glob("*.npz"))
            assert len(profiles) >= 1, "No residue profile .npz files created"

            # Verify a profile has expected keys
            data = np.load(profiles[0], allow_pickle=True)
            assert "ca" in data, "Missing 'ca' key in profile .npz"
            assert "curvature" in data, "Missing 'curvature' key"
            assert "protein_geometry" in data, "Missing 'protein_geometry' key"
            assert data["ca"].ndim == 2 and data["ca"].shape[1] == 3
            assert data["protein_geometry"].shape == (len(GEOM_FEATURE_NAMES),)

            # Check protein-level features matrix
            assert config.geometry_protein_features_path.exists(), (
                "geometry_protein_features.npz not created"
            )
            feat_data = np.load(config.geometry_protein_features_path)
            assert "accessions" in feat_data
            assert "geometry_matrix" in feat_data
            assert "feature_names" in feat_data
            n_proteins = feat_data["geometry_matrix"].shape[0]
            assert n_proteins >= 1
            assert feat_data["geometry_matrix"].shape[1] == len(GEOM_FEATURE_NAMES)

    def test_resumability(self):
        """Second run should skip all proteins (0 new computed)."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            config = _setup_config_with_pdbs(tmpdir, n_pdbs=3)

            # First run
            run_geometry_features(config)
            profiles_first = set(
                p.name for p in config.geometry_residue_profiles_dir.glob("*.npz")
            )

            # Second run should skip all
            run_geometry_features(config)
            profiles_second = set(
                p.name for p in config.geometry_residue_profiles_dir.glob("*.npz")
            )

            # Same set of profiles
            assert profiles_first == profiles_second

    def test_corrupt_pdb_skipped(self):
        """A corrupt PDB file should be skipped gracefully."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            config = _setup_config_with_pdbs(tmpdir, n_pdbs=2)

            # Add a corrupt PDB
            corrupt_path = config.pdb_cache_dir / "AF-CORRUPT-F1-model_v1.pdb"
            corrupt_path.write_text("THIS IS NOT A VALID PDB FILE\nGARBAGE")

            # Should not raise
            run_geometry_features(config)

            # Corrupt accession should not have a profile
            assert not (config.geometry_residue_profiles_dir / "CORRUPT.npz").exists()

            # Other proteins should still have profiles
            profiles = list(config.geometry_residue_profiles_dir.glob("*.npz"))
            assert len(profiles) >= 1
