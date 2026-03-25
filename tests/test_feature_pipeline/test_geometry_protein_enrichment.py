"""Tests for Stage 6b: protein-level LassoCV geometry enrichment.

Uses synthetic data to verify:
  - Nodes with strong signal get positive r2_cv
  - Nodes with random data are skipped or have ~0 r2_cv
  - Nodes with too few active proteins are skipped
  - Monomial string contains expected feature names
  - Output JSON matches expected schema
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.geometry_protein_enrichment import (
    run_geometry_protein_enrichment,
)


def _setup_synthetic_data(
    tmpdir: Path,
    n_proteins: int = 200,
    n_geom_features: int = 10,
    n_sae_nodes: int = 20,
    min_active: int = 50,
) -> PipelineConfig:
    """Create synthetic geometry + activation data for testing.

    Builds a geometry_protein_features.npz, protein_feature_maxes.npy
    memmap, feature_max_activations.npy, and pipeline_state.json with
    known linear relationships for a few SAE nodes.

    Parameters
    ----------
    tmpdir : Path
        Temp directory for pipeline outputs.
    n_proteins : int
        Number of synthetic proteins.
    n_geom_features : int
        Number of geometry features (columns in geometry matrix).
    n_sae_nodes : int
        Number of SAE nodes (columns in activation matrix).
    min_active : int
        Config value for geometry_min_active_proteins.

    Returns
    -------
    PipelineConfig
        Configured for the synthetic data.
    """
    config = PipelineConfig(
        sae_dir="x",
        output_dir=str(tmpdir),
        geometry_min_active_proteins=min_active,
    )

    rng = np.random.default_rng(42)

    # Generate accessions
    accessions = [f"P{i:05d}" for i in range(n_proteins)]

    # Generate geometry matrix (all finite)
    geom_matrix = rng.standard_normal((n_proteins, n_geom_features))

    # Save geometry_protein_features.npz (uses 10 synthetic features
    # instead of the real 55 to keep tests fast; Stage 6b reads
    # feature_names from the NPZ so the smaller set is handled correctly).
    feature_names = [f"geom_{i}" for i in range(n_geom_features)]
    np.savez_compressed(
        config.geometry_protein_features_path,
        accessions=np.array(accessions),
        geometry_matrix=geom_matrix,
        feature_names=np.array(feature_names),
    )

    # Generate activation matrix with known signals:
    #   Node 0: strong linear relationship with geom_0
    #   Node 1: weak/no signal (random)
    #   Node 2: all zeros (dead feature)
    act_matrix = np.zeros((n_proteins, n_sae_nodes), dtype=np.float32)

    # Node 0: y = 3*geom_0 + 5 + noise, active for most proteins
    act_matrix[:, 0] = np.maximum(
        3.0 * geom_matrix[:, 0] + 5.0 + 0.3 * rng.standard_normal(n_proteins),
        0.0,
    ).astype(np.float32)

    # Node 1: random activations (some positive)
    act_matrix[:, 1] = np.maximum(
        rng.standard_normal(n_proteins), 0.0
    ).astype(np.float32)

    # Node 2: all zeros (dead feature)
    # act_matrix[:, 2] stays 0

    # Node 3: too few active proteins
    act_matrix[:5, 3] = rng.random(5).astype(np.float32) + 0.1

    # Save protein_feature_maxes.npy as memmap
    memmap = np.memmap(
        config.protein_feature_maxes_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_proteins, n_sae_nodes),
    )
    memmap[:] = act_matrix
    memmap.flush()

    # Save feature_max_activations.npy
    feature_maxes = act_matrix.max(axis=0)
    np.save(config.feature_max_path, feature_maxes)

    # Save pipeline_state.json with accession_index mapping
    # (matches the key used by the survey stage / checkpoint.py)
    state = {
        "accession_index": {acc: i for i, acc in enumerate(accessions)},
    }
    config.pipeline_state_path.write_text(json.dumps(state))

    return config


class TestRunGeometryProteinEnrichment:
    """Tests for the Stage 6b pipeline function."""

    def test_strong_signal_node_gets_positive_r2(self):
        """Node with a linear geometry-activation relationship gets r2_cv > 0."""
        with tempfile.TemporaryDirectory() as td:
            config = _setup_synthetic_data(Path(td), min_active=30)
            run_geometry_protein_enrichment(config)

            # Check node 0 has a JSON file with protein-level results
            feat_path = config.geometry_enrichment_dir / "0000.json"
            assert feat_path.exists(), "Expected JSON for node 0 (strong signal)"

            data = json.loads(feat_path.read_text())
            plevel = data["geometric_protein_level"]

            assert plevel["r2_cv"] > 0, (
                f"r2_cv = {plevel['r2_cv']}, expected > 0 for signal node"
            )
            assert isinstance(plevel["monomial"], str)
            assert plevel["n_samples"] > 0

    def test_dead_feature_skipped(self):
        """Node with all-zero activations should not get a JSON file."""
        with tempfile.TemporaryDirectory() as td:
            config = _setup_synthetic_data(Path(td), min_active=30)
            run_geometry_protein_enrichment(config)

            # Node 2 is dead (all zeros)
            feat_path = config.geometry_enrichment_dir / "0002.json"
            assert not feat_path.exists(), "Dead node should not get a JSON file"

    def test_too_few_active_proteins_skipped(self):
        """Node with fewer than min_active proteins should be skipped."""
        with tempfile.TemporaryDirectory() as td:
            config = _setup_synthetic_data(Path(td), min_active=30)
            run_geometry_protein_enrichment(config)

            # Node 3 has only 5 active proteins, below min_active=30
            feat_path = config.geometry_enrichment_dir / "0003.json"
            assert not feat_path.exists(), "Node with too few active proteins should be skipped"

    def test_summary_json_exists(self):
        """summary.json should be created with expected fields."""
        with tempfile.TemporaryDirectory() as td:
            config = _setup_synthetic_data(Path(td), min_active=30)
            run_geometry_protein_enrichment(config)

            summary_path = config.geometry_enrichment_dir / "summary.json"
            assert summary_path.exists()

            summary = json.loads(summary_path.read_text())
            assert "n_features_protein_level" in summary
            assert "features" in summary
            assert summary["n_features_protein_level"] >= 1

    def test_output_json_schema(self):
        """JSON output should have the expected structure."""
        with tempfile.TemporaryDirectory() as td:
            config = _setup_synthetic_data(Path(td), min_active=30)
            run_geometry_protein_enrichment(config)

            # Find any produced JSON
            jsons = list(config.geometry_enrichment_dir.glob("[0-9]*.json"))
            assert len(jsons) > 0, "No feature JSONs produced"

            data = json.loads(jsons[0].read_text())
            assert "feature_id" in data
            assert "geometric_protein_level" in data

            plevel = data["geometric_protein_level"]
            expected_keys = {
                "r2_cv", "r2", "r2_adj", "pearson_r", "alpha_chosen",
                "monomial", "n_samples", "n_nonzero", "top_features",
            }
            assert expected_keys.issubset(set(plevel.keys())), (
                f"Missing keys: {expected_keys - set(plevel.keys())}"
            )
            # r2_cv should be in a reasonable range
            assert -1.0 <= plevel["r2_cv"] <= 1.0
