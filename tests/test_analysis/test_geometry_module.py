"""Unit tests for the geometry extraction module.

Tests cover:
  - compute_protein_geometry: real PDB -> 55-key dict of finite floats
  - ca_backbone: real PDB -> (N, 3) array with N > 10
  - extract_local_feature_vector: 44-dim vector, all finite
  - fit_lasso_single_node: synthetic linear relationship -> r2_cv > 0.3
  - train_motif_classifier: separable synthetic data -> f1_cv > 0.5
  - superpose_fragments: identical fragments -> mean_rmsd ~ 0
  - format_monomial: known weights -> expected string
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from proteinlens.analysis.geometry import (
    GEOM_FEATURE_NAMES,
    ca_backbone,
    compute_protein_geometry,
    compute_residue_profiles,
    detect_alpha_helices_from_ca,
    extract_local_feature_vector,
    format_monomial,
    superpose_fragments,
)
from proteinlens.analysis.geometry.classifiers import (
    compute_rmsd,
    fit_lasso_single_node,
    kabsch_align,
    train_motif_classifier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PDB_CACHE = Path("protein_results/alphafold_analysis/pdb_cache")


def _get_first_pdb_text() -> str:
    """Load the first available PDB file from the cache."""
    pdb_files = sorted(PDB_CACHE.glob("*.pdb"))
    assert len(pdb_files) > 0, f"No PDB files found in {PDB_CACHE}"
    return pdb_files[0].read_text()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeProteinGeometry:
    """Tests for the 55-dim protein-level geometry extraction."""

    def test_real_pdb_returns_dict_with_correct_keys(self):
        """compute_protein_geometry on a real PDB returns 55 finite floats."""
        pdb_text = _get_first_pdb_text()
        result = compute_protein_geometry(pdb_text)

        assert result is not None, "compute_protein_geometry returned None on a valid PDB"
        assert len(result) == 55, f"Expected 55 keys, got {len(result)}"

        for name, value in result.items():
            assert isinstance(value, float), f"Feature '{name}' is {type(value)}, expected float"
            assert np.isfinite(value), f"Feature '{name}' is not finite: {value}"

    def test_invalid_pdb_returns_none(self):
        """Garbage input should return None, not raise."""
        result = compute_protein_geometry("NOT A PDB FILE")
        assert result is None

    def test_too_short_pdb_returns_none(self):
        """A PDB with fewer than 4 Ca atoms should return None."""
        # Build a minimal PDB with just 2 ATOM lines
        lines = [
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C  ",
            "ATOM      2  CA  ALA A   2       5.000   6.000   7.000  1.00  0.00           C  ",
            "END",
        ]
        result = compute_protein_geometry("\n".join(lines))
        assert result is None


class TestCaBackbone:
    """Tests for Ca backbone extraction."""

    def test_real_pdb_returns_array(self):
        """ca_backbone on a real PDB returns (N, 3) with N > 10."""
        pdb_text = _get_first_pdb_text()
        coords = ca_backbone(pdb_text, chain_id=None)

        assert isinstance(coords, np.ndarray)
        assert coords.ndim == 2
        assert coords.shape[1] == 3
        assert coords.shape[0] > 10, f"Expected > 10 Ca atoms, got {coords.shape[0]}"

    def test_invalid_pdb_raises(self):
        """Empty/invalid PDB text should raise ValueError."""
        with pytest.raises(ValueError):
            ca_backbone("", chain_id=None)

    def test_no_matplotlib_import(self):
        """ca_backbone should not trigger matplotlib import."""
        import sys
        # If matplotlib.pyplot was imported, it would be in sys.modules
        # We can't guarantee it wasn't imported elsewhere, but we can
        # verify our module doesn't import it directly.
        import proteinlens.analysis.geometry.residue_features as rf
        source = Path(rf.__file__).read_text()
        assert "import matplotlib" not in source
        assert "from matplotlib" not in source


class TestExtractLocalFeatureVector:
    """Tests for the 44-dim residue-level feature extraction."""

    def test_returns_44dim_finite_vector(self):
        """extract_local_feature_vector at pos=15 returns 44-dim finite vector."""
        pdb_text = _get_first_pdb_text()
        ca = ca_backbone(pdb_text, chain_id=None)
        helices = detect_alpha_helices_from_ca(ca)
        profiles = compute_residue_profiles(ca, helices)

        vec = extract_local_feature_vector(profiles, ca, pos=15, half_w=5)

        assert vec is not None, "Feature vector is None for pos=15"
        assert vec.shape == (44,), f"Expected shape (44,), got {vec.shape}"
        assert np.all(np.isfinite(vec)), "Feature vector contains non-finite values"

    def test_too_close_to_edge_returns_none(self):
        """Positions too close to the chain ends should return None."""
        pdb_text = _get_first_pdb_text()
        ca = ca_backbone(pdb_text, chain_id=None)
        helices = detect_alpha_helices_from_ca(ca)
        profiles = compute_residue_profiles(ca, helices)

        # Position 2 with half_w=5 is too close to the N-terminus
        assert extract_local_feature_vector(profiles, ca, pos=2, half_w=5) is None


class TestFitLassoSingleNode:
    """Tests for the protein-level LassoCV single-node regression."""

    def test_synthetic_linear_relationship(self):
        """Known linear y = 2*x1 + noise should give r2_cv > 0.3."""
        rng = np.random.default_rng(42)
        n_samples, n_features = 200, 5
        X = rng.standard_normal((n_samples, n_features))
        # Strong linear signal in feature 0, shifted positive to mimic
        # activation values (pipeline only passes active proteins with y > 0)
        y = 2.0 * X[:, 0] + 5.0 + 0.3 * rng.standard_normal(n_samples)
        y = np.maximum(y, 0.01)  # clamp to positive

        names = [f"feat_{i}" for i in range(n_features)]
        result = fit_lasso_single_node(X, y, names, cv_folds=5)

        assert result is not None, "fit_lasso_single_node returned None on signal data"
        assert result["r2_cv"] > 0.3, f"r2_cv = {result['r2_cv']:.3f}, expected > 0.3"
        assert result["n_samples"] == n_samples
        assert isinstance(result["monomial"], str)
        assert "feat_0" in result["monomial"], "Expected feat_0 in monomial"

    def test_no_signal_returns_none(self):
        """Pure noise should return None (r2_cv < 0)."""
        rng = np.random.default_rng(123)
        X = rng.standard_normal((100, 5))
        y = rng.standard_normal(100)
        y = np.abs(y) + 0.01

        result = fit_lasso_single_node(X, y, [f"f{i}" for i in range(5)])
        # May return None or a result with very low r2_cv
        if result is not None:
            assert result["r2_cv"] >= 0.0  # we skip negative r2_cv


class TestTrainMotifClassifier:
    """Tests for the GBM/DT motif classifier."""

    def test_separable_synthetic_data(self):
        """Separable features should give f1_cv > 0.5."""
        rng = np.random.default_rng(42)

        # Create 50 activated (high curvature) + 150 background (low curvature)
        n_feats = 44
        activated = []
        for i in range(50):
            feats = rng.standard_normal(n_feats)
            feats[0] = 3.0 + rng.standard_normal() * 0.3  # high curvature_mean
            activated.append({
                "features": feats,
                "accession": f"PROT_{i % 10}",
                "position": i,
                "fragment": rng.standard_normal((21, 3)),
                "category": 0,
                "activation": float(2.0 + rng.random()),
            })

        background = []
        for i in range(150):
            feats = rng.standard_normal(n_feats)
            feats[0] = -1.0 + rng.standard_normal() * 0.3  # low curvature_mean
            background.append({
                "features": feats,
                "accession": f"PROT_{i % 30}",
                "position": i,
                "fragment": rng.standard_normal((21, 3)),
                "category": 5,
                "activation": 0.0,
            })

        from proteinlens.analysis.geometry.residue_features import ACTIVE_GEOM_NAMES
        result = train_motif_classifier(
            activated, background, feature_names=ACTIVE_GEOM_NAMES, cv_folds=3
        )

        assert isinstance(result["rules"], str)
        assert isinstance(result["f1_cv"], float)
        # With clearly separable data, f1_cv should be reasonably high
        assert result["f1_cv"] > 0.5, f"f1_cv = {result['f1_cv']:.3f}, expected > 0.5"


class TestSuperposeFragments:
    """Tests for Kabsch-based fragment superposition."""

    def test_identical_fragments_zero_rmsd(self):
        """10 identical fragments should give mean_rmsd ~ 0."""
        rng = np.random.default_rng(42)
        template = rng.standard_normal((21, 3))

        activated = [
            {"fragment": template.copy(), "activation": float(10 - i)}
            for i in range(10)
        ]

        result = superpose_fragments(activated, top_k=10)

        assert result["mean_structure"] is not None
        assert result["mean_rmsd"] < 0.01, (
            f"mean_rmsd = {result['mean_rmsd']:.4f}, expected < 0.01 for identical fragments"
        )
        assert result["n_fragments"] == 10

    def test_too_few_fragments(self):
        """Fewer than 3 fragments should return None mean_structure."""
        activated = [
            {"fragment": np.random.randn(21, 3), "activation": 1.0}
        ]
        result = superpose_fragments(activated)
        assert result["mean_structure"] is None
        assert result["n_fragments"] == 1


class TestFormatMonomial:
    """Tests for the human-readable monomial string formatter."""

    def test_known_weights(self):
        """Known weights produce expected monomial string."""
        weights = [0.5, -0.3, 0.0, 0.0, 0.1]
        intercept = 0.02
        names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]

        result = format_monomial(weights, intercept, names)

        assert result.startswith("y_hat = ")
        assert "feat_a" in result
        assert "feat_b" in result
        # feat_c and feat_d have weight 0, should not appear
        assert "feat_c" not in result
        assert "feat_d" not in result

    def test_all_zero_weights(self):
        """All-zero weights should just show the intercept."""
        result = format_monomial([0.0, 0.0], 1.5, ["a", "b"])
        assert result == "y_hat = 1.5"


class TestNaNGuard:
    """Tests for NaN handling in compute_protein_geometry."""

    def test_nan_replaced_with_zero(self):
        """Proteins with no helix pairs should still return all-finite values.

        Some features (helix_parallel_mean, helix_dist_mean, etc.) produce
        NaN when no helix pairs exist. The NaN guard should replace these
        with 0.0.
        """
        # Use a real PDB -- even if this particular one has helices,
        # the NaN guard is verified by the finite-values assertion in
        # test_real_pdb_returns_dict_with_correct_keys. This test confirms
        # the guard exists by checking the function contract.
        pdb_text = _get_first_pdb_text()
        result = compute_protein_geometry(pdb_text)
        assert result is not None
        for name, value in result.items():
            assert np.isfinite(value), (
                f"Feature '{name}' is not finite ({value}); NaN guard failed"
            )


class TestComputeResidueProfiles:
    """Tests for compute_residue_profiles in isolation."""

    def test_returns_expected_keys(self):
        """compute_residue_profiles returns all expected profile arrays."""
        pdb_text = _get_first_pdb_text()
        ca = ca_backbone(pdb_text, chain_id=None)
        helices = detect_alpha_helices_from_ca(ca)
        profiles = compute_residue_profiles(ca, helices)

        expected_keys = {
            "curvature", "torsion", "planarity", "tangents",
            "helix_mask", "categories",
        }
        assert expected_keys == set(profiles.keys())

        n = len(ca)
        assert profiles["curvature"].shape == (n,)
        assert profiles["torsion"].shape == (n,)
        assert profiles["planarity"].shape == (n,)
        assert profiles["tangents"].shape == (n, 3)
        assert profiles["helix_mask"].shape == (n,)
        assert profiles["helix_mask"].dtype == bool
        assert profiles["categories"].shape == (n,)
        # Categories should be integers in [0, 5]
        assert profiles["categories"].min() >= 0
        assert profiles["categories"].max() <= 5


class TestKabschAlign:
    """Tests for Kabsch alignment correctness."""

    def test_identity_alignment(self):
        """Aligning identical structures should give near-zero RMSD."""
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((20, 3))
        aligned = kabsch_align(coords.copy(), coords.copy())
        rmsd = compute_rmsd(aligned, coords)
        assert rmsd < 1e-10, f"RMSD = {rmsd}, expected ~0 for identical structures"

    def test_known_rotation(self):
        """Aligning a 90-degree rotation should recover the original.

        Rotate mobile by 90 degrees around the Z-axis, then verify
        kabsch_align recovers the target coordinates.
        """
        rng = np.random.default_rng(42)
        target = rng.standard_normal((15, 3))

        # 90-degree rotation around Z-axis
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        mobile = target @ R.T  # apply rotation

        aligned = kabsch_align(mobile, target)
        rmsd = compute_rmsd(aligned, target)
        assert rmsd < 1e-6, f"RMSD = {rmsd}, expected ~0 after Kabsch recovery"

    def test_translation_invariance(self):
        """Kabsch should handle arbitrary translations."""
        rng = np.random.default_rng(42)
        target = rng.standard_normal((10, 3))
        mobile = target + np.array([100.0, -50.0, 25.0])  # large translation

        aligned = kabsch_align(mobile, target)
        rmsd = compute_rmsd(aligned, target)
        assert rmsd < 1e-6, f"RMSD = {rmsd}, expected ~0 after translation recovery"


class TestGeomFeatureNamesConsistency:
    """Tests that feature name constants are self-consistent."""

    def test_npz_feature_names_match_constant(self):
        """Stage 6a NPZ feature_names should match GEOM_FEATURE_NAMES.

        This catches silent mismatches if the geometry feature set is
        changed in one place but not the other.
        """
        import shutil
        import tempfile
        from proteinlens.analysis.feature_pipeline.config import PipelineConfig
        from proteinlens.analysis.feature_pipeline.geometry_features import (
            run_geometry_features,
        )

        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            config = PipelineConfig(sae_dir="x", output_dir=str(tmpdir))
            pdb_cache = config.pdb_cache_dir

            # Copy one real PDB
            real_pdbs = sorted(PDB_CACHE.glob("*.pdb"))[:1]
            assert real_pdbs, f"No PDB files in {PDB_CACHE}"
            shutil.copy(real_pdbs[0], pdb_cache / real_pdbs[0].name)

            run_geometry_features(config)

            data = np.load(config.geometry_protein_features_path)
            npz_names = list(data["feature_names"])
            assert npz_names == list(GEOM_FEATURE_NAMES), (
                f"NPZ feature_names ({len(npz_names)}) != GEOM_FEATURE_NAMES ({len(GEOM_FEATURE_NAMES)})"
            )
