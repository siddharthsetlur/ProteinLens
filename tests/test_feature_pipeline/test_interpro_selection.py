"""Tests for Stage 5a — InterPro stratified selection (bin assignment).

Uses synthetic survey outputs to verify the 11-bin sampling logic:
- "0.0" bin contains only truly inactive proteins (activation == 0)
- 10 normalised bins assign proteins to correct activation ranges
- Deterministic random sampling for reproducibility
- Edge cases: dead features, underpopulated bins
"""

import json

import numpy as np
import pytest

from proteinlens.analysis.feature_pipeline.checkpoint import PipelineState
from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.interpro_selection import (
    run_interpro_selection,
)


@pytest.fixture
def interpro_selection_setup(tmp_path):
    """Set up synthetic survey outputs for InterPro selection testing.

    Creates a scenario with 200 proteins and 3 features:
    - Feature 0: max = 10.0, activations spread across all bins
    - Feature 1: max = 0.0 (dead feature)
    - Feature 2: max = 5.0, only a few proteins activated

    Uses interpro_n_per_bin=3 for compact testing.
    """
    config = PipelineConfig(
        output_dir=tmp_path,
        interpro_n_per_bin=3,
        interpro_collect_residue_activations=False,
    )

    n_proteins = 200
    n_features = 3
    accessions = [f"PROT{i:04d}" for i in range(n_proteins)]

    data = np.zeros((n_proteins, n_features), dtype="float32")

    # Feature 0: max = 10.0, activations spread out
    # Proteins 0-49: activation = 0 (inactive)
    # Proteins 50-59: activation in (0, 1.0] (bin 0.0-0.1)
    # Proteins 60-69: activation in (1.0, 2.0] (bin 0.1-0.2)
    # ... and so on up to bin 0.9-1.0
    for i in range(50, 60):
        data[i, 0] = 0.5  # normalised = 0.05 -> bin [0.0, 0.1]
    for i in range(60, 70):
        data[i, 0] = 1.5  # normalised = 0.15 -> bin [0.1, 0.2]
    for i in range(70, 80):
        data[i, 0] = 2.5  # normalised = 0.25 -> bin [0.2, 0.3]
    for i in range(80, 90):
        data[i, 0] = 3.5  # normalised = 0.35 -> bin [0.3, 0.4]
    for i in range(90, 100):
        data[i, 0] = 4.5  # normalised = 0.45 -> bin [0.4, 0.5]
    for i in range(100, 110):
        data[i, 0] = 5.5  # normalised = 0.55 -> bin [0.5, 0.6]
    for i in range(110, 120):
        data[i, 0] = 6.5  # normalised = 0.65 -> bin [0.6, 0.7]
    for i in range(120, 130):
        data[i, 0] = 7.5  # normalised = 0.75 -> bin [0.7, 0.8]
    for i in range(130, 140):
        data[i, 0] = 8.5  # normalised = 0.85 -> bin [0.8, 0.9]
    for i in range(140, 150):
        data[i, 0] = 9.5  # normalised = 0.95 -> bin [0.9, 1.0]
    data[149, 0] = 10.0  # exactly at max

    # Feature 1: all zeros (dead feature)

    # Feature 2: max = 5.0, sparse activation
    data[160, 2] = 5.0
    data[161, 2] = 4.0
    data[162, 2] = 0.3  # normalised = 0.06 -> bin [0.0, 0.1]

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
    np.save(config.feature_max_path, np.max(data, axis=0))

    # Save pipeline_state.json
    state = PipelineState(config.pipeline_state_path)
    state.set_accession_index({a: i for i, a in enumerate(accessions)})
    state.set_total_proteins(n_proteins)

    return config, accessions, data


class TestInterProBinAssignment:
    """Tests for the 11-bin stratified sampling logic."""

    def test_output_structure(self, interpro_selection_setup):
        """run_interpro_selection should create interpro_selection.json
        with expected top-level keys and per-feature bin structure."""
        config, _, _ = interpro_selection_setup
        result = run_interpro_selection(config)

        assert config.interpro_selection_path.exists()
        assert "per_feature" in result
        assert "all_selected_accessions" in result

        # Each feature should have a "bins" dict with 11 keys
        feat0 = result["per_feature"]["0"]
        assert "bins" in feat0
        assert "0.0" in feat0["bins"]
        assert "0.0-0.1" in feat0["bins"]
        assert "0.9-1.0" in feat0["bins"]
        assert len(feat0["bins"]) == 11  # 1 zero-bin + 10 normalised

    def test_zero_bin_contains_only_inactive(self, interpro_selection_setup):
        """The '0.0' bin should contain only proteins with activation == 0."""
        config, accessions, data = interpro_selection_setup
        result = run_interpro_selection(config)

        feat0_zero_bin = result["per_feature"]["0"]["bins"]["0.0"]
        # All proteins in the zero bin should have activation == 0
        for acc in feat0_zero_bin:
            idx = accessions.index(acc)
            assert data[idx, 0] == 0.0, (
                f"{acc} is in the '0.0' bin but has activation "
                f"{data[idx, 0]} (should be exactly 0.0)"
            )

    def test_zero_bin_respects_n_per_bin(self, interpro_selection_setup):
        """The '0.0' bin should have at most interpro_n_per_bin proteins."""
        config, _, _ = interpro_selection_setup
        result = run_interpro_selection(config)

        feat0_zero_bin = result["per_feature"]["0"]["bins"]["0.0"]
        # We have 50 inactive proteins but n_per_bin=3
        assert len(feat0_zero_bin) == config.interpro_n_per_bin

    def test_zero_bin_deterministic(self, interpro_selection_setup):
        """Running twice with the same seed should produce identical '0.0' bins."""
        config, _, _ = interpro_selection_setup
        result1 = run_interpro_selection(config)
        result2 = run_interpro_selection(config)

        assert (
            result1["per_feature"]["0"]["bins"]["0.0"]
            == result2["per_feature"]["0"]["bins"]["0.0"]
        )

    def test_normalised_bins_correct_range(self, interpro_selection_setup):
        """Proteins in a normalised bin should have activations in that range."""
        config, accessions, data = interpro_selection_setup
        result = run_interpro_selection(config)

        feat_max = 10.0  # Feature 0
        # Check bin 0.5-0.6: absolute range (5.0, 6.0]
        bin_accs = result["per_feature"]["0"]["bins"]["0.5-0.6"]
        for acc in bin_accs:
            idx = accessions.index(acc)
            val = data[idx, 0]
            assert 5.0 < val <= 6.0, (
                f"{acc} in bin 0.5-0.6 has activation {val}, "
                f"expected in (5.0, 6.0] (feat_max={feat_max})"
            )

    def test_lowest_nonzero_bin_excludes_zero(self, interpro_selection_setup):
        """The [0.0-0.1] bin must NOT contain proteins with activation == 0."""
        config, accessions, data = interpro_selection_setup
        result = run_interpro_selection(config)

        bin_0_01 = result["per_feature"]["0"]["bins"]["0.0-0.1"]
        for acc in bin_0_01:
            idx = accessions.index(acc)
            assert data[idx, 0] > 0.0, (
                f"{acc} in bin 0.0-0.1 has activation 0.0 — "
                "should be excluded (use '0.0' bin instead)"
            )

    def test_dead_feature_has_empty_normalised_bins(
        self, interpro_selection_setup
    ):
        """A feature with max == 0 should have all empty normalised bins.
        The '0.0' bin may still contain proteins."""
        config, _, _ = interpro_selection_setup
        result = run_interpro_selection(config)

        feat1 = result["per_feature"]["1"]
        for bin_label, bin_proteins in feat1["bins"].items():
            if bin_label != "0.0":
                assert len(bin_proteins) == 0, (
                    f"Dead feature bin '{bin_label}' should be empty, "
                    f"got {bin_proteins}"
                )

    def test_dead_feature_zero_bin_has_all_proteins(
        self, interpro_selection_setup
    ):
        """For a dead feature (max == 0), every protein has activation 0,
        so the '0.0' bin should have n_per_bin sampled proteins."""
        config, _, _ = interpro_selection_setup
        result = run_interpro_selection(config)

        # Feature 1 is all zeros — all 200 proteins are inactive.
        # With n_per_bin=3 we should get exactly 3.
        feat1_zero_bin = result["per_feature"]["1"]["bins"]["0.0"]
        assert len(feat1_zero_bin) == config.interpro_n_per_bin

    def test_underpopulated_bin(self, interpro_selection_setup):
        """A bin with fewer proteins than n_per_bin should return all of them."""
        config, _, _ = interpro_selection_setup
        result = run_interpro_selection(config)

        # Feature 2: only PROT162 has activation 0.3 (normalised 0.06),
        # which lands in bin [0.0-0.1] (absolute (0, 0.5])
        feat2_bin_0_01 = result["per_feature"]["2"]["bins"]["0.0-0.1"]
        assert "PROT0162" in feat2_bin_0_01
        assert len(feat2_bin_0_01) == 1  # Only 1 protein, fewer than n_per_bin=3

    def test_all_selected_is_union(self, interpro_selection_setup):
        """all_selected_accessions should be the union of all per-feature bins."""
        config, _, _ = interpro_selection_setup
        result = run_interpro_selection(config)

        expected = set()
        for feat_data in result["per_feature"].values():
            for bin_accs in feat_data["bins"].values():
                expected.update(bin_accs)

        assert set(result["all_selected_accessions"]) == expected


class TestReproducibility:
    """Plan item 5.2: verify that Stage 5a is deterministic."""

    def test_identical_output_on_repeated_runs(self, interpro_selection_setup):
        """Running run_interpro_selection twice on the same input must
        produce byte-identical JSON output.  This is essential for
        scientific reproducibility — if the selection changes between
        runs, downstream F1 scores become non-reproducible."""
        config, _, _ = interpro_selection_setup

        # First run
        run_interpro_selection(config)
        with open(config.interpro_selection_path, "r") as f:
            json_1 = f.read()

        # Second run (overwrites the same file)
        run_interpro_selection(config)
        with open(config.interpro_selection_path, "r") as f:
            json_2 = f.read()

        assert json_1 == json_2, (
            "interpro_selection.json differs between two runs on identical input. "
            "This breaks scientific reproducibility."
        )
