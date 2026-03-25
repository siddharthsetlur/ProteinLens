"""Tests for Stage 2 — selection (bin assignment and protein selection).

These tests use synthetic survey outputs to verify the bin logic.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from proteinlens.analysis.feature_pipeline.checkpoint import PipelineState
from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.selection import run_selection


@pytest.fixture
def selection_setup(tmp_path):
    """Set up synthetic survey outputs for selection testing.

    Creates a scenario with 10 proteins and 3 features, where activation
    values are arranged to test bin assignment logic.
    """
    config = PipelineConfig(
        output_dir=tmp_path,
        n_top_per_feature=2,
        n_per_bin=2,
        activation_bins=[0.0, 0.25, 0.5, 0.75, 1.0],
    )

    accessions = [f"PROT{i}" for i in range(10)]
    n_features = 3

    # Create a memmap with controlled activation values.
    # Feature 0: max = 4.0, bins at [0, 1), [1, 2), [2, 3), [3, 4]
    # Feature 1: max = 0.0 (dead feature — should have empty bins)
    # Feature 2: max = 2.0, bins at [0, 0.5), [0.5, 1), [1, 1.5), [1.5, 2]
    data = np.zeros((10, 3), dtype="float32")

    # Feature 0 activations:
    data[0, 0] = 4.0   # PROT0 — top bin [3, 4]
    data[1, 0] = 3.5   # PROT1 — top bin [3, 4]
    data[2, 0] = 2.5   # PROT2 — bin [2, 3)
    data[3, 0] = 1.5   # PROT3 — bin [1, 2)
    data[4, 0] = 0.5   # PROT4 — bin [0, 1)
    data[5, 0] = 0.0   # PROT5 — not activated

    # Feature 1: all zeros (dead feature)

    # Feature 2 activations:
    data[6, 2] = 2.0   # PROT6 — top bin [1.5, 2]
    data[7, 2] = 1.0   # PROT7 — bin [0.5, 1)  — PM NOTE: 1.0 = 0.5*max, edge case
    data[8, 2] = 0.3   # PROT8 — bin [0, 0.5)
    data[9, 2] = 1.8   # PROT9 — top bin [1.5, 2]

    # Save memmap
    memmap_path = config.protein_feature_maxes_path
    mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=(10, 3))
    mm[:] = data
    mm.flush()

    # Save global max
    np.save(config.feature_max_path, np.max(data, axis=0))

    # Save survey_top20.json
    # Feature 0 top-2: PROT0 (4.0), PROT1 (3.5)
    # Feature 1 top-2: empty
    # Feature 2 top-2: PROT6 (2.0), PROT9 (1.8)
    survey_top = {
        "0": [
            {"accession": "PROT0", "max_activation": 4.0},
            {"accession": "PROT1", "max_activation": 3.5},
        ],
        "1": [],
        "2": [
            {"accession": "PROT6", "max_activation": 2.0},
            {"accession": "PROT9", "max_activation": 1.8},
        ],
    }
    with open(config.survey_top_path, "w") as f:
        json.dump(survey_top, f)

    # Save pipeline_state.json
    state = PipelineState(config.pipeline_state_path)
    state.set_accession_index({a: i for i, a in enumerate(accessions)})
    state.set_total_proteins(10)

    return config, accessions, data


class TestRunSelection:
    """Tests for the selection stage."""

    def test_selection_outputs_exist(self, selection_setup):
        """run_selection should create selection.json."""
        config, _, _ = selection_setup
        result = run_selection(config)

        assert config.selection_path.exists()
        assert "per_feature" in result
        assert "all_selected_accessions" in result

    def test_top_proteins_included(self, selection_setup):
        """Top proteins from survey should appear in selection."""
        config, _, _ = selection_setup
        result = run_selection(config)

        feat0 = result["per_feature"]["0"]
        assert "PROT0" in feat0["top"]
        assert "PROT1" in feat0["top"]

    def test_dead_feature_has_empty_bins(self, selection_setup):
        """A feature with max_activation=0 should have all empty bins."""
        config, _, _ = selection_setup
        result = run_selection(config)

        feat1 = result["per_feature"]["1"]
        for bin_label, bin_proteins in feat1["bins"].items():
            assert len(bin_proteins) == 0, (
                f"Dead feature bin {bin_label} should be empty, "
                f"got {bin_proteins}"
            )

    def test_bin_assignment_correctness(self, selection_setup):
        """Proteins should land in the correct normalised bins."""
        config, _, data = selection_setup
        result = run_selection(config)

        feat0_bins = result["per_feature"]["0"]["bins"]

        # Feature 0 max = 4.0
        # PROT4 has activation 0.5 → 0.5/4.0 = 0.125 → bin [0.0, 0.25)
        # But the bin logic uses absolute thresholds:
        #   bin "0.0-0.25" → (0, 1.0]  (0.25 * 4.0 = 1.0)
        #   bin "0.25-0.5" → (1.0, 2.0]
        #   bin "0.5-0.75" → (2.0, 3.0]
        #   bin "0.75-1.0" → (3.0, 4.0]
        assert "PROT4" in feat0_bins["0.0-0.25"]  # 0.5 ∈ (0, 1.0]
        assert "PROT3" in feat0_bins["0.25-0.5"]  # 1.5 ∈ (1.0, 2.0]
        assert "PROT2" in feat0_bins["0.5-0.75"]  # 2.5 ∈ (2.0, 3.0]

        # PROT0 (4.0) and PROT1 (3.5) should be in the top bin
        top_bin = feat0_bins["0.75-1.0"]
        assert "PROT0" in top_bin or "PROT1" in top_bin

    def test_all_selected_is_union(self, selection_setup):
        """all_selected_accessions should be the union of all per-feature selections."""
        config, _, _ = selection_setup
        result = run_selection(config)

        expected = set()
        for feat_data in result["per_feature"].values():
            expected.update(feat_data["top"])
            for bin_accs in feat_data["bins"].values():
                expected.update(bin_accs)

        assert set(result["all_selected_accessions"]) == expected
