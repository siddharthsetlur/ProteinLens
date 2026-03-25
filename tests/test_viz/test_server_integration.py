"""
Integration tests for the SAE feature visualizer server.

Tests the full stack: index builder -> API endpoints -> response schemas.
Validates that field names returned by the API match what the frontend JS expects.

Runs against feature_data_test_500 (which has real InterPro enrichment data)
to catch field name mismatches between the pipeline output and the frontend.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from proteinlens.viz.server import create_app

# --- Fixture: shared test client ---

# Use test_500 if available (has InterPro enrichment), fall back to test_20
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR_500 = PROJECT_ROOT / "feature_data_test_500"
DATA_DIR_20 = PROJECT_ROOT / "feature_data_test_20"


def _pick_data_dir() -> Path:
    """Pick the best available test data directory."""
    if DATA_DIR_500.exists() and (DATA_DIR_500 / "dataset_stats.json").exists():
        return DATA_DIR_500
    if DATA_DIR_20.exists() and (DATA_DIR_20 / "dataset_stats.json").exists():
        return DATA_DIR_20
    pytest.skip("No test data directory found")


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient against the best available data directory."""
    data_dir = _pick_data_dir()
    app = create_app(data_dir)
    return TestClient(app)


@pytest.fixture(scope="module")
def data_dir():
    """Return the data directory path."""
    return _pick_data_dir()


# ================================================================
# API endpoint tests
# ================================================================


class TestStatsEndpoint:
    """Tests for GET /api/stats."""

    def test_stats_returns_200(self, client):
        resp = client.get("/api/stats")
        assert resp.status_code == 200

    def test_stats_has_required_keys(self, client):
        data = client.get("/api/stats").json()
        assert "dataset" in data
        assert "sae" in data
        assert "pipeline" in data

    def test_stats_dataset_fields(self, client):
        ds = client.get("/api/stats").json()["dataset"]
        # Fields the homepage.js renderDatasetCard() accesses
        for field in ["esm_model", "esm_layer", "total_proteins", "total_clusters", "num_features"]:
            assert field in ds, f"dataset.{field} missing — homepage.js will show '—'"

    def test_stats_pipeline_fields(self, client):
        pipeline = client.get("/api/stats").json()["pipeline"]
        # Fields the homepage.js renderPipelineCard() accesses
        for field in ["completed_stages", "feature_count", "interpro_count", "geometry_count"]:
            assert field in pipeline, f"pipeline.{field} missing — homepage.js will show '—'"


class TestIndexEndpoint:
    """Tests for GET /api/index."""

    def test_index_returns_list(self, client):
        data = client.get("/api/index").json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_index_row_fields(self, client):
        """Verify each row has the fields that homepage.js AG Grid column defs reference."""
        row = client.get("/api/index").json()[0]
        expected_fields = [
            "feature_id", "max_activation",
            "pct_proteins_activated", "pct_clusters_activated",
            "interpro_protein_best_f1", "interpro_protein_best_name",
            "interpro_residue_best_f1",
            "geometry_protein_r2_cv", "geometry_residue_gbm_auc_cv",
        ]
        for field in expected_fields:
            assert field in row, f"index row missing '{field}' — AG Grid column will be empty"


class TestFeatureEndpoint:
    """Tests for GET /api/feature/{id}."""

    def test_feature_returns_200(self, client):
        resp = client.get("/api/feature/0")
        assert resp.status_code == 200

    def test_feature_has_required_keys(self, client):
        data = client.get("/api/feature/0").json()
        # Fields accessed by feature_detail.js
        for field in ["top_sequences", "activation_bins", "dataset_coverage"]:
            assert field in data, f"feature.{field} missing — feature_detail.js will fail"

    def test_feature_404_for_missing(self, client):
        resp = client.get("/api/feature/999999")
        assert resp.status_code == 404


class TestInterproFieldNames:
    """
    Verify that InterPro enrichment field names match what feature_detail.js expects.

    This is the most critical test — the original implementation used wrong field names
    (e.g., "f1" instead of "best_f1") causing all InterPro data to display as dashes.
    """

    def _get_interpro_with_data(self, client, data_dir):
        """Find a feature that has InterPro enrichment data and return its JSON."""
        import json
        interpro_dir = data_dir / "interpro_enrichment"
        if not interpro_dir.is_dir():
            pytest.skip("No interpro_enrichment directory")

        for fpath in sorted(interpro_dir.iterdir()):
            if fpath.name == "summary.json" or fpath.suffix != ".json":
                continue
            d = json.loads(fpath.read_text())
            if d.get("protein_level") and len(d["protein_level"]) > 0:
                fid = d.get("feature_id", int(fpath.stem))
                resp = client.get(f"/api/feature/{fid}/interpro")
                if resp.status_code == 200:
                    return resp.json()

        pytest.skip("No features with InterPro enrichment data found")

    def test_protein_level_field_names(self, client, data_dir):
        """Check that protein_level entries have the exact field names the JS accesses."""
        data = self._get_interpro_with_data(client, data_dir)
        entry = data["protein_level"][0]

        # These are the field names used in renderInterproProteinCard() after the fix
        expected = [
            "best_f1", "best_threshold", "best_threshold_normalized",
            "precision_at_best", "recall_at_best",
            "n_true_positives", "n_false_positives", "n_false_negatives",
            "annotation_name",
        ]
        for field in expected:
            assert field in entry, (
                f"InterPro protein_level missing '{field}' — "
                f"feature_detail.js renderInterproProteinCard() will show '—'. "
                f"Available fields: {list(entry.keys())}"
            )

    def test_residue_level_field_names(self, client, data_dir):
        """Check that residue_level entries have the exact field names the JS accesses."""
        data = self._get_interpro_with_data(client, data_dir)
        if not data.get("residue_level"):
            pytest.skip("No residue-level enrichment data")

        entry = data["residue_level"][0]

        # These are the field names used in renderInterproResidueCard() after the fix
        expected = [
            "best_f1", "best_threshold", "best_threshold_normalized",
            "precision_at_best", "recall_at_best",
            "n_residues_in_domain", "n_total_residues",
            "annotation_name",
        ]
        for field in expected:
            assert field in entry, (
                f"InterPro residue_level missing '{field}' — "
                f"feature_detail.js renderInterproResidueCard() will show '—'. "
                f"Available fields: {list(entry.keys())}"
            )


class TestPdbEndpoint:
    """Tests for GET /api/pdb/{accession}."""

    def test_pdb_returns_text(self, client, data_dir):
        """Find a real accession from the pdb_cache and verify the endpoint serves it."""
        pdb_dir = data_dir / "pdb_cache"
        if not pdb_dir.is_dir():
            pytest.skip("No pdb_cache directory")

        pdbs = list(pdb_dir.glob("AF-*-F1-model_v*.pdb"))
        if not pdbs:
            pytest.skip("No PDB files in cache")

        # Extract accession from filename: AF-{accession}-F1-model_v*.pdb
        filename = pdbs[0].name
        accession = filename.split("-")[1]

        resp = client.get(f"/api/pdb/{accession}")
        assert resp.status_code == 200
        assert "ATOM" in resp.text  # PDB files contain ATOM records

    def test_pdb_rejects_traversal(self, client):
        """Verify path traversal attempts are rejected."""
        resp = client.get("/api/pdb/../../etc/passwd")
        assert resp.status_code in (400, 404, 422)


class TestAccessionValidation:
    """Tests for accession input validation."""

    def test_valid_accession_format(self, client):
        resp = client.get("/api/interpro/A0A087X1C5")
        # Should be 200 or 404, never 400
        assert resp.status_code in (200, 404)

    def test_invalid_accession_rejected(self, client):
        resp = client.get("/api/interpro/../../../etc/passwd")
        assert resp.status_code in (400, 404, 422)


class TestPageRoutes:
    """Tests for HTML page routes."""

    def test_homepage_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "SAE Feature Visualizer" in resp.text

    def test_feature_page_serves_html(self, client):
        resp = client.get("/feature/0")
        assert resp.status_code == 200
        assert "Feature Detail" in resp.text
