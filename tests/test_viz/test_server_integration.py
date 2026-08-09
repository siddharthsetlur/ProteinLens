"""Integration tests for the active layer-scoped GeoPedia API."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from proteinlens.viz.server import _load_one_layer, create_app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = [
    PROJECT_ROOT / "feature_data_test_500",
    PROJECT_ROOT / "feature_data_test_20",
]


def _pick_data_dir() -> Path:
    for candidate in CANDIDATES:
        if (candidate / "dataset_stats.json").exists():
            return candidate
    pytest.skip("No GeoPedia test data directory found")


@pytest.fixture(scope="module")
def layer_state():
    return _load_one_layer(_pick_data_dir())


@pytest.fixture(scope="module")
def client(layer_state):
    return TestClient(create_app({layer_state.layer: layer_state}))


def test_top_level_routes(client, layer_state):
    layers = client.get("/api/layers")
    assert layers.status_code == 200
    assert layers.json()[0]["layer"] == layer_state.layer
    assert client.get("/api/landing").status_code == 200
    assert client.get("/api/featured").status_code == 200


def test_layer_stats_contract(client, layer_state):
    response = client.get(f"/api/layers/{layer_state.layer}/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["layer"] == layer_state.layer
    assert {"dataset", "sae", "pipeline"} <= payload.keys()


def test_index_uses_active_seven_method_schema(client, layer_state):
    response = client.get(f"/api/layers/{layer_state.layer}/index")
    assert response.status_code == 200
    rows = response.json()
    assert rows
    row = rows[0]
    assert {"feature_id", "max_activation", "pct_proteins_activated"} <= row.keys()
    for method in range(1, 8):
        assert {f"m{method}_score", f"m{method}_label", f"m{method}_q"} <= row.keys()


def test_significance_and_feature_routes(client, layer_state):
    feature_id = layer_state.feature_index[0]["feature_id"]
    significance = client.get(
        f"/api/layers/{layer_state.layer}/feature/{feature_id}/significance"
    )
    assert significance.status_code == 200
    assert significance.json()["feature_id"] == feature_id

    feature_path = layer_state.analysis_dir / "features" / f"{feature_id:04d}.json"
    feature = client.get(
        f"/api/layers/{layer_state.layer}/feature/{feature_id}"
    )
    if feature_path.exists():
        assert feature.status_code == 200
        assert {"top_sequences", "activation_bins"} <= feature.json().keys()
    else:
        assert feature.status_code == 404


def test_method_coverage_contract(client, layer_state):
    response = client.get(f"/api/layers/{layer_state.layer}/method-coverage")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["methods"]) == 7
    assert payload["total_features"] == len(layer_state.feature_index)
    assert 0 <= payload["total_annotated_n"] <= payload["total_features"]


def test_unknown_layer_and_feature_are_404(client, layer_state):
    assert client.get("/api/layers/999/stats").status_code == 404
    assert (
        client.get(
            f"/api/layers/{layer_state.layer}/feature/999999/significance"
        ).status_code
        == 404
    )


def test_spa_shell(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "GeoPedia" in response.text
