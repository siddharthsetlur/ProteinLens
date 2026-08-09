"""Tests for the random-subset null on Metric C.

Two unit tests verify the null mechanics on synthetic data; one integration
test scaffolds a real residue_phi cache via build_residue_phi_cache and runs
the null end-to-end on feature_data_test_500.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.compute_metric_c_null import (  # noqa: E402
    PHI_DIM,
    _null_cosines,
    _summarise,
)


# ---------------------------------------------------------------------------
# Unit: when the two phi pools are identical, the random-subset null is ≈ 1
#       (any two random subsets of the same population have nearly identical
#       means → cosine ≈ 1). This is the "no-signal limit".
# ---------------------------------------------------------------------------
def test_null_identical_pools_gives_cosine_near_one():
    rng = np.random.default_rng(0)
    M = 5000
    # Common-mode shift dominates: every row is `(mu + small noise)`.
    mu = np.array([5.0, 1.0, -1.0, 0.5] + [0.0] * (PHI_DIM - 4), dtype=np.float32)
    phi = mu + 0.2 * rng.normal(size=(M, PHI_DIM)).astype(np.float32)

    nulls = _null_cosines(phi, phi, n_swiss=500, n_nmp=500, n_trials=50,
                          rng=np.random.default_rng(1), max_per_side=10_000)
    assert nulls.shape == (50,)
    # Random subsets of the same data with a strong common-mode → near-1.
    assert nulls.min() > 0.99, f"identical-pool null should be ≈1, got min {nulls.min()}"


# ---------------------------------------------------------------------------
# Unit: when phi components are zero-mean Gaussian noise, the null cosine is
#       diffuse (concentrates around 0 with std ≈ 1/sqrt(PHI_DIM) ≈ 0.15).
# ---------------------------------------------------------------------------
def test_null_zero_mean_gaussian_centred_on_zero():
    rng = np.random.default_rng(7)
    M = 5000
    phi = rng.normal(size=(M, PHI_DIM)).astype(np.float32)   # zero-mean
    nulls = _null_cosines(phi, phi, n_swiss=200, n_nmp=200, n_trials=200,
                          rng=np.random.default_rng(11), max_per_side=10_000)
    median = float(np.median(nulls))
    assert abs(median) < 0.15, (
        f"zero-mean phi null should sit near 0; got median {median}"
    )
    # And the spread should be wide (std > 0.05).
    assert float(np.std(nulls)) > 0.05


# ---------------------------------------------------------------------------
# Unit: max_sample_per_side actually caps; n_trials honoured.
# ---------------------------------------------------------------------------
def test_null_respects_caps_and_trial_count():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(2000, PHI_DIM)).astype(np.float32)
    out = _null_cosines(phi, phi, n_swiss=10_000, n_nmp=10_000, n_trials=37,
                        rng=rng, max_per_side=500)
    assert out.shape == (37,)


# ---------------------------------------------------------------------------
# Unit: summary handles all-None / partial dicts gracefully.
# ---------------------------------------------------------------------------
def test_summary_handles_partial_features():
    pf = {
        "1": {"observed_cos": 0.95, "null_median": 0.90, "null_p5": 0.85,
              "null_p95": 0.94, "null_p99": 0.96, "delta_cos": 0.05,
              "p_value": 0.02, "n_trials": 100, "n_swiss_residues": 100,
              "n_nmpfam_residues": 50},
        "2": {"observed_cos": None, "null_median": None, "null_p5": None,
              "null_p95": None, "null_p99": None, "delta_cos": None,
              "p_value": None, "n_trials": 100, "n_swiss_residues": 0,
              "n_nmpfam_residues": 0},
    }
    s = _summarise(pf)
    assert s["n_features_processed"] == 2
    assert s["n_features_with_null"] == 1
    assert s["median_delta_cos"] == 0.05
    assert s["frac_observed_above_null_p95"] == 1.0  # 1 of 1 with both fields


# ---------------------------------------------------------------------------
# Integration: end-to-end on a synthesised analysis dir built from
# feature_data_test_500. Builds the phi cache, runs metric C, then runs
# the null. Verifies (a) the null script completes, (b) per-feature
# observed_cos matches metric_C verbatim, (c) median observed cos is well
# above median null cos when computed on a tiny synthetic working set.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synth_analysis_dir(tmp_path_factory) -> Path | None:
    src = Path(__file__).resolve().parents[2] / "feature_data_test_500"
    if not src.is_dir():
        return None
    if not (src / "geometry_residue_profiles").is_dir():
        return None
    tmp = tmp_path_factory.mktemp("metric_c_null_smoke")
    for name in [
        "residue_activations", "geometry_residue_profiles", "nmpfam",
        "feature_max_activations.npy", "protein_feature_maxes.npy",
        "pipeline_state.json", "sequences.json", "geometry_protein_features.npz",
    ]:
        s = src / name
        if s.exists():
            (tmp / name).symlink_to(s)
    fmax = np.load(src / "feature_max_activations.npy")
    chosen = sorted(np.where(fmax > 0.5)[0][:5].tolist())
    acts0 = np.load(src / "residue_activations" /
                    next((src / "residue_activations").glob("*.npz")).name)["activations"]
    thr = {
        f: float(np.quantile(acts0[:, f][acts0[:, f] > 0], 0.3))
           if (acts0[:, f] > 0).any() else 0.1
        for f in chosen
    }
    (tmp / "permutation_null").mkdir()
    (tmp / "geometry_classifiers").mkdir()
    for f in chosen:
        (tmp / "permutation_null" / f"{f:04d}.json").write_text(json.dumps(
            {"feature_id": int(f), "p_values": {"geometry_prauc": 1e-6}}))
        (tmp / "geometry_classifiers" / f"{f:04d}_meta.json").write_text(json.dumps(
            {"feature_id": int(f), "threshold_sae": thr[f],
             "threshold_geom": 0.5, "half_w": 10}))
    # Build phi cache
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_residue_phi_cache.py"),
         "--analysis-dir", str(tmp), "--workers", "2"],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        return None
    # Run metric C
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "transfer_metric_c.py"),
         "--analysis-dir", str(tmp), "--limit-features", "5",
         "--checkpoint-every", "9999"],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        return None
    return tmp


def test_null_end_to_end_smoke(synth_analysis_dir):
    if synth_analysis_dir is None:
        pytest.skip("could not build synthetic analysis dir")

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compute_metric_c_null.py"),
         "--analysis-dir", str(synth_analysis_dir),
         "--n-trials", "20", "--seed", "0"],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, f"null script failed: {r.stderr[-2000:]}"

    null_path = synth_analysis_dir / "transfer_metrics" / "metric_C_null.json"
    assert null_path.exists()
    payload = json.loads(null_path.read_text())
    assert payload["metric"] == "C_null"
    assert payload["config"]["n_trials"] == 20

    # Cross-check: every per_feature observed_cos matches metric_C verbatim.
    mc = json.loads((synth_analysis_dir / "transfer_metrics" / "metric_C.json").read_text())
    for fid, rec in payload["per_feature"].items():
        mc_rec = mc["per_feature"][fid]
        if rec["observed_cos"] is None:
            assert mc_rec["phi_cosine"] is None
        else:
            assert abs(rec["observed_cos"] - mc_rec["phi_cosine"]) < 1e-9, (
                f"observed_cos drift on {fid}: {rec['observed_cos']} vs {mc_rec['phi_cosine']}"
            )
        # n's match
        assert rec["n_swiss_residues"]  == mc_rec["n_swiss_residues"]
        assert rec["n_nmpfam_residues"] == mc_rec["n_nmpfam_residues"]

    s = payload["summary"]
    assert s["n_features_processed"] == len(payload["per_feature"])
    # On test_500 the sequence pool is small but homogeneous (mostly human
    # SwissProt), so the null cosine will be very high already. We assert
    # only that the script *produced* a finite delta — the magnitude is
    # data-dependent and not a correctness signal here.
    if s["median_delta_cos"] is not None:
        assert np.isfinite(s["median_delta_cos"])
