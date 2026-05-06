"""Tests for the rewritten Metric C accumulator.

The rewrite caches phi once per residue per protein and scores all features
in a single matmul. These tests ensure the new accumulator is numerically
equivalent to the legacy per-(feature, position) loop, and that the on-cluster
inputs the script depends on (half_w invariant, protein_feature_maxes.npy
memmap layout) are what the script assumes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.transfer_metric_c import (  # noqa: E402
    PHI_DIM,
    _accumulate_one,
    _load_phi_cache,
    _load_swiss_pmax_memmap,
)
from scripts.build_residue_phi_cache import _compute_phi_npz  # noqa: E402


# ---------------------------------------------------------------------------
# Legacy reference accumulator (oracle for the regression test)
# ---------------------------------------------------------------------------
def _legacy_accumulate(
    acts: np.ndarray,
    phi_mat: np.ndarray,
    phi_valid: np.ndarray,
    n: int,
    feat_arr: np.ndarray,
    thr_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-(feature, position) loop — identical to the original _accumulate_phi
    inner block, but operating on a precomputed phi_mat / phi_valid so we can
    isolate the matmul-vs-loop equivalence question from the extractor itself.
    """
    K = feat_arr.size
    phi_sum = np.zeros((K, PHI_DIM), dtype=np.float64)
    phi_count = np.zeros(K, dtype=np.int64)
    for j, f in enumerate(feat_arr):
        if f >= acts.shape[1]:
            continue
        col = acts[:n, int(f)]
        thr = thr_arr[j]
        active = np.where(col >= thr)[0]
        for ipos in active:
            ipos = int(ipos)
            if not phi_valid[ipos]:
                continue
            phi_sum[j] += phi_mat[ipos]
            phi_count[j] += 1
    return phi_sum, phi_count


# ---------------------------------------------------------------------------
# Synthetic numerical-equivalence test (the core regression gate)
# ---------------------------------------------------------------------------
def test_accumulate_matmul_matches_legacy_loop():
    rng = np.random.default_rng(42)
    n = 80
    K = 12
    n_features = 50  # SAE feature dim
    feat_arr = rng.choice(n_features, size=K, replace=False).astype(np.int64)
    thr_arr = rng.uniform(0.1, 0.6, size=K).astype(np.float32)

    acts = rng.uniform(0.0, 1.0, size=(n, n_features)).astype(np.float32)

    phi_mat = rng.normal(size=(n, PHI_DIM)).astype(np.float64)
    phi_valid = rng.uniform(size=n) > 0.1
    phi_valid[:5] = False
    phi_valid[-5:] = False
    phi_mat[~phi_valid] = 0.0
    bad = rng.choice(np.where(phi_valid)[0], size=2, replace=False)
    phi_mat[bad[0], 3] = np.nan
    phi_mat[bad[1], 7] = np.inf
    phi_valid[bad] = False
    phi_mat[~phi_valid] = 0.0

    sum_ref, cnt_ref = _legacy_accumulate(acts, phi_mat, phi_valid, n, feat_arr, thr_arr)

    sum_new = np.zeros((K, PHI_DIM), dtype=np.float64)
    cnt_new = np.zeros(K, dtype=np.int64)
    _accumulate_one(acts, phi_mat, phi_valid, n, feat_arr, thr_arr, sum_new, cnt_new)

    assert np.array_equal(cnt_new, cnt_ref), (
        f"counts mismatch:\n  new: {cnt_new}\n  ref: {cnt_ref}"
    )
    assert np.allclose(sum_new, sum_ref, atol=1e-10), (
        f"phi_sum drift exceeds 1e-10:\n  max abs diff = {np.abs(sum_new - sum_ref).max()}"
    )
    assert cnt_new.sum() > 0, "test inputs should produce at least one active residue"


def test_accumulate_handles_feature_index_out_of_range():
    rng = np.random.default_rng(7)
    n = 30
    n_features = 8
    feat_arr = np.array([2, 5, 7, 100, 200], dtype=np.int64)  # last two out-of-range
    thr_arr = np.full(feat_arr.size, 0.3, dtype=np.float32)
    acts = rng.uniform(0.0, 1.0, size=(n, n_features)).astype(np.float32)
    phi_mat = rng.normal(size=(n, PHI_DIM))
    phi_valid = np.ones(n, dtype=bool)
    phi_valid[:3] = False
    phi_valid[-3:] = False

    sum_ref, cnt_ref = _legacy_accumulate(acts, phi_mat, phi_valid, n, feat_arr, thr_arr)

    sum_new = np.zeros((feat_arr.size, PHI_DIM))
    cnt_new = np.zeros(feat_arr.size, dtype=np.int64)
    _accumulate_one(acts, phi_mat, phi_valid, n, feat_arr, thr_arr, sum_new, cnt_new)

    assert np.array_equal(cnt_new, cnt_ref)
    assert np.allclose(sum_new, sum_ref, atol=1e-10)
    assert cnt_new[3] == 0 and cnt_new[4] == 0


def test_accumulate_no_active_residues_is_noop():
    n, K, n_features = 20, 5, 10
    feat_arr = np.arange(K, dtype=np.int64)
    thr_arr = np.full(K, 1e6, dtype=np.float32)  # nothing will exceed
    acts = np.zeros((n, n_features), dtype=np.float32)
    phi_mat = np.ones((n, PHI_DIM))
    phi_valid = np.ones(n, dtype=bool)

    sum_new = np.zeros((K, PHI_DIM))
    cnt_new = np.zeros(K, dtype=np.int64)
    _accumulate_one(acts, phi_mat, phi_valid, n, feat_arr, thr_arr, sum_new, cnt_new)

    assert cnt_new.sum() == 0
    assert np.all(sum_new == 0.0)


# ---------------------------------------------------------------------------
# End-to-end equivalence: compute phi via the real extractor on a real
# protein, then compare matmul accumulator vs. legacy loop
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_protein_dir() -> Path | None:
    """A directory with residue_activations/ + geometry_residue_profiles/ on
    disk. Falls back through likely candidates so the test runs locally
    (feature_data_test_500) and on the cluster (cluster analysis dir)."""
    env = os.environ.get("PROTEINLENS_TEST_ANALYSIS_DIR")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "feature_data_test_500",
        repo_root / "feature_data_test_20",
        repo_root / "trained_models" / "layer_4" / "frosty-sweep-15" / "analysis",
        repo_root / "trained_models" / "layer_2" / "firm-sweep-3" / "analysis",
        repo_root / "trained_models" / "layer_6" / "major-sweep-15" / "analysis",
    ]
    for c in candidates:
        if (c / "residue_activations").is_dir() and (c / "geometry_residue_profiles").is_dir():
            return c
    return None


@pytest.fixture(scope="module")
def real_analysis_dir() -> Path | None:
    """Analysis dir that *also* has geometry_classifiers — used by the
    half_w invariant test."""
    env = os.environ.get("PROTEINLENS_TEST_ANALYSIS_DIR")
    if env:
        p = Path(env)
        if p.is_dir() and (p / "geometry_classifiers").is_dir():
            return p
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "trained_models" / "layer_4" / "frosty-sweep-15" / "analysis",
        repo_root / "trained_models" / "layer_2" / "firm-sweep-3" / "analysis",
        repo_root / "trained_models" / "layer_6" / "major-sweep-15" / "analysis",
    ]
    for c in candidates:
        if c.is_dir() and (c / "geometry_classifiers").is_dir():
            return c
    return None


def test_end_to_end_matches_legacy_on_real_protein(real_protein_dir):
    """Compute phi via the real extractor on a real protein, then verify
    the matmul accumulator and the legacy per-position loop agree to 1e-9.

    GBM thresholds are not needed here — we synthesize feat_arr / thr_arr
    so the test runs against any analysis dir with activations + geometry.
    """
    if real_protein_dir is None:
        pytest.skip("no analysis dir with activations + geometry profiles available")

    from scripts.transfer_metric_c import (
        _compute_phi_matrix,
        _load_profiles,
    )

    act_dir = real_protein_dir / "residue_activations"
    geom_dir = real_protein_dir / "geometry_residue_profiles"

    act_stems = sorted(p.stem for p in act_dir.glob("*.npz"))
    geom_stems = set(p.stem for p in geom_dir.glob("*.npz"))
    cand = [s for s in act_stems if s in geom_stems][:5]
    assert cand, f"no protein with both act + geom in {real_protein_dir}"

    HW = 10  # same constant as the script asserts on the cluster
    n_proteins_checked = 0

    for acc in cand:
        try:
            with np.load(act_dir / f"{acc}.npz") as a:
                acts = a["activations"]
        except Exception:  # noqa: BLE001
            continue
        loaded = _load_profiles(geom_dir / f"{acc}.npz")
        if loaded is None:
            continue
        ca, profiles, seq = loaded
        n = int(min(len(ca), acts.shape[0]))
        if n < 2 * HW + 1:
            continue

        # Pick 30 active features at this protein and use the 50th percentile of
        # their activations as the threshold so each contributes some active
        # residues to the accumulators.
        col_max = acts[:n].max(axis=0)
        active_feats = np.where(col_max > 0)[0]
        if active_feats.size < 5:
            continue
        rng = np.random.default_rng(seed=hash(acc) & 0xFFFFFFFF)
        chosen = rng.choice(active_feats, size=min(30, active_feats.size), replace=False)
        feat_arr = np.sort(chosen).astype(np.int64)
        thr_arr = np.asarray(
            [np.quantile(acts[:n, int(f)][acts[:n, int(f)] > 0], 0.5) for f in feat_arr],
            dtype=np.float32,
        )

        phi_mat, phi_valid = _compute_phi_matrix(ca, profiles, seq, n, HW)

        sum_ref, cnt_ref = _legacy_accumulate(
            acts, phi_mat, phi_valid, n, feat_arr, thr_arr
        )
        sum_new = np.zeros_like(sum_ref)
        cnt_new = np.zeros_like(cnt_ref)
        _accumulate_one(
            acts, phi_mat, phi_valid, n, feat_arr, thr_arr, sum_new, cnt_new
        )
        assert np.array_equal(cnt_new, cnt_ref), (
            f"counts mismatch on {acc}"
        )
        max_diff = float(np.abs(sum_new - sum_ref).max())
        assert max_diff < 1e-9, (
            f"phi_sum drift on {acc}: {max_diff} > 1e-9"
        )
        assert cnt_new.sum() > 0, f"no active residues accumulated on {acc}"
        n_proteins_checked += 1

    assert n_proteins_checked >= 1, "no protein exercised the accumulator"


# ---------------------------------------------------------------------------
# Inputs the script depends on
# ---------------------------------------------------------------------------
def test_half_w_constant_in_real_geometry_classifiers(real_analysis_dir):
    if real_analysis_dir is None:
        pytest.skip("no real analysis dir available")
    gbm_dir = real_analysis_dir / "geometry_classifiers"
    if not gbm_dir.is_dir():
        pytest.skip("geometry_classifiers missing")
    metas = list(gbm_dir.glob("*_meta.json"))
    if not metas:
        pytest.skip("no metas")
    seen: set[int] = set()
    for m in metas[:200]:  # sample a few hundred
        try:
            d = json.loads(m.read_text())
            seen.add(int(d["half_w"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    assert seen, "no half_w values parsed"
    assert len(seen) == 1, (
        f"half_w not constant across geometry_classifiers: {sorted(seen)}. "
        "The cached-phi optimisation in transfer_metric_c.py assumes a single half_w."
    )


def test_phi_cache_round_trip_matches_recompute(real_protein_dir, tmp_path):
    """Build the cache for one real protein via the cache-builder, then load
    it via _load_phi_cache, then recompute on the fly via the legacy fallback,
    and assert all three agree to 1e-6 (cache is float32, recompute is
    float64 — this is the only place the precision differs)."""
    if real_protein_dir is None:
        pytest.skip("no analysis dir with activations + geometry profiles available")

    from scripts.transfer_metric_c import (
        _compute_phi_matrix_from_profiles,
        _load_profiles,
    )

    geom_dir = real_protein_dir / "geometry_residue_profiles"
    cands = sorted(geom_dir.glob("*.npz"))[:3]
    if not cands:
        pytest.skip("no geometry profile available")

    HW = 10
    cache_dir = tmp_path / "residue_phi"
    cache_dir.mkdir()

    n_checked = 0
    for geom_npz in cands:
        out_npz = cache_dir / geom_npz.name
        acc, n_total, n_valid = _compute_phi_npz(geom_npz, out_npz, HW)
        if n_total < 2 * HW + 1:
            continue
        loaded = _load_profiles(geom_npz)
        assert loaded is not None
        ca, profiles, seq = loaded
        n = int(len(ca))

        # Legacy recompute
        phi_ref, valid_ref = _compute_phi_matrix_from_profiles(ca, profiles, seq, n, HW)
        # Cache load
        cached = _load_phi_cache(out_npz, n)
        assert cached is not None
        phi_cached, valid_cached = cached

        assert np.array_equal(valid_cached, valid_ref), f"valid mask mismatch on {acc}"
        # phi cache stores float32; recompute is float64. Compare on valid rows.
        # Tolerance reflects float32 precision over the 44-D vector;
        # individual components can range up to ~few hundred (contact counts).
        diff = np.abs(phi_cached[valid_ref] - phi_ref[valid_ref]).max() if valid_ref.any() else 0.0
        assert diff < 1e-4, f"phi diff {diff} > 1e-4 on {acc}"
        n_checked += 1

    assert n_checked >= 1


def test_phi_cache_missing_returns_none(tmp_path):
    """_load_phi_cache should return None when the file doesn't exist or is
    inconsistent — the script falls back to recompute in that case."""
    assert _load_phi_cache(tmp_path / "missing.npz", n=100) is None
    # Wrong shape — phi_dim != PHI_DIM
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad,
                        phi=np.zeros((50, PHI_DIM - 1), dtype=np.float32),
                        valid=np.zeros(50, dtype=bool))
    assert _load_phi_cache(bad, n=50) is None
    # Cache shorter than requested n is fine — caller slices to min(phi_n, n).
    # This is the NMPFam case where geometry profiles come from PDB structures
    # and activations come from the full consensus sequence.
    short = tmp_path / "short.npz"
    np.savez_compressed(short,
                        phi=np.zeros((30, PHI_DIM), dtype=np.float32),
                        valid=np.ones(30, dtype=bool))
    res = _load_phi_cache(short, n=50)
    assert res is not None
    assert res[0].shape == (30, PHI_DIM)
    assert res[1].shape == (30,)
    # Valid cache
    good = tmp_path / "good.npz"
    np.savez_compressed(good,
                        phi=np.ones((50, PHI_DIM), dtype=np.float32),
                        valid=np.ones(50, dtype=bool))
    res = _load_phi_cache(good, n=50)
    assert res is not None
    assert res[0].shape == (50, PHI_DIM)
    assert res[0].dtype == np.float64
    assert res[1].dtype == bool
    assert res[1].all()


def test_protein_feature_maxes_loads_as_memmap(real_analysis_dir):
    if real_analysis_dir is None:
        pytest.skip("no real analysis dir available")
    loaded = _load_swiss_pmax_memmap(real_analysis_dir)
    if loaded is None:
        pytest.skip("pmax inputs not all present in this analysis dir")
    pmax, acc_to_idx = loaded
    assert pmax.dtype == np.float32
    assert pmax.ndim == 2
    assert pmax.shape[0] == len(acc_to_idx)
    assert pmax.shape[1] > 0
    # First row must be readable; just touching it confirms the memmap shape
    # was consistent with the on-disk size.
    _ = float(pmax[0].max())
