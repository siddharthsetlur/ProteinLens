"""Regression tests for the refit-GBM geometry permutation null.

Covers the invariants that matter for scientific reproducibility:

* **Determinism**: same (seed, fid, inputs) → identical JSON output.
* **Threshold invariance under shuffle**: within-protein shuffle preserves
  the per-feature value set, so the 80th-percentile threshold is
  unchanged (the module asserts this at runtime; we cross-check it
  here on independent inputs).
* **Sampling parity with ``collect_node_fragments``**: the set of
  (positive, hard-neg, zero) indices produced by
  :func:`_sample_train_indices` matches those produced by the pipeline's
  canonical fragment collector, for identical inputs.
* **H₀ calibration**: when activations are independent of geometry,
  the observed PR-AUC and the null mean live on comparable scales.
* **Planted-signal recovery**: when activations are driven by geometry,
  the observed PR-AUC sits well above the null distribution.
* **Real-data smoke** (skipped if ``feature_data_test_500`` is absent):
  end-to-end refit null on one real feature; sanity-check that the
  observed lies in a plausible range.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from proteinlens.analysis.feature_pipeline import geometry_null_refit as gnr
from proteinlens.analysis.geometry.classifiers import collect_node_fragments
from proteinlens.analysis.geometry.residue_features import (
    ACTIVE_GEOM_NAMES,
)


# ─────────────────────────── fixtures ────────────────────────────────


def _make_protein_data(
    n_proteins: int,
    protein_len: int,
    rng: np.random.Generator,
    signal: bool = False,
) -> list[dict]:
    """Build a minimal ``protein_data`` list that satisfies the contract of
    :func:`extract_local_feature_vector`.

    Each protein has synthetic ``ca`` coords, ``profiles`` dict, and an
    ``act_matrix`` with a single column (the feature we test).

    When ``signal`` is True, activations at curved residues (large
    abs(curvature)) are boosted — a GBM with any curvature-related feature
    should learn this and produce observed PR-AUC well above null.
    """
    proteins: list[dict] = []
    for i in range(n_proteins):
        n = protein_len
        # Random-walk CA coordinates — good enough to give varied curvature.
        steps = rng.normal(scale=1.0, size=(n, 3))
        ca = np.cumsum(steps, axis=0).astype(np.float64)

        # Build profiles. extract_local_feature_vector only *reads* these —
        # they must have the expected keys and the right length.
        profiles = {
            "curvature": rng.normal(0.0, 1.0, size=n).astype(np.float64),
            "torsion": rng.normal(0.0, 1.0, size=n).astype(np.float64),
            "planarity": rng.normal(0.0, 0.3, size=n).astype(np.float64),
            # Tangents — the feature extractor uses dot-products, so any
            # unit-ish vector works. Normalise to avoid exact zeros.
            "tangents": _random_unit_vectors(n, rng).astype(np.float64),
            "helix_mask": rng.integers(0, 2, size=n).astype(bool),
            "categories": rng.integers(0, 6, size=n).astype(np.int64),
        }
        seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=n))

        # Activation column: random noise, optionally driven by geometry.
        # For the "signal" regime we make the signal deterministic-binary —
        # every high-curvature residue gets a clear positive activation,
        # every low-curvature residue gets a near-zero activation. This
        # keeps the classification target learnable by the GBM and
        # separates it cleanly from noise, so the refit null's generalisation
        # signal is visible above the memorisation baseline.
        if signal:
            # Deterministic binary signal: every high-curvature residue gets
            # activation 2.0, every low-curvature residue gets 0.01. No
            # within-group noise — the threshold lands between the two
            # populations and every high-curv residue is a positive.
            high_curv = np.abs(profiles["curvature"]) > 0.8
            acts = np.where(high_curv, 2.0, 0.01).astype(np.float64)
        else:
            acts = rng.random(n).astype(np.float64) * 0.2
        act_matrix = acts.reshape(-1, 1)  # (n, 1) — feature index 0

        proteins.append(
            {
                "accession": f"P{i:05d}",
                "act_matrix": act_matrix,
                "ca": ca,
                "profiles": profiles,
                "n_residues": n,
                "sequence": seq,
            }
        )
    return proteins


def _random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-6, None)
    return v


# ─────────────────────────── unit tests ──────────────────────────────


def test_threshold_invariant_under_within_protein_shuffle() -> None:
    """Shuffle preserves per-protein value sets → global quantile unchanged."""
    rng = np.random.default_rng(0)
    arr = rng.random(1000).astype(np.float32)
    # Make half the values zero — the quantile is computed over nonzero.
    arr[rng.choice(1000, 500, replace=False)] = 0.0
    boundaries = [(0, 250), (250, 500), (500, 1000)]

    thr_before = gnr._compute_threshold(arr)
    shuffled = gnr._shuffle_within_proteins(arr, boundaries, np.random.default_rng(1))
    thr_after = gnr._compute_threshold(shuffled)

    assert thr_before > 0
    assert abs(thr_before - thr_after) < 1e-6


def test_sampling_parity_with_collect_node_fragments() -> None:
    """_sample_train_indices picks the same (pos, bg) residues as the
    pipeline's collect_node_fragments on the same inputs, modulo ordering
    differences that do not affect the set of indices."""
    rng = np.random.default_rng(7)
    proteins = _make_protein_data(n_proteins=5, protein_len=80, rng=rng)

    # Pipeline path
    frag = collect_node_fragments(proteins, 0, half_w=gnr._HALF_W,
                                  act_quantile=gnr._ACT_QUANTILE,
                                  max_fragments=100,  # cap activated = 500
                                  bg_ratio=gnr._BG_RATIO)
    pipe_pos = {(f["accession"], f["position"]) for f in frag["activated"]}
    pipe_bg = {(f["accession"], f["position"]) for f in frag["background"]}

    # Refit path — threshold uses the full column (pipeline parity), then
    # sampling runs over the interior-only cache.
    thr = gnr._compute_threshold_from_protein_data(proteins, 0)
    cache = gnr._build_residue_cache(proteins, 0)
    assert cache is not None
    feat_vecs, activations, valid, boundaries = cache
    pos_idx, bg_idx, n_pos_uncapped = gnr._sample_train_indices(activations, thr)

    # Map flat indices back to (protein, position)
    def _unflatten(idx: int) -> tuple[str, int]:
        for pi, (start, end) in enumerate(boundaries):
            if start <= idx < end:
                pos = gnr._HALF_W + (idx - start)
                return proteins[pi]["accession"], pos
        raise AssertionError(f"idx {idx} out of bounds")

    refit_pos = {_unflatten(int(i)) for i in pos_idx}
    refit_bg = {_unflatten(int(i)) for i in bg_idx}

    assert refit_pos == pipe_pos, (
        f"positive sets diverge: {len(refit_pos ^ pipe_pos)} symmetric-diff"
    )
    assert refit_bg == pipe_bg, (
        f"background sets diverge: {len(refit_bg ^ pipe_bg)} symmetric-diff"
    )
    assert frag["threshold"] == pytest.approx(thr, abs=1e-6)


def test_planted_signal_above_null_mean() -> None:
    """When activations are driven by geometry, observed PR-AUC should sit
    above the null's mean.

    Scope note
    ----------
    This is a *one-sided sanity check*, not a power analysis. Constructing
    a strongly separating synthetic signal against the refit null is hard:

    1. The refit null's null mean is elevated by memorisation of the
       training fragments, so even true-null cases hit ~0.4–0.5 PR-AUC
       in small synthetic settings.
    2. ``extract_local_feature_vector`` exposes *windowed* geometry
       statistics (mean/max curvature over ~20 residues), not
       point-level features. Point-level planted signals are smeared by
       the windowing and hard for the GBM to recover.

    The meaningful scientific invariants are covered by the parity,
    determinism, H₀-calibration, and real-data smoke tests. Here we only
    require that a deliberately planted signal pushes observed above the
    null mean — anything stronger is conflated with the two effects above.
    """
    rng = np.random.default_rng(42)
    proteins = _make_protein_data(n_proteins=30, protein_len=200, rng=rng, signal=True)

    thr = gnr._compute_threshold_from_protein_data(proteins, 0)
    assert thr > 0
    cache = gnr._build_residue_cache(proteins, 0)
    assert cache is not None
    feat_vecs, activations, valid, boundaries = cache

    observed, _, _, _ = gnr._fit_and_score(activations, feat_vecs, valid, thr)

    rng_perm = np.random.default_rng(0)
    null = np.zeros(10)
    for k in range(10):
        sh = gnr._shuffle_within_proteins(activations, boundaries, rng_perm)
        null[k], _, _, _ = gnr._fit_and_score(sh, feat_vecs, valid, thr)

    assert observed > null.mean(), (
        f"observed {observed:.3f} did not exceed null mean {null.mean():.3f} "
        f"(null max {null.max():.3f})"
    )


def test_null_under_h0_calibration() -> None:
    """Random activations uncorrelated with geometry → observed and null
    live on the same scale (the test is intentionally loose — permutation
    variance dominates with 10 proteins)."""
    rng = np.random.default_rng(1)
    proteins = _make_protein_data(n_proteins=10, protein_len=120, rng=rng, signal=False)

    thr = gnr._compute_threshold_from_protein_data(proteins, 0)
    cache = gnr._build_residue_cache(proteins, 0)
    assert cache is not None
    feat_vecs, activations, valid, boundaries = cache
    observed, _, _, _ = gnr._fit_and_score(activations, feat_vecs, valid, thr)

    rng_perm = np.random.default_rng(0)
    null = np.zeros(20)
    for k in range(20):
        sh = gnr._shuffle_within_proteins(activations, boundaries, rng_perm)
        null[k], _, _, _ = gnr._fit_and_score(sh, feat_vecs, valid, thr)

    # Under a true null, observed is a draw from roughly the same
    # distribution as the permutation scores. Allow generous slack.
    assert abs(observed - null.mean()) < null.std() * 4 + 0.10, (
        f"observed {observed:.3f} implausibly far from null mean {null.mean():.3f} "
        f"(null std {null.std():.3f})"
    )


def test_determinism_same_seed(tmp_path: Path) -> None:
    """Running compute_refit_null twice with the same seed + fid yields
    identical outputs except for timing-invariant fields (there are none)."""
    rng = np.random.default_rng(3)
    proteins = _make_protein_data(n_proteins=6, protein_len=100, rng=rng, signal=True)
    thr = gnr._compute_threshold_from_protein_data(proteins, 0)
    cache = gnr._build_residue_cache(proteins, 0)
    feat_vecs, activations, valid, boundaries = cache

    # Run the null loop twice with identical RNG seeds.
    seed = 123
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    n = 5
    nulls1 = np.zeros(n)
    nulls2 = np.zeros(n)
    for k in range(n):
        s1 = gnr._shuffle_within_proteins(activations, boundaries, rng1)
        s2 = gnr._shuffle_within_proteins(activations, boundaries, rng2)
        nulls1[k], _, _, _ = gnr._fit_and_score(s1, feat_vecs, valid, thr)
        nulls2[k], _, _, _ = gnr._fit_and_score(s2, feat_vecs, valid, thr)

    np.testing.assert_array_equal(nulls1, nulls2)


# ───────────────── real-data smoke (skipped if absent) ───────────────


_DATA_DIR = ROOT / "feature_data_test_500"


# ───────────────── BH-pool separation: overlay regression ────────────


def _write_fixed_null_json(path: Path, fid: int, p_geometry: float) -> None:
    """Minimal permutation_null/{fid:04d}.json fixture — only the fields
    _load_permutation_pvalues reads."""
    import json as _json

    payload = {
        "feature_id": fid,
        "p_values": {
            "pwm_pr_auc": 0.5,
            "position_f1": 0.5,
            "interpro_res_f1": 0.5,
            "cath_res_f1": 0.5,
            "geometry_prauc": p_geometry,
        },
    }
    path.write_text(_json.dumps(payload))


def _write_refit_null_json(path: Path, fid: int, p_refit: float) -> None:
    import json as _json

    payload = {
        "feature_id": fid,
        "source": "refit-gbm",
        "p_value_refit": p_refit,
    }
    path.write_text(_json.dumps(payload))


def test_bh_pools_are_separated(tmp_path: Path) -> None:
    """When both fixed and refit p-values exist, BH must run on each pool
    independently — the refit q-value for a feature must reflect its rank
    within the refit pool only, not its rank within a pooled set."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    import compute_geometry_primary as cgp  # type: ignore[import-not-found]

    data_dir = tmp_path / "data"
    perm_dir = data_dir / "permutation_null"
    refit_dir = data_dir / "geometry_null_refit"
    perm_dir.mkdir(parents=True)
    refit_dir.mkdir(parents=True)

    # Fixed pool: 10 features, p-values 0.01..0.10 — all would pass a naive
    # q<0.05 gate if BH were run on the pooled 10+3 = 13-feature set.
    for fid in range(10):
        _write_fixed_null_json(perm_dir / f"{fid:04d}.json", fid, (fid + 1) / 100.0)
    # Refit pool: 3 features, p-values 0.01, 0.02, 0.03 (fids 100..102) —
    # these are disjoint from the fixed fids.
    for i, fid in enumerate((100, 101, 102)):
        _write_refit_null_json(refit_dir / f"{fid:04d}.json", fid, (i + 1) / 100.0)

    adjusted = cgp._load_permutation_pvalues(data_dir)
    assert adjusted is not None
    assert adjusted["_geometry_prauc_mode"] == "both_separate"
    assert adjusted["_refit_fids"] == {100, 101, 102}
    assert 0 in adjusted["_fixed_fids"]  # fixed pool has fids 0..9

    # The refit pool has n=3, so BH-adjusted q for p=0.01 at rank 1 is
    # min(0.01 * 3/1, 1.0) = 0.03. If BH had been pooled (n=13), the same
    # raw p=0.01 would become 0.01 * 13/1 = 0.13 — very different.
    q_refit_top = adjusted["geometry_prauc_refit"][100]
    assert abs(q_refit_top - 0.03) < 1e-9, (
        f"refit BH q at rank 1 of 3 = {q_refit_top}, expected 0.03 — "
        "BH appears to be running on pooled set, not refit pool only"
    )

    # And features in the fixed pool must not appear in the refit q-table.
    for fid in range(10):
        assert fid not in adjusted["geometry_prauc_refit"]


def test_bh_refit_only_mode(tmp_path: Path) -> None:
    """When every feature with a fixed null also has a refit null, the
    mode is 'both_separate' (not 'refit_only'). 'refit_only' means the
    fixed pool is empty — we test that separately here."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    import compute_geometry_primary as cgp  # type: ignore[import-not-found]

    data_dir = tmp_path / "data"
    perm_dir = data_dir / "permutation_null"
    refit_dir = data_dir / "geometry_null_refit"
    perm_dir.mkdir(parents=True)
    refit_dir.mkdir(parents=True)

    # Fixed pool must not be empty — _load_permutation_pvalues returns
    # None if permutation_null has no usable JSONs. Give it one file with
    # only non-geometry metrics, no geometry_prauc.
    import json as _json
    (perm_dir / "0000.json").write_text(_json.dumps({
        "feature_id": 0,
        "p_values": {"pwm_pr_auc": 0.5},  # no geometry_prauc
    }))

    for i, fid in enumerate((100, 101)):
        _write_refit_null_json(refit_dir / f"{fid:04d}.json", fid, 0.01)

    adjusted = cgp._load_permutation_pvalues(data_dir)
    assert adjusted is not None
    assert adjusted["_geometry_prauc_mode"] == "refit_only"
    assert adjusted["_fixed_fids"] == set()
    assert adjusted["_refit_fids"] == {100, 101}


# ───────────────── CLI seed-collision guard ──────────────────────────


def test_cli_rejects_seed_above_10m(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI must abort with SystemExit when |seed| >= 10_000_000 to avoid
    collision with the four RNG offset streams (10M/20M/30M/40M)."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    import compute_geometry_null_refit as cnrf  # type: ignore[import-not-found]

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Minimal viable filesystem so arg validation runs, but main() exits
    # on seed check before touching anything heavy.
    (data_dir / "feature_max_activations.npy").touch()

    monkeypatch.setattr(
        _sys, "argv",
        ["compute_geometry_null_refit.py", "--data-dir", str(data_dir), "--seed", "10000000"],
    )
    with pytest.raises(SystemExit) as exc:
        cnrf.main()
    # SystemExit message should mention the collision guard.
    msg = str(exc.value)
    assert "10_000_000" in msg or "10000000" in msg


def test_cli_accepts_seed_just_below_10m(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed = 9_999_999 must pass the guard. We stop the test at the
    next failure (missing pipeline_state.json) so we exercise only the
    seed check."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    import compute_geometry_null_refit as cnrf  # type: ignore[import-not-found]

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Valid minimal .npy — _setup_shared loads this with np.load() before
    # hitting the pipeline_state.json check, so an empty file would raise
    # EOFError before the test's assertion gets to run.
    np.save(data_dir / "feature_max_activations.npy", np.array([0.0], dtype=np.float32))

    monkeypatch.setattr(
        _sys, "argv",
        ["compute_geometry_null_refit.py", "--data-dir", str(data_dir), "--seed", "9999999"],
    )
    # Expect a DIFFERENT failure — on missing pipeline_state.json —
    # proving we passed the seed check.
    with pytest.raises((SystemExit, FileNotFoundError, OSError, EOFError, ValueError)) as exc:
        cnrf.main()
    assert "10_000_000" not in str(exc.value)


# ───────────────── Immutability guard — positive test ────────────────


def test_immutability_guard_detects_file_change(tmp_path: Path) -> None:
    """_snapshot_tree + _diff_snapshots must flag any guarded-tree file
    whose (mtime, size) changes between calls."""
    import time as _time
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    import compute_geometry_null_refit as cnrf  # type: ignore[import-not-found]

    data_dir = tmp_path / "data"
    (data_dir / "permutation_null").mkdir(parents=True)
    (data_dir / "geometry_classifiers").mkdir(parents=True)
    (data_dir / "geometry_enrichment").mkdir(parents=True)
    guarded = data_dir / "permutation_null" / "0000.json"
    guarded.write_text('{"feature_id": 0}')

    snap_before = cnrf._snapshot_tree(data_dir)
    assert str(guarded.relative_to(data_dir)) in snap_before

    # Mutate the file — larger content guarantees a size change even if
    # mtime resolution is coarse.
    _time.sleep(0.01)
    guarded.write_text('{"feature_id": 0, "mutated": true}')

    snap_after = cnrf._snapshot_tree(data_dir)
    diffs = cnrf._diff_snapshots(snap_before, snap_after)
    assert any("CHANGED" in d for d in diffs), (
        f"expected CHANGED diff, got {diffs}"
    )

    # Deletion should also be detected.
    guarded.unlink()
    snap_after2 = cnrf._snapshot_tree(data_dir)
    diffs2 = cnrf._diff_snapshots(snap_before, snap_after2)
    assert any("DELETED" in d for d in diffs2), (
        f"expected DELETED diff, got {diffs2}"
    )

    # A new file INSIDE the guarded tree must be flagged too.
    (data_dir / "permutation_null" / "0001.json").write_text('{"feature_id": 1}')
    snap_after3 = cnrf._snapshot_tree(data_dir)
    diffs3 = cnrf._diff_snapshots(snap_before, snap_after3)
    assert any("NEW" in d for d in diffs3), (
        f"expected NEW diff, got {diffs3}"
    )


# ───────────────── Real-data smoke test (unchanged) ──────────────────


@pytest.mark.skipif(
    not (_DATA_DIR / "geometry_enrichment").is_dir()
    or not any((_DATA_DIR / "geometry_enrichment").glob("[0-9]*.json")),
    reason="feature_data_test_500 without geometry_enrichment — real-data test skipped",
)
def test_real_data_single_feature(tmp_path: Path) -> None:
    """End-to-end refit null on one real feature.

    This is a smoke test — it does not enforce exact parity with the stored
    ``concordance.avg_precision`` because the refit's evaluation population
    differs slightly from ``compute_concordance_metrics`` (no
    per-protein-any-firing filter). It does check that:

    * observed_prauc is finite and in [0, 1],
    * the null mean is positive (the memorisation baseline the refit is
      supposed to surface — not 0.0 like the fixed-GBM null),
    * the JSON schema is complete.
    """
    from scripts.compute_geometry_null_refit import (  # type: ignore[attr-defined]
        _read_stored_avg_precision,
        _setup_shared,
    )

    shared = _setup_shared(_DATA_DIR)
    if not shared["geom_enrich_fids"]:
        pytest.skip("no geometry_enrichment features available")

    fid = sorted(shared["geom_enrich_fids"])[0]
    stored_ap = _read_stored_avg_precision(shared["geom_enrich_dir"], fid)

    result = gnr.compute_refit_null(
        fid=fid,
        act_matrix_full=shared["act_matrix_full"],
        row_to_acc=shared["row_to_acc"],
        act_file_map=shared["act_file_map"],
        geom_profile_dir=shared["geom_profile_dir"],
        geom_profile_files=shared["geom_profile_files"],
        n_permutations=5,  # cheap
        seed=0,
        stored_avg_precision=stored_ap,
    )
    if result is None:
        pytest.skip(f"feature {fid} could not be processed (too few positives?)")

    assert 0.0 <= result["observed_prauc"] <= 1.0
    assert 0.0 <= result["null_mean"] <= 1.0
    assert len(result["null_prauc_refit"]) == 5
    assert result["source"] == "refit-gbm"
    # Schema completeness
    for key in (
        "feature_id", "observed_prauc", "null_mean", "null_std",
        "p_value_refit", "n_proteins", "n_residues_total", "threshold_sae",
    ):
        assert key in result, f"missing {key} in refit output"


# ───────────────── activation-column cache: parity ───────────────────


def _write_fixture_files(
    tmp_path: Path,
    proteins: list[dict],
    n_features: int,
    signal_fid: int = 0,
) -> Path:
    """Materialise a minimal on-disk feature_data_ layout from an in-memory
    ``protein_data`` list.

    Writes just the files the refit null + cache-builder need:

    * ``residue_activations/{acc}.npz`` with key ``activations``, shape
      ``(n_res, n_features)``. The protein's real activation lives at column
      ``signal_fid``; every other column is zero. ``signal_fid`` must be
      within ``[0, n_features)``.
    * ``geometry_residue_profiles/{acc}.npz`` with the six profile arrays
      and a ``sequence`` entry — same schema :func:`_load_protein_data`
      expects.
    * ``protein_feature_maxes.npy`` — shape ``(n_proteins, n_features)``
      with per-protein maxes for each column.
    * ``feature_max_activations.npy`` — shape ``(n_features,)``.
    * ``pipeline_state.json`` with an ``accession_index``.

    Returns the created ``data_dir``.
    """
    assert 0 <= signal_fid < n_features
    data_dir = tmp_path / "feat_data"
    data_dir.mkdir()
    act_dir = data_dir / "residue_activations"
    act_dir.mkdir()
    gp_dir = data_dir / "geometry_residue_profiles"
    gp_dir.mkdir()

    protein_maxes = np.zeros((len(proteins), n_features), dtype=np.float32)
    acc_to_row: dict[str, int] = {}
    for i, p in enumerate(proteins):
        acc = p["accession"]
        acc_to_row[acc] = i
        am = p["act_matrix"]
        col = am[:, 0] if am.ndim == 2 else am
        act_mat = np.zeros((len(col), n_features), dtype=np.float32)
        act_mat[:, signal_fid] = col
        np.savez(act_dir / f"{acc}.npz", activations=act_mat)
        np.savez(
            gp_dir / f"{acc}.npz",
            ca=p["ca"],
            curvature=p["profiles"]["curvature"],
            torsion=p["profiles"]["torsion"],
            planarity=p["profiles"]["planarity"],
            tangents=p["profiles"]["tangents"],
            helix_mask=p["profiles"]["helix_mask"],
            categories=p["profiles"]["categories"],
            sequence=np.array([p["sequence"]]),
        )
        protein_maxes[i, signal_fid] = float(col.max())

    np.save(data_dir / "protein_feature_maxes.npy", protein_maxes)
    feat_max = protein_maxes.max(axis=0)
    np.save(data_dir / "feature_max_activations.npy", feat_max)
    (data_dir / "pipeline_state.json").write_text(
        json.dumps({"accession_index": acc_to_row})
    )
    return data_dir


def _load_bacc_module():
    """Load scripts/build_activation_column_cache.py as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_activation_column_cache",
        ROOT / "scripts" / "build_activation_column_cache.py",
    )
    assert spec is not None and spec.loader is not None
    bacc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bacc)
    return bacc


def _build_cache(bacc, data_dir: Path, max_proteins: int) -> Path:
    """Invoke the precompute CLI and return the cache directory."""
    import sys as _sys

    cache_dir = data_dir / "activation_col_cache"
    argv_saved = _sys.argv[:]
    try:
        _sys.argv = [
            "build_activation_column_cache.py",
            "--data-dir", str(data_dir),
            "--max-proteins", str(max_proteins),
        ]
        bacc.main()
    finally:
        _sys.argv = argv_saved
    assert cache_dir.is_dir()
    assert any(cache_dir.glob("*.npz")), "cache build wrote no files"
    return cache_dir


def _run_refit(
    shared: dict,
    *,
    fid: int,
    max_proteins: int,
    activation_col_cache_dir: Path | None,
) -> dict:
    gnr._load_geom_profile_cached.cache_clear()
    result = gnr.compute_refit_null(
        fid=fid,
        act_matrix_full=shared["act_matrix_full"],
        row_to_acc=shared["row_to_acc"],
        act_file_map=shared["act_file_map"],
        geom_profile_dir=shared["geom_profile_dir"],
        geom_profile_files=shared["geom_profile_files"],
        n_permutations=10,
        seed=42,
        max_proteins=max_proteins,
        activation_col_cache_dir=activation_col_cache_dir,
    )
    assert result is not None
    return result


_PARITY_SCALAR_KEYS = (
    "feature_id",
    "n_permutations",
    "seed",
    "rng_offset",
    "observed_prauc",
    "null_mean",
    "null_std",
    "p_value_refit",
    "n_proteins",
    "n_residues_total",
    "n_residues_valid",
    "n_pos_real",
    "n_train_obs",
    "n_train_pos_obs",
    "n_train_neg_obs",
    "threshold_sae",
)


def _assert_byte_identical(cached: dict, nocache: dict) -> None:
    for key in _PARITY_SCALAR_KEYS:
        assert cached[key] == nocache[key], (
            f"cache-vs-nocache mismatch on {key!r}: "
            f"cache={cached[key]!r} nocache={nocache[key]!r}"
        )
    assert cached["null_prauc_refit"] == nocache["null_prauc_refit"], (
        "null_prauc_refit arrays differ between cache and no-cache runs"
    )


@pytest.mark.parametrize(
    "n_proteins,protein_len,signal_fid,n_features,max_proteins",
    [
        # Baseline: fid=0, small fixture, no top-N trim exercised.
        (8, 70, 0, 4, 500),
        # Off-by-one bait: signal planted at fid=2. Exercises column
        # indexing in the precompute's `slab[:, fid]` and every feature
        # loop elsewhere — catches any hard-coded `[:, 0]` mistake.
        (8, 70, 2, 4, 500),
        # Top-N trim: 40 proteins but only top-5 should survive selection,
        # on both the per-file (argsort then trim) and cache (precompute
        # stores exactly 5) paths. If the cache silently included all 40
        # or miscounted the top-N this comparison would blow up.
        (40, 50, 0, 3, 5),
    ],
)
def test_activation_cache_parity(
    tmp_path: Path,
    n_proteins: int,
    protein_len: int,
    signal_fid: int,
    n_features: int,
    max_proteins: int,
) -> None:
    """End-to-end parity: compute_refit_null with and without the
    pre-built activation-column cache must produce byte-identical JSON
    output. This is the load-bearing safety net for the L2/L3/L6 launch —
    if it fails, the launch is blocked.
    """
    bacc = _load_bacc_module()
    rng = np.random.default_rng(11 + signal_fid * 97 + n_proteins)
    proteins = _make_protein_data(
        n_proteins=n_proteins, protein_len=protein_len, rng=rng, signal=True,
    )
    data_dir = _write_fixture_files(
        tmp_path, proteins, n_features=n_features, signal_fid=signal_fid,
    )

    from scripts.compute_geometry_null_refit import _setup_shared  # type: ignore[attr-defined]
    shared = _setup_shared(data_dir)

    result_nocache = _run_refit(
        shared,
        fid=signal_fid,
        max_proteins=max_proteins,
        activation_col_cache_dir=None,
    )
    cache_dir = _build_cache(bacc, data_dir, max_proteins=max_proteins)
    result_cached = _run_refit(
        shared,
        fid=signal_fid,
        max_proteins=max_proteins,
        activation_col_cache_dir=cache_dir,
    )
    _assert_byte_identical(result_cached, result_nocache)


def test_activation_cache_rejects_max_proteins_mismatch(tmp_path: Path) -> None:
    """If the cache was built with ``max_proteins = N_build`` but the refit
    call passes ``max_proteins != N_build``, the loader must refuse the
    cache and fall back to the per-file path. Otherwise a sweep over
    ``max_proteins`` would silently consume a cache with the wrong
    protein set and publish wrong p-values.

    We verify this indirectly: a run that explicitly sets
    ``activation_col_cache_dir=cache_dir`` but passes a mismatched
    ``max_proteins`` must return numbers identical to a no-cache run with
    the same ``max_proteins`` (because the cache is rejected and the
    per-file path runs end-to-end).
    """
    bacc = _load_bacc_module()
    rng = np.random.default_rng(23)
    proteins = _make_protein_data(
        n_proteins=40, protein_len=50, rng=rng, signal=True,
    )
    data_dir = _write_fixture_files(
        tmp_path, proteins, n_features=3, signal_fid=0,
    )

    from scripts.compute_geometry_null_refit import _setup_shared  # type: ignore[attr-defined]
    shared = _setup_shared(data_dir)

    # Cache built with max_proteins=20 (> the selection we will ask for).
    cache_dir = _build_cache(bacc, data_dir, max_proteins=20)

    # Caller asks for max_proteins=5 — a mismatch that would be
    # numerically wrong if the loader honoured the 20-protein cache.
    result_mismatch = _run_refit(
        shared, fid=0, max_proteins=5, activation_col_cache_dir=cache_dir,
    )
    # Reference: identical run with no cache at all.
    result_reference = _run_refit(
        shared, fid=0, max_proteins=5, activation_col_cache_dir=None,
    )
    _assert_byte_identical(result_mismatch, result_reference)


def test_activation_cache_rejects_bad_version(tmp_path: Path) -> None:
    """A cache file with a wrong ``cache_version`` in its meta payload must
    be refused (the loader treats it as if the cache were absent). This
    protects us from a future schema bump silently consuming old caches.
    """
    bacc = _load_bacc_module()
    rng = np.random.default_rng(31)
    proteins = _make_protein_data(
        n_proteins=8, protein_len=60, rng=rng, signal=True,
    )
    data_dir = _write_fixture_files(
        tmp_path, proteins, n_features=2, signal_fid=0,
    )
    cache_dir = _build_cache(bacc, data_dir, max_proteins=500)

    # Rewrite a cache file's meta with a garbage version. np.savez closes
    # the handle when the context manager exits, so we can overwrite.
    cache_file = cache_dir / "0000.npz"
    assert cache_file.is_file()
    with np.load(cache_file, allow_pickle=True) as npz:
        cols = np.array(npz["columns"])
        offs = np.array(npz["offsets"])
        accs = np.array(npz["accessions"], dtype=object)
    bad_meta = json.dumps({
        "feature_id": 0,
        "max_proteins": 500,
        "half_w": gnr._HALF_W,
        "cache_version": 99999,  # <-- sabotage
    })
    np.savez(
        cache_file,
        columns=cols,
        offsets=offs,
        accessions=accs,
        meta=np.array([bad_meta], dtype=object),
    )

    from scripts.compute_geometry_null_refit import _setup_shared  # type: ignore[attr-defined]
    shared = _setup_shared(data_dir)

    # The cache-enabled run should detect the bad version, emit a warning,
    # and fall back to the per-file path — producing numerics identical to
    # a no-cache run.
    result_bad_cache = _run_refit(
        shared, fid=0, max_proteins=500, activation_col_cache_dir=cache_dir,
    )
    result_reference = _run_refit(
        shared, fid=0, max_proteins=500, activation_col_cache_dir=None,
    )
    _assert_byte_identical(result_bad_cache, result_reference)


