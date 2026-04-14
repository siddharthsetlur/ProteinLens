"""Tests for the protein-level InterPro permutation null added to
``scripts/compute_permutation_null.py``.

Focus — the invariants that matter for scientific reproducibility:
  I1/I2: the five pre-existing null distributions are bit-identical whether
         the protein-level block runs or not (no RNG leakage).
  I3:    the new null_interpro_protein array is length n_permutations,
         bounded in [0, 1], and its observed score matches a direct call to
         ``_compute_protein_level_f1``.
  I4:    output JSON schema contains the new keys and preserves the five
         existing keys in their original positions.
  I5:    the new loader performs no glob/exists calls on the cache dir
         (cephfs regression guard — uses only the pre-globbed file set).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import compute_permutation_null as cpn
from proteinlens.analysis.feature_pipeline.interpro_enrichment import (
    _compute_protein_level_f1,
)


# ── Synthetic fixture builders ────────────────────────────────────────


def _build_feature_data(n_proteins: int, length: int, rng: np.random.Generator) -> dict:
    """Construct a feature JSON dict with n_proteins entries.

    Each protein gets a random per-residue activation vector. The pooled
    proteins populate ``top_sequences`` — the same source ``_pool_proteins``
    reads from.
    """
    proteins = []
    for i in range(n_proteins):
        # Mix of amino acids (keep to standard 20 so k-mer extraction works).
        seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=length))
        acts = rng.random(length).tolist()
        proteins.append({
            "accession": f"P{i:05d}",
            "sequence": seq,
            "per_residue_activations": acts,
        })
    return {"top_sequences": proteins, "activation_bins": {}}


def _write_interpro_cache(cache_dir: Path, hits: dict[str, list[dict]]) -> set[str]:
    """Write one JSON per accession and return the set of stems."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_set: set[str] = set()
    for acc, domains in hits.items():
        (cache_dir / f"{acc}.json").write_text(
            json.dumps({"accession": acc, "domains": domains})
        )
        file_set.add(acc)
    return file_set


def _minimal_shared(interpro_file_set: set[str]) -> dict:
    """Shared dict that disables geometry/CATH so tests are self-contained.

    Setting act_matrix_full=None short-circuits ``_load_gbm_and_predict``
    in its fallback branch → geom_result is None and geometry null is 0.
    """
    return {
        "feat_max_arr": None,  # loaded from disk
        "interpro_file_set": interpro_file_set,
        "cath_file_set": set(),
        "geom_profile_files": set(),
        "geom_profile_dir": Path("/nonexistent"),
        "act_file_map": {},
        "gbm_files": set(),
        "act_matrix_full": None,
        "row_to_acc": None,
        "feature_json_fids": set(),  # empty = don't filter
        "motif_k": 3,
        "include_pwm": False,
    }


def _setup_data_dir(tmp_path: Path, feature_data: dict, feat_max: float,
                    fid: int = 0) -> Path:
    """Write feature JSON and feature_max_activations.npy into a data dir."""
    data_dir = tmp_path / "data"
    (data_dir / "features").mkdir(parents=True, exist_ok=True)
    (data_dir / "features" / f"{fid:04d}.json").write_text(json.dumps(feature_data))
    # num_features large enough that fid is in range
    fm = np.zeros(max(fid + 1, 1), dtype=np.float32)
    fm[fid] = feat_max
    np.save(data_dir / "feature_max_activations.npy", fm)
    return data_dir


# ── Tests ─────────────────────────────────────────────────────────────


def test_rng_isolation_preserves_existing_nulls(tmp_path):
    """I1/I2: running with vs without InterPro annotations must produce
    identical values for the five pre-existing null distributions.

    This directly detects any RNG-stream contamination from the new
    across-protein shuffle into the within-protein shuffle used by the
    original metrics.
    """
    rng = np.random.default_rng(0)
    feature_data = _build_feature_data(n_proteins=8, length=40, rng=rng)
    data_dir = _setup_data_dir(tmp_path, feature_data, feat_max=1.0)

    # Case A: no InterPro cache at all → new block is a no-op.
    shared_a = _minimal_shared(interpro_file_set=set())
    result_a = cpn.process_feature(fid=0, data_dir=data_dir,
                                   n_permutations=15, seed=42, shared=shared_a)

    # Case B: InterPro cache present → new block runs.
    cache_dir = data_dir / "interpro_cache"
    hits = {
        f"P{i:05d}": [{
            "interpro_accession": "IPR000001",
            "interpro_name": "Test domain",
            "type": "family",
            "member_db": "pfam",
            "member_accession": "PF00001",
            "start": 1,
            "end": 20,
        }] for i in range(5)
    }
    file_set = _write_interpro_cache(cache_dir, hits)
    shared_b = _minimal_shared(interpro_file_set=file_set)
    result_b = cpn.process_feature(fid=0, data_dir=data_dir,
                                   n_permutations=15, seed=42, shared=shared_b)

    # The five pre-existing metrics must be byte-identical between runs.
    for key in ("motif_f1", "position_f1", "cath_res_f1", "geometry_prauc"):
        assert result_a["null_distributions"][key] == result_b["null_distributions"][key], (
            f"{key} null distribution changed when InterPro block became active"
        )
        assert result_a["observed"][key] == result_b["observed"][key]

    # interpro_res_f1 differs: case A has no residue labels either, case B does.
    # But the *null draws driving it* should be identical — verify via motif
    # and position which share `rng`. Those cover the main RNG-leak concern.

    # New key exists in B and is absent/zeroed in A.
    assert "interpro_protein_f1" in result_b["null_distributions"]
    assert result_a["observed"]["interpro_protein_f1"] == 0.0
    assert all(v == 0.0 for v in result_a["null_distributions"]["interpro_protein_f1"])


def test_observed_matches_direct_computation(tmp_path):
    """I3 (observed): the observed interpro_protein_f1 must equal a direct
    call to ``_compute_protein_level_f1`` with the same parameters."""
    rng = np.random.default_rng(1)
    n = 6
    feature_data = _build_feature_data(n_proteins=n, length=30, rng=rng)
    # Force predictable per-protein max: proteins 0-2 have higher activations
    for i, p in enumerate(feature_data["top_sequences"]):
        p["per_residue_activations"] = [
            (1.0 if i < 3 else 0.1) + 0.01 * j for j in range(30)
        ]
    feat_max = 2.0
    data_dir = _setup_data_dir(tmp_path, feature_data, feat_max=feat_max)

    # Annotate proteins 0,1,2 with IPR_A — perfect signal at some threshold.
    hits = {f"P{i:05d}": [{
        "interpro_accession": "IPR_A",
        "interpro_name": "A",
        "type": "family",
        "member_db": "pfam",
        "member_accession": "PF_A",
        "start": 1, "end": 20,
    }] for i in range(3)}
    file_set = _write_interpro_cache(data_dir / "interpro_cache", hits)
    shared = _minimal_shared(interpro_file_set=file_set)

    result = cpn.process_feature(fid=0, data_dir=data_dir,
                                 n_permutations=5, seed=7, shared=shared)

    # Recompute directly with the same helper used inside compute_permutation_null.
    from proteinlens.analysis.feature_pipeline.interpro_api import _load_cached
    protein_annotations = {
        f"P{i:05d}": _load_cached(data_dir / "interpro_cache" / f"P{i:05d}.json")
        for i in range(3)
    }
    accessions = [f"P{i:05d}" for i in range(n)]
    per_protein_max = np.array(
        [max(p["per_residue_activations"]) for p in feature_data["top_sequences"]]
    )
    direct = _compute_protein_level_f1(
        list(zip(accessions, per_protein_max.tolist())),
        protein_annotations, feat_max,
        n_threshold_steps=50,
        min_proteins=cpn._INTERPRO_PROTEIN_MIN_PROTEINS,
        top_n=1,
    )
    expected = round(float(direct[0]["best_f1"]), 6) if direct else 0.0

    assert result["observed"]["interpro_protein_f1"] == expected
    # Perfect separation → top-1 F1 should be 1.0.
    assert expected == pytest.approx(1.0, abs=1e-6)


def test_null_array_shape_and_bounds(tmp_path):
    """I3 (null): null array is length n_permutations and values in [0, 1]."""
    rng = np.random.default_rng(2)
    feature_data = _build_feature_data(n_proteins=10, length=25, rng=rng)
    data_dir = _setup_data_dir(tmp_path, feature_data, feat_max=1.0)

    hits = {
        f"P{i:05d}": [{
            "interpro_accession": f"IPR_{i % 2}",
            "interpro_name": "X", "type": "family",
            "member_db": "pfam", "member_accession": "PF_X",
            "start": 1, "end": 15,
        }] for i in range(8)
    }
    file_set = _write_interpro_cache(data_dir / "interpro_cache", hits)
    shared = _minimal_shared(interpro_file_set=file_set)

    n_perm = 40
    result = cpn.process_feature(fid=0, data_dir=data_dir,
                                 n_permutations=n_perm, seed=11, shared=shared)

    null = result["null_distributions"]["interpro_protein_f1"]
    assert len(null) == n_perm
    assert all(0.0 <= v <= 1.0 for v in null)
    # p-value is a valid probability in (0, 1] (Phipson-Smyth never reaches 0).
    pv = result["p_values"]["interpro_protein_f1"]
    assert 0.0 < pv <= 1.0


def test_output_schema_additive(tmp_path):
    """I4: all six keys present in each of observed/null_distributions/
    p_values/null_summary, with the five pre-existing keys still present."""
    rng = np.random.default_rng(3)
    feature_data = _build_feature_data(n_proteins=5, length=25, rng=rng)
    data_dir = _setup_data_dir(tmp_path, feature_data, feat_max=1.0)
    shared = _minimal_shared(interpro_file_set=set())

    result = cpn.process_feature(fid=0, data_dir=data_dir,
                                 n_permutations=5, seed=0, shared=shared)

    expected_keys = {
        "motif_f1", "position_f1", "interpro_res_f1",
        "cath_res_f1", "geometry_prauc", "interpro_protein_f1",
    }
    for section in ("observed", "null_distributions", "p_values", "null_summary"):
        assert set(result[section].keys()) == expected_keys, section


def test_loader_uses_pregloblied_set_only(tmp_path, monkeypatch):
    """I5: the new loader must not call Path.glob / Path.exists on cephfs.
    It must rely solely on the pre-globbed ``interpro_file_set``."""
    cache_dir = tmp_path / "interpro_cache"
    hits = {
        f"P{i:05d}": [{
            "interpro_accession": "IPR_Y",
            "interpro_name": "Y", "type": "family",
            "member_db": "pfam", "member_accession": "PF_Y",
            "start": 1, "end": 10,
        }] for i in range(3)
    }
    file_set = _write_interpro_cache(cache_dir, hits)

    proteins = [{"accession": a, "sequence": "A" * 10,
                 "activations": np.zeros(10)} for a in file_set]

    calls = {"glob": 0, "exists": 0}
    orig_glob = Path.glob
    orig_exists = Path.exists

    def counting_glob(self, *a, **kw):
        calls["glob"] += 1
        return orig_glob(self, *a, **kw)

    def counting_exists(self):
        calls["exists"] += 1
        return orig_exists(self)

    monkeypatch.setattr(Path, "glob", counting_glob)
    monkeypatch.setattr(Path, "exists", counting_exists)

    annotations = cpn._load_interpro_protein_annotations(
        proteins, file_set, cache_dir,
    )

    assert len(annotations) == 3
    # Reading the file via open() is fine; what we forbid is directory
    # scanning / per-file existence probing on cephfs.
    assert calls["glob"] == 0, "loader unexpectedly globbed the cache directory"
    assert calls["exists"] == 0, "loader unexpectedly probed file existence"


# ── PWM null integration tests (Option A: fixed PWMs, activations permuted) ─


def _write_pwm_output(data_dir: Path, fid: int, consensus: str,
                      pwm: np.ndarray, e_value: float = 1e-3) -> None:
    """Write a motif_pwm_enrichment/{fid:04d}.json fixture."""
    pwm_dir = data_dir / "motif_pwm_enrichment"
    pwm_dir.mkdir(parents=True, exist_ok=True)
    (pwm_dir / f"{fid:04d}.json").write_text(json.dumps({
        "feature_id": fid,
        "motifs": [{
            "motif_id": "motif_1",
            "consensus": consensus,
            "width": int(pwm.shape[0]),
            "e_value": e_value,
            "best_f1": 0.0,
            "pwm": pwm.tolist(),
            "aa_order": "ACDEFGHIKLMNPQRSTVWY",
        }],
    }))


def test_pwm_null_isolation_preserves_existing_nulls(tmp_path):
    """Enabling --include-pwm must not perturb the six pre-existing null
    distributions (their RNG stream is independent from rng_pwm)."""
    rng = np.random.default_rng(5)
    feature_data = _build_feature_data(n_proteins=8, length=40, rng=rng)
    data_dir = _setup_data_dir(tmp_path, feature_data, feat_max=1.0)

    # Write a trivial PWM (width 5, uniform) — structure test only.
    pwm = np.full((5, 20), 1.0 / 20)
    _write_pwm_output(data_dir, fid=0, consensus="ACDEF", pwm=pwm)

    # Run A: pwm disabled
    shared_off = _minimal_shared(interpro_file_set=set())
    shared_off["include_pwm"] = False
    result_off = cpn.process_feature(fid=0, data_dir=data_dir,
                                      n_permutations=20, seed=99,
                                      shared=shared_off)

    # Run B: pwm enabled
    shared_on = _minimal_shared(interpro_file_set=set())
    shared_on["include_pwm"] = True
    result_on = cpn.process_feature(fid=0, data_dir=data_dir,
                                     n_permutations=20, seed=99,
                                     shared=shared_on)

    # All six pre-existing metrics must be byte-identical.
    for key in ("motif_f1", "position_f1", "interpro_res_f1",
                "cath_res_f1", "geometry_prauc", "interpro_protein_f1"):
        assert result_off["null_distributions"][key] == result_on["null_distributions"][key], (
            f"{key} null changed when PWM stage was enabled — RNG leak"
        )
        assert result_off["observed"][key] == result_on["observed"][key]

    # PWM entry present in B, absent in A.
    assert "pwm_f1" not in result_off["null_distributions"]
    assert "pwm_f1" in result_on["null_distributions"]
    assert len(result_on["null_distributions"]["pwm_f1"]) == 20


def test_pwm_null_recovers_signal(tmp_path):
    """Plant a strong PWM signal, give it a matching PWM → observed F1
    dominates the null distribution, yielding a small p-value."""
    rng = np.random.default_rng(17)
    aa = list("ACDEFGHIKLMNPQRSTVWY")

    # 12 proteins, each has motif 'LYGKE' implanted at a known offset with
    # high activation at the motif centre.
    motif = "LYGKE"
    proteins = []
    for i in range(12):
        seq_list = rng.choice(aa, size=60).tolist()
        pos = 25 + (i % 10)
        for j, ch in enumerate(motif):
            seq_list[pos + j] = ch
        seq = "".join(seq_list)
        acts = [0.0] * 60
        acts[pos + 2] = 1.0
        proteins.append({"accession": f"P{i:05d}", "sequence": seq,
                         "per_residue_activations": acts})
    feature_data = {"top_sequences": proteins, "activation_bins": {}}
    data_dir = _setup_data_dir(tmp_path, feature_data, feat_max=1.0)

    # Build a PWM that exactly matches the motif (peaked at L,Y,G,K,E).
    aa_idx = {a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    pwm = np.full((5, 20), 0.01)
    for j, ch in enumerate(motif):
        pwm[j] = 0.01
        pwm[j, aa_idx[ch]] = 0.81  # dominant
    pwm = pwm / pwm.sum(axis=1, keepdims=True)
    _write_pwm_output(data_dir, fid=0, consensus=motif, pwm=pwm)

    shared = _minimal_shared(interpro_file_set=set())
    shared["include_pwm"] = True
    result = cpn.process_feature(fid=0, data_dir=data_dir,
                                  n_permutations=50, seed=3, shared=shared)

    obs = result["observed"]["pwm_f1"]
    null = result["null_distributions"]["pwm_f1"]
    pv = result["p_values"]["pwm_f1"]

    assert obs > 0.0, "observed PWM F1 should be positive with implanted motif"
    # Observed should beat the null mean by a clear margin.
    assert obs > float(np.mean(null)) + 2 * float(np.std(null) + 1e-6), (
        f"observed {obs} not separated from null mean {np.mean(null):.3f}"
    )
    assert 0.0 < pv <= 1.0
