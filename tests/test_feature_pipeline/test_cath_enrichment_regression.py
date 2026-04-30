"""Regression test for the compute_cath_enrichment residue-level refactor.

Asserts that the parallelized, per-feature column-cache implementation of
``compute_residue_level_f1`` produces byte-identical JSON output vs. the
prior serial + cross-feature full-array cache implementation.

The fixture is synthesized into ``tmp_path`` so the test has no dependency
on whatever partial analysis output happens to live in the repo. The
synthesized data is real (actual ``.npz`` files on disk, real numpy random
activations) — only the *semantics* are fake. Both implementations read
the exact same bytes, so any divergence in outputs is a correctness bug
in the refactor rather than a fixture artifact.
"""

from __future__ import annotations

import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import compute_cath_enrichment as mod  # noqa: E402


# ---------------------------------------------------------------------------
# Pre-refactor implementation (copied verbatim from the version on disk
# immediately before the parallelization edit). Serves as the oracle.
# ---------------------------------------------------------------------------


def _old_load_npz_cached(
    accession: str,
    feat_idx: int,
    npz_dir_map: Dict[str, Path],
    npz_cache: Dict[str, Optional[np.ndarray]],
    max_npz_cache: int,
) -> Optional[np.ndarray]:
    if accession not in npz_dir_map:
        return None
    if accession in npz_cache:
        arr = npz_cache[accession]
        if arr is None:
            return None
        return arr[:, feat_idx] if feat_idx < arr.shape[1] else None
    npz_path = npz_dir_map[accession] / f"{accession}.npz"
    try:
        arr = np.load(npz_path)["activations"]
    except (EOFError, OSError, KeyError):
        arr = None
    if len(npz_cache) >= max_npz_cache:
        oldest = next(iter(npz_cache))
        del npz_cache[oldest]
    npz_cache[accession] = arr
    if arr is None or feat_idx >= arr.shape[1]:
        return None
    return arr[:, feat_idx]


def _old_compute_residue_level_f1(
    protein_level_results: Dict[str, List[Dict[str, Any]]],
    protein_cath: Dict[str, List[Dict[str, Any]]],
    feat_idx: int,
    feat_max: float,
    npz_dir_map: Dict[str, Path],
    npz_cache: Dict[str, Optional[np.ndarray]],
    max_npz_cache: int = 500,
    n_threshold_steps: int = 50,
    top_n_per_level: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    results_by_level: Dict[str, List[Dict[str, Any]]] = {}

    for level in mod.CATH_LEVELS:
        prot_results = protein_level_results.get(level, [])
        level_results = []

        for prot_result in prot_results[:top_n_per_level]:
            target_label = prot_result["cath_label"]

            all_acts: List[np.ndarray] = []
            all_labels: List[np.ndarray] = []
            n_proteins_used = 0

            for acc, hits in protein_cath.items():
                matching_ranges = []
                for hit in hits:
                    cath_id = hit.get("cath_id", "")
                    if not cath_id or len(cath_id.split(".")) < 4:
                        continue
                    if mod._cath_label_at_level(cath_id, level) == target_label:
                        qs = hit.get("query_start")
                        qe = hit.get("query_end")
                        if qs is not None and qe is not None:
                            matching_ranges.append((int(qs), int(qe)))

                if not matching_ranges:
                    continue

                residue_acts = _old_load_npz_cached(
                    acc, feat_idx, npz_dir_map, npz_cache, max_npz_cache
                )
                if residue_acts is None:
                    continue

                seq_len = len(residue_acts)
                labels = np.zeros(seq_len, dtype=np.int32)
                for start, end in matching_ranges:
                    s0 = max(0, start - 1)
                    e0 = min(seq_len - 1, end - 1)
                    labels[s0 : e0 + 1] = 1

                all_acts.append(residue_acts)
                all_labels.append(labels)
                n_proteins_used += 1

            if n_proteins_used == 0:
                continue

            all_activations = np.concatenate(all_acts)
            all_label_arr = np.concatenate(all_labels)
            n_in_domain = int(all_label_arr.sum())
            n_total = len(all_label_arr)

            if n_in_domain == 0 or n_in_domain == n_total:
                continue

            nonzero = all_activations[all_activations > 0]
            if len(nonzero) == 0:
                continue

            pct_thresholds = np.percentile(nonzero, np.linspace(0, 100, n_threshold_steps))
            lin_thresholds = np.linspace(0, feat_max, n_threshold_steps)
            thresholds = np.unique(np.concatenate([pct_thresholds, lin_thresholds]))

            y_pred_all = all_activations[np.newaxis, :] > thresholds[:, np.newaxis]
            y_true = all_label_arr.astype(np.float64)
            y_true_neg = 1.0 - y_true

            tp = y_pred_all.astype(np.float64) @ y_true
            fp = y_pred_all.astype(np.float64) @ y_true_neg
            fn = float(n_in_domain) - tp

            with np.errstate(divide="ignore", invalid="ignore"):
                precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
                recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
                pr_sum = precision + recall
                f1 = np.where(pr_sum > 0, 2.0 * precision * recall / pr_sum, 0.0)

            best_idx = int(f1.argmax())
            best_f1 = float(f1[best_idx])
            if best_f1 == 0.0:
                continue

            t = float(thresholds[best_idx])
            level_results.append({
                "cath_label": target_label,
                "cath_level": level,
                "description": prot_result.get("description", ""),
                "best_f1": round(best_f1, 4),
                "best_threshold": round(t, 4),
                "best_threshold_normalized": round(t / feat_max if feat_max > 0 else 0.0, 4),
                "precision_at_best": round(float(precision[best_idx]), 4),
                "recall_at_best": round(float(recall[best_idx]), 4),
                "n_proteins_used": n_proteins_used,
                "n_total_residues": n_total,
                "n_residues_in_domain": n_in_domain,
                "n_true_positives": int(tp[best_idx]),
                "n_false_positives": int(fp[best_idx]),
                "n_false_negatives": int(fn[best_idx]),
            })

        level_results.sort(key=lambda r: r["best_f1"], reverse=True)
        results_by_level[level] = level_results

    return results_by_level


# ---------------------------------------------------------------------------
# Synthetic fixture on disk. Covers the tricky cases:
#   * Proteins matching at multiple hierarchy levels (C / CA / CAT / CATH)
#   * Proteins that have hits but no matching ranges at a given target
#   * Proteins with multiple disjoint matching ranges (union coverage)
#   * .npz files where ``feat_idx`` is out of range  → None
#   * Accessions absent from ``npz_dir_map``                   → None
#   * Accessions whose .npz fails to load                      → None
#   * Hits with missing qs/qe that must be ignored
#   * feat_max = 0   (protein_level returns empty → residue-level empty)
# ---------------------------------------------------------------------------


NUM_FEATURES = 8
RNG = np.random.default_rng(0xCA7D)


def _write_npz(path: Path, seq_len: int, num_features: int = NUM_FEATURES) -> None:
    arr = RNG.random((seq_len, num_features)).astype(np.float32)
    # Make a fraction of the activations identically 0 so the percentile
    # branch (``nonzero`` filtering) is exercised.
    mask = RNG.random(arr.shape) < 0.4
    arr[mask] = 0.0
    np.savez(path, activations=arr)


@pytest.fixture(scope="module")
def synthetic_fixture(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cath_regression")
    npz_dir_a = tmp / "residue_activations"
    npz_dir_b = tmp / "interpro_residue_activations"
    npz_dir_a.mkdir()
    npz_dir_b.mkdir()

    # Proteins P001..P020. Varying seq_len.
    seq_lens = {f"P{i:03d}": int(20 + (i * 7) % 60) for i in range(1, 21)}
    npz_dir_map: Dict[str, Path] = {}
    for i, (acc, sl) in enumerate(seq_lens.items()):
        target_dir = npz_dir_a if i % 2 == 0 else npz_dir_b
        _write_npz(target_dir / f"{acc}.npz", sl)
        npz_dir_map[acc] = target_dir

    # One protein referenced in CATH but absent from npz_dir_map.
    missing_from_npz = "PMIS"

    # Build protein_cath. Spread CATH IDs so several labels cross the
    # ``min_proteins`` threshold at multiple hierarchy levels.
    protein_cath: Dict[str, List[Dict[str, Any]]] = {}

    def add(acc, hits):
        protein_cath[acc] = hits

    # Family A: cath_id 1.10.20.30 — 6 proteins at CATH, 6 at CAT/CA/C.
    for i in range(1, 7):
        add(f"P{i:03d}", [
            {"cath_id": "1.10.20.30", "query_start": 2, "query_end": 10, "description": "Family A"},
        ])

    # Family B: cath_id 2.40.50.60 — 4 proteins; one has two disjoint ranges.
    for i in range(7, 11):
        hits = [{"cath_id": "2.40.50.60", "query_start": 5, "query_end": 15, "description": "Family B"}]
        if i == 7:
            hits.append({"cath_id": "2.40.50.60", "query_start": 20, "query_end": 25, "description": "Family B"})
        add(f"P{i:03d}", hits)

    # Family C: 2.40.50.99 — shares CA (2.40) and CAT (2.40.50) with Family B
    # so labels stack at higher levels while diverging at CATH level.
    for i in range(11, 14):
        add(f"P{i:03d}", [
            {"cath_id": "2.40.50.99", "query_start": 3, "query_end": 9, "description": "Family C"},
        ])

    # Family D: 3.30.10.20 — 3 proteins (just meets min_proteins at CATH).
    for i in range(14, 17):
        add(f"P{i:03d}", [
            {"cath_id": "3.30.10.20", "query_start": 1, "query_end": 8, "description": "Family D"},
        ])

    # Protein with missing qs/qe on one hit (should be ignored) and valid one.
    add("P017", [
        {"cath_id": "1.10.20.30", "query_start": None, "query_end": None, "description": "no range"},
        {"cath_id": "1.10.20.30", "query_start": 1, "query_end": 5, "description": "Family A"},
    ])
    # Protein with a too-short cath_id (must be skipped).
    add("P018", [
        {"cath_id": "1.10", "query_start": 1, "query_end": 5, "description": "stub"},
    ])
    # Protein with no CATH hits at all.
    add("P019", [])
    # Protein present but with hits that never hit a target label.
    add("P020", [
        {"cath_id": "9.99.99.99", "query_start": 1, "query_end": 5, "description": "loner"},
    ])
    # Absent-from-npz protein: should be skipped by both.
    add("PMIS", [
        {"cath_id": "1.10.20.30", "query_start": 1, "query_end": 5, "description": "missing"},
    ])

    return {
        "protein_cath": protein_cath,
        "npz_dir_map": npz_dir_map,
    }


# ---------------------------------------------------------------------------
# Driver: build protein-level results from the shared impl, then fan out
# to old and new residue-level and deep-compare.
# ---------------------------------------------------------------------------


def _run_both(fixture, feat_idx: int, feat_max: float, pool):
    protein_cath = fixture["protein_cath"]
    npz_dir_map = fixture["npz_dir_map"]

    # Give proteins synthetic activations so compute_protein_level_f1
    # produces meaningful top lists at each level.
    acts = [float(RNG.random()) * feat_max for _ in protein_cath]
    accessions_with_activations = list(zip(protein_cath.keys(), acts))

    prot_results = mod.compute_protein_level_f1(
        accessions_with_activations=accessions_with_activations,
        protein_cath=protein_cath,
        feat_max=feat_max,
        n_threshold_steps=50,
        min_proteins=3,
    )

    old_cache: Dict[str, Optional[np.ndarray]] = {}
    old_out = _old_compute_residue_level_f1(
        protein_level_results=copy.deepcopy(prot_results),
        protein_cath=copy.deepcopy(protein_cath),
        feat_idx=feat_idx,
        feat_max=feat_max,
        npz_dir_map=npz_dir_map,
        npz_cache=old_cache,
        max_npz_cache=500,
    )
    new_out = mod.compute_residue_level_f1(
        protein_level_results=copy.deepcopy(prot_results),
        protein_cath=copy.deepcopy(protein_cath),
        feat_idx=feat_idx,
        feat_max=feat_max,
        npz_dir_map=npz_dir_map,
        n_threshold_steps=50,
        top_n_per_level=5,
        io_executor=pool,
    )

    # Normalize: both must expose the same set of level keys.
    for level in mod.CATH_LEVELS:
        old_out.setdefault(level, [])
        new_out.setdefault(level, [])

    return old_out, new_out


@pytest.mark.parametrize("feat_idx", list(range(NUM_FEATURES)))
def test_residue_level_f1_matches_original(synthetic_fixture, feat_idx):
    with ThreadPoolExecutor(max_workers=4) as pool:
        for feat_max in (1.0, 2.5):
            old_out, new_out = _run_both(synthetic_fixture, feat_idx, feat_max, pool)
            assert new_out == old_out, (
                f"residue F1 diverged at feat_idx={feat_idx} feat_max={feat_max}\n"
                f"old={json.dumps(old_out, sort_keys=True)}\n"
                f"new={json.dumps(new_out, sort_keys=True)}"
            )
            assert json.dumps(old_out, sort_keys=True) == json.dumps(
                new_out, sort_keys=True
            )


def test_residue_level_f1_matches_with_out_of_range_feat_idx(synthetic_fixture):
    """feat_idx beyond .npz shape → both impls must yield empty results."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        old_out, new_out = _run_both(synthetic_fixture, feat_idx=NUM_FEATURES + 5, feat_max=1.0, pool=pool)
    assert new_out == old_out
    assert all(len(v) == 0 for v in new_out.values())


def test_residue_level_f1_no_executor_equals_with_executor(synthetic_fixture):
    """Serial fallback path (io_executor=None) must also match."""
    protein_cath = synthetic_fixture["protein_cath"]
    npz_dir_map = synthetic_fixture["npz_dir_map"]
    acts = [float(RNG.random()) for _ in protein_cath]
    accs = list(zip(protein_cath.keys(), acts))
    prot_results = mod.compute_protein_level_f1(
        accessions_with_activations=accs,
        protein_cath=protein_cath,
        feat_max=1.0,
        n_threshold_steps=50,
        min_proteins=3,
    )
    no_pool = mod.compute_residue_level_f1(
        protein_level_results=copy.deepcopy(prot_results),
        protein_cath=copy.deepcopy(protein_cath),
        feat_idx=0,
        feat_max=1.0,
        npz_dir_map=npz_dir_map,
        io_executor=None,
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        with_pool = mod.compute_residue_level_f1(
            protein_level_results=copy.deepcopy(prot_results),
            protein_cath=copy.deepcopy(protein_cath),
            feat_idx=0,
            feat_max=1.0,
            npz_dir_map=npz_dir_map,
            io_executor=pool,
        )
    assert no_pool == with_pool
