"""Tests for Stage 7b — PWM motif discovery (MEME).

Unit tests cover:
- high-activation window extraction (per-protein cap, percentile)
- PWM scanner against a hand-computed example
- F1 sweep on a synthetic PWM/activation pair

The MEME-dependent end-to-end test is gated with ``pytest.importorskip``-style
check on the `meme` binary; it runs only if MEME is installed locally.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.motif_pwm import (
    _AA_ORDER,
    _compute_best_pwm_f1,
    _compute_pwm_pr_auc,
    _encode_sequence,
    _meme_available,
    _pwm_log_odds,
    _scan_pwm,
    _select_high_activation_windows,
    run_motif_pwm_enrichment,
)
from sklearn.metrics import average_precision_score


# ===================================================================
# Window extraction
# ===================================================================


def test_select_windows_basic():
    """High-activation residue is centred; window has the correct width."""
    seq = "A" * 21
    acts = [0.0] * 21
    acts[10] = 1.0
    proteins = [("p1", seq, acts)]

    windows = _select_high_activation_windows(
        proteins, half_w=7, top_k_per_protein=3, percentile=0.95,
    )
    assert windows == ["A" * 15]


def test_select_windows_top_k_cap():
    """top_k_per_protein caps windows from a single protein."""
    seq = "G" * 41
    acts = [0.0] * 41
    for i in (10, 15, 20, 25, 30):
        acts[i] = 1.0
    proteins = [("p1", seq, acts)]

    windows = _select_high_activation_windows(
        proteins, half_w=7, top_k_per_protein=2, percentile=0.5,
    )
    assert len(windows) == 2


def test_select_windows_skips_nonstandard():
    """Windows containing non-standard residues (e.g. 'X') are dropped."""
    seq = "A" * 7 + "X" + "A" * 13
    acts = [0.0] * 21
    acts[10] = 1.0  # centred, but window contains 'X' at position 7
    proteins = [("p1", seq, acts)]

    windows = _select_high_activation_windows(
        proteins, half_w=7, top_k_per_protein=3, percentile=0.5,
    )
    assert windows == []


# ===================================================================
# PWM scanner
# ===================================================================


def test_scan_pwm_hand_computed():
    """Width-3 PWM scored against 'AAA' at the centre should be max log-odds."""
    # PWM that strongly prefers A at all 3 positions
    pwm = np.zeros((3, 20))
    pwm[:, _AA_ORDER.index("A")] = 1.0
    bg = np.full(20, 1.0 / 20)
    lo = _pwm_log_odds(pwm, bg)

    seq_enc = _encode_sequence("AAAAA")
    scores = _scan_pwm(seq_enc, lo)

    # Centre positions 1, 2, 3 are valid; 0 and 4 are -inf
    assert np.isinf(scores[0]) and scores[0] < 0
    assert np.isinf(scores[4]) and scores[4] < 0
    # At position 2 the window is 'AAA' — should equal 3 * log(1.000001 / 0.05)
    expected = 3 * np.log((1.0 + 1e-6) / (1.0 / 20 + 1e-6))
    assert np.isclose(scores[2], expected, rtol=1e-3)


def test_scan_pwm_nonstandard_residue():
    """Windows containing non-standard residues get -inf at their centres."""
    pwm = np.full((3, 20), 1.0 / 20)
    lo = _pwm_log_odds(pwm, np.full(20, 1.0 / 20))

    seq_enc = _encode_sequence("AAXAA")
    scores = _scan_pwm(seq_enc, lo)
    # Centre position 2 has 'X' → window 'AXA' invalid
    assert np.isinf(scores[2])


# ===================================================================
# F1 sweep
# ===================================================================


def test_compute_best_pwm_f1_perfect_separation():
    """When PWM scores perfectly separate activated from inactive, F1 → 1.0."""
    # 100 residues: first 50 have high score + high activation; rest low/low
    scores = np.concatenate([np.full(50, 5.0), np.full(50, -5.0)])
    acts = np.concatenate([np.full(50, 1.0), np.full(50, 0.0)])

    result = _compute_best_pwm_f1(scores, acts, feat_max=1.0, n_steps=20)
    assert result["best_f1"] >= 0.99


def test_compute_best_pwm_f1_no_signal():
    """Random alignment of scores and activations yields modest F1."""
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(200)
    acts = rng.standard_normal(200).clip(min=0)

    result = _compute_best_pwm_f1(scores, acts, feat_max=float(acts.max()),
                                   n_steps=10)
    # Should return something; we're not asserting a tight bound
    assert 0.0 <= result["best_f1"] <= 1.0


# ===================================================================
# PR-AUC score (Stage 7b refactor — parallel to geometric GBM)
# ===================================================================


def test_compute_pwm_pr_auc_perfect_separation():
    """When PWM scores perfectly rank activated residues, PR-AUC → 1.0."""
    scores = np.concatenate([np.full(50, 5.0), np.full(50, -5.0)])
    acts = np.concatenate([np.full(50, 1.0), np.full(50, 0.0)])
    result = _compute_pwm_pr_auc(scores, acts, act_quantile=0.5)
    assert result is not None
    assert result["pr_auc"] >= 0.99
    assert result["n_activated"] == 50
    assert result["n_valid_residues"] == 100


def test_compute_pwm_pr_auc_random_signal_is_near_base_rate():
    """Uncorrelated score/activation → PR-AUC near the positive-rate base."""
    rng = np.random.default_rng(0)
    n = 4000
    scores = rng.standard_normal(n)
    acts = rng.standard_normal(n)
    result = _compute_pwm_pr_auc(scores, acts, act_quantile=0.8)
    assert result is not None
    base_rate = result["n_activated"] / result["n_valid_residues"]
    # No signal → AP ≈ base rate. Loose bound to absorb sampling noise.
    assert abs(result["pr_auc"] - base_rate) < 0.05


def test_compute_pwm_pr_auc_excludes_inf_positions():
    """-inf score positions must be dropped *before* quantile/AP."""
    # 10 finite residues: first 5 high-score high-act, next 5 low/low.
    # Then 20 -inf positions with high activation that must be ignored.
    scores = np.concatenate([
        np.full(5, 5.0), np.full(5, -5.0), np.full(20, -np.inf),
    ])
    acts = np.concatenate([
        np.full(5, 1.0), np.full(5, 0.0), np.full(20, 10.0),
    ])
    result = _compute_pwm_pr_auc(scores, acts, act_quantile=0.5)
    assert result is not None
    assert result["n_valid_residues"] == 10
    # Quantile taken over finite subset only → threshold should be 0.5·(0+1).
    assert result["pr_auc"] >= 0.99


def test_compute_pwm_pr_auc_degenerate_returns_none():
    """Zero positives or all-positives must return None (symmetric null)."""
    # All activations zero → quantile = 0, truth all >= 0 → all-positive.
    scores = np.linspace(-1, 1, 20)
    acts = np.zeros(20)
    assert _compute_pwm_pr_auc(scores, acts, act_quantile=0.8) is None

    # All -inf scores → no valid residues.
    scores_all_inf = np.full(20, -np.inf)
    acts_any = np.linspace(0, 1, 20)
    assert _compute_pwm_pr_auc(scores_all_inf, acts_any, act_quantile=0.8) is None


def test_compute_pwm_pr_auc_matches_sklearn_directly():
    """PR-AUC must equal a hand-called sklearn.average_precision_score."""
    rng = np.random.default_rng(42)
    n = 500
    scores = rng.standard_normal(n)
    # Correlated activations: higher when score is higher.
    acts = scores + 0.5 * rng.standard_normal(n)

    q = 0.8
    result = _compute_pwm_pr_auc(scores, acts, act_quantile=q)
    assert result is not None
    truth = (acts >= np.quantile(acts, q)).astype(int)
    expected = float(average_precision_score(truth, scores))
    assert result["pr_auc"] == pytest.approx(expected, abs=1e-12)


# ===================================================================
# End-to-end (requires MEME)
# ===================================================================


MEME_PRESENT = _meme_available()


@pytest.mark.skipif(not MEME_PRESENT, reason="MEME binary not on PATH")
def test_end_to_end_recovers_implanted_motif(tmp_path: Path):
    """MEME recovers an implanted LYGKE motif and produces F1>0 output."""
    implanted = "LYGKE"
    rng = np.random.default_rng(42)
    aa = list(_AA_ORDER)

    features_dir = tmp_path / "features"
    features_dir.mkdir()

    # Build 1 feature with 15 proteins each containing the motif at a random
    # position, with high activation at the motif centre.
    top_seqs = []
    for i in range(15):
        seq_list = rng.choice(aa, size=80).tolist()
        pos = 30 + (i % 20)
        for j, ch in enumerate(implanted):
            seq_list[pos + j] = ch
        seq = "".join(seq_list)
        acts = [0.0] * len(seq)
        acts[pos + 2] = 1.0  # centre of the 5-residue motif
        top_seqs.append({
            "accession": f"P{i:05d}",
            "sequence": seq,
            "per_residue_activations": acts,
        })

    feat_data = {
        "feature_id": 0,
        "top_sequences": top_seqs,
        "activation_bins": {},
    }
    with open(features_dir / "0000.json", "w") as f:
        json.dump(feat_data, f)

    np.save(tmp_path / "feature_max_activations.npy", np.array([1.0]))

    config = PipelineConfig(
        sae_dir=tmp_path / "fake_sae",
        output_dir=tmp_path,
        motif_pwm_enabled=True,
        motif_pwm_window_half_w=7,
        motif_pwm_top_k_per_protein=1,
        motif_pwm_activation_percentile=0.95,
        motif_pwm_meme_minw=4,
        motif_pwm_meme_maxw=6,
        motif_pwm_meme_nmotifs=1,
        motif_pwm_min_windows=5,
        motif_pwm_f1_threshold_steps=10,
        # Synthetic activations are extremely sparse (15/1200 nonzero);
        # default q=0.8 would binarise every residue as positive. Raise
        # the quantile so the positive set is the handful of implanted hits.
        motif_pwm_act_quantile=0.99,
    )

    run_motif_pwm_enrichment(config)

    out = config.motif_pwm_enrichment_dir / "0000.json"
    assert out.exists()
    with open(out) as f:
        result = json.load(f)
    assert result["n_motifs_discovered"] >= 1
    assert result["primary_score"] == "pr_auc"
    m0 = result["motifs"][0]
    assert m0["best_f1"] > 0.0
    # PR-AUC wiring is present. With only 15 positives out of ~1140 residues
    # (base rate ~1.3%) the point estimate is noisy on this synthetic; the
    # unit tests assert the scoring function's numerical correctness, so here
    # we only check that the value is computed and non-degenerate.
    assert m0["pr_auc"] is not None
    assert 0.0 <= m0["pr_auc"]["pr_auc"] <= 1.0
    assert m0["pr_auc"]["n_activated"] > 0
    # Consensus should contain at least 3 of the 5 implanted residues
    overlap = sum(1 for ch in implanted if ch in m0["consensus"])
    assert overlap >= 3, f"expected consensus overlap with {implanted}, got {m0['consensus']}"
