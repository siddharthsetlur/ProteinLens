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
    _encode_sequence,
    _meme_available,
    _pwm_log_odds,
    _scan_pwm,
    _select_high_activation_windows,
    run_motif_pwm_enrichment,
)


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
    )

    run_motif_pwm_enrichment(config)

    out = config.motif_pwm_enrichment_dir / "0000.json"
    assert out.exists()
    with open(out) as f:
        result = json.load(f)
    assert result["n_motifs_discovered"] >= 1
    m0 = result["motifs"][0]
    assert m0["best_f1"] > 0.0
    # Consensus should contain at least 3 of the 5 implanted residues
    overlap = sum(1 for ch in implanted if ch in m0["consensus"])
    assert overlap >= 3, f"expected consensus overlap with {implanted}, got {m0['consensus']}"
