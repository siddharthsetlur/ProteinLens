"""Stage 7b — PWM motif discovery per SAE feature (MEME).

Complements Stage 7 (fixed 3-mer F1 search) with a richer motif model.
For each feature we:

1. Extract fixed-width windows around the highest-activation residues
   across the feature's pooled top-activating proteins.
2. Run MEME (EM-based PWM discovery) on those windows to recover up to
   N position-weight matrices.
3. Scan each discovered PWM across the feature's full-sequence pool and
   compute residue-level F1 via a **joint (activation, pwm-score)
   threshold sweep**.

Designed for sparse-autoencoder features with ~20 top-activating
proteins — MEME's EM is appropriate at low N; STREME is not.

**Not directly comparable to Stage 7.** Stage 7 argmaxes F1 over a 1-D
sweep of the activation threshold τ alone (k-mers are exact-match, no
score threshold). Stage 7b argmaxes over a 2-D (τ, σ) grid where σ is
the PWM log-odds cutoff. Optimising over an extra free threshold
inflates the max F1 under the null, so the *magnitude* of ``pwm_f1`` is
not directly comparable to ``motif_f1`` even though both are F1 scores.
Ranking / p-values from ``compute_permutation_null.py --include-pwm`` are
still valid (observed and null use the same grid) — it is only the
point estimate that is biased upward by the extra degree of freedom.

**Background model.** Log-odds use the empirical AA frequency of the
pooled sequences for this feature by default
(``config.motif_pwm_background="empirical"``). A uniform 1/20 background
can be selected with ``"uniform"`` but biases scoring toward motifs
containing rare amino acids (W, C, M); uniform is retained mainly for
debugging / backward compatibility.

**Reproducibility.** MEME's EM is stochastic; we pass ``--seed`` from
``config.motif_pwm_meme_seed`` (default 0) so reruns produce identical
PWMs.

**Outputs:**
- ``motif_pwm_enrichment/{feat_idx:04d}.json`` — per-feature results
  with up to N PWMs, each scored by best residue-level F1. Includes
  ``sweep_type: "2d_tau_sigma"`` to make the protocol self-documenting.
- ``motif_pwm_enrichment/summary.json`` — keyed by feature id.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.motif_enrichment import (
    _pool_proteins_for_feature,
)

_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX = {aa: i for i, aa in enumerate(_AA_ORDER)}


# ===================================================================
# Window extraction
# ===================================================================


def _select_high_activation_windows(
    proteins: List[Tuple[str, str, List[float]]],
    half_w: int,
    top_k_per_protein: int,
    percentile: float,
) -> List[str]:
    """Extract fixed-width windows centred on high-activation residues.

    For each protein, picks up to *top_k_per_protein* residues whose activation
    exceeds the *percentile* quantile of that protein's activations, then
    extracts the ``2*half_w+1``-residue window around each. Windows containing
    non-standard amino acids or crossing sequence boundaries are skipped.

    Args:
        proteins: List of ``(accession, sequence, per_residue_activations)``.
        half_w: Half-width — window length is ``2*half_w+1``.
        top_k_per_protein: Cap on windows per protein (prevents long proteins
            from dominating MEME's input).
        percentile: Per-protein quantile threshold (0–1).

    Returns:
        List of window strings (each of length ``2*half_w+1``).
    """
    w = 2 * half_w + 1
    windows: List[str] = []
    for _acc, seq, pra in proteins:
        if len(seq) != len(pra) or len(seq) < w:
            continue
        acts = np.asarray(pra, dtype=np.float64)
        if acts.max() <= 0:
            continue
        thr = float(np.quantile(acts, percentile))
        # Require strictly positive activation so that features with mostly-zero
        # activations (common for sparse SAEs) don't pull in neutral positions.
        pos = np.arange(len(acts))
        cand = np.where((acts >= thr) & (acts > 0) &
                        (pos >= half_w) & (pos < len(acts) - half_w))[0]
        if cand.size == 0:
            continue
        # Take top-K by activation
        order = cand[np.argsort(-acts[cand])][:top_k_per_protein]
        for i in order:
            win = seq[i - half_w : i + half_w + 1]
            if len(win) == w and all(ch in _AA_INDEX for ch in win):
                windows.append(win)
    return windows


# ===================================================================
# MEME runner
# ===================================================================


def _meme_available() -> bool:
    """Return True iff the `meme` binary is resolvable on PATH."""
    return shutil.which("meme") is not None


def _write_fasta(windows: List[str], path: Path) -> None:
    """Write windows to a FASTA file with synthetic IDs."""
    with open(path, "w") as f:
        for i, w in enumerate(windows):
            f.write(f">w{i}\n{w}\n")


def _parse_meme_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """Parse ``meme.xml`` into a list of motif dicts.

    Each motif dict contains:
      - ``id``: MEME motif id (e.g. ``motif_1``)
      - ``name``: consensus-like name from MEME
      - ``width``: int
      - ``e_value``: float
      - ``pwm``: ``(width, 20)`` float array, probabilities in ``_AA_ORDER``
      - ``consensus``: best-residue-per-column string
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Alphabet letter -> column index mapping in MEME's probability rows.
    alphabet_letters: List[str] = []
    alphabet = root.find("./training_set/alphabet") or root.find("./alphabet")
    if alphabet is not None:
        for letter in alphabet.findall("letter"):
            sym = letter.get("symbol")
            if sym is not None:
                alphabet_letters.append(sym)

    motifs: List[Dict[str, Any]] = []
    for m in root.findall("./motifs/motif"):
        width = int(m.get("width", "0"))
        e_value = float(m.get("e_value", "1.0"))
        mid = m.get("id", "motif")
        name = m.get("name", mid)

        probs_arrays: List[List[float]] = []
        for pos in m.findall("./probabilities/alphabet_matrix/alphabet_array"):
            row = [0.0] * len(alphabet_letters)
            for v in pos.findall("value"):
                letter = v.get("letter_id") or v.get("letter")
                if letter is None:
                    continue
                try:
                    idx = alphabet_letters.index(letter)
                except ValueError:
                    continue
                row[idx] = float(v.text or "0")
            probs_arrays.append(row)

        if not probs_arrays:
            continue

        raw = np.asarray(probs_arrays, dtype=np.float64)  # (width, |alpha|)

        # Reindex columns into canonical _AA_ORDER (20 standard AAs).
        pwm = np.zeros((raw.shape[0], 20), dtype=np.float64)
        for j, letter in enumerate(alphabet_letters):
            if letter in _AA_INDEX:
                pwm[:, _AA_INDEX[letter]] = raw[:, j]

        # Normalise each row to sum 1 (MEME may use pseudocounts).
        row_sums = pwm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        pwm = pwm / row_sums

        consensus = "".join(_AA_ORDER[int(i)] for i in pwm.argmax(axis=1))
        motifs.append({
            "id": mid,
            "name": name,
            "width": width or pwm.shape[0],
            "e_value": e_value,
            "pwm": pwm,
            "consensus": consensus,
        })

    return motifs


_MEME_SUPPORTS_SEED: Optional[bool] = None


def _meme_supports_seed() -> bool:
    """Return True if the installed MEME accepts ``-seed``.

    MEME 4.11.2 (bioconda) does not have ``-seed``; 5.x does. We probe once
    and cache the result so per-feature calls don't fork extra processes.
    """
    global _MEME_SUPPORTS_SEED
    if _MEME_SUPPORTS_SEED is not None:
        return _MEME_SUPPORTS_SEED
    try:
        r = subprocess.run(
            ["meme", "-h"], capture_output=True, text=True, timeout=5,
        )
        # MEME prints help to stderr regardless of returncode.
        help_text = (r.stdout or "") + (r.stderr or "")
        _MEME_SUPPORTS_SEED = "-seed" in help_text
    except (subprocess.TimeoutExpired, FileNotFoundError):
        _MEME_SUPPORTS_SEED = False
    return _MEME_SUPPORTS_SEED


def _run_meme(
    windows: List[str],
    minw: int,
    maxw: int,
    nmotifs: int,
    timeout_s: int,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Run MEME on the given windows and return parsed motifs.

    Raises:
        RuntimeError: If `meme` is not on PATH.
    """
    if not _meme_available():
        raise RuntimeError(
            "MEME binary not found on PATH. Install via "
            "`conda install -c bioconda meme` and ensure the `meme` "
            "executable is available."
        )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fasta = tmp_path / "windows.fasta"
        out_dir = tmp_path / "meme_out"
        _write_fasta(windows, fasta)

        cmd = [
            "meme", str(fasta),
            "-protein",
            "-oc", str(out_dir),
            "-nmotifs", str(nmotifs),
            "-minw", str(minw),
            "-maxw", str(maxw),
            "-mod", "zoops",
            "-nostatus",
        ]
        # MEME 4.11.2 (bioconda) does not accept -seed; newer builds do.
        # Probe once and include it only if supported. Determinism across
        # builds is then best-effort — the permutation null uses its own
        # RNG so statistical validity does not depend on this flag.
        if _meme_supports_seed():
            cmd += ["-seed", str(seed)]
        _ = seed  # silence unused-arg warning when flag not supported
        try:
            subprocess.run(
                cmd, check=True, timeout=timeout_s,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except subprocess.TimeoutExpired:
            return []
        except subprocess.CalledProcessError:
            return []

        xml_path = out_dir / "meme.xml"
        if not xml_path.exists():
            return []
        return _parse_meme_xml(xml_path)


# ===================================================================
# PWM scanner
# ===================================================================


def _encode_sequence(seq: str) -> np.ndarray:
    """Return int array of AA indices (20 = non-standard, use as mask)."""
    enc = np.full(len(seq), 20, dtype=np.int8)
    for i, ch in enumerate(seq):
        idx = _AA_INDEX.get(ch)
        if idx is not None:
            enc[i] = idx
    return enc


def _pwm_log_odds(pwm: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Convert a PWM (width, 20) to log-odds vs. background (20,)."""
    eps = 1e-6
    return np.log((pwm + eps) / (bg[None, :] + eps))


def _scan_pwm(
    seq_enc: np.ndarray,
    log_odds: np.ndarray,
) -> np.ndarray:
    """Score every centre position in an encoded sequence against a PWM.

    Returns an array of length ``len(seq_enc)`` with scores. Positions too
    close to either end (centre would require out-of-bounds residues) and
    windows containing non-standard residues are filled with ``-inf``.
    """
    w = log_odds.shape[0]
    half = w // 2
    n = len(seq_enc)
    scores = np.full(n, -np.inf, dtype=np.float64)
    if n < w:
        return scores
    # Build (n - w + 1, w) view of window starts
    # Valid centre positions: start + half, for start in [0, n-w]
    for start in range(0, n - w + 1):
        window = seq_enc[start : start + w]
        if (window == 20).any():
            continue
        scores[start + half] = log_odds[np.arange(w), window].sum()
    return scores


# ===================================================================
# F1 sweep over PWM score thresholds
# ===================================================================


def _compute_best_pwm_f1(
    pwm_scores: np.ndarray,
    activations: np.ndarray,
    feat_max: float,
    n_steps: int,
) -> Dict[str, float]:
    """Sweep PWM-score thresholds at a fixed activation threshold policy.

    We mirror Stage 7: positive = residues with activation > τ (swept), then
    find the PWM-score cutoff σ that best separates activated from
    non-activated. To keep dimensionality sane, we sweep both:
      - τ over n_steps thresholds of activation
      - σ over n_steps thresholds of PWM score (in the finite-score range)
    and return the (τ, σ) pair maximising F1.

    Only finite PWM scores contribute to the positive/negative candidate set;
    -inf positions (out-of-bounds or non-standard) are excluded.
    """
    valid = np.isfinite(pwm_scores)
    if not valid.any() or feat_max <= 0:
        return {}
    s = pwm_scores[valid]
    a = activations[valid]

    tau_grid = np.linspace(0, feat_max, n_steps + 1)[1:]
    s_lo, s_hi = float(s.min()), float(s.max())
    if s_lo == s_hi:
        return {}
    sig_grid = np.linspace(s_lo, s_hi, n_steps + 1)[:-1]

    # activated_matrix[t, i]: True if a[i] > tau_grid[t]
    activated = a[None, :] > tau_grid[:, None]          # (n_steps, N)
    n_activated = activated.sum(axis=1).astype(float)   # (n_steps,)

    # predicted_matrix[k, i]: True if s[i] >= sig_grid[k]
    predicted = s[None, :] >= sig_grid[:, None]          # (n_steps, N)
    n_predicted = predicted.sum(axis=1).astype(float)    # (n_steps,)

    # tp[t, k] = sum_i activated[t, i] & predicted[k, i]
    #         = activated @ predicted.T
    tp = activated.astype(np.float32) @ predicted.astype(np.float32).T  # (nt, nk)
    fp = n_predicted[None, :] - tp
    fn = n_activated[:, None] - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    best = np.unravel_index(int(np.argmax(f1)), f1.shape)
    t_idx, k_idx = int(best[0]), int(best[1])
    return {
        "best_f1": round(float(f1[t_idx, k_idx]), 4),
        "best_activation_threshold": round(float(tau_grid[t_idx]), 4),
        "best_activation_threshold_normalized": round(
            float(tau_grid[t_idx]) / feat_max, 4
        ),
        "best_pwm_threshold": round(float(sig_grid[k_idx]), 4),
        "precision_at_best": round(float(precision[t_idx, k_idx]), 4),
        "recall_at_best": round(float(recall[t_idx, k_idx]), 4),
        "n_predicted_positive": int(n_predicted[k_idx]),
        "n_activated": int(n_activated[t_idx]),
        "n_true_positives": int(tp[t_idx, k_idx]),
    }


# ===================================================================
# PR-AUC score (parallel to Stage 6c geometric GBM)
# ===================================================================


def _compute_pwm_pr_auc(
    pwm_scores: np.ndarray,
    activations: np.ndarray,
    act_quantile: float,
) -> Optional[Dict[str, float]]:
    """PR-AUC of PWM log-odds against activation > quantile.

    Parallel to Stage 6c: truth = residue activation >= per-feature quantile
    of the valid-position activations; predictor = PWM log-odds per residue.
    Threshold-free along the predictor axis, single fixed threshold on truth
    — so the returned number is directly comparable to
    ``geometric_residue_level.concordance.avg_precision`` from Stage 6c.

    -inf score positions (out-of-bounds / non-standard) are masked *before*
    the quantile is computed so observed and null see the same residue set.

    **Sparse-feature fallback.** SAE features are typically sparse: if
    fewer than ``(1 - act_quantile)`` of valid residues carry non-zero
    activation, ``np.quantile`` collapses to 0 and ``a >= 0`` would flag
    every residue as positive. We detect this and fall back to ``a > 0``
    (every non-zero residue is a positive), which matches the natural
    semantics for a sparse feature. The returned dict records the actual
    threshold used and a ``fallback_nonzero`` flag so the fallback is
    never silent.

    **Observed/null asymmetry warning.** This helper returns ``None`` on
    degenerate binarisation; the null-script helper
    ``_best_pwm_pr_auc_across`` converts None to 0.0 so the p-value stays
    well-defined. Per-feature JSON surfaces the raw ``None`` and must not
    be mixed with the null-script's ``null_distributions.pwm_pr_auc``
    without explicit handling of the None case.
    """
    valid = np.isfinite(pwm_scores)
    if not valid.any():
        return None
    s = pwm_scores[valid]
    a = activations[valid]
    if s.size == 0:
        return None

    q = float(np.quantile(a, act_quantile))
    fallback_nonzero = False
    if q <= 0.0:
        # Sparse-feature fallback: quantile is uninformative (every zero
        # residue would be positive). Use strict non-zero activation as
        # the positive class instead.
        truth = (a > 0.0).astype(np.int8)
        fallback_nonzero = True
    else:
        truth = (a >= q).astype(np.int8)
    n_pos = int(truth.sum())
    # Still degenerate (e.g. all activations are zero, or all are equal
    # and above q): PR-AUC is not informative, surface as "no score".
    # Callers must handle observed and null symmetrically.
    if n_pos == 0 or n_pos == truth.size:
        return None

    ap = float(average_precision_score(truth, s))
    return {
        "pr_auc": ap,
        "activation_threshold": q,
        "act_quantile": act_quantile,
        "fallback_nonzero": fallback_nonzero,
        "n_activated": n_pos,
        "n_valid_residues": int(truth.size),
    }


# ===================================================================
# Per-feature analysis
# ===================================================================


_UNIFORM_BG = np.full(20, 1.0 / 20, dtype=np.float64)


def _empirical_aa_background(proteins: List[Tuple[str, str, List[float]]]) -> np.ndarray:
    """Return empirical AA frequency over the pooled sequences, shape (20,).

    Non-standard residues are excluded. Falls back to uniform if the pool
    contains zero valid residues (pathological edge case).
    """
    counts = np.zeros(20, dtype=np.float64)
    for _acc, seq, _pra in proteins:
        for ch in seq:
            idx = _AA_INDEX.get(ch)
            if idx is not None:
                counts[idx] += 1.0
    total = counts.sum()
    if total <= 0:
        return _UNIFORM_BG.copy()
    # Pseudocount to avoid zero entries for AAs absent from the pool
    # (keeps log-odds finite and doesn't skew frequent-AA logs materially).
    counts += 1.0
    return counts / counts.sum()


def _analyze_feature_pwm(
    feature_data: Dict[str, Any],
    feat_max: float,
    config: PipelineConfig,
) -> Optional[Dict[str, Any]]:
    proteins = _pool_proteins_for_feature(feature_data)
    if not proteins:
        return None

    windows = _select_high_activation_windows(
        proteins,
        half_w=config.motif_pwm_window_half_w,
        top_k_per_protein=config.motif_pwm_top_k_per_protein,
        percentile=config.motif_pwm_activation_percentile,
    )
    if len(windows) < config.motif_pwm_min_windows:
        return None

    motifs = _run_meme(
        windows,
        minw=config.motif_pwm_meme_minw,
        maxw=config.motif_pwm_meme_maxw,
        nmotifs=config.motif_pwm_meme_nmotifs,
        timeout_s=config.motif_pwm_meme_timeout_s,
        seed=config.motif_pwm_meme_seed,
    )
    if not motifs:
        return None

    # Select background model: empirical (per-feature AA freq) or uniform.
    if getattr(config, "motif_pwm_background", "empirical") == "uniform":
        bg = _UNIFORM_BG
    else:
        bg = _empirical_aa_background(proteins)

    # Score each motif against the full-sequence pool.
    motif_results: List[Dict[str, Any]] = []
    for motif in motifs:
        log_odds = _pwm_log_odds(motif["pwm"], bg)
        all_scores: List[np.ndarray] = []
        all_acts: List[np.ndarray] = []
        for _acc, seq, pra in proteins:
            if len(seq) != len(pra):
                continue
            enc = _encode_sequence(seq)
            scores = _scan_pwm(enc, log_odds)
            all_scores.append(scores)
            all_acts.append(np.asarray(pra, dtype=np.float64))
        if not all_scores:
            continue
        scores_cat = np.concatenate(all_scores)
        acts_cat = np.concatenate(all_acts)

        f1_result = _compute_best_pwm_f1(
            scores_cat, acts_cat, feat_max,
            n_steps=config.motif_pwm_f1_threshold_steps,
        )
        if not f1_result:
            continue

        # PR-AUC is the primary Stage 7b score (parallel to Stage 6c GBM).
        # F1 fields are retained as a diagnostic. Missing pr_auc (degenerate
        # binarisation) is rendered as an explicit null so downstream code
        # can distinguish "not computable" from "computed as zero".
        pr_auc_result = _compute_pwm_pr_auc(
            scores_cat, acts_cat,
            act_quantile=config.motif_pwm_act_quantile,
        )

        motif_results.append({
            "motif_id": motif["id"],
            "consensus": motif["consensus"],
            "width": motif["width"],
            "e_value": motif["e_value"],
            # Full precision so downstream permutation-null sees the same
            # PWM values Stage 7b scored with (avoids rounding drift).
            "pwm": motif["pwm"].tolist(),
            "aa_order": _AA_ORDER,
            "pr_auc": pr_auc_result,
            **f1_result,
        })

    if not motif_results:
        return None

    # Sort by PR-AUC (primary); motifs with no PR-AUC sort last.
    motif_results.sort(
        key=lambda r: (r.get("pr_auc") or {}).get("pr_auc", -1.0),
        reverse=True,
    )

    return {
        "feature_id": feature_data["feature_id"],
        "feature_max_activation": round(float(feat_max), 6),
        "n_proteins_evaluated": len(proteins),
        "n_windows": len(windows),
        "n_motifs_discovered": len(motifs),
        "sweep_type": "2d_tau_sigma",
        "primary_score": "pr_auc",
        "background_model": (
            "uniform" if getattr(config, "motif_pwm_background", "empirical") == "uniform"
            else "empirical"
        ),
        "motifs": motif_results,
    }


# ===================================================================
# Public API
# ===================================================================


def run_motif_pwm_enrichment(config: PipelineConfig) -> None:
    """Execute the optional PWM motif discovery stage (Stage 7b).

    Gated by ``config.motif_pwm_enabled``; a no-op if the flag is False.
    Per-feature outputs mirror Stage 7's layout under
    ``motif_pwm_enrichment/``. Existing outputs are not re-computed.
    """
    if not config.motif_pwm_enabled:
        print("[motif_pwm] disabled (set motif_pwm_enabled=True to run).")
        return

    if not _meme_available():
        raise RuntimeError(
            "MEME binary not found on PATH. Install with "
            "`conda install -c bioconda meme` before running this stage."
        )

    global_max = np.load(config.feature_max_path)
    num_features = len(global_max)

    out_dir = config.motif_pwm_enrichment_dir
    features_dir = config.features_dir

    n_analyzed = 0
    n_skipped = 0
    summary_features: Dict[str, Dict[str, Any]] = {}

    for feat_idx in tqdm(range(num_features), desc="[motif_pwm]"):
        feat_max = float(global_max[feat_idx])
        if feat_max == 0:
            n_skipped += 1
            continue

        out_path = out_dir / f"{feat_idx:04d}.json"
        if out_path.exists():
            try:
                with open(out_path) as f:
                    existing = json.load(f)
                if existing.get("motifs"):
                    m0 = existing["motifs"][0]
                    summary_features[str(feat_idx)] = {
                        "best_consensus": m0["consensus"],
                        "best_f1": m0["best_f1"],
                        "best_pr_auc": (m0.get("pr_auc") or {}).get("pr_auc"),
                        "e_value": m0["e_value"],
                        "n_motifs": len(existing["motifs"]),
                    }
                n_analyzed += 1
            except (json.JSONDecodeError, KeyError):
                pass
            continue

        feat_path = features_dir / f"{feat_idx:04d}.json"
        if not feat_path.exists():
            n_skipped += 1
            continue

        with open(feat_path) as f:
            feature_data = json.load(f)

        result = _analyze_feature_pwm(feature_data, feat_max, config)
        if result is None:
            n_skipped += 1
            continue

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        m0 = result["motifs"][0]
        summary_features[str(feat_idx)] = {
            "best_consensus": m0["consensus"],
            "best_f1": m0["best_f1"],
            "best_pr_auc": (m0.get("pr_auc") or {}).get("pr_auc"),
            "e_value": m0["e_value"],
            "n_motifs": len(result["motifs"]),
        }
        n_analyzed += 1

    summary = {
        "n_features_analyzed": n_analyzed,
        "n_features_skipped": n_skipped,
        "window_width": 2 * config.motif_pwm_window_half_w + 1,
        "meme_minw": config.motif_pwm_meme_minw,
        "meme_maxw": config.motif_pwm_meme_maxw,
        "features": summary_features,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[motif_pwm] Analyzed {n_analyzed} features, "
        f"skipped {n_skipped}. Output: {out_dir}/"
    )

    from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog

    wlog({
        "motif_pwm/analyzed": n_analyzed,
        "motif_pwm/skipped": n_skipped,
    })
