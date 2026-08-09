#!/usr/bin/env python3
"""Compute permutation-based null distributions for geometry-primary classification.

For each SAE feature, tests the null hypothesis that there is no association
between the per-residue activation pattern and each annotation structure
(sequence motifs, positional predicates, InterPro/CATH domain boundaries,
and local 3D geometry).

**Shuffle mechanism:**
Within-protein permutation of activation values.  For each protein
independently, randomly permute the per-residue activation values for the
feature under test.  This preserves (a) the marginal activation distribution
within each protein, (b) protein-level activation magnitude, (c) the
annotation structure (k-mer positions, domain boundaries, etc.), and
(d) protein boundaries.  It breaks only the residue-level association
between activation and annotation.

**P-value computation (one-sided, Phipson & Smyth 2010):**
    p = (1 + #{perm_score >= observed_score}) / (1 + K)

**Outputs:**
Per-feature checkpoint JSONs in ``{data_dir}/permutation_null/{fid:04d}.json``
containing observed scores, full null distributions (K values per metric),
raw p-values, and null summary statistics.  These are consumed by
``compute_geometry_primary.py`` which applies Benjamini-Hochberg FDR
correction across features.

Usage::

    python scripts/compute_permutation_null.py --data-dir /data/feature_data
    python scripts/compute_permutation_null.py --data-dir feature_data_cluster --n-permutations 10  # quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proteinlens.analysis.feature_pipeline.motif_enrichment import (
    _compute_best_motif_f1,
    _extract_kmers_with_activations,
)
from proteinlens.analysis.feature_pipeline.motif_pwm import (
    _compute_best_pwm_f1,
    _compute_pwm_pr_auc,
    _empirical_aa_background,
    _encode_sequence as _pwm_encode_sequence,
    _pwm_log_odds,
    _scan_pwm,
)
from proteinlens.analysis.feature_pipeline.position_enrichment import (
    _build_predicate_indices,
)
from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.interpro_api import _load_cached
from proteinlens.analysis.feature_pipeline.interpro_enrichment import (
    InterProDomain,
    _compute_protein_level_f1,
)

# RNG seed offset for the protein-level-InterPro across-protein shuffle.
# A *separate* Generator is used so the pre-existing within-protein rng
# produces a byte-identical draw sequence — this preserves the five existing
# null distributions exactly (regression-critical for FDR downstream).
_PROT_RNG_OFFSET = 10_000_000

# Defaults are passed explicitly from the CLI (whose default is read from
# PipelineConfig) so observed and null scores use the same threshold grid.
_INTERPRO_PROTEIN_MIN_PROTEINS = 3

logger = logging.getLogger(__name__)


# RNG offset for PWM null — uses its own stream so that enabling --include-pwm
# does not shift the six pre-existing null distributions (they share `rng`).
_PWM_RNG_OFFSET = 20_000_000

_PWM_UNIFORM_BG = np.full(20, 1.0 / 20, dtype=np.float64)


# RNG offset for the CATH protein-level null. Own stream so the six
# pre-existing null distributions stay byte-identical.
_CATH_PROT_RNG_OFFSET = 30_000_000

_CATH_PROTEIN_MIN_PROTEINS = 3
_CATH_LEVELS = ("C", "CA", "CAT", "CATH")


def _cath_label_at_level(cath_id: str, level: str) -> str:
    parts = cath_id.split(".")
    n = _CATH_LEVELS.index(level) + 1
    return ".".join(parts[:n])


# ── Data loading helpers ──────────────────────────────────────────────


def _load_observed_pwms(
    fid: int, pwm_dir: Path,
) -> tuple[list[dict] | None, str]:
    """Load the PWMs discovered by Stage 7b for a feature.

    Returns ``(motifs, background_model)`` where ``motifs`` is a list of
    motif dicts each with keys ``consensus``, ``width``, ``e_value``,
    ``best_f1``, ``pwm`` (np.ndarray, width x 20) — or ``None`` if no PWM
    output exists for this feature. ``background_model`` is the string
    read from the JSON's ``background_model`` field (defaults to
    ``"empirical"`` if absent for older outputs) so the null scoring uses
    the same log-odds background as Stage 7b did.

    Schema issues are logged but do not raise — the feature just becomes
    "no PWM null computed" rather than silently poisoning a run.
    """
    path = pwm_dir / f"{fid:04d}.json"
    if not path.exists():
        return None, "empirical"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("feature %d: PWM JSON exists but failed to parse (%s)", fid, e)
        return None, "empirical"
    motifs = data.get("motifs", [])
    if not motifs:
        return None, "empirical"
    bg_model = str(data.get("background_model", "empirical"))
    out = []
    for m in motifs:
        pwm_list = m.get("pwm")
        if pwm_list is None:
            continue
        pwm_arr = np.asarray(pwm_list, dtype=np.float64)
        if pwm_arr.ndim != 2 or pwm_arr.shape[1] != 20:
            logger.warning(
                "feature %d: skipping motif with bad PWM shape %s",
                fid, pwm_arr.shape,
            )
            continue
        out.append({
            "consensus": m.get("consensus", ""),
            "width": int(m.get("width", pwm_arr.shape[0])),
            "e_value": float(m.get("e_value", 1.0)),
            "best_f1": float(m.get("best_f1", 0.0)),
            "pwm": pwm_arr,
        })
    if not out:
        logger.warning("feature %d: PWM JSON present but no usable motifs", fid)
        return None, bg_model
    return out, bg_model


def _compute_pooled_pwm_scores(
    proteins: list[dict], pwms: list[dict], background_model: str,
) -> list[np.ndarray]:
    """Pre-scan each PWM over the pooled residue sequence once.

    Returns a list (same order as ``pwms``) of 1-D score arrays with the same
    length as the concatenated pooled activations. -inf at positions where the
    PWM window is out of bounds or overlaps non-standard residues.

    ``background_model`` must match the setting used by Stage 7b for this
    feature (read from the per-feature JSON) so that log-odds scores are
    reproduced exactly. Supported: ``"empirical"`` (pooled AA frequency) or
    ``"uniform"`` (1/20).
    """
    if background_model == "uniform":
        bg = _PWM_UNIFORM_BG
    else:
        # _empirical_aa_background expects (acc, seq, pra) tuples; re-pack.
        bg = _empirical_aa_background(
            [(p["accession"], p["sequence"], None) for p in proteins]
        )
    # Precompute log-odds once per PWM (shared across proteins).
    log_odds_per_pwm = [_pwm_log_odds(m["pwm"], bg) for m in pwms]
    scores_per_pwm: list[list[np.ndarray]] = [[] for _ in pwms]
    for p in proteins:
        enc = _pwm_encode_sequence(p["sequence"])
        for i, lo in enumerate(log_odds_per_pwm):
            scores_per_pwm[i].append(_scan_pwm(enc, lo))
    return [np.concatenate(s) for s in scores_per_pwm]


def _best_pwm_f1_across(
    pwm_scores: list[np.ndarray],
    activations: np.ndarray,
    feat_max: float,
    n_steps: int,
) -> float:
    """Max F1 across all PWMs given pre-computed per-residue scores."""
    best = 0.0
    for s in pwm_scores:
        r = _compute_best_pwm_f1(s, activations, feat_max, n_steps=n_steps)
        if r and r.get("best_f1", 0.0) > best:
            best = float(r["best_f1"])
    return best


def _best_pwm_pr_auc_across(
    pwm_scores: list[np.ndarray],
    activations: np.ndarray,
    act_quantile: float,
) -> float:
    """Max PR-AUC across all PWMs given pre-computed per-residue scores.

    Parallel to ``_best_pwm_f1_across`` but threshold-free along the predictor
    axis. Degenerate binarisations (the scoring function returns None) are
    treated as 0.0 so the returned float is always well-defined — this is
    required for the Phipson & Smyth p-value to be well-behaved when some
    permutations happen to produce a degenerate split.

    REVIEW: this helper is used for BOTH observed and null in this script,
    so the None->0.0 fallback is symmetric here. The per-feature Stage 7b
    JSON (motif_pwm.py) surfaces None instead; callers computing p-values
    must NOT mix observed values from the Stage 7b JSON with null
    distributions from this script.
    """
    best = 0.0
    for s in pwm_scores:
        r = _compute_pwm_pr_auc(s, activations, act_quantile=act_quantile)
        if r and r.get("pr_auc", 0.0) > best:
            best = float(r["pr_auc"])
    return best


def _pool_proteins(feature_data: dict) -> list[dict]:
    """Pool proteins from feature JSON, deduplicating by accession.

    Returns list of dicts with keys: accession, sequence, activations (1D array).
    Tracks protein boundaries for within-protein shuffling.
    """
    seen = set()
    proteins = []
    for source in [feature_data.get("top_sequences", []),
                   *[v for k, v in sorted(feature_data.get("activation_bins", {}).items())
                     if isinstance(v, list)]]:
        for p in source:
            acc = p.get("accession", "")
            if acc in seen or not acc:
                continue
            seen.add(acc)
            seq = p.get("sequence", "")
            acts = p.get("per_residue_activations")
            if seq and acts:
                proteins.append({
                    "accession": acc,
                    "sequence": seq,
                    "activations": np.array(acts, dtype=np.float64),
                })
    return proteins


def _load_interpro_labels(
    proteins: list[dict], interpro_file_set: set[str], interpro_cache_dir: Path
) -> np.ndarray | None:
    """Build pooled residue-level InterPro domain labels (1=inside, 0=outside).

    Returns 1D bool array of same length as pooled activations, or None if
    no InterPro data is available.

    Args:
        interpro_file_set: Pre-globbed set of accession stems in interpro_cache_dir.
    """
    labels = []
    any_domains = False
    for p in proteins:
        n = len(p["activations"])
        res_labels = np.zeros(n, dtype=bool)
        acc = p["accession"]
        if acc in interpro_file_set:
            cache_path = interpro_cache_dir / f"{acc}.json"
            try:
                raw = json.loads(cache_path.read_text())
                # Cache files use the ``{"accession": ..., "domains": [...]}``
                # layout written by ``interpro_api._save_cache``; earlier runs
                # used a bare list, so accept both shapes defensively.
                domains = raw.get("domains", []) if isinstance(raw, dict) else raw
                for d in domains:
                    start = max(0, d.get("start", 1) - 1)  # 1-based to 0-based
                    end = min(n, d.get("end", 0))  # 1-based inclusive
                    if start < end:
                        res_labels[start:end] = True
                        any_domains = True
            except (json.JSONDecodeError, OSError, AttributeError, TypeError):
                pass
        labels.append(res_labels)

    if not any_domains:
        return None
    return np.concatenate(labels)


def _load_cath_labels(
    proteins: list[dict], cath_file_set: set[str], cath_cache_dir: Path
) -> np.ndarray | None:
    """Build pooled residue-level CATH domain labels (1=inside, 0=outside).

    Takes max across all CATH hierarchy levels (any domain hit counts).

    Args:
        cath_file_set: Pre-globbed set of accession stems in cath_cache_dir.
    """
    labels = []
    any_domains = False
    for p in proteins:
        n = len(p["activations"])
        res_labels = np.zeros(n, dtype=bool)
        acc = p["accession"]
        if acc in cath_file_set:
            cache_path = cath_cache_dir / f"{acc}.json"
            try:
                hits = json.loads(cache_path.read_text())
                for h in hits:
                    qs = h.get("query_start")
                    qe = h.get("query_end")
                    if qs is None or qe is None:
                        continue
                    start = max(0, qs - 1)
                    end = min(n, qe)
                    if start < end:
                        res_labels[start:end] = True
                        any_domains = True
            except (json.JSONDecodeError, OSError):
                pass
        labels.append(res_labels)

    if not any_domains:
        return None
    return np.concatenate(labels)


def _load_interpro_protein_annotations(
    proteins: list[dict],
    interpro_file_set: set[str],
    interpro_cache_dir: Path,
) -> dict[str, list[InterProDomain]]:
    """Load full InterPro domain lists keyed by accession, for protein-level F1.

    Sibling of ``_load_interpro_labels`` (which builds residue-level boolean
    labels from the same JSON files). Both loaders read the same on-disk
    cache but project it differently; keeping them separate means the
    residue-level path is completely untouched and its null distribution
    remains bit-reproducible.

    Uses the pre-globbed ``interpro_file_set`` from the worker's shared
    state — no per-file ``.exists()`` / ``.glob()`` calls on cephfs.

    Returns an empty dict (not None) if no proteins have cached annotations,
    so callers can use a simple truthiness guard.
    """
    annotations: dict[str, list[InterProDomain]] = {}
    for p in proteins:
        acc = p["accession"]
        if acc not in interpro_file_set:
            continue
        try:
            annotations[acc] = _load_cached(interpro_cache_dir / f"{acc}.json")
        except (json.JSONDecodeError, OSError):
            # Silently drop corrupt cache entries — matches residue loader.
            continue
    return annotations


def _compute_interpro_protein_f1(
    accessions: list[str],
    per_protein_max: np.ndarray,
    protein_annotations: dict[str, list[InterProDomain]],
    feat_max: float,
    n_steps: int,
    min_proteins: int = _INTERPRO_PROTEIN_MIN_PROTEINS,
) -> float:
    """Thin wrapper around ``_compute_protein_level_f1`` returning top-1 F1.

    Centralising the call site guarantees the *observed* and *null* paths
    use bit-identical parameters — which matters because their equality is
    the definition of the null hypothesis here.
    """
    if not protein_annotations:
        return 0.0
    results = _compute_protein_level_f1(
        list(zip(accessions, per_protein_max.tolist())),
        protein_annotations,
        feat_max,
        n_threshold_steps=n_steps,
        min_proteins=min_proteins,
        top_n=1,
    )
    return float(results[0]["best_f1"]) if results else 0.0


def _load_cath_protein_annotations(
    proteins: list[dict],
    cath_file_set: set[str],
    cath_cache_dir: Path,
) -> dict[str, list[str]]:
    """Load CATH domain IDs per accession for the protein-level null.

    Sibling of ``_load_cath_labels`` (residue-level boolean labels from the
    same JSON files). Mirrors the InterPro split at
    ``_load_interpro_protein_annotations`` so the residue-level path — and
    its null distribution — stays bit-reproducible.
    """
    annotations: dict[str, list[str]] = {}
    for p in proteins:
        acc = p["accession"]
        if acc not in cath_file_set:
            continue
        try:
            hits = json.loads((cath_cache_dir / f"{acc}.json").read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cath_ids = [
            h["cath_id"] for h in hits
            if h.get("cath_id") and len(h["cath_id"].split(".")) >= 4
        ]
        if cath_ids:
            annotations[acc] = cath_ids
    return annotations


def _compute_cath_protein_f1(
    accessions: list[str],
    per_protein_max: np.ndarray,
    cath_annotations: dict[str, list[str]],
    feat_max: float,
    n_steps: int,
    min_proteins: int = _CATH_PROTEIN_MIN_PROTEINS,
) -> float:
    """Best protein-level F1 across all CATH labels at all hierarchy levels.

    Pooled scalar analog of ``_compute_interpro_protein_f1`` — returns one
    number so downstream FDR sees a single CATH-protein null per feature.
    Labels from all four CATH levels (C, CA, CAT, CATH) compete for the
    top spot; ties and filtering follow ``compute_cath_enrichment.py``
    (``min_proteins`` per label, linear threshold sweep over ``feat_max``).
    """
    if not cath_annotations or feat_max <= 0 or not accessions:
        return 0.0

    activations = np.asarray(per_protein_max, dtype=np.float64)

    label_accs: dict[tuple[str, str], set[str]] = {}
    for acc in accessions:
        for cath_id in cath_annotations.get(acc, []):
            for level in _CATH_LEVELS:
                key = (level, _cath_label_at_level(cath_id, level))
                label_accs.setdefault(key, set()).add(acc)

    eligible = [accs for accs in label_accs.values() if len(accs) >= min_proteins]
    if not eligible:
        return 0.0

    thresholds = np.linspace(0, feat_max, n_steps + 1)
    y_pred = activations[np.newaxis, :] > thresholds[:, np.newaxis]
    pred_sums = y_pred.sum(axis=1).astype(np.float64)
    y_pred_f = y_pred.astype(np.float64)

    y_true = np.array(
        [[1.0 if acc in accs else 0.0 for acc in accessions] for accs in eligible],
        dtype=np.float64,
    )
    true_sums = y_true.sum(axis=1)
    tp = y_true @ y_pred_f.T
    fp = pred_sums[np.newaxis, :] - tp
    fn = true_sums[:, np.newaxis] - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        pr_sum = precision + recall
        f1 = np.where(pr_sum > 0, 2.0 * precision * recall / pr_sum, 0.0)

    return float(f1.max()) if f1.size else 0.0


def _compute_domain_f1(
    all_activations: np.ndarray,
    domain_labels: np.ndarray,
    feat_max: float,
    n_steps: int = 50,
) -> float:
    """Compute best F1 across threshold sweep for domain boundary labels.

    Same logic as the motif/position F1 but with a single "annotation"
    (inside-domain vs outside-domain).

    Note: uses a linear threshold grid (n_steps points) rather than the
    pipeline's hybrid percentile+linear grid.  The permutation test is
    internally consistent (same grid for observed and null), so p-values
    are valid.  The absolute observed F1 may differ slightly from the
    pipeline's reported F1.
    """
    N = len(all_activations)
    if N == 0 or feat_max <= 0 or domain_labels.sum() == 0:
        return 0.0

    thresholds = np.linspace(0, feat_max, n_steps + 1)[1:]
    activated_matrix = all_activations[None, :] > thresholds[:, None]  # (T, N)
    n_activated = activated_matrix.sum(axis=1).astype(float)  # (T,)

    idx = np.where(domain_labels)[0]
    tp = activated_matrix[:, idx].sum(axis=1).astype(float)
    # Reuses the _compute_best_motif_f1 convention where the "annotation"
    # (domain positions) is treated as the prediction and activation as the
    # label.  The names below follow that convention for consistency:
    #   fn_or_fp_a = domain positions not activated
    #   fn_or_fp_b = activated positions not in domain
    # F1 is symmetric in these terms so the result is correct regardless.
    fn_or_fp_a = float(len(idx)) - tp
    fn_or_fp_b = n_activated - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        prec_or_rec_a = np.where(tp + fn_or_fp_a > 0, tp / (tp + fn_or_fp_a), 0.0)
        prec_or_rec_b = np.where(tp + fn_or_fp_b > 0, tp / (tp + fn_or_fp_b), 0.0)
        f1 = np.where(
            prec_or_rec_a + prec_or_rec_b > 0,
            2 * prec_or_rec_a * prec_or_rec_b / (prec_or_rec_a + prec_or_rec_b),
            0.0,
        )
    return float(f1.max()) if len(f1) > 0 else 0.0


# ── Geometry PR-AUC helpers ───────────────────────────────────────────


def _load_gbm_and_predict(
    fid: int,
    proteins: list[dict],
    data_dir: Path,
    shared: dict,
) -> tuple[np.ndarray, np.ndarray, float, list[tuple[int, int]]] | None:
    """Load saved GBM, compute geometry predictions for all interior residues.

    Returns (sae_activations, geom_predictions, threshold, geom_protein_boundaries)
    for all interior residues, or None if GBM not available.
    The geom_protein_boundaries are (start, end) pairs into the returned arrays,
    NOT into the pooled all_activations array.

    Args:
        shared: Dict with pre-loaded data to avoid per-feature I/O:
            - geom_profile_files: set of accession stems with geometry profiles
            - geom_profile_dir: Path to geometry_residue_profiles/
            - act_file_map: dict mapping accession -> Path for .npz activation files
            - gbm_files: set of fid stems with saved GBM models
    """
    from proteinlens.analysis.geometry.residue_features import (
        extract_local_feature_vector,
        select_features,
        ACTIVE_GEOM_NAMES,
    )

    gbm_files = shared.get("gbm_files", set())
    padded = f"{fid:04d}"
    gbm_dir = data_dir / "geometry_classifiers"
    gbm = None
    threshold = None
    half_w = 10  # default

    if padded in gbm_files:
        # Load pre-saved GBM
        try:
            gbm = joblib.load(gbm_dir / f"{padded}_gbm.pkl")
            meta = json.loads((gbm_dir / f"{padded}_meta.json").read_text())
            threshold = meta["threshold_sae"]
            half_w = meta["half_w"]
        except Exception:
            gbm = None

    if gbm is None:
        # Fallback: retrain GBM from geometry profiles + activations.
        # Must replicate the pipeline's protein selection exactly: top 500
        # proteins by activation from the full protein_feature_maxes memmap.
        from proteinlens.analysis.geometry.classifiers import (
            collect_node_fragments,
            train_motif_classifier,
        )

        geom_profile_dir = shared["geom_profile_dir"]
        geom_profile_files = shared.get("geom_profile_files", set())
        act_file_map = shared.get("act_file_map", {})

        # Load protein-level max activations (same as pipeline stage 6c)
        act_matrix_full = shared.get("act_matrix_full")
        row_to_acc = shared.get("row_to_acc")
        if act_matrix_full is None or row_to_acc is None:
            return None

        # Select top 500 proteins by activation for this feature (matching pipeline)
        node_col = act_matrix_full[:, fid]
        active_rows = np.where(node_col > 0)[0]
        if len(active_rows) > 500:
            top_idx = np.argsort(node_col[active_rows])[-500:]
            active_rows = active_rows[top_idx]

        # Build protein_data matching pipeline format
        protein_data = []
        for row_idx in active_rows:
            acc = row_to_acc.get(int(row_idx))
            if acc is None or acc not in geom_profile_files:
                continue
            act_path = act_file_map.get(acc)
            if act_path is None:
                continue
            try:
                act_mat = np.load(act_path)["activations"]
                gp = np.load(geom_profile_dir / f"{acc}.npz", allow_pickle=True)
                ca = np.array(gp["ca"])
                # No `if k in gp` guard — match pipeline which requires all 6 keys.
                # Missing keys raise KeyError, caught by outer try/except (protein skipped).
                profiles = {k: np.array(gp[k])[:len(ca)]
                            for k in ("curvature", "torsion", "planarity", "tangents", "helix_mask", "categories")}
                n = min(len(ca), act_mat.shape[0])
                if n < 20:
                    continue
                seq_arr = gp.get("sequence", np.array([""]))
                protein_data.append({
                    "accession": acc,
                    "act_matrix": act_mat[:n],
                    "ca": ca[:n],
                    "profiles": profiles,
                    "n_residues": n,
                    "sequence": str(seq_arr[0]) if len(seq_arr) > 0 else "",
                })
            except Exception:
                continue

        if len(protein_data) < 2:
            return None

        try:
            # Uses function defaults which match PipelineConfig defaults:
            #   act_quantile=0.80, max_fragments=100, bg_ratio=3, cv_folds=5
            # If the pipeline was run with non-default config, the retrained
            # GBM may differ.  This is acceptable since no saved GBM exists.
            frag_result = collect_node_fragments(protein_data, fid, half_w=half_w)
            activated = frag_result["activated"]
            background = frag_result["background"]
            threshold = frag_result["threshold"]

            if len(activated) < 20 or len(background) < 20:
                return None

            clf_result = train_motif_classifier(
                activated, background,
                feature_names=list(ACTIVE_GEOM_NAMES),
            )
            gbm = clf_result["tree"]
            if gbm is None:
                return None

            # Save retrained GBM for future runs
            gbm_dir.mkdir(parents=True, exist_ok=True)
            try:
                joblib.dump(gbm, gbm_dir / f"{padded}_gbm.pkl")
                meta = {
                    "feature_id": fid,
                    "threshold_sae": float(threshold),
                    "threshold_geom": float(clf_result["optimal_threshold"]),
                    "half_w": half_w,
                    "n_pos": clf_result["n_pos"],
                    "n_neg": clf_result["n_neg"],
                }
                (gbm_dir / f"{padded}_meta.json").write_text(json.dumps(meta))
            except Exception:
                pass
        except Exception:
            return None

    geom_profile_files = shared.get("geom_profile_files", set())
    geom_profile_dir = shared["geom_profile_dir"]
    act_file_map = shared.get("act_file_map", {})

    all_sae = []
    all_geom = []
    geom_protein_boundaries = []

    for p in proteins:
        acc = p["accession"]

        # Load per-residue activations from pre-mapped path
        act_path = act_file_map.get(acc)
        if act_path is None:
            continue

        try:
            act_data = np.load(act_path)["activations"]
        except Exception:
            continue

        n_residues = act_data.shape[0]
        if n_residues < 2 * half_w + 1:
            continue

        # Load geometry profiles (only if pre-globbed as available)
        if acc not in geom_profile_files:
            continue

        profiles = None
        ca = None
        gp_path = geom_profile_dir / f"{acc}.npz"
        try:
            gp = np.load(gp_path, allow_pickle=True)
            ca = np.array(gp["ca"])
            # Require all 6 profile keys (matching pipeline). Missing keys
            # raise KeyError, caught by except → protein skipped.
            profiles = {k: np.array(gp[k])[:len(ca)]
                        for k in ("curvature", "torsion", "planarity", "tangents", "helix_mask", "categories")}
        except Exception:
            continue

        n = min(len(ca), n_residues)
        sae_col = act_data[:n, fid]
        seq = p.get("sequence", "")

        # Extract features and predict for interior residues
        protein_start = len(all_sae)
        for pos in range(half_w, n - half_w):
            fv = extract_local_feature_vector(profiles, ca, pos, half_w, seq)
            if fv is None:
                continue
            fv_sel = select_features(fv).reshape(1, -1)
            try:
                prob = gbm.predict_proba(fv_sel)
                geom_prob = float(prob[0, 1]) if prob.shape[1] > 1 else float(prob[0, 0])
            except Exception:
                continue

            all_sae.append(float(sae_col[pos]))
            all_geom.append(geom_prob)

        protein_end = len(all_sae)
        if protein_end > protein_start:
            geom_protein_boundaries.append((protein_start, protein_end))

    if len(all_sae) < 20:
        return None

    return np.array(all_sae), np.array(all_geom), threshold, geom_protein_boundaries


# ── Within-protein shuffle ────────────────────────────────────────────


def _shuffle_within_proteins(
    all_activations: np.ndarray,
    protein_boundaries: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle activation values independently within each protein.

    Args:
        all_activations: Pooled 1D activation array.
        protein_boundaries: List of (start, end) index pairs into the pooled array.
        rng: NumPy random generator.

    Returns:
        Copy of all_activations with values shuffled within each protein segment.
    """
    shuffled = all_activations.copy()
    for start, end in protein_boundaries:
        rng.shuffle(shuffled[start:end])
    return shuffled


# ── Per-feature permutation worker ────────────────────────────────────


def process_feature(
    fid: int,
    data_dir: Path,
    n_permutations: int,
    seed: int,
    shared: dict | None = None,
    threshold_steps: int | None = None,
) -> dict[str, Any] | None:
    """Run permutation testing for a single feature across all 5 metrics.

    Args:
        shared: Pre-loaded shared data to avoid per-feature I/O on cephfs.
            Must contain: feat_max_arr, interpro_file_set, cath_file_set,
            geom_profile_files, geom_profile_dir, act_file_map, gbm_files.

    Returns the full result dict, or None if the feature cannot be processed.
    """
    from sklearn.metrics import average_precision_score

    if shared is None:
        shared = {}
    if threshold_steps is None:
        threshold_steps = PipelineConfig.__dataclass_fields__[
            "interpro_f1_threshold_steps"
        ].default

    feature_json_fids = shared.get("feature_json_fids", set())
    if feature_json_fids and fid not in feature_json_fids:
        return None

    feat_path = data_dir / "features" / f"{fid:04d}.json"
    try:
        feature_data = json.loads(feat_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    feat_max_arr = shared.get("feat_max_arr")
    if feat_max_arr is None:
        feat_max_arr = np.load(data_dir / "feature_max_activations.npy")
    feat_max = float(feat_max_arr[fid])
    if feat_max <= 0:
        return None

    # Pool proteins
    proteins = _pool_proteins(feature_data)
    if len(proteins) < 2:
        return None

    # Build pooled activation array and protein boundaries
    all_activations_list = []
    protein_boundaries = []
    seq_lengths = []
    offset = 0
    for p in proteins:
        n = len(p["activations"])
        all_activations_list.append(p["activations"])
        protein_boundaries.append((offset, offset + n))
        seq_lengths.append(n)
        offset += n

    all_activations = np.concatenate(all_activations_list)
    total_residues = len(all_activations)

    if total_residues < 10:
        return None

    # ── Build annotation structures (fixed, not shuffled) ──

    # 1. K-mer indices and k-mer-filtered activation array
    #    The pipeline builds both k-mer indices and activations from the same
    #    filtered extraction (skipping non-standard AAs and boundary residues),
    #    so we must do the same here for consistency.
    k = shared.get("motif_k", 3)
    all_kmers: list[str] = []
    motif_acts: list[float] = []
    motif_protein_boundaries: list[tuple[int, int]] = []
    motif_offset = 0
    for p in proteins:
        pairs = _extract_kmers_with_activations(p["sequence"], p["activations"].tolist(), k)
        protein_start = motif_offset
        for kmer, act in pairs:
            all_kmers.append(kmer)
            motif_acts.append(act)
            motif_offset += 1
        if motif_offset > protein_start:
            motif_protein_boundaries.append((protein_start, motif_offset))

    motif_activations = np.array(motif_acts, dtype=np.float64)

    kmer_indices: dict[str, list[int]] = {}
    for i, kmer in enumerate(all_kmers):
        kmer_indices.setdefault(kmer, []).append(i)
    kmer_idx_arrays = {km: np.array(idxs) for km, idxs in kmer_indices.items()}

    # 2. Position predicate indices
    predicate_indices = _build_predicate_indices(seq_lengths, total_residues)

    # 3. InterPro domain labels (using pre-globbed file set)
    interpro_cache_dir = data_dir / "interpro_cache"
    interpro_file_set = shared.get("interpro_file_set", set())
    interpro_labels = _load_interpro_labels(proteins, interpro_file_set, interpro_cache_dir)

    # 4. CATH domain labels (using pre-globbed file set)
    cath_cache_dir = data_dir / "cath_enrichment" / "cache"
    cath_file_set = shared.get("cath_file_set", set())
    cath_labels = _load_cath_labels(proteins, cath_file_set, cath_cache_dir)

    # 4b. InterPro protein-level annotations (for the protein-level null).
    # Reuses the same pre-globbed interpro_file_set — no extra cephfs I/O.
    protein_annotations = _load_interpro_protein_annotations(
        proteins, interpro_file_set, interpro_cache_dir,
    )
    # Per-protein max activation is derived from the SAME per-residue arrays
    # already loaded for the other metrics; no additional data source.
    accessions = [p["accession"] for p in proteins]
    per_protein_max = np.array(
        [float(p["activations"].max()) for p in proteins], dtype=np.float64,
    )

    # 4c. CATH protein-level annotations. Same shuffle unit as 4b (across
    # proteins), but the F1 is pooled across all four CATH levels.
    cath_protein_annotations = _load_cath_protein_annotations(
        proteins, cath_file_set, cath_cache_dir,
    )

    # 5. Geometry PR-AUC (load GBM, get predictions; uses pre-globbed file sets)
    geom_result = _load_gbm_and_predict(fid, proteins, data_dir, shared)

    # 6. PWM motifs (optional; only if --include-pwm and Stage 7b output exists)
    pwm_motifs: list[dict] | None = None
    pwm_scores_pooled: list[np.ndarray] = []
    pwm_bg_model: str = "empirical"
    if shared.get("include_pwm", False):
        pwm_dir = data_dir / "motif_pwm_enrichment"
        pwm_motifs, pwm_bg_model = _load_observed_pwms(fid, pwm_dir)
        if pwm_motifs:
            pwm_scores_pooled = _compute_pooled_pwm_scores(
                proteins, pwm_motifs, pwm_bg_model,
            )

    # ── Compute observed scores ──

    n_steps = threshold_steps
    min_count = 5

    # Motif F1 (uses k-mer-filtered activation array, not full pooled array)
    motif_results = _compute_best_motif_f1(
        kmer_idx_arrays, motif_activations, feat_max,
        n_steps=n_steps, min_count=min_count, top_n=1,
    )
    motif_f1_obs = motif_results[0]["best_f1"] if motif_results else 0.0

    # Position F1
    position_results = _compute_best_motif_f1(
        predicate_indices, all_activations, feat_max,
        n_steps=n_steps, min_count=1, top_n=1,
    )
    position_f1_obs = position_results[0]["best_f1"] if position_results else 0.0

    # InterPro residue F1
    interpro_f1_obs = 0.0
    if interpro_labels is not None:
        interpro_f1_obs = _compute_domain_f1(all_activations, interpro_labels, feat_max, n_steps)

    # CATH residue F1
    cath_f1_obs = 0.0
    if cath_labels is not None:
        cath_f1_obs = _compute_domain_f1(all_activations, cath_labels, feat_max, n_steps)

    # Geometry PR-AUC
    geom_prauc_obs = 0.0
    geom_boundaries = []
    if geom_result is not None:
        sae_arr, geom_preds, geom_threshold, geom_boundaries = geom_result
        sae_binary = (sae_arr >= geom_threshold).astype(int)
        if sae_binary.sum() > 0 and sae_binary.sum() < len(sae_binary):
            geom_prauc_obs = float(average_precision_score(sae_binary, geom_preds))

    # InterPro protein-level F1 (best across eligible codes, top-1).
    # Observed uses the *un*-shuffled per-protein max vector.
    interpro_protein_f1_obs = _compute_interpro_protein_f1(
        accessions, per_protein_max, protein_annotations, feat_max, n_steps,
    )

    cath_protein_f1_obs = _compute_cath_protein_f1(
        accessions, per_protein_max, cath_protein_annotations, feat_max, n_steps,
    )

    # PWM motif F1 (observed). Recomputed from the pooled per-residue scores
    # rather than read from Stage 7b JSON so the scoring matches the null
    # bit-for-bit (Stage 7b rounds the F1 to 4 dp and uses the same sweep).
    pwm_f1_obs = 0.0
    pwm_pr_auc_obs = 0.0
    pwm_act_quantile = 0.0  # only meaningful when the PWM branch runs
    if pwm_motifs and pwm_scores_pooled:
        # Single source of truth: shared["pwm_act_quantile"] is set from
        # PipelineConfig.motif_pwm_act_quantile at CLI time. Raise loudly
        # if absent — defaulting here would silently diverge from Stage 7b
        # and from Stage 6c's geometry_act_quantile.
        if "pwm_act_quantile" not in shared:
            raise KeyError(
                "shared['pwm_act_quantile'] missing — must be set from "
                "PipelineConfig.motif_pwm_act_quantile for observed/null parity."
            )
        pwm_act_quantile = float(shared["pwm_act_quantile"])
        pwm_f1_obs = _best_pwm_f1_across(
            pwm_scores_pooled, all_activations, feat_max, n_steps,
        )
        # PR-AUC mirrors Stage 6c: threshold-free along the predictor axis,
        # truth binarised at a fixed quantile. Magnitudes on the same footing
        # as geometry_prauc.
        pwm_pr_auc_obs = _best_pwm_pr_auc_across(
            pwm_scores_pooled, all_activations, act_quantile=pwm_act_quantile,
        )

    # ── Permutation loop ──

    rng = np.random.default_rng(seed + fid)
    # Independent RNG for the across-protein shuffle. Must not share state
    # with `rng` above or the five pre-existing null distributions would
    # shift — they are consumed downstream for BH-FDR and are part of
    # published results.
    rng_protein = np.random.default_rng(seed + fid + _PROT_RNG_OFFSET)

    null_motif = np.zeros(n_permutations)
    null_position = np.zeros(n_permutations)
    null_interpro = np.zeros(n_permutations)
    null_cath = np.zeros(n_permutations)
    null_geom = np.zeros(n_permutations)
    null_interpro_protein = np.zeros(n_permutations)
    null_cath_protein = np.zeros(n_permutations)
    null_pwm = np.zeros(n_permutations)
    null_pwm_pr_auc = np.zeros(n_permutations)

    # Independent RNG for PWM shuffle so enabling --include-pwm does NOT
    # perturb the six pre-existing null distributions (which share `rng`).
    rng_pwm = np.random.default_rng(seed + fid + _PWM_RNG_OFFSET)

    # Independent RNG for the CATH protein-level across-protein shuffle.
    # Must not share state with any other rng — preserves byte-identical
    # output for the pre-existing null distributions.
    rng_cath_protein = np.random.default_rng(seed + fid + _CATH_PROT_RNG_OFFSET)

    for k_perm in range(n_permutations):
        # Shuffle full pooled activations within each protein (for position/InterPro/CATH)
        shuffled = _shuffle_within_proteins(all_activations, protein_boundaries, rng)

        # Shuffle k-mer-filtered activations within each protein (for motif F1)
        shuffled_motif = _shuffle_within_proteins(motif_activations, motif_protein_boundaries, rng)

        # Motif F1 with shuffled k-mer-filtered activations
        perm_motif = _compute_best_motif_f1(
            kmer_idx_arrays, shuffled_motif, feat_max,
            n_steps=n_steps, min_count=min_count, top_n=1,
        )
        null_motif[k_perm] = perm_motif[0]["best_f1"] if perm_motif else 0.0

        # Position F1 with shuffled activations
        perm_pos = _compute_best_motif_f1(
            predicate_indices, shuffled, feat_max,
            n_steps=n_steps, min_count=1, top_n=1,
        )
        null_position[k_perm] = perm_pos[0]["best_f1"] if perm_pos else 0.0

        # InterPro residue F1 with shuffled activations
        if interpro_labels is not None:
            null_interpro[k_perm] = _compute_domain_f1(shuffled, interpro_labels, feat_max, n_steps)

        # CATH residue F1 with shuffled activations
        if cath_labels is not None:
            null_cath[k_perm] = _compute_domain_f1(shuffled, cath_labels, feat_max, n_steps)

        # Geometry PR-AUC with shuffled labels
        if geom_result is not None:
            # Shuffle the SAE activations within each protein segment
            # using geometry-specific boundaries (interior residues only)
            shuffled_sae = _shuffle_within_proteins(sae_arr, geom_boundaries, rng)
            shuffled_binary = (shuffled_sae >= geom_threshold).astype(int)
            if shuffled_binary.sum() > 0 and shuffled_binary.sum() < len(shuffled_binary):
                null_geom[k_perm] = float(average_precision_score(shuffled_binary, geom_preds))

        # InterPro protein-level F1 with activations shuffled ACROSS proteins.
        # This is a different shuffle unit from the residue-level ones above:
        # breaks which-protein-activates vs which-protein-has-annotation while
        # preserving the per-protein activation magnitude distribution and
        # the accession->annotation mapping. Uses its own RNG stream
        # (rng_protein) so the existing null arrays above are unaffected.
        # Placed last in the loop body as a further safety guard against
        # any future bug here leaking into earlier null values.
        if protein_annotations:
            shuffled_max = per_protein_max.copy()
            rng_protein.shuffle(shuffled_max)
            null_interpro_protein[k_perm] = _compute_interpro_protein_f1(
                accessions, shuffled_max, protein_annotations, feat_max, n_steps,
            )

        # CATH protein-level: same across-protein shuffle unit as InterPro
        # protein-level, separate RNG stream so adding this null does not
        # perturb any pre-existing distribution.
        if cath_protein_annotations:
            shuffled_max_cath = per_protein_max.copy()
            rng_cath_protein.shuffle(shuffled_max_cath)
            null_cath_protein[k_perm] = _compute_cath_protein_f1(
                accessions, shuffled_max_cath, cath_protein_annotations,
                feat_max, n_steps,
            )

        # PWM motif F1 with shuffled activations. PWMs + per-residue scores
        # stay fixed; only activations are permuted within protein. Uses its
        # own RNG stream (rng_pwm) — see _PWM_RNG_OFFSET rationale above.
        if pwm_motifs and pwm_scores_pooled:
            shuffled_pwm_acts = _shuffle_within_proteins(
                all_activations, protein_boundaries, rng_pwm,
            )
            null_pwm[k_perm] = _best_pwm_f1_across(
                pwm_scores_pooled, shuffled_pwm_acts, feat_max, n_steps,
            )
            # Reuse the SAME shuffle for PR-AUC — no extra rng draws, so the
            # pwm_f1 null is byte-identical whether or not PR-AUC is enabled.
            # This is covered by a regression test.
            null_pwm_pr_auc[k_perm] = _best_pwm_pr_auc_across(
                pwm_scores_pooled, shuffled_pwm_acts,
                act_quantile=pwm_act_quantile,
            )

    # ── Compute p-values (Phipson & Smyth 2010) ──

    def _pvalue(observed: float, null_dist: np.ndarray) -> float:
        return float((1 + np.sum(null_dist >= observed)) / (1 + len(null_dist)))

    def _null_summary(null_dist: np.ndarray) -> dict:
        return {
            "mean": round(float(null_dist.mean()), 6),
            "std": round(float(null_dist.std()), 6),
            "p95": round(float(np.percentile(null_dist, 95)), 6),
            "p99": round(float(np.percentile(null_dist, 99)), 6),
        }

    result = {
        "feature_id": fid,
        "n_permutations": n_permutations,
        "threshold_steps": threshold_steps,
        "scoring_provenance_version": 1,
        "seed": seed,
        "n_proteins": len(proteins),
        "n_residues": total_residues,
        "observed": {
            "motif_f1": round(motif_f1_obs, 6),
            "position_f1": round(position_f1_obs, 6),
            "interpro_res_f1": round(interpro_f1_obs, 6),
            "cath_res_f1": round(cath_f1_obs, 6),
            "geometry_prauc": round(geom_prauc_obs, 6),
            # New protein-level InterPro entry — appended last so existing
            # keys keep their order/position and diff cleanly.
            "interpro_protein_f1": round(interpro_protein_f1_obs, 6),
            "cath_protein_f1": round(cath_protein_f1_obs, 6),
        },
        "null_distributions": {
            "motif_f1": [round(float(v), 6) for v in null_motif],
            "position_f1": [round(float(v), 6) for v in null_position],
            "interpro_res_f1": [round(float(v), 6) for v in null_interpro],
            "cath_res_f1": [round(float(v), 6) for v in null_cath],
            "geometry_prauc": [round(float(v), 6) for v in null_geom],
            "interpro_protein_f1": [round(float(v), 6) for v in null_interpro_protein],
            "cath_protein_f1": [round(float(v), 6) for v in null_cath_protein],
        },
        "p_values": {
            "motif_f1": round(_pvalue(motif_f1_obs, null_motif), 6),
            "position_f1": round(_pvalue(position_f1_obs, null_position), 6),
            "interpro_res_f1": round(_pvalue(interpro_f1_obs, null_interpro), 6),
            "cath_res_f1": round(_pvalue(cath_f1_obs, null_cath), 6),
            "geometry_prauc": round(_pvalue(geom_prauc_obs, null_geom), 6),
            "interpro_protein_f1": round(
                _pvalue(interpro_protein_f1_obs, null_interpro_protein), 6,
            ),
            "cath_protein_f1": round(
                _pvalue(cath_protein_f1_obs, null_cath_protein), 6,
            ),
        },
        "null_summary": {
            "motif_f1": _null_summary(null_motif),
            "position_f1": _null_summary(null_position),
            "interpro_res_f1": _null_summary(null_interpro),
            "cath_res_f1": _null_summary(null_cath),
            "geometry_prauc": _null_summary(null_geom),
            "interpro_protein_f1": _null_summary(null_interpro_protein),
            "cath_protein_f1": _null_summary(null_cath_protein),
        },
    }

    # Append PWM entries only when --include-pwm is active AND Stage 7b output
    # exists for this feature. Omitting keys (vs. writing zeros) lets downstream
    # consumers reliably detect "PWM null was not computed for this feature".
    if pwm_motifs and pwm_scores_pooled:
        result["observed"]["pwm_f1"] = round(pwm_f1_obs, 6)
        result["null_distributions"]["pwm_f1"] = [
            round(float(v), 6) for v in null_pwm
        ]
        result["p_values"]["pwm_f1"] = round(_pvalue(pwm_f1_obs, null_pwm), 6)
        result["null_summary"]["pwm_f1"] = _null_summary(null_pwm)

        # PR-AUC: structurally parallel to geometry_prauc. Same shuffle
        # (rng_pwm) as pwm_f1 above, so pwm_f1's null is unaffected.
        result["observed"]["pwm_pr_auc"] = round(pwm_pr_auc_obs, 6)
        result["null_distributions"]["pwm_pr_auc"] = [
            round(float(v), 6) for v in null_pwm_pr_auc
        ]
        result["p_values"]["pwm_pr_auc"] = round(
            _pvalue(pwm_pr_auc_obs, null_pwm_pr_auc), 6,
        )
        result["null_summary"]["pwm_pr_auc"] = _null_summary(null_pwm_pr_auc)
        result["pwm_act_quantile"] = pwm_act_quantile

        result["n_pwms"] = len(pwm_motifs)
        # Self-documenting fields so downstream consumers know how this
        # p-value was computed. Option A: PWMs fixed, activations permuted
        # within-protein. Tests "does this PWM's score predict activation
        # better than chance?", NOT "did MEME find a real motif?".
        result["pwm_null_type"] = "option_A_fixed_pwm_permuted_activations"
        result["pwm_background_model"] = pwm_bg_model
    return result


# ── Module-level worker (must be picklable for ProcessPoolExecutor) ───

_worker_state: dict[str, Any] = {}


def _worker(fid: int) -> tuple[int, str, dict | None]:
    """Process a single feature. Uses module-level _worker_state for shared data.

    Returns (fid, status, summary) where summary is a lightweight dict with
    only p-values and metadata (not the full null distributions) to minimize
    pickle transfer from worker to parent.
    """
    s = _worker_state
    try:
        result = process_feature(
            fid,
            s["data_dir"],
            s["n_permutations"],
            s["seed"],
            s["shared"],
            threshold_steps=s["threshold_steps"],
        )
        if result is None:
            return fid, "skipped", None
        out_path = s["perm_dir"] / f"{fid:04d}.json"
        # Atomic write: tmp + rename. If the worker is SIGKILL'd (OOM, eviction)
        # mid-write, the visible file is either fully-written or absent —
        # never a 0-byte stub that the resume glob would mistake for "done".
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(result, indent=2))
        tmp_path.rename(out_path)
        # Return lightweight summary (p-values + metadata) instead of full
        # result with null distributions to reduce pickle overhead.
        summary = {
            "p_values": result["p_values"],
            "observed": result["observed"],
            "n_proteins": result["n_proteins"],
            "n_residues": result["n_residues"],
        }
        return fid, "done", summary
    except Exception as e:
        logger.error("Feature %d failed: %s", fid, e)
        return fid, f"error: {e}", None


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("feature_data_cluster"),
        help="Pipeline output directory",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=100,
        help="Number of permutations per feature (default: 100)",
    )
    parser.add_argument(
        "--threshold-steps",
        type=int,
        default=PipelineConfig.__dataclass_fields__[
            "interpro_f1_threshold_steps"
        ].default,
        help=(
            "Threshold grid used for every F1 observed/null score. The default "
            "is read from PipelineConfig and is recorded in every output."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Null output directory (default: DATA_DIR/permutation_null)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (per-feature seed = base + feature_id)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Log progress and summary statistics to Weights & Biases",
    )
    parser.add_argument(
        "--include-pwm", action="store_true",
        help=(
            "Also compute a null distribution for the PWM motif F1 from "
            "Stage 7b (requires motif_pwm_enrichment/ outputs). PWMs are "
            "held fixed; only activations are permuted within protein. "
            "Emits both pwm_f1 (legacy 2-D sweep) and pwm_pr_auc (threshold-"
            "free, parallel to geometry_prauc)."
        ),
    )
    # Default pulled from PipelineConfig so the three touchpoints
    # (PipelineConfig, this CLI, shared dict) cannot drift apart.
    parser.add_argument(
        "--pwm-act-quantile", type=float,
        default=PipelineConfig.__dataclass_fields__["motif_pwm_act_quantile"].default,
        help=(
            "Activation quantile used to binarise truth for pwm_pr_auc. "
            "Default is read from PipelineConfig.motif_pwm_act_quantile "
            "to keep Stage 7b observed scores and the null in lock-step. "
            "DEVIATING FROM geometry_act_quantile BREAKS DIRECT "
            "COMPARABILITY with geometry_prauc — change both or neither."
        ),
    )
    args = parser.parse_args()

    # ── Optional W&B init ──
    wb_run = None
    if args.wandb:
        import wandb
        wb_run = wandb.init(
            project="proteinlens-pipeline",
            name="permutation-null",
            tags=["permutation", "null-distribution"],
            config={
                "n_permutations": args.n_permutations,
                "seed": args.seed,
                "workers": args.workers,
                "data_dir": str(args.data_dir),
            },
        )

    data_dir = args.data_dir
    perm_dir = args.output_dir or (data_dir / "permutation_null")
    perm_dir.mkdir(parents=True, exist_ok=True)

    # Discover features
    feat_max_arr = np.load(data_dir / "feature_max_activations.npy")
    n_features = len(feat_max_arr)
    all_fids = [i for i in range(n_features) if feat_max_arr[i] > 0]

    # Check for completed features (resume) — single glob.
    # Skip 0-byte stubs left by SIGKILL'd workers in older runs (pre atomic
    # write fix). Without this, those stubs are treated as "done" and the
    # corresponding features are never recomputed; downstream consumers then
    # JSONDecodeError on empty input.
    done_fids = set()
    stale_stubs = 0
    incompatible_files: list[str] = []
    for fpath in perm_dir.glob("*.json"):
        try:
            if fpath.stat().st_size == 0:
                stale_stubs += 1
                continue
            existing = json.loads(fpath.read_text())
            if (
                existing.get("threshold_steps") != args.threshold_steps
                or existing.get("n_permutations") != args.n_permutations
            ):
                incompatible_files.append(fpath.name)
                continue
            fid = int(fpath.stem)
            done_fids.add(fid)
        except (ValueError, OSError, json.JSONDecodeError):
            pass
    if incompatible_files:
        raise SystemExit(
            f"{len(incompatible_files)} existing null files have missing or "
            "incompatible threshold_steps/n_permutations metadata. Refusing "
            "to mix snapshots; choose a fresh --output-dir."
        )
    if stale_stubs:
        print(f"  Skipping {stale_stubs} zero-byte stub(s) — will recompute")

    todo = [fid for fid in all_fids if fid not in done_fids]

    # ── Pre-glob all directories once to avoid per-feature stat() calls ──
    print("Pre-globbing directories for cephfs I/O optimization...")

    # InterPro cache: glob once, build accession set
    interpro_cache_dir = data_dir / "interpro_cache"
    interpro_file_set = set()
    if interpro_cache_dir.is_dir():
        interpro_file_set = {p.stem for p in interpro_cache_dir.glob("*.json")}
    print(f"  InterPro cache: {len(interpro_file_set)} files")

    # CATH cache: glob once
    cath_cache_dir = data_dir / "cath_enrichment" / "cache"
    cath_file_set = set()
    if cath_cache_dir.is_dir():
        cath_file_set = {p.stem for p in cath_cache_dir.glob("*.json")}
    print(f"  CATH cache: {len(cath_file_set)} files")

    # Geometry profiles: glob once
    geom_profile_dir = data_dir / "geometry_residue_profiles"
    geom_profile_files = set()
    if geom_profile_dir.is_dir():
        geom_profile_files = {p.stem for p in geom_profile_dir.glob("*.npz")}
    print(f"  Geometry profiles: {len(geom_profile_files)} files")

    # Residue activations: glob both dirs once, build accession -> path map
    act_file_map: dict[str, Path] = {}
    for act_dir_name in ("residue_activations", "interpro_residue_activations"):
        act_dir = data_dir / act_dir_name
        if act_dir.is_dir():
            for p in act_dir.glob("*.npz"):
                if p.stem not in act_file_map:
                    act_file_map[p.stem] = p
    print(f"  Residue activations: {len(act_file_map)} files")

    # GBM classifiers: glob once
    gbm_dir = data_dir / "geometry_classifiers"
    gbm_files = set()
    if gbm_dir.is_dir():
        gbm_files = {p.stem.replace("_gbm", "") for p in gbm_dir.glob("*_gbm.pkl")}
    print(f"  GBM classifiers: {len(gbm_files)} files")

    # K-mer length: read once from any motif enrichment JSON (pipeline config constant)
    motif_k = 3
    motif_summary = data_dir / "motif_enrichment" / "summary.json"
    if motif_summary.exists():
        try:
            motif_k = json.loads(motif_summary.read_text()).get("k", 3)
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    else:
        # Fallback: read from first available per-feature motif JSON
        motif_dir = data_dir / "motif_enrichment"
        if motif_dir.is_dir():
            for mf in motif_dir.iterdir():
                if mf.suffix == ".json" and mf.name != "summary.json":
                    try:
                        motif_k = json.loads(mf.read_text()).get("k", 3)
                    except (json.JSONDecodeError, OSError, KeyError):
                        pass
                    break
    print(f"  Motif k-mer length: {motif_k}")

    # Protein-level activation matrix + accession index (for GBM retrain fallback)
    # Matches the pipeline's protein selection: top 500 by activation from full memmap.
    act_matrix_full = None
    row_to_acc = None
    pipeline_state_path = data_dir / "pipeline_state.json"
    protein_maxes_path = data_dir / "protein_feature_maxes.npy"
    if pipeline_state_path.exists() and protein_maxes_path.exists():
        state = json.loads(pipeline_state_path.read_text())
        acc_to_idx = state.get("accession_index", {})
        # Filter to proteins with BOTH geometry profiles AND activation files,
        # matching the pipeline's `available` dict so top-500 selection is identical.
        row_to_acc = {v: k for k, v in acc_to_idx.items()
                      if k in geom_profile_files and k in act_file_map}
        n_proteins = len(acc_to_idx)
        act_matrix_full = np.memmap(
            protein_maxes_path, dtype="float32", mode="r",
            shape=(n_proteins, len(feat_max_arr)),
        )
        print(f"  Protein maxes memmap: {n_proteins} x {len(feat_max_arr)}")
    else:
        print("  WARNING: protein_feature_maxes.npy or pipeline_state.json not found")
        print("           GBM retrain fallback will be disabled for geometry PR-AUC")

    # Feature JSONs: glob once to avoid per-feature exists()
    features_dir = data_dir / "features"
    feature_json_fids = set()
    if features_dir.is_dir():
        for p in features_dir.glob("*.json"):
            try:
                feature_json_fids.add(int(p.stem))
            except ValueError:
                pass
    print(f"  Feature JSONs: {len(feature_json_fids)} files")

    # Shared data dict — passed to all workers to avoid per-feature I/O
    shared = {
        "feat_max_arr": feat_max_arr,
        "interpro_file_set": interpro_file_set,
        "cath_file_set": cath_file_set,
        "geom_profile_files": geom_profile_files,
        "geom_profile_dir": geom_profile_dir,
        "act_file_map": act_file_map,
        "gbm_files": gbm_files,
        "motif_k": motif_k,
        "feature_json_fids": feature_json_fids,
        "act_matrix_full": act_matrix_full,
        "row_to_acc": row_to_acc,
        "include_pwm": args.include_pwm,
        "pwm_act_quantile": args.pwm_act_quantile,
    }

    print("=" * 60)
    print("Permutation Null Distribution Computation")
    print("=" * 60)
    print(f"  Data dir:        {data_dir}")
    print(f"  N permutations:  {args.n_permutations}")
    print(f"  Threshold steps: {args.threshold_steps}")
    print(f"  Seed:            {args.seed}")
    print(f"  Workers:         {args.workers}")
    print(f"  Total features:  {len(all_fids)}")
    print(f"  Already done:    {len(done_fids)}")
    print(f"  To process:      {len(todo)}")
    print("=" * 60)

    if not todo:
        print("All features already processed.")
        return

    t0 = time.time()

    # Set module-level state for picklable worker function
    global _worker_state
    _worker_state.update({
        "data_dir": data_dir,
        "perm_dir": perm_dir,
        "n_permutations": args.n_permutations,
        "threshold_steps": args.threshold_steps,
        "seed": args.seed,
        "shared": shared,
    })

    n_done = 0
    n_skipped = 0
    n_error = 0
    # Collect p-values for final summary
    all_pvalues: dict[str, list[float]] = {
        "motif_f1": [], "position_f1": [], "interpro_res_f1": [],
        "cath_res_f1": [], "geometry_prauc": [],
    }
    if args.include_pwm:
        all_pvalues["pwm_f1"] = []
        all_pvalues["pwm_pr_auc"] = []

    def _handle_result(fid_result: int, status: str, result: dict | None) -> None:
        nonlocal n_done, n_skipped, n_error
        if status == "done":
            n_done += 1
            if result and wb_run:
                pv = result.get("p_values", {})
                wb_run.log({
                    "progress/completed": n_done + len(done_fids),
                    "progress/total": len(all_fids),
                    "progress/pct": 100 * (n_done + len(done_fids)) / max(len(all_fids), 1),
                    "feature/id": fid_result,
                    "feature/n_proteins": result.get("n_proteins", 0),
                    "feature/n_residues": result.get("n_residues", 0),
                    "pvalue/motif_f1": pv.get("motif_f1", 1.0),
                    "pvalue/position_f1": pv.get("position_f1", 1.0),
                    "pvalue/interpro_res_f1": pv.get("interpro_res_f1", 1.0),
                    "pvalue/cath_res_f1": pv.get("cath_res_f1", 1.0),
                    "pvalue/geometry_prauc": pv.get("geometry_prauc", 1.0),
                    "observed/geometry_prauc": result.get("observed", {}).get("geometry_prauc", 0),
                    "observed/motif_f1": result.get("observed", {}).get("motif_f1", 0),
                })
            if result:
                for m in all_pvalues:
                    pv = result.get("p_values", {}).get(m)
                    if pv is not None:
                        all_pvalues[m].append(pv)
        elif status == "skipped":
            n_skipped += 1
        else:
            n_error += 1

    if args.workers <= 1:
        for fid in tqdm(todo, desc="Permutation testing"):
            fid_result, status, result = _worker(fid)
            _handle_result(fid_result, status, result)
    else:
        # Use fork context explicitly — _worker reads module-level _worker_state
        # which is set in main() and must be visible to children via COW.
        # spawn re-imports the module and sees empty _worker_state.
        fork_ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=fork_ctx) as executor:
            futures = {executor.submit(_worker, fid): fid for fid in todo}
            pbar = tqdm(total=len(todo), desc="Permutation testing")
            for future in as_completed(futures):
                fid_result, status, result = future.result()
                _handle_result(fid_result, status, result)
                pbar.update(1)
            pbar.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s: {n_done} completed, {n_skipped} skipped, {n_error} errors")

    # ── Final W&B summary ──
    if wb_run:
        summary = {
            "total_features": len(all_fids),
            "completed": n_done + len(done_fids),
            "skipped": n_skipped,
            "errors": n_error,
            "resumed_from": len(done_fids),
            "elapsed_seconds": elapsed,
        }
        for m, pvals in all_pvalues.items():
            if pvals:
                arr = np.array(pvals)
                summary[f"n_significant_{m}"] = int((arr < 0.05).sum())
                summary[f"pct_significant_{m}"] = round(100 * (arr < 0.05).mean(), 1)
                summary[f"median_pvalue_{m}"] = round(float(np.median(arr)), 4)
                # Log p-value histogram
                import wandb as _wb
                wb_run.log({f"hist/{m}_pvalues": _wb.Histogram(arr, num_bins=20)})
        wb_run.summary.update(summary)
        wb_run.finish()


if __name__ == "__main__":
    main()
