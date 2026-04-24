"""Refit-GBM permutation null for geometry PR-AUC.

Rationale
---------
The existing geometry PR-AUC null in ``scripts/compute_permutation_null.py``
holds the GBM fixed (trained once on real labels) while shuffling labels. The
observed PR-AUC is therefore inflated by the GBM's training-set fit, while
the null collapses toward the base rate — the observed-null gap is dominated
by memorisation rather than generalisation.

This module implements a **refit null**: for each permutation the GBM is
retrained from scratch on the shuffled labels (sampled activated / background
fragments re-derived from the shuffled activations). Both observed and null
then include the same "memorisation of training fragments" component, which
cancels when computing the p-value — the remaining gap reflects the GBM's
true generalisation on real vs shuffled labels.

Design invariants (scientific-code safety)
-------------------------------------------
* **No existing file is mutated.** This module only reads; the CLI that wraps
  it writes only to ``<data-dir>/geometry_null_refit/``.
* **GBM hyperparameters are copied verbatim** from
  ``proteinlens/analysis/geometry/classifiers.py:460–467``. If you change
  them there, update :data:`_GBM_HYPER` here in lock-step or observed-parity
  diagnostics will flag the drift.
* **Sampling replicates ``collect_node_fragments``** bit-for-bit:

  - 80th-percentile activation threshold over every interior-residue nonzero
    activation in the top-500 activating proteins.
  - Activated positions capped at ``_MAX_ACTIVATED = 500`` after sorting by
    activation strength descending (stable sort to match Python's
    ``list.sort``).
  - Background is sampled with ``numpy.random.default_rng(42)`` — the same
    seed ``collect_node_fragments`` uses internally — so the draws line up
    with the pipeline even though we do not call that function directly.
  - Total background size ``n_pos_uncapped * bg_ratio`` (uncapped — matches
    ``collect_node_fragments`` line 220), split roughly 50/50 between hard
    negatives (0 < activation < threshold) and zeros.

* **Permutation RNG** is ``np.random.default_rng(seed + fid +
  _GEOM_REFIT_RNG_OFFSET)``. ``_GEOM_REFIT_RNG_OFFSET`` is picked to not
  collide with the three offsets in ``compute_permutation_null.py``:

  - ``_PROT_RNG_OFFSET``  = 10_000_000
  - ``_PWM_RNG_OFFSET``   = 20_000_000
  - ``_CATH_PROT_RNG_OFFSET`` = 30_000_000

* **Within-protein shuffle** preserves the value set per protein, so the
  quantile-derived threshold is invariant under shuffling; this is asserted
  per permutation as a defensive check.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import numpy as np

from proteinlens.analysis.geometry.residue_features import (
    ACTIVE_GEOM_NAMES,
    extract_local_feature_vector,
    select_features,
)

logger = logging.getLogger(__name__)


# Geom profiles are per-protein and identical across features, so the same
# file is reopened many times across a refit run (top-500 protein sets
# overlap heavily across features). An in-process LRU keyed by (dir,
# accession) keeps the N most-recent parsed profiles resident. Under fork()
# each worker gets its own empty cache.
#
# Entry size: ~(L, 3) float64 ca + six length-L float64 profile arrays +
# a sequence string. Empirically ~100-120 KB per entry for L≈2000 residues,
# ~30-40 KB for shorter chains. At maxsize=500 × 120 KB × 16 workers this
# caps at ~1 GB of pod memory (headroom for a 32 GB node). The hit rate at
# 500 is still ≥90% in practice because each feature's top-500 protein set
# overlaps heavily with its immediate neighbours'.
@functools.lru_cache(maxsize=500)
def _load_geom_profile_cached(
    geom_profile_dir: Path, acc: str
) -> dict[str, Any] | None:
    """Return ``{"ca": ..., "profiles": {...}, "sequence": str}`` or None on failure.

    ``profiles`` carries the six arrays the refit loader needs. The cache key
    includes ``geom_profile_dir`` so callers that override the profile
    location via CLI don't cross-contaminate.
    """
    path = geom_profile_dir / f"{acc}.npz"
    try:
        with np.load(path, allow_pickle=True) as gp:
            ca = np.array(gp["ca"])
            profiles = {
                k: np.array(gp[k])[: len(ca)]
                for k in (
                    "curvature",
                    "torsion",
                    "planarity",
                    "tangents",
                    "helix_mask",
                    "categories",
                )
            }
            seq_arr = gp.get("sequence", np.array([""]))
            seq = str(seq_arr[0]) if len(seq_arr) > 0 else ""
        return {"ca": ca, "profiles": profiles, "sequence": seq}
    except (OSError, KeyError, ValueError) as e:
        logger.debug("geom load failed for %s: %s", acc, e)
        return None


# RNG offset for this module's permutation shuffle. Distinct from all offsets
# in scripts/compute_permutation_null.py so that enabling the refit null does
# not alter the RNG streams consumed by existing null metrics.
_GEOM_REFIT_RNG_OFFSET = 40_000_000


# Activation-column-cache schema version. Bump whenever the on-disk layout
# of ``activation_col_cache/{fid:04d}.npz`` changes. The loader refuses any
# cache whose ``meta["cache_version"]`` does not match — better to silently
# fall back than to consume a stale layout with the wrong semantics.
_CACHE_VERSION = 1


# Pipeline defaults — match proteinlens/analysis/feature_pipeline/config.py:
#   geometry_act_quantile           = 0.80
#   geometry_fragment_half_w        = 10
#   geometry_frag_top_k             = 100  → cap on activated = 5× = 500
#   geometry_bg_ratio               = 3
#   geometry_min_activated_positions = 200
_ACT_QUANTILE = 0.80
_HALF_W = 10
_BG_RATIO = 3
_MAX_ACTIVATED = 500  # collect_node_fragments uses max_fragments * 5

# Seeds — match collect_node_fragments + train_motif_classifier exactly.
_BG_SAMPLING_SEED = 42
_GBM_RANDOM_STATE = 42

# GBM hyperparameters — verbatim copy of classifiers.py:460–467.
# min_samples_leaf depends on len(X) at fit time, so it is set per call.
_GBM_HYPER: dict[str, Any] = dict(
    n_estimators=80,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    random_state=_GBM_RANDOM_STATE,
)


# ──────────────────────────────────────────────────────────────────────
# Protein data loading — mirrors compute_permutation_null.py:602–639
# ──────────────────────────────────────────────────────────────────────


def _iter_cache_entries(
    cache_path: Path,
    expected_max_proteins: int,
) -> list[tuple[str, np.ndarray]] | None:
    """Return ``[(accession, activation_column), ...]`` in refit-iteration order.

    The cache's ``meta`` payload is checked against the caller's expectations
    before any data is returned:

    * ``meta["cache_version"]`` must equal :data:`_CACHE_VERSION`. A mismatch
      implies the on-disk layout has drifted from what this loader knows how
      to interpret — consuming it would silently produce wrong numbers.
    * ``meta["max_proteins"]`` must equal ``expected_max_proteins``. The cache
      stores the top-N proteins the precompute was told to materialise, so a
      run that asks for a different N would get the wrong protein set (extra
      proteins flowing into the fit, or missing proteins the per-file path
      would have included). Refusing the cache is strictly safer than a
      silent truncation.

    Returns ``None`` in any of these cases — the caller treats this as
    "cache absent" and falls back to the per-file loader, which is
    numerically authoritative.
    """
    try:
        with np.load(cache_path, allow_pickle=True) as npz:
            accessions = npz["accessions"]
            columns = npz["columns"]
            offsets = npz["offsets"]
            meta_arr = npz["meta"] if "meta" in npz.files else None
    except (OSError, KeyError, ValueError) as e:
        logger.warning("cache read failed for %s: %s — falling back", cache_path, e)
        return None

    # Meta check. A missing ``meta`` key indicates a cache built by a
    # script version that did not yet write metadata — refuse it. Only
    # verified caches are allowed through.
    if meta_arr is None:
        logger.warning(
            "cache %s has no meta payload (pre-versioning build); falling back",
            cache_path,
        )
        return None
    try:
        import json as _json
        meta = _json.loads(str(meta_arr[0]))
    except (ValueError, IndexError, TypeError) as e:
        logger.warning("cache %s has malformed meta (%s); falling back", cache_path, e)
        return None
    cache_version = meta.get("cache_version")
    cache_max = meta.get("max_proteins")
    if cache_version != _CACHE_VERSION:
        logger.warning(
            "cache %s has version %s (expected %d); falling back",
            cache_path, cache_version, _CACHE_VERSION,
        )
        return None
    if cache_max != expected_max_proteins:
        logger.warning(
            "cache %s was built with max_proteins=%s but caller requested %d; "
            "falling back to per-file loader",
            cache_path, cache_max, expected_max_proteins,
        )
        return None

    out: list[tuple[str, np.ndarray]] = []
    for i, acc in enumerate(accessions):
        col = np.ascontiguousarray(
            columns[offsets[i]: offsets[i + 1]], dtype=np.float32
        )
        out.append((str(acc), col))
    return out


def _load_protein_data(
    fid: int,
    act_matrix_full: np.ndarray,
    row_to_acc: dict[int, str],
    act_file_map: dict[str, Path],
    geom_profile_dir: Path,
    geom_profile_files: set[str],
    max_proteins: int = 500,
    activation_col_cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Return ``protein_data`` for the top-``max_proteins`` activating proteins.

    Replicates the fallback logic in ``compute_permutation_null.py:602–639``.
    Each dict has ``accession``, ``act_matrix`` (``(n,)`` column), ``ca``,
    ``profiles``, ``n_residues``, ``sequence``.

    Two load paths, chosen by the presence of ``activation_col_cache_dir/
    {fid:04d}.npz``:

    * **Cached** — one file open per feature for activations. Accession order
      is read from the cache; it is guaranteed to match the refit iteration
      order because ``build_activation_column_cache.py`` writes entries sorted
      by the same ``active_rows`` array computed here.
    * **Per-file** — the original loader, one ``.npz`` per protein for
      activations. Used when no cache is present so the module remains
      functional in any environment.

    In both paths, geom profiles go through :func:`_load_geom_profile_cached`,
    a per-process LRU so the same profile is not reparsed on every feature.
    """
    cache_path: Path | None = None
    if activation_col_cache_dir is not None:
        candidate = activation_col_cache_dir / f"{fid:04d}.npz"
        if candidate.is_file():
            cache_path = candidate

    # ── Cached path ───────────────────────────────────────────────────
    if cache_path is not None:
        entries = _iter_cache_entries(cache_path, expected_max_proteins=max_proteins)
        if entries is None:
            # Cache missing/mismatched/corrupt — fall through to per-file.
            cache_path = None
        else:
            protein_data: list[dict[str, Any]] = []
            for acc, act_col in entries:
                if acc not in geom_profile_files:
                    continue
                gp = _load_geom_profile_cached(geom_profile_dir, acc)
                if gp is None:
                    continue
                ca = gp["ca"]
                n = min(len(ca), act_col.shape[0])
                if n < 2 * _HALF_W + 1:
                    continue
                profiles = {k: v[:n] for k, v in gp["profiles"].items()}
                protein_data.append(
                    {
                        "accession": acc,
                        "act_matrix": act_col[:n],
                        "ca": ca[:n],
                        "profiles": profiles,
                        "n_residues": n,
                        "sequence": gp["sequence"],
                    }
                )
            return protein_data

    # ── Per-file path (original) ──────────────────────────────────────
    node_col = act_matrix_full[:, fid]
    active_rows = np.where(node_col > 0)[0]
    if len(active_rows) > max_proteins:
        top_idx = np.argsort(node_col[active_rows])[-max_proteins:]
        active_rows = active_rows[top_idx]

    protein_data_fallback: list[dict[str, Any]] = []
    for row_idx in active_rows:
        acc = row_to_acc.get(int(row_idx))
        if acc is None or acc not in geom_profile_files:
            continue
        act_path = act_file_map.get(acc)
        if act_path is None:
            continue
        try:
            # Load only column `fid` — the full (n_residues, n_features) matrix
            # is ~12MB per protein at L4, and we only ever read one column.
            with np.load(act_path) as _npz:
                act_col = np.ascontiguousarray(
                    _npz["activations"][:, fid], dtype=np.float32
                )
        except (OSError, KeyError, ValueError) as e:
            logger.debug("skip protein %s for fid %d (activations): %s", acc, fid, e)
            continue
        gp = _load_geom_profile_cached(geom_profile_dir, acc)
        if gp is None:
            continue
        ca = gp["ca"]
        n = min(len(ca), act_col.shape[0])
        if n < 2 * _HALF_W + 1:
            continue
        profiles = {k: v[:n] for k, v in gp["profiles"].items()}
        protein_data_fallback.append(
            {
                "accession": acc,
                # 1-D column — downstream readers already handle .ndim==1.
                "act_matrix": act_col[:n],
                "ca": ca[:n],
                "profiles": profiles,
                "n_residues": n,
                "sequence": gp["sequence"],
            }
        )
    return protein_data_fallback


# ──────────────────────────────────────────────────────────────────────
# Residue-level cache: feat_vecs + activations + per-protein boundaries
# ──────────────────────────────────────────────────────────────────────


def _build_residue_cache(
    protein_data: list[dict[str, Any]], fid: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]] | None:
    """Extract per-residue data for every interior residue, once.

    Returns
    -------
    feat_vecs : (N, D) float32 — may contain non-finite rows (see ``valid``).
    activations : (N,) float32 — per-residue SAE activation value.
    valid : (N,) bool — ``True`` iff the corresponding row of ``feat_vecs``
        has no NaN/inf. Used to align with ``train_motif_classifier``'s NaN
        filter (``classifiers.py:426``) for training, and to restrict the
        evaluation population for scoring (matches
        ``compute_concordance_metrics:686`` and
        ``_load_gbm_and_predict:731–734``).
    boundaries : list of ``(start, end)`` into the pooled arrays, one per
        protein. Used by the within-protein shuffle.

    ``None`` if no valid residues across all proteins.
    """
    feat_list: list[np.ndarray] = []
    act_list: list[float] = []
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    # Use NaN sentinel for positions where extract_local_feature_vector fails —
    # this matches train_motif_classifier's downstream filter exactly.
    feat_dim = len(ACTIVE_GEOM_NAMES)
    nan_row = np.full(feat_dim, np.nan, dtype=np.float32)

    for pdata in protein_data:
        n = pdata["n_residues"]
        start = cursor
        profiles = pdata["profiles"]
        ca = pdata["ca"]
        seq = pdata.get("sequence", "")
        # If act_matrix is (n, D) take column fid; if already (n,) take as-is.
        am = pdata["act_matrix"]
        col = am[:, fid] if am.ndim == 2 and am.shape[1] > fid else am
        for pos in range(_HALF_W, n - _HALF_W):
            fv = extract_local_feature_vector(profiles, ca, pos, _HALF_W, sequence=seq)
            if fv is None:
                row = nan_row
            else:
                try:
                    row = select_features(fv).astype(np.float32, copy=False)
                except Exception:
                    row = nan_row
            feat_list.append(row)
            act_list.append(float(col[pos]))
            cursor += 1
        boundaries.append((start, cursor))

    if cursor == 0:
        return None

    feat_vecs = np.asarray(feat_list, dtype=np.float32)
    activations = np.asarray(act_list, dtype=np.float32)
    valid = np.isfinite(feat_vecs).all(axis=1)
    return feat_vecs, activations, valid, boundaries


# ──────────────────────────────────────────────────────────────────────
# Within-protein shuffle
# ──────────────────────────────────────────────────────────────────────


def _shuffle_within_proteins(
    arr: np.ndarray,
    boundaries: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle values independently within each (start, end) segment.

    Matches ``scripts/compute_permutation_null.py:_shuffle_within_proteins``
    byte-for-byte given the same RNG state.
    """
    shuffled = arr.copy()
    for start, end in boundaries:
        rng.shuffle(shuffled[start:end])
    return shuffled


# ──────────────────────────────────────────────────────────────────────
# Compute threshold — all-residue activations, before feat-validity filter
# ──────────────────────────────────────────────────────────────────────


def _compute_threshold(activations: np.ndarray) -> float:
    """80th-percentile of nonzero activations; median fallback if zero.

    Matches ``collect_node_fragments:164–190`` semantically.

    Note: ``collect_node_fragments`` computes the threshold from **every**
    position of the raw ``act_matrix`` column (including boundary residues
    that are never used as fragment centres). Callers that restrict
    ``activations`` to interior residues therefore get a slightly different
    threshold. To match the pipeline bit-for-bit, feed this function the
    full-column nonzero activations (see
    :func:`_compute_threshold_from_protein_data`).
    """
    nz = activations[activations > 0]
    if len(nz) < 20:
        return 0.0
    thr = float(np.quantile(nz, _ACT_QUANTILE))
    if thr <= 0:
        thr = float(np.median(nz))
    return thr


def _compute_threshold_from_protein_data(
    protein_data: list[dict[str, Any]], fid: int
) -> float:
    """Pipeline-identical threshold: 80th-percentile of nonzero activations
    across **every** residue of every protein (not just interior positions).
    """
    nonzero_chunks: list[np.ndarray] = []
    for pdata in protein_data:
        am = pdata["act_matrix"]
        col = am[:, fid] if am.ndim == 2 and am.shape[1] > fid else am
        nz = np.asarray(col)
        nz = nz[nz > 0]
        if nz.size:
            nonzero_chunks.append(nz.astype(np.float64, copy=False))
    if not nonzero_chunks:
        return 0.0
    all_nz = np.concatenate(nonzero_chunks)
    if all_nz.size < 20:
        return 0.0
    thr = float(np.quantile(all_nz, _ACT_QUANTILE))
    if thr <= 0:
        thr = float(np.median(all_nz))
    return thr


# ──────────────────────────────────────────────────────────────────────
# Sampling — replicate collect_node_fragments exactly
# ──────────────────────────────────────────────────────────────────────


def _sample_train_indices(
    activations: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Select training-set indices matching ``collect_node_fragments``.

    Returns
    -------
    pos_idx : (<=_MAX_ACTIVATED,) int64 — positions of top-activating
        residues, sorted by activation descending with a stable sort
        (matches Python ``list.sort`` tiebreaking by insertion order).
    bg_idx : (n_bg,) int64 — background positions, ordered as
        ``[hard_negatives, zeros]``; within each half, order is the
        ``rng.choice`` draw order using seed 42.
    n_pos_uncapped : int — count of positions with ``activation >=
        threshold`` before the ``_MAX_ACTIVATED`` cap. Drives the
        background-size calculation (``n_bg = bg_ratio * n_pos_uncapped``).
    """
    is_pos = activations >= threshold
    is_hneg = (activations > 0) & (~is_pos)
    is_zero = activations == 0

    pos_unsorted = np.flatnonzero(is_pos)
    hneg_indices = np.flatnonzero(is_hneg)
    zero_indices = np.flatnonzero(is_zero)

    n_pos_uncapped = int(pos_unsorted.size)
    if n_pos_uncapped == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 0

    # Sort by activation desc (stable — matches Python list.sort). Negate then
    # stable-sort ascending = descending with stable ties.
    order = np.argsort(-activations[pos_unsorted], kind="stable")
    pos_sorted = pos_unsorted[order]
    pos_idx = pos_sorted[:_MAX_ACTIVATED].astype(np.int64, copy=False)

    bg_rng = np.random.default_rng(_BG_SAMPLING_SEED)
    n_bg_total = min(
        int(hneg_indices.size + zero_indices.size),
        n_pos_uncapped * _BG_RATIO,
    )
    n_hard = min(int(hneg_indices.size), n_bg_total // 2)
    n_zero = min(int(zero_indices.size), n_bg_total - n_hard)

    # Replicate collect_node_fragments:225–235 verbatim. Only call
    # rng.choice when the pool is strictly larger than the draw count —
    # otherwise slice in residue order.
    if n_hard > 0 and hneg_indices.size > n_hard:
        draw = bg_rng.choice(hneg_indices.size, size=n_hard, replace=False)
        hneg_kept = hneg_indices[draw]
    else:
        hneg_kept = hneg_indices[:n_hard]

    if n_zero > 0 and zero_indices.size > n_zero:
        draw = bg_rng.choice(zero_indices.size, size=n_zero, replace=False)
        zero_kept = zero_indices[draw]
    else:
        zero_kept = zero_indices[:n_zero]

    bg_idx = np.concatenate([hneg_kept, zero_kept]).astype(np.int64, copy=False)
    return pos_idx, bg_idx, n_pos_uncapped


# ──────────────────────────────────────────────────────────────────────
# Fit + score — matches train_motif_classifier's GBM bit-for-bit
# ──────────────────────────────────────────────────────────────────────


def _fit_and_score(
    activations: np.ndarray,
    feat_vecs: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> tuple[float, int, int, int]:
    """Train a GBM on sampled (activated, background) fragments, score every
    valid-feat residue, return PR-AUC against activations >= threshold.

    Parameters
    ----------
    activations : (N,) float32 — per-residue SAE activation values (real or
        shuffled).
    feat_vecs : (N, D) float32 — per-residue feature vectors (may contain
        non-finite rows).
    valid : (N,) bool — True iff feat_vecs[i] is finite.
    threshold : float — SAE activation threshold for positives.

    Returns
    -------
    pr_auc, n_train_valid, n_pos_train, n_neg_train : float, int, int, int
    """
    # sklearn's import is expensive at module load; defer until needed.
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import average_precision_score

    pos_idx, bg_idx, _ = _sample_train_indices(activations, threshold)
    if pos_idx.size < 20 or bg_idx.size < 20:
        return 0.0, 0, int(pos_idx.size), int(bg_idx.size)

    train_idx = np.concatenate([pos_idx, bg_idx])
    X = feat_vecs[train_idx]
    y = np.concatenate(
        [
            np.ones(pos_idx.size, dtype=np.int8),
            np.zeros(bg_idx.size, dtype=np.int8),
        ]
    )

    # NaN/inf filter — matches train_motif_classifier:426.
    row_valid = np.all(np.isfinite(X), axis=1)
    X = X[row_valid]
    y = y[row_valid]
    if len(X) < 40 or int(y.sum()) < 10 or int(y.sum()) == len(y):
        return 0.0, int(len(X)), int(pos_idx.size), int(bg_idx.size)

    gbm = GradientBoostingClassifier(
        **_GBM_HYPER,
        min_samples_leaf=max(5, int(0.02 * len(X))),
    )
    gbm.fit(X, y)

    # Score every VALID-feat residue. The evaluation population here matches
    # compute_concordance_metrics:686 (which filters to finite feat_vec rows)
    # and _load_gbm_and_predict:731–734 (same filter).
    if not np.any(valid):
        return 0.0, int(len(X)), int(pos_idx.size), int(bg_idx.size)

    probs_valid = gbm.predict_proba(feat_vecs[valid])[:, 1]
    binary_valid = (activations[valid] >= threshold).astype(np.int8)

    n_pos = int(binary_valid.sum())
    n_eval = int(binary_valid.size)
    if n_pos == 0 or n_pos == n_eval:
        return 0.0, int(len(X)), int(pos_idx.size), int(bg_idx.size)

    pr_auc = float(average_precision_score(binary_valid, probs_valid))
    return pr_auc, int(len(X)), int(pos_idx.size), int(bg_idx.size)


# ──────────────────────────────────────────────────────────────────────
# Top-level per-feature driver
# ──────────────────────────────────────────────────────────────────────


def compute_refit_null(
    fid: int,
    act_matrix_full: np.ndarray,
    row_to_acc: dict[int, str],
    act_file_map: dict[str, Path],
    geom_profile_dir: Path,
    geom_profile_files: set[str],
    n_permutations: int = 100,
    seed: int = 0,
    max_proteins: int = 500,
    stored_avg_precision: float | None = None,
    observed_warn_delta: float = 0.05,
    observed_parity_strict: bool = False,
    activation_col_cache_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Compute the refit-GBM permutation null for a single feature.

    The observed PR-AUC is computed on the same residue population as the null
    (all valid-feat interior residues of the top-``max_proteins`` activating
    proteins). This differs slightly from the stored
    ``geometry_enrichment/*.json:concordance.avg_precision`` because the
    enrichment stage's ``compute_concordance_metrics`` additionally filters to
    proteins with at least one residue above threshold. The delta is recorded
    in the output JSON as ``observed_parity_delta`` for diagnostic.

    Observed-parity policy
    ----------------------
    * ``observed_parity_strict = False`` (default, exploratory mode) — if the
      delta exceeds ``observed_warn_delta`` a warning is logged but the
      feature is still processed.
    * ``observed_parity_strict = True`` (paper-grade mode) — if the delta
      exceeds ``observed_warn_delta`` the feature is **skipped** (returns
      ``None``) so no refit JSON is written. Use this when the output will
      feed publishable numbers; it guarantees the refit evaluation
      population closely matches the pipeline's enrichment-stage
      population.

    Returns None if the feature cannot be processed (too few activating
    proteins, too few positives, or parity-strict abort).
    """
    protein_data = _load_protein_data(
        fid,
        act_matrix_full,
        row_to_acc,
        act_file_map,
        geom_profile_dir,
        geom_profile_files,
        max_proteins=max_proteins,
        activation_col_cache_dir=activation_col_cache_dir,
    )
    if len(protein_data) < 2:
        return None

    # Threshold uses the FULL column (boundary + interior residues) — this is
    # what the pipeline's ``collect_node_fragments`` does. Computing on
    # interior-only activations gives a subtly lower quantile because the
    # boundary residues are typically lower-activation, shifting the
    # distribution.
    threshold = _compute_threshold_from_protein_data(protein_data, fid)
    if threshold <= 0:
        return None

    cache = _build_residue_cache(protein_data, fid)
    if cache is None:
        return None
    feat_vecs, activations, valid, boundaries = cache
    if activations.size < 20:
        return None

    n_positives_real = int((activations >= threshold).sum())
    if n_positives_real < 20:
        return None

    # ── Observed ────────────────────────────────────────────────
    observed, n_train_obs, n_pos_obs, n_neg_obs = _fit_and_score(
        activations, feat_vecs, valid, threshold
    )
    if observed == 0.0 and n_train_obs == 0:
        return None

    parity_delta = None
    if stored_avg_precision is not None:
        parity_delta = abs(observed - stored_avg_precision)
        if parity_delta > observed_warn_delta:
            if observed_parity_strict:
                logger.error(
                    "fid %d: observed %.6f vs stored avg_precision %.6f (delta %.6f) "
                    "exceeds strict threshold %.3f; SKIPPING feature",
                    fid,
                    observed,
                    stored_avg_precision,
                    parity_delta,
                    observed_warn_delta,
                )
                return None
            logger.warning(
                "fid %d: observed %.6f vs stored avg_precision %.6f (delta %.6f) — "
                "consider investigating protein-filter divergence",
                fid,
                observed,
                stored_avg_precision,
                parity_delta,
            )

    # ── Null loop ───────────────────────────────────────────────
    rng_perm = np.random.default_rng(seed + fid + _GEOM_REFIT_RNG_OFFSET)
    null_prauc = np.zeros(n_permutations, dtype=np.float64)
    for k in range(n_permutations):
        shuffled_acts = _shuffle_within_proteins(activations, boundaries, rng_perm)
        # Defensive: shuffle preserves the value set per protein, so the
        # interior-residue value multiset is unchanged → any quantile of the
        # interior values is invariant. Assert once to catch regressions in
        # ``_shuffle_within_proteins`` (e.g., accidental mutation of the
        # underlying buffer).
        if k == 0:
            assert np.array_equal(np.sort(shuffled_acts), np.sort(activations)), (
                "_shuffle_within_proteins did not produce a permutation of "
                "the original activation multiset"
            )

        null_val, _, _, _ = _fit_and_score(
            shuffled_acts, feat_vecs, valid, threshold
        )
        null_prauc[k] = null_val

    # Phipson & Smyth (2010) one-sided p-value.
    p_value = float((1 + np.sum(null_prauc >= observed)) / (n_permutations + 1))

    return {
        "feature_id": int(fid),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "rng_offset": int(_GEOM_REFIT_RNG_OFFSET),
        "observed_prauc": round(float(observed), 6),
        "observed_prauc_stored": (
            round(float(stored_avg_precision), 6)
            if stored_avg_precision is not None
            else None
        ),
        "observed_parity_delta": (
            round(float(parity_delta), 6) if parity_delta is not None else None
        ),
        "null_prauc_refit": [round(float(v), 6) for v in null_prauc.tolist()],
        "null_mean": round(float(null_prauc.mean()), 6),
        "null_std": round(float(null_prauc.std()), 6),
        "p_value_refit": round(float(p_value), 6),
        "n_proteins": int(len(protein_data)),
        "n_residues_total": int(feat_vecs.shape[0]),
        "n_residues_valid": int(valid.sum()),
        "n_pos_real": int(n_positives_real),
        "n_train_obs": int(n_train_obs),
        "n_train_pos_obs": int(n_pos_obs),
        "n_train_neg_obs": int(n_neg_obs),
        "threshold_sae": round(float(threshold), 6),
        "geometry_act_quantile": _ACT_QUANTILE,
        "half_w": _HALF_W,
        "bg_ratio": _BG_RATIO,
        "max_activated": _MAX_ACTIVATED,
        "gbm_hyperparams": dict(_GBM_HYPER),
        "source": "refit-gbm",
        "script_version": "compute_geometry_null_refit v1",
    }
