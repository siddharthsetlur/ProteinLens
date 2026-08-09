"""Stage 7 — Sequence motif (k-mer) F1 enrichment per SAE feature.

For each SAE feature, discovers short amino-acid motifs (k-mers) whose
presence at a residue position predicts high feature activation at that
position.

**Approach:**
For every feature we pool all proteins (from ``top_sequences`` and every
``activation_bins`` bucket), extract overlapping k-mers at each residue
position, then ask: "does the k-mer at position *i* predict that
activation[i] exceeds a threshold?"  We sweep 100 evenly-spaced
thresholds from 0 to the feature's global max and report the threshold
that maximises F1 for each k-mer.

**Vectorised F1 computation:**
Pre-compute an ``activated_matrix`` of shape ``(n_thresholds, N)`` where
entry ``[t, i]`` is True when ``activation[i] > threshold[t]``.  For a
k-mer with occurrence index set ``idx``, true positives at each
threshold are simply ``activated_matrix[:, idx].sum(axis=1)``.  This
avoids a Python loop over thresholds.

**Outputs:**
- ``motif_enrichment/{feat_idx:04d}.json`` — per-feature enrichment
  with top motifs, their F1, precision, recall, and counts.
- ``motif_enrichment/summary.json`` — quick-lookup summary keyed by
  feature id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from proteinlens.analysis.feature_pipeline.config import PipelineConfig

# 20 standard amino acids — any character outside this set is skipped
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ===================================================================
# Helper: extract k-mers paired with activations (checklist 7.2.2)
# ===================================================================


def _extract_kmers_with_activations(
    sequence: str,
    activations: List[float],
    k: int,
) -> List[Tuple[str, float]]:
    """Extract overlapping k-mers from a protein sequence, paired with activations.

    For each residue position *i* (from ``k // 2`` to ``len(sequence) - k // 2``),
    the k-mer centred on *i* is ``sequence[i - k//2 : i + k//2 + 1]``.  Positions
    where any character in the k-mer falls outside the 20 standard amino acids are
    skipped.

    Args:
        sequence: Amino-acid sequence (one-letter codes).
        activations: Per-residue activation values, same length as *sequence*.
        k: k-mer length (must be odd for symmetric centering; default 3).

    Returns:
        List of ``(kmer_string, activation_at_centre)`` tuples.  Empty if the
        sequence is shorter than *k* or if ``len(sequence) != len(activations)``
        (a rare data inconsistency that is silently skipped).
    """
    if len(sequence) != len(activations):
        return []
    half = k // 2
    n = len(sequence)
    if n < k:
        return []

    pairs: List[Tuple[str, float]] = []
    for i in range(half, n - half):
        kmer = sequence[i - half : i + half + 1]
        # Skip if any residue is non-standard
        if all(ch in _VALID_AA for ch in kmer):
            pairs.append((kmer, activations[i]))
    return pairs


# ===================================================================
# Helper: pool proteins for a feature (checklist 7.2.3)
# ===================================================================


def _pool_proteins_for_feature(
    feature_data: Dict[str, Any],
) -> List[Tuple[str, str, List[float]]]:
    """Collect all proteins associated with a feature, deduplicated by accession.

    Extracts proteins from both ``top_sequences`` and every bin in
    ``activation_bins``.  Each protein must have a non-null
    ``per_residue_activations`` array.

    Args:
        feature_data: The parsed per-feature JSON (as written by Stage 4
            assembly).  Expected keys: ``top_sequences`` (list of protein
            dicts) and ``activation_bins`` (dict of bin-label -> list of
            protein dicts).

    Returns:
        List of ``(accession, sequence, per_residue_activations)`` tuples,
        one per unique accession.  Order follows first appearance (top
        sequences first, then bins in sorted key order).
    """
    seen: set = set()
    result: List[Tuple[str, str, List[float]]] = []

    def _add(entry: Dict[str, Any]) -> None:
        acc = entry.get("accession")
        if acc is None or acc in seen:
            return
        pra = entry.get("per_residue_activations")
        seq = entry.get("sequence")
        if pra is None or seq is None:
            return
        seen.add(acc)
        result.append((acc, seq, pra))

    for entry in feature_data.get("top_sequences", []):
        _add(entry)
    for bin_label in sorted(feature_data.get("activation_bins", {}).keys()):
        for entry in feature_data["activation_bins"][bin_label]:
            _add(entry)

    return result


# ===================================================================
# Vectorised F1 computation (checklist 7.2.4)
# ===================================================================


def _compute_best_motif_f1(
    kmer_indices: Dict[str, np.ndarray],
    all_activations: np.ndarray,
    feat_max: float,
    n_steps: int,
    min_count: int,
    top_n: int,
) -> List[Dict[str, Any]]:
    """Find the best F1 score for each k-mer across a sweep of activation thresholds.

    For each k-mer whose occurrence count meets *min_count*, we compute
    precision, recall, and F1 at every threshold in a single vectorised
    operation over the pre-computed ``activated_matrix``.

    **Key arrays:**

    - ``thresholds``: shape ``(n_steps,)`` — evenly spaced from
      ``feat_max / (n_steps + 1)`` to ``feat_max``.
    - ``activated_matrix``: shape ``(n_steps, N)`` — boolean mask where
      ``activated_matrix[t, i]`` is True when ``all_activations[i] >
      thresholds[t]``.
    - For k-mer with index set ``idx``:
        - ``tp = activated_matrix[:, idx].sum(axis=1)``
        - ``fp = len(idx) - tp``
        - ``fn = n_activated - tp``  (where ``n_activated`` is total
          activated residues at each threshold)

    Args:
        kmer_indices: Dict mapping each k-mer string to a 1-D int array
            of positions in the pooled activation vector where that k-mer
            occurs.
        all_activations: 1-D float array of activation values for all
            pooled residue positions.
        feat_max: Global maximum activation for this feature (used to
            define the threshold range).
        n_steps: Number of thresholds to sweep.
        min_count: Minimum occurrences for a k-mer to be tested.
        top_n: Number of top-scoring motifs to return.

    Returns:
        List of result dicts sorted by ``best_f1`` descending, each
        containing: ``motif``, ``best_f1``, ``best_threshold``,
        ``best_threshold_normalized``, ``precision_at_best``,
        ``recall_at_best``, ``n_occurrences``, ``n_true_positives``,
        ``n_false_positives``, ``n_false_negatives``, ``interpretation``.
        Empty list if no k-mer passes the *min_count* filter.
    """
    N = len(all_activations)
    if N == 0 or feat_max <= 0:
        return []

    # Threshold grid: exclude 0 (everything would be "activated")
    thresholds = np.linspace(0, feat_max, n_steps + 1)[1:]  # (n_steps,)

    # activated_matrix[t, i] = True when activation[i] > threshold[t]
    activated_matrix = all_activations[None, :] > thresholds[:, None]  # (n_steps, N)
    n_activated = activated_matrix.sum(axis=1)  # (n_steps,)

    results: List[Dict[str, Any]] = []

    for kmer, idx in kmer_indices.items():
        if len(idx) < min_count:
            continue

        # TP: k-mer positions that are also activated
        tp = activated_matrix[:, idx].sum(axis=1).astype(float)  # (n_steps,)
        fp = float(len(idx)) - tp       # predicted positive but not activated
        fn = n_activated.astype(float) - tp  # activated but k-mer not present

        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f1 = np.where(
                precision + recall > 0,
                2 * precision * recall / (precision + recall),
                0.0,
            )

        best_idx = int(np.argmax(f1))
        best_f1 = float(f1[best_idx])
        best_thresh = float(thresholds[best_idx])

        results.append({
            "motif": kmer,
            "best_f1": round(best_f1, 4),
            "best_threshold": round(best_thresh, 4),
            "best_threshold_normalized": round(best_thresh / feat_max, 4),
            "precision_at_best": round(float(precision[best_idx]), 4),
            "recall_at_best": round(float(recall[best_idx]), 4),
            "n_occurrences": int(len(idx)),
            "n_true_positives": int(tp[best_idx]),
            "n_false_positives": int(fp[best_idx]),
            "n_false_negatives": int(fn[best_idx]),
            "interpretation": (
                f"Motif {kmer} predicts activation > {best_thresh:.2f} "
                f"({best_thresh / feat_max:.0%} of max) with F1={best_f1:.2f}"
            ),
        })

    # Sort by F1 descending, keep top_n
    results.sort(key=lambda r: r["best_f1"], reverse=True)
    return results[:top_n]


# ===================================================================
# Per-feature analysis (checklist 7.2.5)
# ===================================================================


def _analyze_feature(
    feature_data: Dict[str, Any],
    feat_max: float,
    config: PipelineConfig,
) -> Optional[Dict[str, Any]]:
    """Run the full motif enrichment analysis for a single feature.

    Steps:
        1. Pool proteins from ``top_sequences`` and ``activation_bins``.
        2. Extract k-mers paired with per-residue activations.
        3. Build a ``kmer_indices`` dict mapping each k-mer to its
           position indices in the pooled activation array.
        4. Run the vectorised F1 sweep to find the best motif(s).

    Args:
        feature_data: Parsed per-feature JSON from Stage 4.
        feat_max: Global maximum activation for this feature.
        config: Pipeline configuration (supplies k, min_count, etc.).

    Returns:
        Result dict conforming to the per-feature JSON schema (see plan
        checklist 7.2.7), or ``None`` if there are no proteins or no
        eligible k-mers.
    """
    k = config.motif_kmer_k

    # Step 1: pool proteins
    proteins = _pool_proteins_for_feature(feature_data)
    if not proteins:
        return None

    # Step 2: extract k-mers with activations
    all_kmers: List[str] = []
    all_acts: List[float] = []
    for _acc, seq, pra in proteins:
        pairs = _extract_kmers_with_activations(seq, pra, k)
        for kmer, act in pairs:
            all_kmers.append(kmer)
            all_acts.append(act)

    if not all_kmers:
        return None

    all_activations = np.array(all_acts, dtype=np.float64)

    # Step 3: build kmer -> index array mapping
    kmer_indices: Dict[str, List[int]] = {}
    for i, kmer in enumerate(all_kmers):
        kmer_indices.setdefault(kmer, []).append(i)
    kmer_idx_arrays = {km: np.array(idxs) for km, idxs in kmer_indices.items()}

    # Step 4: vectorised F1
    top_motifs = _compute_best_motif_f1(
        kmer_idx_arrays,
        all_activations,
        feat_max,
        n_steps=config.motif_f1_threshold_steps,
        min_count=config.motif_min_count,
        top_n=config.motif_top_n,
    )

    if not top_motifs:
        return None

    return {
        "feature_id": feature_data["feature_id"],
        "feature_max_activation": round(float(feat_max), 6),
        "n_proteins_evaluated": len(proteins),
        "n_total_residues": len(all_activations),
        "n_unique_kmers_tested": sum(
            1 for idxs in kmer_idx_arrays.values()
            if len(idxs) >= config.motif_min_count
        ),
        "k": k,
        "top_motifs": top_motifs,
    }


# ===================================================================
# Public API (checklist 7.2.6)
# ===================================================================


def run_motif_enrichment(config: PipelineConfig) -> None:
    """Execute the sequence motif F1 enrichment stage (Stage 7).

    For each SAE feature, loads its assembled JSON (top sequences and
    activation bins), extracts k-mers at every residue position, and
    finds the motif(s) whose presence best predicts high activation via
    an F1-maximising threshold sweep.

    **Resumability:** Features whose output JSON already exists in
    ``config.motif_enrichment_dir`` are skipped.

    **Outputs:**
        - ``motif_enrichment/{feat_idx:04d}.json`` — per-feature results.
        - ``motif_enrichment/summary.json`` — keyed by feature id, with
          ``best_motif``, ``best_motif_f1``, and ``n_kmers_tested``.

    Args:
        config: Pipeline configuration.  Requires that Stage 4 (assembly)
            has completed — specifically, ``feature_max_activations.npy``
            and per-feature JSONs in ``features/`` must exist.

    Raises:
        FileNotFoundError: If ``feature_max_activations.npy`` is missing.
    """
    # ── Load global max activations ──
    global_max = np.load(config.feature_max_path)  # (num_features,)
    num_features = len(global_max)

    out_dir = config.motif_enrichment_dir
    features_dir = config.features_dir

    n_analyzed = 0
    n_skipped = 0
    summary_features: Dict[str, Dict[str, Any]] = {}

    for feat_idx in tqdm(range(num_features), desc="[motif_enrichment]"):
        feat_max = float(global_max[feat_idx])

        # Skip dead features
        if feat_max == 0:
            n_skipped += 1
            continue

        # Resumability: skip if output already exists
        out_path = out_dir / f"{feat_idx:04d}.json"
        if out_path.exists():
            # Still include in summary from existing file
            try:
                with open(out_path) as f:
                    existing = json.load(f)
                if existing.get("top_motifs"):
                    fid_str = str(feat_idx)
                    summary_features[fid_str] = {
                        "best_motif": existing["top_motifs"][0]["motif"],
                        "best_motif_f1": existing["top_motifs"][0]["best_f1"],
                        "n_kmers_tested": existing.get("n_unique_kmers_tested", 0),
                    }
                n_analyzed += 1
            except (json.JSONDecodeError, KeyError):
                pass
            continue

        # Load per-feature JSON
        feat_path = features_dir / f"{feat_idx:04d}.json"
        if not feat_path.exists():
            n_skipped += 1
            continue

        with open(feat_path) as f:
            feature_data = json.load(f)

        result = _analyze_feature(feature_data, feat_max, config)

        if result is None:
            n_skipped += 1
            continue

        # Write per-feature output
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        # Add to summary
        fid_str = str(feat_idx)
        if result["top_motifs"]:
            summary_features[fid_str] = {
                "best_motif": result["top_motifs"][0]["motif"],
                "best_motif_f1": result["top_motifs"][0]["best_f1"],
                "n_kmers_tested": result["n_unique_kmers_tested"],
            }

        n_analyzed += 1

    # ── Write summary JSON (checklist 7.2.8) ──
    summary = {
        "n_features_analyzed": n_analyzed,
        "n_features_skipped": n_skipped,
        "k": config.motif_kmer_k,
        "features": summary_features,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[motif_enrichment] Analyzed {n_analyzed} features, "
        f"skipped {n_skipped}. "
        f"Wrote results to {out_dir}/"
    )

    from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog

    wlog({
        "motif_enrichment/analyzed": n_analyzed,
        "motif_enrichment/skipped": n_skipped,
    })
