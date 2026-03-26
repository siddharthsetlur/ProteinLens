"""Stage 5c — InterPro F1 enrichment analysis per SAE feature.

For each feature, computes how well InterPro annotations predict feature
activation at both the **protein level** and the **amino-acid (residue)
level**.

**Protein-level F1:**
For each annotation code, we ask: "does having this annotation predict
that the protein activates this feature?"  We sweep activation thresholds
and report the threshold that maximises F1.

**Residue-level F1:**
For the top protein-level annotations, we ask: "does a residue falling
inside a domain boundary predict high activation at that position?"
We sweep activation thresholds on per-residue values and report the best.

**Threshold sweep:**
Rather than testing every unique activation value (which can be very
large), we test ``interpro_f1_threshold_steps`` evenly-spaced thresholds
from 0 to ``feature_max``.  This gives consistent resolution across
features and keeps runtime bounded.

Outputs:
- ``interpro_enrichment/{feat_idx:04d}.json`` — per-feature enrichment
  with both protein-level and residue-level results.
- ``interpro_enrichment/summary.json`` — quick-lookup summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.interpro_api import (
    InterProDomain,
    _load_cached,
)


# ===================================================================
# Public API
# ===================================================================


def run_interpro_enrichment(config: PipelineConfig) -> None:
    """Execute the InterPro F1 enrichment stage (Stage 5c).

    For each SAE feature, loads its InterPro selection (proteins from
    the 11 activation bins), retrieves cached InterPro annotations for
    each protein, and computes protein-level and residue-level F1 scores
    for each annotation code.

    Writes per-feature JSON files to ``config.interpro_enrichment_dir``
    and a summary to ``summary.json``.

    Args:
        config: Pipeline configuration.  Requires that Stages 5a and 5b
            have completed.

    Raises:
        FileNotFoundError: If required upstream outputs are missing.
    """
    # ── Load upstream data ──
    global_max = np.load(config.feature_max_path)  # (num_features,)
    num_features = len(global_max)

    with open(config.interpro_selection_path, "r") as f:
        interpro_selection = json.load(f)

    # Load the memmap for protein-level max activations
    with open(config.pipeline_state_path, "r") as f:
        pipeline_state = json.load(f)
    n_proteins = pipeline_state["total_proteins"]
    acc_index = pipeline_state["accession_index"]  # acc -> row_idx

    protein_maxes = np.memmap(
        config.protein_feature_maxes_path,
        dtype="float32",
        mode="r",
        shape=(n_proteins, num_features),
    )

    # ── Process each feature ──
    n_analyzed = 0
    n_skipped = 0
    summary_features: Dict[str, Dict[str, Any]] = {}

    for feat_idx in tqdm(range(num_features), desc="InterPro enrichment"):
        feat_max = float(global_max[feat_idx])

        # Skip features that never fire
        if feat_max == 0:
            n_skipped += 1
            continue

        feat_key = str(feat_idx)
        feat_selection = interpro_selection["per_feature"].get(feat_key, {})
        bins = feat_selection.get("bins", {})

        # Collect all accessions from all bins with their max activations
        accessions_with_activations = _collect_accessions_with_activations(
            bins, acc_index, protein_maxes, feat_idx
        )

        if len(accessions_with_activations) < config.interpro_min_proteins:
            n_skipped += 1
            continue

        # Load InterPro annotations for all proteins
        protein_annotations = _load_annotations_for_proteins(
            [acc for acc, _ in accessions_with_activations],
            config.interpro_cache_dir,
        )

        # Count proteins with any annotations
        n_with_annotations = sum(
            1 for acc, _ in accessions_with_activations
            if protein_annotations.get(acc)
        )

        if n_with_annotations < config.interpro_min_proteins:
            n_skipped += 1
            continue

        # ── Protein-level F1 (checklist 4.2) ──
        protein_level_results = _compute_protein_level_f1(
            accessions_with_activations=accessions_with_activations,
            protein_annotations=protein_annotations,
            feat_max=feat_max,
            n_threshold_steps=config.interpro_f1_threshold_steps,
            min_proteins=config.interpro_min_proteins,
            top_n=config.interpro_top_annotations,
        )

        # ── Residue-level F1 (checklist 4.3) ──
        residue_level_results = _compute_residue_level_f1(
            protein_level_results=protein_level_results,
            protein_annotations=protein_annotations,
            feat_idx=feat_idx,
            feat_max=feat_max,
            config=config,
        )

        # ── Count only annotations that met the min_proteins threshold ──
        # (i.e. those that actually went through the F1 threshold sweep)
        annotation_protein_counts: Dict[str, Set[str]] = {}
        for acc, _ in accessions_with_activations:
            for domain in protein_annotations.get(acc, []):
                code = domain.interpro_accession
                if code not in annotation_protein_counts:
                    annotation_protein_counts[code] = set()
                annotation_protein_counts[code].add(acc)
        n_annotations_tested = sum(
            1 for code, prots in annotation_protein_counts.items()
            if len(prots) >= config.interpro_min_proteins
        )

        # ── Write per-feature JSON (checklist 4.4) ──
        enrichment_data = {
            "feature_id": feat_idx,
            "feature_max_activation": feat_max,
            "n_proteins_evaluated": len(accessions_with_activations),
            "n_proteins_with_annotations": n_with_annotations,
            "n_unique_annotations_tested": n_annotations_tested,
            "protein_level": protein_level_results,
            "residue_level": residue_level_results,
        }

        out_path = config.interpro_enrichment_dir / f"{feat_idx:04d}.json"
        with open(out_path, "w") as f:
            json.dump(enrichment_data, f, indent=2)

        n_analyzed += 1

        # ── Add to summary (checklist 4.5) ──
        if protein_level_results:
            top_prot = protein_level_results[0]
            summary_entry: Dict[str, Any] = {
                "top_protein_annotation": top_prot["annotation_code"],
                "top_protein_annotation_name": top_prot["annotation_name"],
                "top_protein_f1": top_prot["best_f1"],
                "top_residue_annotation": None,
                "top_residue_f1": None,
            }
            if residue_level_results:
                top_res = residue_level_results[0]
                summary_entry["top_residue_annotation"] = top_res["annotation_code"]
                summary_entry["top_residue_f1"] = top_res["best_f1"]
            summary_features[feat_key] = summary_entry

    # ── Write summary JSON (checklist 4.5) ──
    summary = {
        "n_features_analyzed": n_analyzed,
        "n_features_skipped": n_skipped,
        "features": summary_features,
    }
    summary_path = config.interpro_enrichment_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[interpro_enrichment] Analyzed {n_analyzed} features, "
        f"skipped {n_skipped}. "
        f"Wrote results to {config.interpro_enrichment_dir}/"
    )
    from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog

    wlog({
        "interpro_enrichment/analyzed": n_analyzed,
        "interpro_enrichment/skipped": n_skipped,
    })


# ===================================================================
# Protein-level F1 (checklist item 4.2)
# ===================================================================


def _compute_protein_level_f1(
    accessions_with_activations: List[Tuple[str, float]],
    protein_annotations: Dict[str, List[InterProDomain]],
    feat_max: float,
    n_threshold_steps: int,
    min_proteins: int,
    top_n: int,
) -> List[Dict[str, Any]]:
    """Compute protein-level F1 for each annotation code.

    For each unique InterPro annotation present across the selected
    proteins, we sweep activation thresholds and find the one that best
    separates proteins *with* the annotation (positive) from those
    *without* (negative).

    The sweep tests ``n_threshold_steps`` evenly-spaced thresholds from
    0 to ``feat_max``.  At each threshold ``t``, a protein is predicted
    positive if its max activation > t.

    Args:
        accessions_with_activations: List of ``(accession, max_activation)``
            tuples for all proteins in this feature's InterPro selection.
        protein_annotations: Dict mapping accession to its list of
            ``InterProDomain`` objects.
        feat_max: Global max activation for this feature.
        n_threshold_steps: Number of evenly-spaced thresholds to sweep.
        min_proteins: Minimum number of proteins with an annotation for
            it to be tested.
        top_n: Number of top annotations to return (by best F1).

    Returns:
        List of result dicts (one per annotation), sorted by best_f1
        descending.  Includes all annotations within 0.05 of the top
        F1 if there are ties.
    """
    # Build arrays for efficient vectorised threshold sweep
    accessions = [acc for acc, _ in accessions_with_activations]
    activations = np.array(
        [act for _, act in accessions_with_activations], dtype=np.float64
    )

    # Collect all unique annotation codes and count occurrences
    annotation_proteins: Dict[str, Set[str]] = {}
    annotation_meta: Dict[str, Dict[str, str]] = {}

    for acc in accessions:
        for domain in protein_annotations.get(acc, []):
            code = domain.interpro_accession
            if code not in annotation_proteins:
                annotation_proteins[code] = set()
                # Store metadata from the first occurrence
                annotation_meta[code] = {
                    "annotation_name": domain.interpro_name,
                    "annotation_type": domain.type,
                    "member_db": domain.member_db,
                    "member_accession": domain.member_accession,
                }
            annotation_proteins[code].add(acc)

    # Filter out annotations with too few proteins
    eligible_codes = [
        code for code, proteins in annotation_proteins.items()
        if len(proteins) >= min_proteins
    ]

    if not eligible_codes:
        return []

    # Build threshold array: evenly spaced from 0 to feat_max
    thresholds = np.linspace(0, feat_max, n_threshold_steps + 1)

    results: List[Dict[str, Any]] = []

    for code in eligible_codes:
        # y_true: 1 if protein has this annotation, 0 otherwise
        y_true = np.array(
            [1 if acc in annotation_proteins[code] else 0 for acc in accessions],
            dtype=np.int32,
        )

        best_f1 = 0.0
        best_result: Optional[Dict[str, Any]] = None

        for t in thresholds:
            # y_pred: 1 if protein activation > threshold
            y_pred = (activations > t).astype(np.int32)

            tp, fp, fn, precision, recall, f1 = _compute_f1_from_arrays(
                y_true, y_pred
            )

            if f1 > best_f1:
                best_f1 = f1
                n_with = int(y_true.sum())
                n_without = len(y_true) - n_with
                meta = annotation_meta[code]

                best_result = {
                    "annotation_code": code,
                    "annotation_name": meta["annotation_name"],
                    "annotation_type": meta["annotation_type"],
                    "member_db": meta["member_db"],
                    "member_accession": meta["member_accession"],
                    "best_f1": round(f1, 4),
                    "best_threshold": round(float(t), 4),
                    "best_threshold_normalized": round(
                        float(t) / feat_max if feat_max > 0 else 0.0, 4
                    ),
                    "precision_at_best": round(precision, 4),
                    "recall_at_best": round(recall, 4),
                    "n_proteins_with_annotation": n_with,
                    "n_proteins_without_annotation": n_without,
                    "n_true_positives": int(tp),
                    "n_false_positives": int(fp),
                    "n_false_negatives": int(fn),
                    "interpretation": (
                        f"Proteins with activation > {round(float(t), 2)} "
                        f"({round(float(t) / feat_max * 100 if feat_max > 0 else 0, 0):.0f}% of max) "
                        f"are predicted by annotation {code} with F1={round(f1, 2)}"
                    ),
                }

        if best_result is not None:
            results.append(best_result)

    # Sort by best_f1 descending
    results.sort(key=lambda r: r["best_f1"], reverse=True)

    # Keep top_n, but include all annotations within 0.05 of the best F1
    if results:
        best_f1_overall = results[0]["best_f1"]
        cutoff = best_f1_overall - 0.05
        # Start with top_n, then extend to include ties
        n_keep = min(top_n, len(results))
        while n_keep < len(results) and results[n_keep]["best_f1"] >= cutoff:
            n_keep += 1
        results = results[:n_keep]

    return results


# ===================================================================
# Residue-level F1 (checklist item 4.3)
# ===================================================================


def _compute_residue_level_f1(
    protein_level_results: List[Dict[str, Any]],
    protein_annotations: Dict[str, List[InterProDomain]],
    feat_idx: int,
    feat_max: float,
    config: PipelineConfig,
) -> List[Dict[str, Any]]:
    """Compute residue-level F1 for the top protein-level annotations.

    For each annotation that scored well at the protein level, we ask:
    "does a residue being inside a domain boundary predict high activation
    at that position?"

    We concatenate all residues across all proteins that have both this
    annotation AND per-residue activation data, then sweep thresholds
    on the activation values.  Thresholds are based on percentiles of
    non-zero activations for better coverage of the activation distribution.

    Args:
        protein_level_results: Output of ``_compute_protein_level_f1``.
        protein_annotations: Dict mapping accession to InterProDomain list.
        feat_idx: Feature index for extracting the correct column from
            per-residue activation matrices.
        feat_max: Global max activation for normalising thresholds.
        config: Pipeline configuration (for directory paths and settings).

    Returns:
        List of residue-level result dicts, one per annotation tested.
    """
    results: List[Dict[str, Any]] = []

    for prot_result in protein_level_results:
        code = prot_result["annotation_code"]

        # Collect all proteins that have this annotation AND per-residue data
        all_activations_list: List[np.ndarray] = []
        all_labels_list: List[np.ndarray] = []
        n_proteins_used = 0

        for acc, domains in protein_annotations.items():
            # Check if this protein has the annotation
            matching_domains = [
                d for d in domains if d.interpro_accession == code
            ]
            if not matching_domains:
                continue

            # Try to load per-residue activations
            residue_acts = load_residue_activations(acc, config)
            if residue_acts is None:
                continue

            # Extract the column for this feature
            if feat_idx >= residue_acts.shape[1]:
                # PM FLAG: This should not happen if the SAE dimensions are
                # consistent. If it does, we skip this protein silently.
                continue

            feat_acts = residue_acts[:, feat_idx]  # (seq_len,)
            seq_len = len(feat_acts)

            # Build residue-level labels: 1 if residue is inside any domain
            # boundary for this annotation, 0 otherwise.
            # InterPro positions are 1-based inclusive, so we convert to
            # 0-based: residue i is in-domain if start-1 <= i <= end-1.
            labels = np.zeros(seq_len, dtype=np.int32)
            for d in matching_domains:
                # Convert 1-based inclusive to 0-based inclusive
                start_0 = d.start - 1
                end_0 = d.end - 1
                # Clamp to sequence length to avoid index errors
                start_0 = max(0, start_0)
                end_0 = min(seq_len - 1, end_0)
                labels[start_0 : end_0 + 1] = 1

            all_activations_list.append(feat_acts)
            all_labels_list.append(labels)
            n_proteins_used += 1

        if n_proteins_used == 0:
            continue

        # Concatenate across all proteins
        all_activations = np.concatenate(all_activations_list)
        all_labels = np.concatenate(all_labels_list)
        n_total_residues = len(all_labels)
        n_in_domain = int(all_labels.sum())

        if n_in_domain == 0 or n_in_domain == n_total_residues:
            # All or none are in-domain — F1 is trivial/degenerate
            continue

        # Sweep thresholds using percentiles of non-zero activations
        # for better coverage of the activation distribution
        nonzero_acts = all_activations[all_activations > 0]
        if len(nonzero_acts) == 0:
            continue

        percentile_thresholds = np.percentile(
            nonzero_acts,
            np.linspace(0, 100, config.interpro_f1_threshold_steps),
        )
        # Also include evenly-spaced absolute thresholds for coverage
        linear_thresholds = np.linspace(0, feat_max, config.interpro_f1_threshold_steps)
        thresholds = np.unique(
            np.concatenate([percentile_thresholds, linear_thresholds])
        )

        best_f1 = 0.0
        best_res_result: Optional[Dict[str, Any]] = None

        for t in thresholds:
            y_pred = (all_activations > t).astype(np.int32)
            tp, fp, fn, precision, recall, f1 = _compute_f1_from_arrays(
                all_labels, y_pred
            )

            if f1 > best_f1:
                best_f1 = f1
                best_res_result = {
                    "annotation_code": code,
                    "annotation_name": prot_result["annotation_name"],
                    "member_db": prot_result["member_db"],
                    "member_accession": prot_result["member_accession"],
                    "best_f1": round(f1, 4),
                    "best_threshold": round(float(t), 4),
                    "best_threshold_normalized": round(
                        float(t) / feat_max if feat_max > 0 else 0.0, 4
                    ),
                    "precision_at_best": round(precision, 4),
                    "recall_at_best": round(recall, 4),
                    "n_proteins_used": n_proteins_used,
                    "n_total_residues": n_total_residues,
                    "n_residues_in_domain": n_in_domain,
                    "n_true_positives": int(tp),
                    "n_false_positives": int(fp),
                    "n_false_negatives": int(fn),
                    "interpretation": (
                        f"Residues with activation > {round(float(t), 2)} "
                        f"({round(float(t) / feat_max * 100 if feat_max > 0 else 0, 0):.0f}% of max) "
                        f"overlap with {code} domains with F1={round(f1, 2)}"
                    ),
                }

        if best_res_result is not None:
            results.append(best_res_result)

    return results


# ===================================================================
# Shared utilities (checklist item 4.1)
# ===================================================================


def load_residue_activations(
    accession: str,
    config: PipelineConfig,
) -> Optional[np.ndarray]:
    """Load per-residue activations for a protein, checking both directories.

    Checks ``config.residue_activations_dir`` (Stage 3 output) first,
    then ``config.interpro_residue_activations_dir`` (Stage 5a output).

    Args:
        accession: UniProt accession string.
        config: Pipeline configuration.

    Returns:
        Numpy array of shape ``(seq_len, num_features)`` with float32
        activation values, or ``None`` if no .npz file exists in either
        directory.
    """
    # Check Stage 3 directory first (more likely to exist)
    npz_path = config.residue_activations_dir / f"{accession}.npz"
    if npz_path.exists():
        return np.load(npz_path)["activations"]

    # Check Stage 5a directory
    npz_path = config.interpro_residue_activations_dir / f"{accession}.npz"
    if npz_path.exists():
        return np.load(npz_path)["activations"]

    return None


# ===================================================================
# Internal helpers
# ===================================================================


def _collect_accessions_with_activations(
    bins: Dict[str, List[str]],
    acc_index: Dict[str, int],
    protein_maxes: np.ndarray,
    feat_idx: int,
) -> List[Tuple[str, float]]:
    """Flatten all bins into a list of (accession, max_activation) tuples.

    Looks up each protein's max activation for this feature from the
    survey memmap.

    Args:
        bins: Dict of bin_label -> list of accession strings from the
            InterPro selection.
        acc_index: Accession -> row index mapping for the memmap.
        protein_maxes: ``(n_proteins, num_features)`` memmap array.
        feat_idx: Feature index column to read from the memmap.

    Returns:
        De-duplicated list of ``(accession, max_activation)`` tuples.
    """
    seen: Set[str] = set()
    result: List[Tuple[str, float]] = []

    for bin_accs in bins.values():
        for acc in bin_accs:
            if acc in seen:
                continue
            seen.add(acc)

            # Look up activation from the memmap
            if acc in acc_index:
                row = int(acc_index[acc])
                activation = float(protein_maxes[row, feat_idx])
            else:
                # PM FLAG: This accession is in the selection but not in
                # the memmap index. This shouldn't happen in a normal
                # pipeline run. We include it with activation 0.0 but
                # this may skew results.
                activation = 0.0

            result.append((acc, activation))

    return result


def _load_annotations_for_proteins(
    accessions: List[str],
    cache_dir: Path,
) -> Dict[str, List[InterProDomain]]:
    """Load cached InterPro annotations for a list of proteins.

    Args:
        accessions: List of UniProt accession strings.
        cache_dir: Directory containing cached InterPro JSON files
            (one per protein, created by Stage 5b).

    Returns:
        Dict mapping accession to list of ``InterProDomain`` objects.
        Proteins with no cache file are mapped to an empty list.
    """
    result: Dict[str, List[InterProDomain]] = {}
    for acc in accessions:
        cache_path = cache_dir / f"{acc}.json"
        if cache_path.exists():
            result[acc] = _load_cached(cache_path)
        else:
            result[acc] = []
    return result


def _compute_f1_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[int, int, int, float, float, float]:
    """Compute precision, recall, and F1 from binary label arrays.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        Tuple of ``(tp, fp, fn, precision, recall, f1)``.
        If precision + recall == 0, F1 is returned as 0.0.
    """
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return tp, fp, fn, precision, recall, f1
