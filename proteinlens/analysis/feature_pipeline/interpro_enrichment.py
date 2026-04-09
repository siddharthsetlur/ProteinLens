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

    Supports resumption: features with existing output JSON files are
    skipped.  Their summary entries are loaded from the existing files.

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

    # ── Resolve directory paths once (each config property calls mkdir) ──
    residue_act_dir = config.residue_activations_dir
    interpro_act_dir = config.interpro_residue_activations_dir
    interpro_cache_dir = config.interpro_cache_dir
    enrichment_dir = config.interpro_enrichment_dir

    # ── Pre-glob available .npz files once ──
    # Record which directory each accession lives in for direct loading.
    npz_dir_map: Dict[str, Path] = {}
    for d in (residue_act_dir, interpro_act_dir):
        if d.exists():
            for p in d.glob("*.npz"):
                # First directory wins (Stage 3 preferred over Stage 5a)
                if p.stem not in npz_dir_map:
                    npz_dir_map[p.stem] = d
    print(f"[interpro_enrichment] {len(npz_dir_map)} .npz files available.")

    # ── Preload ALL InterPro annotations into memory (one-time I/O) ──
    # Each JSON is ~1-5 KB; 50K files ≈ 50-250 MB in memory.
    # This replaces ~2.8M per-feature file reads with a single pass.
    all_annotations: Dict[str, List[InterProDomain]] = {}
    interpro_cached: Set[str] = set()
    if interpro_cache_dir.exists():
        cache_files = list(interpro_cache_dir.glob("*.json"))
        print(f"[interpro_enrichment] Preloading {len(cache_files)} InterPro annotations...")
        for p in tqdm(cache_files, desc="Loading InterPro cache", leave=False):
            acc = p.stem
            interpro_cached.add(acc)
            try:
                all_annotations[acc] = _load_cached(p)
            except (json.JSONDecodeError, OSError):
                all_annotations[acc] = []
    print(f"[interpro_enrichment] {len(all_annotations)} annotations loaded into memory.")

    # ── Pre-glob existing enrichment outputs for resume ──
    already_computed: Set[int] = set()
    if enrichment_dir.exists():
        for p in enrichment_dir.glob("*.json"):
            if p.stem == "summary":
                continue
            try:
                already_computed.add(int(p.stem))
            except ValueError:
                pass
    print(f"[interpro_enrichment] {len(already_computed)} features already computed.")

    # ── NPZ LRU cache (shared across features) ──
    _npz_cache: Dict[str, Optional[np.ndarray]] = {}
    MAX_NPZ_CACHE = 500

    # ── Rebuild summary from already-computed features ──
    n_resumed = 0
    summary_features: Dict[str, Dict[str, Any]] = {}
    if already_computed:
        for feat_idx in sorted(already_computed):
            feat_key = str(feat_idx)
            out_path = enrichment_dir / f"{feat_idx:04d}.json"
            try:
                with open(out_path, "r") as f:
                    existing = json.load(f)
                prot_results = existing.get("protein_level", [])
                res_results = existing.get("residue_level", [])
                if prot_results:
                    top_prot = prot_results[0]
                    entry: Dict[str, Any] = {
                        "top_protein_annotation": top_prot["annotation_code"],
                        "top_protein_annotation_name": top_prot["annotation_name"],
                        "top_protein_f1": top_prot["best_f1"],
                        "top_residue_annotation": None,
                        "top_residue_f1": None,
                    }
                    if res_results:
                        top_res = res_results[0]
                        entry["top_residue_annotation"] = top_res["annotation_code"]
                        entry["top_residue_f1"] = top_res["best_f1"]
                    summary_features[feat_key] = entry
                n_resumed += 1
            except (json.JSONDecodeError, KeyError, OSError):
                already_computed.discard(feat_idx)
        print(f"[interpro_enrichment] Resumed {n_resumed} features from previous run.")

    # ── Process each feature ──
    n_analyzed = 0
    n_skipped = 0

    for feat_idx in tqdm(range(num_features), desc="InterPro enrichment"):
        if feat_idx in already_computed:
            continue

        feat_key = str(feat_idx)
        feat_max = float(global_max[feat_idx])

        if feat_max == 0:
            n_skipped += 1
            continue

        feat_selection = interpro_selection["per_feature"].get(feat_key, {})
        bins = feat_selection.get("bins", {})

        accessions_with_activations = _collect_accessions_with_activations(
            bins, acc_index, protein_maxes, feat_idx
        )

        if len(accessions_with_activations) < config.interpro_min_proteins:
            n_skipped += 1
            continue

        # Look up annotations from the preloaded dict (zero I/O)
        protein_annotations = {
            acc: all_annotations.get(acc, [])
            for acc, _ in accessions_with_activations
        }

        n_with_annotations = sum(
            1 for acc, _ in accessions_with_activations
            if protein_annotations.get(acc)
        )

        if n_with_annotations < config.interpro_min_proteins:
            n_skipped += 1
            continue

        # ── Protein-level F1 (vectorized) ──
        protein_level_results = _compute_protein_level_f1(
            accessions_with_activations=accessions_with_activations,
            protein_annotations=protein_annotations,
            feat_max=feat_max,
            n_threshold_steps=config.interpro_f1_threshold_steps,
            min_proteins=config.interpro_min_proteins,
            top_n=config.interpro_top_annotations,
        )

        # ── Residue-level F1 ──
        residue_level_results = _compute_residue_level_f1(
            protein_level_results=protein_level_results,
            protein_annotations=protein_annotations,
            feat_idx=feat_idx,
            feat_max=feat_max,
            n_threshold_steps=config.interpro_f1_threshold_steps,
            npz_cache=_npz_cache,
            max_npz_cache=MAX_NPZ_CACHE,
            npz_dir_map=npz_dir_map,
        )

        # ── Count annotations that met the min_proteins threshold ──
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

        # ── Write per-feature JSON ──
        enrichment_data = {
            "feature_id": feat_idx,
            "feature_max_activation": feat_max,
            "n_proteins_evaluated": len(accessions_with_activations),
            "n_proteins_with_annotations": n_with_annotations,
            "n_unique_annotations_tested": n_annotations_tested,
            "protein_level": protein_level_results,
            "residue_level": residue_level_results,
        }

        out_path = enrichment_dir / f"{feat_idx:04d}.json"
        with open(out_path, "w") as f:
            json.dump(enrichment_data, f, indent=2)

        n_analyzed += 1

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

    # ── Write summary JSON ──
    summary = {
        "n_features_analyzed": n_analyzed + n_resumed,
        "n_features_skipped": n_skipped,
        "features": summary_features,
    }
    summary_path = enrichment_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if n_resumed:
        print(f"[interpro_enrichment] Resumed: {n_resumed} features already computed.")
    print(
        f"[interpro_enrichment] Analyzed {n_analyzed} features, "
        f"skipped {n_skipped}. "
        f"Wrote results to {enrichment_dir}/"
    )
    from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog

    wlog({
        "interpro_enrichment/analyzed": n_analyzed,
        "interpro_enrichment/resumed": n_resumed,
        "interpro_enrichment/skipped": n_skipped,
    })


# ===================================================================
# Protein-level F1 — fully vectorized (checklist item 4.2)
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

    Fully vectorized: broadcasts all thresholds across all proteins in
    numpy, then computes tp/fp/fn for every (code, threshold) pair via
    matrix multiplication.  No Python loops over thresholds.

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
    accessions = [acc for acc, _ in accessions_with_activations]
    activations = np.array(
        [act for _, act in accessions_with_activations], dtype=np.float64
    )
    N = len(accessions)

    # Collect unique annotation codes and their protein sets
    annotation_proteins: Dict[str, Set[str]] = {}
    annotation_meta: Dict[str, Dict[str, str]] = {}

    for acc in accessions:
        for domain in protein_annotations.get(acc, []):
            code = domain.interpro_accession
            if code not in annotation_proteins:
                annotation_proteins[code] = set()
                annotation_meta[code] = {
                    "annotation_name": domain.interpro_name,
                    "annotation_type": domain.type,
                    "member_db": domain.member_db,
                    "member_accession": domain.member_accession,
                }
            annotation_proteins[code].add(acc)

    eligible_codes = [
        code for code, proteins in annotation_proteins.items()
        if len(proteins) >= min_proteins
    ]

    if not eligible_codes:
        return []

    # Thresholds: (T,)
    thresholds = np.linspace(0, feat_max, n_threshold_steps + 1)
    T = len(thresholds)

    # y_pred for all thresholds at once: (T, N) bool
    # y_pred[t, n] = activations[n] > thresholds[t]
    y_pred_all = activations[np.newaxis, :] > thresholds[:, np.newaxis]  # (T, N)

    # Precompute sum of predictions per threshold: (T,)
    pred_sums = y_pred_all.sum(axis=1).astype(np.float64)

    # Convert y_pred to float for matrix multiply
    y_pred_float = y_pred_all.astype(np.float64)  # (T, N)

    # Build y_true matrix for all eligible codes: (K, N)
    acc_set_list = [annotation_proteins[code] for code in eligible_codes]
    y_true_matrix = np.array(
        [[1.0 if acc in s else 0.0 for acc in accessions] for s in acc_set_list],
        dtype=np.float64,
    )  # (K, N)

    # true_sums[k] = number of proteins with annotation k
    true_sums = y_true_matrix.sum(axis=1)  # (K,)

    # tp[k, t] = sum(y_true[k] & y_pred[t]) via matmul
    tp = y_true_matrix @ y_pred_float.T  # (K, T)

    # fp[k, t] = pred_sums[t] - tp[k, t]
    fp = pred_sums[np.newaxis, :] - tp  # (K, T)

    # fn[k, t] = true_sums[k] - tp[k, t]
    fn = true_sums[:, np.newaxis] - tp  # (K, T)

    # Precision, recall, F1 with zero-safe division
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        pr_sum = precision + recall
        f1 = np.where(pr_sum > 0, 2.0 * precision * recall / pr_sum, 0.0)

    # For each code, find the threshold with the best F1
    best_t_idx = f1.argmax(axis=1)  # (K,)
    best_f1_vals = f1[np.arange(len(eligible_codes)), best_t_idx]

    results: List[Dict[str, Any]] = []
    for i, code in enumerate(eligible_codes):
        bf1 = float(best_f1_vals[i])
        if bf1 == 0.0:
            continue
        ti = int(best_t_idx[i])
        t = float(thresholds[ti])
        meta = annotation_meta[code]
        n_with = int(true_sums[i])
        prec = float(precision[i, ti])
        rec = float(recall[i, ti])
        tp_val = int(tp[i, ti])
        fp_val = int(fp[i, ti])
        fn_val = int(fn[i, ti])

        results.append({
            "annotation_code": code,
            "annotation_name": meta["annotation_name"],
            "annotation_type": meta["annotation_type"],
            "member_db": meta["member_db"],
            "member_accession": meta["member_accession"],
            "best_f1": round(bf1, 4),
            "best_threshold": round(t, 4),
            "best_threshold_normalized": round(
                t / feat_max if feat_max > 0 else 0.0, 4
            ),
            "precision_at_best": round(prec, 4),
            "recall_at_best": round(rec, 4),
            "n_proteins_with_annotation": n_with,
            "n_proteins_without_annotation": N - n_with,
            "n_true_positives": tp_val,
            "n_false_positives": fp_val,
            "n_false_negatives": fn_val,
            "interpretation": (
                f"Proteins with activation > {round(t, 2)} "
                f"({round(t / feat_max * 100 if feat_max > 0 else 0, 0):.0f}% of max) "
                f"are predicted by annotation {code} with F1={round(bf1, 2)}"
            ),
        })

    results.sort(key=lambda r: r["best_f1"], reverse=True)

    if results:
        best_f1_overall = results[0]["best_f1"]
        cutoff = best_f1_overall - 0.05
        n_keep = min(top_n, len(results))
        while n_keep < len(results) and results[n_keep]["best_f1"] >= cutoff:
            n_keep += 1
        results = results[:n_keep]

    return results


# ===================================================================
# Residue-level F1 — vectorized threshold sweep (checklist item 4.3)
# ===================================================================


def _compute_residue_level_f1(
    protein_level_results: List[Dict[str, Any]],
    protein_annotations: Dict[str, List[InterProDomain]],
    feat_idx: int,
    feat_max: float,
    n_threshold_steps: int,
    npz_cache: Dict[str, Optional[np.ndarray]],
    max_npz_cache: int,
    npz_dir_map: Dict[str, Path],
) -> List[Dict[str, Any]]:
    """Compute residue-level F1 for the top protein-level annotations.

    For each annotation that scored well at the protein level, we ask:
    "does a residue being inside a domain boundary predict high activation
    at that position?"

    Threshold sweep is vectorized via numpy broadcasting.

    Args:
        protein_level_results: Output of ``_compute_protein_level_f1``.
        protein_annotations: Dict mapping accession to InterProDomain list.
        feat_idx: Feature index for extracting the correct column from
            per-residue activation matrices.
        feat_max: Global max activation for normalising thresholds.
        n_threshold_steps: Number of threshold steps for the sweep.
        npz_cache: Mutable LRU cache of loaded activation arrays
            (shared across features to avoid repeated cephfs reads).
        max_npz_cache: Maximum entries in the npz cache.
        npz_dir_map: Pre-built mapping of accession -> directory Path.

    Returns:
        List of residue-level result dicts, one per annotation tested.
    """
    results: List[Dict[str, Any]] = []

    for prot_result in protein_level_results:
        code = prot_result["annotation_code"]

        all_activations_list: List[np.ndarray] = []
        all_labels_list: List[np.ndarray] = []
        n_proteins_used = 0

        for acc, domains in protein_annotations.items():
            matching_domains = [
                d for d in domains if d.interpro_accession == code
            ]
            if not matching_domains:
                continue

            residue_acts = _load_residue_cached(
                acc, npz_cache, max_npz_cache, npz_dir_map,
            )
            if residue_acts is None:
                continue

            if feat_idx >= residue_acts.shape[1]:
                continue

            feat_acts = residue_acts[:, feat_idx]
            seq_len = len(feat_acts)

            labels = np.zeros(seq_len, dtype=np.int32)
            for d in matching_domains:
                start_0 = max(0, d.start - 1)
                end_0 = min(seq_len - 1, d.end - 1)
                labels[start_0 : end_0 + 1] = 1

            all_activations_list.append(feat_acts)
            all_labels_list.append(labels)
            n_proteins_used += 1

        if n_proteins_used == 0:
            continue

        all_activations = np.concatenate(all_activations_list)
        all_labels = np.concatenate(all_labels_list)
        n_total_residues = len(all_labels)
        n_in_domain = int(all_labels.sum())

        if n_in_domain == 0 or n_in_domain == n_total_residues:
            continue

        nonzero_acts = all_activations[all_activations > 0]
        if len(nonzero_acts) == 0:
            continue

        percentile_thresholds = np.percentile(
            nonzero_acts,
            np.linspace(0, 100, n_threshold_steps),
        )
        linear_thresholds = np.linspace(0, feat_max, n_threshold_steps)
        thresholds = np.unique(
            np.concatenate([percentile_thresholds, linear_thresholds])
        )

        # Vectorized threshold sweep: (T, R) bool via broadcasting
        # all_activations: (R,), thresholds: (T,)
        y_pred_all = all_activations[np.newaxis, :] > thresholds[:, np.newaxis]  # (T, R)

        y_true = all_labels.astype(np.float64)  # (R,)
        y_true_neg = 1.0 - y_true

        # tp[t] = sum(y_true & y_pred[t])
        tp = y_pred_all.astype(np.float64) @ y_true           # (T,)
        fp = y_pred_all.astype(np.float64) @ y_true_neg        # (T,)
        fn = float(n_in_domain) - tp                            # (T,)

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
        results.append({
            "annotation_code": code,
            "annotation_name": prot_result["annotation_name"],
            "member_db": prot_result["member_db"],
            "member_accession": prot_result["member_accession"],
            "best_f1": round(best_f1, 4),
            "best_threshold": round(t, 4),
            "best_threshold_normalized": round(
                t / feat_max if feat_max > 0 else 0.0, 4
            ),
            "precision_at_best": round(float(precision[best_idx]), 4),
            "recall_at_best": round(float(recall[best_idx]), 4),
            "n_proteins_used": n_proteins_used,
            "n_total_residues": n_total_residues,
            "n_residues_in_domain": n_in_domain,
            "n_true_positives": int(tp[best_idx]),
            "n_false_positives": int(fp[best_idx]),
            "n_false_negatives": int(fn[best_idx]),
            "interpretation": (
                f"Residues with activation > {round(t, 2)} "
                f"({round(t / feat_max * 100 if feat_max > 0 else 0, 0):.0f}% of max) "
                f"overlap with {code} domains with F1={round(best_f1, 2)}"
            ),
        })

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


def _load_residue_cached(
    accession: str,
    npz_cache: Dict[str, Optional[np.ndarray]],
    max_npz_cache: int,
    npz_dir_map: Dict[str, Path],
) -> Optional[np.ndarray]:
    """Load per-residue activations with LRU caching — zero cephfs metadata ops.

    Uses ``npz_dir_map`` (built from the startup glob) to go directly
    to the file without any ``exists()`` or ``mkdir`` calls.

    Args:
        accession: UniProt accession string.
        npz_cache: Mutable LRU cache dict (accession -> array or None).
        max_npz_cache: Max entries before FIFO eviction.
        npz_dir_map: Pre-built mapping of accession -> directory Path.

    Returns:
        Numpy array of shape ``(seq_len, num_features)``, or ``None``.
    """
    if accession not in npz_dir_map:
        return None

    if accession in npz_cache:
        return npz_cache[accession]

    npz_path = npz_dir_map[accession] / f"{accession}.npz"
    try:
        arr = np.load(npz_path)["activations"]
    except (EOFError, OSError, KeyError):
        arr = None

    if len(npz_cache) >= max_npz_cache:
        oldest_key = next(iter(npz_cache))
        del npz_cache[oldest_key]

    npz_cache[accession] = arr
    return arr


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

            if acc in acc_index:
                row = int(acc_index[acc])
                activation = float(protein_maxes[row, feat_idx])
            else:
                activation = 0.0

            result.append((acc, activation))

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
