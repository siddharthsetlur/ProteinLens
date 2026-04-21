#!/usr/bin/env python3
"""Standalone CATH enrichment analysis for SAE features.

Fetches CATH domain classifications for the top-activated proteins per
feature via the CATH FunFHMMer sequence search API, then computes
protein-level and residue-level F1 enrichment at each level of the
CATH hierarchy (C, CA, CAT, CATH).

Only fetches CATH for proteins in selection.json top lists (~14K unique),
not the full 50K protein set.

API workflow (async, one sequence per call):
  1. POST /search/by_funfhmmer  → task_id
  2. GET  /check/{task_id}      → poll until done
  3. GET  /results/{task_id}    → CATH hits with residue ranges

Usage:
    python scripts/compute_cath_enrichment.py feature_data_cluster/

Outputs (all under <data_dir>/cath_enrichment/):
    cache/{accession}.json   — slim per-protein CATH hits
    {feat_idx:04d}.json      — per-feature F1 at each CATH level
    summary.json             — quick-lookup summary
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

# ===================================================================
# CATH API
# ===================================================================

CATH_REST_URL = "https://www.cathdb.info/version/v4_3_0/api/rest/uniprot/{accession}"

CATH_LEVELS = ("C", "CA", "CAT", "CATH")


def _cath_label_at_level(cath_id: str, level: str) -> str:
    """Extract CATH label at a given hierarchy level.

    E.g. "3.90.1150.10" → C="3", CA="3.90", CAT="3.90.1150", CATH="3.90.1150.10"
    """
    parts = cath_id.split(".")
    n = CATH_LEVELS.index(level) + 1
    return ".".join(parts[:n])


def fetch_cath_for_protein(
    accession: str,
    cache_dir: Path,
    session,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """Look up CATH domains for a UniProt accession via the REST API.

    Returns cached result if available.  Each hit is:
        {"cath_id": "1.10.630.10", "query_start": 24, "query_end": 515,
         "description": "Cytochrome P450"}
    """
    import requests

    cache_path = cache_dir / f"{accession}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    hits: List[Dict[str, Any]] = []
    definitive = False
    for attempt in range(max_retries):
        try:
            resp = session.get(
                CATH_REST_URL.format(accession=accession),
                headers={"Accept": "application/json"},
                timeout=15,
            )
            if resp.status_code == 200:
                try:
                    hits = _parse_rest_response(resp.json())
                    definitive = True
                    break
                except ValueError:
                    # HTML/error page served with 200 — treat as transient
                    pass
            elif resp.status_code == 404:
                definitive = True
                break
        except (requests.Timeout, requests.ConnectionError):
            pass
        time.sleep(2 ** attempt)

    if definitive:
        _save_cache(cache_path, hits)
    return hits


def _parse_rest_response(data) -> List[Dict[str, Any]]:
    """Parse the CATH REST /uniprot/ response into slim hit dicts."""
    if not isinstance(data, list):
        return []
    hits = []
    for block in data:
        if not isinstance(block, dict) or block.get("abbr") != "CATHHMM":
            continue
        for entry in block.get("entries", []):
            cath_id = entry.get("resource_id", "")
            if not cath_id or len(cath_id.split(".")) < 4:
                continue
            desc = entry.get("option3", "")
            range_str = entry.get("option1", "")
            query_start, query_end = None, None
            if range_str and "-" in range_str:
                parts = range_str.split("-", 1)
                try:
                    query_start = int(parts[0])
                    query_end = int(parts[1])
                except ValueError:
                    pass
            hits.append({
                "cath_id": cath_id,
                "query_start": query_start,
                "query_end": query_end,
                "description": desc,
            })
    return hits


def _save_cache(cache_path: Path, hits: List[Dict[str, Any]]) -> None:
    with open(cache_path, "w") as f:
        json.dump(hits, f, separators=(",", ":"))


# ===================================================================
# CATH fetch orchestrator
# ===================================================================


def run_cath_fetch(
    accessions: List[str],
    cache_dir: Path,
    n_workers: int = 4,
) -> None:
    """Fetch CATH annotations for all accessions via REST API.

    Results go straight to cache files — no bulk dict returned.
    """
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)

    todo = [acc for acc in accessions if not (cache_dir / f"{acc}.json").exists()]
    already = len(accessions) - len(todo)
    print(f"[cath_fetch] {len(accessions)} proteins, {already} cached, {len(todo)} to fetch")

    if not todo:
        return

    pbar = tqdm(total=len(todo), desc="Fetching CATH")

    def _worker(accs: List[str]) -> None:
        s = requests.Session()
        for acc in accs:
            fetch_cath_for_protein(acc, cache_dir, s)
            pbar.update(1)

    chunks = [todo[i::n_workers] for i in range(n_workers)]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_worker, chunk) for chunk in chunks]
        for fut in as_completed(futs):
            fut.result()
    pbar.close()


def load_cath_cached(accession: str, cache_dir: Path) -> List[Dict[str, Any]]:
    """Load CATH hits from cache for a single protein."""
    cp = cache_dir / f"{accession}.json"
    if cp.exists():
        with open(cp) as f:
            return json.load(f)
    return []


# ===================================================================
# F1 enrichment (protein-level)
# ===================================================================


def compute_protein_level_f1(
    accessions_with_activations: List[Tuple[str, float]],
    protein_cath: Dict[str, List[Dict[str, Any]]],
    feat_max: float,
    n_threshold_steps: int = 50,
    min_proteins: int = 3,
    top_n: int = 10,
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute protein-level F1 at each CATH hierarchy level."""
    accessions = [acc for acc, _ in accessions_with_activations]
    activations = np.array([act for _, act in accessions_with_activations], dtype=np.float64)
    N = len(accessions)

    if N == 0 or feat_max == 0:
        return {level: [] for level in CATH_LEVELS}

    # Collect labels per level
    level_label_accs: Dict[str, Dict[str, Set[str]]] = {
        level: {} for level in CATH_LEVELS
    }
    label_desc: Dict[str, str] = {}

    for acc in accessions:
        for hit in protein_cath.get(acc, []):
            cath_id = hit.get("cath_id", "")
            desc = hit.get("description", "")
            if not cath_id or len(cath_id.split(".")) < 4:
                continue
            for level in CATH_LEVELS:
                label = _cath_label_at_level(cath_id, level)
                if label not in level_label_accs[level]:
                    level_label_accs[level][label] = set()
                level_label_accs[level][label].add(acc)
                if label not in label_desc or len(desc) > len(label_desc.get(label, "")):
                    label_desc[label] = desc

    thresholds = np.linspace(0, feat_max, n_threshold_steps + 1)
    y_pred_all = activations[np.newaxis, :] > thresholds[:, np.newaxis]
    pred_sums = y_pred_all.sum(axis=1).astype(np.float64)
    y_pred_float = y_pred_all.astype(np.float64)

    results_by_level: Dict[str, List[Dict[str, Any]]] = {}

    for level in CATH_LEVELS:
        eligible = {
            label: accs for label, accs in level_label_accs[level].items()
            if len(accs) >= min_proteins
        }
        if not eligible:
            results_by_level[level] = []
            continue

        codes = list(eligible.keys())
        K = len(codes)

        y_true_matrix = np.array(
            [[1.0 if acc in eligible[code] else 0.0 for acc in accessions] for code in codes],
            dtype=np.float64,
        )
        true_sums = y_true_matrix.sum(axis=1)
        tp = y_true_matrix @ y_pred_float.T
        fp = pred_sums[np.newaxis, :] - tp
        fn = true_sums[:, np.newaxis] - tp

        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            pr_sum = precision + recall
            f1 = np.where(pr_sum > 0, 2.0 * precision * recall / pr_sum, 0.0)

        best_t_idx = f1.argmax(axis=1)
        best_f1_vals = f1[np.arange(K), best_t_idx]

        level_results = []
        for i, code in enumerate(codes):
            bf1 = float(best_f1_vals[i])
            if bf1 == 0.0:
                continue
            ti = int(best_t_idx[i])
            t = float(thresholds[ti])
            level_results.append({
                "cath_label": code,
                "cath_level": level,
                "description": label_desc.get(code, ""),
                "best_f1": round(bf1, 4),
                "best_threshold": round(t, 4),
                "best_threshold_normalized": round(t / feat_max, 4),
                "precision_at_best": round(float(precision[i, ti]), 4),
                "recall_at_best": round(float(recall[i, ti]), 4),
                "n_proteins_with_label": int(true_sums[i]),
                "n_proteins_total": N,
                "n_true_positives": int(tp[i, ti]),
                "n_false_positives": int(fp[i, ti]),
                "n_false_negatives": int(fn[i, ti]),
            })

        level_results.sort(key=lambda r: r["best_f1"], reverse=True)
        results_by_level[level] = level_results[:top_n]

    return results_by_level


# ===================================================================
# F1 enrichment (residue-level)
# ===================================================================


def compute_residue_level_f1(
    protein_level_results: Dict[str, List[Dict[str, Any]]],
    protein_cath: Dict[str, List[Dict[str, Any]]],
    feat_idx: int,
    feat_max: float,
    npz_dir_map: Dict[str, Path],
    n_threshold_steps: int = 50,
    top_n_per_level: int = 5,
    io_executor: Optional[ThreadPoolExecutor] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute residue-level F1 for top CATH labels at each hierarchy level.

    Numerical equivalence is preserved vs. the prior serial implementation:
      * Per-target protein order follows ``protein_cath.items()`` insertion
        order (same as the original inner loop), so ``all_acts`` /
        ``all_labels`` concatenations are identical element-by-element.
      * Threshold grid (``pct`` + ``lin``, then ``np.unique``) and the
        TP/FP/FN formulas are unchanged.
      * Matching-range collection is identical: a hit contributes its
        ``(qs, qe)`` to ``(level, label)`` iff
        ``_cath_label_at_level(cath_id, level) == label``.

    Speedups vs. the original:
      * Each needed ``.npz`` is loaded at most once per feature (was once
        per matching (level, label) per protein).
      * Only the ``feat_idx`` column is retained (was a full
        ``seq_len × num_features`` array held in a cross-feature cache).
      * Loads run in parallel via ``io_executor`` (cephfs benefits a lot).
    """
    # Collect targets in original iteration order: (level, label) × top-N.
    targets: List[Tuple[str, str, str]] = []
    for level in CATH_LEVELS:
        for r in protein_level_results.get(level, [])[:top_n_per_level]:
            targets.append((level, r["cath_label"], r.get("description", "")))

    results_by_level: Dict[str, List[Dict[str, Any]]] = {level: [] for level in CATH_LEVELS}
    if not targets:
        return results_by_level

    target_keys: Set[Tuple[str, str]] = {(lvl, lbl) for (lvl, lbl, _) in targets}

    # Single pass over protein_cath: bucket matching ranges per target,
    # preserving protein_cath insertion order within each bucket.
    target_to_accs: Dict[Tuple[str, str], List[Tuple[str, List[Tuple[int, int]]]]] = {
        key: [] for key in target_keys
    }
    for acc, hits in protein_cath.items():
        per_key: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        for hit in hits:
            cath_id = hit.get("cath_id", "")
            if not cath_id or len(cath_id.split(".")) < 4:
                continue
            qs = hit.get("query_start")
            qe = hit.get("query_end")
            if qs is None or qe is None:
                continue
            qs_i, qe_i = int(qs), int(qe)
            for level in CATH_LEVELS:
                key = (level, _cath_label_at_level(cath_id, level))
                if key in target_keys:
                    per_key.setdefault(key, []).append((qs_i, qe_i))
        for key, ranges in per_key.items():
            target_to_accs[key].append((acc, ranges))

    # Gather unique accessions needed and load only column feat_idx, in parallel.
    needed_accs: Set[str] = set()
    for lst in target_to_accs.values():
        for acc, _ in lst:
            needed_accs.add(acc)

    def _load_col(acc: str) -> Tuple[str, Optional[np.ndarray]]:
        if acc not in npz_dir_map:
            return acc, None
        try:
            with np.load(npz_dir_map[acc] / f"{acc}.npz") as npz:
                arr = npz["activations"]
                if feat_idx >= arr.shape[1]:
                    return acc, None
                col = arr[:, feat_idx].copy()
        except (EOFError, OSError, KeyError):
            return acc, None
        return acc, col

    residue_cols: Dict[str, Optional[np.ndarray]] = {}
    if needed_accs:
        if io_executor is not None:
            for acc, col in io_executor.map(_load_col, needed_accs):
                residue_cols[acc] = col
        else:
            for acc in needed_accs:
                a, col = _load_col(acc)
                residue_cols[a] = col

    # Compute F1 per target with identical math to the original.
    for (level, target_label, description) in targets:
        prot_list = target_to_accs.get((level, target_label), [])

        all_acts: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        n_proteins_used = 0

        for acc, matching_ranges in prot_list:
            residue_acts = residue_cols.get(acc)
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
        results_by_level[level].append({
            "cath_label": target_label,
            "cath_level": level,
            "description": description,
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

    for level in CATH_LEVELS:
        results_by_level[level].sort(key=lambda r: r["best_f1"], reverse=True)

    return results_by_level


# ===================================================================
# Main
# ===================================================================


def main():
    parser = argparse.ArgumentParser(description="CATH enrichment for SAE features")
    parser.add_argument("data_dir", type=Path, help="Feature data directory")
    parser.add_argument("--workers", type=int, default=4, help="Parallel CATH API workers")
    parser.add_argument(
        "--residue-workers",
        type=int,
        default=16,
        help="Parallel threads for residue-level .npz loads (cephfs benefits from high parallelism).",
    )
    parser.add_argument("--threshold-steps", type=int, default=50)
    parser.add_argument("--min-proteins", type=int, default=3)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to W&B")
    args = parser.parse_args()

    # ── Optional W&B init ──
    if args.wandb:
        import wandb
        wandb.init(project="proteinlens-pipeline", name="cath-enrichment", tags=["cath"])

    data_dir = args.data_dir
    cath_dir = data_dir / "cath_enrichment"
    cache_dir = cath_dir / "cache"
    cath_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Load upstream data ──
    print("[cath] Loading data...")

    with open(data_dir / "selection.json") as f:
        selection = json.load(f)

    with open(data_dir / "pipeline_state.json") as f:
        pipeline_state = json.load(f)
    n_proteins = pipeline_state["total_proteins"]
    acc_index = pipeline_state["accession_index"]

    global_max = np.load(data_dir / "feature_max_activations.npy")
    num_features = len(global_max)

    protein_maxes = np.memmap(
        data_dir / "protein_feature_maxes.npy",
        dtype="float32",
        mode="r",
        shape=(n_proteins, num_features),
    )

    # ── Collect unique top proteins across all features ──
    # Only these ~14K need CATH fetches.  They double as negatives for
    # other features (a protein that's top-20 for feature A is typically
    # low-activation for feature B), so no extra API calls are needed.
    per_feature = selection["per_feature"]
    all_top_accessions: Set[str] = set()
    for feat_data in per_feature.values():
        all_top_accessions.update(feat_data.get("top", []))
    print(f"[cath] {len(all_top_accessions)} unique proteins across all feature top lists.")

    # ── Fetch CATH only for top proteins ──
    run_cath_fetch(sorted(all_top_accessions), cache_dir, n_workers=args.workers)

    # ── Preload all CATH cache into memory ──
    print("[cath] Preloading CATH cache into memory...")
    all_cath: Dict[str, List[Dict[str, Any]]] = {}
    cache_files = list(cache_dir.glob("*.json"))
    for p in tqdm(cache_files, desc="Loading CATH cache"):
        acc = p.stem
        with open(p) as f:
            all_cath[acc] = json.load(f)
    print(f"[cath] Loaded {len(all_cath)} CATH entries into memory.")

    n_with_cath = sum(
        1 for acc in all_top_accessions
        if all_cath.get(acc)
    )
    print(f"[cath] {n_with_cath}/{len(all_top_accessions)} proteins have CATH hits.")

    if args.wandb:
        wandb.log({
            "cath_fetch/total_proteins": len(all_top_accessions),
            "cath_fetch/with_hits": n_with_cath,
            "cath_fetch/hit_rate": n_with_cath / max(len(all_top_accessions), 1),
        })

    # ── Also load interpro_selection for bin proteins (negatives) ──
    with open(data_dir / "interpro_selection.json") as f:
        interpro_selection = json.load(f)

    # ── Pre-glob .npz files for residue-level analysis ──
    npz_dir_map: Dict[str, Path] = {}
    for d_name in ("residue_activations", "interpro_residue_activations"):
        d = data_dir / d_name
        if d.exists():
            for p in d.glob("*.npz"):
                if p.stem not in npz_dir_map:
                    npz_dir_map[p.stem] = d
    print(f"[cath] {len(npz_dir_map)} .npz files available for residue-level analysis.")

    # ── Check already-computed features ──
    already_computed: Set[int] = set()
    for p in cath_dir.glob("*.json"):
        if p.stem in ("summary",):
            continue
        try:
            already_computed.add(int(p.stem))
        except ValueError:
            pass
    print(f"[cath] {len(already_computed)} features already computed, resuming.")

    # ── Load existing summaries for resumed features ──
    summary: Dict[str, Dict[str, Any]] = {}
    n_analyzed = 0
    n_skipped = 0

    for feat_idx in sorted(already_computed):
        out_path = cath_dir / f"{feat_idx:04d}.json"
        try:
            with open(out_path) as f:
                existing = json.load(f)
            summary[str(feat_idx)] = existing.get("summary", {})
        except (json.JSONDecodeError, OSError):
            already_computed.discard(feat_idx)

    # ── Shared thread pool for parallel residue-level .npz loads ──
    io_executor = ThreadPoolExecutor(max_workers=args.residue_workers)

    # ── Process each feature ──
    for feat_idx in tqdm(range(num_features), desc="CATH enrichment"):
        if feat_idx in already_computed:
            continue

        feat_max = float(global_max[feat_idx])
        if feat_max == 0:
            n_skipped += 1
            continue

        feat_key = str(feat_idx)
        feat_top = per_feature.get(feat_key, {}).get("top", [])
        feat_bins = interpro_selection["per_feature"].get(feat_key, {}).get("bins", {})

        if not feat_top:
            n_skipped += 1
            continue

        # Collect top proteins (positives) + bin proteins that have
        # cached CATH data (negatives — they're top proteins for other
        # features, so CATH was already fetched for them).
        seen: Set[str] = set()
        accessions_with_activations: List[Tuple[str, float]] = []

        def _add_acc(acc: str) -> None:
            if acc in seen:
                return
            seen.add(acc)
            if acc in acc_index:
                activation = float(protein_maxes[int(acc_index[acc]), feat_idx])
            else:
                activation = 0.0
            accessions_with_activations.append((acc, activation))

        for acc in feat_top:
            _add_acc(acc)
        for bin_accs in feat_bins.values():
            for acc in bin_accs:
                if acc in all_top_accessions:  # only if CATH was fetched
                    _add_acc(acc)

        # Load CATH from preloaded memory
        feat_cath: Dict[str, List[Dict[str, Any]]] = {}
        for acc, _ in accessions_with_activations:
            feat_cath[acc] = all_cath.get(acc, [])

        n_with = sum(1 for hits in feat_cath.values() if hits)
        if n_with < args.min_proteins:
            n_skipped += 1
            continue

        # ── Protein-level F1 ──
        prot_results = compute_protein_level_f1(
            accessions_with_activations=accessions_with_activations,
            protein_cath=feat_cath,
            feat_max=feat_max,
            n_threshold_steps=args.threshold_steps,
            min_proteins=args.min_proteins,
        )

        # ── Residue-level F1 ──
        res_results = compute_residue_level_f1(
            protein_level_results=prot_results,
            protein_cath=feat_cath,
            feat_idx=feat_idx,
            feat_max=feat_max,
            npz_dir_map=npz_dir_map,
            io_executor=io_executor,
        )

        # ── Summary ──
        feat_summary: Dict[str, Any] = {}
        for level in CATH_LEVELS:
            pr = prot_results.get(level, [])
            rr = res_results.get(level, [])
            feat_summary[level] = {
                "top_protein_label": pr[0]["cath_label"] if pr else None,
                "top_protein_f1": pr[0]["best_f1"] if pr else None,
                "top_protein_description": pr[0]["description"] if pr else None,
                "top_residue_label": rr[0]["cath_label"] if rr else None,
                "top_residue_f1": rr[0]["best_f1"] if rr else None,
            }

        out_data = {
            "feature_id": feat_idx,
            "feature_max_activation": feat_max,
            "n_proteins_evaluated": len(accessions_with_activations),
            "n_proteins_with_cath": n_with,
            "protein_level": prot_results,
            "residue_level": res_results,
            "summary": feat_summary,
        }
        with open(cath_dir / f"{feat_idx:04d}.json", "w") as f:
            json.dump(out_data, f, indent=2)

        summary[feat_key] = feat_summary
        n_analyzed += 1

    io_executor.shutdown(wait=True)

    # ── Write summary ──
    summary_data = {
        "n_features_analyzed": n_analyzed + len(already_computed),
        "n_features_skipped": n_skipped,
        "n_proteins_with_cath": n_with_cath,
        "n_proteins_total": len(all_top_accessions),
        "features": summary,
    }
    with open(cath_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print(
        f"[cath] Done. Analyzed {n_analyzed} features "
        f"(+{len(already_computed)} resumed), skipped {n_skipped}. "
        f"Results in {cath_dir}/"
    )

    if args.wandb:
        wandb.log({
            "cath_enrichment/analyzed": n_analyzed,
            "cath_enrichment/resumed": len(already_computed),
            "cath_enrichment/skipped": n_skipped,
            "cath_enrichment/n_proteins_with_cath": n_with_cath,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
