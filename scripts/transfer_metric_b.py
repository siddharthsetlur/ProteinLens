#!/usr/bin/env python3
"""Metric B — predictive transfer of the SwissProt-trained GBM to NMPFams.

For every SAE feature with q<0.05 on the geometric annotation method (m7) AND
at least one NMPFam hit, take the GBM trained on SwissProt residues and apply
it to NMPFam residues (already done in run_nmpfam.py phase 4 — the
``geom_prob_profile`` arrays are stored). Compute PR-AUC of those probabilities
against the actual SAE activation pattern on NMPFams. The trivial-classifier
null is the prevalence of active residues; we report PR-AUC / prevalence and
the fraction of features beating 2× prevalence.

If geometry were SwissProt-specific the ratio collapses to 1.

Output:
    {analysis_dir}/transfer_metrics/metric_B.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)
Q_SIG = 0.05


def _bh(pvals: list[float | None]) -> list[float | None]:
    n = len(pvals)
    idx = [(i, p) for i, p in enumerate(pvals) if p is not None]
    if not idx:
        return [None] * n
    idx.sort(key=lambda x: x[1])
    m = len(idx)
    out: list[float | None] = [None] * n
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        orig, p = idx[rank]
        q = min(1.0, p * m / (rank + 1))
        if q < running_min:
            running_min = q
        out[orig] = running_min
    return out


def load_geometry_qvalues(analysis_dir: Path) -> dict[int, float]:
    """BH q-values for geometry_prauc, mirroring index_builder."""
    pn_dir = analysis_dir / "permutation_null"
    pairs: list[tuple[int, float | None]] = []
    for p in sorted(pn_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        fid = int(d["feature_id"])
        pv = (d.get("p_values") or {}).get("geometry_prauc")
        pairs.append((fid, float(pv) if pv is not None else None))
    pvals = [p for _, p in pairs]
    qvals = _bh(pvals)
    return {fid: q for (fid, _), q in zip(pairs, qvals) if q is not None}


def metric_b_for_feature(payload: dict) -> dict | None:
    """Concatenate per-residue arrays across NMPFam hits and compute PR-AUC."""
    thr = payload.get("activation_threshold_sae")
    hits = payload.get("nmpfam_hits", [])
    if not hits or thr is None:
        return None

    y_true_chunks, y_score_chunks = [], []
    for h in hits:
        sae = np.asarray(h.get("sae_activation_profile", []), dtype=np.float32)
        geom = np.asarray(h.get("geom_prob_profile", []), dtype=np.float32)
        n = min(len(sae), len(geom))
        if n == 0:
            continue
        y_true_chunks.append(sae[:n] >= thr)
        y_score_chunks.append(geom[:n])
    if not y_true_chunks:
        return None

    y_true = np.concatenate(y_true_chunks)
    y_score = np.concatenate(y_score_chunks)
    n_residues = int(len(y_true))
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return {
            "n_residues": n_residues,
            "n_active": 0,
            "prevalence": 0.0,
            "pr_auc": None,
            "ratio": None,
            "beats_2x": False,
            "n_hits": len(hits),
            "note": "no NMPFam residues exceed the SAE activation threshold",
        }

    prevalence = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true.astype(int), y_score))
    ratio = pr_auc / max(prevalence, 1e-12)
    return {
        "n_residues": n_residues,
        "n_active": n_pos,
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "ratio": ratio,
        "beats_2x": bool(ratio >= 2.0),
        "n_hits": len(hits),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-dir", type=Path, required=True)
    args = ap.parse_args()

    analysis = args.analysis_dir.resolve()
    if not analysis.is_dir():
        raise SystemExit(f"Not a directory: {analysis}")

    out_dir = analysis / "transfer_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading geometry q-values …")
    qvals = load_geometry_qvalues(analysis)
    sig_set = {fid for fid, q in qvals.items() if q is not None and q < Q_SIG}
    logger.info("Geometry q<0.05: %d / %d features", len(sig_set), len(qvals))

    nmp_dir = analysis / "nmpfam" / "nmpfam_enrichment"
    nmp_files = sorted(
        p for p in nmp_dir.glob("*.json")
        if p.stem.isdigit() and p.name != "summary.json"
    )
    logger.info("NMPFam-enrichment files: %d", len(nmp_files))

    per_feature: dict[int, dict] = {}
    n_total = 0
    n_with_metric = 0
    for path in nmp_files:
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        fid = int(d["feature_id"])
        if fid not in sig_set:
            continue
        n_total += 1
        m = metric_b_for_feature(d)
        if m is None:
            continue
        m["geometry_q"] = qvals.get(fid)
        per_feature[fid] = m
        if m.get("pr_auc") is not None:
            n_with_metric += 1

    # Aggregate stats
    ratios = [v["ratio"] for v in per_feature.values() if v.get("ratio") is not None]
    pr_aucs = [v["pr_auc"] for v in per_feature.values() if v.get("pr_auc") is not None]
    prevs = [v["prevalence"] for v in per_feature.values() if v.get("pr_auc") is not None]
    n_beats_2x = sum(1 for r in ratios if r >= 2.0)
    n_beats_5x = sum(1 for r in ratios if r >= 5.0)
    n_beats_10x = sum(1 for r in ratios if r >= 10.0)

    summary = {
        "n_geometry_q_significant_features": len(sig_set),
        "n_features_with_nmpfam_hits": n_total,
        "n_features_with_pr_auc": n_with_metric,
        "median_pr_auc": float(np.median(pr_aucs)) if pr_aucs else None,
        "median_prevalence": float(np.median(prevs)) if prevs else None,
        "median_ratio": float(np.median(ratios)) if ratios else None,
        "mean_ratio": float(np.mean(ratios)) if ratios else None,
        "frac_beats_1x": (sum(1 for r in ratios if r > 1.0) / len(ratios)) if ratios else None,
        "frac_beats_2x": (n_beats_2x / len(ratios)) if ratios else None,
        "frac_beats_5x": (n_beats_5x / len(ratios)) if ratios else None,
        "frac_beats_10x": (n_beats_10x / len(ratios)) if ratios else None,
        "ratio_quartiles": (
            np.quantile(ratios, [0.25, 0.5, 0.75]).tolist() if ratios else None
        ),
    }

    out = {
        "metric": "B",
        "description": (
            "Predictive transfer of SwissProt-trained GBM to NMPFams. "
            "For each feature with q<0.05 on geometric annotation (m7) and "
            "≥1 NMPFam hit, PR-AUC of GBM-predicted probabilities against "
            "actual SAE activations on NMPFam residues, with prevalence "
            "(trivial classifier) as the null. Reported per feature: "
            "prevalence, PR-AUC, ratio = PR-AUC / prevalence."
        ),
        "summary": summary,
        "per_feature": per_feature,
    }
    (out_dir / "metric_B.json").write_text(json.dumps(out, indent=2))
    logger.info(
        "Wrote %d per-feature records. median ratio=%.2f, frac>=2x=%.2f, frac>=5x=%.2f",
        len(per_feature),
        summary["median_ratio"] or 0.0,
        summary["frac_beats_2x"] or 0.0,
        summary["frac_beats_5x"] or 0.0,
    )


if __name__ == "__main__":
    main()
