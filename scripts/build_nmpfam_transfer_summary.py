"""Build the per-layer Table 4 + filtered transfer-feature list for the
GeoPedia case study 03 ("Geometric annotation captures transferable
annotation to metagenomic proteins").

For every SAE feature with at least one NMPFam hit, we compute the
per-metagenomic-protein PR-AUC of the pre-trained Swiss-Prot GBM
(geom_prob_profile vs. sae_activation_profile > activation_threshold_sae).
We then aggregate to a single record per feature:

    {feature_id, n_hits, n_strong, max_prauc, median_prauc,
     sequences_annotated, geometry_padj, top_hits[25]}

The ``features`` list in the output is gated to match Table 4 column 3:
*geometry q-significant AND median PR-AUC > 0.5*. The ``table4`` block
reproduces all five columns of the paper table for this layer.

Reads:
  {analysis_dir}/geometry_primary_analysis.json
  {analysis_dir}/nmpfam/nmpfam_enrichment/*.json

Writes:
  {analysis_dir}/nmpfam_transfer_summary.json

Usage:
    python scripts/build_nmpfam_transfer_summary.py --analysis-dir analysis/l4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

Q_SIG = 0.05          # BH significance gate
PRAUC_GATE = 0.5      # Table 4 column 3 threshold + per-hit "strong" cutoff


def _fixed_geometry_qvalues(analysis_dir: Path) -> dict[int, float]:
    """BH-correct fixed-score geometry p-values from the raw null snapshot."""
    raw: dict[int, float] = {}
    for path in sorted((analysis_dir / "permutation_null").glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            value = payload["p_values"].get("geometry_prauc")
            if value is not None:
                raw[int(payload["feature_id"])] = float(value)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if not raw:
        raise SystemExit("No fixed-score geometry permutation p-values found")
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[int, float] = {}
    running = 1.0
    for index in range(len(ordered) - 1, -1, -1):
        fid = ordered[index]
        running = min(running, raw[fid] * len(ordered) / (index + 1), 1.0)
        adjusted[fid] = running
    return adjusted


def _per_hit_prauc(hit: dict, threshold: float) -> float | None:
    """Per-protein PR-AUC of geom_prob_profile against the SAE truth mask.

    Returns ``None`` for hits where truth has no positives or no negatives —
    PR-AUC is undefined in those degenerate cases.
    """
    acts = np.asarray(hit.get("sae_activation_profile") or [], dtype=np.float32)
    probs = np.asarray(hit.get("geom_prob_profile") or [], dtype=np.float32)
    if acts.size == 0 or acts.size != probs.size:
        return None
    truth = (acts > threshold).astype(np.int8)
    n_pos = int(truth.sum())
    if n_pos == 0 or n_pos == truth.size:
        return None
    return float(average_precision_score(truth, probs))


def _summarise_feature(payload: dict) -> dict | None:
    """Compute per-feature transfer aggregates from one nmpfam_enrichment file."""
    fid = int(payload.get("feature_id", -1))
    if fid < 0:
        return None
    hits = payload.get("nmpfam_hits") or []
    if not hits:
        return None
    threshold = float(payload.get("activation_threshold_sae") or 0.0)

    per_hit = []
    for h in hits:
        prauc = _per_hit_prauc(h, threshold)
        if prauc is None:
            continue
        per_hit.append({
            "family_id": h.get("family_id"),
            "category": h.get("category"),
            "sequence_count": int(h.get("sequence_count") or 0),
            "n_residues": int(h.get("n_residues") or 0),
            "max_sae_activation": float(h.get("max_sae_activation") or 0.0),
            "max_geom_prob": float(h.get("max_geom_prob") or 0.0),
            "n_agree": int(h.get("n_agree") or 0),
            "prauc": prauc,
        })
    if not per_hit:
        return None

    praucs = np.asarray([h["prauc"] for h in per_hit], dtype=np.float32)
    strong = [h for h in per_hit if h["prauc"] > PRAUC_GATE]
    sequences_annotated = sum(h["sequence_count"] for h in strong)

    # Top-N hits sorted by PR-AUC descending — the SPA shows this list directly.
    top_hits = sorted(per_hit, key=lambda h: -h["prauc"])[:25]

    return {
        "feature_id": fid,
        "n_hits": len(per_hit),
        "n_strong": len(strong),
        "max_prauc": float(praucs.max()),
        "median_prauc": float(np.median(praucs)),
        "mean_prauc": float(praucs.mean()),
        "sequences_annotated": sequences_annotated,
        "activation_threshold_sae": threshold,
        "top_hits": top_hits,
        # Internal full list used for the table union. Removed before output.
        "_strong_hits": strong,
    }


def build_summary(
    analysis_dir: Path,
    n_nmpfam_families: int = 50_000,
    n_nmpfam_sequences: int = 10_000_000,
) -> dict:
    enrichment_dir = analysis_dir / "nmpfam" / "nmpfam_enrichment"
    if not enrichment_dir.is_dir():
        raise SystemExit(f"NMPFam enrichment dir missing: {enrichment_dir}")

    # Recompute the paper-primary fixed-score geometry q-values from the raw
    # null snapshot. geometry_primary_analysis.json may be stale or may have
    # been generated in the separate refit robustness mode.
    gp_path = analysis_dir / "geometry_primary_analysis.json"
    if not gp_path.exists():
        raise SystemExit(f"geometry_primary_analysis.json missing in {analysis_dir}")
    with open(gp_path) as f:
        gp = json.load(f)
    geom_q = _fixed_geometry_qvalues(analysis_dir)
    # n_features_total isn't always recorded in geometry_primary_analysis.json;
    # fall back to dataset_stats.num_features (== SAE dictionary size).
    n_features_total = gp.get("n_features_total")
    if not n_features_total:
        ds_path = analysis_dir / "dataset_stats.json"
        if ds_path.exists():
            try:
                with open(ds_path) as f:
                    n_features_total = int(json.load(f).get("num_features") or 0)
            except (json.JSONDecodeError, OSError, ValueError):
                pass
    if not n_features_total:
        n_features_total = len(geom_q) or sum(1 for _ in enrichment_dir.glob("*.json"))

    files = sorted(enrichment_dir.glob("*.json"))
    logger.info("scanning %d nmpfam enrichment files in %s", len(files), enrichment_dir)

    n_with_hits = 0
    n_qsig_with_hits = 0
    feature_records: list[dict] = []

    for i, fpath in enumerate(files):
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        rec = _summarise_feature(data)
        if rec is None:
            continue
        n_with_hits += 1
        q = geom_q.get(rec["feature_id"])
        rec["geometry_padj"] = q
        rec["geom_qsig"] = (q is not None and q < Q_SIG)
        if rec["geom_qsig"]:
            n_qsig_with_hits += 1
        feature_records.append(rec)
        if (i + 1) % 1000 == 0:
            logger.info("  scanned %d / %d features", i + 1, len(files))

    # Table 4 column 3: geom q-sig AND median PR-AUC > 0.5
    gated = [
        r for r in feature_records
        if r["geom_qsig"] and r["median_prauc"] > PRAUC_GATE
    ]
    gated.sort(key=lambda r: -r["max_prauc"])

    # Table 4 columns 4 and 5: NMPFam families hit at PR-AUC > 0.5 by one of the
    # COLUMN-3 GATED features (deduplicated), and the union sequence_count across
    # those families.
    #
    # The union runs over `gated`, not over every feature with hits. Unioning over
    # all features inflates layer 4 to 38,846 families / 7,733,244 sequences
    # against the paper's 3,875 / 757,802; restricting to the gated set reproduces
    # both exactly. Two independent quantities matching to the digit is what fixes
    # the estimand -- see tests/test_analysis/test_nmpfam_transfer_summary.py.
    matched_families: dict[str, int] = {}
    for r in gated:
        for h in r["_strong_hits"]:
            fam = h.get("family_id")
            if not fam:
                continue
            sequence_count = h.get("sequence_count") or 0
            previous = matched_families.get(fam)
            if previous is not None and previous != sequence_count:
                raise ValueError(
                    f"inconsistent sequence_count for NMPFam {fam}: "
                    f"{previous} versus {sequence_count}"
                )
            matched_families[fam] = sequence_count
    n_families_matched = len(matched_families)
    n_sequences_annotated = sum(matched_families.values())

    table4 = {
        "n_features_total": n_features_total,
        "n_with_nmpfam_hits": n_with_hits,
        "n_qsig_with_hits": n_qsig_with_hits,
        "n_features_median_prauc_above_gate": len(gated),
        "n_families_matched": n_families_matched,
        "n_sequences_annotated": n_sequences_annotated,
        "n_nmpfam_families": n_nmpfam_families,
        "n_nmpfam_sequences": n_nmpfam_sequences,
        "pct_with_nmpfam_hits": round(100 * n_with_hits / max(n_features_total, 1), 2),
        "pct_qsig_of_with_hits": round(100 * n_qsig_with_hits / max(n_with_hits, 1), 2),
        "pct_features_median_above_gate": round(100 * len(gated) / max(n_features_total, 1), 2),
        "pct_families_matched": round(
            100 * n_families_matched / max(n_nmpfam_families, 1), 2
        ),
        "pct_sequences_annotated": round(
            100 * n_sequences_annotated / max(n_nmpfam_sequences, 1), 2
        ),
        "prauc_gate": PRAUC_GATE,
        "q_gate": Q_SIG,
        "q_source": "fixed_score_permutation_raw_p",
        "n_geometry_q_tested": len(geom_q),
    }

    for record in feature_records:
        record.pop("_strong_hits", None)

    return {
        "table4": table4,
        # Full gated list (Table 4 column 3) — sorted by max PR-AUC descending.
        "features": gated,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", required=True, type=Path)
    p.add_argument(
        "--n-nmpfam-families", type=int, default=50_000,
        help="Table 4 family denominator (paper release: 50,000)",
    )
    p.add_argument(
        "--n-nmpfam-sequences", type=int, default=10_000_000,
        help="Table 4 sequence denominator (paper release: 10,000,000)",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output path (default: ANALYSIS_DIR/nmpfam_transfer_summary.json)",
    )
    args = p.parse_args()

    analysis = args.analysis_dir.resolve()
    out = args.output or (analysis / "nmpfam_transfer_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(
        analysis,
        n_nmpfam_families=args.n_nmpfam_families,
        n_nmpfam_sequences=args.n_nmpfam_sequences,
    )
    out.write_text(json.dumps(summary, separators=(",", ":")))
    logger.info(
        "wrote %s · %d gated features · matched %d families · %d sequences",
        out,
        summary["table4"]["n_features_median_prauc_above_gate"],
        summary["table4"]["n_families_matched"],
        summary["table4"]["n_sequences_annotated"],
    )


if __name__ == "__main__":
    main()
