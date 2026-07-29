#!/usr/bin/env python3
"""Recompute paper Table 1 (annotation coverage) over a sweep of q thresholds.

Table 1 reports, per layer and per annotation method, the percentage of SAE
features whose BH-corrected permutation q-value falls below 0.05.  Nothing
about that number requires re-running the permutation test: the per-feature
raw p-values are already on disk, so a different significance threshold is a
pure re-thresholding of the same data.

Pipeline reproduced here, per (layer, method)
---------------------------------------------
1. Read the raw one-sided permutation p-value from
   ``{analysis}/permutation_null/{fid}.json`` -> ``p_values[<metric>]``.
   These come from ``scripts/compute_permutation_null.py``: K = 100
   within-protein shuffles of the residue activation vector, annotation held
   fixed, Phipson & Smyth estimator ``p = (1 + #{null >= obs}) / (1 + K)``.
2. Apply Benjamini-Hochberg across every feature that has a p-value for that
   method (m = the count in the ``n_tested`` column), independently per
   method and per layer -- matching the paper's stated correction.
3. Count features with ``q < threshold``.  The denominator is always the full
   dictionary (10,240), so features with no p-value -- dead features, and
   features whose permutation job did not produce output -- count as
   unannotated, exactly as in ``scripts/paper_tables.py``.

Two q-value sources
-------------------
``--q-source raw`` (default) does steps 1-3 above from ``permutation_null/``.

``--q-source padj`` instead reads the cached ``*_padj`` fields out of
``geometry_primary_analysis.json``, which is what ``paper_tables.py`` and the
Geopedia viewer use.  Those caches are the *same* BH computation, but were
written from an older, smaller snapshot of ``permutation_null/`` and were
never regenerated after the remaining permutation jobs finished.  The cached
feature set is a strict subset of the raw one (L2 -655, L4 -104, L6 -62
features), so ``padj`` systematically *under*-counts coverage.  ``raw`` is the
complete and self-consistent source; ``padj`` exists here only to reproduce
the published rows and to audit the gap.

Method -> metric key mapping
----------------------------
    InterPro Protein     interpro_protein_f1     (no cached padj exists)
    InterPro Residue     interpro_res_f1
    CATH Protein         cath_protein_f1         (--include-cath)
    CATH Residue         cath_res_f1             (--include-cath)
    Sequence Position    position_f1
    Sequence MEME Motif  pwm_pr_auc
    Geometric            geometry_prauc

Resolution floor
----------------
With K = 100 the smallest attainable raw p is 1/(K+1) = 0.009901, so after BH
the smallest attainable q for a method is ``0.009901 * m / (#features tied at
that minimum p)`` -- reported as ``q_min``.  Thresholds at or below ``q_min``
select nothing and describe the granularity of the null, not a stricter
biological criterion.  Reaching q < 0.001 requires re-running
``compute_permutation_null.py`` with roughly 10x the permutations.

Error bars
----------
``--csv`` writes one tidy row per (layer, method, threshold) carrying
``n_significant`` / ``n_total`` alongside the percentage and a Wilson 95%
score interval, so downstream plotting can use the interval directly or
recompute its own from the counts.

Usage::

    python scripts/paper_table1_q_sweep.py \
        --analysis "Layer 2=analysis/l2" \
        --analysis "Layer 4=analysis/l4" \
        --analysis "Layer 6=analysis/l6" \
        --csv analysis/table1_q_sweep.csv \
        --json analysis/table1_q_sweep.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proteinlens.viz.index_builder import _bh_correct  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]

# (method id, display name, permutation_null p_values key, cached padj key)
# Paper order.  The padj key is None for the protein-level methods: no cached
# q-value is written for them, index_builder BH-corrects them in memory.
METHOD_SPECS: list[tuple[int, str, str, str | None]] = [
    (1, "InterPro Protein", "interpro_protein_f1", None),
    (2, "InterPro Residue", "interpro_res_f1", "interpro_res_f1_padj"),
    (3, "CATH Protein", "cath_protein_f1", None),
    (4, "CATH Residue", "cath_res_f1", "cath_res_f1_padj"),
    (5, "Sequence Position", "position_f1", "position_f1_padj"),
    (6, "Sequence MEME Motif", "pwm_pr_auc", "motif_pr_auc_padj"),
    (7, "Geometric", "geometry_prauc", "geometry_prauc_padj"),
]

PAPER_METHOD_IDS = [1, 2, 5, 6, 7]
CATH_METHOD_IDS = [3, 4]

Z95 = 1.959963984540054


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, returned as percentages.

    Preferred over the normal approximation because coverage here sits near
    the boundaries (some methods exceed 94%), where the Wald interval
    overshoots 100%.  Note the caveat in ``--csv`` help: SAE features are not
    independent draws, so this is a descriptive interval, not a claim about
    resampling the dictionary.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half)


def load_raw_pvalues(analysis_dir: Path) -> dict[str, dict[int, float]]:
    """Read every ``permutation_null/*.json`` into ``{metric: {fid: raw p}}``.

    Files that fail to parse are skipped with a warning rather than aborting
    the run -- a truncated JSON from a killed worker should cost one feature,
    not the whole table.
    """
    perm_dir = analysis_dir / "permutation_null"
    if not perm_dir.is_dir():
        raise SystemExit(f"no permutation_null/ directory in {analysis_dir}")

    raw: dict[str, dict[int, float]] = {spec[2]: {} for spec in METHOD_SPECS}
    n_files = n_bad = 0
    for path in perm_dir.iterdir():
        if path.suffix != ".json":
            continue
        try:
            d = json.loads(path.read_text())
            fid = int(d["feature_id"])
            pvals = d["p_values"]
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            n_bad += 1
            continue
        n_files += 1
        for _, _, pkey, _ in METHOD_SPECS:
            v = pvals.get(pkey)
            if v is not None:
                raw[pkey][fid] = float(v)
    logger.info(
        "%s: read %d permutation files (%d unreadable)", analysis_dir, n_files, n_bad
    )
    return raw


def load_cached_padj(analysis_dir: Path) -> dict[str, dict[int, float]]:
    """Read the cached ``*_padj`` q-values from geometry_primary_analysis.json."""
    gp_path = analysis_dir / "geometry_primary_analysis.json"
    if not gp_path.exists():
        raise SystemExit(f"no geometry_primary_analysis.json in {analysis_dir}")
    features = json.loads(gp_path.read_text()).get("features", {})
    out: dict[str, dict[int, float]] = {}
    for _, _, pkey, qkey in METHOD_SPECS:
        if qkey is None:
            out[pkey] = {}
            continue
        out[pkey] = {
            int(fid): float(v[qkey])
            for fid, v in features.items()
            if v.get(qkey) is not None
        }
    return out


def build_qvalues(
    analysis_dir: Path, q_source: str, n_features: int
) -> dict[str, list[float | None]]:
    """Return ``{metric: [q per feature id, None where untested]}``.

    For ``raw``, BH runs here over exactly the features that carry a p-value
    for that metric.  For ``padj``, the cached q-values are passed through
    unchanged -- BH was already applied when they were written.
    """
    if q_source == "padj":
        cached = load_cached_padj(analysis_dir)
        return {
            pkey: [cached[pkey].get(fid) for fid in range(n_features)]
            for _, _, pkey, _ in METHOD_SPECS
        }

    raw = load_raw_pvalues(analysis_dir)
    qvals: dict[str, list[float | None]] = {}
    for _, _, pkey, _ in METHOD_SPECS:
        per_fid = raw[pkey]
        fids = sorted(per_fid)
        corrected = _bh_correct([per_fid[f] for f in fids])
        lookup = dict(zip(fids, corrected))
        qvals[pkey] = [lookup.get(fid) for fid in range(n_features)]
    return qvals


def n_features_of(analysis_dir: Path) -> int:
    """Dictionary size -- the Table 1 denominator, dead features included."""
    path = analysis_dir / "feature_max_activations.npy"
    if path.exists():
        return int(len(np.load(path)))
    ds = json.loads((analysis_dir / "dataset_stats.json").read_text())
    return int(ds["num_features"])


def parse_analysis_arg(spec: str) -> tuple[str, Path]:
    """Split a ``LABEL=PATH`` argument; a bare path uses its dirname as label."""
    if "=" in spec:
        label, _, path = spec.partition("=")
        return label.strip(), Path(path.strip())
    path = Path(spec)
    return path.name, path


def _render(headers: list[str], rows: list[list[str]]) -> str:
    """Render a fixed-width text table."""
    widths = [
        max([len(headers[i])] + [len(r[i]) for r in rows]) for i in range(len(headers))
    ]
    sep = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)), sep]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--analysis", action="append", required=True, metavar="LABEL=PATH",
        help="Analysis directory, optionally prefixed with a row label. Repeatable.",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
        help=f"q thresholds to sweep (default: {DEFAULT_THRESHOLDS})",
    )
    parser.add_argument(
        "--q-source", choices=["raw", "padj"], default="raw",
        help=(
            "raw (default): BH over permutation_null/*.json, the complete "
            "source. padj: cached q-values from geometry_primary_analysis.json, "
            "written from an older smaller snapshot and so an undercount."
        ),
    )
    parser.add_argument(
        "--include-cath", action="store_true",
        help="Add CATH Protein / CATH Residue columns and fold them into the union.",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help=(
            "Write a tidy row per (layer, method, threshold) with counts and a "
            "Wilson 95%% interval, for plotting error bars. The interval treats "
            "features as independent binomial draws, which they are not -- see "
            "the module docstring on the K=100 resolution floor, the dominant "
            "source of uncertainty here."
        ),
    )
    parser.add_argument("--json", type=Path, default=None, help="Write full numbers here.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )

    method_ids = sorted(
        PAPER_METHOD_IDS + (CATH_METHOD_IDS if args.include_cath else [])
    )
    specs = [s for s in METHOD_SPECS if s[0] in method_ids]
    thresholds = sorted(args.thresholds)

    results: dict[str, Any] = {
        "q_source": args.q_source,
        "thresholds": thresholds,
        "method_ids": method_ids,
        "n_permutations": 100,
        "min_attainable_raw_p": 1.0 / 101.0,
        "rows": [],
    }
    csv_rows: list[dict[str, Any]] = []

    for spec_arg in args.analysis:
        label, path = parse_analysis_arg(spec_arg)
        if not path.is_dir():
            raise SystemExit(f"analysis dir not found: {path}")
        n_total = n_features_of(path)
        qvals = build_qvalues(path, args.q_source, n_total)

        per_method: dict[int, Any] = {}
        for mid, name, pkey, _ in specs:
            qs = qvals[pkey]
            tested = [q for q in qs if q is not None]
            per_method[mid] = {
                "name": name,
                "metric_key": pkey,
                "n_tested": len(tested),
                "q_min": min(tested) if tested else None,
                "by_threshold": {},
            }
            for t in thresholds:
                k = sum(1 for q in tested if q < t)
                lo, hi = wilson_interval(k, n_total)
                per_method[mid]["by_threshold"][f"{t:g}"] = {
                    "n_significant": k, "pct": 100.0 * k / n_total,
                    "ci_lo": lo, "ci_hi": hi,
                }
                csv_rows.append({
                    "layer": label, "method": name, "method_id": mid,
                    "metric_key": pkey, "q_threshold": t,
                    "n_significant": k, "n_tested": len(tested), "n_total": n_total,
                    "pct": round(100.0 * k / n_total, 4),
                    "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                    "err_lo": round(100.0 * k / n_total - lo, 4),
                    "err_hi": round(hi - 100.0 * k / n_total, 4),
                })

        union: dict[str, Any] = {}
        for t in thresholds:
            k = sum(
                1
                for fid in range(n_total)
                if any(
                    (q := qvals[pkey][fid]) is not None and q < t
                    for _, _, pkey, _ in specs
                )
            )
            lo, hi = wilson_interval(k, n_total)
            union[f"{t:g}"] = {
                "n_significant": k, "pct": 100.0 * k / n_total,
                "ci_lo": lo, "ci_hi": hi,
            }
            csv_rows.append({
                "layer": label, "method": "TOTAL (union)", "method_id": 0,
                "metric_key": "union", "q_threshold": t,
                "n_significant": k, "n_tested": n_total, "n_total": n_total,
                "pct": round(100.0 * k / n_total, 4),
                "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                "err_lo": round(100.0 * k / n_total - lo, 4),
                "err_hi": round(hi - 100.0 * k / n_total, 4),
            })

        results["rows"].append({
            "label": label, "analysis_dir": str(path), "n_features": n_total,
            "methods": per_method, "union": union,
        })

    # ── Per-layer view: coverage vs threshold, with the attainability floor ──
    print(f"q-value source: {args.q_source}\n")
    for r in results["rows"]:
        print(f"{r['label']}  (n_features = {r['n_features']})")
        headers = ["method", "n_tested", "q_min"] + [f"q<{t:g}" for t in thresholds]
        body = []
        for mid, _, _, _ in specs:
            m = r["methods"][mid]
            body.append(
                [m["name"], str(m["n_tested"]),
                 f"{m['q_min']:.4f}" if m["q_min"] is not None else "-"]
                + [f"{m['by_threshold'][f'{t:g}']['pct']:.2f}" for t in thresholds]
            )
        body.append(
            ["TOTAL (union)", str(r["n_features"]), "-"]
            + [f"{r['union'][f'{t:g}']['pct']:.2f}" for t in thresholds]
        )
        print(_render(headers, body))
        print()

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Wrote {args.csv}  ({len(csv_rows)} rows)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
