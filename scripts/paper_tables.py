#!/usr/bin/env python3
"""Emit paper Tables 1 and 2 from one or more finished analysis directories.

**Table 1 — annotation coverage.** For each annotation method, the percentage
of SAE features whose permutation-test BH q-value is below 0.05.  The
``Total`` column is the union: features significant under *at least one* of
the reported methods.  The denominator is always the full dictionary
(``len(feature_max_activations.npy)``), so dead features count as unannotated.

**Table 2 — geometric PR-AUC distribution.** Among the features that are
*geometrically annotated* (method 7, q < 0.05), the percentage falling in each
PR-AUC bin.  Each row sums to 100 by construction.

Both tables read the same artefacts the Geopedia viewer reads, via
``proteinlens.viz.index_builder``, so the paper numbers and the interface
cannot drift apart.  q-values are not recomputed here: methods 2/5/6/7 carry
their BH-corrected ``*_padj`` from ``geometry_primary_analysis.json``, and
methods 1/3 are BH-corrected in memory from ``permutation_null/*.json``.

CATH (methods 3 and 4) is optional — it is a separate cluster job and is not
part of the published tables.  Pass ``--include-cath`` to add those columns
(and fold them into the union) when the artefacts are present.

Usage::

    # Published ESM2-8M layers
    python scripts/paper_tables.py \\
        --analysis "Layer 2=trained_models/layer_2/firm-sweep-3/analysis" \\
        --analysis "Layer 4=trained_models/layer_4/frosty-sweep-15/analysis" \\
        --analysis "Layer 6=trained_models/layer_6/major-sweep-15/analysis"

    # ESM2-35M generalization run
    python scripts/paper_tables.py \\
        --analysis "ESM2-35M Layer 6=/data/feature_data_esm35m_l6" \\
        --json /data/feature_data_esm35m_l6/paper_tables.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proteinlens.viz.index_builder import (  # noqa: E402
    METHODS,
    Q_SIGNIFICANCE_THRESHOLD,
    _is_sig,
    build_feature_index,
)

# Method ids reported in paper Table 1, in column order.  CATH (3, 4) is
# excluded by default: it comes from a separate job and is not a column of
# the published table.
PAPER_METHOD_IDS = [1, 2, 5, 6, 7]
CATH_METHOD_IDS = [3, 4]

GEOMETRY_METHOD_ID = 7

# Table 2 bin edges on geometric PR-AUC.  The last edge is just above 1 so a
# PR-AUC of exactly 1.0 lands in the final bin instead of falling outside.
PR_AUC_BIN_EDGES = [0.0, 0.3, 0.6, 1.0001]
PR_AUC_BIN_LABELS = ["PR-AUC 0.0-0.3", "PR-AUC 0.3-0.6", "PR-AUC > 0.6"]

_METHOD_NAME = {m["id"]: m["name"] for m in METHODS}


def parse_analysis_arg(spec: str) -> tuple[str, Path]:
    """Split a ``LABEL=PATH`` CLI argument.

    Parameters
    ----------
    spec : str
        Either ``"Layer 4=/path/to/analysis"`` or a bare path, in which case
        the directory name becomes the row label.

    Returns
    -------
    (label, path)
    """
    if "=" in spec:
        label, _, path = spec.partition("=")
        return label.strip(), Path(path.strip())
    path = Path(spec)
    return path.name, path


def table1_row(index: list[dict[str, Any]], method_ids: list[int]) -> dict[str, Any]:
    """Compute one Table 1 row from a feature index.

    Parameters
    ----------
    index : list of dict
        Rows from :func:`build_feature_index`, one per SAE feature, each
        carrying ``m{k}_q`` for k = 1..7.
    method_ids : list of int
        Which methods to report, and to take the union over.

    Returns
    -------
    dict
        Per-method counts and percentages plus the union ("total annotated").
    """
    total = len(index)
    if total == 0:
        raise ValueError("empty feature index — is the analysis dir populated?")

    per_method_n = {
        k: sum(1 for r in index if _is_sig(r.get(f"m{k}_q"))) for k in method_ids
    }
    union_n = sum(
        1 for r in index if any(_is_sig(r.get(f"m{k}_q")) for k in method_ids)
    )
    return {
        "n_features": total,
        "total_annotated_n": union_n,
        "total_annotated_pct": 100.0 * union_n / total,
        "methods": {
            k: {
                "name": _METHOD_NAME[k],
                "n": per_method_n[k],
                "pct": 100.0 * per_method_n[k] / total,
            }
            for k in method_ids
        },
    }


def table2_row(index: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute one Table 2 row: PR-AUC bin occupancy among geometric hits.

    The population is features with ``m7_q < 0.05`` *and* a non-null geometric
    PR-AUC (``m7_score``).  Percentages are of that population, so the bins
    sum to 100.
    """
    scores = [
        float(r["m7_score"])
        for r in index
        if _is_sig(r.get(f"m{GEOMETRY_METHOD_ID}_q")) and r.get("m7_score") is not None
    ]
    n = len(scores)
    counts, _ = np.histogram(np.asarray(scores, dtype=float), bins=PR_AUC_BIN_EDGES)
    return {
        "n_geometric": n,
        "bins": [
            {"label": label, "n": int(c), "pct": (100.0 * int(c) / n) if n else 0.0}
            for label, c in zip(PR_AUC_BIN_LABELS, counts)
        ],
    }


def _render(headers: list[str], rows: list[list[str]]) -> str:
    """Render a fixed-width text table."""
    widths = [
        max([len(headers[i])] + [len(r[i]) for r in rows])
        for i in range(len(headers))
    ]
    sep = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)), sep]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--analysis",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help=(
            "Analysis directory to summarise, optionally prefixed with a row "
            "label. Repeat for one row per layer/model."
        ),
    )
    parser.add_argument(
        "--include-cath",
        action="store_true",
        help=(
            "Add the CATH Protein / CATH Residue columns and fold them into "
            "the union. Off by default — CATH is not in the published tables."
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write the full numbers (raw counts included) to this JSON file.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help=(
            "Log both tables to Weights & Biases (project proteinlens-pipeline) "
            "as wandb.Table artefacts plus flat run-summary scalars."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="proteinlens-pipeline",
        help="wandb project name (default: proteinlens-pipeline)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show index_builder log output."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )

    method_ids = sorted(
        PAPER_METHOD_IDS + (CATH_METHOD_IDS if args.include_cath else [])
    )

    results: dict[str, Any] = {
        "q_threshold": Q_SIGNIFICANCE_THRESHOLD,
        "method_ids": method_ids,
        "pr_auc_bin_edges": PR_AUC_BIN_EDGES,
        "rows": [],
    }

    for spec in args.analysis:
        label, path = parse_analysis_arg(spec)
        if not path.is_dir():
            raise SystemExit(f"analysis dir not found: {path}")
        index = build_feature_index(path)
        results["rows"].append(
            {
                "label": label,
                "analysis_dir": str(path),
                "table1": table1_row(index, method_ids),
                "table2": table2_row(index),
            }
        )

    # ── Table 1 ──
    headers = ["", "% Total annotated"] + [f"% {_METHOD_NAME[k]}" for k in method_ids]
    rows = [
        [r["label"], f"{r['table1']['total_annotated_pct']:.2f}"]
        + [f"{r['table1']['methods'][k]['pct']:.2f}" for k in method_ids]
        for r in results["rows"]
    ]
    print("Table 1: Percentage of annotated SAE features across annotation methods")
    print(f"(annotated by a method when its q < {Q_SIGNIFICANCE_THRESHOLD};")
    print(" Total = union over the reported methods)")
    print()
    print(_render(headers, rows))
    print()
    for r in results["rows"]:
        t1 = r["table1"]
        print(
            f"  {r['label']}: {t1['total_annotated_n']}/{t1['n_features']} "
            f"features annotated"
        )
    print()

    # ── Table 2 ──
    headers2 = [""] + PR_AUC_BIN_LABELS
    rows2 = [
        [r["label"]] + [f"{b['pct']:.2f}" for b in r["table2"]["bins"]]
        for r in results["rows"]
    ]
    print("Table 2: PR-AUC values of geometrically annotated features")
    print("(percentages are of the geometrically annotated set, so rows sum to 100)")
    print()
    print(_render(headers2, rows2))
    print()
    for r in results["rows"]:
        print(f"  {r['label']}: n = {r['table2']['n_geometric']} geometrically annotated")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json}")

    if args.wandb:
        _log_to_wandb(results, headers, rows, headers2, rows2, args.wandb_project)


def _log_to_wandb(
    results: dict[str, Any],
    headers: list[str],
    rows: list[list[str]],
    headers2: list[str],
    rows2: list[list[str]],
    project: str,
) -> None:
    """Log both tables to W&B as tables plus flat summary scalars.

    The wandb.Table rows carry the same strings that were printed, so the
    dashboard and stdout cannot disagree.  The run summary additionally gets
    one float per cell (``table1/<label>/<column>``) so values are plottable
    and comparable across runs.
    """
    import wandb

    run = wandb.init(project=project, name="paper-tables", tags=["paper", "tables"])

    run.log(
        {
            "table1_annotation_coverage": wandb.Table(columns=headers, data=rows),
            "table2_geometric_pr_auc": wandb.Table(columns=headers2, data=rows2),
        }
    )

    summary: dict[str, float] = {}
    for r in results["rows"]:
        label = r["label"]
        t1 = r["table1"]
        summary[f"table1/{label}/total_annotated_pct"] = t1["total_annotated_pct"]
        summary[f"table1/{label}/n_features"] = float(t1["n_features"])
        for k, m in t1["methods"].items():
            summary[f"table1/{label}/{m['name']}"] = m["pct"]
        t2 = r["table2"]
        summary[f"table2/{label}/n_geometric"] = float(t2["n_geometric"])
        for b in t2["bins"]:
            summary[f"table2/{label}/{b['label']}"] = b["pct"]
    run.summary.update(summary)
    run.finish()
    print(f"Logged both tables to W&B project '{project}'.")


if __name__ == "__main__":
    main()
