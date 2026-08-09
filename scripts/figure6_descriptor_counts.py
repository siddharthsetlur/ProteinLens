#!/usr/bin/env python3
"""Generate Figure 6 descriptor counts from layer-4 geometry artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

AA_COVARIATES = {
    "frac_charged",
    "frac_polar",
    "frac_hydrophobic",
    "frac_aromatic",
    "frac_tiny",
    "frac_small",
}


def compute_counts(
    analysis_dir: Path,
    pr_auc_threshold: float = 0.3,
    importance_threshold: float = 0.1,
) -> tuple[Counter[str], dict]:
    counts: Counter[str] = Counter()
    n_scanned = 0
    n_eligible = 0
    malformed: list[str] = []
    for path in sorted((analysis_dir / "geometry_enrichment").glob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            payload = json.loads(path.read_text())
            residue = payload["geometric_residue_level"]
            score = float(residue["concordance"]["avg_precision"])
            importances = residue["feature_importances"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            malformed.append(path.name)
            continue
        n_scanned += 1
        if score <= pr_auc_threshold:
            continue
        n_eligible += 1
        for descriptor, importance in importances.items():
            if float(importance) > importance_threshold:
                counts[descriptor] += 1
    provenance = {
        "analysis_dir": str(analysis_dir),
        "n_scanned": n_scanned,
        "n_features_pr_auc_above_threshold": n_eligible,
        "pr_auc_operator": ">",
        "pr_auc_threshold": pr_auc_threshold,
        "importance_operator": ">",
        "importance_threshold": importance_threshold,
        "malformed_files": malformed,
    }
    return counts, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("trained_models/layer_4/frosty-sweep-15/analysis"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reproduction_outputs")
    )
    parser.add_argument("--pr-auc-threshold", type=float, default=0.3)
    parser.add_argument("--importance-threshold", type=float, default=0.1)
    args = parser.parse_args()

    counts, provenance = compute_counts(
        args.analysis_dir,
        pr_auc_threshold=args.pr_auc_threshold,
        importance_threshold=args.importance_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    rows = [
        {
            "descriptor": descriptor,
            "count": count,
            "descriptor_class": (
                "amino_acid_covariate"
                if descriptor in AA_COVARIATES
                else "geometry"
            ),
        }
        for descriptor, count in ordered
    ]
    csv_path = args.output_dir / "figure6_descriptor_counts.csv"
    fieldnames = ["descriptor", "count", "descriptor_class"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "figure6_descriptor_counts.json"
    json_path.write_text(
        json.dumps({"provenance": provenance, "rows": rows}, indent=2)
    )

    try:
        import matplotlib.pyplot as plt

        plot_rows = list(reversed(rows))
        fig, axis = plt.subplots(figsize=(9, max(4, 0.25 * len(plot_rows))))
        colors = [
            "#d95f02"
            if row["descriptor_class"] == "amino_acid_covariate"
            else "#1b9e77"
            for row in plot_rows
        ]
        axis.barh(
            [row["descriptor"] for row in plot_rows],
            [row["count"] for row in plot_rows],
            color=colors,
        )
        axis.set_xlabel("Number of features with geometry PR-AUC > 0.30")
        axis.set_ylabel("Descriptor with GBM importance > 0.10")
        fig.tight_layout()
        fig.savefig(args.output_dir / "figure6_descriptor_counts.svg")
        fig.savefig(args.output_dir / "figure6_descriptor_counts.png", dpi=200)
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; wrote machine-readable Figure 6 data only")
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
