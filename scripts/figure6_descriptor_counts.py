#!/usr/bin/env python3
"""Generate Figure 6 descriptor counts from layer-4 geometry artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

RESIDUE_COMPOSITION = {
    "frac_hydrophobic",
    "frac_charged",
    "frac_polar",
    "frac_gly_pro",
    "frac_aromatic",
}
CONTACT_PACKING = {
    "contact_density_8A",
    "contact_density_12A",
    "long_range_contacts_8A",
    "long_range_contacts_12A",
    "max_seq_sep_contact_8A",
    "mean_seq_sep_contact_8A",
    "contact_order_local",
    "min_spatial_dist_long",
}


def descriptor_family(descriptor: str) -> str:
    if descriptor in RESIDUE_COMPOSITION:
        return "Residue composition"
    if descriptor in CONTACT_PACKING:
        return "Contact / packing"
    return "Geometry"


def fixed_geometry_qvalues(analysis_dir: Path) -> dict[int, float]:
    raw: dict[int, float] = {}
    for path in sorted((analysis_dir / "permutation_null").glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            value = payload["p_values"].get("geometry_prauc")
            if value is not None:
                raw[int(payload["feature_id"])] = float(value)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[int, float] = {}
    running = 1.0
    for index in range(len(ordered) - 1, -1, -1):
        fid = ordered[index]
        running = min(running, raw[fid] * len(ordered) / (index + 1), 1.0)
        adjusted[fid] = running
    return adjusted


def compute_counts(
    analysis_dir: Path,
    pr_auc_threshold: float = 0.3,
    importance_threshold: float = 0.1,
) -> tuple[dict[str, Counter[str]], dict]:
    counts = {"0.3-0.6": Counter(), ">0.6": Counter()}
    geometry_q = fixed_geometry_qvalues(analysis_dir)
    n_scanned = 0
    n_eligible = 0
    malformed: list[str] = []
    for path in sorted((analysis_dir / "geometry_enrichment").glob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            payload = json.loads(path.read_text())
            fid = int(payload["feature_id"])
            residue = payload["geometric_residue_level"]
            score = float(residue["concordance"]["avg_precision"])
            importances = residue["feature_importances"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            malformed.append(path.name)
            continue
        n_scanned += 1
        if geometry_q.get(fid, 1.0) >= 0.05 or score <= pr_auc_threshold:
            continue
        n_eligible += 1
        if not importances:
            continue
        descriptor, importance = max(importances.items(), key=lambda item: item[1])
        if float(importance) > importance_threshold:
            score_bin = "0.3-0.6" if score <= 0.6 else ">0.6"
            counts[score_bin][descriptor] += 1
    provenance = {
        "analysis_dir": str(analysis_dir),
        "n_scanned": n_scanned,
        "n_features_pr_auc_above_threshold": n_eligible,
        "pr_auc_operator": ">",
        "pr_auc_threshold": pr_auc_threshold,
        "importance_operator": ">",
        "importance_threshold": importance_threshold,
        "q_source": "fixed_score_permutation_raw_p",
        "q_threshold": 0.05,
        "descriptor_rule": "single highest-importance descriptor per feature",
        "n_geometry_q_tested": len(geometry_q),
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
    descriptors = set(counts["0.3-0.6"]) | set(counts[">0.6"])
    ordered = sorted(
        descriptors,
        key=lambda name: (
            -(counts["0.3-0.6"][name] + counts[">0.6"][name]),
            name,
        ),
    )
    rows = [
        {
            "descriptor": descriptor,
            "count_0.3_0.6": counts["0.3-0.6"][descriptor],
            "count_gt_0.6": counts[">0.6"][descriptor],
            "count": counts["0.3-0.6"][descriptor] + counts[">0.6"][descriptor],
            "descriptor_family": descriptor_family(descriptor),
        }
        for descriptor in ordered
    ]
    csv_path = args.output_dir / "figure6_descriptor_counts.csv"
    fieldnames = [
        "descriptor",
        "count_0.3_0.6",
        "count_gt_0.6",
        "count",
        "descriptor_family",
    ]
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
        from matplotlib.patches import Patch

        plot_rows = list(reversed(rows))
        fig, axis = plt.subplots(figsize=(9, max(4, 0.25 * len(plot_rows))))
        labels = [row["descriptor"] for row in plot_rows]
        low_counts = [row["count_0.3_0.6"] for row in plot_rows]
        high_counts = [row["count_gt_0.6"] for row in plot_rows]
        axis.barh(
            labels, low_counts, color="#80b1d3", label="0.3-0.6"
        )
        axis.barh(
            labels, high_counts, left=low_counts, color="#fb8072", label="> 0.6"
        )
        family_colors = {
            "Geometry": "#1b9e77",
            "Contact / packing": "#7570b3",
            "Residue composition": "#d95f02",
        }
        for tick, row in zip(axis.get_yticklabels(), plot_rows):
            tick.set_color(family_colors[row["descriptor_family"]])
        axis.set_xlabel("Number of features with geometry PR-AUC > 0.30")
        axis.set_ylabel("Descriptor with GBM importance > 0.10")
        pr_auc_legend = axis.legend(title="Geometry PR-AUC", loc="upper right")
        axis.add_artist(pr_auc_legend)
        axis.legend(
            handles=[
                Patch(color=color, label=family)
                for family, color in family_colors.items()
            ],
            title="Descriptor family",
            loc="lower right",
        )
        fig.tight_layout()
        fig.savefig(args.output_dir / "figure6_descriptor_counts.svg")
        fig.savefig(args.output_dir / "figure6_descriptor_counts.png", dpi=200)
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; wrote machine-readable Figure 6 data only")
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
