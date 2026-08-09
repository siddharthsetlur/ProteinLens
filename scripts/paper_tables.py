#!/usr/bin/env python3
"""Generate paper Tables 1 and 2 from one internally consistent snapshot.

The primary path recomputes BH q-values from fixed-score permutation p-values.
It never mixes cached q-values, refit-GBM q-values, or feature-wise fallback
sources. Missing artifacts are reported rather than imputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

Q_THRESHOLD = 0.05
PAPER_METHODS = [
    (1, "InterPro Protein", "interpro_protein_f1"),
    (2, "InterPro Residue", "interpro_res_f1"),
    (5, "Sequence Position", "position_f1"),
    (6, "Sequence MEME Motif", "pwm_pr_auc"),
    (7, "Geometric", "geometry_prauc"),
]
PR_AUC_BINS = [0.0, 0.3, 0.6, 1.0000001]
PR_AUC_LABELS = ["0.0-0.3", "0.3-0.6", ">0.6"]
DEFAULT_ANALYSES = [
    ("Layer 2", Path("trained_models/layer_2/firm-sweep-3/analysis")),
    ("Layer 4", Path("trained_models/layer_4/frosty-sweep-15/analysis")),
    ("Layer 6", Path("trained_models/layer_6/major-sweep-15/analysis")),
]


def bh_correct(values: dict[int, float]) -> dict[int, float]:
    ordered = sorted(values, key=values.get)
    out: dict[int, float] = {}
    running = 1.0
    n_values = len(ordered)
    for index in range(n_values - 1, -1, -1):
        fid = ordered[index]
        running = min(running, values[fid] * n_values / (index + 1), 1.0)
        out[fid] = running
    return out


def dictionary_size(analysis_dir: Path) -> int:
    activations = analysis_dir / "feature_max_activations.npy"
    if activations.exists():
        return int(np.load(activations, mmap_mode="r").shape[0])
    stats = json.loads((analysis_dir / "dataset_stats.json").read_text())
    return int(stats["num_features"])


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_qvalues(
    analysis_dir: Path,
) -> tuple[dict[str, dict[int, float]], dict[str, Any]]:
    raw = {metric: {} for _, _, metric in PAPER_METHODS}
    null_dir = analysis_dir / "permutation_null"
    if not null_dir.is_dir():
        raise FileNotFoundError(f"missing permutation null directory: {null_dir}")

    n_files = 0
    malformed: list[str] = []
    n_permutations: set[int] = set()
    threshold_steps: set[int] = set()
    missing_threshold_metadata = 0
    for path in sorted(null_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            fid = int(payload["feature_id"])
            p_values = payload["p_values"]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            malformed.append(path.name)
            continue
        n_files += 1
        if payload.get("n_permutations") is not None:
            n_permutations.add(int(payload["n_permutations"]))
        if payload.get("threshold_steps") is None:
            missing_threshold_metadata += 1
        else:
            threshold_steps.add(int(payload["threshold_steps"]))
        for _, _, metric in PAPER_METHODS:
            value = p_values.get(metric)
            if value is not None:
                raw[metric][fid] = float(value)

    qvalues = {metric: bh_correct(values) for metric, values in raw.items()}
    provenance = {
        "q_source": "fixed_score_permutation_raw_p",
        "bh_scope": "independent per layer and annotation method",
        "n_json_files": n_files,
        "malformed_files": malformed,
        "n_permutations_values": sorted(n_permutations),
        "threshold_steps_values": sorted(threshold_steps),
        "files_missing_threshold_steps_metadata": missing_threshold_metadata,
        "n_tested": {metric: len(values) for metric, values in raw.items()},
    }
    return qvalues, provenance


def geometry_scores(analysis_dir: Path) -> tuple[dict[int, float], str | None]:
    path = analysis_dir / "geometry_primary_analysis.json"
    payload = json.loads(path.read_text())
    scores = {
        int(fid): float(item["geom_pr_auc"])
        for fid, item in payload.get("features", {}).items()
        if item.get("geom_pr_auc") is not None
    }
    return scores, sha256(path)


def build_row(label: str, analysis_dir: Path) -> dict[str, Any]:
    n_features = dictionary_size(analysis_dir)
    qvalues, provenance = load_fixed_qvalues(analysis_dir)
    scores, score_sha = geometry_scores(analysis_dir)

    methods: dict[str, Any] = {}
    significant_sets: list[set[int]] = []
    for method_id, name, metric in PAPER_METHODS:
        significant = {
            fid for fid, qvalue in qvalues[metric].items() if qvalue < Q_THRESHOLD
        }
        significant_sets.append(significant)
        methods[str(method_id)] = {
            "name": name,
            "metric_key": metric,
            "n": len(significant),
            "pct": 100.0 * len(significant) / n_features,
        }
    union = set().union(*significant_sets)

    geometric_ids = significant_sets[-1]
    missing_scores = sorted(geometric_ids - scores.keys())
    binned_scores = [scores[fid] for fid in geometric_ids if fid in scores]
    counts, _ = np.histogram(np.asarray(binned_scores), bins=PR_AUC_BINS)
    denominator = len(binned_scores)
    return {
        "label": label,
        "analysis_dir": str(analysis_dir),
        "table1": {
            "n_features": n_features,
            "total_annotated_n": len(union),
            "total_annotated_pct": 100.0 * len(union) / n_features,
            "methods": methods,
        },
        "table2": {
            "n_geometric_q_significant": len(geometric_ids),
            "n_geometric_with_score": denominator,
            "missing_score_feature_ids": missing_scores,
            "bins": [
                {
                    "label": bin_label,
                    "n": int(count),
                    "pct": 100.0 * int(count) / denominator if denominator else 0.0,
                }
                for bin_label, count in zip(PR_AUC_LABELS, counts)
            ],
        },
        "provenance": {
            **provenance,
            "geometry_primary_analysis_sha256": score_sha,
        },
    }


def parse_analysis(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(spec)
    return path.name, path


def render(results: dict[str, Any]) -> None:
    names = [name for _, name, _ in PAPER_METHODS]
    print("Table 1 (% dictionary features; fixed-score permutation q < 0.05)")
    print("layer\ttotal\t" + "\t".join(names))
    for row in results["rows"]:
        values = [row["table1"]["total_annotated_pct"]]
        values.extend(
            row["table1"]["methods"][str(mid)]["pct"]
            for mid, _, _ in PAPER_METHODS
        )
        print(row["label"] + "\t" + "\t".join(f"{value:.2f}" for value in values))
    print("\nTable 2 (% geometrically q-significant features)")
    print("layer\t" + "\t".join(PR_AUC_LABELS))
    for row in results["rows"]:
        print(
            row["label"]
            + "\t"
            + "\t".join(
                f"{item['pct']:.2f}" for item in row["table2"]["bins"]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        action="append",
        help="LABEL=PATH; repeat per layer (defaults to canonical local paths)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--require-threshold-metadata",
        action="store_true",
        help="Fail when legacy null files do not record threshold_steps",
    )
    args = parser.parse_args()
    analyses = (
        [parse_analysis(spec) for spec in args.analysis]
        if args.analysis
        else DEFAULT_ANALYSES
    )
    rows = []
    for label, path in analyses:
        if not path.is_dir():
            raise SystemExit(f"analysis directory not found: {path}")
        row = build_row(label, path)
        if (
            args.require_threshold_metadata
            and row["provenance"]["files_missing_threshold_steps_metadata"]
        ):
            raise SystemExit(
                f"{label}: null artifacts lack threshold_steps provenance; "
                "a strict snapshot claim is not possible"
            )
        rows.append(row)
    results = {
        "schema_version": 1,
        "q_threshold": Q_THRESHOLD,
        "q_source": "fixed_score_permutation_raw_p",
        "rows": rows,
    }
    render(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
