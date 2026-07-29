#!/usr/bin/env python
"""Estimate GPU/CPU/memory usage per pipeline stage from wandb.

wandb automatically records system metrics (GPU util, GPU memory, CPU%,
process RSS) in the background for every `wandb.init()` run, even when the
code never explicitly logs them. This script pulls that data via the wandb
API for a representative layer (default: layer 4, ESM2-8M) and:

  1. Reports whole-run wall-clock time + system-metric summary for the SAE
     training run.
  2. Auto-discovers `proteinlens-pipeline` project runs tied to that layer
     (feature pipeline, permutation-null, CATH enrichment, ...) and, where a
     run logs `{"stage": name}` markers (run_feature_pipeline.py does this),
     buckets the system-metrics time series between consecutive markers to
     get a per-stage breakdown within a single run.

Stages that were never run with `--wandb` enabled won't have any data here —
cross-reference k8s Job resource requests/limits (k8s/job-*.yaml) as a
static allocation proxy in that case.

Usage:
    conda activate interplm
    wandb login   # if not already authenticated
    python scripts/estimate_pipeline_compute.py --output pipeline_compute.json
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import wandb

ENTITY = "s-setlur-university-of-edinburgh"
SAE_PROJECT = "ProteinLens"
PIPELINE_PROJECT = "proteinlens-pipeline"

SYSTEM_SKIP_COLS = {"_timestamp", "_runtime", "_step"}


def system_history(run, samples: int = 100_000) -> pd.DataFrame | None:
    hist = run.history(stream="system", samples=samples, pandas=True)
    if hist is None or hist.empty:
        return None
    return hist


def summarize_columns(df: pd.DataFrame) -> dict:
    out = {}
    for col in df.columns:
        if col in SYSTEM_SKIP_COLS:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        out[col] = {"mean": float(series.mean()), "max": float(series.max())}
    return out


def run_summary(run) -> dict:
    sys_hist = system_history(run)
    runtime_s = run.summary.get("_runtime")
    return {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "runtime_s": runtime_s,
        "runtime_h": (runtime_s / 3600) if runtime_s else None,
        "system_metrics": summarize_columns(sys_hist) if sys_hist is not None else None,
        "has_system_metrics": sys_hist is not None,
    }


def stage_breakdown(run) -> list[dict] | None:
    """Bucket system metrics into windows between consecutive `stage` log markers."""
    main_hist = run.history(samples=100_000, pandas=True)
    if main_hist is None or "stage" not in main_hist.columns:
        return None

    markers = main_hist.loc[main_hist["stage"].notna(), ["_timestamp", "stage"]]
    markers = markers.sort_values("_timestamp").reset_index(drop=True)
    if markers.empty:
        return None

    sys_hist = system_history(run)
    if sys_hist is None:
        return None

    boundaries = list(markers["_timestamp"]) + [sys_hist["_timestamp"].max() + 1]
    rows = []
    for i, stage in enumerate(markers["stage"]):
        lo, hi = boundaries[i], boundaries[i + 1]
        window = sys_hist[(sys_hist["_timestamp"] >= lo) & (sys_hist["_timestamp"] < hi)]
        row = {"stage": stage, "duration_s": hi - lo, "duration_h": (hi - lo) / 3600}
        if not window.empty:
            for col, stats in summarize_columns(window).items():
                row[f"{col}_mean"] = stats["mean"]
                row[f"{col}_max"] = stats["max"]
        rows.append(row)
    return rows


def find_layer_runs(api, entity: str, project: str, layer: int) -> list:
    matched = []
    for r in api.runs(f"{entity}/{project}"):
        cfg = r.config
        if cfg.get("esm_layer") == layer or f"layer_{layer}" in json.dumps(cfg):
            matched.append(r)
    return matched


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--sae-project", default=SAE_PROJECT)
    p.add_argument("--pipeline-project", default=PIPELINE_PROJECT)
    p.add_argument("--layer", type=int, default=4)
    p.add_argument(
        "--sae-run-id",
        default="86tryizu",
        help="wandb run id of the representative SAE training run (default: frosty-sweep-15, layer 4, ESM2-8M)",
    )
    p.add_argument(
        "--pipeline-run-ids",
        nargs="*",
        default=None,
        help="Explicit wandb run ids in --pipeline-project to include. If omitted, "
        "auto-discovers runs whose config references the target layer.",
    )
    p.add_argument("--output", default="pipeline_compute_estimate.json")
    args = p.parse_args()

    api = wandb.Api()
    report: dict = {}

    print(f"Fetching SAE training run {args.sae_run_id} from {args.entity}/{args.sae_project} ...")
    sae_run = api.run(f"{args.entity}/{args.sae_project}/{args.sae_run_id}")
    report["sae_training"] = run_summary(sae_run)

    if args.pipeline_run_ids:
        pipeline_runs = [api.run(f"{args.entity}/{args.pipeline_project}/{rid}") for rid in args.pipeline_run_ids]
    else:
        print(f"Auto-discovering {args.pipeline_project} runs for layer {args.layer} ...")
        pipeline_runs = find_layer_runs(api, args.entity, args.pipeline_project, args.layer)
        print(f"  Found {len(pipeline_runs)}: {[r.id for r in pipeline_runs]}")

    report["pipeline_runs"] = []
    for run in pipeline_runs:
        entry = run_summary(run)
        entry["stage_breakdown"] = stage_breakdown(run)
        report["pipeline_runs"].append(entry)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {args.output}")

    # Human-readable summary
    print("\n=== SAE training ===")
    st = report["sae_training"]
    print(f"  run: {st['run_name']} ({st['run_id']}), runtime: {st['runtime_h']:.2f}h" if st["runtime_h"] else "  no runtime recorded")
    print(f"  system metrics logged: {st['has_system_metrics']}")

    print("\n=== Pipeline runs ===")
    for entry in report["pipeline_runs"]:
        print(f"  run: {entry['run_name']} ({entry['run_id']}), runtime: {entry['runtime_h']}, system metrics: {entry['has_system_metrics']}")
        if entry["stage_breakdown"]:
            for row in entry["stage_breakdown"]:
                print(f"    - {row['stage']}: {row['duration_h']:.3f}h")


if __name__ == "__main__":
    main()
