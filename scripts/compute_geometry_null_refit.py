#!/usr/bin/env python3
"""Compute refit-GBM permutation null for geometry PR-AUC.

This is an **additive** companion to ``compute_permutation_null.py``. It
never modifies any existing file. Output is written to a new directory
``<data-dir>/geometry_null_refit/{fid:04d}.json`` which
``compute_geometry_primary.py`` reads as an independent BH pool alongside
the fixed-GBM null — the two pools are BH-corrected separately and never
mixed, because they use different observed statistics and different null
distributions (see ``_load_permutation_pvalues`` in that script).

For the rationale and invariants see
``proteinlens/analysis/feature_pipeline/geometry_null_refit.py``.

Usage
-----
Sanity check on a few features (dry-run directory)::

    python scripts/compute_geometry_null_refit.py \\
        --data-dir trained_models/layer_4/frosty-sweep-15/analysis \\
        --dry-run-features 5 --n-permutations 20

Full layer (parallel)::

    python scripts/compute_geometry_null_refit.py \\
        --data-dir trained_models/layer_4/frosty-sweep-15/analysis \\
        --workers 16

Resume-safe: existing output JSONs are skipped. Kill and restart.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proteinlens.analysis.feature_pipeline.geometry_null_refit import (
    compute_refit_null,
)

logger = logging.getLogger("geometry_null_refit")


# ── Tree-immutability guard ────────────────────────────────────────────
#
# Snapshot (mtime, size) for every file under paths that must never change.
# Re-check at shutdown; abort with non-zero exit if any diverge.


_GUARDED_SUBDIRS = (
    "permutation_null",
    "geometry_classifiers",
    "geometry_enrichment",
)


def _snapshot_tree(data_dir: Path) -> dict[str, tuple[float, int]]:
    snap: dict[str, tuple[float, int]] = {}
    for sub in _GUARDED_SUBDIRS:
        sub_path = data_dir / sub
        if not sub_path.is_dir():
            continue
        for root, _dirs, files in os.walk(sub_path):
            for name in files:
                p = Path(root) / name
                try:
                    st = p.stat()
                except OSError:
                    continue
                snap[str(p.relative_to(data_dir))] = (st.st_mtime, st.st_size)
    return snap


def _diff_snapshots(
    before: dict[str, tuple[float, int]], after: dict[str, tuple[float, int]]
) -> list[str]:
    diffs: list[str] = []
    for rel, stat_b in before.items():
        if rel not in after:
            diffs.append(f"DELETED: {rel}")
            continue
        stat_a = after[rel]
        if stat_b != stat_a:
            diffs.append(
                f"CHANGED: {rel} (mtime {stat_b[0]} -> {stat_a[0]}, "
                f"size {stat_b[1]} -> {stat_a[1]})"
            )
    for rel in after:
        if rel not in before:
            diffs.append(f"NEW (in guarded tree): {rel}")
    return diffs


# ── Shared-data loader — mirrors compute_permutation_null.py:1324–1435 ─


def _setup_shared(
    data_dir: Path,
    geom_profile_dir_override: Path | None = None,
    act_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Glob directories once and return a dict suitable for worker inheritance.

    The geometry residue profiles (``.npz`` per-protein backbone files) and
    residue-activation ``.npz`` files may live outside ``data_dir`` on
    deployments where intermediate per-protein data is kept on a separate
    mount from the analysis outputs. Override paths let the caller point at
    the shared location.
    """
    pipeline_state_path = data_dir / "pipeline_state.json"
    protein_maxes_path = data_dir / "protein_feature_maxes.npy"
    feat_max_path = data_dir / "feature_max_activations.npy"
    if not feat_max_path.exists():
        raise FileNotFoundError(f"{feat_max_path} not found")
    feat_max_arr = np.load(feat_max_path)
    n_features = int(len(feat_max_arr))

    if not (pipeline_state_path.exists() and protein_maxes_path.exists()):
        raise FileNotFoundError(
            "pipeline_state.json and protein_feature_maxes.npy are required "
            "for the refit null (top-500 protein selection uses the full "
            "protein-level activation matrix)."
        )
    state = json.loads(pipeline_state_path.read_text())
    acc_to_idx: dict[str, int] = state.get("accession_index", {})
    n_proteins = len(acc_to_idx)

    # Glob once. Candidate locations for geometry_residue_profiles, in order:
    #   1. explicit override (--geom-profile-dir)
    #   2. <data-dir>/geometry_residue_profiles/
    #   3. <data-dir>/geometry_enrichment/geometry_residue_profiles/
    #   (this nested form is used by deployments that co-locate profiles
    #    with the enrichment outputs)
    if geom_profile_dir_override is not None:
        geom_profile_dir = geom_profile_dir_override
    else:
        geom_profile_dir = data_dir / "geometry_residue_profiles"
        if not geom_profile_dir.is_dir():
            nested = data_dir / "geometry_enrichment" / "geometry_residue_profiles"
            if nested.is_dir():
                geom_profile_dir = nested
    geom_profile_files: set[str] = set()
    if geom_profile_dir.is_dir():
        geom_profile_files = {p.stem for p in geom_profile_dir.glob("*.npz")}

    act_file_map: dict[str, Path] = {}
    if act_dir_override is not None:
        if act_dir_override.is_dir():
            for p in act_dir_override.glob("*.npz"):
                if p.stem not in act_file_map:
                    act_file_map[p.stem] = p
    else:
        for act_dir_name in ("residue_activations", "interpro_residue_activations"):
            act_dir = data_dir / act_dir_name
            if act_dir.is_dir():
                for p in act_dir.glob("*.npz"):
                    if p.stem not in act_file_map:
                        act_file_map[p.stem] = p

    # Row -> accession map, filtered to proteins with BOTH geom profile + activations.
    # Matches compute_permutation_null.py:1397–1398 so the top-500 selection is identical.
    row_to_acc = {
        v: k
        for k, v in acc_to_idx.items()
        if k in geom_profile_files and k in act_file_map
    }

    act_matrix_full = np.memmap(
        protein_maxes_path,
        dtype="float32",
        mode="r",
        shape=(n_proteins, n_features),
    )

    # Geometry enrichment summaries — needed for stored_avg_precision lookup.
    # We do NOT preload every JSON (would be expensive); per-worker lookup is fine.
    geom_enrich_dir = data_dir / "geometry_enrichment"
    geom_enrich_fids: set[int] = set()
    if geom_enrich_dir.is_dir():
        for p in geom_enrich_dir.glob("*.json"):
            if p.name == "summary.json":
                continue
            try:
                geom_enrich_fids.add(int(p.stem))
            except ValueError:
                continue

    return {
        "feat_max_arr": feat_max_arr,
        "n_features": n_features,
        "geom_profile_dir": geom_profile_dir,
        "geom_profile_files": geom_profile_files,
        "act_file_map": act_file_map,
        "row_to_acc": row_to_acc,
        "act_matrix_full": act_matrix_full,
        "geom_enrich_dir": geom_enrich_dir,
        "geom_enrich_fids": geom_enrich_fids,
    }


def _read_stored_avg_precision(geom_enrich_dir: Path, fid: int) -> float | None:
    """Look up ``concordance.avg_precision`` for this feature, if present."""
    p = geom_enrich_dir / f"{fid:04d}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    res = d.get("geometric_residue_level") or {}
    conc = res.get("concordance") or {}
    v = conc.get("avg_precision")
    return float(v) if v is not None else None


# ── Atomic write ───────────────────────────────────────────────────────


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


# ── Worker glue ────────────────────────────────────────────────────────

# Module-level state populated by _worker_init via fork inheritance.
_WORKER_STATE: dict[str, Any] = {}


def _worker_init(shared: dict[str, Any]) -> None:
    _WORKER_STATE.update(shared)


def _worker_process(
    fid: int,
    out_dir: Path,
    n_permutations: int,
    seed: int,
    max_proteins: int,
    observed_parity_strict: bool,
) -> tuple[int, str, dict[str, Any] | None, bool]:
    """Return (fid, status, result_summary, stored_ap_present).

    Status is one of: written, skipped, already_exists, error.
    ``result_summary`` holds the per-feature scalar fields main() logs to
    wandb (None when status != "written"). Keeping it small keeps
    inter-process pickling cheap vs returning the full output dict.
    """
    out_path = out_dir / f"{fid:04d}.json"
    if out_path.exists():
        return fid, "already_exists", None, False

    stored_ap = _read_stored_avg_precision(_WORKER_STATE["geom_enrich_dir"], fid)
    try:
        result = compute_refit_null(
            fid=fid,
            act_matrix_full=_WORKER_STATE["act_matrix_full"],
            row_to_acc=_WORKER_STATE["row_to_acc"],
            act_file_map=_WORKER_STATE["act_file_map"],
            geom_profile_dir=_WORKER_STATE["geom_profile_dir"],
            geom_profile_files=_WORKER_STATE["geom_profile_files"],
            n_permutations=n_permutations,
            seed=seed,
            max_proteins=max_proteins,
            stored_avg_precision=stored_ap,
            observed_parity_strict=observed_parity_strict,
        )
    except (ValueError, OverflowError):
        # Numerical issues (e.g., RNG seed overflow) must surface — not
        # swallowed by the generic Exception branch below.
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("fid %d: worker crashed: %s", fid, e)
        return fid, "error", None, stored_ap is not None

    if result is None:
        return fid, "skipped", None, stored_ap is not None

    _atomic_write_json(out_path, result)
    summary = {
        "observed_prauc": float(result["observed_prauc"]),
        "null_mean": float(result["null_mean"]),
        "null_std": float(result["null_std"]),
        "p_value_refit": float(result["p_value_refit"]),
        "n_proteins": int(result["n_proteins"]),
        "n_residues_total": int(result["n_residues_total"]),
        "observed_parity_delta": (
            float(result["observed_parity_delta"])
            if result.get("observed_parity_delta") is not None
            else None
        ),
    }
    return fid, "written", summary, stored_ap is not None


# ── CLI ────────────────────────────────────────────────────────────────


def _parse_range(s: str) -> tuple[int, int]:
    a, b = s.split("-", 1)
    return int(a), int(b)


def _build_fid_list(shared: dict[str, Any], args: argparse.Namespace) -> list[int]:
    n_features = shared["n_features"]
    if args.fids_from:
        text = Path(args.fids_from).read_text()
        fids = [int(x.strip()) for x in text.split() if x.strip()]
    elif args.feature_range:
        lo, hi = _parse_range(args.feature_range)
        fids = list(range(lo, hi))
    else:
        fids = list(range(n_features))

    # Restrict to features with geometry enrichment data on disk. Without the
    # stored avg_precision we still run, but the enrichment dir is the
    # authoritative source of features that have a GBM worth testing.
    enrich_fids = shared["geom_enrich_fids"]
    if enrich_fids:
        fids = [f for f in fids if f in enrich_fids]

    if args.dry_run_features and args.dry_run_features > 0:
        fids = fids[: args.dry_run_features]
    return fids


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=500,
        help="Top-N activating proteins per feature (matches pipeline default 500).",
    )
    parser.add_argument(
        "--fids-from",
        type=Path,
        default=None,
        help="Newline/whitespace-separated feature IDs to process (overrides range).",
    )
    parser.add_argument(
        "--feature-range",
        type=str,
        default=None,
        help="Half-open range 'A-B' of feature IDs (e.g., 0-500).",
    )
    parser.add_argument(
        "--dry-run-features",
        type=int,
        default=0,
        help="Process only the first N features; write to geometry_null_refit_dryrun/.",
    )
    parser.add_argument(
        "--allow-non-empty-output",
        action="store_true",
        help="Do not abort if the output directory already has files (normal resume).",
    )
    parser.add_argument(
        "--geom-profile-dir",
        type=Path,
        default=None,
        help=(
            "Override path to geometry_residue_profiles/. "
            "Default: <data-dir>/geometry_residue_profiles/ or "
            "<data-dir>/geometry_enrichment/geometry_residue_profiles/."
        ),
    )
    parser.add_argument(
        "--act-dir",
        type=Path,
        default=None,
        help=(
            "Override path to per-protein residue-activation .npz files. "
            "Default: <data-dir>/residue_activations/ and "
            "<data-dir>/interpro_residue_activations/ (merged)."
        ),
    )
    parser.add_argument(
        "--observed-parity-strict",
        action="store_true",
        help=(
            "Abort (skip feature) when |observed - stored concordance.avg_precision| "
            "exceeds --observed-parity-warn-delta. Use for paper-grade runs where "
            "the refit protein set must match the enrichment stage closely."
        ),
    )
    parser.add_argument(
        "--observed-parity-warn-delta",
        type=float,
        default=0.05,
        help=(
            "Absolute PR-AUC delta above which the observed-parity diagnostic "
            "fires (warn by default; skip when --observed-parity-strict)."
        ),
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log progress and summary statistics to Weights & Biases.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(f"--data-dir {data_dir} is not a directory")

    # RNG-offset collision guard. The offsets in compute_permutation_null.py
    # (10M, 20M, 30M) and in this module (40M) are spaced 10M apart, so any
    # |seed| >= 10M can collide `seed + fid + offset_a` with
    # `seed' + fid + offset_b` for some offset pair. Refuse such seeds
    # rather than silently producing invalid nulls.
    if abs(args.seed) >= 10_000_000:
        raise SystemExit(
            f"--seed must satisfy |seed| < 10_000_000 to avoid RNG-offset "
            f"collision with the four offset streams (10M / 20M / 30M / "
            f"40M). Got {args.seed}. Pick a smaller seed (any int in "
            f"[-9_999_999, 9_999_999] is safe)."
        )

    # ── Optional W&B init ── (same project + tag scheme as
    # compute_permutation_null.py so both nulls show up side-by-side.)
    wb_run = None
    if args.wandb:
        import wandb  # local import: wandb is optional at import time

        wb_run = wandb.init(
            project="proteinlens-pipeline",
            name="refit-null",
            tags=["permutation", "null-distribution", "refit-gbm", "geometry"],
            config={
                "script": "compute_geometry_null_refit.py",
                "n_permutations": args.n_permutations,
                "seed": args.seed,
                "workers": args.workers,
                "max_proteins": args.max_proteins,
                "observed_parity_strict": args.observed_parity_strict,
                "observed_parity_warn_delta": args.observed_parity_warn_delta,
                "data_dir": str(args.data_dir),
                "geom_profile_dir": str(args.geom_profile_dir) if args.geom_profile_dir else None,
                "act_dir": str(args.act_dir) if args.act_dir else None,
            },
        )

    out_name = "geometry_null_refit_dryrun" if args.dry_run_features > 0 else "geometry_null_refit"
    out_dir = data_dir / out_name
    out_dir.mkdir(exist_ok=True)

    # Preflight logging
    logger.info("data_dir = %s", data_dir)
    logger.info("out_dir  = %s", out_dir)
    logger.info("n_permutations = %d, seed = %d, workers = %d",
                args.n_permutations, args.seed, args.workers)

    # Immutability snapshot (guarded trees)
    t0 = time.time()
    snap_before = _snapshot_tree(data_dir)
    logger.info("snapshotted %d files across guarded subdirs in %.1fs",
                len(snap_before), time.time() - t0)

    # Shared data
    shared = _setup_shared(
        data_dir,
        geom_profile_dir_override=args.geom_profile_dir,
        act_dir_override=args.act_dir,
    )
    logger.info("glob: %d geom_profile_files, %d act_file_map, %d row_to_acc",
                len(shared["geom_profile_files"]), len(shared["act_file_map"]),
                len(shared["row_to_acc"]))
    logger.info("geom_enrich_fids: %d", len(shared["geom_enrich_fids"]))

    fids = _build_fid_list(shared, args)
    logger.info("features to consider: %d", len(fids))

    existing = sum(1 for f in fids if (out_dir / f"{f:04d}.json").exists())
    if existing and args.dry_run_features == 0 and not args.allow_non_empty_output:
        logger.info("resume: %d features already have output and will be skipped", existing)

    # Parallel execution. The post-run immutability check runs in a
    # `finally` so that crashes, SIGTERM, SystemExit, and normal
    # completion all trigger the guarded-tree verification. Ctrl+C
    # (SIGINT) still bypasses Python `finally` on Linux if the signal
    # arrives during a blocking syscall in a worker — document that
    # and instruct the user to verify manually.
    ctx = get_context("fork")
    n_written = n_skipped = n_existing = n_error = 0
    n_with_stored_ap = 0
    n_without_stored_ap = 0
    null_means: list[float] = []
    p_values: list[float] = []
    observed_pr_aucs: list[float] = []
    parity_deltas: list[float] = []
    parity_flags = 0
    t_run = time.time()
    t_run_elapsed = 0.0
    total_fids = len(fids)

    try:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(shared,),
        ) as pool:
            futures = {
                pool.submit(
                    _worker_process,
                    fid,
                    out_dir,
                    args.n_permutations,
                    args.seed,
                    args.max_proteins,
                    args.observed_parity_strict,
                ): fid
                for fid in fids
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="refit-null"):
                fid = futures[fut]
                try:
                    _fid, status, summary, stored_ap_present = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.exception("fid %d: future raised: %s", fid, e)
                    n_error += 1
                    continue
                if stored_ap_present:
                    n_with_stored_ap += 1
                else:
                    n_without_stored_ap += 1
                if status == "written" and summary is not None:
                    n_written += 1
                    null_means.append(summary["null_mean"])
                    p_values.append(summary["p_value_refit"])
                    observed_pr_aucs.append(summary["observed_prauc"])
                    if summary.get("observed_parity_delta") is not None:
                        parity_deltas.append(summary["observed_parity_delta"])
                    if summary["null_mean"] < 0.10:
                        parity_flags += 1
                    if wb_run is not None:
                        n_processed = n_written + n_skipped + n_existing + n_error
                        wb_run.log({
                            "progress/completed": n_processed,
                            "progress/total": total_fids,
                            "progress/pct": 100 * n_processed / max(total_fids, 1),
                            "feature/id": fid,
                            "feature/n_proteins": summary["n_proteins"],
                            "feature/n_residues": summary["n_residues_total"],
                            "pvalue/geometry_prauc_refit": summary["p_value_refit"],
                            "observed/geometry_prauc": summary["observed_prauc"],
                            "null/mean": summary["null_mean"],
                            "null/std": summary["null_std"],
                            "parity/delta": summary.get("observed_parity_delta"),
                        })
                elif status == "already_exists":
                    n_existing += 1
                elif status == "skipped":
                    n_skipped += 1
                else:
                    n_error += 1
        t_run_elapsed = time.time() - t_run
    finally:
        # Always snapshot and diff, even on exception / early exit.
        snap_after = _snapshot_tree(data_dir)
        diffs = _diff_snapshots(snap_before, snap_after)
        # Persist the diff list next to the output dir so that an
        # aborted run still leaves an auditable record.
        try:
            report_path = out_dir / "_immutability_check.json"
            report_payload = {
                "n_guarded_files": len(snap_before),
                "n_diffs": len(diffs),
                "diffs": diffs[:200],  # cap to keep the report small
                "snapshot_key": "(mtime, size)",
                "completed_cleanly": not diffs,
            }
            _atomic_write_json(report_path, report_payload)
        except OSError as e:
            logger.warning("could not write immutability report: %s", e)

        if diffs:
            logger.error("GUARDED TREE MUTATED (%d diffs):", len(diffs))
            for d in diffs[:20]:
                logger.error("  %s", d)
            if len(diffs) > 20:
                logger.error("  ... (%d more)", len(diffs) - 20)
            raise SystemExit(2)

    # Summary
    logger.info("=" * 60)
    logger.info("refit-null complete in %.1fs", t_run_elapsed)
    logger.info("  written: %d   skipped: %d   already_existed: %d   errors: %d",
                n_written, n_skipped, n_existing, n_error)
    if n_without_stored_ap:
        logger.info(
            "  stored avg_precision present for %d features, missing for %d "
            "(parity diagnostic disabled on the latter)",
            n_with_stored_ap, n_without_stored_ap,
        )
    if null_means:
        arr = np.asarray(null_means)
        logger.info(
            "  null_mean distribution over written features: "
            "min=%.3f p25=%.3f median=%.3f p75=%.3f max=%.3f",
            float(arr.min()), float(np.percentile(arr, 25)),
            float(np.median(arr)), float(np.percentile(arr, 75)),
            float(arr.max()),
        )
    if parity_flags:
        logger.warning(
            "  %d features had null_mean < 0.10 (expected >0.10 under refit); "
            "spot-check a few outputs",
            parity_flags,
        )
    logger.info(
        "guarded trees mtime+size identical ✓ (%d files)", len(snap_before)
    )

    # ── Final W&B summary ──
    if wb_run is not None:
        summary_payload: dict[str, Any] = {
            "total_features_considered": total_fids,
            "written": n_written,
            "skipped": n_skipped,
            "already_existed": n_existing,
            "errors": n_error,
            "features_with_stored_ap": n_with_stored_ap,
            "features_without_stored_ap": n_without_stored_ap,
            "elapsed_seconds": t_run_elapsed,
            "immutability_diffs": len(diffs),
        }
        if p_values:
            pv_arr = np.array(p_values, dtype=np.float64)
            summary_payload["n_significant_geometry_prauc_refit"] = int((pv_arr < 0.05).sum())
            summary_payload["pct_significant_geometry_prauc_refit"] = round(
                100.0 * float((pv_arr < 0.05).mean()), 2
            )
            summary_payload["median_pvalue_geometry_prauc_refit"] = round(
                float(np.median(pv_arr)), 4
            )
            import wandb as _wb

            wb_run.log({
                "hist/pvalue_refit": _wb.Histogram(pv_arr, num_bins=20),
                "hist/observed_prauc": _wb.Histogram(np.array(observed_pr_aucs), num_bins=20),
                "hist/null_mean": _wb.Histogram(np.array(null_means), num_bins=20),
            })
            if parity_deltas:
                wb_run.log({
                    "hist/parity_delta": _wb.Histogram(
                        np.array(parity_deltas), num_bins=20
                    ),
                })
                summary_payload["median_parity_delta"] = round(
                    float(np.median(parity_deltas)), 4
                )
                summary_payload["p95_parity_delta"] = round(
                    float(np.percentile(parity_deltas, 95)), 4
                )
        wb_run.summary.update(summary_payload)
        wb_run.finish()


if __name__ == "__main__":
    main()
