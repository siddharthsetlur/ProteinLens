#!/usr/bin/env python
"""Compute per-(feature, family) PR-AUC of SwissProt-trained GBM on NMPFam residues.

Single-process, low-memory. Writes incrementally to CSV (resume-safe).

For each layer:
  - Filter features to q geom < 0.05 (via metric_B.json per_feature keys).
  - For each feature JSON in nmpfam_enrichment/, iterate hits.
    - PR-AUC = average_precision_score(sae > sae_threshold, geom_prob_profile)
  - Append rows (feature_id, family_id, pr_auc, n_residues, n_active) to CSV.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

# Limit BLAS threads BEFORE numpy import — keeps memory + CPU bounded.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import orjson
from sklearn.metrics import average_precision_score


def existing_done_fids(csv_path: Path) -> set[int]:
    """Read fids already processed from a partial CSV (resume support)."""
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            done.add(int(row["feature_id"]))
    return done


def run_layer(layer: int, base_dir: Path, out_dir: Path, threshold: float):
    enrichment_dir = base_dir / "analysis" / "nmpfam" / "nmpfam_enrichment"
    metric_b_path = base_dir / "analysis" / "transfer_metrics" / "metric_B.json"

    print(f"[layer {layer}] reading q-filter from {metric_b_path}", flush=True)
    mb = orjson.loads(metric_b_path.read_bytes())
    qsig_fids = set(int(f) for f in mb["per_feature"].keys())
    del mb
    gc.collect()

    files = sorted(enrichment_dir.glob("*.json"))
    files = [f for f in files if f.stem.isdigit() and int(f.stem) in qsig_fids]
    print(f"[layer {layer}] {len(files)} feature jsons after q-filter", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"layer_{layer}_per_family_prauc.csv"
    progress_path = out_dir / f"layer_{layer}_progress.txt"

    # Resume: skip features already processed
    done_fids = existing_done_fids(csv_path)
    if done_fids:
        print(f"[layer {layer}] resuming — {len(done_fids)} fids already in CSV", flush=True)
    files = [f for f in files if int(f.stem) not in done_fids]

    write_header = not csv_path.exists()
    fh = open(csv_path, "a", newline="")
    writer = csv.writer(fh)
    if write_header:
        writer.writerow(["feature_id", "family_id", "pr_auc", "n_residues", "n_active"])
        fh.flush()

    t0 = time.time()
    pairs_written = 0
    feats_done = 0

    for fpath in files:
        try:
            with open(fpath, "rb") as f:
                d = orjson.loads(f.read())
        except Exception as e:
            print(f"[layer {layer}] skipping {fpath.name}: {e}", flush=True)
            feats_done += 1
            continue

        sae_thresh = d.get("activation_threshold_sae")
        fid = int(d["feature_id"])
        rows = []
        for hit in d.get("nmpfam_hits", []):
            sae = np.asarray(hit["sae_activation_profile"], dtype=np.float32)
            y = (sae > sae_thresh).astype(np.int8)
            n_active = int(y.sum())
            n_res = int(len(y))
            if n_active == 0 or n_active == n_res:
                continue
            geom = np.asarray(hit["geom_prob_profile"], dtype=np.float32)
            pr = float(average_precision_score(y, geom))
            rows.append([fid, hit["family_id"], f"{pr:.5f}", n_res, n_active])
            del sae, y, geom

        if rows:
            writer.writerows(rows)
            pairs_written += len(rows)
        feats_done += 1
        # Drop and clean up
        d = None
        rows = None

        # Light bookkeeping every 100 feats
        if feats_done % 100 == 0:
            fh.flush()
            os.fsync(fh.fileno())
            elapsed = time.time() - t0
            rate = feats_done / elapsed if elapsed > 0 else 0
            eta = (len(files) - feats_done) / rate if rate > 0 else float("inf")
            msg = (f"[layer {layer}] {feats_done}/{len(files)} feats  "
                   f"({rate:.1f}/s, ETA {eta/60:.1f}m)  pairs={pairs_written}")
            print(msg, flush=True)
            progress_path.write_text(msg + "\n")
        if feats_done % 500 == 0:
            gc.collect()

    fh.flush()
    os.fsync(fh.fileno())
    fh.close()

    elapsed = time.time() - t0
    print(f"[layer {layer}] done in {elapsed/60:.1f}m. {feats_done} feats, {pairs_written} pairs.", flush=True)


def summarize(out_dir: Path, layer: int, threshold: float):
    csv_path = out_dir / f"layer_{layer}_per_family_prauc.csv"
    if not csv_path.exists():
        print(f"[layer {layer}] no CSV yet")
        return
    fam_max: dict[str, float] = {}
    n_pairs = 0
    with open(csv_path, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fam = row["family_id"]
            pr = float(row["pr_auc"])
            n_pairs += 1
            cur = fam_max.get(fam)
            if cur is None or pr > cur:
                fam_max[fam] = pr
    n_strong = sum(1 for v in fam_max.values() if v > threshold)
    print(f"[layer {layer}] pairs={n_pairs}, unique families={len(fam_max)}, "
          f"families with max PR-AUC > {threshold}: {n_strong}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True, choices=[2, 4, 6])
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/home/s2721407/Desktop/ProteinLens/per_family_prauc_cache"))
    ap.add_argument("--summarize-only", action="store_true",
                    help="Skip computation, just summarize existing CSV.")
    args = ap.parse_args()

    bases = {
        2: Path("/home/s2721407/Desktop/ProteinLens/trained_models/layer_2/firm-sweep-3"),
        4: Path("/home/s2721407/Desktop/ProteinLens/trained_models/layer_4/frosty-sweep-15"),
        6: Path("/home/s2721407/Desktop/ProteinLens/trained_models/layer_6/major-sweep-15"),
    }

    if not args.summarize_only:
        run_layer(args.layer, bases[args.layer], args.out_dir, args.threshold)
    summarize(args.out_dir, args.layer, args.threshold)


if __name__ == "__main__":
    main()
