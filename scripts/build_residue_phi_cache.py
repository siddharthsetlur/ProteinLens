#!/usr/bin/env python3
"""Build the per-residue 44-D phi cache.

Walks ``{analysis_dir}/geometry_residue_profiles/`` (and the parallel
NMPFam directory) and, for every protein/family, materialises the 44-D
local-geometry vector phi(p) at every interior residue position. Writes
one ``.npz`` per accession under ``{analysis_dir}/residue_phi/`` with:

    phi   : (n, 44) float32 — phi[p] at every position; rows where phi is
            invalid (boundary, None, or non-finite) are zeros.
    valid : (n,)   bool     — True iff phi[p] was successfully computed
                              and finite.
    half_w: scalar int      — the window size used (constant = 10).

This is a one-shot. Once written, every downstream geometry analysis
(transfer_metric_c, future permutation-null variants, geometry_residue_
enrichment, etc.) loads phi instead of recomputing the O(n²) contact-
density step. Without the cache, each consumer pays ~30 CPU-hours per
run for the same φ.

Usage:
    python scripts/build_residue_phi_cache.py \
        --analysis-dir /data/feature_data_relu_l4 \
        --half-w 10 \
        --workers 4

Idempotent: skips accessions whose target ``residue_phi/{acc}.npz``
already exists. Use ``--rebuild`` to overwrite.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proteinlens.analysis.geometry.residue_features import extract_local_feature_vector  # noqa: E402

PHI_DIM = 44


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Loader for the geometry_residue_profiles npz schema (matches geometry_features.py:138)
# ---------------------------------------------------------------------------
def _load_profiles(npz_path: Path) -> tuple[np.ndarray, dict, str] | None:
    try:
        with np.load(npz_path, allow_pickle=True) as g:
            ca = np.asarray(g["ca"])
            profiles = {
                k: np.asarray(g[k]) for k in
                ("curvature", "torsion", "planarity", "tangents", "helix_mask", "categories")
            }
            seq_arr = g.get("sequence", np.array([""]))
            seq = str(seq_arr[0]) if len(seq_arr) > 0 else ""
        return ca, profiles, seq
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Per-protein phi computation. Pure function — one source of truth for the
# entire pipeline. Writes phi (n, 44) float32 + valid (n,) bool.
# ---------------------------------------------------------------------------
def _compute_phi_npz(
    geom_npz: Path,
    out_npz: Path,
    half_w: int,
) -> tuple[str, int, int]:
    """Returns (accession, n_residues, n_valid). 0/0 means skipped/failed."""
    acc = geom_npz.stem
    loaded = _load_profiles(geom_npz)
    if loaded is None:
        return acc, 0, 0
    ca, profiles, seq = loaded
    n = int(len(ca))
    if n < 2 * half_w + 1:
        # Still write an empty cache so the consumer can ``np.load`` it.
        phi = np.zeros((n, PHI_DIM), dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
    else:
        phi = np.zeros((n, PHI_DIM), dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        ca_n = ca[:n]
        seq_n = seq[:n] if seq else ""
        for pos in range(half_w, n - half_w):
            fv = extract_local_feature_vector(profiles, ca_n, pos, half_w, sequence=seq_n)
            if fv is None:
                continue
            if not np.all(np.isfinite(fv)):
                continue
            phi[pos] = fv.astype(np.float32, copy=False)
            valid[pos] = True
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, phi=phi, valid=valid, half_w=np.int32(half_w))
    return acc, n, int(valid.sum())


# ---------------------------------------------------------------------------
# Worker entry-point — used by ProcessPoolExecutor. We re-import inside the
# worker only if it crashed across a fork (it doesn't here, but keeping the
# pattern simple).
# ---------------------------------------------------------------------------
def _worker(args: tuple[str, str, int]) -> tuple[str, int, int]:
    geom_str, out_str, half_w = args
    return _compute_phi_npz(Path(geom_str), Path(out_str), half_w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _process_dir(
    label: str,
    geom_dir: Path,
    out_dir: Path,
    half_w: int,
    workers: int,
    rebuild: bool,
) -> None:
    if not geom_dir.is_dir():
        log(f"  {label}: {geom_dir} not a directory — skipping.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Glob once per dir (cephfs hygiene).
    log(f"  {label}: globbing {geom_dir} …")
    t0 = time.time()
    geom_files = sorted(geom_dir.glob("*.npz"))
    log(f"  {label}: {len(geom_files)} candidates ({time.time()-t0:.1f}s)")

    existing = {p.stem for p in out_dir.glob("*.npz")} if not rebuild else set()
    todo = [p for p in geom_files if p.stem not in existing]
    log(f"  {label}: {len(todo)} to compute (skipping {len(geom_files) - len(todo)} already cached)")
    if not todo:
        return

    args = [(str(p), str(out_dir / p.name), half_w) for p in todo]

    n_done = 0
    n_residues = 0
    n_valid = 0
    if workers <= 1:
        for arg in tqdm(args, desc=label, file=sys.stdout, mininterval=2.0):
            _, n, v = _worker(arg)
            n_residues += n
            n_valid += v
            n_done += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_worker, a) for a in args]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=label, file=sys.stdout, mininterval=2.0):
                _, n, v = fut.result()
                n_residues += n
                n_valid += v
                n_done += 1
    log(f"  {label}: wrote {n_done} files, {n_residues} residues seen, "
        f"{n_valid} valid phi rows.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis-dir", type=Path, required=True)
    ap.add_argument("--half-w", type=int, default=10,
                    help="Window half-size used by extract_local_feature_vector. "
                         "Must match the value used by every consumer; the saved "
                         "GBM half_w is 10 across all features in this project.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--rebuild", action="store_true",
                    help="Overwrite existing residue_phi/{acc}.npz files.")
    ap.add_argument("--swissprot-only", action="store_true",
                    help="Skip the NMPFam pass.")
    ap.add_argument("--nmpfam-only", action="store_true",
                    help="Skip the SwissProt pass.")
    args = ap.parse_args()

    analysis = args.analysis_dir.resolve()
    if not analysis.is_dir():
        raise SystemExit(f"Not a directory: {analysis}")

    log(f"build_residue_phi_cache starting on {analysis}")
    log(f"  half_w={args.half_w}, workers={args.workers}, rebuild={args.rebuild}")

    if not args.nmpfam_only:
        _process_dir(
            label="SwissProt",
            geom_dir=analysis / "geometry_residue_profiles",
            out_dir=analysis / "residue_phi",
            half_w=args.half_w,
            workers=args.workers,
            rebuild=args.rebuild,
        )

    if not args.swissprot_only:
        _process_dir(
            label="NMPFam",
            geom_dir=analysis / "nmpfam" / "geometry_residue_profiles",
            out_dir=analysis / "nmpfam" / "residue_phi",
            half_w=args.half_w,
            workers=args.workers,
            rebuild=args.rebuild,
        )

    log("Done.")


if __name__ == "__main__":
    main()
