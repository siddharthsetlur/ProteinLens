"""Precompute per-feature activation-column caches for the refit-GBM null.

Why this exists
---------------
``scripts/compute_geometry_null_refit.py`` opens 500 per-protein
``residue_activations/*.npz`` files per feature to extract a single column
from each. On CephFS that is the dominant cost (metadata round-trips +
decompression of the full ``(n_res, n_features)`` matrix per protein just
to discard all but one column).

This script is a one-shot prepass that inverts the iteration: it reads each
protein's ``.npz`` exactly once and distributes its column-``fid`` data into
per-feature cache files. After the cache is built, the refit loader opens
**one** cache file per feature instead of 500.

Output layout
-------------
``<data-dir>/activation_col_cache/{fid:04d}.npz`` with keys:

* ``columns``    — float32, concatenated column-``fid`` activations for all
                   protein records belonging to this feature, in the exact
                   iteration order used by ``_load_protein_data`` (ascending
                   by per-protein max activation for this feature).
* ``offsets``    — int64 of length ``len(accessions) + 1``, cumulative
                   residue offsets so ``columns[offsets[i]:offsets[i+1]]`` is
                   the column for ``accessions[i]``.
* ``accessions`` — object array of protein accessions.
* ``meta``       — dict stored via ``np.savez`` with the cache-building
                   parameters (``max_proteins``, ``half_w``, version).

Design invariants
-----------------
* **The cache must reproduce ``_load_protein_data`` exactly.** The refit
  loader, when reading from the cache, must produce a ``protein_data`` list
  that is indistinguishable (accession order, column values, residue counts)
  from what the per-file loader would build.
* **Skip policy matches.** Proteins skipped by the refit loader (missing
  geom profile, missing activation file, ``n_residues < 2*half_w + 1``,
  ``.npz`` load error) are skipped here too.
* **Resume-safe.** A pre-existing ``{fid:04d}.npz`` is kept unless
  ``--force`` is given, so a crashed precompute can be restarted cheaply.
* **No mutation of any input.** Only writes under ``activation_col_cache/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# Single source of truth for _HALF_W and _CACHE_VERSION — import directly
# from the refit-null module so a change there doesn't silently diverge the
# cache's on-disk layout from what the loader expects.
from proteinlens.analysis.feature_pipeline.geometry_null_refit import (
    _CACHE_VERSION,
    _HALF_W,
)

logger = logging.getLogger("build_activation_column_cache")

_MIN_N_RESIDUES = 2 * _HALF_W + 1  # 21


def _compute_top_proteins_per_feature(
    act_matrix_full: np.ndarray,
    max_proteins: int,
) -> list[np.ndarray]:
    """For each feature fid, return the ordered top-``max_proteins`` row indices.

    Replicates the selection in ``_load_protein_data`` exactly — top-N by
    activation, final order is ascending activation (because ``argsort``
    returns ascending and we take the trailing ``max_proteins``).
    """
    n_features = act_matrix_full.shape[1]
    top_rows: list[np.ndarray] = []
    for fid in range(n_features):
        node_col = act_matrix_full[:, fid]
        active_rows = np.where(node_col > 0)[0]
        if len(active_rows) > max_proteins:
            top_idx = np.argsort(node_col[active_rows])[-max_proteins:]
            active_rows = active_rows[top_idx]
        top_rows.append(active_rows.astype(np.int64, copy=False))
    return top_rows


def _invert_top_proteins(
    top_rows: list[np.ndarray],
    row_to_acc: dict[int, str],
    act_file_map: dict[str, Path],
    geom_profile_files: set[str],
) -> tuple[dict[str, list[int]], dict[int, dict[str, int]]]:
    """Build protein→feature and feature→{acc: order_idx} indices.

    ``order_idx`` is the position each accession occupies in ``top_rows[fid]``.
    When we later write the per-feature cache, entries are sorted by that
    position to match ``_load_protein_data``'s iteration order byte-for-byte.
    """
    protein_features: dict[str, list[int]] = defaultdict(list)
    feature_entries: dict[int, dict[str, int]] = defaultdict(dict)
    for fid, rows in enumerate(top_rows):
        for order_idx, row_idx in enumerate(rows):
            acc = row_to_acc.get(int(row_idx))
            if acc is None:
                continue
            if acc not in geom_profile_files or acc not in act_file_map:
                continue
            protein_features[acc].append(fid)
            feature_entries[fid][acc] = order_idx
    return protein_features, feature_entries


def _glob_inputs(data_dir: Path) -> dict[str, Any]:
    """Return the same shared-data dict that the refit null's ``_setup_shared``
    would produce, but without memmapping any array until needed.
    """
    pipeline_state = json.loads((data_dir / "pipeline_state.json").read_text())
    acc_to_row: dict[str, int] = pipeline_state.get("accession_index", {})
    n_proteins = len(acc_to_row)

    feat_max_path = data_dir / "feature_max_activations.npy"
    n_features = int(len(np.load(feat_max_path, mmap_mode="r")))

    protein_maxes_path = data_dir / "protein_feature_maxes.npy"
    act_matrix_full = np.memmap(
        protein_maxes_path,
        dtype="float32",
        mode="r",
        shape=(n_proteins, n_features),
    )

    # geom profiles — same candidate locations as _setup_shared
    geom_profile_dir = data_dir / "geometry_residue_profiles"
    if not geom_profile_dir.is_dir():
        nested = data_dir / "geometry_enrichment" / "geometry_residue_profiles"
        if nested.is_dir():
            geom_profile_dir = nested
    geom_profile_files: set[str] = set()
    if geom_profile_dir.is_dir():
        geom_profile_files = {p.stem for p in geom_profile_dir.glob("*.npz")}

    # activations — merge residue_activations/ + interpro_residue_activations/
    act_file_map: dict[str, Path] = {}
    for sub in ("residue_activations", "interpro_residue_activations"):
        d = data_dir / sub
        if d.is_dir():
            for p in d.glob("*.npz"):
                if p.stem not in act_file_map:
                    act_file_map[p.stem] = p

    row_to_acc: dict[int, str] = {
        v: k for k, v in acc_to_row.items()
        if k in act_file_map and k in geom_profile_files
    }

    return {
        "n_features": n_features,
        "n_proteins": n_proteins,
        "act_matrix_full": act_matrix_full,
        "geom_profile_dir": geom_profile_dir,
        "geom_profile_files": geom_profile_files,
        "act_file_map": act_file_map,
        "row_to_acc": row_to_acc,
    }


def _already_cached(cache_dir: Path, fid: int) -> bool:
    return (cache_dir / f"{fid:04d}.npz").exists()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    # ``np.savez`` appends ``.npz`` to the filename if it is not already
    # present, which breaks the simple ``<path>.tmp`` + rename pattern (the
    # written file ends up as ``<path>.tmp.npz``). Pass an open file handle so
    # the bytes land at exactly the path we named, then rename atomically.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def _sweep_stale_tmp(cache_dir: Path) -> int:
    """Delete any ``.npz.tmp`` files left by a previously-killed precompute.

    Correctness is not affected — ``_already_cached`` checks only the final
    filename — but stale ``.tmp`` files consume inodes on a CephFS PVC that
    has already hit its quota once. Run once at startup.
    """
    n = 0
    for p in cache_dir.glob("*.npz.tmp"):
        try:
            p.unlink()
            n += 1
        except OSError:
            continue
    return n


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-dir>/activation_col_cache).",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=500,
        help="Top-N activating proteins per feature. Must match the value "
             "passed to compute_geometry_null_refit.py.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing per-feature cache files (default: skip).",
    )
    parser.add_argument(
        "--dry-run-proteins",
        type=int,
        default=0,
        help="Process only the first N proteins (debugging).",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(f"--data-dir {data_dir} is not a directory")

    cache_dir: Path = args.cache_dir or (data_dir / "activation_col_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_stale = _sweep_stale_tmp(cache_dir)
    if n_stale:
        logger.info("swept %d stale .npz.tmp files left from prior runs", n_stale)

    t0 = time.time()
    shared = _glob_inputs(data_dir)
    logger.info(
        "inputs: n_features=%d n_proteins=%d n_act_files=%d n_geom_files=%d row_to_acc=%d",
        shared["n_features"], shared["n_proteins"],
        len(shared["act_file_map"]), len(shared["geom_profile_files"]),
        len(shared["row_to_acc"]),
    )

    # 1. Compute top-N per feature.
    logger.info("computing top-%d per feature ...", args.max_proteins)
    top_rows = _compute_top_proteins_per_feature(
        shared["act_matrix_full"], args.max_proteins
    )

    # 2. Invert: which features need each protein, and at what position.
    logger.info("building protein→feature index ...")
    protein_features, feature_entries = _invert_top_proteins(
        top_rows,
        shared["row_to_acc"],
        shared["act_file_map"],
        shared["geom_profile_files"],
    )
    logger.info(
        "indexed %d proteins covering %d features (elapsed: %.1fs)",
        len(protein_features), len(feature_entries), time.time() - t0,
    )

    # 3. Determine which features still need caching (resume path).
    pending_fids = [
        fid for fid in feature_entries
        if args.force or not _already_cached(cache_dir, fid)
    ]
    logger.info(
        "%d features need caching (%d already present)",
        len(pending_fids), len(feature_entries) - len(pending_fids),
    )
    if not pending_fids:
        logger.info("nothing to do; exiting.")
        return

    pending_fid_set = set(pending_fids)

    # 4. For each protein we need, allocate per-feature in-memory buffers.
    # Structure: feature_buffers[fid] maps order_idx -> (accession, column).
    # Using an ordered dict keyed by order_idx lets us skip sort at the end.
    feature_buffers: dict[int, dict[int, tuple[str, np.ndarray]]] = {
        fid: {} for fid in pending_fid_set
    }

    # 5. One-pass read over the proteins that are referenced by pending features.
    needed_accs = sorted(
        acc for acc, fids in protein_features.items()
        if any(f in pending_fid_set for f in fids)
    )
    if args.dry_run_proteins > 0:
        needed_accs = needed_accs[: args.dry_run_proteins]
    logger.info("will read %d protein files", len(needed_accs))

    geom_profile_dir: Path = shared["geom_profile_dir"]
    act_file_map: dict[str, Path] = shared["act_file_map"]

    n_skipped_short = 0
    n_skipped_error = 0
    for acc in tqdm(needed_accs, desc="proteins"):
        act_path = act_file_map[acc]
        try:
            with np.load(act_path) as npz_act:
                act_mat = npz_act["activations"]
                # Determine n using the geom profile length — matches
                # _load_protein_data which does n = min(len(ca), shape[0]).
                with np.load(geom_profile_dir / f"{acc}.npz", allow_pickle=True) as gp:
                    ca_len = len(gp["ca"])
                n = min(ca_len, act_mat.shape[0])
                if n < _MIN_N_RESIDUES:
                    n_skipped_short += 1
                    continue
                # Extract columns for every pending feature that needs this protein.
                needed_here = [f for f in protein_features[acc] if f in pending_fid_set]
                if not needed_here:
                    continue
                # Bulk slice then per-feature copy — avoids materialising the whole
                # matrix twice and keeps per-column ops sequential.
                slab = np.asarray(act_mat[:n], dtype=np.float32)
            # npz context closed — `slab` is a detached copy.
        except (OSError, KeyError, ValueError) as e:
            logger.debug("skip %s: %s", acc, e)
            n_skipped_error += 1
            continue

        # Look up this protein's position in each feature's iteration.
        # feature_entries[fid] is a dict {acc: order_idx} — O(1) lookup.
        for fid in needed_here:
            order_idx = feature_entries[fid].get(acc)
            if order_idx is None:
                continue
            col = np.ascontiguousarray(slab[:, fid], dtype=np.float32)
            feature_buffers[fid][order_idx] = (acc, col)

    logger.info(
        "protein pass done: skipped_short=%d skipped_error=%d (elapsed: %.1fs)",
        n_skipped_short, n_skipped_error, time.time() - t0,
    )

    # 6. Write per-feature caches.
    logger.info("writing %d per-feature caches ...", len(pending_fid_set))
    for fid in tqdm(sorted(pending_fid_set), desc="features"):
        buf = feature_buffers[fid]
        if not buf:
            # No proteins survived for this feature — skip (refit will skip too).
            continue
        # Iterate in ascending-activation order (== ascending order_idx, since
        # feature_entries was built from top_rows which is already sorted).
        ordered_idx = sorted(buf.keys())
        entries = [buf[oi] for oi in ordered_idx]
        accessions = np.array([a for a, _ in entries], dtype=object)
        cols_list = [c for _, c in entries]
        columns = np.concatenate(cols_list).astype(np.float32, copy=False)
        offsets = np.zeros(len(entries) + 1, dtype=np.int64)
        np.cumsum([c.size for c in cols_list], out=offsets[1:])
        meta = {
            "feature_id": int(fid),
            "max_proteins": int(args.max_proteins),
            "half_w": int(_HALF_W),
            "min_n_residues": int(_MIN_N_RESIDUES),
            "cache_version": int(_CACHE_VERSION),
        }
        _atomic_savez(
            cache_dir / f"{fid:04d}.npz",
            columns=columns,
            offsets=offsets,
            accessions=accessions,
            meta=np.array([json.dumps(meta)], dtype=object),
        )

    logger.info(
        "cache build done: wrote %d features to %s (total elapsed: %.1fs)",
        len(pending_fid_set), cache_dir, time.time() - t0,
    )


if __name__ == "__main__":
    main()
