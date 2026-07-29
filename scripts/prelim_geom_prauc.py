#!/usr/bin/env python3
"""Geometry PR-AUC distribution from a finished geometry leg (stage 6c).

Reads ``geometry_enrichment/*.json`` and reports the distribution of
``geometric_residue_level.concordance.avg_precision`` -- the same value that
``compute_geometry_primary.py`` stores as ``geom_pr_auc`` and that the viewer
and ``paper_tables.py`` read as ``m7_score``.

These PR-AUCs are FINAL as soon as stage 6c finishes; the permutation null
adds no score, only the ``q < 0.05`` gate.  So this script says nothing about
Table 1's ``% Geometric`` and nothing about Table 2's published percentages --
both are q-gated, and the gate preferentially removes low-PR-AUC features, so
the unfiltered histogram below sits lower than the final Table 2 row.  What it
does give is the shape of the PR-AUC landscape and the fitted-feature count,
before the null lands.

stdlib-only (no numpy) so it runs in a plain data-access pod as well as in the
pipeline image.

Usage::

    python scripts/prelim_geom_prauc.py --data-dir /data/feature_data_esm35m_l6
"""

from __future__ import annotations

import argparse
import ast
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Identical to scripts/paper_tables.py, so the preview bins the same way the
# paper table does.  The last edge sits just above 1 so a PR-AUC of exactly
# 1.0 lands in the final bin rather than falling outside.
BIN_EDGES = [0.0, 0.3, 0.6, 1.0001]
BIN_LABELS = ["PR-AUC 0.0-0.3", "PR-AUC 0.3-0.6", "PR-AUC > 0.6"]


def npy_length(path: Path) -> int:
    """Element count of a 1-D ``.npy`` file, parsed from its header alone.

    Avoids a numpy dependency so the script runs in pods without the conda
    environment.  Handles both the v1 (2-byte header length) and v2 (4-byte)
    layouts.
    """
    with path.open("rb") as fh:
        if fh.read(6) != b"\x93NUMPY":
            raise ValueError(f"{path} is not a .npy file")
        major = fh.read(1)[0]
        fh.read(1)  # minor version, unused
        width = 2 if major == 1 else 4
        (hlen,) = struct.unpack("<H" if major == 1 else "<I", fh.read(width))
        header = ast.literal_eval(fh.read(hlen).decode("latin1").strip())
    n = 1
    for dim in header["shape"]:
        n *= dim
    return n


def load_pr_auc(geom_dir: Path, workers: int) -> dict[int, float]:
    """``{feature_id: geometry PR-AUC}`` over every fitted feature.

    One directory listing followed by parallel reads -- no per-file
    ``exists()``, which matters on cephfs.  Features whose concordance block
    has no ``avg_precision`` (GBM not fitted) are simply absent from the
    result, matching ``_load_enrichment_scores`` in
    ``compute_geometry_primary.py``.
    """
    paths = [
        p for p in geom_dir.iterdir()
        if p.suffix == ".json" and p.name != "summary.json"
    ]

    def _read(path: Path) -> tuple[int, float] | None:
        try:
            d = json.loads(path.read_text())
            ap = (
                d.get("geometric_residue_level", {})
                .get("concordance", {})
                .get("avg_precision")
            )
            return (int(d["feature_id"]), float(ap)) if ap is not None else None
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return {r[0]: r[1] for r in ex.map(_read, paths) if r is not None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir", type=Path, required=True,
        help="Pipeline output dir (e.g. /data/feature_data_esm35m_l6)",
    )
    ap.add_argument("--workers", type=int, default=16, help="Parallel JSON readers")
    args = ap.parse_args()

    geom_dir = args.data_dir / "geometry_enrichment"
    if not geom_dir.is_dir():
        raise SystemExit(f"no geometry_enrichment/ under {args.data_dir}")

    geom = load_pr_auc(geom_dir, args.workers)
    if not geom:
        raise SystemExit(f"no fitted features with avg_precision in {geom_dir}")

    n_dict = npy_length(args.data_dir / "feature_max_activations.npy")
    vals = sorted(geom.values())
    n = len(vals)

    counts = [0] * (len(BIN_EDGES) - 1)
    for v in vals:
        for b in range(len(counts)):
            if BIN_EDGES[b] <= v < BIN_EDGES[b + 1]:
                counts[b] += 1
                break

    def pct(frac: float) -> float:
        """Percentile by nearest-rank on the sorted values."""
        return vals[min(n - 1, int(frac * n))]

    print(f"{args.data_dir}")
    print(f"  dictionary : {n_dict} features")
    print(f"  GBM fitted : {n} ({100.0 * n / n_dict:.2f}% of dictionary)")
    print(
        f"  PR-AUC     : p25 {pct(0.25):.3f}   median {pct(0.50):.3f}   "
        f"p75 {pct(0.75):.3f}   p95 {pct(0.95):.3f}   max {vals[-1]:.3f}"
    )
    print()
    print("PR-AUC bins over ALL fitted features -- NOT Table 2, which is")
    print("restricted to q < 0.05 and so will sit higher than this:")
    print()
    for label, c in zip(BIN_LABELS, counts):
        print(
            f"  {label:<16} {c:>6}   {100.0 * c / n:>6.2f}% of fitted   "
            f"{100.0 * c / n_dict:>6.2f}% of dictionary"
        )
    print()
    for thr in (0.3, 0.5, 0.6, 0.8):
        k = sum(1 for v in vals if v >= thr)
        print(
            f"  PR-AUC >= {thr:.1f} : {k:>6}   {100.0 * k / n:>6.2f}% of fitted   "
            f"{100.0 * k / n_dict:>6.2f}% of dictionary"
        )


if __name__ == "__main__":
    main()
