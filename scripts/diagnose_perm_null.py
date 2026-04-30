#!/usr/bin/env python3
"""One-shot diagnostic: run process_feature on a single fid with full traceback.

Usage (inside the pod, assuming /workspace is the repo root and /data holds the
feature_data dir):

    python /workspace/scripts/diagnose_perm_null.py /data/feature_data_relu_l4 0
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.compute_permutation_null as cpn  # noqa: E402


def build_shared(data_dir: Path) -> dict:
    feat_max_arr = np.load(data_dir / "feature_max_activations.npy")
    state = json.loads((data_dir / "pipeline_state.json").read_text())
    acc_to_idx = state.get("accession_index", {})

    geom_profile_dir = data_dir / "geometry_residue_profiles"
    geom_profile_files = {p.stem for p in geom_profile_dir.glob("*.npz")} if geom_profile_dir.is_dir() else set()

    act_file_map: dict[str, Path] = {}
    for name in ("residue_activations", "interpro_residue_activations"):
        d = data_dir / name
        if d.is_dir():
            for p in d.glob("*.npz"):
                act_file_map.setdefault(p.stem, p)

    row_to_acc = {
        v: k for k, v in acc_to_idx.items()
        if k in geom_profile_files and k in act_file_map
    }

    protein_maxes = data_dir / "protein_feature_maxes.npy"
    act_matrix_full = np.memmap(
        protein_maxes, dtype="float32", mode="r",
        shape=(len(acc_to_idx), len(feat_max_arr)),
    )

    interpro_cache = data_dir / "interpro_cache"
    cath_cache = data_dir / "cath_enrichment" / "cache"
    gbm_dir = data_dir / "geometry_classifiers"
    features_dir = data_dir / "features"

    return {
        "feat_max_arr": feat_max_arr,
        "interpro_file_set": {p.stem for p in interpro_cache.glob("*.json")} if interpro_cache.is_dir() else set(),
        "cath_file_set": {p.stem for p in cath_cache.glob("*.json")} if cath_cache.is_dir() else set(),
        "geom_profile_files": geom_profile_files,
        "geom_profile_dir": geom_profile_dir,
        "act_file_map": act_file_map,
        "gbm_files": {p.stem.replace("_gbm", "") for p in gbm_dir.glob("*_gbm.pkl")} if gbm_dir.is_dir() else set(),
        "motif_k": 3,
        "feature_json_fids": {int(p.stem) for p in features_dir.glob("*.json") if p.stem.isdigit()},
        "act_matrix_full": act_matrix_full,
        "row_to_acc": row_to_acc,
        "include_pwm": True,
        "pwm_act_quantile": 0.80,
    }


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: diagnose_perm_null.py <data_dir> <fid>", file=sys.stderr)
        sys.exit(2)
    data_dir = Path(sys.argv[1])
    fid = int(sys.argv[2])

    shared = build_shared(data_dir)
    print(f"[diag] fid={fid} data_dir={data_dir}")
    print(f"[diag] interpro_cache={len(shared['interpro_file_set'])}"
          f" cath_cache={len(shared['cath_file_set'])}"
          f" geom_profiles={len(shared['geom_profile_files'])}"
          f" act_files={len(shared['act_file_map'])}"
          f" gbm_files={len(shared['gbm_files'])}")

    try:
        result = cpn.process_feature(fid, data_dir, 3, 42, shared)
    except Exception:
        print("=" * 60)
        print("TRACEBACK:")
        traceback.print_exc()
        sys.exit(1)

    if result is None:
        print("[diag] returned None (feature skipped)")
    else:
        print(f"[diag] OK. observed keys: {sorted(result['observed'].keys())}")


if __name__ == "__main__":
    main()
