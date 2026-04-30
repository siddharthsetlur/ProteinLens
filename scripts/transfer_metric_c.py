#!/usr/bin/env python3
"""Metric C — consistency of geometric primitives across databases.

For each feature with q<0.05 on geometric annotation (m7) AND ≥1 NMPFam hit,
compute the mean of the 44-D phi vector over (a) SwissProt activating residues
and (b) NMPFam activating residues, then report the cosine similarity. This
is the same construction as Figure 4C in the paper but across databases
instead of within a domain.

Inputs (read-only — paths under {analysis_dir}):
    permutation_null/*.json                    geometry q-values
    geometry_classifiers/{fid:04d}_meta.json   threshold_sae, half_w
    pipeline_state.json                        total_proteins, accession_index
    feature_max_activations.npy                gives num_features for the memmap shape
    protein_feature_maxes.npy                  raw float32 memmap (n_proteins, num_features)
    residue_activations/{accession}.npz        SwissProt SAE activations
    geometry_residue_profiles/{accession}.npz  SwissProt geometry profiles
    sequences.json                             SwissProt accession→sequence (optional)
    nmpfam/feature_maxes.npy                   NMPFam per-family max (.npy header — np.load OK)
    nmpfam/family_index.json                   NMPFam ID → row index
    nmpfam/families.json                       NMPFam consensus seqs (optional fallback)
    nmpfam/residue_activations/{F…}.npz
    nmpfam/geometry_residue_profiles/{F…}.npz

Output:
    {analysis_dir}/transfer_metrics/metric_C.json

Performance: half_w is constant across every saved GBM (audited: 100%
half_w=10), so the 44-D phi vector at residue p is identical for every
feature that fires there. We therefore compute phi once per residue per
protein, then score all features in one (n × K) bool mask · (n × 44) matmul.
This eliminates the ~6.4 s/protein cost of the per-(feature, position) call
to extract_local_feature_vector and brings the whole job down to ~3 h.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proteinlens.analysis.geometry.residue_features import extract_local_feature_vector  # noqa: E402

Q_SIG = 0.05
PHI_DIM = 44


def log(msg: str) -> None:
    """Single flushed-write logger so kubectl logs streams reliably."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# BH q-values (mirrors index_builder + transfer_metric_b)
# ---------------------------------------------------------------------------
def _bh(pvals: list[float | None]) -> list[float | None]:
    n = len(pvals)
    idx = [(i, p) for i, p in enumerate(pvals) if p is not None]
    if not idx:
        return [None] * n
    idx.sort(key=lambda x: x[1])
    m = len(idx)
    out: list[float | None] = [None] * n
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        orig, p = idx[rank]
        q = min(1.0, p * m / (rank + 1))
        if q < running_min:
            running_min = q
        out[orig] = running_min
    return out


def load_geometry_qvalues(analysis: Path) -> dict[int, float]:
    pn_dir = analysis / "permutation_null"
    pairs: list[tuple[int, float | None]] = []
    for p in sorted(pn_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        fid = int(d["feature_id"])
        pv = (d.get("p_values") or {}).get("geometry_prauc")
        pairs.append((fid, float(pv) if pv is not None else None))
    qvals = _bh([p for _, p in pairs])
    return {fid: q for (fid, _), q in zip(pairs, qvals) if q is not None}


# ---------------------------------------------------------------------------
# Profile + activation loaders
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


def _load_swiss_pmax_memmap(analysis: Path) -> tuple[np.memmap, dict[str, int]] | None:
    """Open protein_feature_maxes.npy as a raw float32 memmap.

    The file has no .npy header — survey.py writes it via ``np.memmap(...,
    mode="w+")``. Shape comes from pipeline_state.json (total_proteins) and
    feature_max_activations.npy (num_features).
    """
    pmax_path = analysis / "protein_feature_maxes.npy"
    state_path = analysis / "pipeline_state.json"
    fmax_path = analysis / "feature_max_activations.npy"
    if not (pmax_path.exists() and state_path.exists() and fmax_path.exists()):
        return None
    try:
        state = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    n_proteins = int(state.get("total_proteins") or 0)
    acc_to_idx_raw = state.get("accession_index") or {}
    if n_proteins <= 0 or not acc_to_idx_raw:
        return None
    num_features = int(np.load(fmax_path).shape[0])
    expected_size = n_proteins * num_features * 4
    actual_size = pmax_path.stat().st_size
    if expected_size != actual_size:
        log(
            f"  WARNING pmax size mismatch: expected {expected_size}, got {actual_size}; "
            f"refusing memmap, falling back."
        )
        return None
    pmax = np.memmap(pmax_path, dtype="float32", mode="r", shape=(n_proteins, num_features))
    acc_to_idx = {str(k): int(v) for k, v in acc_to_idx_raw.items()}
    return pmax, acc_to_idx


# ---------------------------------------------------------------------------
# Per-protein phi matrix
#
# Preferred path: load from {analysis}/residue_phi/{acc}.npz (or the parallel
# nmpfam dir), built once by scripts/build_residue_phi_cache.py. Falls back to
# on-the-fly recomputation only when the cache is missing — that path is the
# legacy slow loop (~30 CPU-h per full SwissProt run) and exists only so the
# script keeps working pre-cache.
# ---------------------------------------------------------------------------
def _load_phi_cache(npz_path: Path, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Load phi cache. Returns (phi_mat float64, phi_valid bool) or None.

    The caller passes ``n`` (= ``acts.shape[0]``) as a hint for slicing.
    The returned arrays may be shorter or longer than ``n`` — NMPFam
    geometry profiles are built from PDB structures while activations are
    over the full consensus sequence, so the lengths often differ. The
    caller is expected to set ``n_used = min(phi.shape[0], n)`` and slice
    both arrays before accumulation.
    """
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path) as z:
            phi = np.asarray(z["phi"])
            valid = np.asarray(z["valid"], dtype=bool)
    except Exception:  # noqa: BLE001
        return None
    if phi.ndim != 2 or phi.shape[1] != PHI_DIM:
        return None
    if valid.shape[0] != phi.shape[0]:
        return None
    return phi.astype(np.float64, copy=False), valid.copy()


def _compute_phi_matrix_from_profiles(
    ca: np.ndarray,
    profiles: dict,
    seq: str,
    n: int,
    hw: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: compute phi at every interior residue using the unchanged
    extractor. Used only when the residue_phi cache is missing — see
    scripts/build_residue_phi_cache.py for the one-shot that fills it."""
    phi_mat = np.zeros((n, PHI_DIM), dtype=np.float64)
    phi_valid = np.zeros(n, dtype=bool)
    if n < 2 * hw + 1:
        return phi_mat, phi_valid
    ca_n = ca[:n]
    seq_n = seq[:n] if seq else ""
    for pos in range(hw, n - hw):
        fv = extract_local_feature_vector(profiles, ca_n, pos, hw, sequence=seq_n)
        if fv is None:
            continue
        if not np.all(np.isfinite(fv)):
            continue
        phi_mat[pos] = fv
        phi_valid[pos] = True
    return phi_mat, phi_valid


# Backwards-compatible alias used by the test suite.
def _compute_phi_matrix(ca, profiles, seq, n, hw):
    return _compute_phi_matrix_from_profiles(ca, profiles, seq, n, hw)


def _accumulate_one(
    acts: np.ndarray,
    phi_mat: np.ndarray,
    phi_valid: np.ndarray,
    n: int,
    feat_arr: np.ndarray,
    thr_arr: np.ndarray,
    phi_sum: np.ndarray,
    phi_count: np.ndarray,
) -> None:
    """Score every feature against cached phi via a single matmul.

    Equivalent to the legacy double loop:
        for j, f in enumerate(feat_arr):
            for pos in range(hw, n-hw):
                if acts[pos, f] >= thr_arr[j] and phi_valid[pos]:
                    phi_sum[j] += phi_mat[pos]
                    phi_count[j] += 1
    """
    if n <= 0 or feat_arr.size == 0:
        return
    feat_local = feat_arr[feat_arr < acts.shape[1]]
    if feat_local.size == 0:
        return
    if feat_local.size != feat_arr.size:
        # Defensive: drop thresholds for any feature index past acts width.
        keep = feat_arr < acts.shape[1]
        thr_local = thr_arr[keep]
    else:
        thr_local = thr_arr
    sub = np.asarray(acts[:n, feat_local], dtype=np.float32)        # (n, K)
    mask = (sub >= thr_local[None, :]) & phi_valid[:n, None]         # (n, K)
    if not mask.any():
        return
    contrib = mask.T.astype(np.float64) @ phi_mat[:n]                # (K, 44)
    counts = mask.sum(axis=0).astype(np.int64)                       # (K,)
    if feat_local.size == feat_arr.size:
        phi_sum += contrib
        phi_count += counts
    else:
        idx = np.where(feat_arr < acts.shape[1])[0]
        phi_sum[idx] += contrib
        phi_count[idx] += counts


# ---------------------------------------------------------------------------
# Output payload (kept identical to the legacy schema)
# ---------------------------------------------------------------------------
def _build_payload(qvals, feat_arr, swiss_sum, swiss_cnt, nmp_sum, nmp_cnt,
                   partial: bool, stage: str) -> dict:
    per_feature: dict[str, dict] = {}
    cos_vals: list[float] = []
    for j, fid in enumerate(feat_arr):
        s_cnt, n_cnt = int(swiss_cnt[j]), int(nmp_cnt[j])
        rec = {
            "n_swiss_residues": s_cnt,
            "n_nmpfam_residues": n_cnt,
            "phi_dim": PHI_DIM,
            "geometry_q": qvals.get(int(fid)),
        }
        if s_cnt == 0 or n_cnt == 0:
            rec["phi_cosine"] = None
            per_feature[str(int(fid))] = rec
            continue
        phi_s = swiss_sum[j] / s_cnt
        phi_n = nmp_sum[j] / n_cnt
        denom = float(np.linalg.norm(phi_s) * np.linalg.norm(phi_n))
        cos = float(phi_s @ phi_n / denom) if denom > 0 else None
        rec["phi_cosine"] = cos
        if cos is not None:
            cos_vals.append(cos)
        per_feature[str(int(fid))] = rec

    summary = {
        "stage": stage,
        "partial": partial,
        "n_features_processed": len(per_feature),
        "n_features_with_phi_cosine": len(cos_vals),
        "median_cosine": float(np.median(cos_vals)) if cos_vals else None,
        "mean_cosine":   float(np.mean(cos_vals)) if cos_vals else None,
        "frac_above_0_5": (sum(1 for c in cos_vals if c > 0.5) / len(cos_vals)) if cos_vals else None,
        "frac_above_0_8": (sum(1 for c in cos_vals if c > 0.8) / len(cos_vals)) if cos_vals else None,
        "frac_above_0_9": (sum(1 for c in cos_vals if c > 0.9) / len(cos_vals)) if cos_vals else None,
        "cosine_quartiles": (
            np.quantile(cos_vals, [0.25, 0.5, 0.75]).tolist() if cos_vals else None
        ),
    }
    return {
        "metric": "C",
        "description": (
            "Cosine similarity between mean 44-D phi over SwissProt activating residues "
            "and mean phi over NMPFam activating residues, per feature. q<0.05 geometry filter."
        ),
        "summary": summary,
        "per_feature": per_feature,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis-dir", type=Path, required=True)
    ap.add_argument("--limit-features", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=2000,
                    help="Write a partial metric_C.json after this many proteins/families.")
    args = ap.parse_args()

    analysis = args.analysis_dir.resolve()
    if not analysis.is_dir():
        raise SystemExit(f"Not a directory: {analysis}")

    swiss_act_dir   = analysis / "residue_activations"
    swiss_geom_dir  = analysis / "geometry_residue_profiles"
    swiss_phi_dir   = analysis / "residue_phi"
    swiss_seq_path  = analysis / "sequences.json"
    nmp_dir         = analysis / "nmpfam"
    nmp_act_dir     = nmp_dir / "residue_activations"
    nmp_geom_dir    = nmp_dir / "geometry_residue_profiles"
    nmp_phi_dir     = nmp_dir / "residue_phi"
    nmp_fmax_path   = nmp_dir / "feature_maxes.npy"
    nmp_findex_path = nmp_dir / "family_index.json"
    nmp_fams_path   = nmp_dir / "families.json"
    gbm_dir         = analysis / "geometry_classifiers"

    for p in (swiss_act_dir, swiss_geom_dir, nmp_act_dir, nmp_geom_dir, gbm_dir):
        if not p.is_dir():
            raise SystemExit(f"Missing input directory: {p}")
    for p in (nmp_fmax_path, nmp_findex_path):
        if not p.exists():
            raise SystemExit(f"Missing input file: {p}")

    out_dir = analysis / "transfer_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metric_C.json"

    log(f"Metric C starting on {analysis}")

    # 1. Geometry q-values
    log("Loading geometry q-values …")
    qvals = load_geometry_qvalues(analysis)
    sig_set = {fid for fid, q in qvals.items() if q is not None and q < Q_SIG}
    log(f"  q<{Q_SIG}: {len(sig_set)} features (of {len(qvals)} with permutation null)")

    # 2. GBM meta thresholds + half_w (must be constant for the cached-phi
    #    optimisation; warn loudly if it isn't and recompute per-feature).
    log("Reading GBM meta files …")
    thresholds: dict[int, float] = {}
    half_w: dict[int, int] = {}
    for fid in sig_set:
        meta = gbm_dir / f"{fid:04d}_meta.json"
        if not meta.exists():
            continue
        try:
            m = json.loads(meta.read_text())
            thresholds[fid] = float(m["threshold_sae"])
            half_w[fid] = int(m["half_w"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    feature_set = sorted(thresholds.keys())
    log(f"  features with saved GBM meta: {len(feature_set)}")

    hw_values = {half_w[f] for f in feature_set}
    if len(hw_values) != 1:
        raise SystemExit(
            f"half_w not constant across features ({sorted(hw_values)}). "
            "The cached-phi optimisation requires a single half_w."
        )
    HW = int(next(iter(hw_values)))
    log(f"  half_w (constant): {HW}")

    # 3. NMPFam feature_maxes — proper .npy, np.load is fine.
    log("Loading nmpfam/feature_maxes.npy …")
    t0 = time.time()
    nmp_fmax = np.load(nmp_fmax_path)  # (n_families, n_features)
    log(f"  shape={nmp_fmax.shape}, dtype={nmp_fmax.dtype} ({time.time()-t0:.1f}s)")

    # 4. Filter features to those activating any NMPFam family
    feat_arr_full = np.asarray(feature_set, dtype=np.int64)
    feat_arr_full = feat_arr_full[feat_arr_full < nmp_fmax.shape[1]]
    thr_arr_full  = np.asarray([thresholds[int(f)] for f in feat_arr_full], dtype=np.float32)

    log("Filtering to features with ≥1 NMPFam family above threshold …")
    nmp_cols = nmp_fmax[:, feat_arr_full]                  # (n_families, k_feat)
    has_any = (nmp_cols >= thr_arr_full[None, :]).any(axis=0)
    keep_idx = np.where(has_any)[0]
    if args.limit_features:
        keep_idx = keep_idx[: args.limit_features]
    feat_arr = feat_arr_full[keep_idx]
    thr_arr  = thr_arr_full[keep_idx]
    log(f"  features in working set: {feat_arr.size}")

    # 5. Identify NMPFam families to process
    log("Identifying NMPFam families to scan …")
    fam_active_mask = (nmp_fmax[:, feat_arr] >= thr_arr[None, :]).any(axis=1)
    fam_rows = np.where(fam_active_mask)[0]
    family_index = json.loads(nmp_findex_path.read_text())
    row_to_fid = {int(v): k for k, v in family_index.items()}
    nmpfam_ids = [row_to_fid[int(r)] for r in fam_rows if int(r) in row_to_fid]
    log(f"  {len(nmpfam_ids)} NMPFam families to scan")
    del nmp_fmax  # free ~2 GB

    # 6. SwissProt accession list — proper memmap loader of pmax.
    log("Building SwissProt accession list …")
    swiss_acts_set  = {p.stem for p in swiss_act_dir.glob("*.npz")}
    swiss_geoms_set = {p.stem for p in swiss_geom_dir.glob("*.npz")}
    log(f"  on-disk: {len(swiss_acts_set)} act, {len(swiss_geoms_set)} geom")
    on_disk = swiss_acts_set & swiss_geoms_set

    pmax_loaded = _load_swiss_pmax_memmap(analysis)
    if pmax_loaded is not None:
        swiss_pmax, acc_to_idx = pmax_loaded
        log(f"  pmax memmap: {swiss_pmax.shape} ({swiss_pmax.dtype})")
        # Build candidate set from any-activates-above-threshold rows.
        pmax_cols = np.asarray(swiss_pmax[:, feat_arr])           # (n_proteins, K) float32
        prot_active = (pmax_cols >= thr_arr[None, :]).any(axis=1)
        del pmax_cols
        cand_rows = np.where(prot_active)[0]
        idx_to_acc = {v: k for k, v in acc_to_idx.items()}
        cand_accs = {idx_to_acc[int(r)] for r in cand_rows if int(r) in idx_to_acc}
        log(f"  candidate proteins (pmax pre-filter): {len(cand_accs)}")
        swiss_accs = sorted(cand_accs & on_disk)
        del swiss_pmax
    else:
        log("  WARNING pmax memmap not loadable — scanning every on-disk accession.")
        swiss_accs = sorted(on_disk)

    log(f"  SwissProt proteins to scan: {len(swiss_accs)}")

    # SwissProt sequence fallback (npz already stores `sequence` but if it's
    # empty we use sequences.json as backup).
    swiss_seq_fallback: dict[str, str] = {}
    if swiss_seq_path.exists():
        try:
            sj = json.loads(swiss_seq_path.read_text())
            if isinstance(sj, dict):
                swiss_seq_fallback = {k: v for k, v in sj.items() if isinstance(v, str)}
            elif isinstance(sj, list):
                for entry in sj:
                    if isinstance(entry, dict) and entry.get("accession"):
                        swiss_seq_fallback[entry["accession"]] = entry.get("sequence", "")
        except (json.JSONDecodeError, OSError):
            pass

    # NMPFam sequence fallback
    nm_seq_fallback: dict[str, str] = {}
    if nmp_fams_path.exists():
        for f in json.loads(nmp_fams_path.read_text()):
            sq = (f.get("consensus_sequence") or "").rstrip("*")
            if sq:
                nm_seq_fallback[f["ID"]] = sq

    # 7. Accumulators (vectorised — index by working-set position)
    K = feat_arr.size
    phi_swiss_sum   = np.zeros((K, PHI_DIM), dtype=np.float64)
    phi_swiss_count = np.zeros(K, dtype=np.int64)
    phi_nmp_sum     = np.zeros((K, PHI_DIM), dtype=np.float64)
    phi_nmp_count   = np.zeros(K, dtype=np.int64)

    def _save_partial(stage: str) -> None:
        ratios = _build_payload(
            qvals, feat_arr, phi_swiss_sum, phi_swiss_count,
            phi_nmp_sum, phi_nmp_count, partial=True, stage=stage,
        )
        out_path.write_text(json.dumps(ratios, indent=2))
        log(f"  checkpoint: wrote {out_path} ({stage})")

    # Glob the cache once per dir so we know up front how many proteins
    # have a precomputed phi (cephfs hygiene).
    swiss_phi_cached = {p.stem for p in swiss_phi_dir.glob("*.npz")} if swiss_phi_dir.is_dir() else set()
    nmp_phi_cached   = {p.stem for p in nmp_phi_dir.glob("*.npz")} if nmp_phi_dir.is_dir() else set()
    log(f"  residue_phi cache: SwissProt {len(swiss_phi_cached)}, NMPFam {len(nmp_phi_cached)}")

    # ------------------------------------------------------------------
    # 8. SwissProt loop — load phi from cache when available, else compute.
    # ------------------------------------------------------------------
    log(f"Scanning SwissProt ({len(swiss_accs)} proteins) …")
    pbar = tqdm(swiss_accs, desc="SwissProt", file=sys.stdout, mininterval=2.0)
    n_done = 0
    n_cache_hits = 0
    n_recomputed = 0
    for acc in pbar:
        try:
            with np.load(swiss_act_dir / f"{acc}.npz") as a:
                acts = np.asarray(a["activations"])  # force full read
        except Exception:  # noqa: BLE001
            continue
        n_acts = int(acts.shape[0])

        phi_loaded: tuple[np.ndarray, np.ndarray] | None = None
        if acc in swiss_phi_cached:
            phi_loaded = _load_phi_cache(swiss_phi_dir / f"{acc}.npz", n_acts)
        if phi_loaded is None:
            loaded = _load_profiles(swiss_geom_dir / f"{acc}.npz")
            if loaded is None:
                continue
            ca, profiles, seq = loaded
            if not seq:
                seq = swiss_seq_fallback.get(acc, "")
            n = int(min(len(ca), n_acts))
            if n < 2 * HW + 1:
                n_done += 1
                continue
            phi_mat, phi_valid = _compute_phi_matrix_from_profiles(ca, profiles, seq, n, HW)
            n_recomputed += 1
        else:
            phi_mat, phi_valid = phi_loaded
            n = int(min(phi_mat.shape[0], n_acts))
            if n < 2 * HW + 1:
                n_done += 1
                continue
            phi_mat = phi_mat[:n]
            phi_valid = phi_valid[:n]
            n_cache_hits += 1

        _accumulate_one(acts, phi_mat, phi_valid, n,
                        feat_arr, thr_arr,
                        phi_swiss_sum, phi_swiss_count)
        n_done += 1
        if args.checkpoint_every and n_done % args.checkpoint_every == 0:
            _save_partial(f"swissprot {n_done}/{len(swiss_accs)}")
    log(f"  SwissProt done: {n_done} proteins (cache hits={n_cache_hits}, "
        f"recomputed={n_recomputed}); total residues seen = {int(phi_swiss_count.sum())}")

    # ------------------------------------------------------------------
    # 9. NMPFam loop — same cache pattern.
    # ------------------------------------------------------------------
    log(f"Scanning NMPFams ({len(nmpfam_ids)} families) …")
    pbar = tqdm(sorted(nmpfam_ids), desc="NMPFams", file=sys.stdout, mininterval=2.0)
    n_done = 0
    n_cache_hits = 0
    n_recomputed = 0
    for nmpfid in pbar:
        try:
            with np.load(nmp_act_dir / f"{nmpfid}.npz") as a:
                acts = np.asarray(a["activations"])
        except Exception:  # noqa: BLE001
            continue
        n_acts = int(acts.shape[0])

        phi_loaded = None
        if nmpfid in nmp_phi_cached:
            phi_loaded = _load_phi_cache(nmp_phi_dir / f"{nmpfid}.npz", n_acts)
        if phi_loaded is None:
            loaded = _load_profiles(nmp_geom_dir / f"{nmpfid}.npz")
            if loaded is None:
                continue
            ca, profiles, seq = loaded
            if not seq:
                seq = nm_seq_fallback.get(nmpfid, "")
            n = int(min(len(ca), n_acts))
            if n < 2 * HW + 1:
                n_done += 1
                continue
            phi_mat, phi_valid = _compute_phi_matrix_from_profiles(ca, profiles, seq, n, HW)
            n_recomputed += 1
        else:
            phi_mat, phi_valid = phi_loaded
            n = int(min(phi_mat.shape[0], n_acts))
            if n < 2 * HW + 1:
                n_done += 1
                continue
            phi_mat = phi_mat[:n]
            phi_valid = phi_valid[:n]
            n_cache_hits += 1

        _accumulate_one(acts, phi_mat, phi_valid, n,
                        feat_arr, thr_arr,
                        phi_nmp_sum, phi_nmp_count)
        n_done += 1
        if args.checkpoint_every and n_done % args.checkpoint_every == 0:
            _save_partial(f"nmpfam {n_done}/{len(nmpfam_ids)}")
    log(f"  NMPFams done: {n_done} families (cache hits={n_cache_hits}, "
        f"recomputed={n_recomputed}); total residues seen = {int(phi_nmp_count.sum())}")

    # ------------------------------------------------------------------
    # 10. Final payload
    # ------------------------------------------------------------------
    payload = _build_payload(
        qvals, feat_arr, phi_swiss_sum, phi_swiss_count,
        phi_nmp_sum, phi_nmp_count, partial=False, stage="final",
    )
    out_path.write_text(json.dumps(payload, indent=2))
    s = payload["summary"]
    mc = s["median_cosine"]
    f8 = s["frac_above_0_8"]
    mc_s = f"{mc:.3f}" if mc is not None else "n/a"
    f8_s = f"{f8:.2f}" if f8 is not None else "n/a"
    log(
        f"Wrote {out_path}. n_features={s['n_features_with_phi_cosine']}/{s['n_features_processed']}, "
        f"median cosine={mc_s}, frac>0.8={f8_s}"
    )


if __name__ == "__main__":
    main()
