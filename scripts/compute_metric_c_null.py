#!/usr/bin/env python3
"""Random-subset null distribution for Metric C.

Metric C reports the cosine between mean-phi over SwissProt-active residues
and mean-phi over NMPFam-active residues, per feature. The raw cosine is
inflated by common-mode structure in the 44-D phi vector — natural-protein
amino-acid composition + bulk contact-density geometry are similar across
*any* two residue subsets of two natural databases, so the cosine of two
means is anchored well above 0 even for non-discriminating features.

This script estimates that anchor empirically. For each feature *f*:

    1. Look up the actual N_swiss and N_nmpfam from metric_C.json.
    2. Sample N_swiss random valid SwissProt residues and N_nmpfam random
       valid NMPFam residues, uniformly across all proteins/families.
    3. Compute the cosine between the means of those two random subsets.
    4. Repeat ``--n-trials`` times. Record the null distribution.

Output is ``{analysis_dir}/transfer_metrics/metric_C_null.json``, mirroring
the metric_C schema with extra ``null_median`` / ``null_p95`` / ``delta_cos``
/ ``p_value`` fields per feature, plus a population-level summary that
collapses each feature's observed cosine relative to its own null.

Usage::

    python scripts/compute_metric_c_null.py \\
        --analysis-dir /data/feature_data_relu_l4 \\
        --n-trials 100 \\
        --max-sample-per-side 50000 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

PHI_DIM = 44


def log(msg: str) -> None:
    """Single flushed-write logger so kubectl logs streams reliably."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Cache loader: stack every residue_phi/{acc}.npz into one big (M, 44) matrix
# of valid rows only. Discards invalid rows (boundary / non-finite).
# ---------------------------------------------------------------------------
def _load_all_phi(phi_dir: Path) -> np.ndarray:
    if not phi_dir.is_dir():
        raise SystemExit(f"Missing residue_phi dir: {phi_dir}")
    files = sorted(phi_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No residue_phi npz files in {phi_dir}")

    # First pass: count total valid rows so we can pre-allocate.
    total = 0
    for f in tqdm(files, desc=f"counting valid in {phi_dir.name}",
                  file=sys.stdout, mininterval=2.0):
        try:
            with np.load(f) as z:
                total += int(np.asarray(z["valid"], dtype=bool).sum())
        except Exception:  # noqa: BLE001
            continue

    out = np.empty((total, PHI_DIM), dtype=np.float32)
    cursor = 0
    for f in tqdm(files, desc=f"loading {phi_dir.name}",
                  file=sys.stdout, mininterval=2.0):
        try:
            with np.load(f) as z:
                phi = np.asarray(z["phi"], dtype=np.float32)
                valid = np.asarray(z["valid"], dtype=bool)
        except Exception:  # noqa: BLE001
            continue
        if phi.ndim != 2 or phi.shape[1] != PHI_DIM:
            continue
        if valid.shape[0] != phi.shape[0]:
            continue
        rows = phi[valid]
        n = rows.shape[0]
        if n == 0:
            continue
        out[cursor:cursor + n] = rows
        cursor += n
    return out[:cursor]


# ---------------------------------------------------------------------------
# Per-feature null. Vectorised over n_trials.
# ---------------------------------------------------------------------------
def _null_cosines(
    phi_swiss: np.ndarray,           # (M_swiss, 44) float32
    phi_nmp: np.ndarray,             # (M_nmp, 44) float32
    n_swiss: int,
    n_nmp: int,
    n_trials: int,
    rng: np.random.Generator,
    max_per_side: int,
) -> np.ndarray:
    """Returns shape (n_trials,) float64 — one cosine per random draw."""
    # Cap the per-trial sample size for compute. Above ~50k samples the null
    # variance is < 1% of the mean already (CLT), so this is a near-lossless
    # bound on runtime for the largest features.
    n_s = min(int(n_swiss), max_per_side, phi_swiss.shape[0])
    n_n = min(int(n_nmp), max_per_side, phi_nmp.shape[0])
    if n_s <= 0 or n_n <= 0:
        return np.zeros(n_trials, dtype=np.float64)

    out = np.empty(n_trials, dtype=np.float64)
    for t in range(n_trials):
        idx_s = rng.integers(0, phi_swiss.shape[0], size=n_s)
        idx_n = rng.integers(0, phi_nmp.shape[0], size=n_n)
        mu_s = phi_swiss[idx_s].mean(axis=0).astype(np.float64)
        mu_n = phi_nmp[idx_n].mean(axis=0).astype(np.float64)
        denom = float(np.linalg.norm(mu_s) * np.linalg.norm(mu_n))
        out[t] = float(mu_s @ mu_n / denom) if denom > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _summarise(per_feature: dict) -> dict:
    obs    = np.array([v["observed_cos"]      for v in per_feature.values() if v["observed_cos"]      is not None], dtype=float)
    nullm  = np.array([v["null_median"]       for v in per_feature.values() if v["null_median"]       is not None], dtype=float)
    delta  = np.array([v["delta_cos"]         for v in per_feature.values() if v["delta_cos"]         is not None], dtype=float)
    pvals  = np.array([v["p_value"]           for v in per_feature.values() if v["p_value"]           is not None], dtype=float)

    # "Significant" = observed cosine above null p95 of its OWN feature.
    # Restricted to features where both fields exist (i.e. ones with a null).
    sig = np.array(
        [v["observed_cos"] > v["null_p95"]
         for v in per_feature.values()
         if v["observed_cos"] is not None and v["null_p95"] is not None],
        dtype=bool,
    )

    def _pct(a, q):
        return float(np.percentile(a, q)) if a.size else None

    return {
        "n_features_processed":        len(per_feature),
        "n_features_with_null":        int(nullm.size),
        "median_observed_cos":         float(np.median(obs))   if obs.size   else None,
        "median_null_cos":             float(np.median(nullm)) if nullm.size else None,
        "median_delta_cos":            float(np.median(delta)) if delta.size else None,
        "mean_delta_cos":              float(np.mean(delta))   if delta.size else None,
        "delta_quartiles":             [_pct(delta, 25), _pct(delta, 50), _pct(delta, 75)],
        "frac_observed_above_null_p95": float(sig.mean())      if sig.size   else None,
        "median_p_value":              float(np.median(pvals)) if pvals.size else None,
        "frac_p_value_below_0_05":     float((pvals < 0.05).mean()) if pvals.size else None,
        "frac_p_value_below_0_01":     float((pvals < 0.01).mean()) if pvals.size else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis-dir", type=Path, required=True)
    ap.add_argument("--n-trials", type=int, default=100,
                    help="Random-subset draws per feature (default 100).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-sample-per-side", type=int, default=50_000,
                    help="Cap N_swiss / N_nmpfam per trial for compute. "
                         "Above ~50k the null is already CLT-tight; raising "
                         "this gains < 1% precision and adds linear cost.")
    ap.add_argument("--limit-features", type=int, default=None,
                    help="Process only the first N features (debugging).")
    args = ap.parse_args()

    analysis = args.analysis_dir.resolve()
    if not analysis.is_dir():
        raise SystemExit(f"Not a directory: {analysis}")

    metric_c_path = analysis / "transfer_metrics" / "metric_C.json"
    if not metric_c_path.exists():
        raise SystemExit(f"Missing metric_C.json — run transfer_metric_c.py first: {metric_c_path}")

    out_path = analysis / "transfer_metrics" / "metric_C_null.json"

    log(f"compute_metric_c_null starting on {analysis}")
    log(f"  n_trials={args.n_trials}, seed={args.seed}, max_sample_per_side={args.max_sample_per_side}")

    # 1. Load metric_C.json — gives us the observed cosine + matched N per feature.
    log("Loading metric_C.json …")
    mC = json.loads(metric_c_path.read_text())
    per_feature_obs = mC["per_feature"]
    log(f"  features in metric_C: {len(per_feature_obs)}")

    # 2. Pre-load both phi caches into one big (M, 44) matrix each.
    log("Pre-loading SwissProt residue_phi cache …")
    t0 = time.time()
    phi_swiss = _load_all_phi(analysis / "residue_phi")
    log(f"  swiss valid rows: {phi_swiss.shape[0]:,} ({time.time()-t0:.1f}s, {phi_swiss.nbytes/1e9:.2f} GB)")

    log("Pre-loading NMPFam residue_phi cache …")
    t0 = time.time()
    phi_nmp = _load_all_phi(analysis / "nmpfam" / "residue_phi")
    log(f"  nmpfam valid rows: {phi_nmp.shape[0]:,} ({time.time()-t0:.1f}s, {phi_nmp.nbytes/1e9:.2f} GB)")

    # 3. Per-feature null draw.
    rng = np.random.default_rng(args.seed)
    fids = list(per_feature_obs.keys())
    if args.limit_features:
        fids = fids[: args.limit_features]
    log(f"Computing null for {len(fids)} features …")

    per_feature_null: dict[str, dict] = {}
    pbar = tqdm(fids, desc="null", file=sys.stdout, mininterval=2.0)
    for fid in pbar:
        rec = per_feature_obs[fid]
        observed = rec.get("phi_cosine")
        n_s = int(rec.get("n_swiss_residues", 0) or 0)
        n_n = int(rec.get("n_nmpfam_residues", 0) or 0)
        if observed is None or n_s == 0 or n_n == 0:
            per_feature_null[fid] = {
                "observed_cos":     observed,
                "n_swiss_residues": n_s,
                "n_nmpfam_residues": n_n,
                "null_median":      None,
                "null_p5":          None,
                "null_p95":         None,
                "null_p99":         None,
                "delta_cos":        None,
                "p_value":          None,
                "n_trials":         args.n_trials,
            }
            continue

        nulls = _null_cosines(
            phi_swiss, phi_nmp, n_s, n_n, args.n_trials, rng,
            max_per_side=args.max_sample_per_side,
        )
        nm = float(np.median(nulls))
        per_feature_null[fid] = {
            "observed_cos":     float(observed),
            "n_swiss_residues": n_s,
            "n_nmpfam_residues": n_n,
            "null_median":      nm,
            "null_p5":          float(np.percentile(nulls, 5)),
            "null_p95":         float(np.percentile(nulls, 95)),
            "null_p99":         float(np.percentile(nulls, 99)),
            "delta_cos":        float(observed - nm),
            "p_value":          float((nulls >= observed).mean()),
            "n_trials":         int(args.n_trials),
        }

    # 4. Build payload and write.
    summary = _summarise(per_feature_null)
    payload = {
        "metric": "C_null",
        "description": (
            "Random-subset null for Metric C. For each feature, samples "
            "N_swiss and N_nmpfam random valid residues uniformly across "
            "the residue_phi cache, computes the cosine between their "
            "means, repeats n_trials times. Reports per-feature "
            "delta_cos = observed - null_median and an empirical p-value."
        ),
        "config": {
            "n_trials":             args.n_trials,
            "seed":                 args.seed,
            "max_sample_per_side":  args.max_sample_per_side,
            "phi_dim":              PHI_DIM,
        },
        "summary": summary,
        "per_feature": per_feature_null,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    s = summary
    log(f"Wrote {out_path}.")
    log(f"  median observed cos = {s['median_observed_cos']:.4f}" if s["median_observed_cos"] is not None else "  median observed cos = n/a")
    log(f"  median null     cos = {s['median_null_cos']:.4f}"     if s["median_null_cos"]     is not None else "  median null cos = n/a")
    log(f"  median delta    cos = {s['median_delta_cos']:.4f}"    if s["median_delta_cos"]    is not None else "  median delta cos = n/a")
    log(f"  frac observed > null p95 = {s['frac_observed_above_null_p95']:.3f}" if s["frac_observed_above_null_p95"] is not None else "")
    log(f"  median empirical p-value = {s['median_p_value']:.4f}" if s["median_p_value"] is not None else "")


if __name__ == "__main__":
    main()
