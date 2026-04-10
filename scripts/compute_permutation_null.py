#!/usr/bin/env python3
"""Compute permutation-based null distributions for geometry-primary classification.

For each SAE feature, tests the null hypothesis that there is no association
between the per-residue activation pattern and each annotation structure
(sequence motifs, positional predicates, InterPro/CATH domain boundaries,
and local 3D geometry).

**Shuffle mechanism:**
Within-protein permutation of activation values.  For each protein
independently, randomly permute the per-residue activation values for the
feature under test.  This preserves (a) the marginal activation distribution
within each protein, (b) protein-level activation magnitude, (c) the
annotation structure (k-mer positions, domain boundaries, etc.), and
(d) protein boundaries.  It breaks only the residue-level association
between activation and annotation.

**P-value computation (one-sided, Phipson & Smyth 2010):**
    p = (1 + #{perm_score >= observed_score}) / (1 + K)

**Outputs:**
Per-feature checkpoint JSONs in ``{data_dir}/permutation_null/{fid:04d}.json``
containing observed scores, full null distributions (K values per metric),
raw p-values, and null summary statistics.  These are consumed by
``compute_geometry_primary.py`` which applies Benjamini-Hochberg FDR
correction across features.

Usage::

    python scripts/compute_permutation_null.py --data-dir /data/feature_data
    python scripts/compute_permutation_null.py --data-dir feature_data_cluster --n-permutations 10  # quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proteinlens.analysis.feature_pipeline.motif_enrichment import (
    _compute_best_motif_f1,
    _extract_kmers_with_activations,
)
from proteinlens.analysis.feature_pipeline.position_enrichment import (
    POSITION_PREDICATES,
    _build_predicate_indices,
)

logger = logging.getLogger(__name__)

# Standard amino acids for k-mer extraction
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ── Data loading helpers ──────────────────────────────────────────────


def _pool_proteins(feature_data: dict) -> list[dict]:
    """Pool proteins from feature JSON, deduplicating by accession.

    Returns list of dicts with keys: accession, sequence, activations (1D array).
    Tracks protein boundaries for within-protein shuffling.
    """
    seen = set()
    proteins = []
    for source in [feature_data.get("top_sequences", []),
                   *[v for k, v in sorted(feature_data.get("activation_bins", {}).items())
                     if isinstance(v, list)]]:
        for p in source:
            acc = p.get("accession", "")
            if acc in seen or not acc:
                continue
            seen.add(acc)
            seq = p.get("sequence", "")
            acts = p.get("per_residue_activations")
            if seq and acts:
                proteins.append({
                    "accession": acc,
                    "sequence": seq,
                    "activations": np.array(acts, dtype=np.float64),
                })
    return proteins


def _load_interpro_labels(
    proteins: list[dict], interpro_cache_dir: Path
) -> np.ndarray | None:
    """Build pooled residue-level InterPro domain labels (1=inside, 0=outside).

    Returns 1D bool array of same length as pooled activations, or None if
    no InterPro data is available.
    """
    if not interpro_cache_dir.is_dir():
        return None

    labels = []
    any_domains = False
    for p in proteins:
        n = len(p["activations"])
        res_labels = np.zeros(n, dtype=bool)
        cache_path = interpro_cache_dir / f"{p['accession']}.json"
        if cache_path.exists():
            try:
                domains = json.loads(cache_path.read_text())
                for d in domains:
                    start = max(0, d.get("start", 1) - 1)  # 1-based to 0-based
                    end = min(n, d.get("end", 0))  # 1-based inclusive
                    if start < end:
                        res_labels[start:end] = True
                        any_domains = True
            except (json.JSONDecodeError, OSError):
                pass
        labels.append(res_labels)

    if not any_domains:
        return None
    return np.concatenate(labels)


def _load_cath_labels(
    proteins: list[dict], cath_cache_dir: Path
) -> np.ndarray | None:
    """Build pooled residue-level CATH domain labels (1=inside, 0=outside).

    Takes max across all CATH hierarchy levels (any domain hit counts).
    """
    if not cath_cache_dir.is_dir():
        return None

    labels = []
    any_domains = False
    for p in proteins:
        n = len(p["activations"])
        res_labels = np.zeros(n, dtype=bool)
        cache_path = cath_cache_dir / f"{p['accession']}.json"
        if cache_path.exists():
            try:
                hits = json.loads(cache_path.read_text())
                for h in hits:
                    start = max(0, h.get("query_start", 1) - 1)
                    end = min(n, h.get("query_end", 0))
                    if start < end:
                        res_labels[start:end] = True
                        any_domains = True
            except (json.JSONDecodeError, OSError):
                pass
        labels.append(res_labels)

    if not any_domains:
        return None
    return np.concatenate(labels)


def _compute_domain_f1(
    all_activations: np.ndarray,
    domain_labels: np.ndarray,
    feat_max: float,
    n_steps: int = 50,
) -> float:
    """Compute best F1 across threshold sweep for domain boundary labels.

    Same logic as the motif/position F1 but with a single "annotation"
    (inside-domain vs outside-domain).
    """
    N = len(all_activations)
    if N == 0 or feat_max <= 0 or domain_labels.sum() == 0:
        return 0.0

    thresholds = np.linspace(0, feat_max, n_steps + 1)[1:]
    activated_matrix = all_activations[None, :] > thresholds[:, None]  # (T, N)
    n_activated = activated_matrix.sum(axis=1).astype(float)  # (T,)

    idx = np.where(domain_labels)[0]
    tp = activated_matrix[:, idx].sum(axis=1).astype(float)
    fp = float(len(idx)) - tp
    fn = n_activated - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    return float(f1.max()) if len(f1) > 0 else 0.0


# ── Geometry PR-AUC helpers ───────────────────────────────────────────


def _load_gbm_and_predict(
    fid: int,
    proteins: list[dict],
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Load saved GBM, compute geometry predictions for all interior residues.

    Returns (sae_activations, geom_predictions, threshold) for all interior
    residues, or None if GBM not available.
    """
    gbm_dir = data_dir / "geometry_classifiers"
    gbm_path = gbm_dir / f"{fid:04d}_gbm.pkl"
    meta_path = gbm_dir / f"{fid:04d}_meta.json"

    if not gbm_path.exists() or not meta_path.exists():
        return None

    try:
        gbm = joblib.load(gbm_path)
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None

    threshold = meta["threshold_sae"]
    half_w = meta["half_w"]

    # Load geometry profiles
    geom_path = data_dir / "geometry_protein_features.npz"
    if not geom_path.exists():
        return None

    # Import feature extraction
    from proteinlens.analysis.geometry.residue_features import (
        extract_local_feature_vector,
        select_features,
    )

    geom_data = np.load(geom_path, allow_pickle=True)
    geom_accessions = set(str(a) for a in geom_data.get("accessions", []))

    # Also try loading from per-protein npz files
    geom_profile_dir = data_dir / "geometry_residue_profiles"
    if not geom_accessions and geom_profile_dir.is_dir():
        geom_accessions = {p.stem for p in geom_profile_dir.glob("*.npz")}

    # Load residue activations for geometry computation
    act_dir = data_dir / "residue_activations"
    ipro_act_dir = data_dir / "interpro_residue_activations"

    all_sae = []
    all_geom = []

    for p in proteins:
        acc = p["accession"]

        # Load per-residue activations from .npz
        act_path = None
        for d in [act_dir, ipro_act_dir]:
            candidate = d / f"{acc}.npz"
            if candidate.exists():
                act_path = candidate
                break
        if act_path is None:
            continue

        try:
            act_data = np.load(act_path)["activations"]
        except Exception:
            continue

        n_residues = act_data.shape[0]
        if n_residues < 2 * half_w + 1:
            continue

        # Load geometry profiles for this protein
        profiles = None
        ca = None

        # Try per-protein npz
        if geom_profile_dir.is_dir():
            gp_path = geom_profile_dir / f"{acc}.npz"
            if gp_path.exists():
                try:
                    gp = np.load(gp_path, allow_pickle=True)
                    ca = np.array(gp["ca"])
                    profiles = {
                        "curvature": np.array(gp["curvature"]),
                        "torsion": np.array(gp["torsion"]),
                        "planarity": np.array(gp["planarity"]),
                        "tangents": np.array(gp["tangents"]) if "tangents" in gp else None,
                        "helix_mask": np.array(gp["helix_mask"]) if "helix_mask" in gp else None,
                        "categories": np.array(gp["categories"]) if "categories" in gp else None,
                    }
                except Exception:
                    pass

        if ca is None or profiles is None:
            continue

        n = min(len(ca), n_residues)
        sae_col = act_data[:n, fid]
        seq = p.get("sequence", "")

        # Extract features and predict for interior residues
        for pos in range(half_w, n - half_w):
            fv = extract_local_feature_vector(profiles, ca, pos, half_w, seq)
            if fv is None:
                continue
            fv_sel = select_features(fv).reshape(1, -1)
            try:
                prob = gbm.predict_proba(fv_sel)
                geom_prob = float(prob[0, 1]) if prob.shape[1] > 1 else float(prob[0, 0])
            except Exception:
                continue

            all_sae.append(float(sae_col[pos]))
            all_geom.append(geom_prob)

    if len(all_sae) < 20:
        return None

    return np.array(all_sae), np.array(all_geom), threshold


# ── Within-protein shuffle ────────────────────────────────────────────


def _shuffle_within_proteins(
    all_activations: np.ndarray,
    protein_boundaries: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle activation values independently within each protein.

    Args:
        all_activations: Pooled 1D activation array.
        protein_boundaries: List of (start, end) index pairs into the pooled array.
        rng: NumPy random generator.

    Returns:
        Copy of all_activations with values shuffled within each protein segment.
    """
    shuffled = all_activations.copy()
    for start, end in protein_boundaries:
        rng.shuffle(shuffled[start:end])
    return shuffled


# ── Per-feature permutation worker ────────────────────────────────────


def process_feature(
    fid: int,
    data_dir: Path,
    n_permutations: int,
    seed: int,
) -> dict[str, Any] | None:
    """Run permutation testing for a single feature across all 5 metrics.

    Returns the full result dict, or None if the feature cannot be processed.
    """
    feat_path = data_dir / "features" / f"{fid:04d}.json"
    if not feat_path.exists():
        return None

    feature_data = json.loads(feat_path.read_text())
    feat_max_arr = np.load(data_dir / "feature_max_activations.npy")
    feat_max = float(feat_max_arr[fid])
    if feat_max <= 0:
        return None

    # Pool proteins
    proteins = _pool_proteins(feature_data)
    if len(proteins) < 2:
        return None

    # Build pooled activation array and protein boundaries
    all_activations_list = []
    protein_boundaries = []
    seq_lengths = []
    offset = 0
    for p in proteins:
        n = len(p["activations"])
        all_activations_list.append(p["activations"])
        protein_boundaries.append((offset, offset + n))
        seq_lengths.append(n)
        offset += n

    all_activations = np.concatenate(all_activations_list)
    total_residues = len(all_activations)

    if total_residues < 10:
        return None

    # ── Build annotation structures (fixed, not shuffled) ──

    # 1. K-mer indices
    k = 3
    all_kmers = []
    for p in proteins:
        pairs = _extract_kmers_with_activations(p["sequence"], p["activations"].tolist(), k)
        for kmer, _ in pairs:
            all_kmers.append(kmer)

    kmer_indices = {}
    for i, kmer in enumerate(all_kmers):
        kmer_indices.setdefault(kmer, []).append(i)
    kmer_idx_arrays = {km: np.array(idxs) for km, idxs in kmer_indices.items()}

    # 2. Position predicate indices
    predicate_indices = _build_predicate_indices(seq_lengths, total_residues)

    # 3. InterPro domain labels
    interpro_cache_dir = data_dir / "interpro_cache"
    interpro_labels = _load_interpro_labels(proteins, interpro_cache_dir)

    # 4. CATH domain labels
    cath_cache_dir = data_dir / "cath_enrichment" / "cache"
    cath_labels = _load_cath_labels(proteins, cath_cache_dir)

    # 5. Geometry PR-AUC (load GBM, get predictions)
    geom_result = _load_gbm_and_predict(fid, proteins, data_dir)

    # ── Compute observed scores ──

    n_steps = 50
    min_count = 5

    # Motif F1
    motif_results = _compute_best_motif_f1(
        kmer_idx_arrays, all_activations, feat_max,
        n_steps=n_steps, min_count=min_count, top_n=1,
    )
    motif_f1_obs = motif_results[0]["best_f1"] if motif_results else 0.0

    # Position F1
    position_results = _compute_best_motif_f1(
        predicate_indices, all_activations, feat_max,
        n_steps=n_steps, min_count=1, top_n=1,
    )
    position_f1_obs = position_results[0]["best_f1"] if position_results else 0.0

    # InterPro residue F1
    interpro_f1_obs = 0.0
    if interpro_labels is not None:
        interpro_f1_obs = _compute_domain_f1(all_activations, interpro_labels, feat_max, n_steps)

    # CATH residue F1
    cath_f1_obs = 0.0
    if cath_labels is not None:
        cath_f1_obs = _compute_domain_f1(all_activations, cath_labels, feat_max, n_steps)

    # Geometry PR-AUC
    geom_prauc_obs = 0.0
    if geom_result is not None:
        from sklearn.metrics import average_precision_score
        sae_arr, geom_preds, geom_threshold = geom_result
        sae_binary = (sae_arr >= geom_threshold).astype(int)
        if sae_binary.sum() > 0 and sae_binary.sum() < len(sae_binary):
            geom_prauc_obs = float(average_precision_score(sae_binary, geom_preds))

    # ── Permutation loop ──

    rng = np.random.default_rng(seed + fid)

    null_motif = np.zeros(n_permutations)
    null_position = np.zeros(n_permutations)
    null_interpro = np.zeros(n_permutations)
    null_cath = np.zeros(n_permutations)
    null_geom = np.zeros(n_permutations)

    for k_perm in range(n_permutations):
        # Shuffle activations within each protein
        shuffled = _shuffle_within_proteins(all_activations, protein_boundaries, rng)

        # Motif F1 with shuffled activations
        perm_motif = _compute_best_motif_f1(
            kmer_idx_arrays, shuffled, feat_max,
            n_steps=n_steps, min_count=min_count, top_n=1,
        )
        null_motif[k_perm] = perm_motif[0]["best_f1"] if perm_motif else 0.0

        # Position F1 with shuffled activations
        perm_pos = _compute_best_motif_f1(
            predicate_indices, shuffled, feat_max,
            n_steps=n_steps, min_count=1, top_n=1,
        )
        null_position[k_perm] = perm_pos[0]["best_f1"] if perm_pos else 0.0

        # InterPro residue F1 with shuffled activations
        if interpro_labels is not None:
            null_interpro[k_perm] = _compute_domain_f1(shuffled, interpro_labels, feat_max, n_steps)

        # CATH residue F1 with shuffled activations
        if cath_labels is not None:
            null_cath[k_perm] = _compute_domain_f1(shuffled, cath_labels, feat_max, n_steps)

        # Geometry PR-AUC with shuffled labels
        if geom_result is not None:
            from sklearn.metrics import average_precision_score
            # Shuffle the SAE activations within each protein segment
            # For geometry, we need to shuffle the binary labels
            # The geometry predictions are from the fixed GBM
            shuffled_sae = _shuffle_within_proteins(sae_arr, protein_boundaries, rng)
            shuffled_binary = (shuffled_sae >= geom_threshold).astype(int)
            if shuffled_binary.sum() > 0 and shuffled_binary.sum() < len(shuffled_binary):
                null_geom[k_perm] = float(average_precision_score(shuffled_binary, geom_preds))

    # ── Compute p-values (Phipson & Smyth 2010) ──

    def _pvalue(observed: float, null_dist: np.ndarray) -> float:
        return float((1 + np.sum(null_dist >= observed)) / (1 + len(null_dist)))

    def _null_summary(null_dist: np.ndarray) -> dict:
        return {
            "mean": round(float(null_dist.mean()), 6),
            "std": round(float(null_dist.std()), 6),
            "p95": round(float(np.percentile(null_dist, 95)), 6),
            "p99": round(float(np.percentile(null_dist, 99)), 6),
        }

    result = {
        "feature_id": fid,
        "n_permutations": n_permutations,
        "seed": seed,
        "n_proteins": len(proteins),
        "n_residues": total_residues,
        "observed": {
            "motif_f1": round(motif_f1_obs, 6),
            "position_f1": round(position_f1_obs, 6),
            "interpro_res_f1": round(interpro_f1_obs, 6),
            "cath_res_f1": round(cath_f1_obs, 6),
            "geometry_prauc": round(geom_prauc_obs, 6),
        },
        "null_distributions": {
            "motif_f1": [round(float(v), 6) for v in null_motif],
            "position_f1": [round(float(v), 6) for v in null_position],
            "interpro_res_f1": [round(float(v), 6) for v in null_interpro],
            "cath_res_f1": [round(float(v), 6) for v in null_cath],
            "geometry_prauc": [round(float(v), 6) for v in null_geom],
        },
        "p_values": {
            "motif_f1": round(_pvalue(motif_f1_obs, null_motif), 6),
            "position_f1": round(_pvalue(position_f1_obs, null_position), 6),
            "interpro_res_f1": round(_pvalue(interpro_f1_obs, null_interpro), 6),
            "cath_res_f1": round(_pvalue(cath_f1_obs, null_cath), 6),
            "geometry_prauc": round(_pvalue(geom_prauc_obs, null_geom), 6),
        },
        "null_summary": {
            "motif_f1": _null_summary(null_motif),
            "position_f1": _null_summary(null_position),
            "interpro_res_f1": _null_summary(null_interpro),
            "cath_res_f1": _null_summary(null_cath),
            "geometry_prauc": _null_summary(null_geom),
        },
    }
    return result


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("feature_data_cluster"),
        help="Pipeline output directory",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=100,
        help="Number of permutations per feature (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (per-feature seed = base + feature_id)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    perm_dir = data_dir / "permutation_null"
    perm_dir.mkdir(parents=True, exist_ok=True)

    # Discover features
    feat_max = np.load(data_dir / "feature_max_activations.npy")
    n_features = len(feat_max)
    all_fids = [i for i in range(n_features) if feat_max[i] > 0]

    # Check for completed features (resume)
    done_fids = set()
    for fpath in perm_dir.glob("*.json"):
        try:
            fid = int(fpath.stem)
            done_fids.add(fid)
        except ValueError:
            pass

    todo = [fid for fid in all_fids if fid not in done_fids]

    print("=" * 60)
    print("Permutation Null Distribution Computation")
    print("=" * 60)
    print(f"  Data dir:        {data_dir}")
    print(f"  N permutations:  {args.n_permutations}")
    print(f"  Seed:            {args.seed}")
    print(f"  Workers:         {args.workers}")
    print(f"  Total features:  {len(all_fids)}")
    print(f"  Already done:    {len(done_fids)}")
    print(f"  To process:      {len(todo)}")
    print("=" * 60)

    if not todo:
        print("All features already processed.")
        return

    t0 = time.time()

    def _worker(fid: int) -> tuple[int, str]:
        try:
            result = process_feature(fid, data_dir, args.n_permutations, args.seed)
            if result is None:
                return fid, "skipped"
            out_path = perm_dir / f"{fid:04d}.json"
            out_path.write_text(json.dumps(result, indent=2))
            return fid, "done"
        except Exception as e:
            logger.error("Feature %d failed: %s", fid, e)
            return fid, f"error: {e}"

    n_done = 0
    n_skipped = 0
    n_error = 0

    if args.workers <= 1:
        for fid in tqdm(todo, desc="Permutation testing"):
            fid_result, status = _worker(fid)
            if status == "done":
                n_done += 1
            elif status == "skipped":
                n_skipped += 1
            else:
                n_error += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, fid): fid for fid in todo}
            pbar = tqdm(total=len(todo), desc="Permutation testing")
            for future in as_completed(futures):
                fid_result, status = future.result()
                if status == "done":
                    n_done += 1
                elif status == "skipped":
                    n_skipped += 1
                else:
                    n_error += 1
                pbar.update(1)
            pbar.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s: {n_done} completed, {n_skipped} skipped, {n_error} errors")


if __name__ == "__main__":
    main()
