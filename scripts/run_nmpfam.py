#!/usr/bin/env python3
"""NMPFam inference: activate SAE nodes on metagenomic proteins, score with presaved GBMs.

Pipeline (four phases):

  Phase 1 — Fetch NMPFams families and consensus sequences.
  Phase 2 — Run ESM → SAE inference on consensus sequences.
  Phase 3 — Stream PDB structures and compute residue geometry profiles.
  Phase 4 — For each SAE node with a saved GBM at
            {data_dir}/geometry_classifiers/{id:04d}_gbm.pkl, find NMPFam hits
            (feature_maxes > activation_fraction × global_max), run the
            presaved GBM on each hit's residue windows, and save per-feature
            JSON for the frontend.

Strictly read-only w.r.t. existing pipeline outputs — no retraining,
no overwriting of geometry_classifiers/ or geometry_enrichment/ data.

Outputs land under ``{data_dir}/nmpfam/``:
  families.json, family_index.json, feature_maxes.npy
  fasta/{fid}.fasta
  residue_activations/{fid}.npz
  geometry_residue_profiles/{fid}.npz
  nmpfam_enrichment/{feat_id:04d}.json
  nmpfam_enrichment/summary.json

Usage:
    python scripts/run_nmpfam.py --data-dir feature_data_cluster
    python scripts/run_nmpfam.py --data-dir feature_data_cluster --n-families 5
    python scripts/run_nmpfam.py --data-dir feature_data_cluster --phase 4
    python scripts/run_nmpfam.py --data-dir feature_data_cluster --start-phase 2 --end-phase 3
    PIPELINE_WORKERS=8 python scripts/run_nmpfam.py --data-dir feature_data_cluster

Cluster split (phases 1/3/4 on CPU pods, phase 2 on a GPU pod; phases 2 and 3
can run in parallel after phase 1 populates families.json):

    # CPU:  FASTA gather
    python scripts/run_nmpfam.py --data-dir feature_data_cluster --phase 1
    # GPU:  ESM + SAE inference
    python scripts/run_nmpfam.py --data-dir feature_data_cluster --phase 2
    # CPU:  PDB fetch + geometry (can run concurrently with phase 2)
    python scripts/run_nmpfam.py --data-dir feature_data_cluster --phase 3
    # CPU:  presaved-GBM scoring
    PIPELINE_WORKERS=8 python scripts/run_nmpfam.py --data-dir feature_data_cluster --phase 4
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import requests
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog
from proteinlens.analysis.geometry.residue_features import (
    ACTIVE_GEOM_NAMES,
    ca_backbone,
    compute_residue_profiles,
    detect_alpha_helices_from_ca,
    extract_local_feature_vector,
    select_features,
)
from proteinlens.embedders.esm import ESM
from proteinlens.sae.inference import load_sae
from proteinlens.utils import get_device

logger = logging.getLogger(__name__)

NMPFAM_API_BASE = "https://bib.fleming.gr/NMPFamsDB/api"
NMPFAM_DATA_BASE = "https://bib.fleming.gr/NMPFamsDB/data"


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Fetch and sample NMPFams families
# ═══════════════════════════════════════════════════════════════════════


def fetch_all_families(session: requests.Session) -> list[dict]:
    url = f"{NMPFAM_API_BASE}/families"
    logger.info("Fetching NMPFams family list from %s", url)
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    families = resp.json()
    logger.info("Got %d families from NMPFams API", len(families))
    return families


def sample_families_with_structures(
    families: list[dict], n_target: int, seed: int = 42
) -> list[dict]:
    with_pdb = [f for f in families if f.get("PDB") == "Y"]
    logger.info("%d / %d families have PDB structures", len(with_pdb), len(families))

    if len(with_pdb) <= n_target:
        logger.info("Fewer families with PDB than target — using all %d", len(with_pdb))
        return with_pdb

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for f in with_pdb:
        by_cat[f.get("Category", "Unknown")].append(f)

    rng = random.Random(seed)
    sampled: list[dict] = []
    remaining = n_target
    cat_items = sorted(by_cat.items(), key=lambda x: len(x[1]))

    for i, (_cat, members) in enumerate(cat_items):
        cats_left = len(cat_items) - i
        n_alloc = min(len(members), max(1, remaining // cats_left))
        sampled.extend(rng.sample(members, n_alloc))
        remaining -= n_alloc

    if remaining > 0:
        sampled_ids = {id(f) for f in sampled}
        all_unsampled = [f for f in with_pdb if id(f) not in sampled_ids]
        if all_unsampled:
            sampled.extend(rng.sample(all_unsampled, min(remaining, len(all_unsampled))))

    logger.info("Sampled %d families across %d categories", len(sampled), len(by_cat))
    return sampled


def _parse_first_sequence(fasta_text: str) -> str | None:
    lines = fasta_text.strip().split("\n")
    seq_lines: list[str] = []
    found_header = False
    for line in lines:
        if line.startswith(">"):
            if found_header:
                break
            found_header = True
            continue
        if found_header:
            seq_lines.append(line.strip())
    seq = "".join(seq_lines)
    return seq if seq else None


def download_consensus_sequence(
    family_id: str, fasta_dir: Path, session: requests.Session,
    fasta_cached: set[str],
) -> str | None:
    """Download FASTA for a family; return the first (consensus) sequence.

    Uses the prebuilt ``fasta_cached`` set (populated from one glob at startup)
    instead of ``Path.exists()`` — cephfs-friendly.
    """
    fasta_path = fasta_dir / f"{family_id}.fasta"
    if family_id in fasta_cached:
        try:
            return _parse_first_sequence(fasta_path.read_text())
        except OSError:
            pass
    url = f"{NMPFAM_DATA_BASE}/fasta/{family_id}.fasta"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            logger.warning("FASTA download failed for %s: HTTP %d", family_id, resp.status_code)
            return None
        fasta_path.write_text(resp.text)
        fasta_cached.add(family_id)
        return _parse_first_sequence(resp.text)
    except requests.RequestException as e:
        logger.warning("FASTA download error for %s: %s", family_id, e)
        return None


def fetch_pdb_text(family_id: str, session: requests.Session) -> str | None:
    url = f"{NMPFAM_DATA_BASE}/pdb/{family_id}.pdb"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.text
        logger.warning("PDB fetch failed for %s: HTTP %d", family_id, resp.status_code)
    except requests.RequestException as e:
        logger.warning("PDB fetch error for %s: %s", family_id, e)
    return None


def run_phase1_fetch(n_families: int, data_dir: Path, seed: int = 42) -> list[dict]:
    nmpfam_dir = data_dir / "nmpfam"
    nmpfam_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir = nmpfam_dir / "fasta"
    fasta_dir.mkdir(exist_ok=True)
    families_path = nmpfam_dir / "families.json"

    session = requests.Session()
    session.headers["User-Agent"] = "ProteinLens/1.0 (research)"

    # Single glob up front — no per-file exists() later.
    fasta_cached: set[str] = {p.stem for p in fasta_dir.glob("*.fasta")}
    print(f"[phase1] {len(fasta_cached)} FASTA files already cached")

    families_cached = families_path.is_file()
    if families_cached:
        logger.info("Loading cached families from %s", families_path)
        with open(families_path) as f:
            sampled = json.load(f)
        for fam in tqdm(sampled, desc="Filling FASTA cache"):
            fid = fam["ID"]
            if fam.get("consensus_sequence"):
                continue
            seq = download_consensus_sequence(fid, fasta_dir, session, fasta_cached)
            if seq:
                fam["consensus_sequence"] = seq
                fam["consensus_length"] = len(seq)
                time.sleep(0.1)
        sampled_with_seq = [f for f in sampled if f.get("consensus_sequence")]
        with open(families_path, "w") as f:
            json.dump(sampled, f, indent=2)
        return sampled_with_seq

    all_families = fetch_all_families(session)
    sampled = sample_families_with_structures(all_families, n_families, seed)

    with open(families_path, "w") as f:
        json.dump(sampled, f, indent=2)

    print(f"[phase1] Downloading consensus sequences for {len(sampled)} families...")
    for fam in tqdm(sampled, desc="Downloading FASTA"):
        fid = fam["ID"]
        seq = download_consensus_sequence(fid, fasta_dir, session, fasta_cached)
        if seq:
            fam["consensus_sequence"] = seq
            fam["consensus_length"] = len(seq)
        time.sleep(0.1)

    sampled = [f for f in sampled if f.get("consensus_sequence")]
    logger.info("Got sequences for %d families", len(sampled))

    cat_counts: dict[str, int] = defaultdict(int)
    seq_lengths = [f.get("consensus_length", 0) for f in sampled]
    for f in sampled:
        cat_counts[f.get("Category", "Unknown")] += 1

    wlog({
        "phase1/n_families_sampled": len(sampled),
        "phase1/n_categories": len(cat_counts),
        "phase1/mean_seq_length": float(np.mean(seq_lengths)) if seq_lengths else 0,
        "phase1/median_seq_length": float(np.median(seq_lengths)) if seq_lengths else 0,
    })

    with open(families_path, "w") as f:
        json.dump(sampled, f, indent=2)
    return sampled


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: SAE inference on NMPFam consensus sequences
# ═══════════════════════════════════════════════════════════════════════


def run_phase2_inference(
    families: list[dict],
    sae_dir: Path,
    data_dir: Path,
    esm_model: str,
    esm_layer: int,
    max_seq_len: int,
    device: str | None,
) -> np.ndarray:
    nmpfam_dir = data_dir / "nmpfam"
    act_dir = nmpfam_dir / "residue_activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    device = device or get_device()
    print(f"[phase2] Loading ESM ({esm_model}) and SAE ({sae_dir}) on {device}...")

    esm = ESM(esm_model, device=device)
    sae = load_sae(str(sae_dir), device=device)
    n_features = sae.dict_size

    # Single glob — one cephfs listing instead of 50k exists() calls.
    act_cached: set[str] = {p.stem for p in act_dir.glob("*.npz")}
    print(f"[phase2] {len(act_cached)} activation files already cached")

    family_index: dict[str, int] = {}
    feature_maxes = np.zeros((len(families), n_features), dtype=np.float32)

    n_computed = 0
    n_cached = 0
    n_skipped = 0

    print(f"[phase2] Running inference on {len(families)} families...")
    for i, fam in enumerate(tqdm(families, desc="SAE inference")):
        fid = fam["ID"]
        family_index[fid] = i
        seq = fam.get("consensus_sequence", "")

        if not seq or len(seq) > max_seq_len:
            if seq and len(seq) > max_seq_len:
                logger.info("Skipping %s: seq too long (%d > %d)", fid, len(seq), max_seq_len)
            n_skipped += 1
            continue

        npz_path = act_dir / f"{fid}.npz"
        if fid in act_cached:
            try:
                acts = np.load(npz_path)["activations"]
                feature_maxes[i] = acts.max(axis=0)
                n_cached += 1
                continue
            except Exception:
                pass

        embeddings = esm.embed_single_sequence(seq, esm_layer)
        embeddings_tensor = torch.tensor(embeddings, device=device)
        with torch.no_grad():
            normed_input, _ = sae._normalize_input_and_get_norms(embeddings_tensor)
            activations = sae.encode(normed_input)
        acts = activations.cpu().numpy()

        np.savez_compressed(npz_path, activations=acts)
        act_cached.add(fid)
        feature_maxes[i] = acts.max(axis=0)
        n_computed += 1

        if (n_computed + n_cached) % 500 == 0:
            wlog({
                "phase2/families_processed": n_computed + n_cached,
                "phase2/families_computed": n_computed,
                "phase2/families_cached": n_cached,
            })

    np.save(nmpfam_dir / "feature_maxes.npy", feature_maxes)
    with open(nmpfam_dir / "family_index.json", "w") as f:
        json.dump(family_index, f)

    nonzero_per_family = (feature_maxes > 0).sum(axis=1)
    nonzero_per_feature = (feature_maxes > 0).sum(axis=0)
    wlog({
        "phase2/n_families_total": len(families),
        "phase2/n_computed": n_computed,
        "phase2/n_cached": n_cached,
        "phase2/n_skipped_too_long": n_skipped,
        "phase2/mean_active_features_per_family": float(np.mean(nonzero_per_family)),
        "phase2/mean_active_families_per_feature": float(np.mean(nonzero_per_feature)),
        "phase2/n_features_with_any_hit": int((nonzero_per_feature > 0).sum()),
    })

    print(f"[phase2] Saved feature_maxes ({feature_maxes.shape}) and {len(family_index)} family activations")
    return feature_maxes


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Geometry profiles for NMPFam proteins
# ═══════════════════════════════════════════════════════════════════════


def run_phase3_geometry(families: list[dict], data_dir: Path) -> None:
    nmpfam_dir = data_dir / "nmpfam"
    geom_dir = nmpfam_dir / "geometry_residue_profiles"
    geom_dir.mkdir(parents=True, exist_ok=True)

    # Single glob up front.
    geom_cached: set[str] = {p.stem for p in geom_dir.glob("*.npz")}
    print(f"[phase3] {len(geom_cached)} geometry files already cached")

    session = requests.Session()
    session.headers["User-Agent"] = "ProteinLens/1.0 (research)"

    n_computed = 0
    n_cached_hit = 0
    n_failed = 0

    print(f"[phase3] Computing geometry profiles for {len(families)} families...")
    for fam in tqdm(families, desc="Geometry profiles"):
        fid = fam["ID"]
        if fid in geom_cached:
            n_cached_hit += 1
            continue

        pdb_text = fetch_pdb_text(fid, session)
        if pdb_text is None:
            n_failed += 1
            continue

        try:
            ca = ca_backbone(pdb_text, chain_id=None)
            if len(ca) < 10:
                n_failed += 1
                continue
            helices = detect_alpha_helices_from_ca(ca)
            profiles = compute_residue_profiles(ca, helices)

            seq = fam.get("consensus_sequence", "")
            geom_path = geom_dir / f"{fid}.npz"
            np.savez_compressed(
                geom_path,
                ca=ca,
                curvature=profiles["curvature"],
                torsion=profiles["torsion"],
                planarity=profiles["planarity"],
                tangents=profiles["tangents"],
                helix_mask=profiles["helix_mask"],
                categories=profiles["categories"],
                sequence=np.array([seq[:len(ca)]]),
            )
            geom_cached.add(fid)
            n_computed += 1
        except Exception as e:
            logger.warning("Geometry failed for %s: %s", fid, e)
            n_failed += 1

        time.sleep(0.1)

    wlog({
        "phase3/n_computed": n_computed,
        "phase3/n_cached": n_cached_hit,
        "phase3/n_failed": n_failed,
        "phase3/success_rate": (n_computed + n_cached_hit) / max(len(families), 1),
    })
    print(f"[phase3] Geometry: {n_computed} new, {n_cached_hit} cached, {n_failed} failed")


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Score NMPFam proteins with presaved GBMs
# ═══════════════════════════════════════════════════════════════════════
#
# For each SAE node that has a presaved GBM (from the pipeline's
# geometry_residue_enrichment stage), load the GBM + metadata and run
# predict_proba on NMPFam proteins that activate that node.  No retraining.


_shared: dict = {}


def _score_nmpfam_protein(
    fid: str,
    node_idx: int,
    nmpfam_act: np.ndarray,
    nmpfam_geom: dict,
    gbm,
    activation_threshold: float,
    geom_threshold: float,
    half_w: int,
) -> dict | None:
    """Run a presaved GBM on one NMPFam protein for one SAE node."""
    node_acts = nmpfam_act[:, node_idx]
    ca = nmpfam_geom["ca"]
    n = min(len(ca), len(node_acts))

    if n < 2 * half_w + 1:
        return None

    profiles = {
        k: nmpfam_geom[k][:n]
        for k in ("curvature", "torsion", "planarity", "tangents", "helix_mask", "categories")
    }
    seq_arr = nmpfam_geom.get("sequence", np.array([""]))
    seq = str(seq_arr[0]) if len(seq_arr) > 0 else ""

    batch_fvs: list[np.ndarray] = []
    batch_positions: list[int] = []
    for pos in range(half_w, n - half_w):
        fv = extract_local_feature_vector(profiles, ca[:n], pos, half_w, sequence=seq)
        if fv is not None and np.all(np.isfinite(fv)):
            batch_fvs.append(select_features(fv))
            batch_positions.append(pos)

    if not batch_fvs:
        return None

    X = np.array(batch_fvs)
    probs = gbm.predict_proba(X)

    geom_prob_profile = [0.0] * n
    for j, pos in enumerate(batch_positions):
        p = probs[j]
        geom_prob_profile[pos] = float(p[1] if len(p) > 1 else p[0])

    sae_profile = [float(node_acts[i]) for i in range(n)]

    concordance_labels: list[str] = []
    n_agree = 0
    n_sae_only = 0
    n_geom_only = 0
    for pos in range(n):
        sae_active = float(node_acts[pos]) >= activation_threshold
        geom_active = geom_prob_profile[pos] >= geom_threshold
        if sae_active and geom_active:
            concordance_labels.append("agree")
            n_agree += 1
        elif sae_active and not geom_active:
            concordance_labels.append("fn")
            n_sae_only += 1
        elif not sae_active and geom_active:
            concordance_labels.append("fp")
            n_geom_only += 1
        else:
            concordance_labels.append("tn")

    active_mask = node_acts[:n] >= activation_threshold
    mean_at_active = (
        float(np.mean(np.array(geom_prob_profile)[active_mask]))
        if active_mask.any() else 0.0
    )

    return {
        "family_id": fid,
        "n_residues": n,
        "sequence": seq[:n],
        "sae_activation_profile": sae_profile,
        "geom_prob_profile": geom_prob_profile,
        "concordance_labels": concordance_labels,
        "max_sae_activation": float(node_acts[:n].max()),
        "max_geom_prob": max(geom_prob_profile) if geom_prob_profile else 0.0,
        "mean_geom_prob_at_active": mean_at_active,
        "n_agree": n_agree,
        "n_sae_only": n_sae_only,
        "n_geom_only": n_geom_only,
    }


def _process_node_phase4(ni: int) -> tuple[int, str]:
    """Score all NMPFam hits for one SAE node using its presaved GBM."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    s = _shared

    nmpfam_feature_maxes: np.ndarray = s["nmpfam_feature_maxes"]
    global_max: np.ndarray = s["global_max"]
    activation_fraction: float = s["activation_fraction"]
    nmpfam_act_files: dict[str, Path] = s["nmpfam_act_files"]
    nmpfam_geom_files: dict[str, Path] = s["nmpfam_geom_files"]
    row_to_fid: dict[int, str] = s["row_to_fid"]
    fam_meta: dict[str, dict] = s["fam_meta"]
    gbm_dir: Path = s["gbm_dir"]
    inference_dir: Path = s["inference_dir"]

    # Per-worker LRU caches (populated lazily). Cap at 500 each.
    if not hasattr(_process_node_phase4, "_act_cache"):
        _process_node_phase4._act_cache = {}
        _process_node_phase4._act_order = []
        _process_node_phase4._geom_cache = {}
        _process_node_phase4._geom_order = []
    _act_cache = _process_node_phase4._act_cache
    _act_order = _process_node_phase4._act_order
    _geom_cache = _process_node_phase4._geom_cache
    _geom_order = _process_node_phase4._geom_order
    _LRU_MAX = 500

    def _load_act(fid: str) -> np.ndarray | None:
        if fid in _act_cache:
            return _act_cache[fid]
        path = nmpfam_act_files.get(fid)
        if path is None:
            return None
        try:
            arr = np.load(path)["activations"]
        except Exception:
            return None
        _act_cache[fid] = arr
        _act_order.append(fid)
        if len(_act_order) > _LRU_MAX:
            _act_cache.pop(_act_order.pop(0), None)
        return arr

    def _load_geom(fid: str) -> dict | None:
        if fid in _geom_cache:
            return _geom_cache[fid]
        path = nmpfam_geom_files.get(fid)
        if path is None:
            return None
        try:
            g = np.load(path, allow_pickle=True)
            data = {
                "ca": np.array(g["ca"]),
                "curvature": np.array(g["curvature"]),
                "torsion": np.array(g["torsion"]),
                "planarity": np.array(g["planarity"]),
                "tangents": np.array(g["tangents"]),
                "helix_mask": np.array(g["helix_mask"]),
                "categories": np.array(g["categories"]),
                "sequence": np.array(g.get("sequence", np.array([""]))),
            }
        except Exception:
            return None
        _geom_cache[fid] = data
        _geom_order.append(fid)
        if len(_geom_order) > _LRU_MAX:
            _geom_cache.pop(_geom_order.pop(0), None)
        return data

    # ── Load presaved GBM + meta ──
    padded = f"{ni:04d}"
    try:
        gbm = joblib.load(gbm_dir / f"{padded}_gbm.pkl")
        meta = json.loads((gbm_dir / f"{padded}_meta.json").read_text())
        activation_threshold = float(meta["threshold_sae"])
        geom_threshold = float(meta["threshold_geom"])
        half_w = int(meta["half_w"])
    except Exception as e:
        logger.warning("Failed to load GBM/meta for node %d: %s", ni, e)
        return (ni, "skipped")

    # ── Identify NMPFam hits for this node ──
    gmax = float(global_max[ni])
    if gmax <= 0:
        return (ni, "skipped")

    hit_threshold = activation_fraction * gmax
    node_col = nmpfam_feature_maxes[:, ni]
    hit_rows = np.where(node_col > hit_threshold)[0]
    if len(hit_rows) == 0:
        return (ni, "no_hits")

    hit_rows = hit_rows[np.argsort(node_col[hit_rows])[::-1]]

    # ── Score each hit with the presaved GBM ──
    scored_hits: list[dict] = []
    for row_idx in hit_rows:
        fid = row_to_fid.get(int(row_idx))
        if fid is None:
            continue
        nmpfam_act = _load_act(fid)
        nmpfam_geom = _load_geom(fid)
        if nmpfam_act is None or nmpfam_geom is None:
            continue

        result = _score_nmpfam_protein(
            fid, ni, nmpfam_act, nmpfam_geom, gbm,
            activation_threshold, geom_threshold, half_w,
        )
        if result is None:
            continue

        m = fam_meta.get(fid, {})
        result["category"] = m.get("Category", "Unknown")
        result["sequence_count"] = m.get("SequenceCount", 0)
        result["nmpfams_url"] = f"https://bib.fleming.gr/NMPFamsDB/family/{fid}"
        scored_hits.append(result)

    if not scored_hits:
        return (ni, "no_hits")

    out = {
        "feature_id": ni,
        "feature_global_max": round(gmax, 4),
        "activation_threshold_sae": round(activation_threshold, 4),
        "geom_threshold": round(geom_threshold, 4),
        "half_w": half_w,
        "n_nmpfam_hits": len(scored_hits),
        "nmpfam_hits": scored_hits,
    }
    (inference_dir / f"{ni:04d}.json").write_text(json.dumps(out, indent=2))
    return (ni, "scored")


def run_phase4_gbm_inference(
    data_dir: Path, activation_fraction: float = 0.5,
) -> None:
    """Phase 4: Load presaved GBMs and score NMPFam hits per SAE node."""
    nmpfam_dir = data_dir / "nmpfam"
    inference_dir = nmpfam_dir / "nmpfam_enrichment"
    inference_dir.mkdir(parents=True, exist_ok=True)

    gbm_dir = data_dir / "geometry_classifiers"
    if not gbm_dir.is_dir():
        print(f"[phase4] ERROR: no geometry_classifiers/ at {gbm_dir}")
        return

    # ── Single glob each: cephfs-friendly ──
    gbm_ids: set[int] = set()
    for p in gbm_dir.glob("*_gbm.pkl"):
        stem = p.stem.replace("_gbm", "")
        try:
            gbm_ids.add(int(stem))
        except ValueError:
            continue
    print(f"[phase4] {len(gbm_ids)} presaved GBMs found")

    nmpfam_act_dir = nmpfam_dir / "residue_activations"
    nmpfam_geom_dir = nmpfam_dir / "geometry_residue_profiles"
    nmpfam_act_files = {p.stem: p for p in nmpfam_act_dir.glob("*.npz")}
    nmpfam_geom_files = {p.stem: p for p in nmpfam_geom_dir.glob("*.npz")}
    print(f"[phase4] {len(nmpfam_act_files)} activation files, "
          f"{len(nmpfam_geom_files)} geometry files")

    # Resume: skip nodes whose output already exists
    inference_done = {p.stem for p in inference_dir.glob("????.json")}

    # ── Load SwissProt global max and NMPFam per-family max matrix ──
    global_max_path = data_dir / "feature_max_activations.npy"
    if not global_max_path.exists():
        print(f"[phase4] ERROR: {global_max_path} not found")
        return
    global_max = np.load(global_max_path)
    n_features = len(global_max)

    feature_maxes_path = nmpfam_dir / "feature_maxes.npy"
    if not feature_maxes_path.exists():
        print(f"[phase4] ERROR: {feature_maxes_path} not found (run phase 2 first)")
        return
    nmpfam_feature_maxes = np.load(feature_maxes_path)

    family_index_path = nmpfam_dir / "family_index.json"
    if not family_index_path.exists():
        print(f"[phase4] ERROR: {family_index_path} not found (run phase 2 first)")
        return
    family_index = json.loads(family_index_path.read_text())
    row_to_fid = {int(v): k for k, v in family_index.items()}

    families_path = nmpfam_dir / "families.json"
    fam_meta: dict[str, dict] = {}
    if families_path.is_file():
        fam_meta = {f["ID"]: f for f in json.loads(families_path.read_text())}

    # ── Filter to nodes with a GBM, within feature range, not already done ──
    nodes_to_process = sorted(
        ni for ni in gbm_ids
        if ni < n_features and f"{ni:04d}" not in inference_done
    )
    print(f"[phase4] {len(nodes_to_process)} nodes to process "
          f"({len(inference_done)} already done)")

    # ── Publish shared state before fork ──
    _shared.update({
        "nmpfam_feature_maxes": nmpfam_feature_maxes,
        "global_max": global_max,
        "activation_fraction": activation_fraction,
        "nmpfam_act_files": nmpfam_act_files,
        "nmpfam_geom_files": nmpfam_geom_files,
        "row_to_fid": row_to_fid,
        "fam_meta": fam_meta,
        "gbm_dir": gbm_dir,
        "inference_dir": inference_dir,
    })

    n_workers = min(
        int(os.environ.get("PIPELINE_WORKERS", "1")),
        len(nodes_to_process) or 1,
    )
    n_scored = 0
    n_skipped = 0
    n_no_hits = 0

    pbar = tqdm(total=len(nodes_to_process), desc="GBM inference")
    n_total = len(nodes_to_process)
    log_every = max(1, n_total // 50)

    def _handle_result(ni: int, result: str) -> None:
        nonlocal n_scored, n_skipped, n_no_hits
        if result == "scored":
            n_scored += 1
        elif result == "no_hits":
            n_no_hits += 1
        else:
            n_skipped += 1
        done = n_scored + n_no_hits + n_skipped
        if done % log_every == 0 or done == n_total:
            wlog({
                "phase4/nodes_processed": done,
                "phase4/nodes_scored": n_scored,
                "phase4/nodes_no_hits": n_no_hits,
                "phase4/nodes_skipped": n_skipped,
                "phase4/progress_pct": round(100 * done / n_total, 1),
            })
        pbar.update(1)

    if n_workers > 1:
        print(f"[phase4] Processing {n_total} nodes with {n_workers} workers")
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            for ni, result in pool.imap_unordered(_process_node_phase4, nodes_to_process):
                _handle_result(ni, result)
    else:
        print(f"[phase4] Processing {n_total} nodes serially "
              "(set PIPELINE_WORKERS=N for parallel)")
        for ni in nodes_to_process:
            try:
                _, result = _process_node_phase4(ni)
                _handle_result(ni, result)
            except Exception:
                logger.exception("Error processing node %d", ni)
                _handle_result(ni, "skipped")

    pbar.close()

    wlog({
        "phase4/final_scored": n_scored,
        "phase4/final_no_hits": n_no_hits,
        "phase4/final_skipped": n_skipped,
        "phase4/score_rate": n_scored / max(n_total, 1),
    })
    print(f"[phase4] Done: {n_scored} scored, {n_no_hits} no NMPFam hits, {n_skipped} skipped")

    _write_summary(inference_dir)


def _write_summary(inference_dir: Path) -> None:
    features: dict[str, Any] = {}
    for p in sorted(inference_dir.glob("????.json")):
        try:
            d = json.loads(p.read_text())
            fid = str(d["feature_id"])
            hits = d.get("nmpfam_hits", [])
            mean_probs = [h["mean_geom_prob_at_active"] for h in hits
                          if h.get("mean_geom_prob_at_active", 0) > 0]
            features[fid] = {
                "n_nmpfam_hits": d["n_nmpfam_hits"],
                "mean_geom_prob_across_hits": (
                    round(float(np.mean(mean_probs)), 4) if mean_probs else 0.0
                ),
                "max_geom_prob_across_hits": (
                    round(max(h.get("max_geom_prob", 0) for h in hits), 4) if hits else 0.0
                ),
            }
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    all_hit_counts = [v["n_nmpfam_hits"] for v in features.values()]
    all_mean_probs = [v["mean_geom_prob_across_hits"] for v in features.values()
                      if v["mean_geom_prob_across_hits"] > 0]

    summary = {"n_features_scored": len(features), "features": features}
    (inference_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    wlog({
        "summary/n_features_scored": len(features),
        "summary/total_nmpfam_hits": sum(all_hit_counts) if all_hit_counts else 0,
        "summary/mean_hits_per_feature": float(np.mean(all_hit_counts)) if all_hit_counts else 0,
        "summary/mean_geom_prob_across_features": float(np.mean(all_mean_probs)) if all_mean_probs else 0,
        "summary/median_geom_prob_across_features": float(np.median(all_mean_probs)) if all_mean_probs else 0,
    })
    print(f"[summary] Wrote summary for {len(features)} features to {inference_dir / 'summary.json'}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _resolve_phases(args: argparse.Namespace) -> set[int]:
    if args.phase is not None:
        return {args.phase}
    if args.start_phase is not None:
        end = args.end_phase if args.end_phase is not None else 4
        return set(range(args.start_phase, end + 1))
    return {1, 2, 3, 4}


def _load_cached_families(data_dir: Path) -> list[dict]:
    """Load families.json produced by phase 1, filter to entries with sequences."""
    families_path = data_dir / "nmpfam" / "families.json"
    if not families_path.is_file():
        raise SystemExit(
            f"ERROR: {families_path} not found. Run --phase 1 first to populate it."
        )
    families = json.loads(families_path.read_text())
    families_with_seq = [f for f in families if f.get("consensus_sequence")]
    if not families_with_seq:
        raise SystemExit(
            f"ERROR: {families_path} has no families with consensus_sequence. "
            "Re-run --phase 1 to download FASTAs."
        )
    return families_with_seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NMPFam SAE activation + presaved-GBM geometry inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_nmpfam.py --data-dir feature_data_cluster
  python scripts/run_nmpfam.py --data-dir feature_data_cluster --n-families 5
  python scripts/run_nmpfam.py --data-dir feature_data_cluster --phase 4
  python scripts/run_nmpfam.py --data-dir feature_data_cluster --start-phase 2 --end-phase 3
  PIPELINE_WORKERS=8 python scripts/run_nmpfam.py --data-dir feature_data_cluster

Cluster split (phases 1/3/4 on CPU pods, phase 2 on GPU; phases 2 and 3
can run in parallel after phase 1):
  python scripts/run_nmpfam.py --data-dir feature_data_cluster --phase 1   # CPU
  python scripts/run_nmpfam.py --data-dir feature_data_cluster --phase 2   # GPU
  python scripts/run_nmpfam.py --data-dir feature_data_cluster --phase 3   # CPU
  PIPELINE_WORKERS=8 python scripts/run_nmpfam.py --data-dir feature_data_cluster --phase 4
""",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("feature_data_cluster"))
    parser.add_argument("--sae-dir", type=Path, default=Path("trained_models/fiery-sweep"))
    parser.add_argument("--n-families", type=int, default=50000)
    parser.add_argument("--activation-fraction", type=float, default=0.5)
    parser.add_argument("--esm-model", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--esm-layer", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4], default=None,
        help="Run only this phase (default: run all four).",
    )
    phase_group.add_argument(
        "--start-phase", type=int, choices=[1, 2, 3, 4], default=None,
        help="First phase to run (inclusive). Use with --end-phase for a range.",
    )
    parser.add_argument(
        "--end-phase", type=int, choices=[1, 2, 3, 4], default=None,
        help="Last phase to run (inclusive). Defaults to 4 if omitted.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="proteinlens-nmpfam")
    args = parser.parse_args()
    if args.end_phase is not None and args.start_phase is None:
        parser.error("--end-phase requires --start-phase")
    if args.start_phase is not None and args.end_phase is not None \
            and args.start_phase > args.end_phase:
        parser.error(
            f"--start-phase {args.start_phase} comes after "
            f"--end-phase {args.end_phase}"
        )
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    phases_to_run = _resolve_phases(args)
    phases_label = ",".join(str(p) for p in sorted(phases_to_run))

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config={
                "data_dir": str(args.data_dir),
                "sae_dir": str(args.sae_dir),
                "n_families": args.n_families,
                "activation_fraction": args.activation_fraction,
                "esm_model": args.esm_model,
                "esm_layer": args.esm_layer,
                "phases": phases_label,
                "pipeline_workers": os.environ.get("PIPELINE_WORKERS", "1"),
                "seed": args.seed,
            },
        )

    print("=" * 70)
    print("NMPFam SAE + Presaved-GBM Inference")
    print("=" * 70)
    print(f"  Data dir:        {args.data_dir}")
    print(f"  SAE dir:         {args.sae_dir}")
    print(f"  N families:      {args.n_families}")
    print(f"  Hit threshold:   {args.activation_fraction} × global max")
    print(f"  Phases:          {phases_label}")
    print(f"  Workers:         {os.environ.get('PIPELINE_WORKERS', '1')}")
    print("=" * 70)

    t0 = time.time()
    nmpfam_dir = args.data_dir / "nmpfam"

    families: list[dict] | None = None
    if 1 in phases_to_run:
        print("\n>>> Phase 1: Fetch NMPFams families")
        families = run_phase1_fetch(args.n_families, args.data_dir, args.seed)
        print(f"    {len(families)} families with consensus sequences")
    elif phases_to_run & {2, 3}:
        families = _load_cached_families(args.data_dir)
        print(f"\n[main] Loaded {len(families)} cached families from phase 1")

    if 2 in phases_to_run:
        phase2_done = (
            (nmpfam_dir / "feature_maxes.npy").is_file()
            and (nmpfam_dir / "family_index.json").is_file()
        )
        if phase2_done:
            print("\n>>> Phase 2: SAE inference [SKIPPED — feature_maxes.npy exists]")
        else:
            print("\n>>> Phase 2: SAE inference")
            run_phase2_inference(
                families, args.sae_dir, args.data_dir,
                args.esm_model, args.esm_layer, args.max_seq_len, args.device,
            )

    if 3 in phases_to_run:
        geom_dir = nmpfam_dir / "geometry_residue_profiles"
        if geom_dir.is_dir():
            existing_geom = {p.stem for p in geom_dir.glob("*.npz")}
            family_ids = {f["ID"] for f in families}
            missing_geom = family_ids - existing_geom
            if not missing_geom:
                print(f"\n>>> Phase 3: Geometry profiles [SKIPPED — all {len(existing_geom)} exist]")
            else:
                print(f"\n>>> Phase 3: Geometry profiles [{len(existing_geom)} cached, {len(missing_geom)} remaining]")
                run_phase3_geometry(families, args.data_dir)
        else:
            print("\n>>> Phase 3: Geometry profiles")
            run_phase3_geometry(families, args.data_dir)

    if 4 in phases_to_run:
        print("\n>>> Phase 4: Presaved-GBM inference")
        run_phase4_gbm_inference(args.data_dir, args.activation_fraction)

    elapsed = time.time() - t0
    wlog({"total_time_seconds": elapsed})
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
