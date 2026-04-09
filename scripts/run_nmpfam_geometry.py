#!/usr/bin/env python3
"""Metagenomic geometry inference: retrain SwissProt classifiers, score NMPFam proteins.

Overview
--------
This script extends the NMPFam (Novel Metagenomic Protein Families) analysis
with geometric classifier inference.  It answers the question: *do the local
backbone geometry patterns that an SAE feature learned from SwissProt also
appear in unseen metagenomic proteins that activate the same feature?*

The pipeline has four phases:

  Phase 1 — Fetch NMPFams families from the API and download consensus
            sequences.  We request one representative per cluster, targeting
            up to 50 000 proteins for broad coverage of metagenomic dark matter.

  Phase 2 — Run ESM → SAE inference on the NMPFam consensus sequences to
            obtain per-residue SAE activations.

  Phase 3 — Stream PDB structures from the NMPFams API and compute residue-
            level geometry profiles (curvature, torsion, planarity, tangent
            vectors, helix masks).

  Phase 4 — For each SAE feature that has a geometry enrichment JSON from the
            SwissProt pipeline (Stage 6c):

            a) Retrain the GBM classifier from SwissProt data.  All classifiers
               use random_state=42 throughout (GBM, DT, RF, KFold splits,
               background sampling RNG), so this produces **identical** models
               to Stage 6c.  The training data comes from the existing
               geometry_residue_profiles/*.npz and residue_activations/*.npz
               files — the same files Stage 6c used.

            b) Run predict_proba on every NMPFam protein that activates this
               feature above a threshold.  This is pure inference — no new
               training, no modification of existing metrics.

            c) Write results to nmpfam/geometry_inference/NNNN.json.

Outputs
-------
All outputs go to ``{data_dir}/nmpfam/``.  **No existing pipeline data is
modified** — the geometry_enrichment/ directory and all Stage 0–8 outputs are
read-only inputs.

  nmpfam/
    families.json                    — sampled family metadata (cached)
    family_index.json                — family_id → row index mapping
    feature_maxes.npy                — (n_families, n_features) max activations
    fasta/{family_id}.fasta          — consensus sequences (cached)
    residue_activations/{fid}.npz    — per-residue SAE activations
    geometry_residue_profiles/{fid}.npz — per-residue geometry descriptors
    geometry_inference/
      {node:04d}.json                — per-feature NMPFam geometry predictions
      summary.json                   — aggregated stats across all features

Reproducibility
---------------
Classifier retraining is deterministic because:
  - GBM: random_state=42, fixed hyperparameters
  - Background sampling: np.random.default_rng(42)
  - CV splits: StratifiedKFold(random_state=42) or GroupKFold (deterministic)
  - RF: random_state=42

The retrained classifiers are numerically identical to those used in Stage 6c,
so geometry_inference results are directly comparable to the concordance metrics
already reported.

Usage
-----
    # Full run (50k families, ~2-4h on GPU node)
    python scripts/run_nmpfam_geometry.py --data-dir feature_data_cluster

    # Quick test (5 families)
    python scripts/run_nmpfam_geometry.py --data-dir feature_data_cluster --n-families 5

    # Skip NMPFam download/inference if already done (Phase 4 only)
    python scripts/run_nmpfam_geometry.py --data-dir feature_data_cluster --phase4-only

    # Control parallelism for classifier retraining
    PIPELINE_WORKERS=8 python scripts/run_nmpfam_geometry.py --data-dir feature_data_cluster
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proteinlens.analysis.geometry.classifiers import (
    collect_node_fragments,
    train_motif_classifier,
)
from proteinlens.analysis.geometry.residue_features import (
    ACTIVE_GEOM_NAMES,
    ca_backbone,
    compute_residue_profiles,
    detect_alpha_helices_from_ca,
    extract_local_feature_vector,
    select_features,
)
from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog
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
    """Query the NMPFams /families endpoint and return all families."""
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
    """Stratified sample across Category, keeping only families with PDB structures."""
    with_pdb = [f for f in families if f.get("PDB") == "Y"]
    logger.info("%d / %d families have PDB structures", len(with_pdb), len(families))

    if len(with_pdb) <= n_target:
        logger.info("Fewer families with PDB than target — using all %d", len(with_pdb))
        return with_pdb

    # Proportional allocation across categories
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for f in with_pdb:
        by_cat[f.get("Category", "Unknown")].append(f)

    rng = random.Random(seed)
    sampled: list[dict] = []
    remaining = n_target
    cat_items = sorted(by_cat.items(), key=lambda x: len(x[1]))

    for i, (cat, members) in enumerate(cat_items):
        cats_left = len(cat_items) - i
        n_alloc = min(len(members), max(1, remaining // cats_left))
        sampled.extend(rng.sample(members, n_alloc))
        remaining -= n_alloc

    if remaining > 0:
        all_unsampled = [f for f in with_pdb if f not in sampled]
        if all_unsampled:
            sampled.extend(rng.sample(all_unsampled, min(remaining, len(all_unsampled))))

    logger.info("Sampled %d families across %d categories", len(sampled), len(by_cat))
    return sampled


def _parse_first_sequence(fasta_text: str) -> str | None:
    """Extract the first sequence from FASTA text."""
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
    family_id: str, fasta_dir: Path, session: requests.Session
) -> str | None:
    """Download FASTA for a family; return the first (consensus) sequence."""
    fasta_path = fasta_dir / f"{family_id}.fasta"
    if fasta_path.exists():
        return _parse_first_sequence(fasta_path.read_text())
    url = f"{NMPFAM_DATA_BASE}/fasta/{family_id}.fasta"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            logger.warning("FASTA download failed for %s: HTTP %d", family_id, resp.status_code)
            return None
        fasta_path.write_text(resp.text)
        return _parse_first_sequence(resp.text)
    except requests.RequestException as e:
        logger.warning("FASTA download error for %s: %s", family_id, e)
        return None


def fetch_pdb_text(family_id: str, session: requests.Session) -> str | None:
    """Fetch PDB text from NMPFams (not cached — small files, streamed)."""
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
    """Phase 1: Fetch NMPFams families, stratified sample, download FASTA."""
    nmpfam_dir = data_dir / "nmpfam"
    nmpfam_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir = nmpfam_dir / "fasta"
    fasta_dir.mkdir(exist_ok=True)
    families_path = nmpfam_dir / "families.json"

    session = requests.Session()
    session.headers["User-Agent"] = "ProteinLens/1.0 (research)"

    # Resume from cached family list if available
    if families_path.exists():
        logger.info("Loading cached families from %s", families_path)
        with open(families_path) as f:
            sampled = json.load(f)
        # Fill in sequences from cache or re-download missing ones
        n_missing = 0
        for fam in tqdm(sampled, desc="Checking FASTA cache"):
            fid = fam["ID"]
            if fam.get("consensus_sequence"):
                continue
            seq = download_consensus_sequence(fid, fasta_dir, session)
            if seq:
                fam["consensus_sequence"] = seq
                fam["consensus_length"] = len(seq)
            if not (fasta_dir / f"{fid}.fasta").exists():
                n_missing += 1
                time.sleep(0.1)
        if n_missing:
            print(f"  Downloaded {n_missing} missing FASTA files")
        # Update families.json with any newly filled sequences
        sampled_with_seq = [f for f in sampled if f.get("consensus_sequence")]
        with open(families_path, "w") as f:
            json.dump(sampled, f, indent=2)
        return sampled_with_seq

    # Fresh fetch
    all_families = fetch_all_families(session)
    sampled = sample_families_with_structures(all_families, n_families, seed)

    # Save families.json immediately so restarts use the resume path
    with open(families_path, "w") as f:
        json.dump(sampled, f, indent=2)

    print(f"[phase1] Downloading consensus sequences for {len(sampled)} families...")
    for fam in tqdm(sampled, desc="Downloading FASTA"):
        fid = fam["ID"]
        seq = download_consensus_sequence(fid, fasta_dir, session)
        if seq:
            fam["consensus_sequence"] = seq
            fam["consensus_length"] = len(seq)
        time.sleep(0.1)

    sampled = [f for f in sampled if f.get("consensus_sequence")]
    logger.info("Got sequences for %d families", len(sampled))

    # Category breakdown for logging
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

    # Update families.json with sequences filled in
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
    """Phase 2: Run ESM → SAE on consensus sequences, save per-residue activations."""
    nmpfam_dir = data_dir / "nmpfam"
    act_dir = nmpfam_dir / "residue_activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    device = device or get_device()
    print(f"[phase2] Loading ESM ({esm_model}) and SAE ({sae_dir}) on {device}...")

    esm = ESM(esm_model, device=device)
    sae = load_sae(str(sae_dir), device=device)
    n_features = sae.dict_size

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
        if npz_path.exists():
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

    # Summary stats
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
    """Phase 3: Stream PDBs from NMPFams API, compute residue geometry profiles."""
    nmpfam_dir = data_dir / "nmpfam"
    geom_dir = nmpfam_dir / "geometry_residue_profiles"
    geom_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "ProteinLens/1.0 (research)"

    n_computed = 0
    n_cached = 0
    n_failed = 0

    print(f"[phase3] Computing geometry profiles for {len(families)} families...")
    for fam in tqdm(families, desc="Geometry profiles"):
        fid = fam["ID"]
        geom_path = geom_dir / f"{fid}.npz"
        if geom_path.exists():
            n_cached += 1
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
            n_computed += 1
        except Exception as e:
            logger.warning("Geometry failed for %s: %s", fid, e)
            n_failed += 1

        time.sleep(0.1)

    wlog({
        "phase3/n_computed": n_computed,
        "phase3/n_cached": n_cached,
        "phase3/n_failed": n_failed,
        "phase3/success_rate": (n_computed + n_cached) / max(len(families), 1),
    })
    print(f"[phase3] Geometry: {n_computed} new, {n_cached} cached, {n_failed} failed")


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Retrain SwissProt classifiers and score NMPFam proteins
# ═══════════════════════════════════════════════════════════════════════
#
# This is the core contribution of this script.  For each SAE node:
#
#   1. We re-collect activated/background fragments from the SAME SwissProt
#      geometry and activation data that Stage 6c used.
#
#   2. We retrain the GBM with identical hyperparameters and random_state=42,
#      producing a numerically identical classifier.
#
#   3. We then run predict_proba on NMPFam proteins that activate this node,
#      extracting local feature vectors from their geometry profiles and
#      passing them through the retrained GBM.
#
# The result tells us: for each NMPFam protein × SAE feature, does the
# geometry at activated residues match the structural motif the classifier
# learned from SwissProt?  High predicted probabilities indicate the
# metagenomic protein has the same local backbone geometry as the SwissProt
# training set — a strong signal that the SAE feature has generalised to
# novel sequence space.


# ── Module-level shared state for multiprocessing (set before fork) ──
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
    """Run geometry classifier inference on a single NMPFam protein for one node.

    Returns a dict with per-residue geometry probabilities and concordance
    labels, or None if the protein can't be scored (too short, no geometry, etc.).
    """
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

    # Extract feature vectors for all scorable positions
    batch_fvs: list[np.ndarray] = []
    batch_positions: list[int] = []
    for pos in range(half_w, n - half_w):
        fv = extract_local_feature_vector(profiles, ca[:n], pos, half_w, sequence=seq)
        if fv is not None and np.all(np.isfinite(fv)):
            batch_fvs.append(select_features(fv))
            batch_positions.append(pos)

    if not batch_fvs:
        return None

    # Batched predict_proba — single call for all positions
    X = np.array(batch_fvs)
    probs = gbm.predict_proba(X)

    geom_prob_profile = [0.0] * n
    for j, pos in enumerate(batch_positions):
        p = probs[j]
        geom_prob_profile[pos] = float(p[1] if len(p) > 1 else p[0])

    sae_profile = [float(node_acts[i]) for i in range(n)]

    # Concordance: does SAE activation agree with geometry prediction?
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

    return {
        "family_id": fid,
        "n_residues": n,
        "sequence": seq[:n],
        "sae_activation_profile": sae_profile,
        "geom_prob_profile": geom_prob_profile,
        "concordance_labels": concordance_labels,
        "max_sae_activation": float(node_acts[:n].max()),
        "max_geom_prob": max(geom_prob_profile),
        "mean_geom_prob_at_active": (
            float(np.mean([geom_prob_profile[i] for i in range(n) if node_acts[i] >= activation_threshold]))
            if any(node_acts[i] >= activation_threshold for i in range(n))
            else 0.0
        ),
        "n_agree": n_agree,
        "n_sae_only": n_sae_only,
        "n_geom_only": n_geom_only,
    }


def _process_node_phase4(ni: int) -> tuple[int, str]:
    """Retrain SwissProt classifier for node ni, then score NMPFam proteins.

    This function runs in a forked worker process.  All shared data is
    read-only via the module-level _shared dict (copy-on-write after fork).

    Returns (node_idx, status) where status is 'scored', 'skipped', or 'no_hits'.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    s = _shared

    # ── Unpack shared SwissProt data ──
    act_matrix_full = s["act_matrix_full"]
    geom_cache = s["geom_cache"]
    row_to_acc = s["row_to_acc"]
    sp_available = s["sp_available"]
    half_w = s["half_w"]
    cfg = s["config_params"]
    inference_dir: Path = s["inference_dir"]

    # ── Unpack shared NMPFam data ──
    nmpfam_feature_maxes = s["nmpfam_feature_maxes"]
    nmpfam_act_cache = s["nmpfam_act_cache"]
    nmpfam_geom_cache = s["nmpfam_geom_cache"]
    row_to_fid = s["row_to_fid"]
    global_max = s["global_max"]
    activation_fraction = s["activation_fraction"]

    # ── Per-process SwissProt activation LRU cache ──
    if not hasattr(_process_node_phase4, "_act_cache"):
        _process_node_phase4._act_cache = {}
        _process_node_phase4._act_order = []

    _act_cache = _process_node_phase4._act_cache
    _act_order = _process_node_phase4._act_order
    _ACT_CACHE_MAX = 500

    def _load_sp_activations(acc: str) -> np.ndarray | None:
        if acc in _act_cache:
            return _act_cache[acc]
        path = sp_available.get(acc)
        if path is None:
            return None
        try:
            arr = np.load(path, allow_pickle=True)["activations"]
        except Exception:
            return None
        _act_cache[acc] = arr
        _act_order.append(acc)
        if len(_act_order) > _ACT_CACHE_MAX:
            evict = _act_order.pop(0)
            _act_cache.pop(evict, None)
        return arr

    # ── Step A: Rebuild SwissProt protein_data for this node ──
    # (Same logic as Stage 6c _process_node, lines 201-234)
    node_col = act_matrix_full[:, ni]
    active_rows = np.where(node_col > 0)[0]

    if len(active_rows) > 500:
        top_idx = np.argsort(node_col[active_rows])[-500:]
        active_rows = active_rows[top_idx]

    protein_data: list[dict] = []
    for row_idx in active_rows:
        acc = row_to_acc.get(int(row_idx))
        if acc is None:
            continue
        g = geom_cache.get(acc)
        if g is None:
            continue
        act_mat = _load_sp_activations(acc)
        if act_mat is None:
            continue
        ca = g["ca"]
        n = min(len(ca), act_mat.shape[0])
        if n < 20:
            continue
        seq_arr = g.get("sequence", np.array([""]))
        protein_data.append({
            "accession": acc,
            "act_matrix": act_mat[:n],
            "ca": ca[:n],
            "profiles": {k: g[k][:n] for k in ("curvature", "torsion", "planarity", "tangents", "helix_mask", "categories")},
            "n_residues": n,
            "sequence": str(seq_arr[0]) if len(seq_arr) > 0 else "",
        })

    if not protein_data:
        return (ni, "skipped")

    total_activated = sum(int(np.sum(p["act_matrix"][:, ni] > 0)) for p in protein_data)
    if total_activated < cfg["geometry_min_activated_positions"]:
        return (ni, "skipped")

    # ── Step B: Collect fragments and retrain classifier (deterministic) ──
    frag_result = collect_node_fragments(
        protein_data, ni, half_w=half_w,
        act_quantile=cfg["geometry_act_quantile"],
        max_fragments=cfg["geometry_frag_top_k"],
        bg_ratio=cfg["geometry_bg_ratio"],
    )
    activated = frag_result["activated"]
    background = frag_result["background"]

    if len(activated) < 20 or len(background) < 20:
        return (ni, "skipped")

    clf_result = train_motif_classifier(
        activated, background,
        feature_names=list(ACTIVE_GEOM_NAMES),
        cv_folds=cfg["geometry_classifier_cv_folds"],
    )

    gbm = clf_result["tree"]
    if gbm is None:
        return (ni, "skipped")

    activation_threshold = frag_result["threshold"]
    geom_threshold = clf_result["optimal_threshold"]

    # ── Step C: Identify NMPFam hits for this node ──
    gmax = float(global_max[ni])
    if gmax <= 0:
        return (ni, "skipped")

    hit_threshold = activation_fraction * gmax
    nmpfam_col = nmpfam_feature_maxes[:, ni]
    hit_rows = np.where(nmpfam_col > hit_threshold)[0]

    if len(hit_rows) == 0:
        return (ni, "no_hits")

    # Sort by activation descending
    hit_rows = hit_rows[np.argsort(nmpfam_col[hit_rows])[::-1]]

    # ── Step D: Score each NMPFam hit with the retrained classifier ──
    scored_hits: list[dict] = []
    for row_idx in hit_rows:
        fid = row_to_fid.get(int(row_idx))
        if fid is None:
            continue
        nmpfam_act = nmpfam_act_cache.get(fid)
        nmpfam_geom = nmpfam_geom_cache.get(fid)
        if nmpfam_act is None or nmpfam_geom is None:
            continue

        hit_result = _score_nmpfam_protein(
            fid, ni, nmpfam_act, nmpfam_geom, gbm,
            activation_threshold, geom_threshold, half_w,
        )
        if hit_result is not None:
            scored_hits.append(hit_result)

    if not scored_hits:
        return (ni, "no_hits")

    # ── Step E: Write results ──
    out = {
        "feature_id": ni,
        "feature_global_max": round(gmax, 4),
        "activation_threshold_sae": round(activation_threshold, 4),
        "optimal_geom_threshold": round(geom_threshold, 4),
        "gbm_auc_cv": clf_result["gbm_auc_cv"],
        "n_swissprot_activated": len(activated),
        "n_swissprot_background": len(background),
        "n_nmpfam_hits": len(scored_hits),
        "nmpfam_hits": scored_hits,
    }
    out_path = inference_dir / f"{ni:04d}.json"
    out_path.write_text(json.dumps(out, indent=2))

    return (ni, "scored")


def run_phase4_geometry_inference(
    data_dir: Path,
    activation_fraction: float = 0.5,
) -> None:
    """Phase 4: Retrain SwissProt GBMs and score NMPFam proteins.

    Reads from:
      - {data_dir}/geometry_residue_profiles/*.npz  (SwissProt, read-only)
      - {data_dir}/residue_activations/*.npz         (SwissProt, read-only)
      - {data_dir}/protein_feature_maxes.npy          (SwissProt, read-only)
      - {data_dir}/pipeline_state.json                (SwissProt, read-only)
      - {data_dir}/feature_max_activations.npy        (SwissProt, read-only)
      - {data_dir}/geometry_enrichment/*.json          (read-only, to know which nodes have classifiers)
      - {data_dir}/nmpfam/residue_activations/*.npz    (NMPFam)
      - {data_dir}/nmpfam/geometry_residue_profiles/*.npz (NMPFam)
      - {data_dir}/nmpfam/feature_maxes.npy             (NMPFam)

    Writes to:
      - {data_dir}/nmpfam/geometry_inference/*.json   (NEW — never overwrites existing pipeline data)
    """
    global _shared
    nmpfam_dir = data_dir / "nmpfam"
    inference_dir = nmpfam_dir / "geometry_inference"
    inference_dir.mkdir(parents=True, exist_ok=True)

    # ── Load SwissProt pipeline state ──
    state_path = data_dir / "pipeline_state.json"
    if not state_path.exists():
        print("[phase4] ERROR: pipeline_state.json not found — Stage 6c data required")
        return
    state = json.loads(state_path.read_text())
    acc_to_idx: dict[str, int] = state.get("accession_index", {})
    n_proteins = len(acc_to_idx)

    # Feature max activations (global, from SwissProt survey)
    global_max_path = data_dir / "feature_max_activations.npy"
    if not global_max_path.exists():
        print("[phase4] ERROR: feature_max_activations.npy not found")
        return
    global_max = np.load(global_max_path)
    n_features = len(global_max)

    # SwissProt protein-feature max matrix (memmap → RAM)
    pfm_path = data_dir / "protein_feature_maxes.npy"
    if not pfm_path.exists():
        print("[phase4] ERROR: protein_feature_maxes.npy not found")
        return
    print(f"[phase4] Loading SwissProt protein-feature max matrix ({n_proteins} × {n_features})...")
    act_matrix_full = np.array(np.memmap(
        pfm_path, dtype=np.float32, mode="r", shape=(n_proteins, n_features),
    ))

    half_w = 10  # Must match Stage 6c config (geometry_fragment_half_w)

    # ── Index SwissProt NPZ files ──
    sp_geom_dir = data_dir / "geometry_residue_profiles"
    sp_act_dir = data_dir / "residue_activations"
    sp_interpro_dir = data_dir / "interpro_residue_activations"

    sp_geom_files = {p.stem: p for p in sp_geom_dir.glob("*.npz")} if sp_geom_dir.exists() else {}
    sp_act_files = {p.stem: p for p in sp_act_dir.glob("*.npz")} if sp_act_dir.exists() else {}
    sp_interpro_files = {p.stem: p for p in sp_interpro_dir.glob("*.npz")} if sp_interpro_dir.exists() else {}

    # Build available map: accession → activation NPZ path (geometry + activations required)
    sp_available: dict[str, Path] = {}
    for acc in acc_to_idx:
        if acc not in sp_geom_files:
            continue
        if acc in sp_act_files:
            sp_available[acc] = sp_act_files[acc]
        elif acc in sp_interpro_files:
            sp_available[acc] = sp_interpro_files[acc]

    print(f"[phase4] SwissProt: {len(sp_available)} proteins with geometry + activations")
    if not sp_available:
        print("[phase4] ERROR: No SwissProt proteins found — download NPZ files from cluster first")
        return

    # ── Preload SwissProt geometry profiles into RAM ──
    def _load_geom(acc: str) -> tuple[str, dict | None]:
        try:
            g = np.load(sp_geom_files[acc], allow_pickle=True)
            return acc, {
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
            return acc, None

    geom_cache: dict[str, dict] = {}
    print(f"[phase4] Preloading {len(sp_available)} SwissProt geometry profiles...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, (acc, data) in enumerate(pool.map(_load_geom, sp_available)):
            if data is not None:
                geom_cache[acc] = data
            if (i + 1) % 5000 == 0:
                logger.info("  Preloaded %d/%d geometry profiles", i + 1, len(sp_available))
    print(f"[phase4] Preloaded {len(geom_cache)} SwissProt geometry profiles")

    row_to_acc = {v: k for k, v in acc_to_idx.items() if k in sp_available}

    # ── Load NMPFam data ──
    nmpfam_fm_path = nmpfam_dir / "feature_maxes.npy"
    nmpfam_fi_path = nmpfam_dir / "family_index.json"
    if not nmpfam_fm_path.exists() or not nmpfam_fi_path.exists():
        print("[phase4] ERROR: NMPFam feature_maxes.npy / family_index.json not found — run Phases 1-3 first")
        return

    nmpfam_feature_maxes = np.load(nmpfam_fm_path)
    with open(nmpfam_fi_path) as f:
        family_index = json.load(f)
    row_to_fid = {v: k for k, v in family_index.items()}

    # Preload NMPFam activations and geometry
    nmpfam_act_dir = nmpfam_dir / "residue_activations"
    nmpfam_geom_dir = nmpfam_dir / "geometry_residue_profiles"

    nmpfam_act_cache: dict[str, np.ndarray] = {}
    nmpfam_geom_cache: dict[str, dict] = {}

    print(f"[phase4] Preloading NMPFam activations and geometry profiles...")
    for fid in tqdm(family_index, desc="Loading NMPFam data"):
        act_path = nmpfam_act_dir / f"{fid}.npz"
        geom_path = nmpfam_geom_dir / f"{fid}.npz"
        if act_path.exists():
            try:
                nmpfam_act_cache[fid] = np.load(act_path)["activations"]
            except Exception:
                pass
        if geom_path.exists():
            try:
                g = np.load(geom_path, allow_pickle=True)
                nmpfam_geom_cache[fid] = {
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
                pass

    n_scorable = len(set(nmpfam_act_cache) & set(nmpfam_geom_cache))
    wlog({
        "phase4/sp_proteins_available": len(sp_available),
        "phase4/sp_geom_profiles_loaded": len(geom_cache),
        "phase4/nmpfam_activations_loaded": len(nmpfam_act_cache),
        "phase4/nmpfam_geom_profiles_loaded": len(nmpfam_geom_cache),
        "phase4/nmpfam_scorable": n_scorable,
    })
    print(f"[phase4] NMPFam: {len(nmpfam_act_cache)} activations, "
          f"{len(nmpfam_geom_cache)} geometry profiles, {n_scorable} scorable")

    # ── Determine which nodes to process ──
    # Only process nodes that had a geometry enrichment JSON from Stage 6c
    geom_enrichment_dir = data_dir / "geometry_enrichment"
    nodes_with_classifiers: set[int] = set()
    for feat_path in geom_enrichment_dir.glob("????.json"):
        try:
            feat_json = json.loads(feat_path.read_text())
            if "geometric_residue_level" in feat_json:
                nodes_with_classifiers.add(feat_json["feature_id"])
        except (json.JSONDecodeError, OSError):
            continue

    # Skip nodes already scored (resumability)
    done_nodes: set[int] = set()
    for p in inference_dir.glob("????.json"):
        try:
            done_nodes.add(int(p.stem))
        except ValueError:
            continue

    nodes_to_process = sorted(nodes_with_classifiers - done_nodes)
    print(f"[phase4] {len(nodes_with_classifiers)} nodes have Stage 6c classifiers, "
          f"{len(done_nodes)} already scored, {len(nodes_to_process)} to process")

    if not nodes_to_process:
        print("[phase4] Nothing to do.")
        _write_summary(inference_dir)
        return

    # ── Set shared state for worker processes ──
    _shared.update({
        "act_matrix_full": act_matrix_full,
        "geom_cache": geom_cache,
        "row_to_acc": row_to_acc,
        "sp_available": sp_available,
        "half_w": half_w,
        "global_max": global_max,
        "inference_dir": inference_dir,
        "activation_fraction": activation_fraction,
        "nmpfam_feature_maxes": nmpfam_feature_maxes,
        "nmpfam_act_cache": nmpfam_act_cache,
        "nmpfam_geom_cache": nmpfam_geom_cache,
        "row_to_fid": row_to_fid,
        "config_params": {
            "geometry_min_activated_positions": 200,
            "geometry_act_quantile": 0.80,
            "geometry_frag_top_k": 100,
            "geometry_bg_ratio": 3,
            "geometry_classifier_cv_folds": 5,
        },
    })

    # ── Parallel execution ──
    n_workers = min(
        int(os.environ.get("PIPELINE_WORKERS", "1")),
        len(nodes_to_process) or 1,
    )
    n_scored = 0
    n_skipped = 0
    n_no_hits = 0

    pbar = tqdm(total=len(nodes_to_process), desc="Geometry inference")

    n_total = len(nodes_to_process)
    log_every = max(1, n_total // 50)  # ~50 log points over the run

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
    """Build summary.json from all per-feature geometry inference outputs."""
    features: dict[str, Any] = {}
    for p in sorted(inference_dir.glob("????.json")):
        try:
            d = json.loads(p.read_text())
            fid = str(d["feature_id"])
            hits = d.get("nmpfam_hits", [])
            mean_geom_probs = [h["mean_geom_prob_at_active"] for h in hits if h.get("mean_geom_prob_at_active", 0) > 0]
            features[fid] = {
                "n_nmpfam_hits": d["n_nmpfam_hits"],
                "gbm_auc_cv": d.get("gbm_auc_cv", 0.0),
                "mean_geom_prob_across_hits": (
                    round(float(np.mean(mean_geom_probs)), 4) if mean_geom_probs else 0.0
                ),
                "max_geom_prob_across_hits": (
                    round(max(h.get("max_geom_prob", 0) for h in hits), 4) if hits else 0.0
                ),
            }
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    # Aggregate stats across all scored features
    all_hit_counts = [v["n_nmpfam_hits"] for v in features.values()]
    all_mean_probs = [v["mean_geom_prob_across_hits"] for v in features.values() if v["mean_geom_prob_across_hits"] > 0]
    all_aucs = [v["gbm_auc_cv"] for v in features.values() if v["gbm_auc_cv"] > 0]

    summary = {
        "n_features_scored": len(features),
        "features": features,
    }
    (inference_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    wlog({
        "summary/n_features_scored": len(features),
        "summary/total_nmpfam_hits": sum(all_hit_counts) if all_hit_counts else 0,
        "summary/mean_hits_per_feature": float(np.mean(all_hit_counts)) if all_hit_counts else 0,
        "summary/mean_geom_prob_across_features": float(np.mean(all_mean_probs)) if all_mean_probs else 0,
        "summary/median_geom_prob_across_features": float(np.median(all_mean_probs)) if all_mean_probs else 0,
        "summary/mean_gbm_auc_cv": float(np.mean(all_aucs)) if all_aucs else 0,
    })
    print(f"[summary] Wrote summary for {len(features)} features to {inference_dir / 'summary.json'}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NMPFam geometry inference: retrain SwissProt classifiers and score metagenomic proteins.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (50k families, GPU recommended for Phase 2)
  python scripts/run_nmpfam_geometry.py --data-dir feature_data_cluster

  # Quick test
  python scripts/run_nmpfam_geometry.py --data-dir feature_data_cluster --n-families 5

  # Phase 4 only (Phases 1-3 already done)
  python scripts/run_nmpfam_geometry.py --data-dir feature_data_cluster --phase4-only

  # Parallel classifier retraining (8 workers)
  PIPELINE_WORKERS=8 python scripts/run_nmpfam_geometry.py --data-dir feature_data_cluster
""",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("feature_data_cluster"),
        help="Pipeline output directory with Stage 6c results (default: feature_data_cluster/)",
    )
    parser.add_argument(
        "--sae-dir", type=Path, default=Path("trained_models/fiery-sweep"),
        help="Path to trained SAE directory",
    )
    parser.add_argument(
        "--n-families", type=int, default=50000,
        help="Number of NMPFams families to sample (default: 50000 — one per cluster)",
    )
    parser.add_argument(
        "--activation-fraction", type=float, default=0.5,
        help="Fraction of global max for NMPFam hit threshold (default: 0.5)",
    )
    parser.add_argument(
        "--esm-model", type=str, default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace ESM model name",
    )
    parser.add_argument(
        "--esm-layer", type=int, default=3,
        help="ESM layer for embeddings",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=1024,
        help="Max sequence length (longer sequences skipped)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="PyTorch device (default: auto-detect)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for family sampling",
    )
    parser.add_argument(
        "--phase4-only", action="store_true",
        help="Skip Phases 1-3 (NMPFam download/inference) and run geometry inference only",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable wandb logging for run progress and metrics",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="proteinlens-nmpfam",
        help="wandb project name (default: proteinlens-nmpfam)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    # ── Optional wandb init ──
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
                "phase4_only": args.phase4_only,
                "pipeline_workers": os.environ.get("PIPELINE_WORKERS", "1"),
                "seed": args.seed,
            },
        )

    print("=" * 70)
    print("NMPFam Geometry Inference")
    print("=" * 70)
    print(f"  Data dir:        {args.data_dir}")
    print(f"  SAE dir:         {args.sae_dir}")
    print(f"  N families:      {args.n_families}")
    print(f"  Hit threshold:   {args.activation_fraction} × global max")
    print(f"  ESM model:       {args.esm_model}")
    print(f"  ESM layer:       {args.esm_layer}")
    print(f"  Phase 4 only:    {args.phase4_only}")
    print(f"  Workers:         {os.environ.get('PIPELINE_WORKERS', '1')}")
    print(f"  wandb:           {args.wandb}")
    print("=" * 70)

    t0 = time.time()

    if not args.phase4_only:
        # Phase 1: Fetch NMPFams families
        print("\n>>> Phase 1: Fetch NMPFams families")
        families = run_phase1_fetch(args.n_families, args.data_dir, args.seed)
        print(f"    {len(families)} families with consensus sequences")

        # Phase 2: SAE inference
        print("\n>>> Phase 2: SAE inference")
        run_phase2_inference(
            families, args.sae_dir, args.data_dir,
            args.esm_model, args.esm_layer, args.max_seq_len, args.device,
        )

        # Phase 3: Geometry profiles
        print("\n>>> Phase 3: Geometry profiles")
        run_phase3_geometry(families, args.data_dir)

    # Phase 4: Retrain classifiers + score NMPFam proteins
    print("\n>>> Phase 4: Geometry classifier inference")
    run_phase4_geometry_inference(args.data_dir, args.activation_fraction)

    elapsed = time.time() - t0
    wlog({"total_time_seconds": elapsed})
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if args.wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
