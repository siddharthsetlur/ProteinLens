"""Stage 4 — Assemble per-feature JSON files for the visualiser.

Reads the selection (Stage 2), per-residue activations (Stage 3), and
survey statistics (Stage 1) to produce one JSON file per SAE feature
plus two shared files (``sequences.json`` and ``dataset_stats.json``).

Each ``features/NNNN.json`` file contains everything the front-end
needs to render a single feature page: top activating sequences with
per-residue colouring, activation range samples, and coverage stats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from tqdm import tqdm

from proteinlens.analysis.feature_pipeline.collection import _has_pdb
from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.data_acquisition import _parse_fasta


# ===================================================================
# Public API
# ===================================================================


def run_assembly(config: PipelineConfig) -> None:
    """Execute the assembly stage (Stage 4).

    Writes:
    - ``features/NNNN.json`` for each feature (0-padded to 4 digits).
    - ``sequences.json`` — ``{accession: sequence}`` for all proteins
      that appear in any feature file.
    - ``dataset_stats.json`` — summary statistics about the dataset.

    Args:
        config: Pipeline configuration.

    Raises:
        FileNotFoundError: If required upstream outputs are missing.
    """
    # ── Load all upstream data ──
    print("[assembly] Loading upstream outputs ...")
    global_max = np.load(config.feature_max_path)
    num_features = len(global_max)

    with open(config.selection_path, "r") as f:
        selection = json.load(f)

    with open(config.survey_top_path, "r") as f:
        survey_top = json.load(f)

    with open(config.survey_coverage_path, "r") as f:
        survey_coverage = json.load(f)

    _, sequences = _parse_fasta(config.fasta_path)

    # Preload cluster map for coverage stats (if available)
    cluster_map: Dict[str, str] = {}
    if config.cluster_map_path.exists():
        with open(config.cluster_map_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    rep, member = parts
                    cluster_map[member] = rep

    # ── Cache of loaded .npz files ──
    # We keep a small LRU-style cache so that proteins appearing in
    # multiple features don't trigger repeated disk reads.  However,
    # each .npz can be large (seq_len x 5120 x 4 bytes), so we cap it.
    _npz_cache: Dict[str, np.ndarray] = {}
    MAX_NPZ_CACHE = 500

    # Track all accessions that appear in the output (for sequences.json)
    all_referenced_accessions: Set[str] = set()

    # ── Assemble each feature ──
    for feat_idx in tqdm(range(num_features), desc="Assembling features"):
        feat_key = str(feat_idx)
        feat_data = _assemble_single_feature(
            feat_idx=feat_idx,
            feat_max=float(global_max[feat_idx]),
            selection_for_feature=selection["per_feature"].get(feat_key, {}),
            survey_top_for_feature=survey_top.get(feat_key, []),
            coverage_for_feature=survey_coverage.get(feat_key, {}),
            sequences=sequences,
            config=config,
            npz_cache=_npz_cache,
            max_cache=MAX_NPZ_CACHE,
        )

        # Track referenced accessions
        for entry in feat_data.get("top_sequences", []):
            all_referenced_accessions.add(entry["accession"])
        for bin_entries in feat_data.get("activation_bins", {}).values():
            for entry in bin_entries:
                all_referenced_accessions.add(entry["accession"])

        # Write per-feature JSON
        out_path = config.features_dir / f"{feat_idx:04d}.json"
        with open(out_path, "w") as f:
            json.dump(feat_data, f, indent=2)

    print(f"[assembly] Wrote {num_features} feature JSON files to {config.features_dir}/")

    # ── Write sequences.json ──
    referenced_sequences = {
        acc: sequences[acc]
        for acc in sorted(all_referenced_accessions)
        if acc in sequences
    }
    with open(config.sequences_path, "w") as f:
        json.dump(referenced_sequences, f, indent=2)
    print(f"[assembly] Wrote {config.sequences_path} ({len(referenced_sequences)} sequences)")

    # ── Write dataset_stats.json ──
    total_clusters = len(set(cluster_map.values())) if cluster_map else len(sequences)
    stats = {
        "total_proteins": len(sequences),
        "total_clusters": total_clusters,
        "num_features": num_features,
        "organism_taxid": config.organism_taxid,
        "esm_model": config.esm_model_name,
        "esm_layer": config.esm_layer,
        "sae_dir": str(config.sae_dir),
        "activation_threshold": config.activation_threshold,
        "n_top_per_feature": config.n_top_per_feature,
        "n_per_bin": config.n_per_bin,
        "activation_bins": config.activation_bins,
        "unique_proteins_in_output": len(referenced_sequences),
    }
    with open(config.dataset_stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[assembly] Wrote {config.dataset_stats_path}")

    print("[assembly] Stage 4 complete.")


# ===================================================================
# Internal helpers
# ===================================================================


def _assemble_single_feature(
    feat_idx: int,
    feat_max: float,
    selection_for_feature: Dict,
    survey_top_for_feature: List[Dict],
    coverage_for_feature: Dict,
    sequences: Dict[str, str],
    config: PipelineConfig,
    npz_cache: Dict[str, np.ndarray],
    max_cache: int,
) -> Dict[str, Any]:
    """Build the JSON dict for one feature.

    The output follows the schema documented in plan.md:

    .. code-block:: json

        {
            "feature_id": 42,
            "max_activation": 3.7,
            "dataset_coverage": { ... },
            "top_sequences": [ ... ],
            "activation_bins": { ... }
        }

    Args:
        feat_idx: Integer feature index.
        feat_max: Global max activation for this feature.
        selection_for_feature: ``{"top": [...], "bins": {...}}`` from
            selection.json.
        survey_top_for_feature: List of ``{"accession", "max_activation"}``
            dicts from survey_top20.json.
        coverage_for_feature: Coverage stats dict from survey_coverage.json.
        sequences: ``{accession: sequence}`` mapping.
        config: Pipeline configuration.
        npz_cache: Mutable dict used as an activation-array cache.
        max_cache: Max number of entries to keep in the cache.

    Returns:
        Dict matching the per-feature JSON schema.
    """
    # ── Top sequences ──
    top_accessions = selection_for_feature.get("top", [])

    # Build a lookup from the survey top list for max_activation values
    survey_lookup: Dict[str, float] = {
        e["accession"]: e["max_activation"]
        for e in survey_top_for_feature
    }

    top_sequences = []
    for acc in top_accessions:
        entry = _build_protein_entry(
            acc, feat_idx, survey_lookup.get(acc, 0.0),
            sequences, config, npz_cache, max_cache,
        )
        if entry is not None:
            top_sequences.append(entry)

    # Sort by max_activation descending
    top_sequences.sort(key=lambda e: e["max_activation"], reverse=True)

    # ── Activation bins ──
    bins_data: Dict[str, List[Dict]] = {}
    bins_selection = selection_for_feature.get("bins", {})
    for bin_label, bin_accessions in bins_selection.items():
        bin_entries = []
        for acc in bin_accessions:
            entry = _build_protein_entry(
                acc, feat_idx, 0.0,  # max_activation will be recomputed from npz
                sequences, config, npz_cache, max_cache,
            )
            if entry is not None:
                bin_entries.append(entry)
        # Sort within bin by max_activation descending
        bin_entries.sort(key=lambda e: e["max_activation"], reverse=True)
        bins_data[bin_label] = bin_entries

    return {
        "feature_id": feat_idx,
        "max_activation": feat_max,
        "dataset_coverage": coverage_for_feature,
        "top_sequences": top_sequences,
        "activation_bins": bins_data,
    }


def _build_protein_entry(
    accession: str,
    feat_idx: int,
    survey_max_activation: float,
    sequences: Dict[str, str],
    config: PipelineConfig,
    npz_cache: Dict[str, np.ndarray],
    max_cache: int,
) -> Optional[Dict[str, Any]]:
    """Build a protein entry dict for a single feature JSON.

    If per-residue activation data is available (from Stage 3), it is
    included.  Otherwise the entry still has metadata but no per-residue
    array.

    Args:
        accession: UniProt accession.
        feat_idx: Feature index to extract from the activation matrix.
        survey_max_activation: Max activation from the survey (fallback
            if no .npz is available).
        sequences: Accession -> sequence mapping.
        config: Pipeline configuration.
        npz_cache: Mutable activation-array cache.
        max_cache: Max cache entries.

    Returns:
        Dict for this protein within the feature JSON, or ``None`` if
        the protein is not in the sequence map.
    """
    if accession not in sequences:
        return None

    seq = sequences[accession]
    pdb_available = _has_pdb(accession, config.pdb_cache_dir)

    # Try to load per-residue activations
    per_residue: Optional[List[float]] = None
    max_activation = survey_max_activation
    mean_activation = 0.0

    npz_path = config.residue_activations_dir / f"{accession}.npz"
    if npz_path.exists():
        # Load from cache or disk
        if accession not in npz_cache:
            if len(npz_cache) >= max_cache:
                # Evict the oldest entry (simple FIFO — good enough here)
                oldest_key = next(iter(npz_cache))
                del npz_cache[oldest_key]
            npz_cache[accession] = np.load(npz_path)["activations"]

        all_activations = npz_cache[accession]  # shape (seq_len, num_features)

        # Extract the column for this feature
        feat_activations = all_activations[:, feat_idx]  # shape (seq_len,)
        per_residue = feat_activations.tolist()
        max_activation = float(feat_activations.max())
        mean_activation = float(feat_activations.mean())

    entry: Dict[str, Any] = {
        "accession": accession,
        "max_activation": max_activation,
        "mean_activation": round(mean_activation, 4),
        "sequence": seq,
        "sequence_length": len(seq),
        "pdb_available": pdb_available,
    }

    if per_residue is not None:
        # Round to 4 decimal places to keep JSON size manageable
        entry["per_residue_activations"] = [round(v, 4) for v in per_residue]
    else:
        # PM NOTE: If we reach here, it means the .npz was not written
        # for this protein (perhaps collection was interrupted).  We
        # still include the entry but flag the missing data so the
        # front-end can handle it gracefully.
        entry["per_residue_activations"] = None

    return entry
