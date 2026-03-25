"""Stage 1 — Survey pass: stream all proteins through ESM2 -> SAE.

This is the first of two passes over the dataset.  For every protein we:

- Compute the per-residue SAE activations (but do **not** store them —
  that would be ~5120 × seq_len × 20 K proteins ≈ hundreds of GB).
- Track the **per-feature global maximum** activation value (needed to
  define normalised activation bins in Stage 2).
- Maintain a **top-N min-heap** per feature, recording which proteins
  have the highest max-activation for that feature.
- Count **coverage** per feature: how many proteins (and how many
  *clusters*) activate it above the threshold.
- Write a **memmap** file of shape ``(n_proteins, n_features)`` holding
  each protein's per-feature max activation.  This is read during
  Stage 2 (selection) to assign proteins to bins.

The pass is **resumable**: a checkpoint is saved every
``config.survey_checkpoint_every`` proteins.  On restart the already-
processed accessions are skipped and the heaps / counters are rebuilt
from the memmap.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from proteinlens.analysis.feature_pipeline.checkpoint import PipelineState
from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.data_acquisition import _parse_fasta
from proteinlens.embedders.esm import ESM
from proteinlens.sae.inference import load_sae
from proteinlens.utils import get_device


# ===================================================================
# Public API
# ===================================================================


def run_survey(
    config: PipelineConfig,
    state: PipelineState,
    member_to_rep: Dict[str, str],
) -> None:
    """Execute the survey pass (Stage 1).

    Streams every protein through ESM2 -> SAE and records per-feature
    statistics.  Writes four output files:

    - ``feature_max_activations.npy`` — shape ``(num_features,)``, the
      global max activation per feature.
    - ``protein_feature_maxes.npy`` — shape ``(n_proteins, num_features)``,
      a memmap of per-protein per-feature max activations.
    - ``survey_top20.json`` — per-feature top-N proteins by max
      activation.
    - ``survey_coverage.json`` — per-feature protein / cluster counts.

    Args:
        config: Pipeline configuration.
        state: Pipeline state for resumability tracking.
        member_to_rep: Dict mapping each accession to its cluster
            representative (from Stage 0b).  Used for cluster-level
            coverage counting.

    Side effects:
        Updates *state* in-place and on disk.
    """
    # ── Load the FASTA ──
    accessions, sequences = _parse_fasta(config.fasta_path)
    n_proteins = len(accessions)
    print(f"[survey] {n_proteins} proteins in FASTA.")

    # ── Build accession -> row index mapping ──
    # This mapping is fixed for the lifetime of the memmap.
    acc_to_idx: Dict[str, int] = {acc: i for i, acc in enumerate(accessions)}
    state.set_accession_index(acc_to_idx)
    state.set_total_proteins(n_proteins)

    # ── Load models ──
    device = config.device or get_device()
    print(f"[survey] Loading SAE from {config.sae_dir} ...")
    sae = load_sae(config.sae_dir, device=device)
    num_features = sae.dict_size
    print(f"[survey] SAE has {num_features} features.")

    print(f"[survey] Loading ESM model {config.esm_model_name} ...")
    esm = ESM(model_name=config.esm_model_name, device=device)

    # ── Initialise or open the memmap ──
    # The memmap stores per-protein per-feature max activations.
    # Shape: (n_proteins, num_features), dtype float32.
    memmap_path = config.protein_feature_maxes_path
    if memmap_path.exists():
        # Re-open an existing memmap (resume scenario)
        protein_maxes = np.memmap(
            memmap_path, dtype="float32", mode="r+",
            shape=(n_proteins, num_features),
        )
        print(f"[survey] Reopened existing memmap {memmap_path}.")
    else:
        # Create a fresh memmap initialised to zeros
        protein_maxes = np.memmap(
            memmap_path, dtype="float32", mode="w+",
            shape=(n_proteins, num_features),
        )
        print(f"[survey] Created new memmap {memmap_path}.")

    # ── Determine which accessions still need processing ──
    already_done: Set[str] = state.get_survey_processed()
    todo = [acc for acc in accessions if acc not in already_done]
    print(
        f"[survey] {len(already_done)} already processed, "
        f"{len(todo)} remaining."
    )

    # ── Stream proteins one at a time ──
    # PM NOTE: We embed one protein at a time because each protein can
    # have a different length, so batching would require padding and
    # masking which complicates the per-residue max computation.  For
    # throughput on GPU, ESM's internal batch_size=1 is still fast
    # because the bottleneck is the SAE encode (5120 features).
    batch_accessions: List[str] = []

    for i, acc in enumerate(tqdm(todo, desc="Survey pass")):
        seq = sequences[acc]
        row_idx = acc_to_idx[acc]

        # Step 1 — Embed with ESM2
        # embed_single_sequence returns shape (seq_len, embedding_dim)
        embeddings = esm.embed_single_sequence(seq, config.esm_layer)
        embeddings_tensor = torch.tensor(embeddings, device=device)

        # Step 2 — Encode with SAE -> shape (seq_len, num_features)
        # We must apply the SAE's input normalization before encoding,
        # because sae.encode() does NOT normalise internally — only
        # sae.forward() does.  For SAEs trained with normalize_to_sqrt_d=False
        # this is a no-op, but for normalised SAEs skipping this would
        # produce incorrect activations.
        with torch.no_grad():
            normed_input, _ = sae._normalize_input_and_get_norms(embeddings_tensor)
            activations = sae.encode(normed_input)

        # Step 3 — Per-feature max across residues -> shape (num_features,)
        per_feature_max = activations.max(dim=0).values.cpu().numpy()

        # Step 4 — Write to memmap
        protein_maxes[row_idx, :] = per_feature_max

        # Track which accessions are done (for checkpoint)
        batch_accessions.append(acc)

        # ── Periodic checkpoint ──
        if len(batch_accessions) >= config.survey_checkpoint_every:
            # Flush the memmap to disk
            protein_maxes.flush()
            state.add_survey_processed(batch_accessions)
            state.save()
            batch_accessions = []

    # ── Final flush ──
    if batch_accessions:
        protein_maxes.flush()
        state.add_survey_processed(batch_accessions)
        state.save()

    # ── Compute derived outputs from the memmap ──
    print("[survey] Computing global feature maxes and top-N heaps ...")
    _compute_and_save_survey_outputs(
        config=config,
        accessions=accessions,
        protein_maxes=protein_maxes,
        num_features=num_features,
        member_to_rep=member_to_rep,
    )

    state.mark_stage_complete("survey")
    print("[survey] Stage 1 complete.")


# ===================================================================
# Internal helpers
# ===================================================================


def _compute_and_save_survey_outputs(
    config: PipelineConfig,
    accessions: List[str],
    protein_maxes: np.ndarray,
    num_features: int,
    member_to_rep: Dict[str, str],
) -> None:
    """Derive and save the four survey output files from the memmap.

    This is a separate function so it can also be called to rebuild
    outputs from an existing memmap without re-running the full survey.

    The function iterates over *features* (columns of the memmap) rather
    than proteins (rows), which is cache-friendly for column-oriented
    queries and avoids holding all 5120 heaps in memory simultaneously.

    Args:
        config: Pipeline configuration.
        accessions: Ordered list of accession strings (row order matches
            the memmap).
        protein_maxes: The ``(n_proteins, num_features)`` memmap array.
        num_features: Number of SAE features (== memmap column count).
        member_to_rep: Accession -> cluster representative mapping.
    """
    n_proteins = len(accessions)

    # ── 1. Global max per feature ──
    # Take the column-wise max of the memmap.
    global_max = np.max(protein_maxes, axis=0)  # shape (num_features,)
    np.save(config.feature_max_path, global_max)
    print(f"[survey]   Saved {config.feature_max_path}")

    # ── 2. Top-N per feature ──
    # For each feature, find the top-N proteins by max activation using
    # a min-heap of size N.
    n_top = config.n_top_per_feature
    top_per_feature: Dict[str, List[Dict]] = {}

    for feat_idx in tqdm(range(num_features), desc="Top-N heaps"):
        # Extract the column for this feature
        col = protein_maxes[:, feat_idx]

        # Use argpartition for efficient top-N extraction
        # (faster than maintaining a heap for the full column)
        if n_proteins <= n_top:
            top_indices = np.arange(n_proteins)
        else:
            # argpartition puts the top-N values in the last N positions
            # (unsorted among themselves)
            top_indices = np.argpartition(col, -n_top)[-n_top:]

        # Sort the top-N by activation (descending)
        top_indices = top_indices[np.argsort(col[top_indices])[::-1]]

        top_per_feature[str(feat_idx)] = [
            {
                "accession": accessions[idx],
                "max_activation": float(col[idx]),
            }
            for idx in top_indices
            if col[idx] > 0  # exclude proteins with zero activation
        ]

    with open(config.survey_top_path, "w") as f:
        json.dump(top_per_feature, f, indent=2)
    print(f"[survey]   Saved {config.survey_top_path}")

    # ── 3. Coverage statistics per feature ──
    # Count how many proteins and how many *clusters* activate each
    # feature above the threshold.
    threshold = config.activation_threshold
    coverage: Dict[str, Dict] = {}

    for feat_idx in tqdm(range(num_features), desc="Coverage stats"):
        col = protein_maxes[:, feat_idx]

        # Boolean mask: which proteins activate this feature?
        activated_mask = col > threshold
        n_activated = int(activated_mask.sum())

        # Which clusters are represented among the activated proteins?
        activated_clusters: Set[str] = set()
        for prot_idx in np.where(activated_mask)[0]:
            acc = accessions[prot_idx]
            rep = member_to_rep.get(acc, acc)  # fallback to self if not in map
            activated_clusters.add(rep)
        n_clusters_activated = len(activated_clusters)

        total_clusters = len(set(member_to_rep.values())) if member_to_rep else n_proteins

        coverage[str(feat_idx)] = {
            "n_proteins_activated": n_activated,
            "n_clusters_activated": n_clusters_activated,
            "pct_proteins_activated": round(
                100.0 * n_activated / n_proteins, 2
            ) if n_proteins > 0 else 0.0,
            "pct_clusters_activated": round(
                100.0 * n_clusters_activated / total_clusters, 2
            ) if total_clusters > 0 else 0.0,
            "total_proteins": n_proteins,
            "total_clusters": total_clusters,
            "activation_threshold": threshold,
        }

    with open(config.survey_coverage_path, "w") as f:
        json.dump(coverage, f, indent=2)
    print(f"[survey]   Saved {config.survey_coverage_path}")
