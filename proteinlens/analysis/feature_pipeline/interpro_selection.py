"""Stage 5a — InterPro stratified selection and per-residue collection.

Selects a stratified sample of proteins across activation levels for each
SAE feature, then computes per-residue activations for any newly selected
proteins.  The stratified sample is designed for InterPro annotation
enrichment analysis (Stage 5c), where we need proteins at every activation
level — not just the top activators.

**Bin layout (11 bins by default):**

- ``"0.0"`` bin: truly inactive proteins with activation == 0.
  Up to ``interpro_n_per_bin`` are sampled randomly (deterministic seed).
- 10 normalised bins ``[0.0-0.1, 0.1-0.2, ..., 0.9-1.0]``:
  each bin selects up to ``interpro_n_per_bin`` proteins by highest
  activation within the bin.

This gives a balanced sample across the full activation range, which is
critical for computing meaningful F1 scores in Stage 5c.

**Per-residue collection:** After selection, any protein that does not
already have a ``.npz`` file (in either ``residue_activations/`` from
Stage 3 or ``interpro_residue_activations/``) is re-embedded through
ESM2 -> SAE to produce per-residue activations.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from proteinlens.analysis.feature_pipeline.config import PipelineConfig


# ===================================================================
# Public API
# ===================================================================


def run_interpro_selection(config: PipelineConfig) -> Dict:
    """Execute the InterPro selection stage (Stage 5a).

    Reads the survey memmap and global max activations, assigns proteins
    to 11 stratified activation bins per feature, writes the selection
    JSON, then computes per-residue activations for any newly selected
    proteins that don't already have .npz files.

    Args:
        config: Pipeline configuration.  Requires that Stage 1 (survey)
            has completed, providing ``protein_feature_maxes.npy``,
            ``feature_max_activations.npy``, and ``pipeline_state.json``.

    Returns:
        The selection dict with keys ``"per_feature"`` and
        ``"all_selected_accessions"``.

    Raises:
        FileNotFoundError: If required survey outputs are missing.
    """
    # ── Load inputs from Stage 1 ──
    global_max = np.load(config.feature_max_path)  # (num_features,)
    num_features = len(global_max)

    with open(config.pipeline_state_path, "r") as f:
        pipeline_state = json.load(f)
    n_proteins = pipeline_state["total_proteins"]
    acc_index = pipeline_state["accession_index"]  # acc -> row_idx

    # Invert index: row_idx -> accession
    idx_to_acc = {int(v): k for k, v in acc_index.items()}

    protein_maxes = np.memmap(
        config.protein_feature_maxes_path,
        dtype="float32",
        mode="r",
        shape=(n_proteins, num_features),
    )

    # ── Build the 11 bins ──
    # Bin 0: "0.0" — truly inactive (activation == 0)
    # Bins 1-10: normalised ranges [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
    normalised_bin_edges = [i / 10.0 for i in range(11)]  # [0.0, 0.1, ..., 1.0]
    normalised_bin_ranges = [
        (normalised_bin_edges[i], normalised_bin_edges[i + 1])
        for i in range(10)
    ]

    rng = np.random.default_rng(seed=42)

    all_selected = set()
    per_feature: Dict[str, Dict] = {}

    for feat_idx in tqdm(range(num_features), desc="InterPro selection"):
        feat_max = float(global_max[feat_idx])
        bins_result: Dict[str, List[str]] = {}

        col = protein_maxes[:, feat_idx]

        # --- "0.0" bin: truly inactive proteins (activation == 0) ---
        zero_mask = col == 0
        zero_indices = np.where(zero_mask)[0]

        if len(zero_indices) <= config.interpro_n_per_bin:
            selected_zero = zero_indices
        else:
            # Deterministic random sample from inactive proteins
            selected_zero = rng.choice(
                zero_indices, size=config.interpro_n_per_bin, replace=False
            )

        bins_result["0.0"] = [idx_to_acc[int(i)] for i in selected_zero]

        # --- 10 normalised bins ---
        if feat_max > 0:
            for low_frac, high_frac in normalised_bin_ranges:
                low_abs = low_frac * feat_max
                high_abs = high_frac * feat_max
                bin_label = f"{low_frac}-{high_frac}"

                # All bins use (low_abs, high_abs] — half-open on the left,
                # closed on the right.  The lowest non-zero bin uses col > 0
                # instead of col > low_abs to exclude truly inactive proteins
                # (which belong in the "0.0" bin).
                if low_frac == 0.0:
                    mask = (col > 0) & (col <= high_abs)
                else:
                    mask = (col > low_abs) & (col <= high_abs)

                candidate_indices = np.where(mask)[0]

                if len(candidate_indices) == 0:
                    bins_result[bin_label] = []
                    continue

                # Select top-N within this bin by activation value
                if len(candidate_indices) <= config.interpro_n_per_bin:
                    selected_indices = candidate_indices
                else:
                    bin_vals = col[candidate_indices]
                    top_in_bin = np.argpartition(
                        bin_vals, -config.interpro_n_per_bin
                    )[-config.interpro_n_per_bin:]
                    selected_indices = candidate_indices[top_in_bin]

                bins_result[bin_label] = [
                    idx_to_acc[int(i)] for i in selected_indices
                ]
        else:
            # Feature never fires — all normalised bins empty
            for low_frac, high_frac in normalised_bin_ranges:
                bins_result[f"{low_frac}-{high_frac}"] = []

        # Accumulate all selected accessions
        for accs in bins_result.values():
            all_selected.update(accs)

        per_feature[str(feat_idx)] = {"bins": bins_result}

    # ── Write selection JSON ──
    selection = {
        "per_feature": per_feature,
        "all_selected_accessions": sorted(all_selected),
    }
    with open(config.interpro_selection_path, "w") as f:
        json.dump(selection, f, indent=2)

    print(
        f"[interpro_selection] Selected {len(all_selected)} unique proteins "
        f"across {num_features} features. "
        f"Saved to {config.interpro_selection_path}."
    )
    from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog

    wlog({
        "interpro_selection/unique_proteins": len(all_selected),
        "interpro_selection/num_features": num_features,
    })

    # ── 2.2: Per-residue activation collection for new proteins ──
    _collect_missing_residue_activations(config, all_selected)

    return selection


# ===================================================================
# Internal helpers
# ===================================================================


def _collect_missing_residue_activations(
    config: PipelineConfig,
    all_selected: set,
) -> None:
    """Compute per-residue activations for newly selected proteins.

    Checks both ``config.residue_activations_dir`` (Stage 3 output) and
    ``config.interpro_residue_activations_dir`` for existing .npz files.
    Only proteins missing from both directories are re-embedded.

    Args:
        config: Pipeline configuration.
        all_selected: Set of all accession strings selected for InterPro
            enrichment.
    """
    # Determine which proteins already have .npz files
    already_cached = set()
    for acc in all_selected:
        if (config.residue_activations_dir / f"{acc}.npz").exists():
            already_cached.add(acc)
        elif (config.interpro_residue_activations_dir / f"{acc}.npz").exists():
            already_cached.add(acc)

    todo = sorted(all_selected - already_cached)

    print(
        f"[interpro_selection] Computing per-residue activations for "
        f"{len(todo)} new proteins ({len(already_cached)} already cached)"
    )

    if not todo:
        return

    # Lazy imports: avoid loading heavy ML models unless needed
    from proteinlens.analysis.feature_pipeline.collection import (
        _compute_residue_activations,
    )
    from proteinlens.analysis.feature_pipeline.data_acquisition import _parse_fasta
    from proteinlens.embedders.esm import ESM
    from proteinlens.sae.inference import load_sae
    from proteinlens.utils import get_device

    device = config.device or get_device()
    print(f"[interpro_selection] Loading SAE from {config.sae_dir} ...")
    sae = load_sae(config.sae_dir, device=device)
    print(f"[interpro_selection] Loading ESM model {config.esm_model_name} ...")
    esm_model = ESM(model_name=config.esm_model_name, device=device)

    _, sequences = _parse_fasta(config.fasta_path)

    n_saved = 0
    n_skipped = 0
    for acc in tqdm(todo, desc="Computing InterPro residue activations"):
        if acc not in sequences:
            # PM NOTE: This mirrors the warning pattern in collection.py.
            # Should not happen if pipeline stages ran in order.
            print(
                f"[interpro_selection] WARNING: {acc} not in FASTA — skipping."
            )
            n_skipped += 1
            continue

        seq = sequences[acc]
        activations = _compute_residue_activations(
            esm_model, sae, seq, config.esm_layer, device
        )
        npz_path = config.interpro_residue_activations_dir / f"{acc}.npz"
        np.savez_compressed(npz_path, activations=activations)
        n_saved += 1

    print(
        f"[interpro_selection] Saved {n_saved} new .npz files, "
        f"skipped {n_skipped}."
    )
    from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog

    wlog({
        "interpro_selection/npz_saved": n_saved,
        "interpro_selection/npz_skipped": n_skipped,
    })
