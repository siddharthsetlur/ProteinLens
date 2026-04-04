"""Stage 8 — Sequence position F1 enrichment per SAE feature.

For each SAE feature, tests whether a residue's position within its
protein sequence predicts high feature activation at that position.

**Biological motivation:**
Many protein features are position-dependent: signal peptides occupy the
first ~20 residues, C-terminal retention signals sit at the end, and
domain boundaries often occur at consistent relative positions.  This
stage discovers such positional biases in SAE feature activations.

**Approach:**
We define 21 binary position predicates (e.g. ``first_10``,
``pct_0_10``, ``third_N``) that map ``(residue_position,
sequence_length)`` to True/False.  For each predicate we reuse the
same vectorised F1 threshold-sweep from Stage 7 (motif enrichment):
sweep 50 evenly-spaced activation thresholds and report the threshold
that maximises F1.

**Predicate categories:**

- *Absolute N-terminal* (``first_5/10/20``): signal peptides,
  methionine-adjacent effects.
- *Absolute C-terminal* (``last_5/10/20``): C-terminal sorting/retention
  signals.
- *Relative decile bins* (``pct_0_10`` through ``pct_90_100``): position-
  dependent features at any relative location, normalised by sequence
  length.
- *Relative thirds* (``third_N/M/C``): coarse N-terminal / middle /
  C-terminal preference.
- *Termini vs interior* (``terminal_10pct``, ``interior_80pct``): edge vs
  core distinction.
- *Absolute middle* (``mid_20pct``): central region of the sequence.

**Outputs:**
- ``position_enrichment/{feat_idx:04d}.json`` — per-feature enrichment
  with top position predicates, their F1, precision, recall, and counts.
- ``position_enrichment/summary.json`` — quick-lookup summary keyed by
  feature id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.motif_enrichment import (
    _compute_best_motif_f1,
    _pool_proteins_for_feature,
)

# ===================================================================
# Position predicates (checklist 8.2.1)
# ===================================================================

POSITION_PREDICATES: Dict[str, Callable[[int, int], bool]] = {
    # Absolute N-terminal
    "first_5": lambda pos, _: pos < 5,
    "first_10": lambda pos, _: pos < 10,
    "first_20": lambda pos, _: pos < 20,
    # Absolute C-terminal
    "last_5": lambda pos, slen: pos >= slen - 5,
    "last_10": lambda pos, slen: pos >= slen - 10,
    "last_20": lambda pos, slen: pos >= slen - 20,
    # Relative decile bins (normalised position)
    "pct_0_10": lambda pos, slen: pos / slen < 0.1,
    "pct_10_20": lambda pos, slen: 0.1 <= pos / slen < 0.2,
    "pct_20_30": lambda pos, slen: 0.2 <= pos / slen < 0.3,
    "pct_30_40": lambda pos, slen: 0.3 <= pos / slen < 0.4,
    "pct_40_50": lambda pos, slen: 0.4 <= pos / slen < 0.5,
    "pct_50_60": lambda pos, slen: 0.5 <= pos / slen < 0.6,
    "pct_60_70": lambda pos, slen: 0.6 <= pos / slen < 0.7,
    "pct_70_80": lambda pos, slen: 0.7 <= pos / slen < 0.8,
    "pct_80_90": lambda pos, slen: 0.8 <= pos / slen < 0.9,
    "pct_90_100": lambda pos, slen: pos / slen >= 0.9,
    # Relative thirds
    "third_N": lambda pos, slen: pos / slen < 1 / 3,
    "third_M": lambda pos, slen: 1 / 3 <= pos / slen < 2 / 3,
    "third_C": lambda pos, slen: pos / slen >= 2 / 3,
    # Termini vs interior
    "terminal_10pct": lambda pos, slen: pos / slen < 0.1 or pos / slen >= 0.9,
    "interior_80pct": lambda pos, slen: 0.1 <= pos / slen < 0.9,
    # Absolute middle
    "mid_20pct": lambda pos, slen: 0.4 <= pos / slen < 0.6,
}


# ===================================================================
# Build predicate indices (checklist 8.2.2)
# ===================================================================


def _build_predicate_indices(
    seq_lengths: List[int],
    total_residues: int,
) -> Dict[str, np.ndarray]:
    """Map each position predicate to the global residue indices where it is True.

    Args:
        seq_lengths: Sequence lengths for each protein, in pooling order.
            The sum must equal *total_residues*.
        total_residues: Total number of residues across all proteins.

    Returns:
        Dict mapping predicate name to a 1-D int array of global indices
        where that predicate evaluates to True.
    """
    # Pre-compute local positions and sequence lengths for every residue
    local_positions = np.empty(total_residues, dtype=np.int64)
    lengths = np.empty(total_residues, dtype=np.int64)
    offset = 0
    for slen in seq_lengths:
        local_positions[offset : offset + slen] = np.arange(slen)
        lengths[offset : offset + slen] = slen
        offset += slen

    result: Dict[str, np.ndarray] = {}
    for name, pred_fn in POSITION_PREDICATES.items():
        # Vectorise by evaluating the predicate on all residues
        mask = np.array(
            [pred_fn(int(local_positions[i]), int(lengths[i]))
             for i in range(total_residues)],
            dtype=bool,
        )
        indices = np.where(mask)[0]
        if len(indices) > 0:
            result[name] = indices

    return result


# ===================================================================
# Per-feature analysis (checklist 8.2.3)
# ===================================================================


def _analyze_feature(
    feature_data: Dict[str, Any],
    feat_max: float,
    config: PipelineConfig,
) -> Optional[Dict[str, Any]]:
    """Run the full position enrichment analysis for a single feature.

    Steps:
        1. Pool proteins from ``top_sequences`` and ``activation_bins``.
        2. Extract per-residue activations and sequence lengths.
        3. Build position predicate index arrays.
        4. Run the vectorised F1 sweep (reused from motif enrichment).

    Args:
        feature_data: Parsed per-feature JSON from Stage 4.
        feat_max: Global maximum activation for this feature.
        config: Pipeline configuration.

    Returns:
        Result dict or ``None`` if there are no proteins.
    """
    # Step 1: pool proteins
    proteins = _pool_proteins_for_feature(feature_data)
    if not proteins:
        return None

    # Step 2: extract activations and sequence lengths
    all_acts: List[float] = []
    seq_lengths: List[int] = []
    for _acc, _seq, pra in proteins:
        all_acts.extend(pra)
        seq_lengths.append(len(pra))

    if not all_acts:
        return None

    all_activations = np.array(all_acts, dtype=np.float64)
    total_residues = len(all_activations)

    # Step 3: build predicate indices
    predicate_indices = _build_predicate_indices(seq_lengths, total_residues)

    if not predicate_indices:
        return None

    # Step 4: vectorised F1 (reuse motif enrichment function)
    top_positions = _compute_best_motif_f1(
        predicate_indices,
        all_activations,
        feat_max,
        n_steps=config.position_f1_threshold_steps,
        min_count=config.position_min_count,
        top_n=config.position_top_n,
    )

    if not top_positions:
        return None

    # Rename "motif" key to "position" in results for clarity
    for entry in top_positions:
        entry["position"] = entry.pop("motif")
        entry["interpretation"] = entry["interpretation"].replace("Motif ", "Position predicate ")

    return {
        "feature_id": feature_data["feature_id"],
        "feature_max_activation": round(float(feat_max), 6),
        "n_proteins_evaluated": len(proteins),
        "n_total_residues": total_residues,
        "n_predicates_tested": sum(
            1 for idxs in predicate_indices.values()
            if len(idxs) >= config.position_min_count
        ),
        "top_positions": top_positions,
    }


# ===================================================================
# Public API (checklist 8.2.4)
# ===================================================================


def run_position_enrichment(config: PipelineConfig) -> None:
    """Execute the sequence position F1 enrichment stage (Stage 8).

    For each SAE feature, loads its assembled JSON (top sequences and
    activation bins), computes position predicate indices for every
    residue, and finds the predicate(s) whose membership best predicts
    high activation via an F1-maximising threshold sweep.

    **Resumability:** Features whose output JSON already exists in
    ``config.position_enrichment_dir`` are skipped.

    **Outputs:**
        - ``position_enrichment/{feat_idx:04d}.json`` — per-feature results.
        - ``position_enrichment/summary.json`` — keyed by feature id, with
          ``best_position``, ``best_position_f1``, and ``n_predicates_tested``.

    Args:
        config: Pipeline configuration.  Requires that Stage 4 (assembly)
            has completed.

    Raises:
        FileNotFoundError: If ``feature_max_activations.npy`` is missing.
    """
    global_max = np.load(config.feature_max_path)
    num_features = len(global_max)

    out_dir = config.position_enrichment_dir
    features_dir = config.features_dir

    n_analyzed = 0
    n_skipped = 0
    summary_features: Dict[str, Dict[str, Any]] = {}

    for feat_idx in tqdm(range(num_features), desc="[position_enrichment]"):
        feat_max = float(global_max[feat_idx])

        if feat_max == 0:
            n_skipped += 1
            continue

        out_path = out_dir / f"{feat_idx:04d}.json"
        if out_path.exists():
            try:
                with open(out_path) as f:
                    existing = json.load(f)
                if existing.get("top_positions"):
                    fid_str = str(feat_idx)
                    summary_features[fid_str] = {
                        "best_position": existing["top_positions"][0]["position"],
                        "best_position_f1": existing["top_positions"][0]["best_f1"],
                        "n_predicates_tested": existing.get("n_predicates_tested", 0),
                    }
                n_analyzed += 1
            except (json.JSONDecodeError, KeyError):
                pass
            continue

        feat_path = features_dir / f"{feat_idx:04d}.json"
        if not feat_path.exists():
            n_skipped += 1
            continue

        with open(feat_path) as f:
            feature_data = json.load(f)

        result = _analyze_feature(feature_data, feat_max, config)

        if result is None:
            n_skipped += 1
            continue

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        fid_str = str(feat_idx)
        if result["top_positions"]:
            summary_features[fid_str] = {
                "best_position": result["top_positions"][0]["position"],
                "best_position_f1": result["top_positions"][0]["best_f1"],
                "n_predicates_tested": result["n_predicates_tested"],
            }

        n_analyzed += 1

    summary = {
        "n_features_analyzed": n_analyzed,
        "n_features_skipped": n_skipped,
        "n_predicates": len(POSITION_PREDICATES),
        "features": summary_features,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[position_enrichment] Analyzed {n_analyzed} features, "
        f"skipped {n_skipped}. "
        f"Wrote results to {out_dir}/"
    )

    from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog

    wlog({
        "position_enrichment/analyzed": n_analyzed,
        "position_enrichment/skipped": n_skipped,
    })
