"""Stage 2 — Selection: decide which proteins need per-residue data.

With the global max activation per feature now known (from Stage 1), we
can define **normalised activation bins** and select a sample of proteins
from each bin.  The union of all selected proteins across all 5120
features is then passed to Stage 3 for full per-residue collection.

The bin edges are expressed as fractions of each feature's global max.
For example, with the default bin edges ``[0.0, 0.25, 0.5, 0.75, 1.0]``
and a feature whose global max is 4.0, the four bins are:

- ``[0.0, 1.0)``  — low activation
- ``[1.0, 2.0)``  — medium-low
- ``[2.0, 3.0)``  — medium-high
- ``[3.0, 4.0]``  — high activation

From each bin we select up to ``config.n_per_bin`` proteins (sampled
by highest activation within the bin).  Combined with the top-N from
the survey, each feature gets at most ``n_top + 4 * n_per_bin`` = 60
proteins (with default settings), but heavy overlap across features
means the actual unique set is much smaller.
"""

from __future__ import annotations

import json
from typing import Dict, List, Set

import numpy as np
from tqdm import tqdm

from proteinlens.analysis.feature_pipeline.config import PipelineConfig


# ===================================================================
# Public API
# ===================================================================


def run_selection(config: PipelineConfig) -> Dict:
    """Execute the selection stage (Stage 2).

    Reads the survey outputs (memmap + top-N JSON + global maxes) and
    determines which proteins require per-residue activation collection.

    Outputs are written to ``config.selection_path`` as JSON with the
    structure::

        {
            "per_feature": {
                "0": {
                    "top": ["P12345", "Q67890", ...],
                    "bins": {
                        "0.75-1.0": ["A11111", ...],
                        "0.5-0.75": [...],
                        ...
                    }
                },
                ...
            },
            "all_selected_accessions": ["P12345", "Q67890", ...]
        }

    Args:
        config: Pipeline configuration.

    Returns:
        The selection dict (same structure as the JSON).

    Raises:
        FileNotFoundError: If required survey outputs are missing.
    """
    # ── Load inputs from Stage 1 ──
    global_max = np.load(config.feature_max_path)          # (num_features,)
    num_features = len(global_max)

    with open(config.survey_top_path, "r") as f:
        survey_top = json.load(f)

    # Load the memmap of per-protein per-feature max activations
    # We need the pipeline_state to know n_proteins for the memmap shape
    with open(config.pipeline_state_path, "r") as f:
        pipeline_state = json.load(f)
    n_proteins = pipeline_state["total_proteins"]
    acc_index = pipeline_state["accession_index"]  # acc -> row_idx

    # Invert the index: row_idx -> accession
    idx_to_acc = {int(v): k for k, v in acc_index.items()}

    protein_maxes = np.memmap(
        config.protein_feature_maxes_path,
        dtype="float32",
        mode="r",
        shape=(n_proteins, num_features),
    )

    # ── Parse bin edges from config ──
    # config.activation_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    # We form bins as consecutive pairs: (0.0, 0.25), (0.25, 0.5), ...
    bin_edges = config.activation_bins
    bin_ranges = [
        (bin_edges[i], bin_edges[i + 1])
        for i in range(len(bin_edges) - 1)
    ]

    # ── Select proteins for each feature ──
    all_selected: Set[str] = set()
    per_feature: Dict[str, Dict] = {}

    for feat_idx in tqdm(range(num_features), desc="Selection"):
        feat_max = global_max[feat_idx]

        # --- Top-N from survey ---
        top_entries = survey_top.get(str(feat_idx), [])
        top_accessions = [e["accession"] for e in top_entries]

        # --- Bin sampling ---
        # Skip features that never activate (max == 0)
        bins_result: Dict[str, List[str]] = {}
        if feat_max > 0:
            col = protein_maxes[:, feat_idx]

            for low_frac, high_frac in bin_ranges:
                # Convert normalised fractions to absolute thresholds
                low_abs = low_frac * feat_max
                high_abs = high_frac * feat_max
                bin_label = f"{low_frac}-{high_frac}"

                # Find proteins whose max activation falls in this bin.
                # Lower bound is exclusive (except for the lowest bin
                # which uses > 0 to exclude zero-activation proteins).
                # Upper bound is inclusive for the top bin, exclusive
                # otherwise.
                if low_frac == 0.0:
                    # Lowest bin: (0, high_abs]  — exclude exactly-zero
                    mask = (col > 0) & (col <= high_abs)
                elif high_frac == 1.0:
                    # Highest bin: (low_abs, high_abs]  — inclusive upper
                    mask = (col > low_abs) & (col <= high_abs)
                else:
                    # Middle bins: (low_abs, high_abs]
                    mask = (col > low_abs) & (col <= high_abs)

                candidate_indices = np.where(mask)[0]

                if len(candidate_indices) == 0:
                    bins_result[bin_label] = []
                    continue

                # Select top-N within this bin (by activation value)
                if len(candidate_indices) <= config.n_per_bin:
                    selected_indices = candidate_indices
                else:
                    # Pick the top-N highest within the bin
                    bin_vals = col[candidate_indices]
                    top_in_bin = np.argpartition(bin_vals, -config.n_per_bin)[
                        -config.n_per_bin:
                    ]
                    selected_indices = candidate_indices[top_in_bin]

                bins_result[bin_label] = [
                    idx_to_acc[int(idx)] for idx in selected_indices
                ]
        else:
            # Feature never fires — all bins empty
            for low_frac, high_frac in bin_ranges:
                bins_result[f"{low_frac}-{high_frac}"] = []

        # Accumulate all selected accessions
        for acc in top_accessions:
            all_selected.add(acc)
        for accs in bins_result.values():
            for acc in accs:
                all_selected.add(acc)

        per_feature[str(feat_idx)] = {
            "top": top_accessions,
            "bins": bins_result,
        }

    # ── Save selection results ──
    selection = {
        "per_feature": per_feature,
        "all_selected_accessions": sorted(all_selected),
    }
    with open(config.selection_path, "w") as f:
        json.dump(selection, f, indent=2)

    print(
        f"[selection] Selected {len(all_selected)} unique proteins across "
        f"{num_features} features. Saved to {config.selection_path}."
    )
    from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog

    wlog({
        "selection/unique_proteins": len(all_selected),
        "selection/num_features": num_features,
    })
    return selection
