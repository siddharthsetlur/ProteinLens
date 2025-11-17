#!/usr/bin/env python
"""Analyze SAE features: find max activating proteins and compute statistics.

This script analyzes a trained SAE by:
1. Finding the top activating proteins for each feature
2. Computing feature statistics (frequency, activation percentages)
3. Saving results for later use (e.g., in dashboard creation)

Run this after training an SAE and before creating a dashboard.
"""

from pathlib import Path
from typing import List, Optional
import yaml
import torch

from proteinlens.sae.inference import load_sae
from proteinlens.analysis.per_protein_tracking import find_max_examples_per_feat
from proteinlens.utils import get_device


def collect_feature_activations(
    sae_dir: Path,
    embeddings_dir: Path,
    metadata_dir: Path,
    output_dir: Optional[Path] = None,
    shards: Optional[List[int]] = None,
    shard_range: Optional[List[int]] = None,
    activation_threshold: float = 0.05,
):
    """
    Analyze SAE features and find max activating proteins.

    Args:
        sae_dir: Directory containing trained SAE model (e.g., models/walkthrough_model/layer_4)
        embeddings_dir: Directory containing embeddings (e.g., data/analysis_embeddings/esm2_8m/layer_4)
        metadata_dir: Directory containing protein metadata (e.g., data/annotations/uniprotkb/processed)
        output_dir: Output directory for results (default: same as sae_dir)
        shards: Shard indices to search (e.g., [0, 1, 2, 3]). Use shard_range instead for ranges.
        shard_range: Shard range [start, end] (inclusive) to search (e.g., [0, 7] for shards 0-7)
        activation_threshold: Minimum activation value to count as 'activated' (default: 0.05)
    """
    # Handle shard arguments
    if shards is not None and shard_range is not None:
        raise ValueError("Cannot specify both shards and shard_range")
    elif shard_range is not None:
        shards = list(range(shard_range[0], shard_range[1] + 1))
    elif shards is None:
        shards = [0]  # Default to shard 0

    # Set output directory
    if output_dir is None:
        output_dir = sae_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SAE Feature Analysis")
    print("=" * 70)
    print(f"SAE directory: {sae_dir}")
    print(f"Embeddings: {embeddings_dir}")
    print(f"Metadata: {metadata_dir}")
    print(f"Output: {output_dir}")
    print(f"Analyzing shards: {shards}")
    print()

    # Load SAE (auto-detects architecture from config)
    device = get_device()
    print(f"Loading SAE on device: {device}")
    sae = load_sae(sae_dir, device=device)
    print(f"SAE loaded: {sae.__class__.__name__} with {sae.dict_size} features, {sae.activation_dim}D embeddings")
    print()

    # Find max activating proteins for each feature
    print("Finding max activating proteins for each feature...")
    print("(This may take several minutes depending on dataset size)")
    print(f"Using activation threshold: {activation_threshold}")
    per_protein_tracker = find_max_examples_per_feat(
        sae=sae,
        aa_embeds_dir=embeddings_dir,
        aa_metadata_dir=metadata_dir,
        shards_to_search=shards,
        activation_threshold=activation_threshold,
        # Uses default loaders: load_shard_embeddings() for embeddings
        # and _get_protein_ids() for metadata
    )
    print("✓ Max activating proteins found")
    print()

    # Save results
    print("Saving results...")

    # 1. Save max activation per feature (for SAE normalization)
    max_activations = per_protein_tracker["max_activation_per_feature"]
    output_path = output_dir / "max_activations_per_feature.pt"
    torch.save(torch.tensor(max_activations), output_path)
    print(f"✓ Max activations saved to: {output_path}")

    # 2. Save feature statistics
    per_feature_statistics = {
        "Per_prot_frequency_of_any_activation": per_protein_tracker[
            "pct_proteins_with_activation"
        ],
        "Per_prot_pct_activated_when_present": per_protein_tracker[
            "avg_pct_activated_when_present"
        ],
    }
    output_path = output_dir / "Per_feature_statistics.yaml"
    with open(output_path, "w") as f:
        yaml.dump(per_feature_statistics, f)
    print(f"✓ Feature statistics saved to: {output_path}")

    # 3. Save max activating examples
    max_examples = per_protein_tracker["max"]
    output_path = output_dir / "Per_feature_max_examples.yaml"
    with open(output_path, "w") as f:
        yaml.dump(max_examples, f)
    print(f"✓ Max examples saved to: {output_path}")

    # 4. Save lower quantile examples
    quantile_examples = per_protein_tracker["lower_quantile"]
    output_path = output_dir / "Per_feature_quantile_examples.yaml"
    with open(output_path, "w") as f:
        yaml.dump(quantile_examples, f)
    print(f"✓ Quantile examples saved to: {output_path}")

    print()
    print("=" * 70)
    print("✅ Feature analysis complete!")
    print("=" * 70)
    print()
    print(f"Results saved to: {output_dir}")
    print()
    print("Summary statistics:")
    print(f"  Total features: {sae.dict_size}")
    print(f"  Features with any activation: {sum(1 for x in per_feature_statistics['Per_prot_frequency_of_any_activation'] if x > 0)}")
    print(f"  Dead features (no activation): {sum(1 for x in per_feature_statistics['Per_prot_frequency_of_any_activation'] if x == 0)}")
    print()


if __name__ == "__main__":
    from tap import tapify
    tapify(collect_feature_activations)
