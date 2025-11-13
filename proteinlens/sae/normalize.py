"""
Normalize SAE model features based on maximum activation values, adjusting the model weights
to maintain the same reconstructions while ensuring that the maximum activation value for each
feature is 1 across the provided dataset.
"""

from pathlib import Path
from typing import Optional

from interplm.sae.dictionary import MatryoshkaBatchTopKSAE
import numpy as np
import torch

from interplm.sae.inference import (
    get_sae_feats_in_batches,
    load_sae,
    split_up_feature_list,
)
from interplm.utils import get_device
from interplm.data_processing.embedding_loader import (
    ShardDataLoader,
    detect_and_create_loader
)

def calculate_feature_statistics(
    sae: torch.nn.Module,
    data_loader: ShardDataLoader,
    n_shards: Optional[int] = None,
    max_features_per_chunk: int = 640,
    max_tokens_per_chunk: int = 25_000,
) -> torch.Tensor:
    """
    Calculate maximum activation value for each SAE feature across all data.

    Args:
        sae: Sparse autoencoder model
        data_loader: ShardDataLoader instance for loading data
        n_shards: Number of data shards to process (None = process all)
        max_features_per_chunk: Maximum features to process at once
        max_tokens_per_chunk: Maximum tokens to process in one batch

    Returns:
        Tensor containing maximum activation value for each feature
    """
    device = get_device()
    num_features = sae.dict_size
    max_per_feat = torch.zeros(num_features, device=device)

    # Get the shard indices to process from the dataloader
    shard_indices = data_loader.get_shard_indices(n_shards)
    
    # binary mask of features that have been non-zero at least once
    is_feature_ever_nonzero = torch.zeros(num_features, device=device)

    for i, shard_idx in enumerate(shard_indices):
        print(f"Processing shard {shard_idx} ({i+1}/{len(shard_indices)})")
        
        # Load embeddings for current shard
        esm_acts = data_loader.load_shard(shard_idx)

        n_features_processed = 0
        n_features_with_max_value_of_0 = 0
        
        # Process features in chunks to manage memory
        for feature_list in split_up_feature_list(
            total_features=num_features, max_feature_chunk_size=max_features_per_chunk
        ):
            # Get SAE features for current chunk
            sae_feats = get_sae_feats_in_batches(
                sae=sae,
                device=device,
                aa_embds=esm_acts,
                chunk_size=max_tokens_per_chunk,
                feat_list=feature_list,
            )

            # print the number of features that have max value of 0
            n_features_with_max_value_of_0 += sum(torch.max(sae_feats, dim=0)[0] == 0)
            n_features_processed += len(feature_list)

            is_feature_ever_nonzero[feature_list] = torch.logical_or(
                is_feature_ever_nonzero[feature_list], (sae_feats > 0).any(dim=0)
            ).to(torch.float)

            # Update maximum values for current feature subset
            max_per_feat[feature_list] = torch.max(
                max_per_feat[feature_list], torch.max(sae_feats, dim=0)[0]
            )

            # Clean up to manage memory
            del sae_feats
            torch.cuda.empty_cache()

        print(
            f"  Shard {shard_idx}: {n_features_with_max_value_of_0}/{n_features_processed} "
            f"features with max value of 0"
        )

    # Final statistics
    print(f"\nFinal statistics:")
    print(f"  Features with max value of 0: {sum(max_per_feat == 0)}")
    print(f"  Features that were non-zero at least once: {sum(is_feature_ever_nonzero)}")

    return max_per_feat


def create_normalized_model(
    sae: torch.nn.Module, max_per_feat: torch.Tensor
) -> torch.nn.Module:
    """
    Create a normalized version of the SAE model based on maximum feature values.
    
    This stores normalization factors in the model's activation_rescale_factor buffer
    rather than modifying weights directly. This ensures consistency across all SAE types
    and allows the normalization to be applied post-ReLU during inference.

    Args:
        sae: Original SAE model
        max_per_feat: Maximum activation values per feature

    Returns:
        SAE model with normalization factors stored
    """
    # Store normalization factors in the model's buffer
    # All Dictionary subclasses have this buffer initialized to ones
    if hasattr(sae, 'activation_rescale_factor'):
        sae.activation_rescale_factor = max_per_feat
    else:
        # Register the buffer if it doesn't exist (shouldn't happen with proper Dictionary subclass)
        sae.register_buffer("activation_rescale_factor", max_per_feat)
    
    return sae


def normalize_sae_features(
    sae_dir: Path, 
    aa_embds_dir: Path, 
    n_shards: Optional[int] = None,
    data_loader: Optional[ShardDataLoader] = None,
    loader_type: Optional[str] = None,
    nested_filename: Optional[str] = None,
) -> None:
    """
    Calculate feature statistics and create a normalized version of the SAE model.

    Args:
        sae_dir: Directory containing SAE model
        aa_embds_dir: Directory containing ESM embeddings
        n_shards: Number of data shards to process (None = all)
        data_loader: Optional pre-configured ShardDataLoader instance
        loader_type: Optional loader type ("flat" or "nested") to force specific loader
        nested_filename: Optional filename for nested loader (default: "activations.pt")
    """
    # Setup paths
    feat_stat_cache = sae_dir / "feature_stats"
    norm_sae_path = sae_dir / f"ae_normalized.pt"

    # Create cache directory
    feat_stat_cache.mkdir(parents=True, exist_ok=True)

    # Create or configure data loader
    if data_loader is None:
        # Auto-detect is the only option now since loader classes are in embedding_loader module
        data_loader = detect_and_create_loader(aa_embds_dir)

    # Load model and calculate statistics
    print("Loading SAE model and calculating feature statistics...")
    sae = load_sae(sae_dir)
    max_per_feat = calculate_feature_statistics(
        sae=sae, 
        data_loader=data_loader, 
        n_shards=n_shards
    )

    # Save statistics
    np.save(feat_stat_cache / "max.npy", max_per_feat.cpu().numpy())

    print("\nCreating normalized SAE model...")
    # All models now use the same normalization approach
    sae_normalized = create_normalized_model(sae, max_per_feat)

    torch.save(sae_normalized.state_dict(), norm_sae_path)
    print(f"Normalized model saved to {norm_sae_path}")


if __name__ == "__main__":
    from tap import tapify

    tapify(normalize_sae_features)