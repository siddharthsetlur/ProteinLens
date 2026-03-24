from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

from interplm.sae.dictionary import Dictionary
from interplm.sae.inference import get_sae_feats_in_batches
from interplm.utils import get_device


def get_random_sample_of_sae_feats(
    sae: Dictionary,
    aa_embds_dir: Path,
    shards_to_search: List[int],
    get_aa_activations_fn: Callable,
    max_samples_per_feature: int = 1000,
    num_feature_chunks: int = 32,
    batch_chunk_size: int = 10_000,
):
    """
    Get a random sample of up to `max_samples_per_feature` nonzero activations for each feature by scanning
    across n_shards shards of PLM embeddings.

    Args:
        sae: Dictionary (SAE model)
        aa_embds_dir: Directory containing amino acid embeddings
        shards_to_search: List of shard indices to process
        get_aa_activations_fn: Function to load activations for a shard
        max_samples_per_feature: Maximum number of nonzero activations to sample per feature (default: 1000)
        num_feature_chunks: Number of chunks to split features into for processing (default: 32)
        batch_chunk_size: Batch size for get_sae_feats_in_batches (default: 10,000)
    """
    device = get_device()

    nonzero_acts_per_feat = defaultdict(list)
    for shard in shards_to_search:
        aa_acts = get_aa_activations_fn(aa_embds_dir, shard, device)

        # iterate through esm_acts and for each feat, add any nonzero acts to nonzero_per_feat
        for feat_chunk_list in np.array_split(range(sae.dict_size), num_feature_chunks):
            sae_feats = (
                get_sae_feats_in_batches(
                    sae=sae,
                    device=device,
                    aa_embds=aa_acts,
                    feat_list=feat_chunk_list,
                    chunk_size=batch_chunk_size,
                    normalize_features=True,  # Normalize to [0,1] range
                )
                .cpu()
                .numpy()
            )

            for i, feature in enumerate(feat_chunk_list):
                # find the nonzero acts and add them to nonzero_acts_per_feat
                nonzero_for_feat = sae_feats[:, i][sae_feats[:, i] != 0]
                # if nonzero_per_feat > max_samples_per_feature, subsample to max_samples_per_feature
                if len(nonzero_for_feat) > max_samples_per_feature:
                    nonzero_for_feat = np.random.choice(
                        nonzero_for_feat, max_samples_per_feature, replace=False
                    )

                nonzero_acts_per_feat[feature] = nonzero_for_feat.tolist()

    return nonzero_acts_per_feat
