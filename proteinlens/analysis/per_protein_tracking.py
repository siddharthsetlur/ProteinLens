"""
Organizes proteins based on their maximum activation value for each feature. Both finds
proteins that have the higest activation value for each feature and finds proteins where
the maximum activation value *within that protein* is in a pre-specified quantile range.
"""

import heapq
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from interplm.sae.dictionary import Dictionary
from interplm.sae.inference import get_sae_feats_in_batches, split_up_feature_list
from interplm.utils import get_device
from interplm.data_processing.embedding_loader import load_shard_embeddings


class PerProteinActivationTracker:
    """
    Tracks and analyzes feature activations across proteins.

    This class maintains various statistics about how features activate across proteins:
    - Top N proteins per feature by maximum activation
    - Top N proteins per feature by percentage of activation
    - Proteins grouped by activation quantile ranges
    - Overall activation statistics across the protein dataset
    - Maximum activation value observed for each feature
    """

    def __init__(
        self,
        num_features: int,
        n_top: int,
        lower_quantile_thresholds: list = [
            (0, 0.2),
            (0.2, 0.4),
            (0.4, 0.6),
            (0.6, 0.8),
            (0.8, 1.0),
        ],
        activation_threshold: float = 0.0,
    ):
        """
        Args:
            num_features: Total number of features to track
            n_top: Number of top proteins to track per feature
            lower_quantile_thresholds: List of tuples defining activation quantile ranges
            activation_threshold: Minimum activation value to count as "activated" (default: 0.0)
        """
        self.num_features = num_features
        self.n_top = n_top
        self.activation_threshold = activation_threshold

        # Initialize min-heaps to track top activations
        # Each heap stores tuples of (activation_value, protein_id)
        # Track by maximum activation
        self.max_heap = [[] for _ in range(num_features)]
        # Track by activation percentage
        self.pct_heap = [[] for _ in range(num_features)]

        # Initialize quantile tracking
        self.lower_quantile_thresholds = lower_quantile_thresholds
        self.lower_quantile_lists = [
            {thresh: set() for thresh in lower_quantile_thresholds}
            for _ in range(num_features)
        ]

        # Initialize global statistics
        self.total_proteins = 0
        self.unique_proteins = set()  # Track unique proteins to avoid double counting
        self.proteins_with_activation = np.zeros(num_features)
        self.total_activation_percentage = np.zeros(num_features)
        self.max_activation_per_feature = np.zeros(num_features)

    def update(
        self, feature_activations: np.ndarray, protein_id: str, feature_ids: List
    ):
        """
        Update tracker with new protein activation data.

        Args:
            feature_activations: 2D array of activation values (amino_acids × features)
            protein_id: Identifier for the current protein
            feature_ids: List of feature indices being processed
        """
        if feature_ids is None:
            feature_ids = list(range(feature_activations.shape[1]))

        # Only increment total_proteins if this is a new protein (avoid double counting)
        if protein_id not in self.unique_proteins:
            self.unique_proteins.add(protein_id)
            self.total_proteins += 1

        # Calculate per-feature statistics
        max_activations = feature_activations.max(
            axis=0
        )  # Maximum activation per feature
        nonzero_counts = (feature_activations > self.activation_threshold).sum(
            axis=0
        )  # Count of activations above threshold
        # Percentage of amino acids activated
        pct_nonzero = nonzero_counts / feature_activations.shape[0]

        # Update global statistics
        has_activation = (nonzero_counts > 0).astype(int)
        self.proteins_with_activation[feature_ids] += has_activation
        # Only add percentage to total if feature is present (has_activation > 0)
        self.total_activation_percentage[feature_ids] += pct_nonzero * has_activation
        # Update max activation values
        self.max_activation_per_feature[feature_ids] = np.maximum(
            self.max_activation_per_feature[feature_ids], max_activations
        )

        # Process each feature
        for i, feature_id in enumerate(feature_ids):
            max_activation = max_activations[i]
            pct_activation = pct_nonzero[i]

            if max_activation > 0:
                # Update top-N heaps if activation is significant
                if len(self.max_heap[feature_id]) < self.n_top:
                    heapq.heappush(
                        self.max_heap[feature_id], (max_activation, protein_id)
                    )
                    heapq.heappush(
                        self.pct_heap[feature_id], (pct_activation, protein_id)
                    )
                else:
                    # Replace lowest value if current activation is higher
                    if max_activation > self.max_heap[feature_id][0][0]:
                        heapq.heapreplace(
                            self.max_heap[feature_id], (max_activation, protein_id)
                        )
                    if pct_activation > self.pct_heap[feature_id][0][0]:
                        heapq.heapreplace(
                            self.pct_heap[feature_id], (pct_activation, protein_id)
                        )

                # Assign to appropriate quantile range
                for start_threshold, end_threshold in self.lower_quantile_lists[
                    feature_id
                ]:
                    if (
                        max_activation > start_threshold
                        and max_activation <= end_threshold
                    ):
                        self.lower_quantile_lists[feature_id][
                            (start_threshold, end_threshold)
                        ].add(protein_id)
                        break

            # Track proteins with zero activation (up to 1000 per feature)
            elif (0.0, 0.0) in self.lower_quantile_lists[feature_id] and len(
                self.lower_quantile_lists[feature_id][(0.0, 0.0)]
            ) < 1_000:
                self.lower_quantile_lists[feature_id][(0.0, 0.0)].add(protein_id)

    def get_results(self) -> Dict[str, Dict[int, List[str]]]:
        """
        Compile and return all tracking results.

        Returns:
            Dictionary containing:
            - 'max': Top proteins by maximum activation
            - 'lower_quantile': Proteins grouped by activation quantiles
            - 'pct': Top proteins by activation percentage
            - 'pct_proteins_with_activation': Percentage of proteins showing any activation
            - 'avg_pct_activated_when_present': Average activation percentage when feature is present
            - 'max_activation_per_feature': Maximum activation value observed for each feature
        """
        # Sort and convert max activation heaps to lists
        max_result = {
            i: [p for _, p in sorted(self.max_heap[i], reverse=True)]
            for i in range(self.num_features)
        }

        # Process quantile results (randomly sample 10 proteins if more are present)
        lower_quantile_results = {
            feat: {quantile: [] for quantile in self.lower_quantile_thresholds}
            for feat in range(self.num_features)
        }
        for feat in range(self.num_features):
            for quantile, quantile_res in self.lower_quantile_lists[feat].items():
                n_res = len(quantile_res)
                quantile_res = list(quantile_res)
                if n_res > 10:
                    quantile_res = np.random.choice(quantile_res, 10, replace=False)
                lower_quantile_results[feat][quantile] = quantile_res

        # Sort and convert percentage activation heaps to lists
        pct_result = {
            i: [p for _, p in sorted(self.pct_heap[i], reverse=True)]
            for i in range(self.num_features)
        }

        # Calculate global statistics
        pct_proteins_with_activation = (
            self.proteins_with_activation / self.total_proteins
        ) * 100
        # Average percentage of AAs activated when feature is present
        avg_pct_activated_when_present = np.divide(
            self.total_activation_percentage,
            self.proteins_with_activation,
            out=np.zeros_like(self.total_activation_percentage, dtype=float),
            where=self.proteins_with_activation != 0,
        ) * 100  # Convert to percentage

        return {
            "max": max_result,
            "lower_quantile": lower_quantile_results,
            "pct": pct_result,
            "pct_proteins_with_activation": pct_proteins_with_activation.tolist(),
            "avg_pct_activated_when_present": avg_pct_activated_when_present.tolist(),
            "max_activation_per_feature": self.max_activation_per_feature.tolist(),
        }


def _get_protein_ids(aa_metadata_dir: Path, shard_num: int) -> pd.Series:
    """
    Convert per-protein metadata to per-amino-acid protein ID mapping.

    This function creates a mapping where each amino acid position gets its protein ID.
    For example, if protein A has length 100, this returns 100 copies of "A".

    This is used for matching amino-acid-level embeddings to their source proteins
    during feature activation analysis. It is NOT a replacement for UniProtMetadata
    which provides full protein-level metadata.

    Args:
        aa_metadata_dir: Directory containing shard_N/protein_data.tsv files
        shard_num: Shard number to load

    Expected format of protein_data.tsv:
        - Entry: protein ID (e.g., UniProt ID)
        - Length: number of amino acids in protein

    Returns:
        Series with protein IDs repeated per amino acid. Length equals total amino acids in shard.

    Example:
        If protein_data.tsv contains:
            Entry  Length
            P12345  50
            Q67890  30

        Returns Series of length 80: ['P12345', 'P12345', ...(50x), 'Q67890', 'Q67890', ...(30x)]
    """
    data = pd.read_csv(
        aa_metadata_dir / f"shard_{shard_num}" / "protein_data.tsv", sep="\t"
    )

    # create a new pandas series with the protein ids repeated for each aa (based on "Length" column)
    return pd.Series(
        np.repeat(data["Entry"], data["Length"]), name="protein_id"
    ).reset_index(drop=True)


def find_max_examples_per_feat(
    sae: Dictionary,
    aa_embeds_dir: Path,
    aa_metadata_dir: Path,
    shards_to_search: List[int],
    feature_chunk_size: int = 200,  # Number of features to process at once
    n_top_proteins_to_track: int = 10,
    lower_quantile_thresholds: List[Tuple[float, float]] = [
        (0, 0.4),
        (0.4, 0.8),
        (0.8, 1.0),
    ],
    activation_threshold: float = 0.05,  # Minimum activation to count as "activated"
):
    """
    Find proteins that maximally activate each feature in the sparse autoencoder.

    This function works with any PLM embeddings stored in the standard sharded format.
    Automatically detects whether protein IDs are stored with embeddings or in separate metadata.

    Embedding Formats Supported:
        1. With metadata: Dict with 'embeddings', 'boundaries', 'protein_ids' keys
        2. Without metadata: Tensor only (uses separate protein_data.tsv from aa_metadata_dir)

    Args:
        sae: Trained sparse autoencoder model
        aa_embeds_dir: Directory containing per-amino-acid embedding shards
        aa_metadata_dir: Directory containing protein metadata (shard_N/protein_data.tsv)
                        Used only if embeddings don't include protein_ids
        shards_to_search: List of data shards to process
        feature_chunk_size: Number of features to process in each chunk
        n_top_proteins_to_track: Number of top proteins to track per feature
        lower_quantile_thresholds: List of activation quantile ranges to track
        activation_threshold: Minimum activation value to count as "activated"

    Returns:
        Dictionary containing activation analysis results from PerProteinActivationTracker:
        - 'max': Top proteins by maximum activation
        - 'lower_quantile': Proteins grouped by activation quantiles
        - 'pct': Top proteins by activation percentage
        - 'pct_proteins_with_activation': Global statistics
        - 'avg_pct_activated_when_present': Global statistics
        - 'max_activation_per_feature': Maximum activation per feature
    """
    total_features = sae.dict_size
    device = get_device()

    # Initialize tracker for all features
    tracker = PerProteinActivationTracker(
        total_features,
        n_top=n_top_proteins_to_track,
        lower_quantile_thresholds=lower_quantile_thresholds,
        activation_threshold=activation_threshold,
    )

    # Process each shard of data
    for shard_idx, shard in enumerate(shards_to_search):
        print(f"Processing shard {shard} ({shard_idx+1}/{len(shards_to_search)})...")

        try:
            # Load embeddings - get full data with metadata if available
            shard_data = load_shard_embeddings(aa_embeds_dir, shard, device, return_tensor_only=False)

            # Extract embeddings and protein IDs
            if isinstance(shard_data, dict) and 'protein_ids' in shard_data and 'boundaries' in shard_data:
                # Format 1: Embeddings stored with protein_ids and boundaries
                aa_embeddings = shard_data['embeddings']
                boundaries = shard_data['boundaries']
                protein_ids = shard_data['protein_ids']

                # Convert boundaries to per-AA protein ID mapping
                uniprot_id_per_aa = []
                for protein_id, (start, end) in zip(protein_ids, boundaries):
                    # Repeat protein ID for each AA in this protein
                    uniprot_id_per_aa.extend([protein_id] * (end - start))
                uniprot_id_per_aa = pd.Series(uniprot_id_per_aa, name="protein_id")
            else:
                # Format 2: Embeddings without metadata, load from separate files
                aa_embeddings = shard_data if isinstance(shard_data, torch.Tensor) else shard_data['embeddings']
                uniprot_id_per_aa = _get_protein_ids(aa_metadata_dir, shard)

            # Check that the number of embeddings matches the number of protein IDs
            assert len(aa_embeddings) == len(uniprot_id_per_aa), (
                f"Number of embeddings ({len(aa_embeddings)}) does not match number of protein IDs "
                f"({len(uniprot_id_per_aa)})"
            )

            # Map amino acid indices to protein IDs
            prot_id_to_idx = defaultdict(list)
            for i, prot_id in enumerate(uniprot_id_per_aa):
                prot_id_to_idx[prot_id].append(i)

            # Process features in chunks to manage memory
            for feature_list in tqdm(
                split_up_feature_list(
                    total_features=total_features,
                    max_feature_chunk_size=feature_chunk_size,
                ),
                desc=f"Processing feature chunks for shard {shard}",
            ):
                # Get SAE features for current chunk
                sae_feats = get_sae_feats_in_batches(
                    sae=sae,
                    device=device,
                    aa_embds=aa_embeddings,
                    chunk_size=25_000,  # Keep original chunk size for speed
                    feat_list=feature_list,
                    normalize_features=True,  # Use normalized features for statistics
                )

                # Process each protein - original approach for speed
                for prot_id, prot_idx in prot_id_to_idx.items():
                    # Get features for this protein (CPU for memory efficiency)
                    protein_features = sae_feats[prot_idx].cpu().numpy()

                    # Update tracker with this protein's activations
                    tracker.update(
                        protein_features,
                        protein_id=prot_id,
                        feature_ids=feature_list,
                    )

                # Free memory after processing all proteins for this feature chunk
                del sae_feats

            # Clear shard data to free memory
            del aa_embeddings
            del uniprot_id_per_aa
            del prot_id_to_idx

        except Exception as e:
            print(f"Error processing shard {shard}: {e}")
            raise e

    return tracker.get_results()
