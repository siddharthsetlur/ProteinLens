#!/usr/bin/env python
"""Extract protein embeddings for annotated proteins from UniProtKB."""

import os
from pathlib import Path
from typing import Type

import pandas as pd
import torch
from tqdm import tqdm

from proteinlens.embedders.base import BaseEmbedder
from proteinlens.embedders import get_embedder


def embed_annotations(
    input_dir: Path,
    output_dir: Path,
    embedder_type: str = "esm",
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    layer: int = 3,
    batch_size: int = 8,
    sequence_column: str = "sequence",
):
    """
    Extract PLM embeddings for proteins with annotations.

    Args:
        input_dir: Directory containing annotation CSV files (shard_*.csv)
        output_dir: Directory to save embeddings
        embedder_type: Type of protein embedder to use (default: esm)
        model_name: Model name/identifier (default: facebook/esm2_t6_8M_UR50D)
        layer: Layer to extract embeddings from (default: 3)
        batch_size: Batch size for processing (default: 8)
        sequence_column: Name of the column containing sequences (default: sequence)
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize protein embedder
    print(f"Loading {embedder_type} embedder: {model_name}")
    embedder = get_embedder(embedder_type, model_name=model_name)

    # Find all annotation shards
    # Look for protein data files in shard subdirectories
    shard_files = sorted(input_dir.glob("shard_*/protein_data.tsv"))
    if not shard_files:
        # Try CSV files directly in the input directory
        shard_files = sorted(input_dir.glob("shard_*.csv"))
    if not shard_files:
        # Try alternative naming patterns
        shard_files = sorted(input_dir.glob("*.csv"))

    if not shard_files:
        raise FileNotFoundError(f"No protein data files found in {input_dir}")

    print(f"Found {len(shard_files)} annotation files to process")

    # Process each shard
    for shard_file in tqdm(shard_files, desc="Processing shards"):
        # Load annotations - handle both CSV and TSV
        if shard_file.suffix == '.tsv':
            df = pd.read_csv(shard_file, sep='\t')
        else:
            df = pd.read_csv(shard_file)

        # Check for sequence column (case-insensitive)
        seq_col = None
        for col in df.columns:
            if col.lower() == sequence_column.lower():
                seq_col = col
                break

        if seq_col is None:
            raise ValueError(f"Column '{sequence_column}' not found in {shard_file}. Available columns: {list(df.columns)}")

        # Get sequences
        sequences = df[seq_col].tolist()

        print(f"\nProcessing {shard_file.name} with {len(sequences)} sequences")

        # Extract embeddings with boundaries
        embeddings_dict = embedder.extract_embeddings_with_boundaries(
            sequences,
            layer=layer,
            batch_size=batch_size
        )

        # Save embeddings - maintain shard directory structure if present
        if "shard_" in str(shard_file.parent.name):
            shard_dir = output_dir / shard_file.parent.name
            shard_dir.mkdir(parents=True, exist_ok=True)
            output_file = shard_dir / "embeddings.pt"
        else:
            output_file = output_dir / f"{shard_file.stem}.pt"

        # Get protein IDs from the dataframe
        protein_ids = df[seq_col].index.tolist() if df.index.name else df['Entry'].tolist() if 'Entry' in df.columns else list(range(len(sequences)))

        # Save embeddings with boundaries and protein IDs
        save_data = {
            'embeddings': embeddings_dict['embeddings'],  # Concatenated tensor
            'boundaries': embeddings_dict['boundaries'],  # List of (start, end) tuples
            'protein_ids': protein_ids  # List of protein identifiers
        }
        torch.save(save_data, output_file)
        print(f"Saved embeddings with boundaries to {output_file}")

    print(f"\nEmbedding extraction complete! All embeddings saved to {output_dir}")


if __name__ == "__main__":
    from tap import tapify
    tapify(embed_annotations)
