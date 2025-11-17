#!/usr/bin/env python
"""
Extract protein embeddings from FASTA files for SAE training.
"""
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from proteinlens.embedders import get_embedder


def main(
    fasta_dir: Path,
    output_dir: Path,
    embedder_type: str = "esm",
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    layers: List[int] = [6],
    batch_size: int = 32,
    shard_index: Optional[int] = None,
):
    """
    Extract protein embeddings from FASTA files for SAE training.
    
    Args:
        fasta_dir: Directory containing FASTA files (sharded)
        output_dir: Directory to save embeddings
        embedder_type: Type of protein embedder to use (default: esm)
        model_name: Model name/identifier (default: facebook/esm2_t6_8M_UR50D)
        layers: Layers to extract (default: [3])
        batch_size: Batch size for processing (default: 8)
        shard_index: Optional: process only a specific shard by index (0-based)
    """
    
    # Initialize protein embedder
    print(f"Loading {embedder_type} embedder: {model_name}")
    embedder = get_embedder(embedder_type, model_name=model_name)
    
    # Get all FASTA files
    fasta_files = sorted(fasta_dir.glob("*.fasta"))
    if not fasta_files:
        fasta_files = sorted(fasta_dir.glob("*.fa"))
    
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found in {fasta_dir}")
    
    # Filter to specific shard if requested
    if shard_index is not None:
        if shard_index < 0 or shard_index >= len(fasta_files):
            raise ValueError(f"Shard index {shard_index} out of range. Found {len(fasta_files)} shards (0-{len(fasta_files)-1})")
        fasta_files = [fasta_files[shard_index]]
        print(f"Processing only shard {shard_index}: {fasta_files[0].name}")
    else:
        print(f"Found {len(fasta_files)} FASTA files to process")
    
    print(f"Extracting embeddings for layers: {layers}")
    print(f"Using batch size: {batch_size}")
    print(f"Device: {embedder.device}")
    
    # Process each FASTA file - extract all layers in one pass
    for fasta_file in tqdm(fasta_files, desc="Processing FASTA files", unit="file"):
        # Extract embeddings for all layers at once
        print(f"\nProcessing {fasta_file.name}...")
        embedder.embed_fasta_file_multiple_layers(
            fasta_file,
            layers=layers,
            output_dir=output_dir,
            batch_size=batch_size,
            shuffle=False
        )
    
    # Print summary
    for layer in layers:
        layer_dir = output_dir / f"layer_{layer}"
        print(f"Layer {layer} embeddings saved to {layer_dir}")
    
    print(f"\nEmbedding extraction complete! Saved to {output_dir}")


if __name__ == "__main__":
    from tap import tapify
    tapify(main)
