"""Protein Embedders for proteinlens."""

from typing import Type
from proteinlens.embedders.base import BaseEmbedder
from proteinlens.embedders.esm import ESM
from proteinlens.embedders.esmc import ESMC_Embedder
from proteinlens.embedders.esm3 import ESM3Embedder


def get_embedder(embedder_type: str, **kwargs) -> BaseEmbedder:
    """Factory function to get a protein embedder instance.
    
    Args:
        embedder_type: Type of embedder ('esm', 'progen2', etc.)
        **kwargs: Additional arguments passed to embedder constructor
        
    Returns:
        Instance of the requested embedder
        
    Raises:
        ValueError: If embedder type is not supported
    """
    embedder_types = {
        'esm': ESM,
        'esm2': ESM,  # Alias
        'esmc': ESMC_Embedder,
        'esm3': ESM3Embedder,
    }
    
    embedder_type_lower = embedder_type.lower()
    if embedder_type_lower not in embedder_types:
        raise ValueError(
            f"Embedder type '{embedder_type}' not supported. "
            f"Available types: {list(embedder_types.keys())}"
        )
    
    return embedder_types[embedder_type_lower](**kwargs)


__all__ = ["BaseEmbedder", "ESM", "ESMC_Embedder", "ESM3Embedder", "get_embedder"]