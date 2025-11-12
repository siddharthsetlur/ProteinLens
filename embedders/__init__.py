"""Protein Embedders for InterPLM."""

from typing import Type
from interplm.embedders.base import BaseEmbedder
from interplm.embedders.esm import ESM


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
    }
    
    embedder_type_lower = embedder_type.lower()
    if embedder_type_lower not in embedder_types:
        raise ValueError(
            f"Embedder type '{embedder_type}' not supported. "
            f"Available types: {list(embedder_types.keys())}"
        )
    
    return embedder_types[embedder_type_lower](**kwargs)


__all__ = ["BaseEmbedder", "ESM", "get_embedder"]