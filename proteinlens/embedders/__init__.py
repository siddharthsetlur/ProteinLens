"""Protein Embedders for proteinlens."""

from typing import Type
from proteinlens.embedders.base import BaseEmbedder
from proteinlens.embedders.amplify import Amplify
from proteinlens.embedders.esm import ESM


def get_embedder(embedder_type: str, **kwargs) -> BaseEmbedder:
    """Factory function to get a protein embedder instance.

    Args:
        embedder_type: Type of embedder ('esm' / 'esm2' / 'amplify').
        **kwargs: Additional arguments passed to embedder constructor

    Returns:
        Instance of the requested embedder

    Raises:
        ValueError: If embedder type is not supported
    """
    embedder_types = {
        'esm': ESM,
        'esm2': ESM,  # Alias
        'amplify': Amplify,
    }

    embedder_type_lower = embedder_type.lower()
    if embedder_type_lower not in embedder_types:
        raise ValueError(
            f"Embedder type '{embedder_type}' not supported. "
            f"Available types: {list(embedder_types.keys())}"
        )

    return embedder_types[embedder_type_lower](**kwargs)


def load_embedder(model_name: str, **kwargs) -> BaseEmbedder:
    """Instantiate the embedder matching a HuggingFace model name.

    AMPLIFY checkpoints (e.g. 'chandar-lab/AMPLIFY_120M') get the AMPLIFY
    embedder; everything else is treated as ESM2, as before.
    """
    if "amplify" in model_name.lower():
        return Amplify(model_name=model_name, **kwargs)
    return ESM(model_name=model_name, **kwargs)


__all__ = ["Amplify", "BaseEmbedder", "ESM", "get_embedder", "load_embedder"]
