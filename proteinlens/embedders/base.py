"""Abstract base class for Protein Embedders."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class BaseEmbedder(ABC):
    """Abstract base class for Protein Embedders.
    
    This class defines the interface that all protein embedder implementations must follow.
    Supports both language models (ESM, ProGen) and structure-based models (AlphaFold).
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize the Embedder.
        
        Args:
            model_name: Name or path of the model to load
            device: Device to run the model on ('cuda', 'cpu', or 'mps')
        """
        self.model_name = model_name
        self.device = device
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the embedder model and any necessary components."""
        pass

    def extract_embeddings_with_boundaries(
        self,
        sequences: List[str],
        layer: int,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        """Extract embeddings and track protein boundaries.

        Default implementation for backward compatibility.
        Embedders should override this for efficiency.

        Args:
            sequences: List of protein sequences
            layer: Layer number to extract
            batch_size: Batch size for processing

        Returns:
            Dictionary with:
                'embeddings': Concatenated tensor or array
                'boundaries': List of (start, end) tuples for each protein
        """
        # Default: call extract_embeddings and assume it returns concatenated embeddings
        embeddings = self.extract_embeddings(sequences, layer, batch_size)

        # Convert to tensor if it's numpy
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)

        # Calculate boundaries assuming embeddings are concatenated
        boundaries = []
        current_pos = 0
        for seq in sequences:
            seq_len = len(seq)
            boundaries.append((current_pos, current_pos + seq_len))
            current_pos += seq_len

        return {
            'embeddings': embeddings,
            'boundaries': boundaries
        }

    @abstractmethod
    def extract_embeddings(
        self, 
        sequences: List[str], 
        layer: int,
        batch_size: int = 8,
        return_contacts: bool = False
    ) -> np.ndarray:
        """Extract embeddings from sequences at specified layer.
        
        Args:
            sequences: List of protein sequences
            layer: Layer number to extract embeddings from
            batch_size: Batch size for processing
            return_contacts: Whether to return contact predictions (if available)
            
        Returns:
            Array of embeddings with shape (n_sequences, max_seq_len, embedding_dim)
        """
        pass
    
    def extract_embeddings_multiple_layers(
        self,
        sequences: List[str],
        layers: List[int],
        batch_size: int = 8,
    ) -> Dict[int, np.ndarray]:
        """Extract embeddings from sequences at multiple layers in a single pass.
        
        Default implementation calls extract_embeddings for each layer separately.
        Override this method for more efficient multi-layer extraction.
        
        Args:
            sequences: List of protein sequences
            layers: List of layer numbers to extract embeddings from
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping layer number to embeddings array
        """
        result = {}
        for layer in layers:
            result[layer] = self.extract_embeddings(sequences, layer, batch_size)
        return result
    
    @abstractmethod
    def embed_single_sequence(
        self,
        sequence: str,
        layer: int
    ) -> np.ndarray:
        """Extract embeddings for a single sequence.
        
        Args:
            sequence: Protein sequence string
            layer: Layer number to extract embeddings from
            
        Returns:
            Embeddings array with shape (seq_len, embedding_dim)
        """
        pass
    
    @abstractmethod
    def embed_fasta_file(
        self,
        fasta_path: Path,
        layer: int,
        output_path: Optional[Path] = None,
        batch_size: int = 8
    ) -> Union[np.ndarray, None]:
        """Extract embeddings for all sequences in a FASTA file.
        
        Args:
            fasta_path: Path to FASTA file
            layer: Layer number to extract embeddings from
            output_path: Optional path to save embeddings
            batch_size: Batch size for processing
            
        Returns:
            Embeddings array or None if saved to file
        """
        pass
    
    def embed_fasta_file_multiple_layers(
        self,
        fasta_path: Path,
        layers: List[int],
        output_dir: Optional[Path] = None,
        batch_size: int = 8
    ) -> Union[Dict[int, np.ndarray], None]:
        """Extract embeddings for all sequences in a FASTA file at multiple layers.
        
        Default implementation calls embed_fasta_file for each layer separately.
        Override this method for more efficient multi-layer extraction.
        
        Args:
            fasta_path: Path to FASTA file
            layers: List of layer numbers to extract embeddings from
            output_dir: Optional directory to save embeddings (creates layer_N subdirs)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping layer to embeddings array, or None if saved to files
        """
        result = {}
        for layer in layers:
            if output_dir:
                layer_output = output_dir / f"layer_{layer}" / f"{fasta_path.stem}.npy"
                layer_output.parent.mkdir(parents=True, exist_ok=True)
                self.embed_fasta_file(fasta_path, layer, layer_output, batch_size)
            else:
                result[layer] = self.embed_fasta_file(fasta_path, layer, None, batch_size)
        
        return None if output_dir else result
    
    @abstractmethod
    def get_embedding_dim(self, layer: int) -> int:
        """Return embedding dimension for specified layer.
        
        Args:
            layer: Layer number
            
        Returns:
            Embedding dimension
        """
        pass
    
    @property
    @abstractmethod
    def available_layers(self) -> List[int]:
        """List of available layers for embedding extraction.
        
        Returns:
            List of layer indices
        """
        pass
    
    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """Maximum sequence length the model can process.
        
        Returns:
            Maximum sequence length
        """
        pass
    
    @abstractmethod
    def tokenize(self, sequences: List[str]) -> Dict:
        """Tokenize sequences for model input.
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            Dictionary with tokenized inputs
        """
        pass
    
    def preprocess_sequence(self, sequence: str) -> str:
        """Preprocess a single sequence before tokenization.
        
        Default implementation just returns the sequence unchanged.
        Override if embedder needs special preprocessing.
        
        Args:
            sequence: Raw protein sequence
            
        Returns:
            Preprocessed sequence
        """
        return sequence