"""ESM (Evolutionary Scale Modeling) embedder for InterPLM."""

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Note: If you see NumPy compatibility warnings, recreate the conda environment
# with: conda env create -f environment.yml

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

from interplm.embedders.base import BaseEmbedder
from interplm.utils import get_device


class ESM(BaseEmbedder):
    """ESM embedder for extracting protein language model embeddings."""
    
    # Model dimensions by architecture
    MODEL_DIMS = {
        "facebook/esm2_t6_8M_UR50D": 320,
        "facebook/esm2_t12_35M_UR50D": 480,
        "facebook/esm2_t30_150M_UR50D": 640,
        "facebook/esm2_t33_650M_UR50D": 1280,
        "facebook/esm2_t36_3B_UR50D": 2560,
        "facebook/esm2_t48_15B_UR50D": 5120,
    }
    
    # Aliases for convenience (without facebook/ prefix)
    MODEL_ALIASES = {
        "esm2_t6_8M_UR50D": "facebook/esm2_t6_8M_UR50D",
        "esm2_t12_35M_UR50D": "facebook/esm2_t12_35M_UR50D",
        "esm2_t30_150M_UR50D": "facebook/esm2_t30_150M_UR50D",
        "esm2_t33_650M_UR50D": "facebook/esm2_t33_650M_UR50D",
        "esm2_t36_3B_UR50D": "facebook/esm2_t36_3B_UR50D",
        "esm2_t48_15B_UR50D": "facebook/esm2_t48_15B_UR50D",
    }
    
    def __init__(
        self, 
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: Optional[str] = None,
        max_length: int = 1024
    ):
        """Initialize ESM embedder.
        
        Args:
            model_name: HuggingFace model identifier (e.g., 'facebook/esm2_t6_8M_UR50D')
                       or shorthand (e.g., 'esm2_t6_8M_UR50D')
            device: Device to run on (cuda/cpu/mps)
            max_length: Maximum sequence length
        """
        # Resolve model name aliases
        if model_name in ESM.MODEL_ALIASES:
            model_name = ESM.MODEL_ALIASES[model_name]
        
        if device is None:
            device = get_device()
        
        super().__init__(model_name, device)
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load ESM model and tokenizer from HuggingFace."""
        # Load tokenizer with clean_up_tokenization_spaces set to avoid warning
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            clean_up_tokenization_spaces=True
        )
        
        # Load model, ignoring mismatched keys for pooler (we don't use it)
        from transformers import logging
        logging.set_verbosity_error()  # Suppress warnings
        
        self.model = EsmModel.from_pretrained(
            self.model_name,
            add_pooling_layer=False  # We don't need the pooler
        )
        
        logging.set_verbosity_warning()  # Reset to default
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
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
            return_contacts: Not used, for compatibility
            
        Returns:
            Array of embeddings with shape (n_sequences, max_seq_len, embedding_dim)
        """
        # Extract just one layer
        embeddings_dict = self.extract_embeddings_multiple_layers(
            sequences, [layer], batch_size
        )
        return embeddings_dict[layer]
    
    def extract_embeddings_multiple_layers(
        self,
        sequences: List[str],
        layers: List[int],
        batch_size: int = 8,
        shuffle: bool = False,
    ) -> Dict[int, torch.Tensor]:
        """Extract embeddings from sequences at multiple layers efficiently.
        
        Args:
            sequences: List of protein sequences
            layers: List of layer numbers to extract
            batch_size: Batch size for processing
            shuffle: If True, shuffle the flattened embeddings (for training)
            
        Returns:
            Dictionary mapping layer number to flattened embeddings tensor
            Shape: (total_tokens, embedding_dim) with CLS/EOS tokens removed
        """
        # Validate layers
        max_layer = self.model.config.num_hidden_layers
        for layer in layers:
            if layer < 0 or layer > max_layer:
                raise ValueError(f"Layer {layer} out of range [0, {max_layer}]")
        
        # Initialize storage
        all_embeddings = {layer: [] for layer in layers}
        
        # Process in batches with progress bar if multiple batches
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        batch_iterator = range(0, len(sequences), batch_size)
        
        if num_batches > 1:

            batch_iterator = tqdm(batch_iterator, desc="Processing batches", total=num_batches)
        
        for i in batch_iterator:
            batch_sequences = sequences[i:i+batch_size]
            
            # Clean sequences
            batch_sequences = [self.preprocess_sequence(seq) for seq in batch_sequences]
            
            # Tokenize with optimizations
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore tokenizer warnings
                inputs = self.tokenizer(
                    batch_sequences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
            
            # Move to device (non-blocking for speed)
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            
            # Get all hidden states in one forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of tensors, one per layer
                
                # Extract requested layers
                for layer in layers:
                    layer_output = hidden_states[layer].detach().cpu()
                    
                    # Always remove CLS and EOS tokens for each sequence in batch
                    # ESM uses: [CLS] seq [EOS] [PAD]...
                    for seq_idx, seq_len in enumerate([len(seq) for seq in batch_sequences]):
                        # Extract only the actual sequence tokens (position 1 to seq_len+1)
                        seq_embeddings = layer_output[seq_idx, 1:seq_len+1, :].detach().cpu()
                        all_embeddings[layer].append(seq_embeddings)
        
        # Process the collected embeddings
        result = {}
        for layer in layers:
            # Concatenate all sequence embeddings into one flat tensor
            layer_tensor = torch.cat(all_embeddings[layer], dim=0)
            
            # Optionally shuffle for training
            if shuffle:
                perm = torch.randperm(layer_tensor.size(0))
                layer_tensor = layer_tensor[perm]
            
            result[layer] = layer_tensor
        
        return result

    def extract_embeddings_with_boundaries(
        self,
        sequences: List[str],
        layer: int,
        batch_size: int = 8,
    ) -> Dict[str, Union[torch.Tensor, List[Tuple[int, int]]]]:
        """Extract embeddings and track protein boundaries.

        Args:
            sequences: List of protein sequences
            layer: Layer number to extract
            batch_size: Batch size for processing

        Returns:
            Dictionary with:
                'embeddings': Concatenated tensor (total_tokens, embedding_dim)
                'boundaries': List of (start, end) tuples for each protein
        """
        # Extract embeddings for the layer
        embeddings_dict = self.extract_embeddings_multiple_layers(
            sequences, [layer], batch_size, shuffle=False  # Never shuffle when tracking boundaries
        )

        # Now we need to track boundaries
        # Re-process to get boundaries (this is a bit inefficient but maintains compatibility)
        boundaries = []
        current_pos = 0

        # Process sequences in same order to calculate boundaries
        for sequence in sequences:
            seq_len = len(sequence)
            boundaries.append((current_pos, current_pos + seq_len))
            current_pos += seq_len

        return {
            'embeddings': embeddings_dict[layer],
            'boundaries': boundaries
        }

    def embed_single_sequence(
        self,
        sequence: str,
        layer: int
    ) -> np.ndarray:
        """Extract embeddings for a single sequence.

        Args:
            sequence: Protein sequence string
            layer: Layer number to extract from

        Returns:
            Embeddings with shape (seq_len, embedding_dim)
        """
        embeddings = self.extract_embeddings([sequence], layer, batch_size=1)
        # Remove batch dimension and padding
        seq_len = len(sequence)
        if isinstance(embeddings, torch.Tensor):
            # Handle both 2D (seq_len, dim) and 3D (batch, seq_len, dim) cases
            if embeddings.ndim == 3:
                return embeddings[0, :seq_len, :].cpu().numpy()
            else:
                return embeddings[:seq_len, :].cpu().numpy()
        else:
            if embeddings.ndim == 3:
                return embeddings[0, :seq_len, :]
            else:
                return embeddings[:seq_len, :]
    
    def embed_fasta_file(
        self,
        fasta_path: Path,
        layer: int,
        output_path: Optional[Path] = None,
        batch_size: int = 8
    ) -> Union[np.ndarray, None]:
        """Extract embeddings for sequences in a FASTA file.
        
        Args:
            fasta_path: Path to FASTA file
            layer: Layer to extract
            output_path: Optional path to save embeddings (.pt or .npy)
            batch_size: Batch size
            
        Returns:
            Embeddings array or None if saved to file
        """
        # Read FASTA file
        sequences = []
        with open(fasta_path, 'r') as f:
            current_seq = []
            for line in f:
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_seq:
                sequences.append(''.join(current_seq))
        
        # Extract embeddings
        embeddings = self.extract_embeddings(sequences, layer, batch_size)
        
        # Save or return
        if output_path:
            output_path = Path(output_path)
            # Always save as .pt files (PyTorch native format)
            if not str(output_path).endswith('.pt'):
                output_path = output_path.with_suffix('.pt')
            
            if isinstance(embeddings, torch.Tensor):
                torch.save(embeddings, output_path)
            else:
                torch.save(torch.from_numpy(embeddings), output_path)
            return None
        else:
            return embeddings
    
    def embed_fasta_file_multiple_layers(
        self,
        fasta_path: Path,
        layers: List[int],
        output_dir: Optional[Path] = None,
        batch_size: int = 8,
        shuffle: bool = False,
    ) -> Union[Dict[int, torch.Tensor], None]:
        """Extract embeddings at multiple layers from a FASTA file.
        
        Args:
            fasta_path: Path to FASTA file
            layers: List of layers to extract
            output_dir: Optional directory to save embeddings
            batch_size: Batch size
            shuffle: If True, shuffle flattened embeddings (for training)
            
        Returns:
            Dictionary of flattened embeddings (CLS/EOS removed) or None if saved
        """
        # Read FASTA file
        sequences = []
        with open(fasta_path, 'r') as f:
            current_seq = []
            for line in f:
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_seq:
                sequences.append(''.join(current_seq))
        
        # Extract all layers at once with larger batch size for speed
        # Increase batch size if using GPU
        effective_batch_size = batch_size * 2 if self.device != "cpu" else batch_size
        
        embeddings_dict = self.extract_embeddings_multiple_layers(
            sequences, layers, effective_batch_size,
            shuffle=shuffle
        )
        
        # Save or return
        if output_dir:
            output_dir = Path(output_dir)
            import yaml
            
            for layer, embeddings in embeddings_dict.items():
                layer_dir = output_dir / f"layer_{layer}"
                layer_dir.mkdir(parents=True, exist_ok=True)
                
                # Create subdirectory for this shard (legacy format)
                shard_dir = layer_dir / fasta_path.stem
                shard_dir.mkdir(parents=True, exist_ok=True)
                
                # Save embeddings as activations.pt
                output_path = shard_dir / "activations.pt"
                torch.save(embeddings, output_path)
                
                # Create metadata.yaml
                metadata = {
                    "model": self.model_name,
                    "layer": layer,
                    "d_model": int(embeddings.shape[1]),  # Ensure int for YAML
                    "total_tokens": int(embeddings.shape[0]),  # Ensure int for YAML
                    "dtype": "float32"
                }
                metadata_path = shard_dir / "metadata.yaml"
                with open(metadata_path, 'w') as f:
                    yaml.dump(metadata, f, default_flow_style=False)
            
            return None
        else:
            return embeddings_dict
    
    def get_embedding_dim(self, layer: int) -> int:
        """Get embedding dimension for specified layer."""
        if self.model:
            return self.model.config.hidden_size
        elif self.model_name in self.MODEL_DIMS:
            return self.MODEL_DIMS[self.model_name]
        else:
            raise ValueError(f"Unknown embedding dimension for {self.model_name}")
    
    @property
    def available_layers(self) -> List[int]:
        """Get list of available layers."""
        if self.model:
            # Layer 0 is embeddings, 1 through n_layers are transformer layers
            return list(range(self.model.config.num_hidden_layers + 1))
        else:
            # Defaults for known models
            return list(range(7))  # ESM2-8M has 6 layers + embedding layer
    
    @property
    def max_sequence_length(self) -> int:
        """Maximum sequence length the model can process."""
        return self.max_length
    
    def tokenize(self, sequences: List[str]) -> Dict:
        """Tokenize sequences for model input."""
        return self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
    
    def preprocess_sequence(self, sequence: str) -> str:
        """Clean and validate protein sequence."""
        # Remove whitespace and make uppercase
        sequence = sequence.strip().upper() 
        return sequence