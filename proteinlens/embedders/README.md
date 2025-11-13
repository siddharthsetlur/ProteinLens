# Adding Your Own Protein Embedder

This guide explains how to add support for a new protein embedding model to InterPLM.

## Overview

InterPLM uses a plugin architecture where each embedder implements the `BaseEmbedder` interface. This allows the rest of the framework (SAE training, analysis, visualization) to work with any protein embedding model - whether it's a language model (ESM, ProGen), structure prediction model (AlphaFold), or any other type.

## Step 1: Create Your Embedder Implementation

Create a new file in this directory (e.g., `my_embedder.py`) that implements the `BaseEmbedder` class:

```python
from interplm.embedders.base import BaseEmbedder
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path

class MyEmbedder(BaseEmbedder):
    """Embedder for MyModel protein embeddings."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name, device)
        # Initialize your model here
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load your model and tokenizer."""
        # Load your model
        # self.model = load_my_model(self.model_name)
        # self.tokenizer = load_my_tokenizer(self.model_name)
        pass
    
    def extract_embeddings(
        self, 
        sequences: List[str], 
        layer: int,
        batch_size: int = 8,
        return_contacts: bool = False
    ) -> np.ndarray:
        """Extract embeddings from sequences."""
        # Process sequences and return embeddings
        # Shape should be (n_sequences, max_seq_len, embedding_dim)
        pass
    
    def embed_single_sequence(self, sequence: str, layer: int) -> np.ndarray:
        """Extract embeddings for a single sequence."""
        # Return shape: (seq_len, embedding_dim)
        pass
    
    def embed_fasta_file(
        self,
        fasta_path: Path,
        layer: int,
        output_path: Optional[Path] = None,
        batch_size: int = 8
    ) -> Optional[np.ndarray]:
        """Process a FASTA file."""
        # Load sequences from FASTA
        # Extract embeddings
        # Save to output_path if provided, else return array
        pass
    
    def get_embedding_dim(self, layer: int) -> int:
        """Return the embedding dimension for the specified layer."""
        # Return integer dimension
        pass
    
    @property
    def available_layers(self) -> List[int]:
        """List available layers for extraction."""
        # Return list of layer indices
        pass
    
    @property
    def max_sequence_length(self) -> int:
        """Maximum sequence length the model can handle."""
        # Return max length
        pass
    
    def tokenize(self, sequences: List[str]) -> Dict:
        """Tokenize sequences."""
        # Return tokenized inputs as dict
        pass
```

## Step 2: Register Your Embedder

Add your embedder to the factory function in `__init__.py`:

```python
from interplm.embedders.my_embedder import MyEmbedder

def get_embedder(embedder_type: str, **kwargs) -> BaseEmbedder:
    embedder_types = {
        'esm': ESM,
        'my_model': MyEmbedder,  # Add your embedder here
    }
    # ...
```

## Step 3: Test Your Implementation

Create a test script to verify your embedder works correctly:

```python
from interplm.embedders import get_embedder

# Initialize your embedder
embedder = get_embedder("my_model", model_name="your_model_name")

# Test single sequence embedding
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
embeddings = embedder.embed_single_sequence(sequence, layer=1)
print(f"Embedding shape: {embeddings.shape}")

# Test batch embedding
sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"
]
batch_embeddings = embedder.extract_embeddings(sequences, layer=1)
print(f"Batch embedding shape: {batch_embeddings.shape}")
```

## Step 4: Use with InterPLM

Once your embedder is implemented, you can use it with all InterPLM features:

```python
from interplm.embedders import get_embedder
from interplm.train import train_sae
from interplm.feature_analysis import collect_top_activating

# Use your embedder for SAE training
embedder = get_embedder("my_model", model_name="model_name")
embeddings = embedder.embed_fasta_file("proteins.fasta", layer=5)

# Train SAE on your embeddings
sae = train_sae(embeddings, config)

# Analyze features (works with any embedder!)
top_proteins = collect_top_activating(sae, embeddings)
```

## Reference Implementation

See `esm.py` for a complete example implementation using the ESM models.
