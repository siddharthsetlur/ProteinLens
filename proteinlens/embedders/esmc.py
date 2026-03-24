"""ESMC (ESM Cambrian) embedder for proteinlens."""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from proteinlens.embedders.base import BaseEmbedder
from proteinlens.utils import get_device


class ESMC_Embedder(BaseEmbedder):
    """ESMC embedder for extracting protein language model embeddings.

    ESMC (ESM Cambrian) is EvolutionaryScale's representation learning model,
    the successor to ESM2 for protein embeddings. It provides per-residue
    embeddings and per-layer hidden states, suitable for SAE training.
    """

    # Model dimensions by name
    MODEL_DIMS = {
        "esmc_300m": 960,
        "esmc_600m": 1152,
    }

    MODEL_LAYERS = {
        "esmc_300m": 30,
        "esmc_600m": 36,
    }

    def __init__(
        self,
        model_name: str = "esmc_300m",
        device: Optional[str] = None,
        max_length: int = 1024,
    ):
        """Initialize ESMC embedder.

        Args:
            model_name: ESMC model identifier ('esmc_300m' or 'esmc_600m')
            device: Device to run on (cuda/cpu/mps). Auto-detected if None.
            max_length: Maximum sequence length.
        """
        if device is None:
            device = get_device()

        super().__init__(model_name, device)
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self) -> None:
        """Load ESMC model from EvolutionaryScale."""
        from esm.models.esmc import ESMC

        self.model = ESMC.from_pretrained(
            self.model_name, device=torch.device(self.device)
        )
        self.model.eval()
        self.tokenizer = self.model.tokenizer

    def extract_embeddings(
        self,
        sequences: List[str],
        layer: int,
        batch_size: int = 8,
        return_contacts: bool = False,
    ) -> np.ndarray:
        """Extract embeddings from sequences at specified layer.

        Args:
            sequences: List of protein sequences
            layer: Layer number to extract embeddings from (0-indexed into hidden_states)
            batch_size: Batch size for processing
            return_contacts: Not used, for compatibility

        Returns:
            Flattened tensor of embeddings (total_tokens, embedding_dim)
        """
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
            layers: List of layer numbers to extract (0-indexed, max = num_layers - 1)
            batch_size: Batch size for processing
            shuffle: If True, shuffle the flattened embeddings (for training)

        Returns:
            Dictionary mapping layer number to flattened embeddings tensor
            Shape: (total_tokens, embedding_dim) with CLS/EOS tokens removed
        """
        # Validate layers
        max_layer = self.MODEL_LAYERS[self.model_name] - 1
        for layer in layers:
            if layer < 0 or layer > max_layer:
                raise ValueError(f"Layer {layer} out of range [0, {max_layer}]")

        # Initialize storage
        all_embeddings = {layer: [] for layer in layers}

        # Process in batches
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        batch_iterator = range(0, len(sequences), batch_size)

        if num_batches > 1:
            batch_iterator = tqdm(
                batch_iterator, desc="Processing batches", total=num_batches
            )

        for i in batch_iterator:
            batch_sequences = sequences[i : i + batch_size]

            # Clean sequences
            batch_sequences = [self.preprocess_sequence(seq) for seq in batch_sequences]

            # Tokenize
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inputs = self.tokenizer(
                    batch_sequences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )

            # Move tokens to device
            input_ids = inputs["input_ids"].to(self.device, non_blocking=True)

            # Forward pass
            with torch.no_grad():
                output = self.model(input_ids)
                # hidden_states shape: [n_layers, B, L, D]
                hidden_states = output.hidden_states

                for layer in layers:
                    # Index into stacked hidden states: [B, L, D]
                    layer_output = hidden_states[layer].detach().float().cpu()

                    # Remove CLS and EOS tokens for each sequence in batch
                    # ESMC uses: [CLS] seq [EOS] [PAD]...
                    for seq_idx, seq_len in enumerate(
                        [len(seq) for seq in batch_sequences]
                    ):
                        seq_embeddings = layer_output[
                            seq_idx, 1 : seq_len + 1, :
                        ].detach().cpu()
                        all_embeddings[layer].append(seq_embeddings)

        # Process the collected embeddings
        result = {}
        for layer in layers:
            layer_tensor = torch.cat(all_embeddings[layer], dim=0)

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
        embeddings_dict = self.extract_embeddings_multiple_layers(
            sequences, [layer], batch_size, shuffle=False
        )

        boundaries = []
        current_pos = 0
        for sequence in sequences:
            seq_len = len(sequence)
            boundaries.append((current_pos, current_pos + seq_len))
            current_pos += seq_len

        return {
            "embeddings": embeddings_dict[layer],
            "boundaries": boundaries,
        }

    def embed_single_sequence(self, sequence: str, layer: int) -> np.ndarray:
        """Extract embeddings for a single sequence.

        Args:
            sequence: Protein sequence string
            layer: Layer number to extract from

        Returns:
            Embeddings with shape (seq_len, embedding_dim)
        """
        embeddings = self.extract_embeddings([sequence], layer, batch_size=1)
        seq_len = len(sequence)
        if isinstance(embeddings, torch.Tensor):
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
        batch_size: int = 8,
    ) -> Union[np.ndarray, None]:
        """Extract embeddings for sequences in a FASTA file.

        Args:
            fasta_path: Path to FASTA file
            layer: Layer to extract
            output_path: Optional path to save embeddings (.pt)
            batch_size: Batch size

        Returns:
            Embeddings tensor or None if saved to file
        """
        sequences = self._read_fasta(fasta_path)
        embeddings = self.extract_embeddings(sequences, layer, batch_size)

        if output_path:
            output_path = Path(output_path)
            if not str(output_path).endswith(".pt"):
                output_path = output_path.with_suffix(".pt")
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
        sequences = self._read_fasta(fasta_path)

        effective_batch_size = batch_size * 2 if self.device != "cpu" else batch_size

        embeddings_dict = self.extract_embeddings_multiple_layers(
            sequences, layers, effective_batch_size, shuffle=shuffle
        )

        if output_dir:
            output_dir = Path(output_dir)
            import yaml

            for layer, embeddings in embeddings_dict.items():
                layer_dir = output_dir / f"layer_{layer}"
                layer_dir.mkdir(parents=True, exist_ok=True)

                shard_dir = layer_dir / fasta_path.stem
                shard_dir.mkdir(parents=True, exist_ok=True)

                output_path = shard_dir / "activations.pt"
                torch.save(embeddings, output_path)

                metadata = {
                    "model": self.model_name,
                    "layer": layer,
                    "d_model": int(embeddings.shape[1]),
                    "total_tokens": int(embeddings.shape[0]),
                    "dtype": "float32",
                }
                metadata_path = shard_dir / "metadata.yaml"
                with open(metadata_path, "w") as f:
                    yaml.dump(metadata, f, default_flow_style=False)

            return None
        else:
            return embeddings_dict

    def get_embedding_dim(self, layer: int) -> int:
        """Get embedding dimension for specified layer."""
        if self.model_name in self.MODEL_DIMS:
            return self.MODEL_DIMS[self.model_name]
        raise ValueError(f"Unknown embedding dimension for {self.model_name}")

    @property
    def available_layers(self) -> List[int]:
        """Get list of available layers (0-indexed)."""
        if self.model_name in self.MODEL_LAYERS:
            return list(range(self.MODEL_LAYERS[self.model_name]))
        return list(range(30))  # Default for esmc_300m

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
            max_length=self.max_length,
        )

    def preprocess_sequence(self, sequence: str) -> str:
        """Clean and validate protein sequence."""
        return sequence.strip().upper()

    @staticmethod
    def _read_fasta(fasta_path: Path) -> List[str]:
        """Read sequences from a FASTA file."""
        sequences = []
        with open(fasta_path, "r") as f:
            current_seq = []
            for line in f:
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_seq:
                sequences.append("".join(current_seq))
        return sequences
