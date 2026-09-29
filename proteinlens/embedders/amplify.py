"""AMPLIFY embedder for proteinlens.

AMPLIFY (Fournier et al., 2024; chandar-lab/AMPLIFY_*) is a BERT-style
protein encoder with pre-norm RMSNorm, SwiGLU and RoPE. Three differences from
the HF ESM2 implementation matter for embedding extraction:

1. ``hidden_states`` has one entry per encoder block and *no* embedding-layer
   entry: ``hidden_states[i]`` is the residual stream after block ``i + 1``.
   HF ESM2 prepends the embedding output, so there ``hidden_states[L]`` is the
   output of block ``L``. We keep the ESM convention in this API -- ``layer=L``
   means "residual stream after block L" -- and read ``hidden_states[L - 1]``.
2. The model expects an *additive* float attention mask (0 = attend,
   -inf = padding); the tokenizer's 0/1 mask trips an assertion. On CUDA the
   remote code uses xformers ``memory_efficient_attention``, whose bias must
   have a sequence length that is a multiple of 8, hence ``pad_to_multiple_of``.
3. The vocabulary has 20 amino acids + B only; X/U/Z/O/J tokenize to
   ``<unk>``. Tokenization is still one token per residue, so the
   ``<bos> seq <eos>`` layout and ``[1 : len(seq) + 1]`` stripping match ESM.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from proteinlens.embedders.base import BaseEmbedder
from proteinlens.utils import get_device


class Amplify(BaseEmbedder):
    """AMPLIFY embedder for extracting per-residue residual-stream activations."""

    # Pinned HF revisions so the remote modelling code cannot change under us.
    MODEL_REVISIONS = {
        "chandar-lab/AMPLIFY_120M": "d918a9e8c64dc43f6ec4996022c3deba14d66470",
    }

    MODEL_ALIASES = {
        "AMPLIFY_120M": "chandar-lab/AMPLIFY_120M",
    }

    def __init__(
        self,
        model_name: str = "chandar-lab/AMPLIFY_120M",
        device: Optional[str] = None,
        max_length: int = 2048,
        revision: Optional[str] = None,
    ):
        """Initialize AMPLIFY embedder.

        Args:
            model_name: HuggingFace model identifier (e.g. 'chandar-lab/AMPLIFY_120M')
                        or shorthand (e.g. 'AMPLIFY_120M').
            device: Device to run on (cuda/cpu/mps).
            max_length: Maximum tokenized length *including* <bos>/<eos>. Longer
                        sequences raise rather than being silently truncated.
            revision: HF revision; defaults to the pinned hash in MODEL_REVISIONS.
        """
        if model_name in Amplify.MODEL_ALIASES:
            model_name = Amplify.MODEL_ALIASES[model_name]

        if device is None:
            device = get_device()

        super().__init__(model_name, device)
        self.max_length = max_length
        self.revision = revision or Amplify.MODEL_REVISIONS.get(model_name)
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self) -> None:
        """Load AMPLIFY model and tokenizer (remote code) from HuggingFace."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, revision=self.revision
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, revision=self.revision
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def _hidden_state_index(self, layer: int) -> int:
        """Map an ESM-convention layer number (output of block L) to AMPLIFY's index."""
        n_layers = self.model.config.num_hidden_layers
        if layer < 1 or layer > n_layers:
            raise ValueError(
                f"Layer {layer} out of range [1, {n_layers}] "
                "(AMPLIFY exposes no embedding-layer hidden state)"
            )
        return layer - 1

    def _forward(self, sequences: List[str]):
        """Tokenize a batch and run the model, returning (input_ids, hidden_states)."""
        # A lone sequence needs no padding and hence no mask (xformers' alignment
        # constraint applies only to the bias), so bs=1 is an exact unpadded pass.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inputs = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                pad_to_multiple_of=8 if len(sequences) > 1 else None,
                truncation=False,
            )
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] > self.max_length:
            raise ValueError(
                f"Tokenized length {input_ids.shape[1]} exceeds max_length {self.max_length}"
            )

        # Additive mask; None when nothing is padded (the model then skips masking).
        mask = inputs["attention_mask"].bool()
        attention_mask = None
        if not mask.all():
            attention_mask = torch.where(mask, 0.0, float("-inf"))
            attention_mask = attention_mask.to(self.device, non_blocking=True)

        input_ids_dev = input_ids.to(self.device, non_blocking=True)
        with torch.no_grad():
            outputs = self.model(
                input_ids_dev, attention_mask=attention_mask, output_hidden_states=True
            )
        return input_ids, outputs.hidden_states

    def extract_embeddings(
        self,
        sequences: List[str],
        layer: int,
        batch_size: int = 8,
        return_contacts: bool = False,
    ) -> torch.Tensor:
        """Extract embeddings from sequences at specified layer.

        Args:
            sequences: List of protein sequences
            layer: Layer number (residual stream after block ``layer``)
            batch_size: Batch size for processing
            return_contacts: Not used, for compatibility

        Returns:
            Flattened embeddings (total_residues, embedding_dim), BOS/EOS removed
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
        """Extract embeddings from sequences at multiple layers.

        Args:
            sequences: List of protein sequences
            layers: List of layer numbers to extract
            batch_size: Batch size for processing
            shuffle: If True, shuffle the flattened embeddings (for training)

        Returns:
            Dictionary mapping layer number to flattened embeddings tensor
            Shape: (total_tokens, embedding_dim) with BOS/EOS tokens removed
        """
        indices = {layer: self._hidden_state_index(layer) for layer in layers}
        eos_id = self.tokenizer.eos_token_id

        all_embeddings = {layer: [] for layer in layers}

        num_batches = (len(sequences) + batch_size - 1) // batch_size
        batch_iterator = range(0, len(sequences), batch_size)
        if num_batches > 1:
            batch_iterator = tqdm(batch_iterator, desc="Processing batches", total=num_batches)

        for i in batch_iterator:
            batch_sequences = [
                self.preprocess_sequence(seq) for seq in sequences[i:i + batch_size]
            ]
            input_ids, hidden_states = self._forward(batch_sequences)

            # One token per residue: <bos> at 0, residues at 1..L, <eos> at L+1.
            seq_lens = [len(seq) for seq in batch_sequences]
            for seq_idx, seq_len in enumerate(seq_lens):
                if input_ids[seq_idx, seq_len + 1].item() != eos_id:
                    raise ValueError(
                        f"Token/residue misalignment for sequence of length {seq_len}"
                    )

            for layer in layers:
                layer_output = hidden_states[indices[layer]].detach().cpu()
                for seq_idx, seq_len in enumerate(seq_lens):
                    all_embeddings[layer].append(layer_output[seq_idx, 1:seq_len + 1, :])

        result = {}
        for layer in layers:
            layer_tensor = torch.cat(all_embeddings[layer], dim=0)
            if shuffle:
                perm = torch.randperm(layer_tensor.size(0))
                layer_tensor = layer_tensor[perm]
            result[layer] = layer_tensor

        return result

    def embed_single_sequence(self, sequence: str, layer: int) -> np.ndarray:
        """Extract embeddings for a single sequence.

        Args:
            sequence: Protein sequence string
            layer: Layer number to extract from

        Returns:
            Embeddings with shape (seq_len, embedding_dim)
        """
        embeddings = self.extract_embeddings([sequence], layer, batch_size=1)
        return embeddings.cpu().numpy()

    @staticmethod
    def _read_fasta(fasta_path: Path) -> List[str]:
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

    def embed_fasta_file(
        self,
        fasta_path: Path,
        layer: int,
        output_path: Optional[Path] = None,
        batch_size: int = 8,
    ) -> Union[torch.Tensor, None]:
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
            torch.save(embeddings, output_path)
            return None
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

        Writes the same ``layer_{L}/<shard>/{activations.pt,metadata.yaml}``
        layout as the ESM embedder, which the SAE data loader consumes.

        Args:
            fasta_path: Path to FASTA file
            layers: List of layers to extract
            output_dir: Optional directory to save embeddings
            batch_size: Batch size
            shuffle: If True, shuffle flattened embeddings (for training)

        Returns:
            Dictionary of flattened embeddings (BOS/EOS removed) or None if saved
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
                shard_dir = output_dir / f"layer_{layer}" / fasta_path.stem
                shard_dir.mkdir(parents=True, exist_ok=True)

                torch.save(embeddings, shard_dir / "activations.pt")

                metadata = {
                    "model": self.model_name,
                    "revision": self.revision,
                    "layer": layer,
                    "d_model": int(embeddings.shape[1]),
                    "total_tokens": int(embeddings.shape[0]),
                    "dtype": "float32",
                }
                with open(shard_dir / "metadata.yaml", "w") as f:
                    yaml.dump(metadata, f, default_flow_style=False)

            return None
        return embeddings_dict

    def get_embedding_dim(self, layer: int) -> int:
        """Get embedding dimension for specified layer."""
        return self.model.config.hidden_size

    @property
    def available_layers(self) -> List[int]:
        """Layers 1..n_layers (no embedding-layer hidden state is exposed)."""
        return list(range(1, self.model.config.num_hidden_layers + 1))

    @property
    def max_sequence_length(self) -> int:
        """Maximum tokenized sequence length (including <bos>/<eos>)."""
        return self.max_length

    def tokenize(self, sequences: List[str]) -> Dict:
        """Tokenize sequences for model input (0/1 attention mask, as returned)."""
        return self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=8,
            truncation=False,
        )

    def preprocess_sequence(self, sequence: str) -> str:
        """Clean protein sequence (strip whitespace, uppercase)."""
        return sequence.strip().upper()
