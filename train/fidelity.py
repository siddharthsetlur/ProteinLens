"""
Calculate the loss (cross entropy) fidelity metric for a Sparse Autoencoder (SAE) trained on ESM embeddings
1. Calculates original cross entropy and cross entropy after zero-ablation.
2. Creates a function to calculate cross entropy using SAE reconstructions.
3. During each evaluation step, uses step 2 to evaluate the current SAE model
   then compares to the original and zero-ablation cross entropy to calculate
   the loss recovered metric.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from esm import pretrained
from nnsight import NNsight
from tqdm import tqdm
from transformers import EsmForMaskedLM

# from interplm.esm.embed import shuffle_individual_parameters  # Not available in public repo
from interplm.sae.intervention import get_esm_output_with_intervention
from interplm.train.evaluation import EvaluationConfig, EvaluationManager
from interplm.utils import get_device


@dataclass
class ESMFidelityConfig(EvaluationConfig):
    model_name: str | None = None
    layer_idx: int | None = None
    corrupt: bool = False

    def build(self) -> "ESMFidelityFunction":
        return ESMFidelityFunction(self)


def calculate_cross_entropy(model_output, batch_tokens, batch_attn_mask):
    """Calculate cross entropy for each sequence in batch, excluding start/end tokens."""
    losses = []
    for j, mask in enumerate(batch_attn_mask):
        length = mask.sum()
        seq_logits = model_output[j, 1 : length - 1]
        seq_tokens = batch_tokens[j, 1 : length - 1]
        loss = F.cross_entropy(seq_logits, seq_tokens)
        losses.append(loss.item())
    return losses


def calculate_loss_recovered(ce_autoencoder, ce_identity, ce_zero_ablation):
    """
    Calculate the loss recovered metric for a Sparse Autoencoder (SAE).

    If the recovered loss is as good as  to the original loss as possible, the
    metric will be 100%. If the recovered loss is as bad as zero-ablating, the
    metric will be 0%.

    Parameters:
    ce_autoencoder (float): Cross-entropy loss when using the SAE's reconstructions
    ce_identity (float): Cross-entropy loss when using the identity function
    ce_zero_ablation (float): Cross-entropy loss when using the zero-ablation function

    Returns:
    float: The loss recovered metric as a percentage
    """

    numerator = ce_autoencoder - ce_identity
    denominator = ce_zero_ablation - ce_identity

    # Avoid division by zero
    if np.isclose(denominator, 0):
        return 0.0

    loss_recovered = 1 - (numerator / denominator)

    # Clip the result to be between 0 and 1
    loss_recovered = np.clip(loss_recovered, 0, 1)

    # Convert to percentage
    return loss_recovered * 100


class ESMFidelityFunction(EvaluationManager):
    def __init__(
        self,
        eval_config: "ESMFidelityConfig",
    ):
        # The super class just sets the config
        super().__init__(eval_config)

        print("Prepping loss fidelity_fn")
        self.device = get_device()

        # Extract config values
        self.model_name = eval_config.model_name
        self.eval_seq_path = eval_config.eval_seq_path
        self.layer_idx = eval_config.layer_idx
        self.batch_size = eval_config.eval_batch_size or 8
        self.corrupt = eval_config.corrupt

        # Load the ESM model and alphabet
        _, alphabet = pretrained.load_model_and_alphabet(self.model_name)
        self.model = EsmForMaskedLM.from_pretrained(f"facebook/{self.model_name}").to(
            self.device
        )

        if self.corrupt:
            raise NotImplementedError(
                "The 'corrupt' parameter requires shuffle_individual_parameters() "
                "which is not available in the public repo"
            )

        batch_converter = alphabet.get_batch_converter()

        # Load evaluation sequences
        with open(self.eval_seq_path, "r") as f:
            eval_seqs = [line.strip() for line in f]

        # Prepare data in the format expected by the batch converter
        data = [(f"protein{i}", seq) for i, seq in enumerate(eval_seqs)]

        # Pre-tokenize and create batches
        self.tokenized_batches = []
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i : i + self.batch_size]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_mask = (batch_tokens != alphabet.padding_idx).to(int)
            self.tokenized_batches.append((batch_tokens, batch_mask))

        self.nnsight_model = NNsight(self.model, device=self.device)
        self.layer_idx = self.layer_idx

        self.orig_loss, self.zero_loss = self._CE_for_orig_and_zero_ablation(
            self.tokenized_batches
        )
        print("Finished initializing ESM Fidelity Function")

    def _calculate_fidelity(self, sae_model) -> dict:
        sae_loss = self._CE_from_sae_recon(self.tokenized_batches, sae_model)

        loss_recovered = calculate_loss_recovered(
            ce_autoencoder=sae_loss,
            ce_identity=self.orig_loss,
            ce_zero_ablation=self.zero_loss,
        )

        return {"pct_loss_recovered": loss_recovered, "CE_w_sae_patching": sae_loss}

    def _CE_for_orig_and_zero_ablation(self, tokenized_batches):
        """Calculate cross entropy for original and zero-ablated outputs."""
        orig_losses, zero_losses = [], []

        for batch_tokens, batch_attn_mask in tqdm(tokenized_batches):
            batch_tokens = batch_tokens.to(self.device)
            batch_attn_mask = batch_attn_mask.to(self.device)

            orig_logits, orig_hidden = get_esm_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
            )

            zero_logits, _ = get_esm_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
                torch.zeros_like(orig_hidden),
            )

            orig_losses.extend(
                calculate_cross_entropy(orig_logits, batch_tokens, batch_attn_mask)
            )
            zero_losses.extend(
                calculate_cross_entropy(zero_logits, batch_tokens, batch_attn_mask)
            )

        return np.mean(orig_losses), np.mean(zero_losses)

    def _CE_from_sae_recon(self, tokenized_batches, sae_model):
        """Calculate cross entropy using SAE reconstructions."""
        sae_losses = []

        for batch_tokens, batch_attn_mask in tokenized_batches:
            batch_tokens = batch_tokens.to(self.device)
            batch_attn_mask = batch_attn_mask.to(self.device)

            _, orig_hidden = get_esm_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
            )

            # Get reconstructions with unnormalize=True for injection into the model
            reconstructions = sae_model(orig_hidden, unnormalize=True)
            sae_logits, _ = get_esm_output_with_intervention(
                self.model,
                self.nnsight_model,
                batch_tokens,
                batch_attn_mask,
                self.layer_idx,
                reconstructions,
            )

            sae_losses.extend(
                calculate_cross_entropy(sae_logits, batch_tokens, batch_attn_mask)
            )

        return np.mean(sae_losses)
