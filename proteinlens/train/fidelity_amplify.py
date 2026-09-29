from __future__ import annotations
"""
Loss (cross entropy) fidelity metric for an SAE trained on AMPLIFY embeddings.

Same metric as ``fidelity.py`` (ESM2): per-sequence CE of the model's logits
against the input tokens, excluding <bos>/<eos>, for (a) the unmodified model,
(b) the layer's residual stream zero-ablated, and (c) the residual stream
replaced by the SAE reconstruction; ``calculate_loss_recovered`` turns these
into % loss recovered.

The intervention is a plain forward hook on the transformer block whose output
is the SAE's training layer. ``layer_idx`` follows the ESM convention used
throughout the codebase -- "residual stream after block L" -- which for
AMPLIFY is ``model.transformer_encoder[L - 1]`` (== ``hidden_states[L - 1]``).
"""

from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from proteinlens.embedders.amplify import Amplify
from proteinlens.train.evaluation import EvaluationManager
from proteinlens.train.fidelity import (
    ESMFidelityConfig,
    ESMFidelityFunction,
    calculate_cross_entropy,
)
from proteinlens.utils import get_device


@dataclass
class AmplifyFidelityConfig(ESMFidelityConfig):
    # Subclasses ESMFidelityConfig so training_run's final evaluation, which
    # checks isinstance(..., ESMFidelityConfig), also runs for AMPLIFY.
    def build(self) -> "AmplifyFidelityFunction":
        return AmplifyFidelityFunction(self)


def additive_attention_mask(attn_mask: torch.Tensor) -> torch.Tensor | None:
    """0/1 tokenizer mask -> AMPLIFY's additive mask (0 attend, -inf pad), or None if unpadded."""
    mask = attn_mask.bool()
    if mask.all():
        return None
    return torch.where(mask, 0.0, float("-inf"))


class AmplifyFidelityFunction(ESMFidelityFunction):
    def __init__(self, eval_config: AmplifyFidelityConfig):
        # Skip ESMFidelityFunction.__init__ (fair-esm alphabet + nnsight).
        EvaluationManager.__init__(self, eval_config)

        print("Prepping AMPLIFY loss fidelity_fn")
        self.device = get_device()

        self.model_name = Amplify.MODEL_ALIASES.get(
            eval_config.model_name, eval_config.model_name
        )
        self.revision = Amplify.MODEL_REVISIONS.get(self.model_name)
        self.eval_seq_path = eval_config.eval_seq_path
        self.layer_idx = eval_config.layer_idx
        self.batch_size = eval_config.eval_batch_size or 8

        if eval_config.corrupt:
            raise NotImplementedError("'corrupt' is not supported for AMPLIFY")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, revision=self.revision
        )
        self.model = (
            AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True, revision=self.revision
            )
            .to(self.device)
            .eval()
        )

        n_layers = self.model.config.num_hidden_layers
        if not 1 <= self.layer_idx <= n_layers:
            raise ValueError(f"layer_idx {self.layer_idx} out of range [1, {n_layers}]")
        self.block = self.model.transformer_encoder[self.layer_idx - 1]

        with open(self.eval_seq_path, "r") as f:
            eval_seqs = [line.strip() for line in f if line.strip()]

        # Right padding, multiple of 8 for the xformers attention bias.
        self.tokenized_batches = []
        for i in range(0, len(eval_seqs), self.batch_size):
            enc = self.tokenizer(
                eval_seqs[i : i + self.batch_size],
                return_tensors="pt",
                padding=True,
                pad_to_multiple_of=8,
                truncation=False,
            )
            self.tokenized_batches.append((enc["input_ids"], enc["attention_mask"]))

        self.fidelity_enabled = True
        self.orig_loss, self.zero_loss = self._CE_for_orig_and_zero_ablation(
            self.tokenized_batches
        )
        print("Finished initializing AMPLIFY Fidelity Function")

    def _output_with_intervention(
        self,
        batch_tokens: torch.Tensor,
        batch_attn_mask: torch.Tensor,
        hidden_state_override: torch.Tensor | None = None,
    ):
        """Run the model, optionally replacing the layer's output; return (logits, original hidden)."""
        captured = {}

        def hook(module, inputs, output):
            captured["hidden"] = output[0]
            if hidden_state_override is not None:
                return (hidden_state_override,) + tuple(output[1:])

        handle = self.block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = self.model(
                    batch_tokens, attention_mask=additive_attention_mask(batch_attn_mask)
                )
        finally:
            handle.remove()
        return out.logits, captured["hidden"]

    def _CE_for_orig_and_zero_ablation(self, tokenized_batches):
        """Calculate cross entropy for original and zero-ablated outputs."""
        orig_losses, zero_losses = [], []

        for batch_tokens, batch_attn_mask in tqdm(tokenized_batches):
            batch_tokens = batch_tokens.to(self.device)
            batch_attn_mask = batch_attn_mask.to(self.device)

            orig_logits, orig_hidden = self._output_with_intervention(
                batch_tokens, batch_attn_mask
            )
            zero_logits, _ = self._output_with_intervention(
                batch_tokens, batch_attn_mask, torch.zeros_like(orig_hidden)
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

            _, orig_hidden = self._output_with_intervention(batch_tokens, batch_attn_mask)

            # Reconstructions with unnormalize=True for injection into the model
            reconstructions = sae_model(orig_hidden, unnormalize=True)
            sae_logits, _ = self._output_with_intervention(
                batch_tokens, batch_attn_mask, reconstructions
            )

            sae_losses.extend(
                calculate_cross_entropy(sae_logits, batch_tokens, batch_attn_mask)
            )

        return np.mean(sae_losses)
