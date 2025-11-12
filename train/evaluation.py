from dataclasses import dataclass
from pathlib import Path

import torch
import torch as t

from interplm.sae.dictionary import Dictionary
from interplm.train.data_loader import DataloaderConfig, ShardedActivationsDataset


@dataclass
class EvaluationConfig:
    # List of sequences to evaluate fidelity on
    eval_seq_path: Path | None = None
    # Directory of embeddings to calculate eval metrics on
    eval_embd_dir: Path | None = None
    # Batch size for evaluation
    eval_batch_size: int | None = None
    # Steps to evaluate on
    eval_steps: int | None = None
    # Normalization files for evaluation data (should match training)
    zscore_means_file: Path | None = None
    zscore_vars_file: Path | None = None
    target_dtype: torch.dtype = torch.float32

    def build(self) -> "EvaluationManager":
        return EvaluationManager(self)


class EvaluationManager:
    def __init__(self, eval_config: EvaluationConfig):
        self.config = eval_config
        self.eval_steps = eval_config.eval_steps
        self.eval_seq_path = eval_config.eval_seq_path
        self.eval_embd_dir = eval_config.eval_embd_dir
        self.eval_batch_size = eval_config.eval_batch_size

        self.eval_activations = (
            DataloaderConfig(
                plm_embd_dir=self.eval_embd_dir,
                batch_size=self.eval_batch_size,
                zscore_means_file=eval_config.zscore_means_file,
                zscore_vars_file=eval_config.zscore_vars_file,
                target_dtype=eval_config.target_dtype,
            ).build()
            if self.eval_embd_dir is not None
            else None
        )

    def _calculate_fidelity(self, features: t.Tensor):
        """By default, we don't calculate fidelity (subclass should override)"""
        return None

    def _should_run_evals_on_valid(self, step):
        return self.eval_embd_dir is not None and step % self.eval_steps == 0

    def calculate_monitoring_metrics(
        self,
        features: t.Tensor,
        activations: t.Tensor,
        reconstructions: t.Tensor,
        sae_model: Dictionary,
    ):
        # Use namespaced metric names from the start
        metrics = {
            "performance/variance_explained": self._calculate_variance_explained(
                activations, reconstructions
            ),
        }
        
        # Only include l0 sparsity for non-BatchTopK SAEs (where it's meaningful)
        # BatchTopK SAE always has exactly k active features
        if not hasattr(sae_model, 'k'):  # Not a BatchTopK SAE
            metrics["performance/l0_sparsity"] = self._calculate_sparsity(features)
            
        # Only include fidelity if it's actually calculated (not None)
        fidelity = self._calculate_fidelity(sae_model)
        if fidelity is not None:
            metrics["performance/fidelity"] = fidelity

        return metrics

    def _calculate_sparsity(self, features):
        """Calculate sparsity-related metrics from feature activations.

        Args:
            features: Feature activations tensor from encoder

        Returns:
            dict: Dictionary of sparsity metrics
        """
        n_nonzero_per_example = (features != 0).float().sum(dim=-1)
        return n_nonzero_per_example.mean().item()

    def _calculate_variance_explained(self, activations, reconstructed):
        """Calculate variance-related metrics comparing original and reconstructed activations.

        Args:
            activations: Original input activations
            reconstructed: Reconstructed activations from autoencoder

        Returns:
            dict: Dictionary of variance metrics
        """
        total_variance = t.var(activations, dim=0).sum()
        residual_variance = t.var(activations - reconstructed, dim=0).sum()
        return (1 - residual_variance / total_variance).item()
