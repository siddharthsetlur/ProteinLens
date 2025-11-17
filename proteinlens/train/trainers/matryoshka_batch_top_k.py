import torch as t
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
# import einops
from collections import namedtuple
from typing import Optional
from math import isclose
from proteinlens.sae.dictionary import (
    MatryoshkaBatchTopKSAE,
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
)
from proteinlens.train.trainers.base_trainer import SAETrainer, SAETrainerConfig
from proteinlens.train.trainers.common import get_lr_schedule, get_autocast_context, geometric_median
from proteinlens.utils import get_device


def apply_temperature(probabilities: list[float], temperature: float) -> list[float]:
    """
    Apply temperature scaling to a list of probabilities using PyTorch.

    Args:
        probabilities (list[float]): Initial probability distribution
        temperature (float): Temperature parameter (> 0)

    Returns:
        list[float]: Scaled and normalized probabilities
    """
    probs_tensor = t.tensor(probabilities, dtype=t.float32)
    logits = t.log(probs_tensor)
    scaled_logits = logits / temperature
    scaled_probs = t.nn.functional.softmax(scaled_logits, dim=0)

    return scaled_probs.tolist()

@dataclass
class MatryoshkaBatchTopKTrainerConfig(SAETrainerConfig):
    # Top-K parameters
    k: int = 10
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000
    k_anneal_steps: Optional[int] = None
    
    # Matryoshka-specific parameters
    group_fractions: list[float] = None
    group_weights: Optional[list[float]] = None
    
    # Dead feature threshold
    dead_feature_threshold: int = 10_000_000
    
    def __post_init__(self):
        # # Auto-compute LR if not provided
        # if self.lr is None:
        #     scale = self.activation_dim * self.expansion_factor / (2**14)
        #     self.lr = 2e-4 / scale**0.5
        
        # # === CRITICAL GROUP CALCULATION LOGIC ===
        # if self.dictionary_size is None:
        #     self.dict_size = self.activation_dim * self.expansion_factor
        # Validation: fractions must sum to 1.0
        assert isclose(sum(self.group_fractions), 1.0), (
            "group_fractions must sum to 1.0"
        )
        
        # Calculate all groups EXCEPT the last one
        group_sizes = [int(f * self.dictionary_size) for f in self.group_fractions[:-1]]
        
        # Put REMAINDER in the last group (handles rounding errors)
        group_sizes.append(self.dictionary_size - sum(group_sizes))
        
        self.group_sizes = group_sizes
        
        # Default group weights to equal if not provided
        if self.group_weights is None:
            self.group_weights = [(1.0 / len(group_sizes))] * len(group_sizes)
        
        # Validation: weights must match number of groups
        assert len(self.group_sizes) == len(self.group_weights), (
            "group_sizes and group_weights must have the same length"
        )
    
    def trainer_cls(self) -> type["MatryoshkaBatchTopKTrainer"]:
        return MatryoshkaBatchTopKTrainer



class MatryoshkaBatchTopKTrainer(SAETrainer):
    """Trainer for Matryoshka Batch Top-K Sparse Autoencoders.
    
    This trainer supports nested dictionary structures where multiple dictionary
    sizes are trained simultaneously. It uses cumulative reconstruction across
    groups to train features at different scales.
    """
    
    def __init__(self, trainer_config: MatryoshkaBatchTopKTrainerConfig):
        super().__init__(
            trainer_config,
            logging_parameters=[
                "training/learning_rate",
                "training/threshold",
                "features/total_dead",
                "features/effective_l0",
                "loss/pre_norm_auxk",
                "loss/min_l2",
                "loss/max_l2",
            ],
        )
        
        # Training parameters
        self.lr = trainer_config.lr
        self.steps = trainer_config.steps
        self.decay_start = trainer_config.decay_start
        self.warmup_steps = trainer_config.warmup_steps
        # self.grad_clip_norm = trainer_config.grad_clip_norm
        
        # Top-K parameters
        self.k = trainer_config.k
        self.threshold_beta = trainer_config.threshold_beta
        self.threshold_start_step = trainer_config.threshold_start_step
        self.k_anneal_steps = trainer_config.k_anneal_steps
        
        # Matryoshka-specific parameters
        self.group_fractions = trainer_config.group_fractions
        self.group_sizes = trainer_config.group_sizes
        self.group_weights = trainer_config.group_weights
        
        # Create the Matryoshka SAE
        self.ae = MatryoshkaBatchTopKSAE(
            activation_dim=trainer_config.activation_dim,
            dict_size=trainer_config.dictionary_size,
            k=trainer_config.k,
            group_sizes=self.group_sizes,
            normalize_to_sqrt_d=trainer_config.normalize_to_sqrt_d,
        )
        
        self.device = get_device()
        self.ae.to(self.device)
        
        # Auxiliary loss parameters
        self.auxk_alpha = trainer_config.auxk_alpha
        self.dead_feature_threshold = trainer_config.dead_feature_threshold
        self.top_k_aux = trainer_config.activation_dim // 2  # Heuristic from paper
        
        # Dead feature tracking
        self.num_tokens_since_fired = t.zeros(
            self.ae.dict_size, dtype=t.long, device=self.device
        )
        self.steps_since_active = t.zeros(
            self.ae.dict_size, dtype=t.long, device=self.device
        )
        # namespace for logging
        setattr(self, "features/total_dead", -1)

        self.optimizer = t.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )
        lr_fn = get_lr_schedule(
            self.steps,
            self.warmup_steps,
            decay_start=self.decay_start,
        )
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def update_annealed_k(
        self, step: int, activation_dim: int, k_anneal_steps: Optional[int] = None
         ) -> None:
        """Update k buffer in-place with annealed value"""
        if k_anneal_steps is None:
            return

        assert 0 <= k_anneal_steps < self.steps, (
            "k_anneal_steps must be >= 0 and < steps."
        )
        assert activation_dim > self.k, "activation_dim must be greater than k"

        step = min(step, k_anneal_steps)
        ratio = step / k_anneal_steps
        annealed_value = activation_dim * (1 - ratio) + self.k * ratio

        # Update in-place
        self.ae.k.fill_(int(annealed_value))
    def update_threshold(self, f: t.Tensor):
        with get_autocast_context(self.device, enabled=False), t.no_grad():
            active = f[f > 0]

            if active.size(0) == 0:
                min_activation = 0.0
            else:
                min_activation = active.min().detach().to(dtype=t.float32)

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )  
    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        """
        Compute the auxiliary loss for the batch top-k SAE.
        """
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        setattr(self, "features/total_dead", int(dead_features.sum()))

        if dead_features.sum() > 0:
            k_aux = min(self.top_k_aux, dead_features.sum())

            # Only look at activations of dead features, mask others to -inf
            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            # Get top-k dead feature activations
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Create sparse tensor with only the top-k dead activations
            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )

            # Use decoder method instead of direct matrix multiplication
            x_reconstruct_aux = self.ae.W_dec(auxk_acts_BF)
            
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            setattr(self, "loss/pre_norm_auxk", l2_loss_aux.item())

            # Normalize by variance of residual to make loss scale-invariant
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(
                residual_BD.shape
            )
            loss_denom = (
                (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            setattr(self, "loss/pre_norm_auxk", -1)
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    @property
    def threshold(self):
        return self.ae.threshold
        
    # Property accessors for namespaced logging parameters
    @property 
    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]
    
    def loss(self, x, step=None, logging=False):
        # The SAE model handles normalization internally
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(
            x, return_active=True, use_threshold=False
        )

        if step > self.threshold_start_step:
            self.update_threshold(f)

        # Initialize reconstruction with bias
        x_reconstruct = t.zeros_like(x) + self.ae.b_dec
        total_l2_loss = 0.0
        l2_losses = t.tensor([]).to(self.device)

        # MATRYOSHKA: Split decoder and activations by groups
        W_dec_chunks = t.split(self.ae.W_dec, self.ae.group_sizes.tolist(), dim=0)
        f_chunks = t.split(f, self.ae.group_sizes.tolist(), dim=1)

        # Cumulative reconstruction across nested groups
        for i in range(self.ae.active_groups):
            W_dec_slice = W_dec_chunks[i]
            acts_slice = f_chunks[i]
            
            # Add this group's contribution to reconstruction
            x_reconstruct = x_reconstruct + acts_slice @ W_dec_slice

            # Compute loss with groups 0...i
            l2_loss = (x - x_reconstruct).pow(2).sum(dim=-1).mean()
            weighted_loss = l2_loss * self.group_weights[i]
            total_l2_loss += weighted_loss
            l2_losses = t.cat([l2_losses, l2_loss.unsqueeze(0)])

        min_l2_loss = l2_losses.min().item()
        max_l2_loss = l2_losses.max().item()
        mean_l2_loss = l2_losses.mean()

        # Update logging attributes
        setattr(self, "features/effective_l0", self.k)
        setattr(self, "loss/min_l2", min_l2_loss)
        setattr(self, "loss/max_l2", max_l2_loss)

        # Track dead features
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        # Update steps since active (additional tracking)
        self.steps_since_active += 1
        self.steps_since_active[did_fire] = 0

        # Compute auxiliary loss on detached residual
        auxk_loss = self.get_auxiliary_loss(
            (x - x_reconstruct).detach(), post_relu_acts_BF
        )
        loss = mean_l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            # Namespaced loss dictionary
            loss_dict = {
                "loss/reconstruction_mean": mean_l2_loss.item(),
                "loss/reconstruction_min": min_l2_loss,
                "loss/reconstruction_max": max_l2_loss,
                "loss/auxiliary": auxk_loss.item(),
                "loss/total": loss.item(),
            }

            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x, x_reconstruct, f, loss_dict
            )
    def update(self, step, x):
        # Initialize bias on first step (removed - handled in parent class)
        # if step == 0:
        #     median = self.geometric_median(x)
        #     self.ae.b_dec.data = median

        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # We must transpose because we are using nn.Parameter, not nn.Linear
        self.ae.W_dec.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.W_dec.T,
            self.ae.W_dec.grad.T,
            self.ae.activation_dim,
            self.ae.dict_size,
        ).T
        # if self.grad_clip_norm is not None:
        #     t.nn.utils.clip_grad_norm_(self.ae.parameters(), self.grad_clip_norm)


        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.update_annealed_k(step, self.ae.activation_dim, self.k_anneal_steps)

        # # Renormalize decoder
        # if hasattr(self.ae, 'decoder'):  # If using nn.Linear
        #     self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
        #         self.ae.decoder.weight,
        #         self.ae.activation_dim,
        #         self.ae.dict_size,
        #     )
        # else:  # If using nn.Parameter (W_dec)
        #     self.ae.W_dec.data = set_decoder_norm_to_unit_norm(
        #         self.ae.W_dec.T,
        #         self.ae.activation_dim,
        #         self.ae.dict_size,
        #     ).T
        # Make sure the decoder is still unit-norm
        # We must transpose because we are using nn.Parameter, not nn.Linear
        self.ae.W_dec.data = set_decoder_norm_to_unit_norm(
            self.ae.W_dec.T, self.ae.activation_dim, self.ae.dict_size
        ).T

        return loss.item()
    def get_per_dimension_mse(self, x):
        """
        Calculate per-dimension MSE for final analysis.
        
        Args:
            x: Input activations to evaluate
            
        Returns:
            torch.Tensor: Per-dimension MSE, shape (activation_dim,)
        """
        x = x.to(self.device)
        with t.no_grad():
            # Get reconstruction
            f, _, _ = self.ae.encode(x, return_active=True, use_threshold=True)
            x_hat = self.ae.decode(f)
            
            # Calculate per-dimension squared error, averaged across batch
            per_dim_mse = (x - x_hat).pow(2).mean(dim=0)
            
        return per_dim_mse
    @classmethod
    def dictionary_cls(cls):
        return MatryoshkaBatchTopKSAE
