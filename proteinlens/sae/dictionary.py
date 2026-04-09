from __future__ import annotations
"""
Defines the SAE classes.
(Originally built off of code from https://github.com/saprmarks/dictionary_learning/blob/2d586e417cd30473e1c608146df47eb5767e2527/dictionary.py)
"""

from abc import ABC, abstractmethod
from typing import Optional

import einops
import torch as t
import torch.nn as nn
import torch.nn.init as init

from proteinlens.utils import get_device


class Dictionary(ABC, nn.Module):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """

    dict_size: int  # number of features in the dictionary
    activation_dim: int  # dimension of the activation vectors

    def __init__(self, normalize_to_sqrt_d=False):
        super().__init__()
        self.normalize_to_sqrt_d = normalize_to_sqrt_d
    
    @property
    def has_normalization_factors(self) -> bool:
        """Check if normalization factors have been calculated (not just initialized to ones)."""
        if not hasattr(self, 'activation_rescale_factor'):
            return False
        # Check if factors are not all ones (which is the default initialization)
        return not t.allclose(self.activation_rescale_factor, t.ones_like(self.activation_rescale_factor))

    def _unnormalize_output(self, x_normalized, original_norms):
        """Un-normalize output back to original scale if normalization was applied."""
        if self.normalize_to_sqrt_d:
            # For √d scaling: x_original = x_normalized * original_norms / √d
            d = x_normalized.shape[-1]
            sqrt_d = t.sqrt(t.tensor(d, dtype=x_normalized.dtype, device=x_normalized.device))
            return x_normalized * original_norms / sqrt_d
        return x_normalized

    def _normalize_input_and_get_norms(self, x):
        """
        Apply Anthropic's √d normalization to input and return normalized input + original norms.
        If normalization is disabled, returns original input and None.
        """
        if self.normalize_to_sqrt_d:
            # Save original norms
            original_norms = t.norm(x, dim=-1, keepdim=True)
            original_norms = t.clamp(original_norms, min=1e-8)
            
            # Apply Anthropic's √d scaling: normalize to unit vectors, then scale by √d
            d = x.shape[-1]
            sqrt_d = t.sqrt(t.tensor(d, dtype=x.dtype, device=x.device))
            normalized = t.nn.functional.normalize(x, p=2, dim=-1) * sqrt_d
            
            return normalized, original_norms
        return x, None

    @abstractmethod
    def encode(self, x, normalize_features: bool = False):
        """
        Encode a vector x in the activation space.
        
        Args:
            x: Input activations to encode
            normalize_features: If True, divide features by activation_rescale_factor
        """
        pass

    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

    @abstractmethod
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        """
        Encode a vector x in the activation space using a subset of features.
        """
        pass

    @classmethod
    def from_pretrained(cls, path: str | None = None, device=None):
        """
        Load a pretrained dictionary from a file.
        """
        raise NotImplementedError(
            "from_pretrained not implemented for this dictionary class"
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Call parent class load method first
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # If activation_rescale_factor wasn't in state dict, it will be in missing_keys
        # Remove it from missing_keys since we handle it here
        norm_factor_key = prefix + "activation_rescale_factor"
        if norm_factor_key in missing_keys:
            missing_keys.remove(norm_factor_key)
            # activation_rescale_factor buffer was already initialized to ones in __init__

        # Load normalize_to_sqrt_d parameter if it exists
        norm_key = prefix + "normalize_to_sqrt_d"
        if norm_key in state_dict:
            self.normalize_to_sqrt_d = state_dict[norm_key].item()
            # Remove from unexpected_keys if it's there
            if norm_key in unexpected_keys:
                unexpected_keys.remove(norm_key)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Save the normalize_to_sqrt_d parameter to state dict."""
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "normalize_to_sqrt_d"] = t.tensor(self.normalize_to_sqrt_d, dtype=t.bool)
    
    def _make_contiguous(self):
        """Make all parameters contiguous (called automatically during init)."""
        for param in self.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()


class ReLUSAE(Dictionary):
    """
    ReLU SAE with separate pre-encoding bias and untied encoder/decoder weights.
    This is the architecture put forward in https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder
    Architecture:
    - Pre-encoding bias applied before encoder
    - Decoder has no bias
    - Independent (untied) encoder and decoder weights
    - Decoder weights are unit-normed
    """

    def __init__(self, activation_dim, dict_size, normalize_to_sqrt_d=False):
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

        self.register_buffer("activation_rescale_factor", t.ones(dict_size))
        self._make_contiguous()

    def encode(self, x, normalize_features: bool = False):
        features = nn.ReLU()(self.encoder(x - self.bias))
        if normalize_features:
            features /= self.activation_rescale_factor
        return features

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, ghost_mask=None, unnormalize=False):
        """
        Forward pass of an autoencoder.

        Args:
            x: activations to be autoencoded (unnormalized)
            output_features: if True, return the encoded features as well as the decoded x
            ghost_mask: if not None, run this autoencoder in "ghost mode" where features are masked
            unnormalize: if True, un-normalize output to original space (for injection/steering).
                        if False (default), keep output in normalized √d space (for training/analysis)
        """
        # Apply unit normalization and get original norms (calculated once)
        x, original_norms = self._normalize_input_and_get_norms(x)

        if ghost_mask is None:  # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)

            # Only un-normalize if explicitly requested (for fidelity/steering)
            if unnormalize:
                x_hat = self._unnormalize_output(x_hat, original_norms)

            if output_features:
                return x_hat, f
            else:
                return x_hat

        else:  # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)

            x_ghost = self.decoder(
                f_ghost
            )  # note that this only applies the decoder weight matrix, no bias
            x_hat = self.decode(f)

            # Un-normalize both outputs
            x_ghost = self._unnormalize_output(x_ghost, original_norms)
            x_hat = self._unnormalize_output(x_hat, original_norms)

            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost

    @t.no_grad()
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        encoder_w_subset = self.encoder.weight[feat_list, :]
        encoder_b_subset = self.encoder.bias[feat_list]
        x, _ = self._normalize_input_and_get_norms(x)
        features = t.nn.ReLU()((x - self.bias) @ encoder_w_subset.T + encoder_b_subset)
        if normalize_features:
            features /= self.activation_rescale_factor[feat_list]
        return features

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        if device is None:
            device = get_device()
        state_dict = t.load(path, map_location=device)
        dict_size, activation_dim = state_dict["encoder.weight"].shape

        # Check if normalize_to_sqrt_d was saved in the state dict
        normalize_to_sqrt_d = False
        if "normalize_to_sqrt_d" in state_dict:
            normalize_to_sqrt_d = state_dict["normalize_to_sqrt_d"].item()

        autoencoder = ReLUSAE(activation_dim, dict_size, normalize_to_sqrt_d=normalize_to_sqrt_d)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder = autoencoder.to(device)
        return autoencoder



class JumpReLUSAE(Dictionary, nn.Module):
    """
    An autoencoder with jump ReLUs.
    """

    def __init__(self, activation_dim, dict_size, device="cpu", normalize_to_sqrt_d=False):
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.W_enc = nn.Parameter(t.empty(activation_dim, dict_size, device=device))
        self.b_enc = nn.Parameter(t.zeros(dict_size, device=device))
        self.W_dec = nn.Parameter(
            t.nn.init.kaiming_uniform_(
                t.empty(dict_size, activation_dim, device=device)
            )
        )
        self.b_dec = nn.Parameter(t.zeros(activation_dim, device=device))
        self.threshold = nn.Parameter(
            t.ones(dict_size, device=device) * 0.001
        )  # Appendix I

        self.apply_b_dec_to_input = False

        self.W_dec.data = self.W_dec / self.W_dec.norm(dim=1, keepdim=True)
        self.W_enc.data = self.W_dec.data.clone().T

        self.register_buffer("activation_rescale_factor", t.ones(dict_size))
        self._make_contiguous()

    def encode(self, x, output_pre_jump=False, normalize_features: bool = False):
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        pre_jump = x @ self.W_enc + self.b_enc

        f = nn.ReLU()(pre_jump * (pre_jump > self.threshold))
        
        if normalize_features:
            f /= self.activation_rescale_factor

        if output_pre_jump:
            return f, pre_jump
        else:
            return f

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded (unnormalized)
        output_features : if True, return the encoded features (and their pre-jump version) as well as the decoded x
        """
        # Apply unit normalization and get original norms (calculated once)
        x, original_norms = self._normalize_input_and_get_norms(x)
        
        f = self.encode(x)
        x_hat = self.decode(f)

        # Un-normalize the output
        x_hat = self._unnormalize_output(x_hat, original_norms)

        if output_features:
            return x_hat, f
        else:
            return x_hat

    def scale_biases(self, scale: float):
        self.b_dec.data *= scale
        self.b_enc.data *= scale
        self.threshold.data *= scale

    @t.no_grad()
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        pre_jump = x @ self.W_enc[:, feat_list] + self.b_enc[feat_list]
        features = nn.ReLU()(pre_jump * (pre_jump > self.threshold[feat_list]))
        if normalize_features:
            features /= self.activation_rescale_factor[feat_list]
        return features

    @classmethod
    def from_pretrained(
        cls,
        path: str | None = None,
        load_from_sae_lens: bool = False,
        dtype: t.dtype = t.float32,
        device: t.device | None = None,
        **kwargs,
    ):
        """
        Load a pretrained autoencoder from a file.
        If sae_lens=True, then pass **kwargs to sae_lens's
        loading function.
        """
        state_dict = t.load(path, map_location=device)
        activation_dim, dict_size = state_dict["W_enc"].shape
        
        # Check if normalize_to_sqrt_d was saved in the state dict
        normalize_to_sqrt_d = False
        if "normalize_to_sqrt_d" in state_dict:
            normalize_to_sqrt_d = state_dict["normalize_to_sqrt_d"].item()
        
        autoencoder = JumpReLUSAE(activation_dim, dict_size, normalize_to_sqrt_d=normalize_to_sqrt_d)
        autoencoder.load_state_dict(state_dict)
        autoencoder = autoencoder.to(dtype=dtype, device=device)

        if device is not None:
            device = autoencoder.W_enc.device
        return autoencoder.to(dtype=dtype, device=device)

class MatryoshkaBatchTopKSAE(Dictionary, nn.Module):
    def __init__(
        self, activation_dim: int, dict_size: int, k: int, group_sizes: list[int], normalize_to_sqrt_d=False
    ):
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert sum(group_sizes) == dict_size, "group sizes must sum to dict_size"
        assert all(s > 0 for s in group_sizes), "all group sizes must be positive"

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

        self.active_groups = len(group_sizes)
        group_indices = [0] + list(t.cumsum(t.tensor(group_sizes), dim=0))
        self.group_indices = group_indices

        self.register_buffer("group_sizes", t.tensor(group_sizes))

        self.W_enc = nn.Parameter(t.empty(activation_dim, dict_size))
        self.b_enc = nn.Parameter(t.zeros(dict_size))
        self.W_dec = nn.Parameter(
            t.nn.init.kaiming_uniform_(t.empty(dict_size, activation_dim))
        )
        self.b_dec = nn.Parameter(t.zeros(activation_dim))

        # We must transpose because we are using nn.Parameter, not nn.Linear
        self.W_dec.data = set_decoder_norm_to_unit_norm(
            self.W_dec.data.T, activation_dim, dict_size
        ).T
        self.W_enc.data = self.W_dec.data.clone().T
        # Initialize normalization factor buffer with ones
        self.register_buffer("activation_rescale_factor", t.ones(dict_size))
        self._make_contiguous()
    def encode(
        self, x: t.Tensor, return_active: bool = False, use_threshold: bool = True, normalize_features: bool = False
    ):
        post_relu_feat_acts_BF = nn.functional.relu(
            (x - self.b_dec) @ self.W_enc + self.b_enc
        )

        if use_threshold:
            encoded_acts_BF = post_relu_feat_acts_BF * (
                post_relu_feat_acts_BF > self.threshold
            )
        else:
            # Flatten and perform batch top-k
            flattened_acts = post_relu_feat_acts_BF.flatten()
            post_topk = flattened_acts.topk(self.k * x.size(0), sorted=False, dim=-1)

            encoded_acts_BF = (
                t.zeros_like(post_relu_feat_acts_BF.flatten())
                .scatter_(-1, post_topk.indices, post_topk.values)
                .reshape(post_relu_feat_acts_BF.shape)
            )

        max_act_index = self.group_indices[self.active_groups]
        encoded_acts_BF[:, max_act_index:] = 0
        if normalize_features:
            encoded_acts_BF /= self.activation_rescale_factor
        if return_active:
            return encoded_acts_BF, encoded_acts_BF.sum(0) > 0, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        return x @ self.W_dec + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False, unnormalize: bool = False):
        """
        Forward pass of an autoencoder.

        Args:
            x: activations to be autoencoded (unnormalized)
            output_features: if True, return the encoded features as well as the decoded x
            unnormalize: if True, un-normalize output to original space (for injection/steering).
                        if False (default), keep output in normalized √d space (for training/analysis)
        """
        # Apply unit normalization and get original norms (calculated once)
        x, original_norms = self._normalize_input_and_get_norms(x)

        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
         # Only un-normalize if explicitly requested (for fidelity/steering)
        if unnormalize:
            x_hat_BD = self._unnormalize_output(x_hat_BD, original_norms)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    @t.no_grad()
    def scale_biases(self, scale: float):
        self.b_enc.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @t.no_grad()
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        encoded_acts_BF = t.nn.ReLU()(
            (x - self.b_dec) @ self.W_enc[:, feat_list] + self.b_enc[feat_list]
        )
        encoded_acts_BF = encoded_acts_BF * (encoded_acts_BF > self.threshold)

        if normalize_features:
            encoded_acts_BF /= self.activation_rescale_factor[feat_list]
        return encoded_acts_BF
    
    @classmethod
    def from_pretrained(
        cls,
        path: str | None = None,
        k: int | None = None,
        threshold: float | None = None,
        device=None,
    ):
        """
        Load a pretrained dictionary from a file.
        """
        if device is None:
            device = get_device()
        state_dict = t.load(path, map_location=device)
        if k is None:
            print(f"Loading k={state_dict['k'].item()} from state dict")
            k = state_dict["k"].item()
        else:
            assert (
                k == state_dict["k"].item()
            ), f"k={k} != {state_dict['k'].item()}=state_dict['k']"

        if threshold is None:
            print(f"Loading threshold={state_dict['threshold'].item()} from state dict")
            threshold = state_dict["threshold"].item()
        else:
            assert (
                threshold == state_dict["threshold"].item()
            ), f"threshold={threshold} != {state_dict['threshold'].item()}=state_dict['threshold']"

        activation_dim, dict_size = state_dict["W_enc"].shape
                
        group_sizes = state_dict["group_sizes"].tolist()

        normalize_to_sqrt_d = state_dict.get("normalize_to_sqrt_d", t.tensor(False)).item()

        autoencoder = MatryoshkaBatchTopKSAE(activation_dim=activation_dim, dict_size=dict_size, k=k, group_sizes=group_sizes, normalize_to_sqrt_d=normalize_to_sqrt_d)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder

class IdentityDict(Dictionary, nn.Module):
    """
    An identity dictionary, i.e. the identity function. This is useful for treating neurons as features.
    """

    def __init__(self, activation_dim=None, normalize_to_sqrt_d=False):
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = activation_dim

    def encode(self, x, normalize_features: bool = False):
        if normalize_features:
            x = x / self.activation_rescale_factor
        return x

    def decode(self, f):
        return f

    @t.no_grad()
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        features = x[:, feat_list]

        if normalize_features:
            features /= self.activation_rescale_factor[feat_list]

        return features

    def forward(self, x, output_features=False, ghost_mask=None):
        if output_features:
            return x, x
        else:
            return x


# The next two functions could be replaced with the ConstrainedAdam Optimizer
@t.no_grad()
def set_decoder_norm_to_unit_norm(
    W_dec_DF: t.nn.Parameter, activation_dim: int, d_sae: int
) -> t.Tensor:
    """There's a major footgun here: we use this with both nn.Linear and nn.Parameter decoders.
    nn.Linear stores the decoder weights in a transposed format (d_model, d_sae). So, we pass the dimensions in
    to catch this error."""

    D, F = W_dec_DF.shape

    assert D == activation_dim
    assert F == d_sae

    eps = t.finfo(W_dec_DF.dtype).eps
    norm = t.norm(W_dec_DF.data, dim=0, keepdim=True)
    W_dec_DF.data /= norm + eps
    return W_dec_DF.data


@t.no_grad()
def remove_gradient_parallel_to_decoder_directions(
    W_dec_DF: t.Tensor,
    W_dec_DF_grad: t.Tensor,
    activation_dim: int,
    d_sae: int,
) -> t.Tensor:
    """There's a major footgun here: we use this with both nn.Linear and nn.Parameter decoders.
    nn.Linear stores the decoder weights in a transposed format (d_model, d_sae). So, we pass the dimensions in
    to catch this error."""

    D, F = W_dec_DF.shape
    assert D == activation_dim
    assert F == d_sae

    normed_W_dec_DF = W_dec_DF / (t.norm(W_dec_DF, dim=0, keepdim=True) + 1e-6)

    parallel_component = einops.einsum(
        W_dec_DF_grad,
        normed_W_dec_DF,
        "d_in d_sae, d_in d_sae -> d_sae",
    )
    W_dec_DF_grad -= einops.einsum(
        parallel_component,
        normed_W_dec_DF,
        "d_sae, d_in d_sae -> d_in d_sae",
    )
    return W_dec_DF_grad
