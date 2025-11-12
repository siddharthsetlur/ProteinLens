from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Union

import h5py
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from torch import Tensor
from tqdm import tqdm
import shutil

from proteinlens.sae.dictionary import ReLUSAE, Dictionary
from proteinlens.train.configs import TrainingRunConfig
from proteinlens.utils import get_device

warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

def load_sae(
    model_dir: Union[str, Path], model_name: str = "ae.pt", device: Optional[str] = None
) -> Dictionary:
    """
    Load a pretrained SAE model in inference mode.

    :param model_path: Path to the saved model state dictionary
    :param device: Target device for model computation ('cpu', 'cuda', etc.).
                  If None, automatically determines the best available device
    :return: Loaded SAE model in eval mode with gradients disabled

    This function:
    1. Loads the model state from disk
    2. Reconstructs the SAE architecture based on the saved config
    3. Moves the model to the specified device
    4. Sets the model to evaluation mode
    5. Disables gradient computation for all parameters
    """
    if device is None:
        device = get_device()
    model_dir = Path(model_dir)

    # Load state dict to the target device
    state_dict = torch.load(
        model_dir / model_name, map_location=torch.device(device), weights_only=True
    )

    config = TrainingRunConfig.from_yaml(model_dir / "config.yaml")
    ae_cls = config.trainer_cfg.trainer_cls().dictionary_cls()

    # Initialize and configure the model
    autoencoder = ae_cls.from_pretrained(model_dir / model_name, device=device)

    return autoencoder


def get_sae_feats_in_batches(
    sae: Dictionary,
    device: str,
    aa_embds: np.ndarray | Tensor,
    chunk_size: int,
    feat_list: Optional[List[int]] = None,
    scale_to_100_in_int8: bool = False,
    normalize_features: bool = False,
) -> Tensor:
    """
    Process large embedding arrays in chunks to generate SAE features.

    :param sae: Trained SAE model
    :param device: Device to perform computations on ('cpu', 'cuda', etc.)
    :param aa_embds: NumPy array of amino acid embeddings to process
    :param chunk_size: Number of embeddings to process in each batch
    :param feat_list: List of feature indices to encode. If None, uses all features
    :return: Tensor containing encoded features for all input embeddings

    This function:
    1. Defaults to using all features if feat_list is None
    2. Processes embeddings in batches to manage memory usage
    3. Shows progress using tqdm
    4. Concatenates all processed batches into a single tensor
    """
    # Use all features if none specified
    if feat_list is None:
        feat_list = list(range(sae.dict_size))

    # Convert input to tensor on specified device
    try:
        if torch.is_tensor(aa_embds):
            aa_embds = aa_embds.to(device)
        elif isinstance(aa_embds, list) and all(torch.is_tensor(x) for x in aa_embds):
            aa_embds = torch.stack(aa_embds, dim=0).to(device)
        else:
            aa_embds = torch.tensor(aa_embds, device=device)
    except Exception as e:
        print(f"Error converting input to tensor: {e}")
        raise e

    all_features = []

    # Process in chunks with progress bar
    for i in range(0, len(aa_embds), chunk_size):
        chunk = aa_embds[i : i + chunk_size]
        features = sae.encode_feat_subset(
            chunk, feat_list, normalize_features=normalize_features
        )
        # if scale_to_100_in_int8:
        #     features = (features * 100).to(torch.int8)
        all_features.append(features)

    # Combine all processed chunks
    all_features = torch.vstack(all_features)
    return all_features


def split_up_feature_list(total_features, max_feature_chunk_size: int = 2560):
    feature_chunk_size = min(max_feature_chunk_size, total_features)
    num_chunks = int(np.ceil(total_features / feature_chunk_size))

    return np.array_split(range(total_features), num_chunks)
