"""
Data loader utilities for sharded protein language model (PLM) activations.

This module provides classes for efficiently loading, batching, and optionally
unnormalizing sharded PLM activation data from disk. It supports both PyTorch
tensor files (.pt) and memory-mapped files (.dat), and can handle z-score
unnormalization using provided statistics. The main entry point is the
`ActivationsDataLoader`, which is configured via `DataloaderConfig`.

Key Classes:
- DataloaderConfig: Configuration for the data loader, including paths, batch size, normalization, etc.
- ZScoreUnnormalizer: Handles unnormalization of z-score normalized activations.
- ActivationsDataLoader: PyTorch DataLoader for batched activations.
- ShardedActivationsDataset: ActivationsDataLoader that loads activations from multiple shards.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import numpy as np
from interplm.utils import get_device


@dataclass
class DataloaderConfig:
    """
    Configuration for the ActivationsDataLoader.

    Args:
        plm_embd_dir (Path): Path to the root directory containing sharded PLM activations.
        batch_size (int, optional): Number of samples per batch. Default is 512.
        seed (int, optional): Random seed for shuffling and reproducibility. Default is 0.
        samples_to_skip (int, optional): Number of initial samples to skip in the dataset (useful for resuming). Default is 0.
        n_shards_to_include (int, optional): If set, only include the first N shards from the directory. Default is None (all shards).
        zscore_means_file (Path or None, optional): Path to file containing means for z-score unnormalization. Default is None.
        zscore_vars_file (Path or None, optional): Path to file containing variances for z-score unnormalization. Default is None.
        target_dtype (torch.dtype, optional): Data type to which activations are cast after unnormalization. Default is torch.float32.
    """
    plm_embd_dir: Path
    batch_size: int = 512
    seed: int = 0
    samples_to_skip: int = 0
    n_shards_to_include: int = None
    zscore_means_file: Path | None = None
    zscore_vars_file: Path | None = None
    target_dtype: torch.dtype = torch.float32

    def build(self) -> "ActivationsDataLoader":
        return ActivationsDataLoader(self)


class ZScoreUnnormalizer:
    """Handles unnormalization of z-score normalized activation data back to original distribution.
    
    This is relevant if you want to store your activations in a low precision format (e.g. fp16) to save memory
    then unnormalize them back to fp32 for training or analysis, but they have large variance such that converting
    each dimension individually makes this process less lossy.
    """
    
    def __init__(
        self,
        zscore_means_file: Path | None = None,
        zscore_vars_file: Path | None = None,  
        target_dtype: torch.dtype = torch.float32,
    ):
        self.means = None
        self.stds = None
        self.target_dtype = target_dtype
        
        if zscore_means_file is not None and zscore_vars_file is not None:
            self._load_zscore_stats(zscore_means_file, zscore_vars_file)
        
    
    def _load_zscore_stats(self, zscore_means_file: Path, zscore_vars_file: Path):
        """Load z-score means and variances from files, convert variances to standard deviations."""
        print(f"Loading z-score unnormalization stats from {zscore_means_file} and {zscore_vars_file}")
        
        # Try different file formats for means
        if zscore_means_file.suffix == '.pt':
            self.means = torch.load(zscore_means_file, map_location='cpu')
        elif zscore_means_file.suffix in ['.npy', '.npz']:
            self.means = torch.from_numpy(np.load(str(zscore_means_file)))
        else:
            # Assume text file with one number per line
            with open(zscore_means_file, 'r') as f:
                self.means = torch.tensor([float(line.strip()) for line in f])
        
        # Load variances and convert to standard deviations
        if zscore_vars_file.suffix == '.pt':
            vars_data = torch.load(zscore_vars_file, map_location='cpu')
        elif zscore_vars_file.suffix in ['.npy', '.npz']:
            vars_data = torch.from_numpy(np.load(str(zscore_vars_file)))
        else:
            # Assume text file with one number per line
            with open(zscore_vars_file, 'r') as f:
                vars_data = torch.tensor([float(line.strip()) for line in f])
        
        # Convert variances to standard deviations
        self.stds = torch.sqrt(vars_data + 1e-8)  # Add small epsilon for numerical stability
        
        # Ensure they're the right shape and dtype
        self.means = self.means.to(dtype=self.target_dtype)
        self.stds = self.stds.to(dtype=self.target_dtype)
        
        print(f"Loaded z-score means shape: {self.means.shape}, vars shape: {vars_data.shape}, computed stds shape: {self.stds.shape}")
    
    def unnormalize_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Convert fp16 z-score normalized data back to fp32 original distribution.
        
        Args:
            batch: Input batch tensor (z-score normalized, typically fp16)
            
        Returns:
            torch.Tensor: Unnormalized data in target dtype (typically fp32)
        """
        # Convert to target dtype (typically fp32)
        batch = batch.to(dtype=self.target_dtype)
        
        # Z-score unnormalize if we have stats
        if self.means is not None and self.stds is not None:
            # Z-score unnormalize: x_original = x_zscore * std + mean
            batch = batch * self.stds + self.means
        
        return batch


class SingleShardActivationLoader(Dataset):
    """Base class for loading activation data from a single shard"""
    
    def __init__(
        self,
        filename: str,
        total_tokens: int,
        d_model: int,
        shuffle: bool = True,
        zscore_unnormalizer: ZScoreUnnormalizer | None = None,
    ):
        self.filename = filename
        self.total_tokens = total_tokens
        self.d_model = d_model
        self.shuffle = shuffle
        self.zscore_unnormalizer = zscore_unnormalizer
        
        # Defer permutation creation until needed to save memory
        self._permutation = None
        self.accessed_indices = set()

    @property
    def permutation(self):
        """Lazy creation of permutation tensor"""
        if self._permutation is None:
            self._permutation = (
                torch.randperm(self.total_tokens) if self.shuffle else torch.arange(self.total_tokens)
            )
        return self._permutation

    def __getitem__(self, idx: int) -> torch.Tensor:
        permuted_idx = self.permutation[idx]
        self.accessed_indices.add(idx)
        
        # Subclasses implement the actual data loading
        result = self._load_data_at_index(permuted_idx)

        # Apply z-score unnormalization if available
        if self.zscore_unnormalizer is not None:
            result = self.zscore_unnormalizer.unnormalize_batch(result)
        
        return result

    def _load_data_at_index(self, idx: int) -> torch.Tensor:
        """Subclasses must implement this method"""
        raise NotImplementedError("Subclasses must implement _load_data_at_index")

    def __len__(self):
        return self.total_tokens


class PtShardActivationLoader(SingleShardActivationLoader):
    """Loader for .pt PyTorch tensor files"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor = None  # Lazy load the tensor

    def _load_data_at_index(self, idx: int) -> torch.Tensor:
        """Load data from .pt file"""
        # Load the entire tensor once
        if self.tensor is None:
            self.tensor = torch.load(
                self.filename, map_location="cpu", weights_only=True
            )
        
        # Memory optimization: free tensor when all indices accessed
        if len(self.accessed_indices) == self.total_tokens:
            result = self.tensor[idx].clone()
            # Clear memory
            self.tensor = None
            self.accessed_indices.clear()
        else:
            result = self.tensor[idx]
        
        return result


class MemmapShardActivationLoader(SingleShardActivationLoader):
    """Lazy-loading loader for .dat memmap files"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Lazy loading - don't open memmap until needed
        self._memmap = None
        self._is_loaded = False

    def _load_data_at_index(self, idx: int) -> torch.Tensor:
        """Load data from memmap file with lazy loading"""
        self._ensure_loaded()
        
        # Read the specific index and copy to avoid memory mapping issues
        data = self._memmap[idx].copy()
        
        # Convert to torch tensor
        result = torch.from_numpy(data).float()
        
        # Close memmap if we've accessed all tokens (shard is done)
        if len(self.accessed_indices) == self.total_tokens:
            self._close_memmap()
        
        return result

    def _ensure_loaded(self):
        """Load the memmap file if not already loaded"""
        if not self._is_loaded:
            try:
                self._memmap = np.memmap(
                    self.filename, dtype=np.float16, mode='r',
                    shape=(self.total_tokens, self.d_model)
                )
                self._is_loaded = True
            except Exception as e:
                raise RuntimeError(f"Failed to load memmap {self.filename}: {e}")

    def _close_memmap(self):
        """Close the memmap to free memory"""
        if self._memmap is not None:
            del self._memmap
            self._memmap = None
            self._is_loaded = False

    def close(self):
        """Explicitly close the memmap"""
        self._close_memmap()

    def __del__(self):
        """Cleanup on destruction"""
        self._close_memmap()


class ShardedActivationsDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        shuffle: bool = True,
        seed: int = None,
        n_shards_to_include: int = None,
        zscore_unnormalizer: ZScoreUnnormalizer | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.datasets = []
        self.total_tokens = 0
        self.d_model = None
        self.cumulative_tokens = [0]
        self.shuffle = shuffle
        self.n_shards_to_include = n_shards_to_include
        self.zscore_unnormalizer = zscore_unnormalizer

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        print("Loading dataset metadata")

        # Load sharded dataset
        self._load_sharded_dataset()

    def clear_memmap_caches(self):
        """
        Clear all memmap caches to free file handles and memory mappings.
        Useful for memory management with large datasets.
        """
        closed_count = 0
        for dataset_info in self.datasets:
            dataset = dataset_info["dataset"]
            
            # Skip if dataset hasn't been created yet (truly lazy!)
            if dataset is None:
                continue
            
            # Handle memmap datasets (MemmapShardActivationLoader)
            if hasattr(dataset, '_close_memmap'):
                dataset._close_memmap()
                closed_count += 1
        
        if closed_count > 0:
            print(f"🧹 Closed {closed_count} memmap caches to free memory")

    def _load_sharded_dataset(self):
        """Load sharded dataset from subdirectories containing activations.pt and metadata"""
        subdirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        if self.n_shards_to_include is not None:
            subdirs = subdirs[: self.n_shards_to_include]

        print(f"Loading metadata from {len(subdirs)} shard directories...")
        for subdir in tqdm(subdirs):
            dataset_info = self._load_shard_info(subdir)
            if dataset_info is not None:
                self.datasets.append(dataset_info)
                self.total_tokens += dataset_info["total_tokens"]
                self.cumulative_tokens.append(self.total_tokens)
                if self.d_model is None:
                    self.d_model = dataset_info["d_model"]
                else:
                    assert self.d_model == dataset_info["d_model"]
            else:
                print(f"Skipping {subdir} - missing activations or metadata")

    def _load_shard_info(self, subdir: Path):
        """Load dataset info from a shard directory

        Supports both .pt and .dat formats with metadata in .json or .yaml
        """
        # Try .pt first (ESM format), then .dat (memmap format)
        activations_path = subdir / "activations.pt"
        is_memmap = False

        if not activations_path.exists():
            activations_path = subdir / "activations.dat"
            is_memmap = True
            if not activations_path.exists():
                return None

        # Load metadata (try json first, then yaml)
        metadata = None
        if (subdir / "metadata.json").exists():
            metadata = json.load(open(subdir / "metadata.json", "r"))
        elif (subdir / "metadata.yaml").exists():
            metadata = yaml.load(open(subdir / "metadata.yaml", "r"), Loader=yaml.FullLoader)

        if metadata is None:
            return None

        # Create appropriate loader based on file type
        if is_memmap:
            dataset = MemmapShardActivationLoader(
                str(activations_path),
                total_tokens=metadata["total_tokens"],
                d_model=metadata["d_model"],
                shuffle=self.shuffle,
                zscore_unnormalizer=self.zscore_unnormalizer,
            )
        else:
            dataset = PtShardActivationLoader(
                str(activations_path),
                total_tokens=metadata["total_tokens"],
                d_model=metadata["d_model"],
                shuffle=self.shuffle,
                zscore_unnormalizer=self.zscore_unnormalizer,
            )

        return {
            "plm_name": metadata.get("model", "unknown"),
            "total_tokens": metadata["total_tokens"],
            "d_model": metadata["d_model"],
            "dataset": dataset,
            "dtype": metadata["dtype"],
            "layer": metadata.get("layer", None),
        }

    def __len__(self):
        return self.total_tokens

    def __getitem__(self, idx: int) -> torch.Tensor:
        dataset_index = (
            next(
                i
                for i, cum_tokens in enumerate(self.cumulative_tokens)
                if cum_tokens > idx
            )
            - 1
        )
        local_idx = idx - self.cumulative_tokens[dataset_index]
        
        # Memory optimization: close other lazy shards when switching to a new shard
        self._manage_lazy_shard_memory(dataset_index)
        
        # Create dataset on-demand if not already created
        dataset_info = self.datasets[dataset_index]
        if dataset_info["dataset"] is None:
            # Create the dataset object only when first accessed
            if "shard_file" in dataset_info:
                # Memmap format
                dataset_info["dataset"] = MemmapShardActivationLoader(
                    dataset_info["shard_file"],
                    total_tokens=dataset_info["total_tokens"],
                    d_model=dataset_info["d_model"],
                    shuffle=self.shuffle,
                    zscore_unnormalizer=self.zscore_unnormalizer,
                )
            else:
                # This shouldn't happen with the new lazy approach, but keeping for safety
                raise RuntimeError("Dataset object is None and no shard_file found!")
        
        return dataset_info["dataset"][local_idx]
    
    def _manage_lazy_shard_memory(self, active_shard_index: int):
        """
        Close memmap files for inactive shards to save memory.
        Only keeps the active shard and 1-2 adjacent shards loaded.
        """
        if not hasattr(self, '_last_active_shard'):
            self._last_active_shard = -1
        
        # Only manage memory if we're switching to a different shard
        if active_shard_index != self._last_active_shard:
            for i, dataset_info in enumerate(self.datasets):
                dataset = dataset_info["dataset"]
                
                # Skip if dataset hasn't been created yet (truly lazy!)
                if dataset is None:
                    continue
                
                # Check if this is a memmap shard
                if hasattr(dataset, '_close_memmap'):
                    # Keep active shard and immediate neighbors loaded
                    should_keep_loaded = abs(i - active_shard_index) <= 1
                    
                    if not should_keep_loaded and hasattr(dataset, '_is_loaded') and dataset._is_loaded:
                        dataset._close_memmap()
            
            self._last_active_shard = active_shard_index


class ActivationsDataLoader(DataLoader):
    def __init__(
        self,
        dataloader_config: DataloaderConfig,
    ):

        self.config = dataloader_config

        # Set seed
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)

        # Initialize z-score unnormalizer if needed
        zscore_unnormalizer = None
        if self.config.zscore_means_file is not None and self.config.zscore_vars_file is not None:
            zscore_unnormalizer = ZScoreUnnormalizer(
                zscore_means_file=self.config.zscore_means_file,
                zscore_vars_file=self.config.zscore_vars_file,
                target_dtype=self.config.target_dtype,
            )

        # Initialize dataset with fixed seed
        acts_dataset = ShardedActivationsDataset(
            self.config.plm_embd_dir,
            seed=self.config.seed,
            shuffle=True,
            n_shards_to_include=self.config.n_shards_to_include,
            zscore_unnormalizer=zscore_unnormalizer,
        )

        device = get_device()

        def collate_fn(batch):
            # Always return single tensor (unnormalized fp32 data)
            return torch.stack(batch).to(device)

        # TODO: Make this work with multi-epoch
        if self.config.samples_to_skip > 0:
            # make a subset of the dataset that skips the first samples_to_skip tokens
            # Create a subset that preserves dataset attributes
            acts_dataset = AttributePreservingSubset(
                acts_dataset, range(self.config.samples_to_skip, len(acts_dataset))
            )

        super().__init__(
            acts_dataset,
            batch_size=dataloader_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )


class AttributePreservingSubset(Subset):
    """
    AttributePreservingSubset is a custom subclass of torch.utils.data.Subset that ensures
    important attributes from the original dataset (such as d_model and zscore_unnormalizer)
    are preserved when creating a subset. This is useful when you want to skip a number of
    samples (e.g., for resuming training) but still need access to dataset-level metadata
    or normalization logic in downstream code. 
    """
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # Preserve key attributes from the original dataset for downstream access
        self.d_model = dataset.d_model
        self.zscore_unnormalizer = getattr(dataset, 'zscore_unnormalizer', None)
        if self.zscore_unnormalizer is None:
            self.zscore_unnormalizer = getattr(dataset, None)
