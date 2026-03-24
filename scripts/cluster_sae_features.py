#!/usr/bin/env python
"""
Spectral clustering of SAE decoder vectors.

Loads a trained SAE, builds a pairwise angular-similarity matrix from the
decoder weight matrix (W_dec), runs Spectral Clustering, and saves the
resulting cluster→feature mapping as a YAML file.

Usage
-----
  python scripts/cluster_sae_features.py \\
      --sae-dir trained_models/fiery-sweep \\
      --n-clusters 500 \\
      --output clusters.yaml

  # Smaller run for a quick smoke-test:
  python scripts/cluster_sae_features.py \\
      --sae-dir trained_models/fiery-sweep \\
      --n-clusters 50 \\
      --output /tmp/test_clusters.yaml \\
      --chunk-size 512

Output YAML format
------------------
  0:
    - 42
    - 1337
    - 5012
  1:
    - 7
    - 99
  ...

Each top-level key is a cluster index (0-based integer).
Each value is a sorted list of SAE feature indices that belong to that cluster.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proteinlens.sae.inference import load_sae
from proteinlens.analysis.feature_clusters import FeatureClusters
from proteinlens.utils import get_device


def main():
    p = argparse.ArgumentParser(
        description="Spectral clustering of SAE decoder weight vectors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--sae-dir",
        required=True,
        help="Path to trained SAE directory (must contain ae.pt and config.yaml).",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        required=True,
        help="Number of clusters for SpectralClustering.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output path for clusters YAML file.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Row chunk size for building the similarity matrix (default: 1024). "
             "Reduce if you hit OOM.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device for loading the SAE (default: auto-detect). "
             "The similarity matrix is always computed on CPU.",
    )
    args = p.parse_args()

    device = args.device or get_device()
    print(f"Device: {device}")

    print(f"\nLoading SAE from {args.sae_dir} …")
    sae = load_sae(args.sae_dir, device=device)
    sae.eval()
    print(f"  SAE loaded: dict_size={sae.dict_size}, activation_dim={sae.activation_dim}")

    print()
    with torch.no_grad():
        fc = FeatureClusters.from_sae(
            sae,
            n_clusters=args.n_clusters,
            chunk_size=args.chunk_size,
            verbose=True,
        )

    print(f"\n{fc}")
    fc.save(args.output)


if __name__ == "__main__":
    main()
