#!/usr/bin/env python
"""
Evaluate a trained SAE on held-out protein sequences.

This script calculates three key metrics:
1. Reconstruction quality (variance explained, MSE)
2. Sparsity (L0, feature activation statistics)
3. Downstream task fidelity (loss recovered on masked language modeling)

Usage:
    python examples/evaluate_sae.py \
        --sae_path models/walkthrough_model/layer_4/ae.pt \
        --fasta_file data/uniprot_shards/shard_0.fasta \
        --model_name esm2_t6_8M_UR50D \
        --layer 4 \
        --output_file results/evaluation_metrics.yaml
"""

from pathlib import Path
from typing import Optional
import yaml
import torch
from tqdm import tqdm

from proteinlens.sae.inference import load_sae
from proteinlens.embedders import get_embedder
from proteinlens.train.fidelity import ESMFidelityConfig
from proteinlens.utils import get_device


def calculate_reconstruction_metrics(sae, embeddings):
    """Calculate reconstruction quality metrics.

    Args:
        sae: Trained SAE model
        embeddings: Original embeddings tensor [n_tokens, d_model]

    Returns:
        dict: Reconstruction metrics
    """
    with torch.no_grad():
        reconstructed = sae(embeddings)

        # MSE
        mse = torch.mean((embeddings - reconstructed) ** 2).item()

        # Variance explained
        total_variance = torch.var(embeddings, dim=0).sum()
        residual_variance = torch.var(embeddings - reconstructed, dim=0).sum()
        variance_explained = (1 - residual_variance / total_variance).item()

        # Per-dimension MSE statistics
        per_dim_mse = torch.mean((embeddings - reconstructed) ** 2, dim=0)

    return {
        "mse": mse,
        "variance_explained": variance_explained,
        "per_dim_mse_mean": per_dim_mse.mean().item(),
        "per_dim_mse_std": per_dim_mse.std().item(),
        "per_dim_mse_max": per_dim_mse.max().item(),
    }


def calculate_sparsity_metrics(sae, embeddings):
    """Calculate sparsity metrics.

    Args:
        sae: Trained SAE model
        embeddings: Original embeddings tensor [n_tokens, d_model]

    Returns:
        dict: Sparsity metrics
    """
    with torch.no_grad():
        features = sae.encode(embeddings)

        # L0 sparsity (average number of active features per token)
        l0_sparsity = (features != 0).float().sum(dim=-1).mean().item()

        # Feature activation frequency (% of tokens that activate each feature)
        feature_active = (features != 0).float().mean(dim=0)

        # Dead features (never activate)
        dead_features = (feature_active == 0).sum().item()

        # Highly active features (activate on >50% of tokens)
        highly_active = (feature_active > 0.5).sum().item()

    return {
        "l0_sparsity": l0_sparsity,
        "l0_sparsity_pct": (l0_sparsity / sae.dict_size) * 100,
        "dead_features": dead_features,
        "dead_features_pct": (dead_features / sae.dict_size) * 100,
        "highly_active_features": highly_active,
        "highly_active_pct": (highly_active / sae.dict_size) * 100,
        "mean_feature_activation_freq": feature_active.mean().item() * 100,
    }


def extract_embeddings_from_fasta(fasta_file, embedder, layer, max_proteins=None):
    """Extract embeddings from a FASTA file.

    Args:
        fasta_file: Path to FASTA file
        embedder: Protein embedder instance
        layer: Layer to extract
        max_proteins: Maximum number of proteins to process (default: all)

    Returns:
        torch.Tensor: Concatenated embeddings [n_tokens, d_model]
    """
    from Bio import SeqIO

    all_embeddings = []
    n_proteins = 0
    n_skipped = 0

    print(f"Extracting embeddings from {fasta_file}...")

    # Count sequences first
    with open(fasta_file) as f:
        n_seqs = sum(1 for _ in SeqIO.parse(f, "fasta"))

    # Limit if requested
    if max_proteins is not None:
        n_seqs = min(n_seqs, max_proteins)

    with open(fasta_file) as f:
        for record in tqdm(SeqIO.parse(f, "fasta"), total=n_seqs, desc="Processing proteins"):
            sequence = str(record.seq)

            # Get embeddings for this protein
            embeddings = embedder.embed_single_sequence(
                sequence,
                layer=layer
            )

            # Convert to torch tensor if needed
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.from_numpy(embeddings)

            # Skip proteins with NaN embeddings (can happen with some ESM models/devices)
            if torch.isnan(embeddings).any():
                n_skipped += 1
                continue

            all_embeddings.append(embeddings)
            n_proteins += 1

            # Stop if we've reached max_proteins
            if max_proteins is not None and n_proteins >= max_proteins:
                break

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)

    print(f"Extracted {all_embeddings.shape[0]:,} tokens from {n_proteins} proteins")
    if n_skipped > 0:
        print(f"  (Skipped {n_skipped} proteins with NaN embeddings)")

    return all_embeddings


def evaluate_sae(
    sae_path: str,
    fasta_file: Path,
    model_name: str,
    layer: int,
    output_file: Optional[Path] = None,
    skip_fidelity: bool = False,
    fidelity_batch_size: int = 8,
    max_proteins: Optional[int] = None,
    device: Optional[str] = None,
):
    """
    Evaluate SAE on held-out protein sequences.

    Args:
        sae_path: Path to SAE model (local path or hf://org/model:layer_N)
        fasta_file: Path to FASTA file with evaluation sequences
        model_name: ESM model name (e.g., esm2_t6_8M_UR50D)
        layer: Layer index to evaluate
        output_file: Path to save results (default: print to stdout)
        skip_fidelity: Skip downstream task fidelity evaluation (faster)
        fidelity_batch_size: Batch size for fidelity evaluation (default: 8)
        max_proteins: Maximum number of proteins to evaluate (default: all)
        device: Device for embedder (cpu/cuda/mps, default: auto-detect)
    """
    # Validate inputs
    if not fasta_file.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

    print("=" * 70)
    print("SAE Evaluation Script")
    print("=" * 70)
    print(f"SAE: {sae_path}")
    print(f"FASTA: {fasta_file}")
    print(f"Model: {model_name}")
    print(f"Layer: {layer}")
    print()

    # Load SAE
    device_str = get_device()
    print(f"Loading SAE on device: {device_str}")

    # Parse sae_path - could be directory or .pt file
    sae_path_obj = Path(sae_path)
    if sae_path_obj.is_file():
        # If it's a .pt file, use the parent directory
        sae_dir = sae_path_obj.parent
        model_name_sae = sae_path_obj.name
    else:
        # If it's a directory, use default model name
        sae_dir = sae_path_obj
        model_name_sae = "ae.pt"

    sae = load_sae(sae_dir, model_name=model_name_sae, device=device_str)

    print(f"SAE loaded: {sae.dict_size} features, {sae.activation_dim}D embeddings")
    print()

    # Load embedder
    print("Loading protein embedder...")
    embedder_type = "esm"

    # MPS has known issues with ESM-2 producing NaN embeddings in loops
    # Force CPU for embedder if MPS is detected and user didn't specify device
    embedder_device = device if device else device_str
    if embedder_device == "mps" and device is None:
        print("⚠️  WARNING: MPS device detected. ESM-2 has known NaN issues on MPS.")
        print("   Using CPU for embedder instead (SAE will still use MPS).")
        embedder_device = "cpu"

    embedder = get_embedder(
        embedder_type,
        model_name=f"facebook/{model_name}",
        device=embedder_device
    )
    print(f"Embedder loaded: {embedder_type} on {embedder_device}")
    print()

    # Extract embeddings
    embeddings = extract_embeddings_from_fasta(
        fasta_file,
        embedder,
        layer,
        max_proteins=max_proteins
    )

    embeddings = embeddings.to(device_str)

    # Calculate metrics
    results = {
        "sae_path": str(sae_path),
        "fasta_file": str(fasta_file),
        "model_name": model_name,
        "layer": layer,
        "n_tokens": embeddings.shape[0],
        "sae_dict_size": sae.dict_size,
        "sae_activation_dim": sae.activation_dim,
    }

    # 1. Reconstruction metrics
    print("\n" + "=" * 70)
    print("1. RECONSTRUCTION QUALITY")
    print("=" * 70)
    recon_metrics = calculate_reconstruction_metrics(sae, embeddings)
    results["reconstruction"] = recon_metrics

    for key, value in recon_metrics.items():
        print(f"  {key}: {value:.6f}")

    # 2. Sparsity metrics
    print("\n" + "=" * 70)
    print("2. SPARSITY")
    print("=" * 70)
    sparsity_metrics = calculate_sparsity_metrics(sae, embeddings)
    results["sparsity"] = sparsity_metrics

    for key, value in sparsity_metrics.items():
        if "pct" in key or "freq" in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value}")

    # 3. Fidelity metrics
    if not skip_fidelity:
        print("\n" + "=" * 70)
        print("3. DOWNSTREAM TASK FIDELITY")
        print("=" * 70)
        print("This measures how well the SAE preserves ESM's ability to predict")
        print("masked tokens. Higher is better (100% = perfect preservation).")
        print()

        # Create temporary file with sequences for fidelity eval
        import tempfile
        from Bio import SeqIO

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_seq_file = f.name
            with open(fasta_file) as fasta_f:
                for i, record in enumerate(SeqIO.parse(fasta_f, "fasta")):
                    if max_proteins is not None and i >= max_proteins:
                        break
                    f.write(str(record.seq) + "\n")

        try:
            fidelity_config = ESMFidelityConfig(
                eval_seq_path=temp_seq_file,
                model_name=model_name,
                layer_idx=layer,
                eval_batch_size=fidelity_batch_size,
            )

            fidelity_eval = fidelity_config.build()
            fidelity_metrics = fidelity_eval._calculate_fidelity(sae)
            results["fidelity"] = fidelity_metrics

            for key, value in fidelity_metrics.items():
                if "pct" in key:
                    print(f"  {key}: {value:.2f}%")
                else:
                    print(f"  {key}: {value:.6f}")
        finally:
            import os
            os.unlink(temp_seq_file)
    else:
        print("\n(Skipping fidelity evaluation)")

    # Save or print results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        print(f"Results saved to: {output_file}")
    else:
        print("\nFull results:")
        print(yaml.dump(results, default_flow_style=False))

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    from tap import tapify
    tapify(evaluate_sae)
