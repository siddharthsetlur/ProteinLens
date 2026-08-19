#!/usr/bin/env python
"""
Basic SAE training example with hardcoded parameters.

This is the simplest way to train an SAE - no configuration files or CLI arguments needed.
Just set the LAYER environment variable and run it:

    export LAYER=4
    python examples/train_basic_sae.py

This script trains a standard ReLU SAE with sensible defaults and is designed
for the README walkthrough. For more control over architecture, hyperparameters,
or to explore different SAE variants, see train_multiple_sae_architectures.py
"""

import os
from pathlib import Path
import torch

from proteinlens.train.configs import (
    TrainingRunConfig,
    DataloaderConfig,
    ReLUTrainerConfig,
    WandbConfig,
    CheckpointConfig
)
from proteinlens.train.fidelity import ESMFidelityConfig
from proteinlens.train.training_run import SAETrainingRun


def main():
    # ========== Configuration ==========
    # These are the settings for the walkthrough

    # Get proteinlens_DATA from environment or use default
    GEOPEDIA_DATA = os.environ.get("GEOPEDIA_DATA", "data")
    MODEL_DIR = os.environ.get("MODEL_DIR", "models")

    # Get LAYER from environment
    LAYER = os.environ.get("LAYER")
    if LAYER is None:
        raise RuntimeError("Environment variable 'LAYER' must be set (e.g., export LAYER=3)")

    # Paths
    EMBEDDINGS_DIR = Path(GEOPEDIA_DATA) / "training_embeddings" / "esm2_8m" / f"layer_{LAYER}"
    EVAL_SEQ_FILE = Path(GEOPEDIA_DATA) / "eval_sequences.txt"
    EVAL_FASTA = Path(GEOPEDIA_DATA) / "eval_shards" / "shard_0.fasta"
    SAVE_DIR = Path(MODEL_DIR) / "relu" / f"layer_{LAYER}"

    # Model dimensions
    EMBEDDING_DIM = 320  # ESM2-8M layer dimension
    # EXPANSION_FACTOR = 32 #InterPLM has an expansion factor of 32
    HIDDEN_SIZE = 10240   # Number of SAE features (32x expansion as in the paper)

    # Training hyperparameters (optimized for convergence)
    BATCH_SIZE = 2048     # Batch size FROM THE inTERpLM pAPER
    LEARNING_RATE = 2e-4 # lr FROM paper was 1
    STEPS = 500000        # num_steps from paper 
    L1_COEFFICIENT = 0.06

    # ========================================

    print("=" * 60)
    print("GeoPedia SAE Training Walkthrough")
    print("=" * 60)
    print(f"Training embeddings: {EMBEDDINGS_DIR}")
    print(f"Evaluation FASTA: {EVAL_FASTA}")
    print(f"Save directory: {SAVE_DIR}")
    print(f"Model: {EMBEDDING_DIM}D → {HIDDEN_SIZE} features")
    print()

    # Create eval sequences file if it doesn't exist
    if not EVAL_SEQ_FILE.exists() and EVAL_FASTA.exists():
        print(f"Creating eval sequences file from {EVAL_FASTA}...")
        from Bio import SeqIO
        EVAL_SEQ_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(EVAL_FASTA) as f_in, open(EVAL_SEQ_FILE, 'w') as f_out:
            for i, record in enumerate(SeqIO.parse(f_in, "fasta")):
                if i >= 100:  # Use 100 sequences for evaluation
                    break
                f_out.write(str(record.seq) + "\n")
        print(f"Created {EVAL_SEQ_FILE} with 100 sequences from held-out eval shard")
        print()

    # Create configuration objects
    dataloader_cfg = DataloaderConfig(
        plm_embd_dir=EMBEDDINGS_DIR,
        batch_size=BATCH_SIZE,
    )

    # Trainer configuration for ReLU SAE
    trainer_cfg = ReLUTrainerConfig(
        activation_dim=EMBEDDING_DIM,
        dictionary_size=HIDDEN_SIZE,
        lr=LEARNING_RATE,
        l1_penalty=L1_COEFFICIENT,
        warmup_steps=50000,  # 10% of total steps for warmup
        decay_start=400000,   # Start decay at 80% of training
        steps=STEPS,
        normalize_to_sqrt_d=False,
    )

    # Evaluation config - use ESMFidelityConfig for comprehensive evaluation
    eval_cfg = ESMFidelityConfig(
        eval_seq_path=EVAL_SEQ_FILE if EVAL_SEQ_FILE.exists() else None,
        model_name="esm2_t6_8M_UR50D",
        layer_idx=int(LAYER),
        eval_steps=10000,  # Don't run during training (only at end)
        eval_batch_size=256,
    )
    
    # W&B config (disabled for walkthrough)
    wandb_cfg = WandbConfig(
        use_wandb=True,
        wandb_entity="<YOUR_WANDB_ENTITY>",      # Your wandb username
        wandb_project="proteinlens-sae",  # Project name (creates if doesn't exist)
        wandb_name="relu_base",      # This specific run's name
        log_steps=100,                     # Log metrics to wandb every 100 steps
    )
    
    # Checkpoint config - only keep latest checkpoint to save space
    checkpoint_cfg = CheckpointConfig(
        save_dir=SAVE_DIR,
        save_steps=2000,  # Save every 2000 steps
        max_ckpts_to_keep=1,  # Only keep the latest checkpoint
    )
    
    # Create combined config
    config = TrainingRunConfig(
        dataloader_cfg=dataloader_cfg,
        trainer_cfg=trainer_cfg,
        eval_cfg=eval_cfg,
        wandb_cfg=wandb_cfg,
        checkpoint_cfg=checkpoint_cfg,
    )
    
    print("Configuration created:")
    print(f"  Steps: {STEPS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE:.1e}")
    print(f"  L1 coefficient: {L1_COEFFICIENT:.1e}")
    print(f"  Warmup steps: {trainer_cfg.warmup_steps}")
    print(f"  Decay start: {trainer_cfg.decay_start}")
    print(f"  Checkpoints: Every {checkpoint_cfg.save_steps} steps")
    print(f"  Fidelity Evaluation: {'Enabled (at end)' if eval_cfg.eval_seq_path else 'Disabled'}")
    if eval_cfg.eval_seq_path:
        print(f"    - Will run comprehensive eval at end of training")
        print(f"    - Eval sequences: {eval_cfg.eval_seq_path}")
    print()
    
    # Create training run and start
    print("Starting training...")
    training_run = SAETrainingRun.from_config(config)
    training_run.run()
    
    print()
    print("=" * 60)
    print(f"Training complete! Model saved to {SAVE_DIR}")
    print(f"- Model weights: {SAVE_DIR}/ae.pt")
    print(f"- Configuration: {SAVE_DIR}/config.yaml")

    # Check if fidelity evaluation was run during training
    eval_results_file = SAVE_DIR / "final_evaluation.yaml"
    if eval_results_file.exists():
        print(f"- Final evaluation: {eval_results_file}")

        # Display the fidelity results
        import yaml
        with open(eval_results_file, 'r') as f:
            results = yaml.unsafe_load(f)
        if 'fidelity' in results:
            fidelity = results['fidelity'].get('pct_loss_recovered', 0)
            print(f"  ✅ Fidelity: {fidelity:.2f}% loss recovered")
    else:
        print("⚠️  Final evaluation was not run during training.")
        print("   This can happen if eval_seq_path is not configured.")
        print(f"   Run comprehensive evaluation with:")
        print(f"   python examples/evaluate_sae.py --sae_path {SAVE_DIR}/ae.pt \\")
        print(f"       --fasta_file {EVAL_FASTA} \\")
        print(f"       --model_name esm2_t6_8M_UR50D --layer {LAYER}")

    print(f"- Ready for normalization and analysis!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())