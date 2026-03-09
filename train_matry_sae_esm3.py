#!/usr/bin/env python
"""
Matryoshka Batch Top-K SAE training for ESM3-open embeddings.

Trains a Matryoshka SAE on pre-extracted ESM3-small-open (1.4B) embeddings.
Set the LAYER environment variable and run:

    export LAYER=24
    python train_matry_sae_esm3.py
"""

import os
from pathlib import Path
import torch

from proteinlens.train.configs import (
    TrainingRunConfig,
    DataloaderConfig,
    MatryoshkaBatchTopKTrainerConfig,
    WandbConfig,
    CheckpointConfig
)
from proteinlens.train.evaluation import EvaluationConfig
from proteinlens.train.training_run import SAETrainingRun


def main():
    # ========== Configuration ==========
    INTERPLM_DATA = os.environ.get("INTERPLM_DATA", "data")
    MODEL_DIR = os.environ.get("MODEL_DIR", "models")

    LAYER = os.environ.get("LAYER")
    if LAYER is None:
        raise RuntimeError("Environment variable 'LAYER' must be set (e.g., export LAYER=24)")

    # Paths - ESM3-open embeddings
    EMBEDDINGS_DIR = Path(INTERPLM_DATA) / "training_embeddings" / "esm3_open" / f"layer_{LAYER}"
    EVAL_EMBD_DIR = Path(INTERPLM_DATA) / "eval_embeddings" / "esm3_open" / f"layer_{LAYER}"
    SAVE_DIR = Path(MODEL_DIR) / "esm3_matryoshka" / f"layer_{LAYER}"

    # ESM3-open dimensions
    EMBEDDING_DIM = 1536
    HIDDEN_SIZE = 49152   # 32x expansion factor (1536 * 32)

    # Training hyperparameters
    BATCH_SIZE = 2048
    LEARNING_RATE = 2e-4
    STEPS = 500000
    FRACTIONS = [1/32, 3/32, 7/32, 15/32, 6/32]  # From Matryoshka Batch Top-K paper
    K = 30

    # ========================================

    print("=" * 60)
    print("ESM3-open SAE Training - MatryoshkaBatchTopK")
    print("=" * 60)
    print(f"Training embeddings: {EMBEDDINGS_DIR}")
    print(f"Save directory: {SAVE_DIR}")
    print(f"Model: {EMBEDDING_DIM}D -> {HIDDEN_SIZE} features")
    print()

    # Create configuration objects
    dataloader_cfg = DataloaderConfig(
        plm_embd_dir=EMBEDDINGS_DIR,
        batch_size=BATCH_SIZE,
    )

    trainer_cfg = MatryoshkaBatchTopKTrainerConfig(
        activation_dim=EMBEDDING_DIM,
        dictionary_size=HIDDEN_SIZE,
        lr=LEARNING_RATE,
        k=K,
        group_fractions=FRACTIONS,
        warmup_steps=50000,
        decay_start=400000,
        steps=STEPS,
        normalize_to_sqrt_d=False,
    )

    # Use base EvaluationConfig (reconstruction metrics only, no fidelity)
    eval_cfg = EvaluationConfig(
        eval_embd_dir=EVAL_EMBD_DIR if EVAL_EMBD_DIR.exists() else None,
        eval_steps=10000,
        eval_batch_size=256,
    )

    wandb_cfg = WandbConfig(
        use_wandb=True,
        wandb_entity="s-setlur-university-of-edinburgh",
        wandb_project="protein-sae-esm3",
        wandb_name=f"esm3_matryoshka_layer{LAYER}",
        log_steps=100,
    )

    checkpoint_cfg = CheckpointConfig(
        save_dir=SAVE_DIR,
        save_steps=2000,
        max_ckpts_to_keep=1,
    )

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
    print(f"  k: {trainer_cfg.k}")
    print(f"  Group fractions: {trainer_cfg.group_fractions}")
    print(f"  Warmup steps: {trainer_cfg.warmup_steps}")
    print(f"  Decay start: {trainer_cfg.decay_start}")
    print(f"  Checkpoints: Every {checkpoint_cfg.save_steps} steps")
    print()

    print("Starting training...")
    training_run = SAETrainingRun.from_config(config)
    training_run.run()

    print()
    print("=" * 60)
    print(f"Training complete! Model saved to {SAVE_DIR}")
    print(f"- Model weights: {SAVE_DIR}/ae.pt")
    print(f"- Configuration: {SAVE_DIR}/config.yaml")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
