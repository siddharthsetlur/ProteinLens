#!/usr/bin/env python
"""
Matryoshka SAE training script for ESM3-open hyperparameter sweeps.
Usage: wandb agent <sweep-id>
"""

import os
from pathlib import Path
import wandb

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
    # Initialize wandb - this reads from the sweep
    run = wandb.init()
    config = wandb.config

    # ========== Fixed Configuration ==========
    INTERPLM_DATA = os.environ.get("INTERPLM_DATA", "data")
    MODEL_DIR = os.environ.get("MODEL_DIR", "models")
    LAYER = os.environ.get("LAYER", "24")

    # Paths - ESM3-open embeddings
    EMBEDDINGS_DIR = Path(INTERPLM_DATA) / "training_embeddings" / "esm3_open" / f"layer_{LAYER}"
    EVAL_EMBD_DIR = Path(INTERPLM_DATA) / "eval_embeddings" / "esm3_open" / f"layer_{LAYER}"

    SAVE_DIR = Path(MODEL_DIR) / "esm3_sweeps" / "matryoshka" / f"layer_{LAYER}" / wandb.run.id

    EMBEDDING_DIM = 1536  # ESM3-open hidden dimension
    STEPS = 500000

    # ========== Sweep Hyperparameters ==========
    LEARNING_RATE = config.learning_rate
    K = config.k
    HIDDEN_SIZE = config.hidden_size
    BATCH_SIZE = config.batch_size
    WARMUP_STEPS = int(config.warmup_ratio * STEPS)
    DECAY_START = int(config.decay_start_ratio * STEPS)

    if hasattr(config, 'group_fractions'):
        FRACTIONS = list(config.group_fractions)
    else:
        FRACTIONS = [0.03125, 0.09375, 0.21875, 0.46875, 0.1875]

    AUXK_ALPHA = config.auxk_alpha if hasattr(config, 'auxk_alpha') else 0.03125

    print(f"Starting sweep run: {wandb.run.name}")
    print(f"Hyperparameters:")
    print(f"  LR: {LEARNING_RATE:.2e}")
    print(f"  k: {K}")
    print(f"  auxk_alpha: {AUXK_ALPHA}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Group fractions: {FRACTIONS}")

    # Create configuration
    dataloader_cfg = DataloaderConfig(
        plm_embd_dir=EMBEDDINGS_DIR,
        batch_size=BATCH_SIZE,
    )

    trainer_cfg = MatryoshkaBatchTopKTrainerConfig(
        activation_dim=EMBEDDING_DIM,
        dictionary_size=HIDDEN_SIZE,
        lr=LEARNING_RATE,
        k=K,
        auxk_alpha=AUXK_ALPHA,
        group_fractions=FRACTIONS,
        warmup_steps=WARMUP_STEPS,
        decay_start=DECAY_START,
        steps=STEPS,
        normalize_to_sqrt_d=False,
    )

    eval_cfg = EvaluationConfig(
        eval_embd_dir=EVAL_EMBD_DIR if EVAL_EMBD_DIR.exists() else None,
        eval_steps=100000,
        eval_batch_size=256,
    )

    wandb_cfg = WandbConfig(
        use_wandb=True,
        wandb_entity="s-setlur-university-of-edinburgh",
        wandb_project="protein-sae-esm3",
        wandb_name=wandb.run.name,
        log_steps=100,
    )

    checkpoint_cfg = CheckpointConfig(
        save_dir=SAVE_DIR,
        save_steps=50000,
        max_ckpts_to_keep=1,
    )

    training_config = TrainingRunConfig(
        dataloader_cfg=dataloader_cfg,
        trainer_cfg=trainer_cfg,
        eval_cfg=eval_cfg,
        wandb_cfg=wandb_cfg,
        checkpoint_cfg=checkpoint_cfg,
    )

    training_run = SAETrainingRun.from_config(training_config)
    training_run.run()
    print(f"Training complete! Model saved to {SAVE_DIR}")

    wandb.finish()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
