#!/usr/bin/env python
"""
ReLU SAE training script for ESMC hyperparameter sweeps.
Usage: wandb agent <sweep-id>
"""

import os
from pathlib import Path
import wandb

from proteinlens.train.configs import (
    TrainingRunConfig,
    DataloaderConfig,
    ReLUTrainerConfig,
    WandbConfig,
    CheckpointConfig
)
from proteinlens.train.evaluation import EvaluationConfig
from proteinlens.train.training_run import SAETrainingRun


def main():
    # Initialize wandb - this reads from the sweep
    run = wandb.init()
    config = wandb.config  # Hyperparameters from sweep

    # ========== Fixed Configuration ==========
    INTERPLM_DATA = os.environ.get("INTERPLM_DATA", "data")
    MODEL_DIR = os.environ.get("MODEL_DIR", "models")
    LAYER = os.environ.get("LAYER", "15")

    # Paths - ESMC-300M embeddings
    EMBEDDINGS_DIR = Path(INTERPLM_DATA) / "training_embeddings" / "esmc_300m" / f"layer_{LAYER}"
    EVAL_EMBD_DIR = Path(INTERPLM_DATA) / "eval_embeddings" / "esmc_300m" / f"layer_{LAYER}"

    # Use wandb run ID for unique save directory
    SAVE_DIR = Path(MODEL_DIR) / "esmc_sweeps" / "relu" / f"layer_{LAYER}" / wandb.run.id

    EMBEDDING_DIM = 960  # ESMC-300M hidden dimension
    STEPS = 500000

    # ========== Sweep Hyperparameters ==========
    LEARNING_RATE = config.learning_rate
    L1_COEFFICIENT = config.l1_coefficient
    HIDDEN_SIZE = config.hidden_size
    BATCH_SIZE = config.batch_size
    WARMUP_STEPS = int(config.warmup_ratio * STEPS)
    DECAY_START = int(config.decay_start_ratio * STEPS)

    print(f"Starting sweep run: {wandb.run.name}")
    print(f"Hyperparameters:")
    print(f"  LR: {LEARNING_RATE:.2e}")
    print(f"  L1: {L1_COEFFICIENT:.2e}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")

    # Create configuration
    dataloader_cfg = DataloaderConfig(
        plm_embd_dir=EMBEDDINGS_DIR,
        batch_size=BATCH_SIZE,
    )

    trainer_cfg = ReLUTrainerConfig(
        activation_dim=EMBEDDING_DIM,
        dictionary_size=HIDDEN_SIZE,
        lr=LEARNING_RATE,
        l1_penalty=L1_COEFFICIENT,
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
        wandb_project="protein-sae-esmc",
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

    # Train
    training_run = SAETrainingRun.from_config(training_config)
    training_run.run()
    print(f"Training complete! Model saved to {SAVE_DIR}")

    wandb.finish()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
