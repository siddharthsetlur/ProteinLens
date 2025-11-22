#!/usr/bin/env python
"""
SAE training script for hyperparameter sweeps.
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
from proteinlens.train.fidelity import ESMFidelityConfig
from proteinlens.train.training_run import SAETrainingRun


def main():
    # Initialize wandb - this reads from the sweep
    run = wandb.init()
    config = wandb.config  # Hyperparameters from sweep
    
    # ========== Fixed Configuration ==========
    INTERPLM_DATA = os.environ.get("INTERPLM_DATA", "data")
    MODEL_DIR = os.environ.get("MODEL_DIR", "models")
    LAYER = os.environ.get("LAYER", "4")
    
    # Paths
    EMBEDDINGS_DIR = Path(INTERPLM_DATA) / "training_embeddings" / "esm2_8m" / f"layer_{LAYER}"
    EVAL_SEQ_FILE = Path(INTERPLM_DATA) / "eval_sequences.txt"
    EVAL_FASTA = Path(INTERPLM_DATA) / "eval_shards" / "shard_0.fasta"
    
    # Use wandb run ID for unique save directory
    SAVE_DIR = Path(MODEL_DIR) / "sweeps" / f"layer_{LAYER}" / wandb.run.id
    
    EMBEDDING_DIM = 320 # ESM2-8M layer dimension
    STEPS = 500000
    
    # ========== Sweep Hyperparameters ==========
    # These come from wandb.config
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
    
    # Create eval sequences if needed
    if not EVAL_SEQ_FILE.exists() and EVAL_FASTA.exists():
        from Bio import SeqIO
        EVAL_SEQ_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(EVAL_FASTA) as f_in, open(EVAL_SEQ_FILE, 'w') as f_out:
            for i, record in enumerate(SeqIO.parse(f_in, "fasta")):
                if i >= 100:
                    break
                f_out.write(str(record.seq) + "\n")
    
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
    
    eval_cfg = ESMFidelityConfig(
        eval_seq_path=EVAL_SEQ_FILE if EVAL_SEQ_FILE.exists() else None,
        model_name="esm2_t6_8M_UR50D",
        layer_idx=int(LAYER),
        eval_steps=100000,  # Evaluate during training (every 100k steps)
        eval_batch_size=256,
    )
    
    # WandB already initialized by sweep agent
    wandb_cfg = WandbConfig(
        use_wandb=True,
        wandb_entity="s-setlur-university-of-edinburgh",
        wandb_project="protein-sae-eidf",
        wandb_name=None,  # Use default name from sweep
        log_steps=100,
    )
    
    checkpoint_cfg = CheckpointConfig(
        save_dir=SAVE_DIR,
        save_steps=50000,  # Save less frequently to save space
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