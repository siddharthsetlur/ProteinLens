# Training SAEs on ESM2 Embeddings — Step-by-Step Guide

This document describes the end-to-end pipeline for training a Sparse Autoencoder (SAE) on a specific layer of ESM2, from data preparation through hyperparameter sweeps with Weights & Biases.

---

## Prerequisites

- Conda environment created from `environment.yml`
- A collection of protein sequences in FASTA format
- Access to a GPU (recommended)
- A Weights & Biases account (for sweeps)

---

## Pipeline Overview

```
FASTA sequences
  → (1) Shard & subset data
  → (2) Extract embeddings from an ESM2 layer
  → (3) Train SAE (single run or W&B sweep)
  → (4) Evaluate trained SAE
```

---

## Step 1: Prepare Your Data

### 1a. Subset your FASTA file (optional)

If you have a large FASTA file and want to filter or sample from it:

```bash
python scripts/subset_fasta.py \
    --input_fasta data/uniprot.fasta \
    --output_fasta data/uniprot_subset.fasta \
    --max_length 1024 \
    --num_sequences 100000
```

### 1b. Shard your FASTA file

Split a single FASTA file into multiple shards for parallel processing and to create a held-out evaluation set:

```bash
python scripts/shard_fasta.py \
    --input_fasta data/uniprot_subset.fasta \
    --output_dir data/training_shards \
    --num_shards 10
```

Keep one shard aside for evaluation (e.g. move the last shard to `data/eval_shards/`).

---

## Step 2: Extract Embeddings

Use `scripts/extract_embeddings.py` to run a forward pass through ESM2 and save the hidden-state activations for the layer(s) you want to train on.

```bash
python scripts/extract_embeddings.py \
    --fasta_dir data/training_shards \
    --output_dir data/training_embeddings/esm2_8m \
    --embedder_type esm \
    --model_name facebook/esm2_t6_8M_UR50D \
    --layers 4 \
    --batch_size 8
```

**What this does:**

1. Loads the ESM2 model via the `ESM` embedder class (`proteinlens/embedders/esm.py`)
2. For each FASTA shard, runs a forward pass with `output_hidden_states=True`
3. Extracts per-residue embeddings from the requested layer(s), removing CLS/EOS tokens
4. Concatenates all residues into a flat tensor of shape `(total_tokens, embedding_dim)`
5. Saves to `output_dir/layer_N/shard_name/activations.pt` with a `metadata.yaml` file

**Key parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--fasta_dir` | Directory containing FASTA shards | `data/training_shards` |
| `--output_dir` | Where to save embedding `.pt` files | `data/training_embeddings/esm2_8m` |
| `--embedder_type` | Embedder to use | `esm` |
| `--model_name` | HuggingFace model ID | `facebook/esm2_t6_8M_UR50D` |
| `--layers` | Layer indices to extract | `4` or `3 4 5` |
| `--batch_size` | Sequences per batch | `8` (doubled internally on GPU) |
| `--shard_index` | Process only one shard (for parallelism) | `0` |

**ESM2 model dimensions:**

| Model | Params | Layers | Hidden Dim |
|-------|--------|--------|------------|
| `esm2_t6_8M_UR50D` | 8M | 6 | 320 |
| `esm2_t12_35M_UR50D` | 35M | 12 | 480 |
| `esm2_t30_150M_UR50D` | 150M | 30 | 640 |
| `esm2_t33_650M_UR50D` | 650M | 33 | 1280 |
| `esm2_t36_3B_UR50D` | 3B | 36 | 2560 |
| `esm2_t48_15B_UR50D` | 15B | 48 | 5120 |

**Also extract embeddings for evaluation sequences** (used for fidelity evaluation during training):

```bash
python scripts/extract_embeddings.py \
    --fasta_dir data/eval_shards \
    --output_dir data/eval_embeddings/esm2_8m \
    --embedder_type esm \
    --model_name facebook/esm2_t6_8M_UR50D \
    --layers 4 \
    --batch_size 8
```

---

## Step 3: Create Evaluation Sequences File

The fidelity evaluation needs a text file with one sequence per line. The training scripts create this automatically from the eval FASTA if it doesn't exist, but you can also create it manually:

```bash
# The training scripts (train_basic_sae.py, train_matry_sae.py) auto-create this
# from data/eval_shards/shard_0.fasta → data/eval_sequences.txt (first 100 sequences)
```

---

## Step 4: Train a Single SAE

### Option A: ReLU SAE

```bash
export LAYER=4
export INTERPLM_DATA=data
export MODEL_DIR=models
python train_basic_sae.py
```

**Script:** `train_basic_sae.py`

This trains a standard ReLU SAE with these hardcoded defaults (designed for ESM2-8M):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `EMBEDDING_DIM` | 320 | ESM2-8M hidden dimension |
| `HIDDEN_SIZE` | 10,240 | 32x expansion factor |
| `BATCH_SIZE` | 2,048 | From the InterPLM paper |
| `LEARNING_RATE` | 2e-4 | |
| `STEPS` | 500,000 | |
| `L1_COEFFICIENT` | 0.06 | Sparsity penalty |
| `WARMUP_STEPS` | 50,000 | 10% of total steps |
| `DECAY_START` | 400,000 | Start LR decay at 80% |

**Data paths** (derived from environment variables):
- Training embeddings: `$INTERPLM_DATA/training_embeddings/esm2_8m/layer_$LAYER`
- Eval sequences: `$INTERPLM_DATA/eval_sequences.txt`
- Eval FASTA: `$INTERPLM_DATA/eval_shards/shard_0.fasta`
- Model output: `$MODEL_DIR/relu/layer_$LAYER`

**What happens during training:**
1. `DataloaderConfig` points to the pre-extracted embeddings directory
2. `ReLUTrainerConfig` configures the SAE architecture and optimizer
3. `ESMFidelityConfig` sets up loss-recovered evaluation (patches SAE reconstructions back into ESM2)
4. `WandbConfig` enables logging to W&B
5. `SAETrainingRun.from_config(config).run()` executes the training loop

**Outputs:**
- `models/relu/layer_4/ae.pt` — trained SAE weights
- `models/relu/layer_4/config.yaml` — full configuration

### Option B: Matryoshka Batch Top-K SAE

```bash
export LAYER=4
export INTERPLM_DATA=data
export MODEL_DIR=models
python train_matry_sae.py
```

**Script:** `train_matry_sae.py`

Same as ReLU but uses `MatryoshkaBatchTopKTrainerConfig` with:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `K` | 30 | Top-k sparsity |
| `FRACTIONS` | [1/32, 3/32, 7/32, 15/32, 6/32] | 5 nested groups |

**Outputs saved to:** `models/matryoshka/layer_$LAYER/`

---

## Step 5: Run a W&B Hyperparameter Sweep

### 5a. ReLU SAE Sweep

**Sweep config:** `relu_sweep.yaml`

```yaml
program: relu_sweep.py
method: bayes
metric:
  name: performance/pct_loss_recovered
  goal: maximize
parameters:
  learning_rate:     # log-uniform [5e-5, 1e-3]
  l1_coefficient:    # log-uniform [0.01, 0.2]
  hidden_size:       # [2560, 5120, 10240, 20480] (8x–64x expansion)
  batch_size:        # [1024, 2048, 4096]
  warmup_ratio:      # uniform [0.05, 0.15]
  decay_start_ratio: # uniform [0.7, 0.9]
early_terminate:
  type: hyperband
  min_iter: 100000
```

**Launch the sweep:**

```bash
# 1. Create the sweep on W&B
wandb sweep relu_sweep.yaml

# 2. Start an agent (runs on your machine)
export LAYER=4
export INTERPLM_DATA=data
export MODEL_DIR=models
wandb agent <your-entity>/<your-project>/<sweep-id>
```

**Sweep script:** `relu_sweep.py`

- Calls `wandb.init()` to receive hyperparameters from the sweep controller
- Reads `wandb.config.learning_rate`, `.l1_coefficient`, `.hidden_size`, `.batch_size`, `.warmup_ratio`, `.decay_start_ratio`
- Fixed: `EMBEDDING_DIM=320`, `STEPS=500000`
- Saves model to `models/sweeps/layer_$LAYER/<wandb_run_id>/`
- Evaluation runs every 100k steps
- Checkpoints saved every 50k steps

### 5b. Matryoshka SAE Sweep

**Sweep config:** `matryoshka_sweep_config.yaml`

```yaml
program: matryoshka_sweep.py
method: bayes
metric:
  name: fidelity/pct_loss_recovered
  goal: maximize
parameters:
  learning_rate:     # log-uniform [5e-5, 5e-4]
  k:                 # [10, 15, 20, 30, 40, 50]
  auxk_alpha:        # [1/32, 1/16, 1/8]
  hidden_size:       # [5120, 10240, 20480]
  batch_size:        # [1024, 2048, 4096]
  warmup_ratio:      # uniform [0.05, 0.15]
  decay_start_ratio: # uniform [0.7, 0.9]
  group_fractions:   # 16 different distributions (3–6 groups)
```

**Launch:**

```bash
wandb sweep matryoshka_sweep_config.yaml
export LAYER=4
wandb agent <your-entity>/<your-project>/<sweep-id>
```

**Sweep script:** `matryoshka_sweep.py`

- Same pattern as `relu_sweep.py` but uses `MatryoshkaBatchTopKTrainerConfig`
- Additional sweep parameters: `k`, `auxk_alpha`, `group_fractions`
- Saves to `models/matryoshka/layer_$LAYER/<wandb_run_id>/`

---

## Step 6: Evaluate a Trained SAE

```bash
python scripts/evaluate_sae.py \
    --sae_path models/relu/layer_4/ae.pt \
    --fasta_file data/eval_shards/shard_0.fasta \
    --model_name esm2_t6_8M_UR50D \
    --layer 4
```

This computes:
- **Fidelity** (% loss recovered): How well SAE reconstructions preserve ESM2's predictions
- **Reconstruction quality**: MSE, variance explained
- **Sparsity**: L0 (average active features), feature activation frequencies
- **Dead features**: Features that never activate

---

## Key Metrics to Monitor

| Metric | W&B Key | What it means |
|--------|---------|---------------|
| % Loss Recovered | `performance/pct_loss_recovered` or `fidelity/pct_loss_recovered` | How much of ESM2's prediction ability is preserved through the SAE |
| Variance Explained | `performance/variance_explained` | Fraction of embedding variance captured by reconstructions |
| L0 Sparsity | `performance/l0` | Average number of active SAE features per token |
| Dead Features (100 steps) | `features/dead_100_steps` | Features that haven't activated in last 100 steps |
| Dead Features (1000 steps) | `features/dead_1000_steps` | Features that haven't activated in last 1000 steps |

---

## File Structure Summary

```
data/
├── training_shards/           # FASTA shards for training
│   ├── shard_0.fasta
│   └── ...
├── eval_shards/               # Held-out FASTA for evaluation
│   └── shard_0.fasta
├── eval_sequences.txt         # One sequence per line (auto-created)
└── training_embeddings/
    └── esm2_8m/
        └── layer_4/
            ├── shard_0/
            │   ├── activations.pt    # (total_tokens, 320) tensor
            │   └── metadata.yaml
            └── shard_1/
                └── ...

models/
├── relu/
│   └── layer_4/
│       ├── ae.pt              # Trained SAE weights
│       └── config.yaml        # Training configuration
├── matryoshka/
│   └── layer_4/
│       └── ...
└── sweeps/
    └── layer_4/
        └── <wandb_run_id>/
            └── ...

# Root-level scripts:
train_basic_sae.py              # Single ReLU SAE training
train_matry_sae.py              # Single Matryoshka SAE training
relu_sweep.py                   # W&B sweep agent for ReLU
matryoshka_sweep.py             # W&B sweep agent for Matryoshka
relu_sweep.yaml                 # W&B sweep config (ReLU)
matryoshka_sweep_config.yaml    # W&B sweep config (Matryoshka)
scripts/extract_embeddings.py   # Embedding extraction
scripts/evaluate_sae.py         # SAE evaluation
```
