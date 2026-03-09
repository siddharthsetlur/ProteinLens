# Training SAEs on ESMC Embeddings — Step-by-Step Guide

This document describes the end-to-end pipeline for training a Sparse Autoencoder (SAE) on a specific layer of ESMC (ESM Cambrian), from data preparation through hyperparameter sweeps with Weights & Biases.

ESMC is EvolutionaryScale's representation learning model — the successor to ESM2 for protein embeddings. It uses the same sequence-in, embeddings-out paradigm as ESM2 but with improved representations. The SAE training pipeline is identical to ESM2; only the embedding extraction step uses a different model.

---

## Prerequisites

- Conda environment created from `environment.yml` with the `esm` package (v3.x) installed
  ```bash
  pip install esm  # EvolutionaryScale's ESM package (provides ESMC)
  ```
- A collection of protein sequences in FASTA format
- Access to a GPU (recommended)
- A Weights & Biases account (for sweeps)

---

## Pipeline Overview

```
FASTA sequences
  → (1) Shard & subset data
  → (2) Extract embeddings from an ESMC layer
  → (3) Train SAE (single run or W&B sweep)
  → (4) Evaluate trained SAE
```

This is the same pipeline as ESM2 (see `docs/esm2_sae_training_guide.md`). The only difference is Step 2 uses the ESMC embedder instead of the ESM embedder.

---

## ESMC Model Details

| Model | Params | Layers | Hidden Dim | Local? |
|-------|--------|--------|------------|--------|
| `esmc_300m` | 300M | 30 | 960 | Yes |
| `esmc_600m` | 600M | 36 | 1152 | Yes |
| `esmc_6b` | 6B | 80 | — | Forge API only |

**Default model:** `esmc_300m` (960-dimensional embeddings, 30 layers)

**Layer indexing:** 0-indexed. Layer 0 is the output of the first transformer block, layer 29 is the output of the last block (for `esmc_300m`).

---

## Step 1: Prepare Your Data

Identical to ESM2. See `docs/esm2_sae_training_guide.md` Step 1.

```bash
# Subset (optional)
python scripts/subset_fasta.py \
    --input_fasta data/uniprot.fasta \
    --output_fasta data/uniprot_subset.fasta \
    --max_length 1024 \
    --num_sequences 100000

# Shard
python scripts/shard_fasta.py \
    --input_fasta data/uniprot_subset.fasta \
    --output_dir data/training_shards \
    --num_shards 10
```

Keep one shard aside for evaluation (e.g. `data/eval_shards/`).

---

## Step 2: Extract ESMC Embeddings

Use the same `scripts/extract_embeddings.py` script, but with `--embedder_type esmc`:

```bash
# Training embeddings
python scripts/extract_embeddings.py \
    --fasta_dir data/training_shards \
    --output_dir data/training_embeddings/esmc_300m \
    --embedder_type esmc \
    --model_name esmc_300m \
    --layers 15 \
    --batch_size 8

# Evaluation embeddings
python scripts/extract_embeddings.py \
    --fasta_dir data/eval_shards \
    --output_dir data/eval_embeddings/esmc_300m \
    --embedder_type esmc \
    --model_name esmc_300m \
    --layers 15 \
    --batch_size 8
```

**What this does:**

1. Loads the ESMC model via the `ESMC_Embedder` class (`proteinlens/embedders/esmc.py`)
2. For each FASTA shard, runs a forward pass through the transformer
3. Extracts per-residue hidden states from the requested layer(s), removing CLS/EOS tokens
4. Concatenates all residues into a flat tensor of shape `(total_tokens, 960)`
5. Saves to `output_dir/layer_N/shard_name/activations.pt` with a `metadata.yaml` file

**Key parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--fasta_dir` | Directory containing FASTA shards | `data/training_shards` |
| `--output_dir` | Where to save embedding `.pt` files | `data/training_embeddings/esmc_300m` |
| `--embedder_type` | Must be `esmc` | `esmc` |
| `--model_name` | ESMC model identifier | `esmc_300m` or `esmc_600m` |
| `--layers` | Layer indices to extract (0-indexed) | `15` or `10 15 20` |
| `--batch_size` | Sequences per batch | `8` (doubled internally on GPU) |
| `--shard_index` | Process only one shard (for parallelism) | `0` |

**Choosing a layer:** ESMC-300M has 30 layers (0–29). A middle layer (e.g. 15) is a reasonable starting point. You can extract multiple layers in one pass to compare.

---

## Step 3: Train a Single SAE

### Option A: ReLU SAE

```bash
export LAYER=15
export INTERPLM_DATA=data
export MODEL_DIR=models
python train_basic_sae_esmc.py
```

**Script:** `train_basic_sae_esmc.py`

Hardcoded defaults for ESMC-300M:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `EMBEDDING_DIM` | 960 | ESMC-300M hidden dimension |
| `HIDDEN_SIZE` | 30,720 | 32x expansion factor |
| `BATCH_SIZE` | 2,048 | |
| `LEARNING_RATE` | 2e-4 | |
| `STEPS` | 500,000 | |
| `L1_COEFFICIENT` | 0.06 | Sparsity penalty |
| `WARMUP_STEPS` | 50,000 | 10% of total steps |
| `DECAY_START` | 400,000 | Start LR decay at 80% |

**Data paths** (derived from environment variables):
- Training embeddings: `$INTERPLM_DATA/training_embeddings/esmc_300m/layer_$LAYER`
- Eval embeddings: `$INTERPLM_DATA/eval_embeddings/esmc_300m/layer_$LAYER`
- Model output: `$MODEL_DIR/esmc_relu/layer_$LAYER`

**Evaluation:** Uses reconstruction metrics only (variance explained, L0 sparsity). Fidelity evaluation (loss recovered) is not yet implemented for ESMC — this requires an ESMC-specific intervention function. Reconstruction metrics are sufficient for hyperparameter optimization.

**Outputs:**
- `models/esmc_relu/layer_15/ae.pt` — trained SAE weights
- `models/esmc_relu/layer_15/config.yaml` — full configuration

### Option B: Matryoshka Batch Top-K SAE

```bash
export LAYER=15
export INTERPLM_DATA=data
export MODEL_DIR=models
python train_matry_sae_esmc.py
```

**Script:** `train_matry_sae_esmc.py`

Same as ReLU but uses `MatryoshkaBatchTopKTrainerConfig` with:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `K` | 30 | Top-k sparsity |
| `FRACTIONS` | [1/32, 3/32, 7/32, 15/32, 6/32] | 5 nested groups |

**Outputs saved to:** `models/esmc_matryoshka/layer_$LAYER/`

---

## Step 4: Run a W&B Hyperparameter Sweep

### 4a. ReLU SAE Sweep

**Sweep config:** `relu_sweep_esmc.yaml`

```yaml
program: relu_sweep_esmc.py
method: bayes
metric:
  name: performance/variance_explained
  goal: maximize
parameters:
  learning_rate:     # log-uniform [5e-5, 1e-3]
  l1_coefficient:    # log-uniform [0.01, 0.2]
  hidden_size:       # [7680, 15360, 30720, 61440] (8x–64x expansion for 960d)
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
wandb sweep relu_sweep_esmc.yaml

# 2. Start an agent
export LAYER=15
export INTERPLM_DATA=data
export MODEL_DIR=models
wandb agent <your-entity>/<your-project>/<sweep-id>
```

**Sweep script:** `relu_sweep_esmc.py`

- Fixed: `EMBEDDING_DIM=960`, `STEPS=500000`
- Saves models to `models/esmc_sweeps/relu/layer_$LAYER/<wandb_run_id>/`
- Optimizes `performance/variance_explained` (since fidelity is not available for ESMC)

### 4b. Matryoshka SAE Sweep

**Sweep config:** `matryoshka_sweep_config_esmc.yaml`

```yaml
program: matryoshka_sweep_esmc.py
method: bayes
metric:
  name: performance/variance_explained
  goal: maximize
parameters:
  learning_rate:     # log-uniform [5e-5, 5e-4]
  k:                 # [10, 15, 20, 30, 40, 50]
  auxk_alpha:        # [1/32, 1/16, 1/8]
  hidden_size:       # [15360, 30720, 61440] (16x–64x expansion)
  batch_size:        # [1024, 2048, 4096]
  warmup_ratio:      # uniform [0.05, 0.15]
  decay_start_ratio: # uniform [0.7, 0.9]
  group_fractions:   # 13 different distributions (3–6 groups)
```

**Launch:**

```bash
wandb sweep matryoshka_sweep_config_esmc.yaml
export LAYER=15
wandb agent <your-entity>/<your-project>/<sweep-id>
```

---

## Differences from ESM2 Pipeline

| Aspect | ESM2 | ESMC |
|--------|------|------|
| Embedder type | `--embedder_type esm` | `--embedder_type esmc` |
| Model name | `facebook/esm2_t6_8M_UR50D` | `esmc_300m` |
| Hidden dim | 320 (8M model) | 960 (300M model) |
| SAE hidden size | 10,240 (32x) | 30,720 (32x) |
| Sweep hidden sizes | [2560, 5120, 10240, 20480] | [7680, 15360, 30720, 61440] |
| Fidelity eval | Yes (ESMFidelityConfig) | No (reconstruction metrics only) |
| Sweep metric | `performance/pct_loss_recovered` | `performance/variance_explained` |
| Training scripts | `train_basic_sae.py` | `train_basic_sae_esmc.py` |
| Sweep scripts | `relu_sweep.py` | `relu_sweep_esmc.py` |
| Sweep configs | `relu_sweep.yaml` | `relu_sweep_esmc.yaml` |
| Embedding paths | `data/.../esm2_8m/layer_N` | `data/.../esmc_300m/layer_N` |
| Model output paths | `models/relu/layer_N` | `models/esmc_relu/layer_N` |

Everything else (SAE architectures, training loop, data loading, checkpointing, W&B logging) is identical.

---

## Key Metrics to Monitor

| Metric | W&B Key | What it means |
|--------|---------|---------------|
| Variance Explained | `performance/variance_explained` | Fraction of embedding variance captured by reconstructions |
| L0 Sparsity | `performance/l0` | Average number of active SAE features per token |
| Dead Features (100 steps) | `features/dead_100_steps` | Features that haven't activated in last 100 steps |
| Dead Features (1000 steps) | `features/dead_1000_steps` | Features that haven't activated in last 1000 steps |

---

## File Structure Summary

```
data/
├── training_shards/                 # FASTA shards for training
├── eval_shards/                     # Held-out FASTA for evaluation
└── training_embeddings/
    └── esmc_300m/
        └── layer_15/
            ├── shard_0/
            │   ├── activations.pt   # (total_tokens, 960) tensor
            │   └── metadata.yaml
            └── ...

models/
├── esmc_relu/
│   └── layer_15/
│       ├── ae.pt                    # Trained SAE weights
│       └── config.yaml
├── esmc_matryoshka/
│   └── layer_15/
│       └── ...
└── esmc_sweeps/
    ├── relu/
    │   └── layer_15/<run_id>/
    └── matryoshka/
        └── layer_15/<run_id>/

# ESMC-specific scripts (root directory):
train_basic_sae_esmc.py              # Single ReLU SAE training
train_matry_sae_esmc.py              # Single Matryoshka SAE training
relu_sweep_esmc.py                   # W&B sweep agent for ReLU
matryoshka_sweep_esmc.py             # W&B sweep agent for Matryoshka
relu_sweep_esmc.yaml                 # W&B sweep config (ReLU)
matryoshka_sweep_config_esmc.yaml    # W&B sweep config (Matryoshka)

# Shared scripts (no changes needed):
scripts/extract_embeddings.py        # Works with --embedder_type esmc
scripts/shard_fasta.py               # Data preparation
scripts/subset_fasta.py              # Data preparation

# New embedder (in proteinlens package):
proteinlens/embedders/esmc.py        # ESMC_Embedder class
```

---

## Future Work

- **ESMC fidelity evaluation:** Implement an `ESMCFidelityConfig` / `ESMCFidelityFunction` analogous to the ESM2 version, which patches SAE reconstructions back into the ESMC model and measures cross-entropy loss recovered. This requires nnsight integration with the ESMC architecture.
- **ESMC-600M support:** The embedder already supports `esmc_600m` (1152-dim, 36 layers). Training scripts would need adjusted `EMBEDDING_DIM` and `HIDDEN_SIZE` values.
