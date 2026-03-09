# Training SAEs on ESM3-open Embeddings — Step-by-Step Guide

This document describes the end-to-end pipeline for training a Sparse Autoencoder (SAE) on a specific layer of ESM3-open (1.4B), from data preparation through hyperparameter sweeps with Weights & Biases.

ESM3 is EvolutionaryScale's multi-modal protein generative model. When used for embedding extraction, we pass sequence-only input (structure, function, and other tracks are automatically mask/padded) and extract per-layer hidden states from the shared TransformerStack. The SAE training pipeline is identical to ESM2 and ESMC; only the embedding extraction step differs.

---

## Prerequisites

- Conda environment created from `environment.yml` with the `esm` package (v3.x) installed
  ```bash
  pip install esm  # EvolutionaryScale's ESM package
  ```
- **HuggingFace authentication** (ESM3-open is a gated model):
  1. Create an account at [huggingface.co](https://huggingface.co)
  2. Go to [EvolutionaryScale/esm3-sm-open-v1](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) and accept the license agreement
  3. Log in locally:
     ```bash
     conda activate interplm
     huggingface-cli login
     ```
- A collection of protein sequences in FASTA format
- Access to a GPU (recommended — 1.4B model is large)
- A Weights & Biases account (for sweeps)

---

## Pipeline Overview

```
FASTA sequences
  → (1) Shard & subset data
  → (2) Extract embeddings from an ESM3 layer
  → (3) Train SAE (single run or W&B sweep)
  → (4) Evaluate trained SAE
```

This is the same pipeline as ESM2 (see `docs/esm2_sae_training_guide.md`) and ESMC (see `docs/esmc_sae_training_guide.md`). The only difference is Step 2 uses the ESM3 embedder.

---

## ESM3-open Model Details

| Model | Params | Layers | Hidden Dim | Access |
|-------|--------|--------|------------|--------|
| `esm3_sm_open_v1` | 1.4B | 48 | 1536 | HuggingFace (gated, free) |

**Note:** ESM3 medium (7B) and large (98B) are only available via EvolutionaryScale's Forge API and are NOT supported by this embedder (which requires local weights).

**Layer indexing:** 0-indexed. Layer 0 is the output of the first transformer block, layer 47 is the output of the last block.

### How ESM3 embedding extraction works

ESM3 is a multi-modal model that jointly processes sequence, structure, secondary structure, SASA, function, and residue annotation tracks. When we extract embeddings for SAE training:

1. Only sequence tokens are provided as real input
2. All other tracks (structure, function, SS8, SASA, etc.) are filled with mask/pad tokens — this is the same default behavior as ESM3's own `forward()` method
3. The encoder fuses all input tracks into a single embedding per residue
4. The `TransformerStack` processes these embeddings through 48 layers
5. We extract per-layer hidden states directly from the TransformerStack (bypassing the output heads), since `ESM3.forward()` discards them

The hidden states represent the model's internal representation conditioned on sequence alone, which is suitable for SAE training.

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

## Step 2: Extract ESM3 Embeddings

Use the same `scripts/extract_embeddings.py` script, but with `--embedder_type esm3`:

```bash
# Training embeddings
python scripts/extract_embeddings.py \
    --fasta_dir data/training_shards \
    --output_dir data/training_embeddings/esm3_open \
    --embedder_type esm3 \
    --model_name esm3_sm_open_v1 \
    --layers 24 \
    --batch_size 4

# Evaluation embeddings
python scripts/extract_embeddings.py \
    --fasta_dir data/eval_shards \
    --output_dir data/eval_embeddings/esm3_open \
    --embedder_type esm3 \
    --model_name esm3_sm_open_v1 \
    --layers 24 \
    --batch_size 4
```

**Note:** ESM3 (1.4B params) requires significantly more GPU memory than ESM2-8M or ESMC-300M. Use smaller batch sizes (4–8) and consider using `--shard_index` for parallel processing across multiple GPUs.

**Key parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--embedder_type` | Must be `esm3` | `esm3` |
| `--model_name` | ESM3 model identifier | `esm3_sm_open_v1` |
| `--layers` | Layer indices to extract (0-indexed, 0–47) | `24` or `20 24 30` |
| `--batch_size` | Sequences per batch (use smaller for ESM3) | `4` |

**Choosing a layer:** ESM3 has 48 layers (0–47). A middle layer (e.g. 24) is a reasonable starting point. You can extract multiple layers in one pass to compare.

---

## Step 3: Train a Single SAE

### Option A: ReLU SAE

```bash
export LAYER=24
export INTERPLM_DATA=data
export MODEL_DIR=models
python train_basic_sae_esm3.py
```

**Script:** `train_basic_sae_esm3.py`

Hardcoded defaults for ESM3-open:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `EMBEDDING_DIM` | 1536 | ESM3-open hidden dimension |
| `HIDDEN_SIZE` | 49,152 | 32x expansion factor |
| `BATCH_SIZE` | 2,048 | |
| `LEARNING_RATE` | 2e-4 | |
| `STEPS` | 500,000 | |
| `L1_COEFFICIENT` | 0.06 | Sparsity penalty |
| `WARMUP_STEPS` | 50,000 | 10% of total steps |
| `DECAY_START` | 400,000 | Start LR decay at 80% |

**Data paths:**
- Training embeddings: `$INTERPLM_DATA/training_embeddings/esm3_open/layer_$LAYER`
- Eval embeddings: `$INTERPLM_DATA/eval_embeddings/esm3_open/layer_$LAYER`
- Model output: `$MODEL_DIR/esm3_relu/layer_$LAYER`

**Evaluation:** Uses reconstruction metrics only (variance explained, L0 sparsity). Fidelity evaluation (loss recovered) is not yet implemented for ESM3.

### Option B: Matryoshka Batch Top-K SAE

```bash
export LAYER=24
export INTERPLM_DATA=data
export MODEL_DIR=models
python train_matry_sae_esm3.py
```

**Script:** `train_matry_sae_esm3.py`

Same as ReLU but uses `MatryoshkaBatchTopKTrainerConfig` with:

| Parameter | Value |
|-----------|-------|
| `K` | 30 |
| `FRACTIONS` | [1/32, 3/32, 7/32, 15/32, 6/32] |

**Outputs saved to:** `models/esm3_matryoshka/layer_$LAYER/`

---

## Step 4: Run a W&B Hyperparameter Sweep

### 4a. ReLU SAE Sweep

**Sweep config:** `relu_sweep_esm3.yaml`

```yaml
program: relu_sweep_esm3.py
method: bayes
metric:
  name: performance/variance_explained
  goal: maximize
parameters:
  learning_rate:     # log-uniform [5e-5, 1e-3]
  l1_coefficient:    # log-uniform [0.01, 0.2]
  hidden_size:       # [12288, 24576, 49152, 98304] (8x–64x expansion for 1536d)
  batch_size:        # [1024, 2048, 4096]
  warmup_ratio:      # uniform [0.05, 0.15]
  decay_start_ratio: # uniform [0.7, 0.9]
early_terminate:
  type: hyperband
  min_iter: 100000
```

**Launch:**

```bash
wandb sweep relu_sweep_esm3.yaml

export LAYER=24
export INTERPLM_DATA=data
export MODEL_DIR=models
wandb agent <your-entity>/<your-project>/<sweep-id>
```

### 4b. Matryoshka SAE Sweep

**Sweep config:** `matryoshka_sweep_config_esm3.yaml`

```bash
wandb sweep matryoshka_sweep_config_esm3.yaml
export LAYER=24
wandb agent <your-entity>/<your-project>/<sweep-id>
```

---

## Comparison: ESM2 vs ESMC vs ESM3

| Aspect | ESM2 (8M) | ESMC (300M) | ESM3-open (1.4B) |
|--------|-----------|-------------|-------------------|
| Type | Sequence model | Sequence model | Multi-modal (seq+struct+func) |
| Hidden dim | 320 | 960 | 1536 |
| Layers | 6 | 30 | 48 |
| SAE hidden size (32x) | 10,240 | 30,720 | 49,152 |
| Embedder type flag | `esm` | `esmc` | `esm3` |
| Auth required | No | No | Yes (HuggingFace gated) |
| GPU memory (batch=4) | ~1 GB | ~2 GB | ~6+ GB |
| Fidelity eval | Yes | No | No |
| Sweep metric | `pct_loss_recovered` | `variance_explained` | `variance_explained` |
| Training scripts | `train_basic_sae.py` | `train_basic_sae_esmc.py` | `train_basic_sae_esm3.py` |
| Embedding paths | `.../esm2_8m/...` | `.../esmc_300m/...` | `.../esm3_open/...` |
| Model output paths | `models/relu/...` | `models/esmc_relu/...` | `models/esm3_relu/...` |

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
    └── esm3_open/
        └── layer_24/
            ├── shard_0/
            │   ├── activations.pt   # (total_tokens, 1536) tensor
            │   └── metadata.yaml
            └── ...

models/
├── esm3_relu/
│   └── layer_24/
│       ├── ae.pt                    # Trained SAE weights
│       └── config.yaml
├── esm3_matryoshka/
│   └── layer_24/
│       └── ...
└── esm3_sweeps/
    ├── relu/
    │   └── layer_24/<run_id>/
    └── matryoshka/
        └── layer_24/<run_id>/

# ESM3-specific scripts (root directory):
train_basic_sae_esm3.py              # Single ReLU SAE training
train_matry_sae_esm3.py              # Single Matryoshka SAE training
relu_sweep_esm3.py                   # W&B sweep agent for ReLU
matryoshka_sweep_esm3.py             # W&B sweep agent for Matryoshka
relu_sweep_esm3.yaml                 # W&B sweep config (ReLU)
matryoshka_sweep_config_esm3.yaml    # W&B sweep config (Matryoshka)

# New embedder (in proteinlens package):
proteinlens/embedders/esm3.py        # ESM3Embedder class

# Shared scripts (no changes needed):
scripts/extract_embeddings.py        # Works with --embedder_type esm3
```

---

## Future Work

- **ESM3 fidelity evaluation:** Implement fidelity measurement by patching SAE reconstructions back into ESM3's transformer and measuring masked language modeling cross-entropy loss recovered.
- **Multi-modal embeddings:** Currently only sequence tokens are used. Future work could explore SAE training on ESM3 hidden states conditioned on structure + sequence input, which may reveal different features.
- **Larger ESM3 models:** The Forge API could enable embedding extraction from ESM3-medium (7B) and ESM3-large (98B) without local GPU requirements.
