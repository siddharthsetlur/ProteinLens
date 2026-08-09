# ProteinLens

> **Acknowledgement.** The SAE training portion of this codebase
> (`proteinlens/sae/`, `proteinlens/train/`, and the top-level
> `train_basic_sae.py` / `train_matry_sae.py` / `relu_sweep.py` /
> `matryoshka_sweep.py` entrypoints) is **heavily based on
> [InterPLM](https://github.com/ElanaPearl/interPLM)** by Elana Simon
> and James Zou. Please cite their work if you use this training
> stack.

Code accompanying the ProteinLens paper. The repo supports three workflows:

1. **Train** Sparse Autoencoders (SAEs) on ESM2 activations.
2. **Run the feature pipeline** to compute per-feature activations, geometric
   features, InterPro / CATH / NMPFam annotation transfer, MEME motif PWM
   enrichment, and BH-corrected permutation null tests.
3. **Load the GeoPedia visualization** to explore SAE features across
   protein structures in a browser.

Trained SAE checkpoints and pre-computed analysis artifacts are released
separately (see [Data release](#data-release)) — the repo itself does
not carry them.

See [the paper reproduction guide](docs/paper_reproduction.md) for provenance
checks, the comparison tolerance, commands, and known missing inputs.

## Install

Requires CUDA-capable GPU and Conda.

```bash
git clone <repo-url> ProteinLens
cd ProteinLens
conda env create -f environment.yml
conda activate interplm
pip install -e .
```

## 1. Train SAEs

The paper uses ESM2-8M (`facebook/esm2_t6_8M_UR50D`) on layers 1–6, with
focus on layers 2 / 4 / 6.

**Single training run** (canonical hyperparameters):

```bash
# ReLU SAE
python train_basic_sae.py

# Matryoshka batch-top-k SAE
python train_matry_sae.py
```

Edit the config inside the script to point at your activation cache and
choose the layer / hidden dim / sparsity penalty.

**W&B sweeps** (used to produce the published checkpoints):

```bash
wandb sweep relu_sweep.yaml
wandb agent <sweep-id>

wandb sweep matryoshka_sweep_config.yaml
wandb agent <sweep-id>
```

The sweep YAMLs control the search space; the launcher scripts
(`relu_sweep.py`, `matryoshka_sweep.py`) are invoked by the agent.

To extract the activations the SAEs train on, the script
`scripts/extract_embeddings.py` writes per-shard activation tensors for
a chosen ESM2 layer over a FASTA corpus (typically a subset of
SwissProt — see `scripts/subset_fasta.py` and `scripts/shard_fasta.py`).

## 2. Run the feature pipeline

`scripts/run_feature_pipeline.py` is the orchestrator. It runs 14 stages
in order, each checkpointed so partial runs resume:

```
0a download                    fetch SwissProt FASTA
0b cluster                     MMseqs2 sequence clustering
1  survey                      stream proteins through ESM2 -> SAE
2  selection                   bin proteins by normalised activation
3  collection                  per-residue activation collection
4  assembly                    per-feature JSON files
5a interpro_selection          choose families for annotation transfer
5b interpro_fetch              fetch InterPro annotations
5c interpro_enrichment         family-level transfer enrichment
6a geometry_features           per-residue geometric features (writhe, curvature, torsion, ...)
6c geometry_residue_enrichment per-feature geometry classifiers + BH q-values
7  motif_enrichment            k-mer motif enrichment
7b motif_pwm                   MEME PWM motifs
8  position_enrichment         residue position enrichment
```

Run the whole pipeline:

```bash
python scripts/run_feature_pipeline.py \
    --paper-layer 4 \
    --output-dir <output-data-dir>
```

A specific stage:

```bash
python scripts/run_feature_pipeline.py --paper-layer 4 --stage geometry_features
```

A small local test (50 proteins):

```bash
python scripts/run_feature_pipeline.py --paper-layer 4 --max-proteins 50
```

Other paper-specific scripts under `scripts/` produce supplementary
analyses (cross-database transfer metrics, RMSD vs. PR-AUC, NMPFam
case studies, MEME case studies, CATH enrichment, refit GBM
permutation null, etc.). Each is documented in its own `--help`.

## 3. Load the visualization

GeoPedia is a single-page React app served by FastAPI. It expects each
SAE's analysis output (the per-feature pipeline outputs from step 2)
under `trained_models/layer_<N>/<run>/analysis/`.

**Multi-layer mode** (recommended): create an `analysis/` directory at
the repo root with `l2/`, `l4/`, `l6/` subdirectories (the symlink layout
used in the paper):

```bash
mkdir -p analysis
ln -s ../trained_models/layer_2/<run>/analysis analysis/l2
ln -s ../trained_models/layer_4/<run>/analysis analysis/l4
ln -s ../trained_models/layer_6/<run>/analysis analysis/l6

python -m proteinlens.viz --analysis-root analysis --port 8050
```

**Single-layer mode**:

```bash
python -m proteinlens.viz \
    --analysis-dir trained_models/layer_4/<run>/analysis \
    --port 8050
```

Then open http://localhost:8050.

## Data release

`trained_models/`, `feature_data_*/`, `results/`, `models/`, `wandb/`,
and the `analysis/` symlink dir are gitignored — they are distributed
separately:

- **Trained SAEs** and **pre-computed analysis directories**: <HF / Zenodo link>
- **Sample feature_data** for testing (`feature_data_test_500/`): <HF / Zenodo link>

After downloading, place each archive under the matching path at the
repo root (`trained_models/layer_4/<run>/`, etc.) and the viz / pipeline
will pick it up.

## Repo layout

```
proteinlens/                    main package
    sae/                        SAE classes (ReLU, Matryoshka batch-top-k)
    train/                      training run, configs, checkpoint manager, trainers
    embedders/                  ESM2 embedder
    analysis/
        feature_pipeline/       14-stage pipeline modules
        geometry/               geometric feature classifiers
        concepts/               concept analysis
    viz/                        FastAPI server + React SPA
        static/index.html       SPA entry
        static/geopedia/        SPA components (jsx)
        static/js/mol_viewer.js 3Dmol wrapper
scripts/                        pipeline driver + supplementary analyses
protein_results/geometry/       residue-level geometric primitives (live-imported)
tests/                          test suite (some integration tests need a
                                local PDB cache and feature_data fixture;
                                see test files for paths)
k8s/                            Dockerfile + INSTRUCTIONS.md for cluster runs
                                (per-run job YAMLs are templated locally)
docs/                           paper-companion docs
```

## Citation

Citation will be added after review.
