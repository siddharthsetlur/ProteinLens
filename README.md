# ProteinLens

> **Acknowledgement.** The SAE training portion of this codebase
> (`proteinlens/sae/`, `proteinlens/train/`, and the top-level
> `train_basic_sae.py` / `train_matry_sae.py` / `relu_sweep.py` /
> `matryoshka_sweep.py` entrypoints) is **heavily based on
> [InterPLM](https://github.com/ElanaPearl/interPLM)** by Elana Simon
> and James Zou. Please cite their work if you use this training
> stack.

Code accompanying the [GeoPedia paper](https://arxiv.org/abs/2608.26419). The repo supports three workflows:

1. **Train** Sparse Autoencoders (SAEs) on masked-LM activations (the paper
   uses ESM2).
2. **Run the feature pipeline** to compute per-feature activations, geometric
   features, InterPro / CATH / NMPFam annotation transfer, MEME motif PWM
   enrichment, and BH-corrected permutation null tests.
3. **Load the GeoPedia visualization** to explore SAE features across
   protein structures in a browser.

Trained SAE checkpoints and pre-computed analysis artifacts are released
separately (see [Data release](#data-release)) — the repo itself does
not carry them.

## Try it now — [geopedia.studio](https://geopedia.studio/)

A live GeoPedia instance is running at **<https://geopedia.studio/>**. Explore
the SAE features across protein structures in the browser — no install, no
downloads, no artifacts to fetch.

Run it locally only if you want to point it at your own analysis directory; see
[Load the visualization](#3-load-the-visualization) below.

## Reproducing the paper — start here

You do **not** need to retrain anything or run the pipeline. The paper's numbers
regenerate from released artifacts on CPU, in well under an hour.

In a checkout, run the bundled Claude Code skill:

```
/reproduce-paper
```

It asks which result you want, downloads only the artifacts that result needs,
extracts them, verifies artifact identity, regenerates the number, and writes a
comparison report to `reproduction_outputs/`.

Prefer to do it by hand? Every step is an ordinary script —
[docs/paper_reproduction.md](docs/paper_reproduction.md) has the commands, the
comparison policy, and the known exclusions.

Full details, the four modes and their costs, and the currently-open
discrepancies are in [Reproducing the paper](#reproducing-the-paper) below.

## Works with any masked protein language model

Nothing in the pipeline is ESM-specific. The paper uses ESM2-8M
(`facebook/esm2_t6_8M_UR50D`), but the SAE, the feature pipeline, and the
visualization operate on hidden-state activations from any masked protein
language model, *mutatis mutandis*. `proteinlens/embedders/` is a plugin
interface for exactly this — see
[proteinlens/embedders/README.md](proteinlens/embedders/README.md) for adding one.

The obvious things to keep straight when swapping the model:

- **Activation dimension** — `trainer_cfg.activation_dim` must match the model's
  hidden size (320 for ESM2-8M), and `dictionary_size` follows from your chosen
  expansion factor.
- **Layer indexing** — layer numbering is per-model, and the interesting layers
  differ; do not assume the paper's 2 / 4 / 6 transfer.
- **Tokenization** — per-residue alignment between activations and sequence
  positions is what the geometry and annotation transfer rely on. Models with
  different special-token conventions need that offset handled.
- **Retraining** — the released SAEs are tied to ESM2-8M and cannot be reused
  against a different model; you need your own training run and your own
  permutation nulls.

## Install

Requires CUDA-capable GPU and Conda.

```bash
git clone <repo-url> ProteinLens
cd ProteinLens
conda env create -f environment.yml
conda activate geopedia
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

> A hosted instance is already running at <https://geopedia.studio/> — use that
> unless you need to browse your own analysis output.

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
separately, on Hugging Face:

| Repo | Type | Contents |
|---|---|---|
| [`Sidd2010/proteinlens-sae-esm2-8m`](https://huggingface.co/Sidd2010/proteinlens-sae-esm2-8m) | model | Trained SAEs (layers 2/4/6), run configs |
| [`Sidd2010/proteinlens-paper-artifacts`](https://huggingface.co/datasets/Sidd2010/proteinlens-paper-artifacts) | dataset | Null tests, enrichment, NMPFam — Tables 1–4 and Figure 6 |
| [`Sidd2010/proteinlens-geopedia-analysis`](https://huggingface.co/datasets/Sidd2010/proteinlens-geopedia-analysis) | dataset | Per-feature payloads for the visualizer, plus `geometry_enrichment` / `cath_enrichment` (needed by Table 3 and Figure 6) |


The layout mirrors the repo, so a download drops straight in at the repo root
(`trained_models/layer_4/frosty-sweep-15/analysis/...`) with no path rewriting.
Directories ship as one `.tar.zst` each — a Hugging Face repo caps at 10,000 files
per directory and several of these hold more. Extract in place:

```bash
sh EXTRACT.sh    # find . -name '*.tar.zst' -execdir tar --zstd -xf {} \; -delete
```

## Reproducing the paper

The quick start is [at the top](#reproducing-the-paper--start-here). This section
is the detail.

The skill lives in `.claude/skills/reproduce-paper/` and is version-controlled
with the code it drives. Name the target you want, e.g.
`/reproduce-paper Tables 1 and 2` or `/reproduce-paper Table 4, layer 6`. Four
modes:

| Mode | What it does | Cost |
|---|---|---|
| **A** (default) | Tables 1–4 and Figure 6 from the released artifacts | CPU, 30–90 min, 32 MB – 17 GB depending on target |
| **B** | Re-run the pipeline from raw sequences | GPU cluster, days–weeks — opt-in only |
| **C** | ESM2-35M layer 6 generalization check (not a paper claim) | CPU, minutes |
| **D** | Figure 5 contact-prediction ablation | CPU, ~15 min |

Download only what you need — the per-target artifact map, with verified sizes
and expected file counts, is in
`.claude/skills/reproduce-paper/references/artifacts.md`. Note that Table 3 and
Figure 6 read from the `geopedia-analysis` repo, not `paper-artifacts`.



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

<!-- PLACEHOLDER — replace with the real entry before release. -->

If you use GeoPedia, the released SAEs, or the analysis artifacts, please cite:

```bibtex
@article{
  title={Interpreting Latent Protein Language Model Features with Geometric Annotations}, 
      author={Siddharth Setlur and Djordje Mihajlovic and Darrick Lee},
      year={2026},
      eprint={2608.26419},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2608.26419},
}
```

Please also cite [InterPLM](https://github.com/ElanaPearl/interPLM) (Simon &
Zou), which the SAE training stack is based on — see the acknowledgement at the
top of this file.
