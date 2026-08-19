# Mode B — re-run the pipeline from scratch

Rebuilds the artifacts instead of downloading them: ESM2 activations → SAE →
14-stage feature pipeline → permutation nulls → tables.

**Confirm with the user before starting.** This is a GPU-cluster job measured in
days, and Mode A answers the reproduction question in under an hour.

## Cost, from the actual runs

From the W&B exports for the published runs. Those exports are **not tracked in
the repo** (W&B run artifacts were deliberately untracked in commit `02b0b8e`), so
the figures below are reproduced here rather than cited to a file:

| Stage | Measured |
|---|---|
| SAE training, layer 4 (`frosty-sweep-15`, run `86tryizu`) | 5.8 h on one A100 |
| Longest finished feature-pipeline runs | 40.4 h, 22.9 h, 14.0 h, 10.2 h |
| Permutation null, layer 4 | 1.2 h to 12.5 h depending on metric set |
| Refit-GBM null (robustness only, not the paper's primary) | 106 h |

Plus a full W&B sweep if you want the published hyperparameters rather than
reusing them, and ~40 GB of intermediates per layer. Most of the crashed rows in
those exports are OOM kills — expect to babysit.

## What will and will not match

**Nothing will be bit-identical.** MMseqs2 clustering, GBM fits, and permutation
draws are stochastic, and the InterPro/AlphaFold/UniProt fetches hit live
databases whose content has moved since the paper's snapshot. The comparison
target is the **tables**, under the same 1.0 pp tolerance as Mode A — never a file
diff, never a checksum against the release.

If a from-scratch run disagrees with the paper by more than tolerance, you cannot
separate "the method doesn't reproduce" from "the annotation databases changed"
without also running Mode A on the released artifacts. Run Mode A first so you
have the controlled comparison.

## Prerequisites

- CUDA GPU, `conda env create -f environment.yml`, `conda activate geopedia`
- MMseqs2 (clustering) and MEME Suite (PWM motifs) on `PATH`
- Network access to UniProt, InterPro (EBI), and AlphaFold
- ~40 GB free per layer

Pinning the annotation databases matters more than anything else here. Record the
InterPro release and the UniProt/SwissProt release date you fetched; without them
a later disagreement is uninterpretable.

## Smoke test first — always

```bash
python scripts/run_feature_pipeline.py --paper-layer 4 --max-proteins 50
```

Minutes, not days. Shakes out missing binaries, auth, and disk before you commit
to the real run.

## 1. Activations

`scripts/subset_fasta.py` and `scripts/shard_fasta.py` prepare the corpus;
`scripts/extract_embeddings.py` writes per-shard activation tensors for one ESM2
layer. The paper uses `facebook/esm2_t6_8M_UR50D`, layers 2 / 4 / 6.

Pre-stage `swissprot_all.fasta` with `curl --retry-all-errors --continue-at -`
rather than relying on the UniProt stream endpoint mid-pipeline — it fails often
enough to cost a run.

## 2. Train the SAE

README §1, and `docs/esm2_sae_training_guide.md` for the details.

```bash
python train_basic_sae.py                  # single run; edit the config in-file
wandb sweep relu_sweep.yaml && wandb agent <sweep-id>   # published checkpoints
```

Published dictionaries are 10,240 features (expansion factor 32 over ESM2-8M's
320 dims). Per-run hyperparameters are in each released `config.yaml` — lr,
`l1_penalty`, and batch size differ per layer, so copy them rather than assuming.

## 3. Feature pipeline (14 stages)

```bash
python scripts/run_feature_pipeline.py --paper-layer 4 --output-dir <analysis-dir>
```

Every stage checkpoints into `pipeline_state.json`, so a re-run resumes. Single
stage: `--stage geometry_features`. Stage order is in README §2.

Stage 6b (`geometry_protein_enrichment`) was intentionally removed from the
pipeline — do not restore it.

## 4. Permutation nulls — the step the README omits

Not one of the 14 stages. It must run after the pipeline and before any table:

```bash
python scripts/compute_permutation_null.py \
    --data-dir <analysis-dir> \
    --n-permutations 100 \
    --threshold-steps 100 \
    --include-pwm \
    --workers <n>
```

- `--include-pwm` is **required**. Table 1's MEME Motif row reads `pwm_pr_auc`,
  which is only emitted with that flag.
- `--threshold-steps 100` matches the manuscript. The default already reads 100
  from `PipelineConfig.interpro_f1_threshold_steps`; pass it explicitly anyway, so
  the value is recorded in every output file. The released snapshot predates that
  provenance field, which is exactly why the layer-2 Table 1 discrepancy cannot be
  diagnosed today — don't recreate that gap.
- Within-protein permutation, one-sided p-values (Phipson & Smyth 2010),
  per-feature seed = `--seed` + feature_id.

Then derive the geometry-primary classification the tables read:

```bash
python scripts/compute_geometry_primary.py --data-dir <analysis-dir>
```

This writes `geometry_primary_analysis.json`. It reads the enrichment outputs, so
it must come after stages 5c/6c/7b/8 and after the nulls.

## 5. Tables

Identical to Mode A step 4 — point the generators at your new analysis directory
via `--analysis LABEL=PATH` / `--data-dir` / `--analysis-dir`.
