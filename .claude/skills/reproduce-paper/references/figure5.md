# Figure 5 — contact-prediction ablation

Three panels, all layer-4-labelled in the caption:

| Panel | Claim | Script |
|---|---|---|
| Left | Ablation strength vs normalized signed target delta for f/4714, 11 strengths, top-20 activating proteins | `benchmark_contact_predictor_scale_sweep.py` |
| Middle | Contact probability by sequence-separation bin vs ablation strength; strongest decrease at bins [25,48] | same sweep |
| Right | 495 proteins across 92 features; target shift vs matched control | `benchmark_contact_predictor_ablation.py` |

Figures come from `generate_contact_scale_sweep_case_study_figures.py` and
`generate_contact_predictor_ablation_figures.py`.

## Read this before running anything

**Figure 5 does not come from the paper's released runs.** Its inputs are:

| | Figure 5 | The paper's layer 4 (Tables 1-4) |
|---|---|---|
| SAE | `trained_models/layer_3/fiery-sweep` (`fiery-sweep-34`) | `trained_models/layer_4/frosty-sweep-15` |
| `layer_idx` | 3 | 4 |
| Dictionary | 5,120 | 10,240 |
| Analysis dir | `feature_data_cluster/` | `.../frosty-sweep-15/analysis/` |
| In the HF release? | **No** | Yes |

The paper's three layers are uniform at 10,240 features. `fiery-sweep` is not one
of them — it is a separate, earlier, half-width layer-3 run, confirmed four ways:

| Check | `fiery-sweep` | The three paper runs |
|---|---|---|
| `ae.pt` `encoder.weight` | **(5120, 320)** | (10240, 320) |
| `expansion_factor` | 16.0 | 32.0 |
| `ae.pt` size | 13 MB | 26 MB |
| `dataset_stats.num_features` | 5,120 (`esm_layer: 3`) | 10,240 |

W&B identity `fiery-sweep-34` / `xkd1maao`, trained on
`training_embeddings/esm2_8m/layer_3`. It appears in neither `paper_manifest.yaml`
nor the release. So this is a different SAE, not the same model under another
layer-indexing convention — the caption says "at layer 4", the weights say 5,120
features at `layer_idx: 3`.

**Therefore "f/4714" in Figure 5 indexes a 5,120-wide dictionary and is not the
same latent as f/4714 in the layer-4 release.**

The identification is not guesswork. The paper names f/4714's strongest geometric
predictor as "mean sequence separation at 8Å", and:

- `feature_data_cluster` f/4714 → `top_geometric_feature: mean_seq_sep_contact_8A` ✓
- `frosty-sweep-15` f/4714 → `contact_density_12A`, `geom_pr_auc` 0.10, `is_geometry_primary: false` ✗

**Consequence: Figure 5 cannot be reproduced from the Hugging Face release.**
Neither `feature_data_cluster/` nor `trained_models/layer_3/fiery-sweep/` is
published. Say so plainly rather than substituting the layer-4 run — substituting
changes the model, the dictionary, and the feature's identity.

## Environment

The `interplm` env cannot run these scripts: its `torch` package reports 2.1.2
while `torch-2.9.1` and `torch-2.11.0` dist-infos also sit in site-packages, and
the installed `transformers` needs `torch.utils._pytree.register_pytree_node`,
added in torch 2.2. It fails at import. This is the env drift
`docs/reproducibility_code_audit.md` flagged.

Requirements: `torch >= 2.2`, `transformers`, plus `h5py`, `einops`, `biopython`,
`numpy`, `scipy`, `scikit-learn`, `matplotlib`.

**No GPU and no ESMFold are needed.** The contact path uses ESM2's own
`predict_contacts` head, not the folding stack, and ESM2-8M runs fine on CPU —
a two-case probe finished in well under a minute. Pass `--device cpu`.

## 1. Parity run

Selection gates are **bypassed** when `--feature-ids` is explicit
(`benchmark_contact_predictor_ablation.py:236-245`); only the contact-descriptor
filter still applies. That is what makes the f/4714 case study runnable — f/4714
would not survive the default gates.

```bash
python scripts/benchmark_contact_predictor_ablation.py \
    --data-dir feature_data_cluster \
    --sae-dir trained_models/layer_3/fiery-sweep \
    --layer 3 --device cpu \
    --feature-ids 4714 \
    --proteins-per-feature 20 \
    --output-dir results/contact_predictor_ablation
```

`--proteins-per-feature 20` matches the paper's "top 20 activating proteins".

## 2. Scale sweep — panels left and middle

```bash
python scripts/benchmark_contact_predictor_scale_sweep.py \
    --source-results-dir results/contact_predictor_ablation \
    --data-dir feature_data_cluster \
    --sae-dir trained_models/layer_3/fiery-sweep \
    --layer 3 --device cpu \
    --case-specs 4714:<ACCESSION> \
    --output-dir results/contact_scale_sweep
```

`--ablation-scales` defaults to `1.0 0.9 … 0.1 0.0` — exactly the paper's 11
strengths from no ablation to full. Do not pass a custom grid; the default *is*
the published setting.

Take `<ACCESSION>` from the parity run's outputs. Without `--case-specs` the
sweep auto-selects via `balanced_signed_target_delta`, which is a different
estimand from "the paper's chosen case".

## 3. Right panel — does not reproduce

The paper reports **92 features across 495 proteins**. A default-gate run of
`benchmark_contact_predictor_ablation.py` over `feature_data_cluster` yields
**5 features / 20 protein-cases**.

This is not a tuning gap. Of 5,000 analysed features: 2,220 carry a contact
descriptor, 41 are also geometry-primary, 5 pass the script's default gates
(`concordance_f1 >= 0.60`, `PR-AUC >= 0.80`, `position_f1 <= 0.10`). A sweep over
`is_geometry_primary` on/off x f1 in {0,.3,.4,.5,.6} x PR-AUC in {0,.3,.5,.6,.7,.8}
produces no 92-feature population from this run at all.

The cause is identifiable. `benchmark_contact_predictor_ablation.py` accepts
`--feature-ids-file`, and explicit ids **bypass every gate** (lines 236-245).
Separately `generate_contact_predictor_ablation_figures.py` aborts on a missing
`results/contact_top_feature_ids_ranked.txt`, which exists in neither the working
tree nor any commit in git history. The 92 features were supplied as a curated
external list that was never published — which is why no gate setting recovers
them, and why the right panel's figure generator cannot run at all.

**Do not tune gates toward 92.** That fits the target and changes the estimand.

## 4. Figures

```bash
python scripts/generate_contact_scale_sweep_case_study_figures.py \
    --results-dir results/contact_scale_sweep --output-dir reproduction_outputs
python scripts/generate_contact_predictor_ablation_figures.py \
    --results-dir results/contact_predictor_ablation --output-dir reproduction_outputs
```

Check each generator's `--help` first; their defaults point at
`results/contact_predictor_ablation_1000`, a run directory that is not in the repo.

## Measured outcome — run of 2026-08-19

Repo `main` @ `8c91c61`, CPU, 16 cases x 11 ablation strengths (176 samples).

| Panel | Result |
|---|---|
| Left | **Reproduced.** Mean target delta rises monotonically 0 -> +3.109 across the 11 strengths; Pearson r = **+0.9982**. All 16 proteins shift the same direction at full ablation (+1.32 to +6.58). |
| Middle | **Reproduced.** Mean contact-probability change at full ablation by band: 3-8 -0.0065, 9-12 -0.0060, 13-24 -0.0123, **25-48 -0.0297**, 49-96 -0.0008, 97+ -0.0002. The [25,48] band is 2.4x the next strongest and decreases monotonically with strength. |
| Right | **Not reproduced.** 5 features / 20 cases vs the paper's 92 / 495. |

The middle panel's bins were **not** imposed to obtain that result. No
sequence-separation binning exists in any shipped script, so the profile was
computed per separation and ranked: the 12 most-responsive separations are
33, 34, 32, 38, 35, 39, 26, 27, 31, 36, 21, 25 — 11 of 12 inside [25,48], peak at
33. The band falls out of the data independently.

**Deviation:** the paper says "top 20 activating proteins"; 16 of the 20-protein
pool survive the eligibility filters, so the case study ran on 16. Not corrected —
raising `--top-sequence-pool` would change the population the paper described.

Full report: `docs/figure5_reproduction.md`.

## What to report

- Panels left/middle: reproduced, **on inputs absent from the release**. Always
  pair the result with that caveat — "reproduces" and "reproducible by a reviewer"
  are different claims here.
- Panel right: not reproduced; name the missing curated feature-id list.
- Provenance: Figure 5's SAE is `fiery-sweep-34`, `layer_idx` 3, 5,120 features.
  This is the headline finding, more important than any number the run produces.
