# Figure 5 reproduction — contact-prediction ablation

Run date: 2026-08-19 · Repo: `main` @ `8c91c61` · Paper: `docs/28622_Interpreting_Latent_Prot.pdf`

## Verdict

| Panel | Claim | Result |
|---|---|---|
| Left | Target delta varies with ablation strength for f/4714 | **Reproduced** — r = +0.9982 over 11 strengths |
| Middle | Contacts drop most at sequence separations [25,48] | **Reproduced** — peak response at sep 33; 11 of the 12 most-responsive separations lie in [25,48] |
| Right | 495 proteins across 92 features vs matched control | **Not reproduced** — population is 20 cases / 5 features; the defining input is unpublished |

Reproduced **on inputs that are not in the Hugging Face release.** See Provenance.

## Provenance — the headline finding

Figure 5 does not come from the paper's released runs.

| | Figure 5 | Paper's layer 4 (Tables 1-4) |
|---|---|---|
| SAE | `trained_models/layer_3/fiery-sweep` (`fiery-sweep-34`) | `trained_models/layer_4/frosty-sweep-15` |
| `layer_idx` | 3 | 4 |
| Dictionary | 5,120 | 10,240 |
| Analysis dir | `feature_data_cluster/` | `.../frosty-sweep-15/analysis/` |
| Published? | **No** | Yes |

Different dictionary sizes means a different SAE, not the same model under another
layer-indexing convention. The caption reads "at layer 4"; the config reads
`layer_idx: 3`, `dictionary_size: 5120`.

The identification is positive, not inferred from the caption. The paper names
f/4714's strongest geometric predictor as "mean sequence separation at 8Å":

- `feature_data_cluster` f/4714 → `mean_seq_sep_contact_8A` ✓
- `frosty-sweep-15` f/4714 → `contact_density_12A`, PR-AUC 0.10, not geometry-primary ✗

**Figure 5 cannot be reproduced from the release.** It reproduces only on a machine
that already holds the two unpublished inputs. No layer-4 substitution was made.

## Method

Environment: `interplm` cannot run these scripts — its `torch` package reports
2.1.2 while `torch-2.9.1` and `torch-2.11.0` dist-infos also sit in site-packages,
and `transformers` 4.57 needs `torch.utils._pytree.register_pytree_node` (torch
≥ 2.2). Used an isolated venv inheriting `causalab` (torch 2.9.0, transformers
4.57.1) plus `einops` and `biopython`. **CPU only** — the contact path uses ESM2's
own `predict_contacts` head, not the folding stack. No GPU, no ESMFold.

```
scripts/benchmark_contact_predictor_ablation.py  --feature-ids 4714 --proteins-per-feature 20
scripts/benchmark_contact_predictor_scale_sweep.py  --case-specs <16 accessions>
scripts/generate_contact_scale_sweep_case_study_figures.py
```
with `--data-dir feature_data_cluster --sae-dir trained_models/layer_3/fiery-sweep --layer 3 --device cpu`.

`--ablation-scales` left at its default `1.0 0.9 … 0.1 0.0` — exactly the paper's
"11 strengths from no ablation to full ablation". Nothing was tuned.

**Deviation:** the paper says "top 20 activating proteins"; 16 of the 20-protein
pool survive the eligibility filters, so the case study runs on 16. Not corrected —
raising `--top-sequence-pool` would change the population the paper described.

## Left panel — reproduced

Mean signed change in the target metric (`patch_weighted_mean_seq_sep`), 16 cases:

| Strength | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|
| Mean Δ | 0.000 | +0.729 | +1.242 | +1.812 | +2.435 | +3.109 |

Monotone across all 11 strengths, **Pearson r = +0.9982**, 176 samples. All 16
proteins shift in the same direction at full ablation (+1.32 to +6.58). This is the
paper's "linear relationship … suggesting a causal relationship".

## Middle panel — reproduced

Mean Δ contact probability vs no-ablation, by separation band (|i−j| ≥ 3):

| Strength | 3-8 | 9-12 | 13-24 | **25-48** | 49-96 | 97+ |
|---|---|---|---|---|---|---|
| 0.5 | −0.0042 | −0.0047 | −0.0061 | **−0.0140** | −0.0002 | −0.0001 |
| 1.0 | −0.0065 | −0.0060 | −0.0123 | **−0.0297** | −0.0008 | −0.0002 |

The [25,48] band is 2.4× the next strongest and decreases monotonically with
strength — the paper's "most strongly decreases predicted contacts at localized
intermediate sequence separations, in particular bins [25,48]".

The paper's bins were **not** imposed to obtain this. No sequence-separation
binning exists in any shipped script, so the profile was computed per separation
and ranked. The 12 most-responsive separations are 33, 34, 32, 38, 35, 39, 26, 27,
31, 36, 21, 25 — 11 of 12 inside [25,48], peak at 33. The band falls out of the
data independently.

## Right panel — not reproduced

| | Paper | This run |
|---|---:|---:|
| Features | 92 | **5** |
| Protein cases | 495 | **20** |

Not a tuning gap. In `feature_data_cluster` (5,000 features): 2,220 have a contact
descriptor, 41 are also geometry-primary, and 5 pass the script's default gates
(`concordance_f1 ≥ 0.60`, `PR-AUC ≥ 0.80`, `position_f1 ≤ 0.10`). A sweep over
`is_geometry_primary` × f1 ∈ {0,.3,.4,.5,.6} × PR-AUC ∈ {0,.3,.5,.6,.7,.8} yields
**no 92-feature population from this run at all**.

The likely explanation is concrete: `benchmark_contact_predictor_ablation.py`
accepts `--feature-ids-file`, and explicit ids **bypass every gate**
(`benchmark_contact_predictor_ablation.py:236-245`). Separately,
`generate_contact_predictor_ablation_figures.py` fails outright on a missing
`results/contact_top_feature_ids_ranked.txt`, which is in neither the working tree
nor any commit in git history. So the 92 features were supplied as a curated
external list that was never published, which is exactly why no gate setting
recovers them.

Gates were **not** tuned toward 92. Doing so would fit the target and change the
estimand.

## What would close the gap

1. Publish `feature_data_cluster/` and `trained_models/layer_3/fiery-sweep/`, or
   state in the paper that Figure 5 uses a different, unreleased SAE.
2. Publish the 92-feature id list and `contact_top_feature_ids_ranked.txt`.
3. Reconcile the caption's "layer 4" with `layer_idx: 3` / 5,120 features.

## Artifacts

Under the session scratchpad (heavy outputs kept out of the repo):
`fig5/parity/` (16 cases), `fig5/sweep/` (16 × 11 scales, contact maps),
`fig5/figures/` (17 PNG/PDF pairs + summary), `fig5/parity_right/` (20 cases).
