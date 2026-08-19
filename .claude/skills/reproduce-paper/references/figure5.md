# Figure 5 — contact-prediction ablation

Three panels:

| Panel | What it shows | Script |
|---|---|---|
| Left | Ablation strength vs normalized signed target delta for f/4714, 11 strengths | `benchmark_contact_predictor_scale_sweep.py` |
| Middle | Contact probability by sequence-separation bin vs ablation strength | same sweep |
| Right | Target shift vs matched control across many features | `benchmark_contact_predictor_ablation.py` |

Figures come from `generate_contact_scale_sweep_case_study_figures.py` and
`generate_contact_predictor_ablation_figures.py`.

## Environment

Needs `torch >= 2.2`, `transformers`, `h5py`, `einops`, `biopython`.

**No GPU and no ESMFold.** The contact path uses ESM2's own `predict_contacts`
head rather than the folding stack, and ESM2-8M runs fine on CPU — a two-case
probe finishes in under a minute. Pass `--device cpu`.

## 1. Parity run

```bash
python scripts/benchmark_contact_predictor_ablation.py \
    --data-dir <analysis-dir> \
    --sae-dir <sae-dir> \
    --layer <N> --device cpu \
    --feature-ids 4714 \
    --proteins-per-feature 20 \
    --output-dir results/contact_predictor_ablation
```

`--proteins-per-feature 20` matches the paper's "top 20 activating proteins";
fewer may survive the eligibility filters, and the run reports how many.

Selection gates are bypassed when `--feature-ids` is explicit
(`benchmark_contact_predictor_ablation.py:236-245`); only the contact-descriptor
filter still applies.

## 2. Scale sweep — panels left and middle

```bash
python scripts/benchmark_contact_predictor_scale_sweep.py \
    --source-results-dir results/contact_predictor_ablation \
    --data-dir <analysis-dir> \
    --sae-dir <sae-dir> \
    --layer <N> --device cpu \
    --case-specs 4714:<ACCESSION> \
    --output-dir results/contact_scale_sweep
```

`--ablation-scales` defaults to `1.0 0.9 … 0.1 0.0` — exactly the paper's 11
strengths from no ablation to full. Do not pass a custom grid; the default *is*
the published setting.

Take `<ACCESSION>` from the parity run's outputs. Without `--case-specs` the
sweep auto-selects via `balanced_signed_target_delta`, a different estimand from
"the paper's chosen case".

## 3. Figures

```bash
python scripts/generate_contact_scale_sweep_case_study_figures.py \
    --results-dir results/contact_scale_sweep --output-dir reproduction_outputs
python scripts/generate_contact_predictor_ablation_figures.py \
    --results-dir results/contact_predictor_ablation --output-dir reproduction_outputs
```

Check each generator's `--help` first — their defaults point at run directories
that may not exist locally.

## Notes

- The middle panel's sequence-separation binning is not implemented in any
  shipped script; compute the profile per separation from the saved contact maps.
- f/4714 refers to a specific dictionary — confirm the feature's top geometric
  descriptor matches the paper's before reporting a case study under that id.
