# Paper reproduction

Artifact identity, numerical regeneration, and manuscript comparison are
separate steps. A close numerical match is not evidence that the right data
were used.

## Obtain the artifacts

The paper's artifacts are released on Hugging Face — see the *Data release*
section of the README for the three repos. They are private until publication.

Download into the repo root; the released layout matches what every generator
expects by default:

```bash
hf download Sidd2010/proteinlens-paper-artifacts --repo-type dataset --local-dir . \
  --include "trained_models/*/*/analysis/permutation_null.tar.zst" \
            "trained_models/*/*/analysis/geometry_primary_analysis.json" \
            "trained_models/*/*/analysis/dataset_stats.json"

find trained_models -name '*.tar.zst' -print -execdir tar --zstd -xf {} \; -delete
```

That subset (~95 MB) is enough for Tables 1 and 2. Tables 3 and 4 and Figure 6
need more, and Table 3 and Figure 6 read `geometry_enrichment/`,
`cath_enrichment/`, and `survey_coverage.json` from the *geopedia-analysis* repo
rather than *paper-artifacts*. The per-target download map, with sizes, is in
`.claude/skills/reproduce-paper/references/artifacts.md`, and `/reproduce-paper`
drives the whole sequence.

## Validate candidate artifacts

Run:

    python scripts/verify_paper_manifest.py \
      --output reproduction_outputs/artifact_identity.json

This checks dataset_stats.json and each SAE config.yaml. Candidate inputs must
independently identify ESM2-8M layers 2, 4, and 6, 10,240-feature dictionaries,
and the runs in paper_manifest.yaml.

Strict validation additionally requires a release snapshot ID and checksums:

    python scripts/verify_paper_manifest.py --strict

It is expected to fail until the external paper artifact release is identified.
Do not call local candidate artifacts the published snapshot while it fails.

## Regenerate lightweight artifacts

Tables 1 and 2:

    python scripts/paper_tables.py \
      --output reproduction_outputs/tables_1_2.json

The generator applies BH independently by method and layer to raw fixed-score
permutation p-values. It does not use cached adjusted p-values or refit-GBM
robustness q-values.

Table 3, repeated per layer:

    python scripts/build_subdomain_case_study.py \
      --data-dir trained_models/layer_4/frosty-sweep-15/analysis \
      --output reproduction_outputs/layer4_table3.json

The Table 3 block uses every eligible InterPro-residue group. The 100-group
limit applies only to the GeoPedia display payload.

Table 4, repeated per layer:

    python scripts/build_nmpfam_transfer_summary.py \
      --analysis-dir trained_models/layer_4/frosty-sweep-15/analysis \
      --output reproduction_outputs/layer4_table4.json

Family unions use every qualifying hit, not the top-25 display list.

Table 4's `nmpfam/nmpfam_enrichment/` inputs were absent when
`docs/reproduction_attempt_report.md` was written, which is why that report
records Table 4 as not regenerable. The release carries them for **layers 2 and
6** — layer 6 holds 9,313 per-feature files, exactly the paper's 90.95% of
10,240 — so Table 4 is now reproducible for those layers.

Layer 4 is **not**: its released `nmpfam_enrichment/` holds 284 files covering
feature ids 8933-10239, against the ~7,904 its own cached
`nmpfam_transfer_summary.json` was built from. The generator does not fail on the
partial input; it silently returns 2.73% where the paper reports 77.78%. The
datastore copy is itself partial, so this is not a transfer defect and re-running
the transfer will not fix it. Check the file count before trusting the output.

Figure 6:

    python scripts/figure6_descriptor_counts.py \
      --output-dir reproduction_outputs

This writes CSV, JSON, SVG, and PNG. It fixed-q gates the geometric
annotations, selects each feature's single highest-importance descriptor
when its importance exceeds 0.1, and stacks counts by the two reported
PR-AUC bins.

## Comparison policy

For reported percentages, differences no larger than 1.0 percentage point are
treated as practically reproduced. Always retain the raw count, denominator,
exact delta, and artifact fingerprint. The tolerance is a reporting rule only:
generators never rescale, select inputs, or change gates to approach the paper.

Missing-data and snapshot mismatches are reported separately from method
discrepancies. Legacy null files do not record their threshold grid, so they
cannot establish whether the manuscript's stated 100-step grid or historical
code's 50-step grid generated the snapshot.

## Known exclusions

- Figure 5's contact-prediction ablation code is now in the repo, and its left and
  middle panels reproduce — but not from the release. It uses an unpublished
  layer-3 SAE (`fiery-sweep`, 5,120 features), not the paper's layer-4 run, and
  its right panel does not reproduce. See `docs/figure5_reproduction.md`.
- Exact renderers/editable sources for Figures 1 through 4 are unavailable.
  Known feature and protein identities are frozen in paper_manifest.yaml.
- Tables 5 and 6 need a pinned W&B export that is not currently identified.
