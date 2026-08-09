# Paper reproduction

Artifact identity, numerical regeneration, and manuscript comparison are
separate steps. A close numerical match is not evidence that the right data
were used.

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

- Figure 5 contact-map ablation code remains with the collaborator. No ESMFold
  steering experiment is substituted.
- Exact renderers/editable sources for Figures 1 through 4 are unavailable.
  Known feature and protein identities are frozen in paper_manifest.yaml.
- Tables 5 and 6 need a pinned W&B export that is not currently identified.
