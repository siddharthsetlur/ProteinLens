# Reproducibility refactor report

Date: 2026-08-09  
Branch: reproducibility-audit-refactor  
Input audit: docs/reproducibility_code_audit.md

## Scientific policy

Changes expose estimands and provenance; they do not target manuscript values.
Percentages use a declared 1.0 percentage-point practical tolerance, with raw
counts and exact deltas retained. No threshold, layer, checkpoint, denominator,
significance source, or snapshot is chosen based on closeness to the paper.

The local candidates are the correct kind of main-table data. Both
dataset_stats.json and the independent SAE configs identify layers 2, 4, and
6. Every run has 10,240 features and the same 556,340-protein,
72,528-cluster survey. The legacy layer-3 fiery-sweep run is not used. The
exact paper release is nevertheless unidentified, so the manifest labels the
local artifacts unresolved and does not invent checksums.

## Changes and justification

### Statistical sources

- scripts/paper_tables.py restores readable Table 1 and 2 generation. It
  recomputes BH q-values from fixed-score raw permutation p-values separately
  by layer and method, preventing mixed raw and stale cached sources.
- scripts/compute_geometry_primary.py now has explicit fixed and refit modes.
  Fixed-score is the paper default; refit is separate robustness analysis.
  Feature-wise fallback between estimands was removed.
- Table 3 and 4 builders independently recompute fixed-score q-values used for
  their gates. Existing geometry-primary files supply scores and labels, not
  unquestioned significance.
- New null files record threshold_steps and a provenance version. Resume
  refuses missing or incompatible grid/permutation metadata, preventing mixed
  snapshots. Historical files remain explicitly unverifiable.

### Table and figure estimands

- Table 3 statistics are computed before the GeoPedia group cap. The output
  distinguishes all eligible groups from displayed groups and records the
  cosine threshold and q source.
- Table 3 now reads MEME/PWM enrichment and refers to paper Figure 3.
- Table 4 family and sequence unions use all qualifying hits. Top 25 remains a
  display limit only. Conflicting family sequence counts fail loudly, and all
  manuscript percentages and denominators are emitted.
- Figure 6 has a deterministic generator using strict PR-AUC above 0.3 and
  descriptor importance above 0.1 rules from its caption.
- Appendix Table 7 GBM settings are centralized and unit tested.
- Position documentation reports 22 predicates, and paper pipeline threshold
  defaults are pinned to 100.

### Identity and operational interfaces

- paper_manifest.yaml names canonical runs, data paths, denominators, figure
  examples, primary null, tolerance, and the missing collaborator dependency.
- scripts/verify_paper_manifest.py checks identity in two metadata sources and
  emits SHA-256 fingerprints. Strict mode cannot pass until release resolution.
- The feature-pipeline CLI no longer silently targets legacy layer 3.
  paper-layer selects 2, 4, or 6; ad-hoc use requires SAE path and ESM layer.
- A CPU reproduction environment was added. The GPU environment includes
  missing test/server dependencies and uses a compatible NumPy version.
- Packaging includes GeoPedia assets, runtime dependencies, and a console
  entry point. Pytest is restricted to tests, registers integration, and
  excludes network integration by default.
- Stale unscoped GeoPedia tests were replaced with the active layer API.

### Existing cleanup reused

The branch was fast-forwarded through cleanup/publication. That lineage removes
tracked bytecode, OS files, logs, obsolete ESM3/ESMC follow-ons, the legacy
multi-page frontend, scratch plans, and misleading ESMFold intervention code;
it also expands ignores and adds the publication README. Untracked analyses
were not added, moved, or deleted.

## Deliberately outstanding

- Figure 5 is excluded because code and inputs remain with the collaborator.
  Reconstructing it from prose or substituting ESMFold changes the experiment.
- Figures 1 through 4 are not claimed exactly reproducible without editable
  sources, camera state, traces, and panel layout.
- Tables 5 and 6 are not generated because the pinned W&B export is unknown.
- Strict artifact validation awaits the correct release snapshot and checksums.
- A compiled offline SPA bundle remains outstanding and does not affect tables.

## Discrepancy classification

Values within 1.0 percentage point are practically reproduced, not exact.
Larger differences are classified first as wrong run identity, different or
incomplete snapshot, missing null provenance, changed estimand, or unexplained.
Only a manuscript-supported estimand error warrants code correction. None
justifies tuning code to a target.
