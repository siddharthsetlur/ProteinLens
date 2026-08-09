# Independent paper reproduction attempt

Date: 2026-08-09  
Tested base refactor commit: `6ed0791711c5d673a76914c7cd1e1637f103eb58`;
the reproduction-driven Figure 6 correction is documented below and included
in the follow-up commit.  
Paper: `docs/28622_Interpreting_Latent_Prot.pdf`

## Conclusion

The local artifacts have the correct declared run identities: ESM2-8M
(`facebook/esm2_t6_8M_UR50D`), layers 2, 4, and 6 (not layer 3), 10,240-feature
dictionaries, and runs `firm-sweep-3`, `frosty-sweep-15`, and
`major-sweep-15`. Each layer has 556,340 proteins and 72,528 clusters.
Non-strict manifest validation passes. Exact paper-snapshot identity remains
unresolved because `paper_manifest.yaml` has no release snapshot ID or release
checksums; strict validation fails for only those reasons.

No input was chosen for numerical agreement and no gate or denominator was
changed to approach the manuscript. Tables 1--3 use raw fixed-score
permutation p-values with BH applied independently per annotation method and
layer. Differences of at most **1.0 percentage point (pp)** are classified as
practically reproduced, while exact deltas, counts, and denominators are still
reported.

Main findings:

- Table 1 reproduces throughout layers 4 and 6. Several layer-2 values are
  1.15--2.30 pp high and do not reproduce.
- All Table 2 values reproduce within tolerance.
- Table 3 is exact at layers 2 and 6; layer 4 differs by one group,
  447/578 versus 448/578 (-0.17 pp).
- Table 4 cannot be regenerated because its raw NMPFam enrichment inputs are
  absent. Cached summaries are close, but lack q-source and snapshot
  provenance and therefore are not an independent reproduction.
- Figure 6 now reproduces the descriptor distribution closely under the
  paper-defined fixed-q population and PR-AUC strata. Exact counts remain
  modestly lower (for example 255 versus 271), consistent with the unresolved
  local-versus-paper snapshot rather than a changed plotting estimand.
- Figures 1--4 have underlying artifacts to varying degrees but no exact
  deterministic renderer. Figure 5 was deliberately ignored.
- Appendix Tables 7 and 8 are supported by source. Tables 5 and 6 are only
  partially supported because pinned W&B/evaluation artifacts are absent.

## Identity and statistical provenance

Identity evidence and local SHA-256 fingerprints are in
`reproduction_outputs/artifact_identity.json`. The strict result is in
`reproduction_outputs/artifact_identity_strict.json`.

| Layer | Null JSON files | Tested: InterPro protein/residue, position, geometry | Tested: PWM | Missing threshold-grid metadata |
|---:|---:|---:|---:|---:|
| 2 | 9,309 | 9,309 each | 9,004 | 9,309 |
| 4 | 9,587 | 9,587 each | 9,579 | 9,587 |
| 6 | 9,743 | 9,743 each | 9,697 | 9,743 |

The files report 100 permutations but none records `threshold_steps`.
Consequently, they cannot prove whether the paper's 100-step threshold grid or
the historical 50-step grid generated this snapshot.

## Table 1: annotation coverage

Source: `reproduction_outputs/tables_1_2.json`; q source:
`fixed_score_permutation_raw_p`; denominator: 10,240 dictionary features.
Delta is the unrounded computed percentage minus the published percentage.

| Layer | Measure | Raw | Computed % | Paper % | Delta pp | Within 1 pp? |
|---:|---|---:|---:|---:|---:|:---:|
| 2 | Total union | 9,014/10,240 | 88.0273 | 86.53 | +1.4973 | No |
| 2 | InterPro protein | 5,746/10,240 | 56.1133 | 56.11 | +0.0033 | Yes |
| 2 | InterPro residue | 4,422/10,240 | 43.1836 | 42.03 | +1.1536 | No |
| 2 | Position | 6,535/10,240 | 63.8184 | 62.22 | +1.5984 | No |
| 2 | MEME motif | 7,335/10,240 | 71.6309 | 69.33 | +2.3009 | No |
| 2 | Geometric | 8,873/10,240 | 86.6504 | 84.38 | +2.2704 | No |
| 4 | Total union | 9,574/10,240 | 93.4961 | 93.50 | -0.0039 | Yes |
| 4 | InterPro protein | 5,811/10,240 | 56.7480 | 56.75 | -0.0020 | Yes |
| 4 | InterPro residue | 6,293/10,240 | 61.4551 | 61.46 | -0.0049 | Yes |
| 4 | Position | 8,109/10,240 | 79.1895 | 79.19 | -0.0005 | Yes |
| 4 | MEME motif | 7,942/10,240 | 77.5586 | 77.56 | -0.0014 | Yes |
| 4 | Geometric | 9,502/10,240 | 92.7930 | 92.79 | +0.0030 | Yes |
| 6 | Total union | 9,704/10,240 | 94.7656 | 94.57 | +0.1956 | Yes |
| 6 | InterPro protein | 5,804/10,240 | 56.6797 | 56.68 | -0.0003 | Yes |
| 6 | InterPro residue | 6,933/10,240 | 67.7051 | 67.70 | +0.0051 | Yes |
| 6 | Position | 7,328/10,240 | 71.5625 | 71.35 | +0.2125 | Yes |
| 6 | MEME motif | 7,915/10,240 | 77.2949 | 77.20 | +0.0949 | Yes |
| 6 | Geometric | 9,678/10,240 | 94.5117 | 94.49 | +0.0217 | Yes |

Because model/layer/run identity is correct and layer 4 exactly matches under
the fixed-score estimand, the layer-2 result is classified **snapshot
incompleteness/difference**, not wrong identity. Missing threshold-grid
metadata prevents a firmer diagnosis.

## Table 2: geometric PR-AUC bins

The denominator is geometrically q-significant features that also have a score
in `geometry_primary_analysis.json`.

| Layer | Bin | Raw/denominator | Computed % | Paper % | Delta pp |
|---:|---|---:|---:|---:|---:|
| 2 | 0.0--0.3 | 7,521/8,641 | 87.0385 | 87.39 | -0.3515 |
| 2 | 0.3--0.6 | 685/8,641 | 7.9273 | 7.74 | +0.1873 |
| 2 | >0.6 | 435/8,641 | 5.0341 | 4.88 | +0.1541 |
| 4 | 0.0--0.3 | 5,982/9,476 | 63.1279 | 63.64 | -0.5121 |
| 4 | 0.3--0.6 | 2,209/9,476 | 23.3115 | 23.08 | +0.2315 |
| 4 | >0.6 | 1,285/9,476 | 13.5606 | 13.28 | +0.2806 |
| 6 | 0.0--0.3 | 7,880/9,676 | 81.4386 | 81.43 | +0.0086 |
| 6 | 0.3--0.6 | 1,533/9,676 | 15.8433 | 15.87 | -0.0267 |
| 6 | >0.6 | 263/9,676 | 2.7181 | 2.70 | +0.0181 |

All reproduce within tolerance. There are 8,873/9,502/9,678 geometric
q-significant features in layers 2/4/6, so 232/26/2 respectively lack a score
and are excluded from these denominators. That incompleteness is explicitly
retained rather than imputed.

## Table 3: InterPro groups split by geometry

The refactored generator computes over every eligible InterPro-residue group;
the 100-group cap affects only the display payload. A group is called
distinguishable when its mean pairwise geometry-importance cosine similarity
is below 0.5.

| Layer | Computed | Paper | Delta pp | Geometry/InterPro q tests |
|---:|---:|---:|---:|---:|
| 2 | 71/120 = 59.17% | 71/120 = 59.17% | 0.00 | 9,309 / 9,309 |
| 4 | 447/578 = 77.34% | 448/578 = 77.51% | -0.17 | 9,587 / 9,587 |
| 6 | 31/45 = 68.89% | 31/45 = 68.89% | 0.00 | 9,743 / 9,743 |

Outputs are `reproduction_outputs/layer2_table3.json`,
`layer4_table3.json`, and `layer6_table3.json`. This is practically
reproduced. The prior 100-group result was a **changed estimand** caused by a
presentation cap; the remaining one-group layer-4 difference is an
**unexplained snapshot difference within tolerance**.

## Table 4: NMPFam transfer

Regeneration stopped because
`analysis/nmpfam/nmpfam_enrichment/*.json` is absent for all three runs. No
other local directory contains the required per-hit profiles, and the
layer-4 `nmpfam_annotation` directory is a different artifact and was not
substituted.

The following values are cached-summary comparisons only. The caches do not
record q source, tested-feature count, or input snapshot checksums.

| Layer | Activation raw/% (paper; delta) | q-sig among hits raw/% (paper; delta) | Median PR-AUC >.5 raw/% (paper; delta) | Families | Sequences |
|---:|---|---|---|---|---|
| 2 | 6,473/10,240 = 63.21 (63.55; -0.34) | 5,898/6,473 = 91.12 (91.72; -0.60) | 166/10,240 = 1.62 (1.65, n=169; -0.03) | 2,369/50k = 4.74% | 440,475/10m = 4.40% |
| 4 | 7,904/10,240 = 77.19 (77.78; -0.59) | 7,390/7,904 = 93.50 (93.50; 0.00) | 375/10,240 = 3.66 (3.67, n=376; -0.01) | 3,875/50k = 7.75% | 757,802/10m = 7.58% |
| 6 | 9,300/10,240 = 90.82 (90.95; -0.13) | 8,946/9,300 = 96.19 (96.18; +0.01) | 420/10,240 = 4.10 (4.10, n=420; 0.00) | 8,108/50k = 16.22% | 1,588,446/10m = 15.88% |

Although every percentage is within tolerance, Table 4 is classified
**missing provenance**, not reproduced.

## Figures

### Figures 1--4, artifact level

- **Figure 1:** layer-4 f/670 feature, geometry, and InterPro files exist, but
  its fixed-null file and exact schematic/editable source do not.
- **Figure 2:** f/894 exists and has local geometry PR-AUC 0.9921. Other panel
  identities and the exact renderer/state are not frozen.
- **Figure 3:** f/6775, f/5508, f/8254, and f/9608 exist. Q15120 occurs in each
  feature's stratified activation-bin sample (not its top-20 list). Their local
  geometry PR-AUCs are 0.7700, 0.8528, 0.6451, and 0.7793. The exact renderer
  is absent.
- **Figure 4:** f/8518 has P37016 in its top sequences and SwissProt geometry
  PR-AUC 0.6206. Its cached transfer record has six NMPFam hits, mean PR-AUC
  0.7365, and contains F011972 at PR-AUC 0.8657. This supports the case-study
  identity and paper's rounded 0.62/~0.73 statements, but the raw NMPFam input
  and exact renderer are absent.

These are identity checks, not figure reconstructions. They are classified
**missing renderer/provenance**. Figure 5/contact-map ablations were not
attempted; no ESMFold or steering experiment was substituted.

### Figure 6

Generated files are `reproduction_outputs/figure6_descriptor_counts.{csv,json,svg,png}`.
The corrected generator scanned 10,129 well-formed layer-4 geometry files and
tested fixed-score geometry q-values for 9,587 features. It retained 3,494
features with q < 0.05 and PR-AUC > 0.3. Each eligible feature contributes at
most once: its single highest-importance descriptor is counted only when that
importance exceeds 0.1. Counts are stacked into the paper's 0.3--0.6 and >0.6
PR-AUC bins and descriptors are assigned to the three displayed families. No
files were malformed. Of the 3,494 eligible features, 3,412 had a winning
descriptor above 0.1.

| Descriptor | Generated | Paper | Count delta |
|---|---:|---:|---:|
| `wide_end_to_end_ratio` | 255 | 271 | -16 |
| `narrow_end_to_end_ratio` | 210 | 216 | -6 |
| `wide_curvature_mean` | 198 | 214 | -16 |
| `end_to_end_ratio` | 164 | 174 | -10 |
| `max_seq_sep_contact_8A` | 150 | 160 | -10 |
| `contact_density_12A` | 151 | 159 | -8 |

Across all 44 descriptors, generated counts sum to 3,412 versus 3,599 in the
paper (-187, or -5.20%). The distribution is considerably closer than this
total difference suggests. For the six descriptors above, generated shares of
their respective 44-descriptor totals differ from the paper shares by -0.056,
+0.153, -0.143, -0.028, -0.049, and +0.008 pp. These derived shares show that
the distribution is close, but the declared 1-pp rule applies to manuscript
percentages, not raw figure counts. Figure 6 therefore remains **not exactly
reproduced in counts**, with a closely matched descriptor distribution.

The source and output now agree with the declared scientific rules rather than
using a value-selected gate: fixed-score q < 0.05, PR-AUC > 0.3, one top
descriptor above 0.1 per feature, and the two paper PR-AUC strata. Since the
release snapshot/checksums are unresolved and no threshold-grid metadata is
available, the remaining count deficit is classified **snapshot
difference/unresolved provenance**. No alternate input or post-hoc gate was
tried. The corrected JSON SHA-256 is
`4785906da3a1bc12aca68ca267af7f1a8acffae1fc2254bbcb7e8168032b4422`.

## Appendix Tables 5--8

- **Table 5:** `proteinlens/train/configs/relu_sweep.yaml` supports the paper's
  parameter ranges, dictionary/batch choices, Bayesian method, objective, and
  Hyperband minimum. Completed-run counts and the final selection rule cannot
  be audited without a pinned W&B export. Partially reproduced; **missing
  provenance**.
- **Table 6:** the three run configs exactly support layer, activation size
  320, dictionary 10,240, expansion 32x, batches, learning rates, L1 values,
  warmup/decay steps, 500,000 configured steps, and epochs. Local final YAMLs
  reproduce CE/loss-recovered values: layer 2 `0.4590413/99.2889%`, layer 4
  `0.4606722/99.3445%`, layer 6 `0.4335957/100%`. Variance explained, L0,
  component losses, and dead-feature counts lack a complete local summary.
  Partially reproduced; **missing provenance**.
- **Table 7:** source constants exactly encode 80 estimators, depth 3,
  learning rate 0.1, subsample 0.8, and
  `max(5, floor(0.02*N))` minimum leaf size. Reproduced from source.
- **Table 8:** source contains the paper's 22 predicates, and current defaults
  use 100 threshold steps for InterPro, position, motif, and PWM. Definition
  reproduced from source; the legacy null artifacts cannot prove they used
  100 steps because their metadata omits it.

## Discrepancy classes

| Class | Findings |
|---|---|
| Wrong identity | None at model/layer/run/dictionary level. Exact snapshot identity remains unresolved. |
| Snapshot incompleteness/difference | Table 1 layer 2; missing Table 2 scores; Figure 6's residual count deficit despite closely reproduced descriptor shares. |
| Missing provenance | Release ID/checksums; null threshold grid; raw Table 4 data/q source; Figures 1--4 renderers; W&B export; complete Table 6 evaluation. |
| Changed estimand | Historical capped Table 3 (now fixed). The corrected Figure 6 generator no longer has an estimand mismatch. |
| Unexplained | One layer-4 Table 3 group; Figure 6's exact raw-count residual remains unresolved at the snapshot level. |

## Commands and execution notes

```text
python scripts/verify_paper_manifest.py --output reproduction_outputs/artifact_identity.json
python scripts/verify_paper_manifest.py --strict --output reproduction_outputs/artifact_identity_strict.json
~/miniconda3/envs/interplm/bin/python scripts/paper_tables.py --output reproduction_outputs/tables_1_2.json
~/miniconda3/envs/interplm/bin/python scripts/build_subdomain_case_study.py --data-dir trained_models/layer_2/firm-sweep-3/analysis --output reproduction_outputs/layer2_table3.json
~/miniconda3/envs/interplm/bin/python scripts/build_subdomain_case_study.py --data-dir trained_models/layer_4/frosty-sweep-15/analysis --output reproduction_outputs/layer4_table3.json
~/miniconda3/envs/interplm/bin/python scripts/build_subdomain_case_study.py --data-dir trained_models/layer_6/major-sweep-15/analysis --output reproduction_outputs/layer6_table3.json
~/miniconda3/envs/interplm/bin/python scripts/build_nmpfam_transfer_summary.py --analysis-dir trained_models/layer_2/firm-sweep-3/analysis --output reproduction_outputs/layer2_table4.json
~/miniconda3/envs/interplm/bin/python scripts/figure6_descriptor_counts.py --analysis-dir trained_models/layer_4/frosty-sweep-15/analysis --output-dir reproduction_outputs
sha256sum reproduction_outputs/figure6_descriptor_counts.json reproduction_outputs/figure6_descriptor_counts.csv
```

The strict identity command intentionally exited 1 after writing its report.
The Table 4 command exited 1 for the missing raw directory; after confirming
the same absence globally, layers 4 and 6 were not redundantly invoked. The
base Python resolves a broken namespace-only NumPy, so NumPy-dependent
generators used the existing `interplm` interpreter (Python 3.11, NumPy
1.26.4). This changes no data or scientific rule.

No source code was edited during this independent attempt. Generated outputs
are under `reproduction_outputs/`; this report is the only added document.
