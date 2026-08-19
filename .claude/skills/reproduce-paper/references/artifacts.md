# Release artifact map

Three repos. Sizes below are **verified against the live repos**, not estimates.

| Repo | Type | Holds |
|---|---|---|
| `Sidd2010/proteinlens-sae-esm2-8m` | model | SAE weights, sanitized `config.yaml`, `final_evaluation.yaml` |
| `Sidd2010/proteinlens-paper-artifacts` | dataset | null tests, enrichment, NMPFam, `geometry_primary_analysis.json` |
| `Sidd2010/proteinlens-geopedia-analysis` | dataset | per-feature payloads, `geometry_enrichment`, `cath_enrichment`, survey files |

All three are **private** until publication — `hf auth login` first.

Run identities: layer 2 = `firm-sweep-3`, layer 4 = `frosty-sweep-15`,
layer 6 = `major-sweep-15`. Substitute for `<RUN>` below.

## The trap

The repo named *paper-artifacts* is **not sufficient for the paper**. Table 3 and
Figure 6 read `geometry_enrichment/`, `cath_enrichment/`, and
`survey_coverage.json`, which live in the *geopedia-analysis* repo because the
visualizer needs them too. Check the matrix before downloading.

## What each target reads

`R` = required, `o` = optional, blank = unused.

| File | Repo | T1-2 | T3 | T4 | Fig 6 |
|---|---|:--:|:--:|:--:|:--:|
| `permutation_null/` | paper | R | R | R | R |
| `geometry_primary_analysis.json` | paper | R | R | R | |
| `dataset_stats.json` | paper | R | | R | |
| `interpro_enrichment/` | paper | | R | | |
| `motif_pwm_enrichment/` | paper | | R | | |
| `nmpfam/nmpfam_enrichment/` | paper | | | R | |
| `geometry_enrichment/` | **viz** | | R | | R |
| `cath_enrichment/` | **viz** | | R | | |
| `survey_coverage.json` | **viz** | | R¹ | | |
| `feature_max_activations.npy` | **viz** | o² | | | |

¹ Without it the sparsity filter silently goes inactive and the Table 3 count is
not comparable to the paper. The generator warns; treat that warning as fatal.
² Only used to read the dictionary size; falls back to `dataset_stats.json`.

`geometry_classifiers/` (264-287 MB per layer) and `interpro_selection.json`
(78-96 MB) are in the paper repo for provenance and refits. **None of the four
table generators read them** — skip unless you are re-deriving
`geometry_primary_analysis.json` with `scripts/compute_geometry_primary.py`.

## Footprint per target, per layer

Compressed download → extracted on disk.

| Target | Layer 2 | Layer 4 | Layer 6 |
|---|---|---|---|
| Tables 1-2 | 30 MB → 160 MB | 32 MB → 165 MB | 33 MB → 170 MB |
| Table 3 | 0.96 GB → 5.5 GB | 1.16 GB → 6.2 GB | 1.26 GB → 6.5 GB |
| Table 4 | 10.3 GB → ~54 GB | *(incomplete — see below)* | 5.9 GB → ~31 GB |
| Figure 6 | — | 1.13 GB → 5.7 GB | — |

Figure 6 is a layer-4 figure. The NMPFam archives are by far the largest objects
in the release and expand 5x; check `df -h` and do one layer at a time.

## Known release defect — Table 4, layer 4

`trained_models/layer_4/frosty-sweep-15/analysis/nmpfam/nmpfam_enrichment/`
contains **284 per-feature files covering ids 8933-10239**, against the ~7,904
the layer's own cached `nmpfam_transfer_summary.json` was built from. The
generator runs happily on it and returns badly wrong numbers — 2.73% of features
with NMPFam activation against the paper's 77.78%.

**Do not report Table 4 for layer 4 from this release.** Layers 2 and 6 are
intact; layer 6 holds 9,313 files, exactly the paper's 90.95% of 10,240.

The cause is upstream of the release: the datastore copy is itself partial, so
re-running the transfer will not fix it. The complete run lived on the cluster PVC.

## Expected file counts after extraction

Verified against the live release. Use these as the post-extraction sanity check.

| Directory | Layer 2 | Layer 4 | Layer 6 |
|---|---:|---:|---:|
| `permutation_null/` | 9,309 | 9,587 | 9,743 |
| `nmpfam/nmpfam_enrichment/` | *unverified* | **284 (broken)** | 9,313 |

A count below these means a truncated download — re-run `hf download`, it resumes.

## Commands

Run from the repo root so the extracted tree lands where the generators look.

`--include` patterns are fnmatch, and `*` crosses `/`. `trained_models/*/*/...`
therefore also matches the four-level ESM2-35M path — harmless, it just pulls that
run's files too. Name the run explicitly to avoid it.

**Tables 1 and 2** — all three layers plus the 35M run, ~140 MB:

```bash
hf download Sidd2010/proteinlens-paper-artifacts --repo-type dataset --local-dir . \
  --include "trained_models/*/*/analysis/permutation_null.tar.zst" \
            "trained_models/*/*/analysis/geometry_primary_analysis.json" \
            "trained_models/*/*/analysis/dataset_stats.json"
```

**Table 3** — one layer:

```bash
hf download Sidd2010/proteinlens-paper-artifacts --repo-type dataset --local-dir . \
  --include "trained_models/layer_4/<RUN>/analysis/permutation_null.tar.zst" \
            "trained_models/layer_4/<RUN>/analysis/geometry_primary_analysis.json" \
            "trained_models/layer_4/<RUN>/analysis/interpro_enrichment.tar.zst" \
            "trained_models/layer_4/<RUN>/analysis/motif_pwm_enrichment.tar.zst"

hf download Sidd2010/proteinlens-geopedia-analysis --repo-type dataset --local-dir . \
  --include "trained_models/layer_4/<RUN>/analysis/geometry_enrichment.tar.zst" \
            "trained_models/layer_4/<RUN>/analysis/cath_enrichment.tar.zst" \
            "trained_models/layer_4/<RUN>/analysis/survey_coverage.json"
```

**Table 4** — one layer; check `df -h` first:

```bash
hf download Sidd2010/proteinlens-paper-artifacts --repo-type dataset --local-dir . \
  --include "trained_models/layer_4/<RUN>/analysis/nmpfam/nmpfam_enrichment.tar.zst" \
            "trained_models/layer_4/<RUN>/analysis/permutation_null.tar.zst" \
            "trained_models/layer_4/<RUN>/analysis/geometry_primary_analysis.json" \
            "trained_models/layer_4/<RUN>/analysis/dataset_stats.json"
```

**Figure 6** — layer 4:

```bash
hf download Sidd2010/proteinlens-paper-artifacts --repo-type dataset --local-dir . \
  --include "trained_models/layer_4/<RUN>/analysis/permutation_null.tar.zst"
hf download Sidd2010/proteinlens-geopedia-analysis --repo-type dataset --local-dir . \
  --include "trained_models/layer_4/<RUN>/analysis/geometry_enrichment.tar.zst"
```

**SAE weights** (only needed to run inference, not for any table):

```bash
hf download Sidd2010/proteinlens-sae-esm2-8m --local-dir .
```

## Extraction

Every directory ships as one `.tar.zst`, because a HF repo caps at 10,000 files per
directory and these hold up to 19,160. Extract in place:

```bash
find trained_models -name '*.tar.zst' -print -execdir tar --zstd -xf {} \; -delete
```

`-delete` reclaims the archive after extraction. Drop it to keep them, and budget
the extra space. Each repo also ships `EXTRACT.sh` containing this command, but a
targeted `--include` download does not fetch it.

Needs `zstd`: `conda install -c conda-forge zstd` or `apt-get install zstd`.

## Resuming and verifying

`hf download` resumes — re-run the identical command after an interruption.
`hf` caches under `$HF_HOME`, so `--local-dir` copies cost disk twice unless
`HF_HUB_ENABLE_HF_TRANSFER=1` and you clear the cache afterwards.

After extraction, sanity-check file counts against
`docs/reproduction_attempt_report.md`: `permutation_null/` should hold 9,309 /
9,587 / 9,743 JSON files for layers 2 / 4 / 6.

## ESM2-35M layer 6 (Mode C)

Path `trained_models/esm2_35m/layer_6/cgvpk5vp/analysis/`, same paper repo.
Has `permutation_null` (30.6 MB), `geometry_primary_analysis.json` (13.8 MB),
`interpro_enrichment`, `motif_pwm_enrichment`, `position_enrichment`, and — in the
viz repo — `geometry_enrichment` (1.66 GB). **No NMPFam and no `cath_enrichment`**,
so Tables 1-2 only.
