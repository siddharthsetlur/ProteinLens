---
name: reproduce-paper
description: Reproduce the results of the ProteinLens paper (Interpreting Latent Protein Representations) from the released Hugging Face artifacts — Tables 1-4 and Figure 6 — or, on explicit request, re-run the full pipeline from scratch. Use when someone wants to verify, reproduce, check, or extend the paper's reported numbers.
---

# Reproduce the ProteinLens paper

Takes a fresh clone from nothing to a reproduction report. The artifacts the
paper reports on are released on Hugging Face.

The manuscript itself is **not tracked in the repo**. `paper_manifest.yaml`
declares its expected path as `docs/28622_Interpreting_Latent_Prot.pdf`; if it is
not there, ask the user for it rather than guessing at the published values —
`docs/reproduction_attempt_report.md` tabulates them, but it is a secondary source.

**Default to Mode A.** Mode B (re-run everything from raw sequences) costs GPU-weeks
and is only entered when the user explicitly asks for it.

## Two rules that override everything else in this skill

1. **Never move a number toward the paper.** Do not rescale, re-gate, cap,
   re-select inputs, swap a q-value source, or substitute a nearby artifact for a
   missing one. A mismatch is a finding to report, not a bug to fix. This is the
   policy in `docs/paper_reproduction.md` and it is the whole point of the exercise.
2. **Some mismatches are the expected result.** Reproducing them correctly is
   success. See *Known discrepancies* below before you start debugging anything.

## Mode A — reproduce from the released artifacts

CPU only. No GPU, no cluster. Budget 30-90 min including downloads.

### Step 0 — preflight

```bash
conda activate geopedia          # the repo's env; the base python has a broken numpy
python -c "import numpy, yaml, matplotlib"
hf auth whoami                   # the repos are PRIVATE until the paper is out
df -h .                          # check against the target's footprint below
```

If `hf auth whoami` fails, the user needs a token from
<https://huggingface.co/settings/tokens> and `hf auth login`. If the repos are
public by the time you read this, no login is needed.

### Step 1 — ask what to reproduce, then download only that

Targets: **Tables 1-2**, **Table 3**, **Table 4**, **Figure 6**, or all of them;
and which layers (2, 4, 6). Do not download a whole repo by reflex — the footprint
ranges from 32 MB to 17 GB depending on the answer.

Read `references/artifacts.md` for the download map. It gives the exact
`hf download --include` command per target, real compressed and extracted sizes,
and — importantly — **which targets need files from the `geopedia-analysis` repo
rather than the `paper-artifacts` repo**. Table 3 and Figure 6 both do.

Download into the repo root so paths line up: every generator defaults to
`trained_models/layer_N/<run>/analysis/`, which is exactly the released layout.

### Step 2 — extract

The directories ship as one `.tar.zst` per directory (a HF repo caps at 10,000
files per directory; these hold up to 19,160). Each repo carries its own helper:

```bash
sh EXTRACT.sh        # find . -name '*.tar.zst' -execdir tar --zstd -xf {} \; -delete
```

Or extract just what you fetched. Extracted trees are 4-6x the compressed size.

### Step 3 — identity check

```bash
python scripts/verify_paper_manifest.py \
    --output reproduction_outputs/artifact_identity.json
```

Must pass. It confirms ESM2-8M, layers 2/4/6, 10,240-feature dictionaries, and the
runs named in `paper_manifest.yaml`, and emits SHA-256 fingerprints.

`--strict` additionally requires `artifact_release.snapshot_id` and per-layer
`sha256` in `paper_manifest.yaml`. If those are unpinned, strict fails **and that
is a provenance gap, not a numerical one** — say so plainly and carry on. Do not
fabricate a snapshot id.

### Step 4 — regenerate

Run only what the chosen target needs. Full command list in
`docs/paper_reproduction.md`; in brief, per layer:

```bash
# Tables 1 and 2 (all layers in one call)
python scripts/paper_tables.py --output reproduction_outputs/tables_1_2.json

# Table 3
python scripts/build_subdomain_case_study.py \
    --data-dir trained_models/layer_4/frosty-sweep-15/analysis \
    --output reproduction_outputs/layer4_table3.json

# Table 4
python scripts/build_nmpfam_transfer_summary.py \
    --analysis-dir trained_models/layer_4/frosty-sweep-15/analysis \
    --output reproduction_outputs/layer4_table4.json

# Figure 6 (layer 4; writes CSV, JSON, SVG, PNG)
python scripts/figure6_descriptor_counts.py --output-dir reproduction_outputs
```

Watch for `WARNING: survey_coverage.json not found` from the Table 3 generator. It
means the sparsity filter is **inactive** and the resulting count is not comparable
to the paper. Fetch that file and re-run rather than reporting the number.

### Step 5 — compare

Read the paper for the published values, and
`docs/reproduction_attempt_report.md` for the previous independent attempt —
it tabulates every Table 1-3 row with its computed value, paper value, and delta,
so it is the fastest cross-check that a run behaved.

Tolerance: differences ≤ `comparison_tolerance_percentage_points` in
`paper_manifest.yaml` (1.0 pp) count as practically reproduced. The tolerance is a
**reporting rule only** — never a licence to adjust an input.

Write `reproduction_outputs/REPRODUCTION_REPORT.md` with one row per reported
value, each classified:

| Class | Meaning |
|---|---|
| reproduced | exact, or equal after rounding |
| within tolerance | ≤ 1.0 pp |
| snapshot difference | > 1.0 pp, identity checks pass, method unchanged |
| not reproducible | a required input is absent — name it |

Always keep the raw count, the denominator, the unrounded delta, and the artifact
fingerprint from Step 3. A percentage with no denominator is not a result.

### Step 6 — report the exclusions rather than working around them

- **Figure 5** (contact-prediction ablation): the code landed in #6 and the left and
  middle panels are runnable — but **not from the release**. Its SAE
  (`trained_models/layer_3/fiery-sweep`, 5,120 features, `layer_idx` 3) and its
  analysis dir (`feature_data_cluster/`) are both unpublished, and are a different
  model from the paper's layer-4 run. See `references/figure5.md` before touching
  it, and never substitute the layer-4 run.
- **Figures 1-4**: underlying data exists to varying degrees, no deterministic
  renderer. Feature and protein identities are frozen in `paper_manifest.yaml`.
- **Tables 5-6**: need a pinned W&B export that is not identified.

## Known discrepancies — expected, do not debug

From `docs/reproduction_attempt_report.md` (2026-08-09), same artifacts:

- **Layer 2, Table 1** runs +1.15 to +2.30 pp high on four of six measures.
  Layers 4 and 6 match. Classified as a snapshot difference; the null files record
  no `threshold_steps`, so they cannot prove whether the paper's 100-step or the
  historical 50-step threshold grid produced the published snapshot.
- **Layer 4, Table 3** gives 447/578 against the paper's 448/578 (-0.17 pp).
- **Figure 6** counts run modestly low (e.g. 255 vs 271) with the distribution
  shape intact.
- Table 3's 100-group cap applies only to the GeoPedia display payload, never to
  the reported statistic. If a run yields ~100 groups, the cap leaked into the
  computation — that one *is* a bug.

- **Table 4 layer 4** reproduces in full, all five columns. Columns 4-5 needed a
  generator fix (it unioned strong hits over all 7,904 features with hits instead
  of the 376 gated ones, giving 38,846 families against the paper's 3,875);
  `tests/test_analysis/test_nmpfam_transfer_summary.py` pins it. If you see 38,846,
  you are on pre-fix code.

Layer-4 NMPFam should extract to 7,965 files. If column 1 comes out at 2.73% instead of ~77%, you have a stale partial download cached — check the file counts in `references/artifacts.md`.

## Mode B — re-run the pipeline from scratch

Only on explicit request. Read `references/from_scratch.md`. State the cost
(GPU cluster, days to weeks, tens of GB) and get confirmation before starting.
Outputs will not be bit-identical — clustering, GBM fits, and permutation draws are
stochastic — so the comparison target is the tables, never the files.

## Mode C — ESM2-35M layer 6 generalization check

Optional, and **not part of the paper's claims** — say so in any report. A second
SAE was trained on ESM2-35M layer 6 and its artifacts are in the same paper repo
under `trained_models/esm2_35m/layer_6/cgvpk5vp/analysis/`. It supports Tables 1-2
only (no NMPFam, no `cath_enrichment`):

```bash
python scripts/paper_tables.py \
    --analysis "ESM2-35M L6=trained_models/esm2_35m/layer_6/cgvpk5vp/analysis" \
    --output reproduction_outputs/tables_1_2_esm35m.json
```

Compare against the committed `analysis/esm35m_l6/paper_tables.json`.

## Mode D — Figure 5, contact-prediction ablation

Read `references/figure5.md` first — it carries the provenance problem, the
environment requirement, and the exact commands.

Three things to know before starting:

1. Figure 5's inputs are **not in the Hugging Face release**, so this mode only
   works on a machine that already holds `feature_data_cluster/` and
   `trained_models/layer_3/fiery-sweep/`. Check for both and stop if absent.
2. `geopedia` cannot run it (broken torch install). Use an environment with
   `torch >= 2.2`, `transformers`, `h5py`, `einops`, `biopython`. CPU is fine.
3. The right panel does not reproduce: 5 features / 20 cases against the paper's
   92 / 495. The 92 came from a curated `--feature-ids-file` that was never
   published. Report it as such; do not tune selection gates toward 92.

Measured on 2026-08-19: left panel r = +0.9982 across the 11 strengths; middle
panel confirms the [25,48] band at 2.4x the next strongest; right panel fails as
above. Details and the full evidence chain in `references/figure5.md`.
