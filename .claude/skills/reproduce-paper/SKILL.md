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
conda activate interplm          # the repo's env; the base python has a broken numpy
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

- **Figure 5** (contact-map ablation): code is with an external collaborator. Do not
  substitute an ESMFold steering experiment.
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

And one release defect, which is **not** a tolerable discrepancy:

- **Table 4 layer 4 cannot be reported from this release.** Its NMPFam input holds
  284 of ~7,904 per-feature files. The generator does not error — it returns 2.73%
  where the paper reports 77.78%. Use layers 2 and 6, which are intact. Verify with
  the file counts in `references/artifacts.md` *before* running the generator.

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
