# ProteinLens → Hugging Face: upload handoff

**What this is.** Instructions to publish the ProteinLens trained SAEs and
analysis artifacts to Hugging Face. Everything runs from a machine that has
(a) the ProteinLens repo checked out and (b) the datastore disk mounted.

**Why you and not me:** my VPN can't reach the datastore right now. Nothing
here is destructive — it reads from the datastore and writes to *your* HF
account. Total upload is about 38 GB across three repos.

> **If you are an AI agent executing this:** follow the steps in order. Do not
> improvise paths — run the Step 0 checks and stop if they fail. Several
> commands print `MISSING`/`!!` warnings that are *data*, not errors: collect
> them and report them back rather than working around them. The one thing you
> must not do is upload raw `config.yaml` files from the datastore (they embed
> absolute machine paths); Step 2's script produces sanitized copies for you.

---

## 0. Prerequisites and sanity checks

You need: the repo, the datastore mounted, Python 3.10+, and an HF account with
a **write** token from <https://huggingface.co/settings/tokens>.

Set these two variables. `DATASTORE` must be the directory that *contains*
`trained_models/`:

```bash
export PL=/path/to/ProteinLens          # repo checkout
export DATASTORE=/path/to/datastore     # contains trained_models/
```

**Check the layout.** This must list files such as `permutation_null`,
`geometry_classifiers`, `interpro_enrichment`:

```bash
ls "$DATASTORE/trained_models/layer_4/frosty-sweep-15/analysis" | head -20
```

If it's empty, the datastore is nested or flattened differently. Find the right
level and re-set `DATASTORE`:

```bash
find "$DATASTORE" -maxdepth 4 -type d -name 'frosty-sweep-15' 2>/dev/null
```

**Check free space.** The build step hard-links (≈0 extra bytes) when the
staging dir is on the *same filesystem* as the datastore, which is why Step 2
stages into `$DATASTORE/_hf_release`. If they end up on different filesystems it
silently falls back to copying and needs ~40 GB:

```bash
df -h "$DATASTORE"
```

---

## 1. Environment

```bash
pip install --upgrade "huggingface_hub[cli,hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1      # faster large uploads
hf auth login                            # paste your write token
```

Use the repo's Python if there is a conda env set up (`geopedia`); otherwise any
Python 3.10+ with `numpy` and `pyyaml` works for the build script.

---

## 2. Build the three bundles

One script does all of it — selects the right files, hard-links them into a
staging tree, **sanitizes machine-specific paths out of the run configs**, writes
sidecars for the raw memmap arrays, and emits checksums.

```bash
cd "$PL"

# models + paper bundles (~4.5 GB combined, fast)
python scripts/build_release.py \
    --source "$DATASTORE" \
    --out "$DATASTORE/_hf_release" \
    --bundles models,paper \
    --mode link \
    --checksums

# viz bundle (~31 GB; skip --checksums, it's slow at this size)
python scripts/build_release.py \
    --source "$DATASTORE" \
    --out "$DATASTORE/_hf_release" \
    --bundles viz \
    --mode link
```

### What the three bundles are

| Bundle | Size | What it enables |
|---|---|---|
| `models` | ~80 MB | Load the trained SAEs and run inference |
| `paper` | ~4.5 GB | Regenerate paper Tables 1–4, 7, 8 and Figure 6 |
| `viz` | ~31 GB | Launch the GeoPedia visualizer |

Bundle trees mirror the layout the code already expects, so a user downloads one
and drops it at the repo root with no path rewriting.

### ⚠️ Record any `MISSING` warnings

The script logs `WARNING MISSING: <path>` for anything it can't find. **Please
copy these lines and send them back.** One is expected and important:

- `analysis/nmpfam/nmpfam_enrichment` — the raw input for **Table 4**. It is
  absent from the local machine; the datastore copy is the only one. If it's
  missing there too, Table 4 can't be independently reproduced and we need to
  know before publishing.

Confirm what landed:

```bash
cat "$DATASTORE/_hf_release/build_report.json"
find "$DATASTORE/_hf_release/paper" -type f | wc -l    # expect ~180,000
```

---

## 3. Upload

Repos are created automatically on first upload (private). If you'd rather
create them explicitly first, that's fine — `hf repo create ... --exist-ok`.

Replace `<YOUR_HF>` with your HF username or org:

```bash
export HF_ORG=<YOUR_HF>

hf upload-large-folder "$HF_ORG/proteinlens-sae-esm2-8m" \
    "$DATASTORE/_hf_release/models" --repo-type=model --private --num-workers=4

hf upload-large-folder "$HF_ORG/proteinlens-paper-artifacts" \
    "$DATASTORE/_hf_release/paper" --repo-type=dataset --private --num-workers=8

hf upload-large-folder "$HF_ORG/proteinlens-geopedia-analysis" \
    "$DATASTORE/_hf_release/viz" --repo-type=dataset --private --num-workers=8
```

**Keep them private for now.** We'll flip them public when the paper is ready.

### Expected behaviour, so you don't think it's broken

- **Long silence at the start of each upload.** It hashes every file before
  sending anything. The paper bundle is ~180k files, so this takes a while with
  no output. It is not hung.
- **`Repo created: ...` on a repo that already exists** is a harmless no-op —
  the uploader always calls `create_repo(exist_ok=True)`.
- **Interrupted?** Re-run the identical command. It resumes from state kept in
  `.cache/.huggingface/` inside the bundle dir.

---

## 4. Verify (please don't skip)

Round-trip one bundle to prove the release is self-sufficient:

```bash
mkdir -p /tmp/pl-roundtrip
hf download "$HF_ORG/proteinlens-paper-artifacts" \
    --repo-type dataset --local-dir /tmp/pl-roundtrip

cd /tmp/pl-roundtrip && sha256sum -c SHA256SUMS | grep -v ': OK$' ; cd -
```

No output from the `grep` means every checksum matched.

---

## 5. Two provenance checks

Both are quick, and both are far cheaper to answer now than after the data is
public and cited.

**(a) Do the permutation nulls record their threshold grid?**

```bash
python - <<'EOF'
import json, glob, os
base = os.environ["DATASTORE"]
for run in ["layer_2/firm-sweep-3", "layer_4/frosty-sweep-15", "layer_6/major-sweep-15"]:
    files = sorted(glob.glob(f"{base}/trained_models/{run}/analysis/permutation_null/*.json"))
    if not files:
        print(f"{run}: NO permutation_null files"); continue
    keys = json.load(open(files[0])).keys()
    print(f"{run}: {len(files)} files, threshold_steps={'YES' if 'threshold_steps' in keys else 'NO'}")
EOF
```

The local copies say `NO`. If the datastore copies say `YES`, they're the newer
provenance-recording generation and we should publish those — it would let the
release prove the paper's 100-step threshold grid, and may explain an
unresolved discrepancy in the layer-2 Table 1 numbers.

**(b) Does the NMPFam raw data exist for all three layers?**

```bash
for run in layer_2/firm-sweep-3 layer_4/frosty-sweep-15 layer_6/major-sweep-15; do
  d="$DATASTORE/trained_models/$run/analysis/nmpfam/nmpfam_enrichment"
  printf '%-40s ' "$run"
  [ -d "$d" ] && echo "$(ls "$d" | wc -l) files" || echo "MISSING"
done
```

---

## 6. Send back

1. The three repo URLs.
2. Any `MISSING` warnings from Step 2.
3. The output of both Step 5 checks.
4. Whether the Step 4 checksum verification passed.

I'll then pin the release into `paper_manifest.yaml` (snapshot ID + per-layer
checksums) and run `scripts/verify_paper_manifest.py --strict`, which is the
check that converts "these are our artifacts" into something a reviewer can
verify. Nothing else is needed from you.

**On ownership:** these will live under your HF account. That's fine for review,
but if we later move them to a lab org the URLs change and any pinned manifest
has to be re-pinned — so if a shared org already exists, it's cheaper to upload
there now. Your call.

---

## Notes on what is deliberately excluded

Not everything on the datastore is published, for good reasons rather than
oversight:

| Excluded | Why |
|---|---|
| `pdb_cache/` (~400 MB/layer) | AlphaFold structures; the viz falls back to the AlphaFold REST API |
| `interpro_cache/` (~210 MB/layer) | Raw EBI responses; re-fetchable from a pinned InterPro release |
| `swissprot_all.fasta` (~175 MB/layer) | UniProt; better shipped as a pinned-release fetch than a redistributed copy |
| `residue_activations/` (~400 MB/layer) | Pipeline intermediate; read by neither the tables nor the viz |
| `geom_refit/`, `geometry_null_refit/` | Layer-4 refit robustness check, not the paper's primary null |
| `protein_feature_maxes.npy` (2 GB/layer) | Opt in with `--include-protein-maxes` if we want it |
| Raw `config.yaml` | Contains absolute `/data/...` paths; the build script publishes sanitized copies |

Contact-map / Figure 5 artifacts are excluded from this release entirely — that
code and its inputs sit with an external collaborator.

---

## Troubleshooting

**Upload appears to hang immediately after "repo created".** Almost certainly
you pointed `upload-large-folder` at `$DATASTORE` rather than at a bundle dir
under `$DATASTORE/_hf_release/`. It globs the *entire* tree before applying any
filter, so aimed at the whole datastore it enumerates millions of files before
sending anything. Always upload the staged bundle directory.

**`build_release.py` says a run directory isn't found.** Your `DATASTORE` is at
the wrong level — re-run the Step 0 `find`.

**Build filled the disk.** Hard-linking fell back to copying because `--out`
landed on a different filesystem from `--source`. Keep `--out` under
`$DATASTORE` as written.

**Cleanup when everything is confirmed uploaded:**

```bash
rm -rf "$DATASTORE/_hf_release"    # removes only hard links, not your data
```
