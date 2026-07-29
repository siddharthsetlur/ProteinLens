# ESM2-35M layer 6 — annotation pipeline (paper Tables 1 & 2)

Runs the full residue-level annotation pipeline on the SAE from the 35M sweep
(`/data/models/sweeps/layer_6/cgvpk5vp`, W&B run `colorful-sweep-1`) and emits
Tables 1 and 2 for the 35M generalization row.

Everything lives on **`proteinlens-esm35m-pvc`** (the 35M claim from
`k8s/pvc-esm35m.yaml`), mounted at `/data`. Pipeline output dir:
`/data/feature_data_esm35m_l6`.

Prerequisite: `k8s/job-esm35m-sae-sweep.yaml` finished and `cgvpk5vp` has
`ae.pt` + `config.yaml` under `/data/models/sweeps/layer_6/`.

## Dependency graph

```
1  prep       cluster                                    CPU   ~1-2 h
                     |
2  gpu        survey -> selection -> collection ->       A100  days
              assembly -> interpro_selection
                     |
        +------------+------------+
        |            |            |
3a interpro   3b geometry   3c seq            <-- ALL THREE AT ONCE
   fetch         6a features    7  motif
   enrichment    6c GBM+PR-AUC  7b MEME PWM
                                8  position
        |            |            |
        +------------+------------+
                     |
4  permnull   100 within-protein shuffles, 8 shards      CPU
                     |
5  tables     BH FDR -> Table 1 + Table 2                CPU   minutes
```

Legs 3a/3b/3c are safe to run concurrently. They write to disjoint
subdirectories, and the geometry leg does not consume anything the InterPro
leg produces: stage 6c unions `residue_activations/` with
`interpro_residue_activations/`, but `interpro_selection` only writes `.npz`
for accessions that have *no* stage-3 `.npz`, and AlphaFold PDBs are fetched
only for stage-3 accessions — so nothing in the InterPro-only set has a
geometry profile, and the union adds no proteins either way.

## 0. Rebuild and push the image

Required — `checkpoint.py`, `compute_permutation_null.py` and the new
`scripts/paper_tables.py` all changed.

```bash
docker build -f k8s/Dockerfile.pipeline -t <REGISTRY>/proteinlens-pipeline:latest .
docker push  <REGISTRY>/proteinlens-pipeline:latest
```

## 1. Prep — MMseqs2 clustering

Stage 0a (download) is skipped: this leg copies the SwissProt FASTA already on
the PVC at `/data/uniprot/uniprot_sprot.fasta`, filtering to <= 1024 residues
(the survey pass does not skip over-length proteins; the ESM tokenizer would
silently truncate them and desynchronise residue indices).

```bash
kubectl create -f k8s/job-esm35m-pipeline-prep.yaml
kubectl logs -f -l app=proteinlens-esm35m-prep
```

## 2. GPU leg — survey through interpro_selection

The long pole. ESM2-35M is roughly 4x the FLOPs of the 8M model, and this
streams all of SwissProt; budget days, not hours. It is checkpointed every
1000 proteins, so an evicted pod resumes.

```bash
kubectl create -f k8s/job-esm35m-pipeline-gpu.yaml
kubectl logs -f -l app=proteinlens-esm35m-gpu
```

## 3. The three annotation legs — launch together

```bash
kubectl create -f k8s/job-esm35m-pipeline-interpro.yaml
kubectl create -f k8s/job-esm35m-pipeline-geometry.yaml
kubectl create -f k8s/job-esm35m-pipeline-seq.yaml

kubectl get pods -l 'app in (proteinlens-esm35m-interpro,proteinlens-esm35m-geometry,proteinlens-esm35m-seq)' -w
```

Which Table 1 column each one feeds:

| leg | stages | Table 1 columns |
|---|---|---|
| interpro | 5b, 5c | `% InterPro Prot.`, `% InterPro Res.` |
| geometry | 6a, 6c | `% Geometric` + all of Table 2 |
| seq | 7, 7b, 8 | `% Seq Motif` (MEME), `% Seq Pos` |

CATH is **not** run — it is not a column of the published Table 1.

## 4. Permutation null — wait for all three legs

This is a barrier. A leg that has not finished produces no null for its
method, and that Table 1 column comes out empty rather than wrong.

8 indexed shards partitioned by `feature_id % 8`. Per-feature RNG seeds are
`42 + feature_id`, so the sharded result is numerically identical to a
single-pod run.

```bash
kubectl create -f k8s/job-esm35m-pipeline-permnull.yaml
kubectl get pods -l app=proteinlens-esm35m-permnull -w
```

Check for stragglers before moving on — the count should equal the number of
non-dead features:

```bash
kubectl exec -it <any-access-pod> -- \
  bash -c 'ls /data/feature_data_esm35m_l6/permutation_null/*.json | wc -l'
```

## 5. Tables

```bash
kubectl create -f k8s/job-esm35m-pipeline-tables.yaml
kubectl logs -f -l app=proteinlens-esm35m-tables
```

Writes `/data/feature_data_esm35m_l6/paper_tables.{txt,json}` and logs both
tables to W&B (`proteinlens-pipeline`, run `paper-tables`) as `wandb.Table`s
plus per-cell summary scalars.

To regenerate locally after pulling the analysis dir back:

```bash
python scripts/paper_tables.py --analysis "ESM2-35M Layer 6=<dir>"
```

## 6. Pull the analysis dir back

With a data-access pod up (`kubectl pvcsync proteinlens-esm35m-pvc up`, note
the PORTNUMBER):

```bash
mkdir -p analysis/esm35m_l6
rsync -rvP \
  -e 'ssh -J <USER>@<GATEWAY_HOST>,<USER>@<INTERNAL_HOST> -p PORTNUMBER' \
  root@localhost:/data/feature_data_esm35m_l6/ \
  analysis/esm35m_l6/
```

## Watch out for

- **Disk.** The PVC is 350Gi and already holds the training embeddings. At
  15,360 features the per-residue `.npz` files are ~1.5x the 8M run's, so
  `residue_activations/` + `interpro_residue_activations/` + `pdb_cache/`
  will run to tens of GB. Check free space before launching leg 2.
- **`--motif-pwm`** is on the seq leg and is required: Table 1's `% Seq Motif`
  is the MEME PWM PR-AUC, not the stage-7 k-mer F1. Without the flag the
  column is empty.
- **`--include-pwm`** is on the permnull leg for the same reason — it is what
  gives the MEME score a null.
