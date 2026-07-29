# ProteinLens ESM2-35M generalization — EIDF cluster instructions

Train a constrained ReLU-SAE sweep on **ESM2-35M, layer 6** (hidden dim 480 →
15,360-feature dictionary). EIDF specifics (user `<USER>`, queue
`<QUEUE>`, gateway jump chain, CephFS PVC, A100-80GB) are baked into
the YAMLs.

Flow: **build image → create PVC → stage SwissProt → extract embeddings → create
sweep → run sweep agent → pull checkpoints back**.

Substitute your own values for `<REGISTRY>`, `<USER>`, `<QUEUE>`,
`<GATEWAY_HOST>` and `<INTERNAL_HOST>` throughout, as in `k8s/INSTRUCTIONS.md`.
`PORTNUMBER` is the port printed by `kubectl pvcsync ... up` and changes every
session.

## 1. Build & push the image

The image must be rebuilt **after** the `relu_sweep.py` env change (it now reads
`EMBED_TAG` / `EMBEDDING_DIM` / `ESM_MODEL_NAME`). From the repo root:

```bash
docker build -f k8s/Dockerfile.pipeline -t <REGISTRY>/proteinlens-pipeline:latest .
docker push  <REGISTRY>/proteinlens-pipeline:latest
```

## 2. Create the fresh PVC

The old `proteinlens-pipeline-pvc` was reclaimed; this is a new dedicated claim.

```bash
kubectl get pvc proteinlens-esm35m-pvc >/dev/null 2>&1 || kubectl create -f k8s/pvc-esm35m.yaml
```

## 3. Stage SwissProt onto the PVC

Bring up the data-access pod (`kubectl pvcsync proteinlens-esm35m-pvc up`, note
the **PORTNUMBER**), then upload a raw SwissProt FASTA to
`/data/uniprot/uniprot_sprot.fasta`. Reuse the copy you already have locally:

```bash
kubectl pvcsync proteinlens-esm35m-pvc up      # note PORTNUMBER

# The PVC starts empty, so /data/uniprot doesn't exist -- create it on the remote
# first (rsync won't mkdir a parent for a file dest). One-shot via --rsync-path:
rsync -rvP --rsync-path="mkdir -p /data/uniprot && rsync" -e 'ssh -J <USER>@<GATEWAY_HOST>,<USER>@<INTERNAL_HOST> -p PORTNUMBER' trained_models/layer_4/frosty-sweep-15/analysis/swissprot_all.fasta root@localhost:/data/uniprot/uniprot_sprot.fasta
```

(Or download fresh — reliable, resumable:
`curl --retry-all-errors --continue-at - -o uniprot_sprot.fasta.gz https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz`
then gunzip and rsync up.)

## 4. Extract ESM2-35M layer-6 activations

```bash
kubectl create -f k8s/job-esm35m-extract.yaml
kubectl get pods -l app=proteinlens-esm35m-extract -w
kubectl logs -f -l app=proteinlens-esm35m-extract
```

Expected ~4–6 h. It writes (under the `/data/data` root the sweep job reads)
`/data/data/training_embeddings/esm2_35m/layer_6/`,
`/data/data/eval_shards/shard_0.fasta`, and `/data/data/eval_sequences.txt`.

## 5. Create the sweep and wire the job

From your laptop / a login shell (needs `wandb` + your login):

```bash
wandb sweep --project proteinlens-sae-esm35m relu_sweep_esm35m.yaml
# -> prints the full sweep path, e.g. <entity>/proteinlens-sae-esm35m/ab12cd34
```

Paste that full path into the `args:` line of `k8s/job-esm35m-sae-sweep.yaml`
(replacing `REPLACE_WITH_ENTITY/.../REPLACE_WITH_SWEEP_ID`). `WANDB_API_KEY` is
already set to the same key as the other proteinlens jobs — the account behind
it provides the default entity, so no separate entity var is needed.

## 6. Run the sweep agents

```bash
kubectl create -f k8s/job-esm35m-sae-sweep.yaml
kubectl get pods -l app=proteinlens-esm35m-sae-sweep -w
kubectl logs -f -l app=proteinlens-esm35m-sae-sweep
```

Worker pattern: `parallelism: 2`, `completions: 4` — 2 agent pods at a time, each
`wandb agent --count 1` (one grid point/pod), 4 runs total under one W&B sweep.
Bump `parallelism` to 4 to run the whole grid at once. ~6–9 h/run; the runs show
up together on wandb.ai. Checkpoints land in `/data/models/sweeps/layer_6/<wandb_run_id>/`.

Delete a job when done (leaves the PVC intact):

```bash
kubectl delete job -l app=proteinlens-esm35m-sae-sweep
```

## 7. Pull the trained SAEs back

With the data-access pod up (step 3), PVC mounted at `/data` there:

```bash
rsync -rvP \
  -e 'ssh -J <USER>@<GATEWAY_HOST>,<USER>@<INTERNAL_HOST> -p PORTNUMBER' \
  root@localhost:/data/models/sweeps/layer_6/ \
  trained_models/layer_6_esm35m/
```

**Adjust `PORTNUMBER`** to your access pod's actual NodePort each session. Pick
the best run by `performance/pct_loss_recovered` in W&B; its
`<run_id>/ae.pt` + `config.yaml` is the `--sae-dir` for the feature pipeline
(geometry + InterPro tables) next.
