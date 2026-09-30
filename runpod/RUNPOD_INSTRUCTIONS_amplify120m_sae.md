# AMPLIFY-120M SAEs on RunPod

Scope: **SAE training only** (smoke test → extraction → 4-run sweep → pick).
The annotation pipeline stays on EIDF (`k8s/CLUSTER_INSTRUCTIONS_amplify120m_pipeline.md`).

RunPod bills per second while a pod exists, so the rules here are: one GPU,
nothing idle, everything durable on a network volume, and the pod stops itself
when the sweep ends. Everything is logged to the W&B project
**`proteinlens-sae-amplify120m`** (entity `s-setlur-university-of-edinburgh`).

## What gets pulled where (no streaming needed)

| Data | Source | Size | Why |
|---|---|---|---|
| SwissProt FASTA | HF private dataset `Sidd2010/ProteinLensData`, `geometry_enrichment/swissprot_all.fasta` | 184 MB | **The same file every ESM run used**: `00_setup.sh` checks sha256 `70ade890…` against the local `trained_models/layer_4/frosty-sweep-15/analysis/swissprot_all.fasta` |
| ESM2-35M SAE `cgvpk5vp` | HF model `Sidd2010/proteinlens-sae-esm2-35m` | 60 MB | reference L0 / variance explained on the same proteins |
| ESM2-35M model, AMPLIFY-120M | HF hub | 150 MB, 480 MB | ESM2-35M activations are **recomputed on the pod** (the model is tiny), not downloaded |

The ESM2-35M training activations were never uploaded to HF, so there is
nothing large to stream. The two comparisons the smoke test needs (residual
stream scale, and the published 35M SAE's L0 on identical proteins) come from
recomputing 1000 proteins through ESM2-35M on the GPU, which takes seconds.

The training subset itself (42k proteins, ≤1022 aa) follows the 35M recipe. The
35M run's `subset_fasta.py` was unseeded, so its exact protein list is not
recoverable; ours is seeded (`--seed 42`) so it is.

## 1. One-off setup (RunPod console)

1. **Network volume** (Storage → Network Volumes): **100 GB**, in a data centre
   that has A100 80GB stock. ~35 GB embeddings + ~5 GB smoke + ~2 GB SAEs +
   HF cache + logs. Pods can only attach volumes from their own data centre.
2. **Template** (Templates → New):
   - Image: `sidd2000/proteinlens-pipeline:amplify` (build and push from the
     `amplify` branch first, see §2)
   - Container start command: `bash -c "sleep infinity"`
   - Container disk: 20 GB. **Volume mount path: `/data`** (not the default
     `/workspace`: the image keeps the repo at `/workspace` and the volume would
     shadow it)
   - Environment variables (as RunPod secrets): `HF_TOKEN` (read token; the
     FASTA dataset is private), `WANDB_API_KEY`,
     `WANDB_ENTITY=s-setlur-university-of-edinburgh`, `RUNPOD_API_KEY` (for
     the self-stop). `RUNPOD_POD_ID` is set by RunPod.
3. **Sweep** (laptop, once):
   ```bash
   wandb sweep --entity s-setlur-university-of-edinburgh --project proteinlens-sae-amplify120m relu_sweep_amplify120m.yaml
   # -> s-setlur-university-of-edinburgh/proteinlens-sae-amplify120m/<sweep_id>
   ```

## 2. Image

```bash
docker build -f k8s/Dockerfile.pipeline -t sidd2000/proteinlens-pipeline:amplify .
docker push  sidd2000/proteinlens-pipeline:amplify
```

## 3. Smoke test pod (gate; ~1 h of A100 time)

Deploy **1× A100 80GB** from the template with the volume attached. In the web
terminal (or SSH; `tmux` so a dropped session doesn't kill the run):

```bash
tmux new -s smoke
cd /workspace && bash runpod/00_setup.sh
bash runpod/10_smoke.sh 2>&1 | tee /data/logs/smoke.log
```

Then **stop the pod** (console, or `python -c 'import os,runpod; runpod.api_key=os.environ["RUNPOD_API_KEY"]; runpod.stop_pod(os.environ["RUNPOD_POD_ID"])'`).
Read `/data/logs/smoke.log` and the W&B `smoke` runs:

| Line | Must hold |
|---|---|
| `GPU single vs CPU single`, `GPU batched vs GPU single` | `PASS` (max rel diff < 1e-3): the xformers path is numerically right |
| `norm_ratio_amplify_over_esm35m` | expect ≈3.5. It's the factor by which a fixed L1 is weaker on AMPLIFY |
| `esm2_35m_l6_sae_cgvpk5vp`: `l0`, `variance_explained` | the reference (35M final: L0 ≈ 457, var ≈ 0.945 at L1 0.05; L0 ≈ 131, var ≈ 0.90 at L1 0.08) |
| `SMOKE lr=0.0003 l1=0.05 …` and `… l1=0.08 …` | `pct_loss_recovered` and `var_explained` clearly non-zero; `dead_frac` not near 1. Compare L0 with the 35M reference above: if it's several times higher at both L1 values, scale `l1_coefficient` in `relu_sweep_amplify120m.yaml` (≈ ×3.5 → `[0.18, 0.28]`) and recreate the sweep before §5 |
| `peak_gpu_mem_gb` | sets the number of concurrent agents: `N = floor(0.8 × 80 / peak)`, capped at 4 |
| `GPU single`/`GPU batched` throughput, extraction `real` time | sizes §4 (42 shards ≈ 7× the smoke extraction) |

## 4. Extraction (same pod type; ~1–3 h)

```bash
tmux new -s extract
cd /workspace && bash runpod/00_setup.sh
bash runpod/20_extract.sh 2>&1 | tee /data/logs/extract.log
```

Writes `/data/data/training_embeddings/amplify_120m/layer_12/` (~35 GB),
`/data/data/eval_shards/shard_0.fasta`, `/data/data/eval_sequences.txt`. Stop
the pod unless going straight to §5 on the same pod.

## 5. Sweep (one pod, N agents on the one GPU, self-stopping)

The 35M sweep averaged **51% GPU utilisation** with one run per GPU (the
trainer is data-loader/Python bound), so one agent per A100 pays for idle
silicon. Run `N` agents concurrently (from the smoke `peak_gpu_mem_gb`); the
W&B controller hands each its grid points and they exit when the sweep is
exhausted. `gpu_monitor.sh` samples `nvidia-smi` every 60 s to
`/data/logs/gpu_monitor.csv` and warns after 20 idle minutes (`AUTO_STOP=1`
makes it stop the pod instead). When all agents exit, `30_sweep.sh` **stops the
pod** through the RunPod API; the volume keeps the checkpoints.

```bash
tmux new -s sweep
cd /workspace && bash runpod/00_setup.sh
bash runpod/30_sweep.sh s-setlur-university-of-edinburgh/proteinlens-sae-amplify120m/<sweep_id> <N> 2>&1 | tee /data/logs/sweep.log
```

Watch: the W&B sweep page (per-run `system.gpu.0.gpu`/`memoryAllocated`), and
`tail -f /data/logs/gpu_monitor.out /data/logs/agent_*.log`. If pod-level GPU
util sits well under ~85% with N agents and memory allows, N+1 is free money;
if it is memory-bound, don't.

Budget: 35M runs took 4.4–4.7 h each on an A100 at 15,360 features; AMPLIFY's
SAE is ~1.8× the parameters, so expect ~6–8 h per run, ~20–30 GPU-hours total,
compressed into ~8–15 h wall time with 2–4 agents.

Checkpoints: `/data/models/sweeps/layer_12/<wandb_run_id>/{ae.pt,config.yaml,final_evaluation.yaml}`.

## 6. Picking the SAE — a judgement call, not `argmax(pct_loss_recovered)`

```bash
python scripts/summarize_sae_sweep.py --project proteinlens-sae-amplify120m
```

prints, per run: lr, L1, % loss recovered, L0, variance explained, hours, next
to the 35M reference values. Note that in the 35M sweep the highest loss
recovered came with **L0 = 457**. Weigh loss recovered against L0 and variance
explained (and the smoke `dead_frac`) and choose; nothing here auto-selects.

## 7. Get the SAE off the volume

On a stopped pod nothing runs, so start a cheap CPU pod (or the same template
on the smallest GPU) with the volume attached and copy out:

```bash
# from the pod: push the chosen run to HF next to the 35M one
huggingface-cli upload Sidd2010/proteinlens-sae-amplify120m /data/models/sweeps/layer_12/<run_id> trained_models/amplify_120m/layer_12/<run_id> --repo-type model
```

or `runpodctl send` / `scp` to the laptop. Then stage it on the EIDF PVC at
`/data/models/sweeps/layer_12/<run_id>/` for the pipeline runbook. Delete the
network volume once the embeddings are no longer needed (it bills monthly).
