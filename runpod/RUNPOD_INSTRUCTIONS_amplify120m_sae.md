# AMPLIFY-120M SAEs on RunPod

Scope: **SAE training only** (smoke test → extraction → 4-run sweep → pick).
The annotation pipeline stays on EIDF (`k8s/CLUSTER_INSTRUCTIONS_amplify120m_pipeline.md`).

RunPod bills per second while a pod exists, so the rules here are: one GPU,
nothing idle, everything durable on a network volume, and the pod stops itself
when the sweep ends. Everything is logged to the W&B project
**`proteinlens-sae-amplify120m`** (entity `s-setlur-university-of-edinburgh`).

Verified against the live account and RunPod API on 2026-09-30: the account has
no pods, volumes, templates or secrets, so nothing is billing yet. Prices and
stock below are from that read; recheck stock right before deploying.

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

## 1. Image (build first; the template references it)

The image is built from `nvidia/cuda`, not a RunPod base, so it has no sshd.
`runpod/start.sh` (the template's start command) brings sshd up and idles;
without it you get **no SSH and no web terminal**. `openssh-server` and `tmux`
are in the Dockerfile.

```bash
docker build -f k8s/Dockerfile.pipeline -t sidd2000/proteinlens-pipeline:amplify .
docker push  sidd2000/proteinlens-pipeline:amplify
```

## 2. One-off account setup

1. **SSH key on the account** (RunPod injects it as `PUBLIC_KEY`; `start.sh` installs it):
   `runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub` (or Settings → SSH keys).
2. **Secrets** (Settings → Secrets, or `runpodctl`/MCP `create-secret`), named
   `hf-token` (read token; the FASTA dataset is private), `wandb-api-key`,
   `runpod-api-key` (for the self-stop). None exist yet.
3. **Agent-driven run?** `AGENT_RUNBOOK_amplify120m_sae.md` is the same flow
   with per-phase hard budgets, tick-by-tick monitoring and gates.
4. **Template** (Templates → New, or MCP `create-template`):
   - Image `sidd2000/proteinlens-pipeline:amplify`; container disk 20 GB
   - Container start command: `/workspace/runpod/start.sh`
   - Ports: `22/tcp`; start SSH: on; Jupyter: off
   - Env: `HF_TOKEN={{ RUNPOD_SECRET_hf-token }}`,
     `WANDB_API_KEY={{ RUNPOD_SECRET_wandb-api-key }}`,
     `RUNPOD_API_KEY={{ RUNPOD_SECRET_runpod-api-key }}`,
     `WANDB_PROJECT=proteinlens-sae-amplify120m`,
     `WANDB_ENTITY=s-setlur-university-of-edinburgh`. `RUNPOD_POD_ID` is
     injected by RunPod.
   - **No volume in the template**: templates can't hold a network-volume mount
     (API rejects it); the volume is attached when the pod is created (§3).
5. **Network volume**: 100 GB, STANDARD tier (~$7/month), in the data centre
   you will deploy in. A volume is pinned to its data centre and the pod must be
   in the same one, so pick the DC from GPU stock at that moment:

   | GPU (secure cloud) | $/h | DCs with stock at verification (all "LOW") | Volumes |
   |---|---|---|---|
   | A100 80GB PCIe | 1.59 | EU-RO-1, US-KS-2 | EU-RO-1 ✓ |
   | A100 SXM 80GB | 1.59 | US-MD-1, US-WA-1 | not in the volume-capable DC list |
   | H100 SXM 80GB | 3.49 | 8 DCs incl. US-NE-1, AP-JP-1, EU-NL-1 (overall MEDIUM) | ✓ |
   | RTX PRO 6000 96GB | 2.09 | 9 DCs incl. EUR-IS-1/2, US-NE-1 | ✓ |

   Default: **A100 PCIe in EU-RO-1** (`create-network-volume` with
   `dataCenter: EU-RO-1`). If A100 stock is gone when you deploy, create the
   volume in US-NE-1 and use an H100 instead; 2.2× the price, but the SAE runs
   are not GPU-bound, so don't pay for it unless A100s are unavailable.
6. **Sweep** (laptop, once):
   ```bash
   wandb sweep --entity s-setlur-university-of-edinburgh --project proteinlens-sae-amplify120m relu_sweep_amplify120m.yaml
   # -> s-setlur-university-of-edinburgh/proteinlens-sae-amplify120m/<sweep_id>
   ```

## 3. Deploying a pod (each of §4–§6)

Console: Deploy → the template → 1× GPU → attach the network volume with
**mount path `/data`** (not the default `/workspace`, which would hide the repo
baked into the image). Or `runpodctl pod create --template-id <id> --gpu-type
"NVIDIA A100 80GB PCIe" --network-volume-id <vol> --volume-mount-path /data
--data-center-id EU-RO-1` (MCP `create-pod`: `mounts.network[{volumeId, path:"/data"}]`).

Connect with `runpodctl ssh info <pod-id>` or the web terminal, then always
`tmux`. Every script starts by importing the template env from PID 1
(`runpod/env.sh`), because RunPod does not put template env vars into SSH or
web-terminal shells.

`00_setup.sh` installs `runpodctl` in the pod and checks it can see the pod,
which is what the self-stop (`runpodctl pod stop $RUNPOD_POD_ID`) relies on.

## 4. Smoke test pod (gate; ~1 h of A100 time ≈ $2)

```bash
tmux new -s smoke
cd /workspace && bash runpod/00_setup.sh
bash runpod/10_smoke.sh 2>&1 | tee /data/logs/smoke.log
runpodctl pod stop "$RUNPOD_POD_ID"     # or from the laptop / console
```

Read `/data/logs/smoke.log` and the W&B `smoke` runs:

| Line | Must hold |
|---|---|
| `GPU single vs CPU single`, `GPU batched vs GPU single` | `PASS` (max rel diff < 1e-3): the xformers path is numerically right |
| `norm_ratio_amplify_over_esm35m` | expect ≈3.5–3.8. It's the factor by which a fixed L1 is weaker on AMPLIFY |
| `esm2_35m_l6_sae_cgvpk5vp`: `l0`, `variance_explained` | the reference (35M final: L0 ≈ 457, var ≈ 0.945 at L1 0.05; L0 ≈ 131, var ≈ 0.90 at L1 0.08) |
| `SMOKE lr=0.0003 l1=0.05 …` and `… l1=0.08 …` | `pct_loss_recovered` and `var_explained` clearly non-zero; `dead_frac` not near 1. Compare L0 with the 35M reference above: if it's several times higher at both L1 values, scale `l1_coefficient` in `relu_sweep_amplify120m.yaml` (≈ ×3.5 → `[0.18, 0.28]`) and recreate the sweep before §6 |
| `peak_gpu_mem_gb` | sets the number of concurrent agents: `N = floor(0.8 × 80 / peak)`, capped at 4 |
| `GPU single`/`GPU batched` throughput, extraction `real` time | sizes §5 (42 shards ≈ 7× the smoke extraction) |

## 5. Extraction (same pod type; ~1–3 h ≈ $2–5)

```bash
tmux new -s extract
cd /workspace && bash runpod/00_setup.sh
bash runpod/20_extract.sh 2>&1 | tee /data/logs/extract.log
```

Writes `/data/data/training_embeddings/amplify_120m/layer_12/` (~35 GB),
`/data/data/eval_shards/shard_0.fasta`, `/data/data/eval_sequences.txt`. Stop
the pod unless going straight to §6 on the same pod.

## 6. Sweep (one pod, N agents on the one GPU, self-stopping)

The 35M sweep averaged **51% GPU utilisation** with one run per GPU (the
trainer is data-loader/Python bound), so one agent per A100 pays for idle
silicon. Run `N` agents concurrently (from the smoke `peak_gpu_mem_gb`); the
W&B controller hands each its grid points and they exit when the sweep is
exhausted. `gpu_monitor.sh` samples `nvidia-smi` every 60 s to
`/data/logs/gpu_monitor.csv` and warns after 20 idle minutes (`AUTO_STOP=1`
makes it stop the pod instead). When all agents exit, `30_sweep.sh` runs
`runpodctl pod stop` on itself; the volume keeps the checkpoints.

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
SAE is ~1.8× the parameters, so expect ~6–8 h per run, ~20–30 GPU-hours total
(≈ $30–50 at $1.59/h), compressed into ~8–15 h wall time with 2–4 agents.

Checkpoints: `/data/models/sweeps/layer_12/<wandb_run_id>/{ae.pt,config.yaml,final_evaluation.yaml}`.

## 7. Picking the SAE — a judgement call, not `argmax(pct_loss_recovered)`

```bash
python scripts/summarize_sae_sweep.py --project proteinlens-sae-amplify120m
```

prints, per run: lr, L1, % loss recovered, L0, variance explained, hours, next
to the 35M reference values. Note that in the 35M sweep the run with L0 = 457
was the one used, and that sweep has no loss-recovered metric at all (its
fidelity eval was disabled). Weigh loss recovered against L0 and variance
explained (and the smoke `dead_frac`) and choose; nothing here auto-selects.

## 8. Get the SAE off the volume, then clean up

A stopped pod runs nothing, so start the template on the cheapest GPU (or a
CPU pod) with the volume attached and copy out:

```bash
# from the pod: push the chosen run to HF next to the 35M one
huggingface-cli upload Sidd2010/proteinlens-sae-amplify120m /data/models/sweeps/layer_12/<run_id> trained_models/amplify_120m/layer_12/<run_id> --repo-type model
```

or `runpodctl send` / `scp`. Then stage it on the EIDF PVC at
`/data/models/sweeps/layer_12/<run_id>/` for the pipeline runbook.

Finally **terminate** the pods (a stopped pod still bills its container disk)
and delete the network volume once the embeddings are no longer needed.
