# Agent runbook: AMPLIFY-120M SAE sweep on RunPod

For an AI agent with the **RunPod MCP tools**, a shell with `ssh`, `runpodctl`
and `wandb` (logged in as entity `s-setlur-university-of-edinburgh`), and this
repo checked out on branch `amplify`. The human runbook
(`RUNPOD_INSTRUCTIONS_amplify120m_sae.md`) explains *why*; this one says
exactly what to call, what to check, and when to stop.

## Hard rules (read before every phase)

1. **Budgets are hard caps on pod spend, per phase.** Spend = pod hours × $/h
   (from the `create-pod` response, `costPerHr`), cross-checked with
   `list-pod-billing`. At the cap the pod is **stopped immediately**, whatever
   it is doing. At 80% it is a WARN in the report.

   | Phase | Hard cap | Expected | Pod hours at $1.59/h |
   |---|---|---|---|
   | Smoke test | **$6** | ~$2 | 3.8 h |
   | Extraction | **$8** | ~$3 | 5.0 h |
   | Sweep | **$60** | $30–50 | 37.7 h |
   | Network volume | 100 GB ≈ $7/month (confirm in the create response) | | |

2. **Only one GPU pod exists at a time**, and it is stopped (`pod-action stop`)
   the moment its phase's script finishes or fails. A running pod with no
   `tmux` session doing work is a bug: stop it.
3. **Monitor continuously**: a tick every **10 min** (smoke, extraction) or
   **15 min** (sweep). A tick that cannot reach the pod or W&B twice in a row
   → stop the pod, report.
4. **State the hourly price before every `create-pod`**, and record every id
   you create in `runpod/agent_state.json` (schema at the end). Only ever
   stop/start/terminate/delete resources whose ids are in that file.
5. **Never create secrets with values you do not have**, never print a
   secret, never edit `relu_sweep_amplify120m.yaml` on your own (the L1
   decision in gate G4 is the user's).
6. **Scientific gates are not negotiable**: a failed gate ends the phase; you
   report and wait. You do not "fix and proceed".
7. **Verify W&B is receiving data before spending on a long phase** (G6).
   Training that is not logged is money spent for nothing.

## Preconditions (verify, do not assume)

| Check | Tool | Pass |
|---|---|---|
| Image exists | `docker manifest inspect sidd2000/proteinlens-pipeline:amplify` | returns a manifest. If not: stop, ask the user to build/push (§1 of the human runbook) |
| Secrets | `list-secrets` | names `hf-token`, `wandb-api-key`, `runpod-api-key` present. If missing: ask the user to create them (console) — you cannot |
| SSH key | `get-ssh-keys` | ≥1 key, and its private key is on this machine |
| Local W&B | `python -c "import wandb; print(wandb.Api().default_entity)"` | prints the entity |
| Sweep | `wandb sweep --entity s-setlur-university-of-edinburgh --project proteinlens-sae-amplify120m relu_sweep_amplify120m.yaml` | creates it; record `sweep_id`. Do this only after gate G4 (the grid may change) |
| No stray pods | `list-pods` | none, or only pods in `agent_state.json` |

## Phase 0: data centre, volume, template (no GPU spend)

1. `get-gpu-type id="NVIDIA A100 80GB PCIe" include=[AVAILABILITY] product=[POD] cloud=SECURE`.
   Pick `dc` = the first of `EU-RO-1`, `US-KS-2` whose availability is not
   `NONE` **and** that appears in `list-data-centers networkVolumeTypes=[STANDARD]`.
   If neither: `get-gpu-type id="NVIDIA H100 80GB HBM3"` (secure $3.49/h,
   halves the pod-hour budgets) and pick a volume-capable DC from its list;
   report the price change to the user before continuing.
2. `create-network-volume {name: "proteinlens-amplify", size: 100, dataCenter: dc, type: STANDARD}` → record `volume_id`, `dc`.
3. `create-template` with:
   ```json
   {"name": "proteinlens-amplify", "image": "sidd2000/proteinlens-pipeline:amplify",
    "args": "/workspace/runpod/start.sh", "disk": 20, "ports": ["22/tcp"],
    "startSsh": true, "startJupyter": false, "category": "NVIDIA",
    "env": {"HF_TOKEN": "{{ RUNPOD_SECRET_hf-token }}",
            "WANDB_API_KEY": "{{ RUNPOD_SECRET_wandb-api-key }}",
            "RUNPOD_API_KEY": "{{ RUNPOD_SECRET_runpod-api-key }}",
            "WANDB_ENTITY": "s-setlur-university-of-edinburgh",
            "WANDB_PROJECT": "proteinlens-sae-amplify120m"}}
   ```
   → record `template_id`. (No mount in the template; the API rejects it.)

## Pod lifecycle (used by every phase)

**Create** (state the price first):
```json
create-pod {"name": "amplify-<phase>", "templateId": template_id, "cloud": "SECURE",
            "gpu": {"id": "NVIDIA A100 80GB PCIe", "count": 1}, "dataCenterIds": [dc],
            "mounts": {"network": [{"volumeId": volume_id, "path": "/data"}]}}
```
Record `pod_id`, `costPerHr`, `created_at` (now, UTC). Poll `get-pod` every 30 s
up to 10 min for `RUNNING` and a populated `ssh` block. Not RUNNING in 10 min →
`pod-action terminate`, retry once; second failure → report (stock).

**Shell**: `runpodctl ssh info <pod_id>` gives the `ssh` command; every remote
command is `ssh <target> '<cmd>'` (non-interactive, returns immediately).
Long work runs detached in tmux with an exit sentinel:
```bash
ssh <target> 'tmux new -d -s <phase> "cd /workspace && bash runpod/00_setup.sh > /data/logs/<phase>.log 2>&1 && bash runpod/<script> >> /data/logs/<phase>.log 2>&1; echo EXIT=\$? >> /data/logs/<phase>.log"'
```
The first `00_setup.sh` on a fresh volume downloads the FASTA (184 MB) and
verifies sha256 `70ade89081c29b30a05fa10e2c100f1fe8faf147570b1c3c0ae4b6b573b42e25`
— the same SwissProt file every ESM run used. A sha mismatch is a hard stop.

**Tick** (every phase, at the phase's cadence), all four, then evaluate:
1. `get-pod pod_id` → status is `RUNNING` (else: the pod died; read
   `stream-pod-logs`, report).
2. `ssh <target> 'tail -n 3 /data/logs/<phase>.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'`
   → note GPU util/memory; a `Traceback`/`Error` line or `EXIT=` ends the phase.
3. Spend = (now − created_at) × costPerHr; every 4th tick also
   `list-pod-billing podId=pod_id bucketSize=hour` and use the larger figure.
   ≥ cap → `pod-action stop` **now**, then report. ≥ 80% → WARN.
4. Phase-specific check (below).

**Finish**: on `EXIT=0` copy the log locally
(`ssh <target> 'cat /data/logs/<phase>.log' > runpod/logs/<phase>.log`), then
`pod-action stop`. On `EXIT=<nonzero>`: same, and the phase has failed.

**Reuse**: the next phase first tries `pod-action start` on the stopped pod
(keeps its disk and the volume attachment). If `start` returns an error
(stock), `pod-action terminate` it and `create-pod` again.

## Phase 1: smoke test (cap $6, cadence 10 min, expect ~1 h)

Script: `runpod/10_smoke.sh`. Phase check per tick: none beyond the common
ones. On `EXIT=0`, evaluate **all** gates from `runpod/logs/smoke.log` and W&B:

| Gate | Where | Pass |
|---|---|---|
| **G1 xformers path correct** | log lines `GPU single vs CPU single`, `GPU batched vs GPU single` | both `PASS` (max rel diff < 1e-3) |
| **G2 same-protein comparison ran** | log JSON from `compare_amplify_esm35m.py`; W&B run `smoke-compare-esm35m` (job_type `smoke`) exists with summary keys `esm2_35m_l6_sae_cgvpk5vp/l0`, `norm_ratio_amplify_over_esm35m` | `norm_ratio_amplify_over_esm35m` in 3–4.5; `esm2_35m_l6_sae_cgvpk5vp/l0` in 350–550 and `variance_explained` in 0.90–0.96 (sanity: the published 35M SAE behaves as on its own run) |
| **G3 AMPLIFY SAEs train** | two `SMOKE lr=0.0003 l1=… ` lines; W&B runs `smoke-lr0.0003-l1{0.05,0.08}-5000steps` **and** their `-summary` runs | `pct_loss_recovered` > 10, `var_explained` > 0.3, `dead_frac` < 0.9 for both |
| **G4 L1 grid still sensible** (user decides) | `smoke/l0` of both vs the 35M references (final 35M: L0 457/377 at L1 0.05, 131/117 at L1 0.08; smoke `esm2_35m_l6_sae_cgvpk5vp/l0` on the same proteins) | Report both L0s and the ratio. If AMPLIFY's L0 at 5k steps is more than ~3× the 35M reference at the same L1, recommend scaling the grid ×3.5 (`[0.18, 0.28]`), **and wait for the user's decision**. Otherwise recommend keeping the grid and continue |
| **G5 capacity plan** | `peak_gpu_mem_gb` (max over the two runs); extraction `real` time; `GPU batched` proteins/s | `N = min(4, floor(64 / peak_gpu_mem_gb))`; `extract_est_h = 7 × smoke_extract_h`; `run_est_h = 500000 / steps_per_s / 3600` where `steps_per_s` comes from the smoke training runs' `_step/_runtime` in W&B. Record all three |

Report: the gate table with the numbers, the smoke spend, and (G4) the
recommendation. **Stop here until the user confirms the grid**; then create
the sweep (preconditions table) and record `sweep_id`.

## Phase 2: extraction (cap $8, cadence 10 min, expect `extract_est_h`)

Script: `runpod/20_extract.sh`. If `extract_est_h × costPerHr > $8` before
starting, report and wait instead of starting. Phase check per tick:
`ssh <target> 'ls /data/data/training_embeddings/amplify_120m/layer_12 2>/dev/null | wc -l'`
must grow (41 shards at the end). On `EXIT=0` verify: 41 shard dirs, each
with `activations.pt` + `metadata.yaml` (`d_model: 640`, `layer: 12`,
`revision: d918a9e8…`), `/data/data/eval_sequences.txt` has 100 lines, and the
logged `du -sh` is 30–40 GB. Stop the pod.

## Phase 3: sweep (cap $60, cadence 15 min)

Before starting: `projected = 4 × run_est_h × costPerHr / N` (+10% for the
fidelity baselines). If `projected > $50`, report and wait: either the user
accepts a smaller N is impossible, or a run count/steps change is theirs to
make. Start the pod, then:
```bash
ssh <target> 'tmux new -d -s sweep "cd /workspace && bash runpod/00_setup.sh > /data/logs/sweep.log 2>&1 && bash runpod/30_sweep.sh s-setlur-university-of-edinburgh/proteinlens-sae-amplify120m/<sweep_id> <N> >> /data/logs/sweep.log 2>&1; echo EXIT=\$? >> /data/logs/sweep.log"'
```
**G6 W&B integration, 10 min after start (mandatory):** `wandb.Api().runs(...)`
in project `proteinlens-sae-amplify120m` shows N runs with `state == running`,
`_step` increasing between two reads 2 min apart, and `system.gpu.0.gpu`
present in their system metrics; the sweep page lists them under `sweep_id`.
If not → `pod-action stop`, read `/data/logs/agent_*.log`, report.

Per-tick phase check:
```bash
python scripts/sweep_progress.py --project proteinlens-sae-amplify120m --pod-started <created_at> --price <costPerHr> --budget 60
ssh <target> 'tail -n 2 /data/logs/gpu_monitor.csv; grep -l "Traceback\|CUDA out of memory" /data/logs/agent_*.log'
```
- `STOP` verdict (spend ≥ $60 or projection > $60) → `pod-action stop` now,
  report with the table. Checkpoints every 50k steps are on the volume; a
  resume is the user's call.
- `WARN` (projection > $48) → include the table in the report, continue.
- `CUDA out of memory` in an agent log → that agent has exited; the sweep
  controller will requeue its grid point on another agent. Note it; if two
  agents OOM, stop the pod and report (N was too high).
- Pod-level GPU util (`gpu_monitor.csv`) averaging < 50% for 30 min with
  N < 4 and memory headroom > `peak_gpu_mem_gb` → report "N+1 possible"; do
  not change N mid-sweep (a new agent is a new process; the user can start
  it with `wandb agent <sweep_id>` in another tmux if they choose).

Completion: `30_sweep.sh` stops the pod itself. When `get-pod` shows
`EXITED` (or `EXIT=` appears in the log), verify in W&B: 4 runs `finished`,
each `_step ≥ 499900`, each with `performance/fidelity.pct_loss_recovered`,
`performance/l0_sparsity`, `performance/variance_explained`. Then:
```bash
python scripts/summarize_sae_sweep.py --project proteinlens-sae-amplify120m
```
Report the table plus the phase spend (`list-pod-billing`). **The choice of
SAE is the user's** (loss recovered vs L0 vs variance explained vs the 35M
reference); ask, do not pick.

## Phase 4: retrieval and cleanup (cap $3)

After the user names `run_id`: `pod-action start` (or create a pod on the
cheapest available GPU with the volume at `/data`), then
```bash
ssh <target> 'cd /workspace && . runpod/env.sh && huggingface-cli upload Sidd2010/proteinlens-sae-amplify120m /data/models/sweeps/layer_12/<run_id> trained_models/amplify_120m/layer_12/<run_id> --repo-type model'
```
Verify with `huggingface_hub.HfApi().list_repo_files("Sidd2010/proteinlens-sae-amplify120m")`
that `ae.pt`, `config.yaml`, `final_evaluation.yaml` are there, then
`pod-action terminate` every pod in `agent_state.json` (a stopped pod still
bills its disk). Keep the volume and template until the user says otherwise
(the embeddings may be wanted for a resume); say so in the final report with
the volume's monthly cost. Final report: total spend per phase from
`list-pod-billing`, the HF path, and the sweep table.

## `runpod/agent_state.json`

```json
{"dc": "EU-RO-1", "volume_id": "", "template_id": "", "sweep_id": "",
 "pods": [{"phase": "smoke", "pod_id": "", "costPerHr": 1.59, "created_at": "", "stopped_at": "", "spend_usd": 0}],
 "gates": {"G1": null, "G2": null, "G3": null, "G4": null, "G5": {"N": null, "peak_gpu_mem_gb": null, "extract_est_h": null, "run_est_h": null}, "G6": null},
 "chosen_run_id": ""}
```
Update it after every mutation and every finished phase; it is the only
record of what this agent is allowed to touch.
