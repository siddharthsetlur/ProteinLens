#!/bin/bash
# runpod/30_sweep.sh -- run the 4-point grid with N agents sharing the one GPU, then stop the pod.
# Usage: runpod/30_sweep.sh <entity>/proteinlens-sae-amplify120m/<sweep_id> <N_AGENTS>
#   N_AGENTS = floor(0.8 * GPU memory / smoke peak_gpu_mem_gb), max 4 (one per grid point).
# Each agent pulls grid points from W&B until the sweep is exhausted, so N agents = N runs at once.
set -euo pipefail
. "$(dirname "$0")/env.sh"      # PID-1 env (secrets, RUNPOD_POD_ID) into this shell
SWEEP=${1:?sweep path}; N=${2:?number of agents}
W=/data; export HF_HOME=$W/hf_cache
cd /workspace   # repo copy baked into the image

# relu_sweep.py reads these (same names as the k8s job env block)
export LAYER=12 EMBED_TAG=amplify_120m EMBEDDING_DIM=640 ESM_MODEL_NAME=chandar-lab/AMPLIFY_120M
export INTERPLM_DATA=$W/data MODEL_DIR=$W/models EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}

nohup bash "$(dirname "$0")/gpu_monitor.sh" > $W/logs/gpu_monitor.out 2>&1 &
MON=$!
for i in $(seq 1 "$N"); do
  nohup wandb agent "$SWEEP" > $W/logs/agent_$i.log 2>&1 &
  sleep 60   # stagger start-up so the fidelity baselines don't all peak at once
done
wait $(jobs -p | grep -v "^$MON$") || true
kill $MON 2>/dev/null || true
echo "=== SWEEP DONE: $(ls -d $W/models/sweeps/layer_12/*/ | wc -l) run dirs under $W/models/sweeps/layer_12 ==="

# Stop billing. The volume (/data) keeps the checkpoints + logs.
runpodctl pod stop "$RUNPOD_POD_ID"
