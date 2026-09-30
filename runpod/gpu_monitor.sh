#!/bin/bash
# runpod/gpu_monitor.sh -- 60 s samples of GPU util/memory to /data/logs/gpu_monitor.csv,
# and a warning (or, with AUTO_STOP=1, a pod stop) if the GPU idles for 20 min while billed.
# W&B also records per-run system.gpu.* metrics; this is the pod-level view across agents.
. "$(dirname "$0")/env.sh"
CSV=/data/logs/gpu_monitor.csv; IDLE=0
echo "time,util_pct,mem_used_mb,mem_total_mb" >> $CSV
while true; do
  L=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "$(date -u +%FT%TZ),$L" >> $CSV
  U=${L%%,*}
  if [ "${U:-0}" -lt 15 ]; then IDLE=$((IDLE+1)); else IDLE=0; fi
  if [ $IDLE -ge 20 ]; then
    echo "$(date -u +%FT%TZ) WARNING: GPU util <15% for 20 min (idle pod = wasted billing)"
    if [ "${AUTO_STOP:-0}" = "1" ]; then
      runpodctl pod stop "$RUNPOD_POD_ID"
    fi
    IDLE=0
  fi
  sleep 60
done
