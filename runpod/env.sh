# runpod/env.sh -- source at the top of every pod script.
# RunPod injects template env vars (secrets, RUNPOD_POD_ID, ...) into PID 1 only;
# SSH and web-terminal shells do NOT get them. Copy them from PID 1's environment.
set -a
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(HF_TOKEN|WANDB_[A-Z_]+|RUNPOD_[A-Z_]+|PUBLIC_KEY)=' | sed 's/^\([A-Z_]*\)=\(.*\)$/\1=${\1:-"\2"}/')"
set +a
export HF_HOME=/data/hf_cache
