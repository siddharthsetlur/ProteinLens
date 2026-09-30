#!/bin/bash
# runpod/00_setup.sh -- first thing to run in a fresh pod (idempotent).
# Checks the secrets, pulls the SwissProt FASTA the ESM runs used, verifies it.
set -euo pipefail
W=/data                           # RunPod network volume, mounted at /data (persists across pods)
mkdir -p $W/uniprot $W/logs $W/hf_cache
export HF_HOME=$W/hf_cache        # model + dataset cache on the volume, not the container disk

for v in HF_TOKEN WANDB_API_KEY WANDB_ENTITY RUNPOD_API_KEY RUNPOD_POD_ID; do
  [ -n "${!v:-}" ] || { echo "ERROR: $v is not set (put it in the pod template env)"; exit 1; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import xformers.ops, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
wandb login --relogin "$WANDB_API_KEY" >/dev/null && echo "wandb ok"
pip install -q runpod            # python SDK used to stop the pod when the sweep finishes

# Same FASTA as every ESM run (local copy = trained_models/layer_4/frosty-sweep-15/analysis/swissprot_all.fasta,
# uploaded to the private HF dataset). sha256 must match or the protein universe differs.
SHA=70ade89081c29b30a05fa10e2c100f1fe8faf147570b1c3c0ae4b6b573b42e25
if ! echo "$SHA  $W/uniprot/uniprot_sprot.fasta" | sha256sum -c --quiet 2>/dev/null; then
  huggingface-cli download Sidd2010/ProteinLensData geometry_enrichment/swissprot_all.fasta \
      --repo-type dataset --local-dir $W/uniprot/_dl
  mv $W/uniprot/_dl/geometry_enrichment/swissprot_all.fasta $W/uniprot/uniprot_sprot.fasta
  rm -rf $W/uniprot/_dl
  echo "$SHA  $W/uniprot/uniprot_sprot.fasta" | sha256sum -c
fi
echo "setup done: $(grep -c '^>' $W/uniprot/uniprot_sprot.fasta) proteins"
