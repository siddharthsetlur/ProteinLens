#!/bin/bash
# runpod/10_smoke.sh -- the gate before the sweep (~1 h on one A100).
#   1. xformers GPU path vs CPU, batched vs single, throughput
#   2. AMPLIFY L12 vs ESM2-35M L6 scale on the same proteins + the published 35M SAE as reference
#   3. 5k-step AMPLIFY SAEs at the two sweep L1 values: L0 / var explained / loss recovered / peak GPU mem
# Everything is logged to W&B project $WANDB_PROJECT (job_type=smoke). Writes under /data/smoke.
set -euo pipefail
. "$(dirname "$0")/env.sh"      # PID-1 env (secrets, RUNPOD_POD_ID) into this shell
W=/data; SM=$W/smoke; export HF_HOME=$W/hf_cache
export WANDB_PROJECT=${WANDB_PROJECT:-proteinlens-sae-amplify120m}
mkdir -p $SM/shards $SM/heldout
cd /workspace   # repo copy baked into the image

echo "=== 1. subset + shard (6000 proteins, <=1022 aa, seeded) ==="
python scripts/subset_fasta.py --input_file $W/uniprot/uniprot_sprot.fasta \
    --output_file $SM/subset.fasta --num_proteins 6000 --max_length 1022 --seed 1
python scripts/shard_fasta.py --input_file $SM/subset.fasta --output_dir $SM/shards --proteins_per_shard 1000

echo "=== 2. GPU (xformers) vs CPU (SDPA), batched vs single, throughput ==="
python scripts/check_amplify_gpu.py --fasta $SM/shards/shard_0.fasta --n 200 --layer 12

echo "=== 3. AMPLIFY L12 vs ESM2-35M L6 on the same 1000 proteins (+ 35M SAE reference from HF) ==="
python scripts/compare_amplify_esm35m.py --fasta $SM/shards/shard_1.fasta --n 1000 \
    --out $SM/compare_esm35m.json --wandb-project $WANDB_PROJECT

echo "=== 4. extraction timing (6 shards) ==="
time python scripts/extract_embeddings.py --fasta_dir $SM/shards --output_dir $SM/emb \
    --embedder_type amplify --model_name chandar-lab/AMPLIFY_120M --layers 12 --batch_size 8
mv $SM/emb/layer_12/shard_5 $SM/heldout/
python -c 'from Bio import SeqIO; recs=list(SeqIO.parse("/data/smoke/shards/shard_5.fasta","fasta"))[:100]; open("/data/smoke/eval_sequences.txt","w").write("\n".join(str(r.seq) for r in recs)+"\n")'

echo "=== 5. 5k-step SAEs (lr 3e-4, 20480 features) ==="
for L1 in 0.05 0.08; do
  python scripts/amplify_smoke_sae.py --embd-dir $SM/emb/layer_12 --eval-seqs $SM/eval_sequences.txt \
      --heldout $SM/heldout/shard_5/activations.pt --lr 3e-4 --l1 $L1 --steps 5000 \
      --eval-batch-size 32 --out $SM/sae_l1_$L1 --wandb-project $WANDB_PROJECT
done

echo "=== SMOKE DONE ==="
