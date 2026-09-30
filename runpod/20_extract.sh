#!/bin/bash
# runpod/20_extract.sh -- training activations for the sweep (same recipe as ESM2-35M:
# 42k proteins <=1022 aa from the same FASTA, 1000/shard, last shard held out for eval).
# Seeded, so the subset is reproducible. Output ~35 GB under /data/data.
set -euo pipefail
. "$(dirname "$0")/env.sh"      # PID-1 env (secrets, RUNPOD_POD_ID) into this shell
W=/data; D=$W/data; export HF_HOME=$W/hf_cache
mkdir -p $D/uniprot_amplify $D/training_shards_amplify $D/eval_shards
cd /workspace   # repo copy baked into the image

python scripts/subset_fasta.py --input_file $W/uniprot/uniprot_sprot.fasta \
    --output_file $D/uniprot_amplify/subset.fasta --num_proteins 42000 --max_length 1022 --seed 42
python scripts/shard_fasta.py --input_file $D/uniprot_amplify/subset.fasta \
    --output_dir $D/training_shards_amplify --proteins_per_shard 1000
LAST=$(ls $D/training_shards_amplify/*.fasta | tail -n1)
mv "$LAST" $D/eval_shards/shard_0.fasta

time python scripts/extract_embeddings.py --fasta_dir $D/training_shards_amplify \
    --output_dir $D/training_embeddings/amplify_120m \
    --embedder_type amplify --model_name chandar-lab/AMPLIFY_120M --layers 12 --batch_size 8

python -c 'from Bio import SeqIO; recs=list(SeqIO.parse("/data/data/eval_shards/shard_0.fasta","fasta"))[:100]; open("/data/data/eval_sequences.txt","w").write("\n".join(str(r.seq) for r in recs)+"\n"); print(f"wrote {len(recs)} eval sequences")'
du -sh $D/training_embeddings/amplify_120m
echo "=== EXTRACT DONE ==="
