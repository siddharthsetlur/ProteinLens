### Extract PLM embeddings (training data for SAE)
As a small test we're going to use Swiss-Prot as suggested (later scale to random subsets of UniProt50). 
**Set the layer to analyze:**
```bash
# Choose which layer to extract and analyze (4 is middle layer for ESM-2-8M)
export LAYER=4
```
For now InterPLMDATA = `/Users/siddharthsetlur/Desktop/PhD/ProteinLens/data`
**Download the sequences from UNiprot**
```bash
# Download sequences
wget -P $INTERPLM_DATA/uniprot/ https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
```
**Select random subset and filter to proteins with length < 1022 for ESM-2 compatibility and Adjust num_proteins to increase the number of proteins kept**

```
python scripts/subset_fasta.py \
    --input_file $INTERPLM_DATA/uniprot/uniprot_sprot.fasta.gz \
    --output_file $INTERPLM_DATA/uniprot/subset.fasta \
    --num_proteins 5000
# Shard fasta into smaller files for training and evaluation
python scripts/shard_fasta.py \
    --input_file $INTERPLM_DATA/uniprot/subset.fasta \
    --output_dir $INTERPLM_DATA/uniprot_shards/ \
    --proteins_per_shard 1000  # 1000/shard -> ~0.5GB/shard for ESM-2 (8M)

# Set aside shard 0 as evaluation data (100 proteins for fidelity testing)
mkdir -p $INTERPLM_DATA/eval_shards/
mv $INTERPLM_DATA/uniprot_shards/shard_0.fasta $INTERPLM_DATA/eval_shards/
```

**Generate protein embeddings for training**
```bash
# Extract embeddings for training shards only (shard_0 is held out for evaluation)
python scripts/extract_embeddings.py \
    --fasta_dir /Users/siddharthsetlur/Desktop/PhD/ProteinLens/data/uniprot_shards/ \
    --output_dir /Users/siddharthsetlur/Desktop/PhD/ProteinLens/data/training_embeddings/esm2_8m/ \
    --embedder_type esm \
    --model_name facebook/esm2_t6_8M_UR50D \
    --layers $LAYER \
    --batch_size 32
```