# SAE Feature Clustering

## 1. Run clustering on the fiery-sweep SAE

```bash
conda activate interplm
python scripts/cluster_sae_features.py \
    --sae-dir trained_models/fiery-sweep \
    --n-clusters 500 \
    --output protein_results/clusters.yaml
```

**What it does:** loads `trained_models/fiery-sweep/ae.pt`, extracts the decoder weight matrix (5120 features × 320 dims), builds a pairwise angular-similarity matrix in row-chunks, and runs scikit-learn `SpectralClustering`. Takes ~5–20 minutes on CPU for 5120 features.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--sae-dir` | required | Path to trained SAE directory |
| `--n-clusters` | required | Number of clusters |
| `--output` | required | Where to write the YAML |
| `--chunk-size` | 1024 | Rows per chunk when building similarity matrix. Reduce if OOM |
| `--device` | auto | Device for loading SAE (similarity matrix always on CPU) |

---

## 2. Output YAML — format and location

The YAML is written to whatever path you pass to `--output`. Recommended location: `protein_results/clusters.yaml`.

**Format:** integer cluster index → sorted list of integer feature indices.

```yaml
0:
- 42
- 1337
- 5012
1:
- 7
- 99
- 204
2:
- 0
- 15
...
```

Every feature index from `0` to `dict_size - 1` (5119 for fiery-sweep) appears exactly once across all cluster lists.

---

## 3. Load the clustering and query it

```python
from proteinlens.analysis.feature_clusters import FeatureClusters

fc = FeatureClusters.from_file("protein_results/clusters.yaml")

# How many clusters?
len(fc)           # e.g. 500

# How many features total?
fc.n_features()   # 5120

# Which feature indices are in cluster 7?
fc.get_features(7)
# [83, 204, 1019, 2847, 3301, ...]

# Which cluster does feature 42 belong to?
fc.get_cluster(42)
# 0

# Top proteins for a cluster, using Per_feature_max_examples.yaml
import yaml
max_examples = yaml.safe_load(open("protein_results/Per_feature_max_examples.yaml"))

proteins = fc.get_top_proteins(
    cluster_idx=7,
    max_examples=max_examples,
    n_per_feature=5,   # take up to 5 top proteins per feature
)
# Returns a deduplicated list sorted by how many cluster features
# activate on each protein (most → first).
# e.g. ['P62807', 'Q9CR21', 'P00698', ...]
```

---

## 4. Intervene on a cluster

### Via `intervene_and_fold.py` (command line)

```bash
# Ablate (zero out) all features in cluster 7:
python scripts/intervene_and_fold.py \
    --accession P00698 \
    --cluster-file protein_results/clusters.yaml \
    --cluster-idx 7 \
    --cluster-action zero

# Amplify all features in cluster 7 by 3×:
python scripts/intervene_and_fold.py \
    --accession P00698 \
    --cluster-file protein_results/clusters.yaml \
    --cluster-idx 7 \
    --cluster-action scale \
    --cluster-value 3.0

# Set all features in cluster 7 to a fixed value:
python scripts/intervene_and_fold.py \
    --accession P00698 \
    --cluster-file protein_results/clusters.yaml \
    --cluster-idx 7 \
    --cluster-action set \
    --cluster-value 5.0

# Add to all features in cluster 7 only at positions 10–30:
python scripts/intervene_and_fold.py \
    --accession P00698 \
    --cluster-file protein_results/clusters.yaml \
    --cluster-idx 7 \
    --cluster-action add \
    --cluster-value 2.0 \
    --cluster-positions "10-30"

# Show top proteins for the cluster at the same time:
python scripts/intervene_and_fold.py \
    --accession P00698 \
    --cluster-file protein_results/clusters.yaml \
    --cluster-idx 7 \
    --cluster-action zero \
    --max-examples protein_results/Per_feature_max_examples.yaml \
    --cluster-top-n 5

# Mix cluster intervention with individual feature interventions:
python scripts/intervene_and_fold.py \
    --accession P00698 \
    --cluster-file protein_results/clusters.yaml \
    --cluster-idx 7 \
    --cluster-action zero \
    --interventions "42:scale:2.0"
```

**Cluster argument reference:**

| Flag | Default | Description |
|------|---------|-------------|
| `--cluster-file` | — | Path to clusters YAML |
| `--cluster-idx` | — | Which cluster to intervene on |
| `--cluster-action` | `zero` | `scale` / `set` / `zero` / `add` |
| `--cluster-value` | `1.0` | Scalar for scale/set/add |
| `--cluster-positions` | all | Position spec, e.g. `"10-30"` or `"5,10,15"` |
| `--max-examples` | — | Path to `Per_feature_max_examples.yaml` for top-protein display |
| `--cluster-top-n` | `3` | Proteins taken per feature in the display |

### Via Python API

```python
from proteinlens.analysis.feature_clusters import FeatureClusters

fc = FeatureClusters.from_file("protein_results/clusters.yaml")

# Build a list of FeatureIntervention objects — one per feature in the cluster.
# These are the same objects used by intervene_and_fold.py internally.
interventions = fc.make_interventions(
    cluster_idx=7,
    action="zero",          # or "scale", "set", "add"
    value=1.0,              # ignored for "zero"
    positions=None,         # or a list of 0-indexed residue positions
)

# Apply them to a feature tensor (seq_len, dict_size):
for iv in interventions:
    iv.apply(features, seq_len)
```

---

## 5. Test script

**File:** `scripts/test_feature_clusters.py`

```bash
conda activate interplm
python scripts/test_feature_clusters.py
```

**What it tests (40 checks total):**

| Section | Checks | What's verified |
|---------|--------|-----------------|
| YAML round-trip | 2 | `save()` then `from_file()` reproduces identical clusters and reverse map |
| Inverse property | 2 | `get_cluster(f) == c` for every `f` in `get_features(c)`, and reverse |
| Error cases | 2 | `KeyError` on unknown cluster index or unknown feature index |
| len / n_features | 2 | Counts are correct |
| `get_top_proteins` | 7 | Vote-ranking, empty-list edge case, string keys, `n_per_feature` truncation |
| `_get_decoder_weights` | 3 | Returns shape `(dict_size, activation_dim)` for `ReLUSAE` (requires `.T` transpose) |
| `_spectral_cluster` | 2 | All features assigned; 3 well-separated synthetic groups are recovered |
| `make_interventions` | 6 | Correct count, feature indices, action, value, positions for each intervention object |
| End-to-end equivalence | 12 | For each of `scale`, `zero`, `set`, `add`: cluster intervention via `make_interventions` produces identical feature tensors, identical SAE-decoded hidden states, and identical ESM2 logits compared to applying the same action feature-by-feature |

The end-to-end test uses the real fiery-sweep SAE and ESM2-8M on a short lysozyme fragment. It requires the SAE to be present at `trained_models/fiery-sweep/` and is skipped automatically if not found.
