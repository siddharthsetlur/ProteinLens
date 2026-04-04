# Feature Data Pipeline for SAE Visualizer

## Context

We have a trained ReLUSAE (5120 features, 320D, ESM2-8M layer 3) at `trained_models/fiery-sweep/`. The end goal is an interprot-style visualizer. Before building any UI, we need to reliably compute per-feature data: top activating sequences, activation range samples, per-residue activation maps, coverage stats, and AlphaFold structures. This data must be extensible (append geometric features later) and structured for web serving.

**Decisions made:**
- Start with human proteome (~20K proteins), scale to full SwissProt later
- Normalized activation bins (% of feature max) for range sampling
- Sequence clustering at ~30% identity via MMseqs2 for deduplication/coverage
- JSON-per-feature storage format with shared PDB cache
- Memmap array for per-protein per-feature max activations (~400MB for human proteome)
- GPU-intensive stages run on EIDF cluster; scaffold K8s job files

---

## Pipeline Architecture: Two-Pass Design

The activation range bins are normalized by each feature's max activation, which is unknown until we've seen all data. This requires two passes:

- **Pass 1 (Survey):** Stream all proteins through ESM2 -> SAE. Record per-feature max activation, maintain top-20 heaps, count coverage stats. Do NOT store per-residue activations (too large).
- **Pass 2 (Collect):** Now knowing each feature's max, re-embed only the selected proteins (~few thousand unique). Store full per-residue activations. Download AlphaFold PDBs.

---

## Output Directory Structure

```
feature_data/
  pipeline_state.json           # resumability checkpoint
  dataset_stats.json            # total proteins, clusters, dedup info
  feature_max_activations.npy   # (5120,) global max per feature
  cluster_map.tsv               # MMseqs2 cluster assignments
  sequences.json                # {accession: sequence} for all stored proteins
  features/
    0000.json                   # per-feature data
    0001.json
    ...
    5119.json
  pdb_cache/
    P12345.pdb
    Q67890.pdb
    ...
```

**Per-feature JSON schema** (`features/NNNN.json`):
```json
{
  "feature_id": 42,
  "max_activation": 3.7,
  "dataset_coverage": {
    "pct_proteins_activated": 12.3,
    "pct_clusters_activated": 8.1,
    "n_proteins_activated": 2460,
    "n_clusters_activated": 1620,
    "total_proteins": 20000,
    "total_clusters": 19500,
    "activation_threshold": 0.05
  },
  "top_sequences": [
    {
      "accession": "P12345",
      "max_activation": 3.7,
      "mean_activation": 0.42,
      "per_residue_activations": [0.0, 0.0, 2.1, 3.7, 1.2, "..."],
      "sequence": "MKTL...",
      "sequence_length": 256,
      "pdb_available": true
    }
  ],
  "activation_bins": {
    "0.75-1.0": [ { "accession": "...", "max_activation": 0, "per_residue_activations": [] } ],
    "0.5-0.75": [],
    "0.25-0.5": [],
    "0.0-0.25": []
  }
}
```

---

## Pipeline Stages

### Stage 0: Data Acquisition & Clustering

**0a. Download SwissProt sequences**
- Use existing `fetch_swissprot_accessions(organism_taxid=9606)` from `protein_results/build_activation_dataset.py`
- Fetch sequences via `fetch_sequence()` (batch with `requests.Session`)
- Save as FASTA file: `feature_data/swissprot_human.fasta`
- Resumable: skip accessions already in FASTA

**0b. Cluster sequences with MMseqs2**
- Run `mmseqs easy-cluster swissprot_human.fasta cluster_result tmp --min-seq-id 0.3`
- Parse output to get cluster representative -> members mapping
- Save as `feature_data/cluster_map.tsv`
- This gives us ~30% identity clusters for coverage stats

**Files to create:**
- `proteinlens/analysis/feature_pipeline/data_acquisition.py`
- `proteinlens/analysis/feature_pipeline/clustering.py`

### Stage 1: Survey Pass

Stream all proteins through ESM2 -> SAE. For each protein, record:
- Per-feature max activation (to find global max)
- Top-20 heap per feature (accession + max activation)
- Coverage counter per feature (how many proteins/clusters activate it)

**Processing approach:**
- Use existing `ESM.embed_single_sequence()` for embedding
- Use `sae.encode()` for SAE features
- Process in batches of 64-128 sequences for GPU efficiency
- Checkpoint after every N proteins (e.g., 1000)

**Key reuse:**
- `PerProteinActivationTracker` from `proteinlens/analysis/per_protein_tracking.py`
- `load_sae()` from `proteinlens/sae/inference.py`
- `ESM` from `proteinlens/embedders/esm.py`

**Output:**
- `feature_data/feature_max_activations.npy` -- shape (5120,)
- `feature_data/protein_feature_maxes.npy` -- shape (n_proteins, 5120), memmap
- `feature_data/survey_top20.json` -- top-20 accessions per feature with max activation values
- `feature_data/survey_coverage.json` -- per-feature protein/cluster activation counts
- `feature_data/pipeline_state.json` -- tracks which proteins have been processed

**Files to create:**
- `proteinlens/analysis/feature_pipeline/survey.py`

### Stage 2: Selection

With global maxes known, determine which proteins need per-residue data:

1. Load `feature_max_activations.npy`
2. For each feature, compute bin thresholds: `[0, 0.25*max, 0.5*max, 0.75*max, max]`
3. Scan through `protein_feature_maxes.npy` to assign proteins to bins per feature
4. Select: top-20 (from survey) + top-10 per bin = up to 60 proteins per feature
5. Compute union of all selected proteins across all 5120 features
6. Expected: ~5K-20K unique proteins (heavy overlap across features)

**Files to create:**
- `proteinlens/analysis/feature_pipeline/selection.py`

### Stage 3: Per-Residue Collection

For each selected protein (~5K-20K):
1. Embed with ESM2 (single sequence -> per-residue embeddings)
2. Run full SAE encode -> (seq_len, 5120) activation matrix
3. Store compressed: `feature_data/residue_activations/{accession}.npz`
4. Download AlphaFold PDB: `feature_data/pdb_cache/{accession}.pdb`

Resumable: skip proteins that already have both `.npz` and `.pdb` files.

**Files to create:**
- `proteinlens/analysis/feature_pipeline/collection.py`

### Stage 4: Assembly

For each feature (0..5119):
1. Look up its top-20 and bin selections from Stage 2
2. For each protein, extract that feature's per-residue activations from the `.npz` file
3. Assemble JSON structure with all fields
4. Write `feature_data/features/NNNN.json`

Also write:
- `feature_data/sequences.json` -- all stored sequences
- `feature_data/dataset_stats.json` -- summary statistics

**Files to create:**
- `proteinlens/analysis/feature_pipeline/assembly.py`

### CLI Entry Point

- `scripts/run_feature_pipeline.py` -- tap-based CLI that orchestrates all stages
  - `--sae-dir` (default: `trained_models/fiery-sweep`)
  - `--output-dir` (default: `feature_data/`)
  - `--organism-taxid` (default: 9606 for human)
  - `--stage` (optional: run only a specific stage)
  - `--max-proteins` (optional: cap for testing)

---

## Extensibility

Adding new computed properties (e.g., geometric features) per feature:
1. Read `features/NNNN.json`
2. Access the stored sequences and PDB files (no recomputation needed)
3. Compute new property
4. Add new key to the JSON dict
5. Write back

---

## Implementation Checklist

Each item is independently verifiable.

### Stage 0: Data Acquisition & Clustering
- [ ] **0.1** Create `PipelineConfig` dataclass in `config.py`
- [ ] **0.2** Create `data_acquisition.py`: resumable SwissProt FASTA download
- [ ] **0.3** Create `clustering.py`: MMseqs2 wrapper + cluster map parsing
- [ ] **0.4** Test: `tests/test_feature_pipeline/test_data_acquisition.py` -- real UniProt API, small query
- [ ] **0.5** Test: `tests/test_feature_pipeline/test_clustering.py` -- real sequences + MMseqs2

### Stage 1: Survey Pass
- [ ] **1.1** Create `survey.py`: ESM2->SAE streaming with top-20 heaps + memmap
- [ ] **1.2** Create `checkpoint.py`: `PipelineState` class for resumability
- [ ] **1.3** Test: `tests/test_feature_pipeline/test_survey.py` -- real SAE + ESM2 on small set

### Stage 2: Selection
- [ ] **2.1** Create `selection.py`: normalized bin assignment + protein selection
- [ ] **2.2** Test: `tests/test_feature_pipeline/test_selection.py` -- real survey output

### Stage 3: Per-Residue Collection
- [ ] **3.1** Create `collection.py`: per-residue activations as `.npz`
- [ ] **3.2** Add PDB fetching via real AlphaFold API
- [ ] **3.3** Test: `tests/test_feature_pipeline/test_collection.py` -- real models + API

### Stage 4: Assembly
- [ ] **4.1** Create `assembly.py`: per-feature JSON assembly
- [ ] **4.2** Write `sequences.json` and `dataset_stats.json`
- [ ] **4.3** Test: `tests/test_feature_pipeline/test_assembly.py` -- real pipeline output

### CLI & Integration
- [ ] **5.1** Create `scripts/run_feature_pipeline.py` CLI
- [ ] **5.2** Integration test: full end-to-end with ~50 real proteins
- [ ] **5.3** Verify extensibility: add/remove fields without breaking pipeline
### Verify end to end on a small sample maybe 50 proteins locally. 
---

## Test Strategy

All tests use **real APIs, real models, and real data** (small subsets for speed). No mocks.

| Test | Type | What it verifies | Requirements |
|------|------|-----------------|-------------|
| test_data_acquisition | Unit | FASTA download, resumability | Network |
| test_clustering | Unit | MMseqs2 integration, cluster map | MMseqs2 + network |
| test_survey | Unit | Top-20 correctness, max tracking | ESM2-8M + SAE + network |
| test_selection | Unit | Normalized bin assignment | Real survey output |
| test_collection | Unit | Per-residue storage, PDB caching | ESM2-8M + SAE + AlphaFold API |
| test_assembly | Unit | JSON schema, data integrity | Real collection output |
| test_integration | Integration | Full pipeline end-to-end | All of the above |

---

# InterPro Annotation Enrichment for Feature Pipeline

## Context

For each SAE feature, we want to find InterPro annotation codes that predict feature activation. This adds interpretability — if a feature's activation is well-predicted by "Pfam PF00155 (Aminotransferase)", that tells us what the feature has learned.

We compute both **protein-level F1** (does having an annotation predict the protein activates?) and **AA-level F1** (does a residue being inside a domain predict high activation at that position?). For each, we sweep activation thresholds and report the optimal split — including the threshold value itself so the visualizer can show e.g. "Pfam PF00155 separates high (>0.8) from low (<0.2) activation with F1=0.87".

Inspired by InterPLM (Simon & Zou, 2024) but with stratified sampling across activation levels rather than their exhaustive all-protein approach.

---

## Implementation Checklist

Each item is a self-contained objective. A reviewer can verify each one independently.

### Phase 1: Config & Plumbing

- [ ] **1.1** Add InterPro config fields to `PipelineConfig` in `config.py`
  - `interpro_n_bins: int = 11` (bins: 0.0, 0.0-0.1, 0.1-0.2, ..., 0.9-1.0 — the "0.0" bin contains truly inactive proteins with activation == 0)
  - `interpro_n_per_bin: int = 50`
  - `interpro_api_rate_limit: float = 5.0`
  - `interpro_f1_threshold_steps: int = 50`
  - `interpro_top_annotations: int = 5`
  - `interpro_min_proteins: int = 3`
  - **Verify**: `PipelineConfig(sae_dir="x", output_dir="y")` still instantiates without errors. New fields have sensible defaults.

- [ ] **1.2** Add derived path properties to `PipelineConfig`
  - `interpro_selection_path` → `output_dir / "interpro_selection.json"`
  - `interpro_cache_dir` → `output_dir / "interpro_cache"` (mkdir in property)
  - `interpro_residue_activations_dir` → `output_dir / "interpro_residue_activations"` (mkdir in property)
  - `interpro_enrichment_dir` → `output_dir / "interpro_enrichment"` (mkdir in property)
  - **Verify**: Access each property, confirm directories are created.

- [ ] **1.3** Register 3 new stages in `scripts/run_feature_pipeline.py`
  - Add `_run_stage_interpro_selection`, `_run_stage_interpro_fetch`, `_run_stage_interpro_enrichment`
  - Append to `STAGES` list: `("interpro_selection", ...)`, `("interpro_fetch", ...)`, `("interpro_enrichment", ...)`
  - Follow exact pattern of existing stage runners (check `is_stage_complete`, run, `mark_stage_complete`)
  - **Verify**: `python scripts/run_feature_pipeline.py --help` shows the new stage names. Running `--stage interpro_selection` with no prior data gives a clear `FileNotFoundError` (not a crash).

### Phase 2: InterPro Selection (Stage 5a)

- [ ] **2.1** Create `proteinlens/analysis/feature_pipeline/interpro_selection.py`
  - Public function: `run_interpro_selection(config: PipelineConfig) -> Dict`
  - Load `protein_feature_maxes.npy` memmap, `feature_max_activations.npy`, `pipeline_state.json` (for accession index)
  - For each feature:
    - Compute 11 bins: a "0.0" bin for truly inactive proteins (activation == 0), then 10 normalized bins `[0.0-0.1, 0.1-0.2, ..., 0.9-1.0]` of feature max
    - Use same bin logic as `selection.py` lines 120-163: mask + `np.argpartition` for top-N within bin
    - The "0.0" bin: select up to `n_per_bin` proteins with activation == 0 (use `np.random.default_rng(seed=42)` for reproducibility)
    - The lowest non-zero bin (0.0-0.1): `col > 0 & col <= 0.1 * feat_max` — excludes activation == 0
    - All bins are treated uniformly — there is no separate "negatives" group. The threshold sweep in Stage 5c determines what counts as positive/negative at F1 computation time.
  - Write `interpro_selection.json` (structure below)
  - **Verify**: Run on real pipeline output. Check JSON has expected structure. Check that proteins in bin "0.5-0.6" actually have normalized activations in that range. Check "0.0" bin proteins all have activation == 0.

- [ ] **2.2** Add per-residue activation collection to Stage 5a
  - After writing selection JSON, compute union of all unique accessions
  - For each, check if `.npz` exists in EITHER `config.residue_activations_dir` or `config.interpro_residue_activations_dir` — skip if so
  - For remaining proteins: load ESM + SAE, call `_compute_residue_activations` (imported from `collection.py`), save to `interpro_residue_activations/{accession}.npz`
  - Print progress: "Computing per-residue activations for N new proteins (M already cached)"
  - **Verify**: Run stage. Check that .npz files appear in `interpro_residue_activations/`. Check that proteins already in `residue_activations/` were NOT recomputed.

- [ ] **2.3** Write unit test `tests/test_interpro_selection.py`
  - Test bin assignment: create a fake `protein_feature_maxes` array with known values, verify proteins land in correct bins
  - Test "0.0" bin: verify all proteins in this bin have activation == 0, and that sampling is deterministic (same seed → same proteins)
  - Test edge cases: feature with max == 0 (should produce all empty bins except possibly the "0.0" bin), feature where one bin has fewer than `n_per_bin` proteins
  - **Verify**: `pytest tests/test_interpro_selection.py -v` passes.

### Phase 3: InterPro API Client (Stage 5b)

- [ ] **3.1** Create `proteinlens/analysis/feature_pipeline/interpro_api.py`
  - Dataclass `InterProDomain`: `interpro_accession`, `interpro_name`, `type` (Family/Domain/etc), `member_db`, `member_accession`, `start` (int, 1-based), `end` (int, 1-based inclusive)
  - `RateLimiter` class: token-bucket, configurable rate
  - `fetch_interpro_annotations(accession, cache_dir, session, rate_limiter) -> List[InterProDomain]`
    - Check `cache_dir/{accession}.json` first — return cached if exists
    - GET `https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{accession}`
    - Parse response: iterate `results`, extract metadata (accession, name, type) + `entry_protein_locations` for domain boundaries + `member_databases` for cross-references
    - One InterPro entry may have multiple fragments (start/end pairs) — create one `InterProDomain` per fragment per member database entry
    - Cache parsed result to `interpro_cache/{accession}.json`
    - On 404: cache `{"accession": "...", "domains": []}` — do NOT re-fetch
    - On transient error (5xx, timeout): exponential backoff, 3 retries
  - **Verify**: Manually call `fetch_interpro_annotations("P12345", ...)` and inspect the cached JSON. Check that domain start/end positions are present and 1-based.

- [ ] **3.2** Public function `run_interpro_fetch(config: PipelineConfig) -> None`
  - Load `interpro_selection.json` → `all_selected_accessions`
  - Filter to accessions not already cached in `interpro_cache_dir`
  - Fetch remaining with rate limiting, progress bar
  - Print summary: "Fetched annotations for N proteins (M already cached, K not in InterPro)"
  - **Verify**: Run stage. Check `interpro_cache/` has one JSON per protein. Re-run — should skip all (0 fetched).

- [ ] **3.3** Write unit test `tests/test_interpro_api.py`
  - Test API response parsing: provide a sample InterPro JSON response (hardcoded), verify correct `InterProDomain` objects are extracted
  - Test caching: verify second call reads from cache, not API
  - Test 404 handling: verify empty domains cached
  - Test rate limiter: verify calls are spaced appropriately
  - Use real API call for one protein (P12345) as integration test — mark with `@pytest.mark.integration`
  - **Verify**: `pytest tests/test_interpro_api.py -v` passes (unit tests). `pytest tests/test_interpro_api.py -v -m integration` for the real API test.

### Phase 4: F1 Enrichment (Stage 5c)

- [ ] **4.1** Create `proteinlens/analysis/feature_pipeline/interpro_enrichment.py`
  - Utility: `load_residue_activations(accession, config) -> Optional[np.ndarray]` — checks both `residue_activations_dir` and `interpro_residue_activations_dir`

- [ ] **4.2** Implement protein-level F1 computation
  - For each feature:
    1. Load its InterPro selection (all 11 bins: "0.0" inactive bin + 10 activation bins) → flat list of accessions with their activation values
    2. Get each protein's max activation from the memmap
    3. Load each protein's InterPro annotations from cache
    4. Collect all unique annotation codes across these proteins
    5. Skip annotations appearing in fewer than `config.interpro_min_proteins` proteins
    6. For each annotation code:
       - `y_true` = 1 if protein has this annotation, 0 otherwise (binary vector, length = n_proteins)
       - Sweep `config.interpro_f1_threshold_steps` evenly-spaced thresholds from 0 to feature_max
       - At each threshold `t`: `y_pred` = 1 if protein activation > t, 0 otherwise
       - Compute precision, recall, F1 at each threshold
       - Record the threshold giving best F1, AND the precision/recall at that threshold
    7. Rank by best F1, keep top `config.interpro_top_annotations`
    8. If multiple annotations have F1 within 0.05 of the best, include all of them
  - **Verify**: Check output JSON for a few features. Verify that reported threshold values are within [0, feature_max]. Verify precision * recall * 2 / (precision + recall) ≈ reported F1.

- [ ] **4.3** Implement AA-level F1 computation
  - For each feature, for each top annotation from protein-level:
    1. Collect all proteins that have this annotation AND have per-residue activations available
    2. For each protein: load per-residue activations, extract column for this feature (1D array of length seq_len)
    3. Build residue-level labels: `y_true[i]` = 1 if residue `i` falls within any domain boundary for this annotation (convert 1-based InterPro positions to 0-based: `start-1` to `end-1` inclusive)
    4. Concatenate all residues across all proteins into `(all_activations, all_labels)`
    5. Sweep thresholds on activation values (use percentiles of non-zero activations for better coverage)
    6. Compute F1 at each threshold, record best
  - **Verify**: For a feature with known domain associations, check that AA-level F1 is reasonable. Verify residue counts make sense (n_in_domain < n_total_residues).

- [ ] **4.4** Write output JSON per feature
  - Path: `interpro_enrichment/{feat_idx:04d}.json`
  - Structure (see Output JSON Schema below)
  - All threshold values and scores must be present for downstream visualization
  - **Verify**: Load a few JSONs, check all required fields are present and types are correct.

- [ ] **4.5** Write summary JSON
  - Path: `interpro_enrichment/summary.json`
  - Structure: `{"features": {"0": {"top_annotation": "IPR000001", "top_f1": 0.87, ...}, ...}}`
  - Quick lookup for the visualizer dashboard
  - **Verify**: Check that every feature with enrichment data has an entry.

- [ ] **4.6** Public function `run_interpro_enrichment(config: PipelineConfig) -> None`
  - Orchestrates 4.1–4.5
  - Skip features with `global_max == 0`
  - Skip features with fewer than `config.interpro_min_proteins` activated proteins
  - Print progress and summary stats
  - **Verify**: Run stage end-to-end. Check `interpro_enrichment/` has JSON files.

- [ ] **4.7** Write unit tests `tests/test_interpro_enrichment.py`
  - **Test F1 computation with known data**: Create synthetic proteins + annotations where the answer is known:
    - 10 proteins with annotation X, activations [5, 5, 5, 5, 5, 1, 1, 1, 1, 1]
    - 10 proteins without annotation X, activations [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    - Best threshold should be ~0.5, F1 should be 1.0 for separating annotated vs not
  - **Test AA-level F1 with known data**: Create a protein with known domain at positions 10-20, high activation at those positions, low elsewhere → F1 should be near 1.0
  - **Test threshold reporting**: Verify the reported threshold is actually the one that gives the reported F1 (recompute manually)
  - **Test edge cases**: Feature with no annotations, annotation on only 1 protein (should be skipped), all proteins have same activation
  - **Verify**: `pytest tests/test_interpro_enrichment.py -v` passes.

### Phase 5: Integration & Verification

- [ ] **5.1** End-to-end test with small dataset
  - Run full pipeline with `--max_proteins 100`
  - Verify all 3 new stages complete
  - Spot-check 5 feature JSONs for correctness
  - **Verify**: No errors in stdout. Output files exist and parse correctly.

- [ ] **5.2** Verify reproducibility
  - Run Stage 5a twice with same input — `interpro_selection.json` should be identical (deterministic negative sampling)
  - **Verify**: `diff` of the two JSON files shows no differences.

---

## Output JSON Schema

### `interpro_selection.json`
```json
{
  "per_feature": {
    "0": {
      "bins": {
        "0.0": ["C33333", "D44444"],
        "0.0-0.1": ["P12345", "Q67890"],
        "0.1-0.2": ["A11111"],
        "0.9-1.0": ["B22222"]
      }
    }
  },
  "all_selected_accessions": ["A11111", "B22222", "C33333", "D44444"]
}
```

### `interpro_cache/{accession}.json`
```json
{
  "accession": "P12345",
  "domains": [
    {
      "interpro_accession": "IPR000796",
      "interpro_name": "Aspartate aminotransferase",
      "type": "Family",
      "member_db": "pfam",
      "member_accession": "PF00155",
      "start": 45,
      "end": 210
    }
  ]
}
```

### `interpro_enrichment/{feat_idx:04d}.json`

This is the primary output consumed by the visualizer. Every field needed for rendering is included.

```json
{
  "feature_id": 42,
  "feature_max_activation": 3.7,
  "n_proteins_evaluated": 350,
  "n_proteins_with_annotations": 280,
  "n_unique_annotations_tested": 120,

  "protein_level": [
    {
      "annotation_code": "IPR000796",
      "annotation_name": "Aspartate aminotransferase",
      "annotation_type": "Family",
      "member_db": "pfam",
      "member_accession": "PF00155",

      "best_f1": 0.87,
      "best_threshold": 2.96,
      "best_threshold_normalized": 0.80,
      "precision_at_best": 0.85,
      "recall_at_best": 0.89,

      "n_proteins_with_annotation": 45,
      "n_proteins_without_annotation": 305,
      "n_true_positives": 38,
      "n_false_positives": 7,
      "n_false_negatives": 5,

      "interpretation": "Proteins with activation > 2.96 (80% of max) are predicted by annotation IPR000796 with F1=0.87"
    }
  ],

  "residue_level": [
    {
      "annotation_code": "IPR000796",
      "annotation_name": "Aspartate aminotransferase",
      "member_db": "pfam",
      "member_accession": "PF00155",

      "best_f1": 0.72,
      "best_threshold": 0.95,
      "best_threshold_normalized": 0.26,
      "precision_at_best": 0.70,
      "recall_at_best": 0.74,

      "n_proteins_used": 42,
      "n_total_residues": 15000,
      "n_residues_in_domain": 3200,
      "n_true_positives": 2368,
      "n_false_positives": 1015,
      "n_false_negatives": 832,

      "interpretation": "Residues with activation > 0.95 (26% of max) overlap with IPR000796 domains with F1=0.72"
    }
  ]
}
```

Key design points for the visualizer:
- `best_threshold` = absolute activation value used for the split
- `best_threshold_normalized` = threshold / feature_max (0-1 scale, for display)
- `interpretation` = human-readable sentence for tooltip/label rendering
- Full confusion matrix counts (TP/FP/FN) so the visualizer can show detailed breakdowns
- Both protein-level and residue-level results in same file, keyed separately

### `interpro_enrichment/summary.json`
```json
{
  "n_features_analyzed": 4800,
  "n_features_skipped": 320,
  "features": {
    "42": {
      "top_protein_annotation": "IPR000796",
      "top_protein_annotation_name": "Aspartate aminotransferase",
      "top_protein_f1": 0.87,
      "top_residue_annotation": "IPR000796",
      "top_residue_f1": 0.72
    }
  }
}
```

---

## Files Summary

| Action | File | What |
|--------|------|------|
| Modify | `proteinlens/analysis/feature_pipeline/config.py` | Add InterPro config fields + path properties |
| Modify | `scripts/run_feature_pipeline.py` | Register 3 new stages |
| Create | `proteinlens/analysis/feature_pipeline/interpro_selection.py` | Stage 5a: sampling + expanded collection |
| Create | `proteinlens/analysis/feature_pipeline/interpro_api.py` | Stage 5b: InterPro API client with caching |
| Create | `proteinlens/analysis/feature_pipeline/interpro_enrichment.py` | Stage 5c: F1 computation + output |
| Create | `tests/test_interpro_selection.py` | Unit tests for bin sampling |
| Create | `tests/test_interpro_api.py` | Unit tests for API parsing + caching |
| Create | `tests/test_interpro_enrichment.py` | Unit tests for F1 computation |

## Key Code to Reuse

| What | Where | How |
|------|-------|-----|
| Bin sampling logic | `selection.py:120-163` | Replicate with 10 bins in `interpro_selection.py` |
| `_compute_residue_activations()` | `collection.py:220-252` | Import directly |
| Remaining-protein check pattern | `collection.py:255-280` | Replicate for dual-directory check |
| LRU .npz loading | `assembly.py` | Replicate pattern in enrichment |
| Stage runner pattern | `run_feature_pipeline.py:53-120` | Follow exactly for 3 new stages |

---

# Geometric Descriptor Enrichment for Feature Pipeline

## Context

The feature pipeline computes InterPro annotation F1 scores per SAE node (Stages 5a-5c). We now add geometric descriptor enrichment — answering "can local 3D geometry predict where/whether this SAE node fires?"

Two proven scripts already do this:
- `protein_results/build_residue_motifs.py` — per-node residue-level GBM classification (F1, AUC, Spearman, fragment superposition)
- `protein_results/build_activation_multiset.py` — per-node protein-level LassoCV regression (R²_cv, Pearson r, monomial)

We extract core functions into a proper `proteinlens/analysis/geometry/` package, then add three pipeline stages (6a, 6b, 6c) producing per-feature JSON with precomputed data for the visualizer frontend.

**Key decisions:**
- Precompute & store all plotting arrays (geom_prob, feature traces, Cα backbone coords, concordance labels) — no model inference at serving time
- Extract geometry code into `proteinlens/analysis/geometry/` (not sys.path hacks from `protein_results/`)
- Protein-level uses R²_cv (regression), not F1

**3D plot feasibility:** The pipeline already has AlphaFold PDB files in `pdb_cache/`. `ca_backbone()` extracts (N, 3) Cα coordinates — sufficient for three.js tube/ribbon rendering. Storing coords for 5 proteins per node is ~3-5KB per protein in JSON. Total is well under 100MB.

---

## Implementation Checklist

Each item is independently verifiable.

### Phase 1: Geometry Extraction Module

- [ ] **1.1** Create `proteinlens/analysis/geometry/__init__.py`
  - Minimal docstring, no logic.
  - **Verify**: `from proteinlens.analysis.geometry import ...` does not error.

- [ ] **1.2** Create `proteinlens/analysis/geometry/protein_features.py`
  - Copy `GEOM_FEATURE_NAMES` (56 names) from `protein_results/build_activation_dataset.py:133-195`
  - Extract `compute_protein_geometry(pdb_text: str) -> dict[str, float] | None` adapted from `build_activation_dataset.py:451-500`
  - Must NOT import matplotlib. Import geometric primitives from `protein_results/geometry/compute_geometric_features.py` (these are already clean: `writhe`, `ca_curvature_profile`, `ca_torsion_profile`, etc.)
  - **Verify**: Call `compute_protein_geometry(open("feature_data/pdb_cache/P12345.pdb").read())` — returns dict with 56 keys, all float values, no matplotlib import triggered.

- [ ] **1.3** Create `proteinlens/analysis/geometry/residue_features.py`
  - Extract `ca_backbone(pdb_text, chain_id=None) -> np.ndarray` from `protein_results/pdb_plotter.py:17-55` — pure BioPython + numpy, no matplotlib
  - Extract `detect_alpha_helices_from_ca(ca) -> list[tuple[int,int]]` from `pdb_plotter.py`
  - Extract `compute_residue_profiles(ca, helices) -> dict` from `build_residue_motifs.py:~270-400`
  - Extract `extract_local_feature_vector(profiles, ca, pos, half_w, sequence) -> np.ndarray | None` from `build_residue_motifs.py:~400-500`
  - Copy `LOCAL_GEOM_NAMES` (44 names), `ACTIVE_GEOM_NAMES`, `FEATURE_GROUPS` from `build_residue_motifs.py:131-206`
  - Copy `select_features(feat_vec) -> np.ndarray` and `set_active_feature_set(choice)` from `build_residue_motifs.py`
  - **Verify**: Load a real PDB from `pdb_cache/`, call `ca_backbone()` -> get (N, 3) array. Call `compute_residue_profiles()` + `extract_local_feature_vector()` at position 15 -> get 44-dim vector. No matplotlib in the import chain.

- [ ] **1.4** Create `proteinlens/analysis/geometry/classifiers.py`
  - Extract `kabsch_align(frag, ref) -> np.ndarray` and `compute_rmsd(a, b) -> float` from `protein_results/kabsch_top_alignment.py`
  - Extract `superpose_fragments(activated, top_k=100) -> dict` from `build_residue_motifs.py:738-801`. Returns `{mean_structure, mean_rmsd, std_rmsd, per_pos_std, n_fragments}`
  - Extract `collect_node_fragments(protein_data, node_idx, half_w=10, act_quantile=0.80, bg_ratio=3) -> dict` from `build_residue_motifs.py:628-734`. Returns `{activated, background, threshold, n_total_active}`
  - Extract `train_motif_classifier(activated, background, feature_names, max_depth=4, cv_folds=5) -> dict` from `build_residue_motifs.py:806-1009`. **Adaptation**: accept `feature_names` param instead of global `ACTIVE_GEOM_NAMES`. Returns `{tree, decision_tree, rules, f1_cv, auc_cv, gbm_auc_cv, rf_auc_cv, lpo_auc, feature_importances, optimal_threshold}`
  - Extract `compute_concordance_metrics(protein_data, node_idx, tree, threshold, geom_threshold, half_w) -> dict` from `build_residue_motifs.py:1856-1990`. Returns `{spearman_r, residue_auroc, avg_precision, cosine_sim, f1, precision, recall, n_residues, n_proteins}`
  - Create `fit_lasso_single_node(X, y, geom_names, cv_folds=5) -> dict | None` — single-node version of `build_activation_multiset.py:88-213` (no loop, no top-N filtering). Returns `{r2, r2_adj, r2_cv, pearson_r, n_samples, n_nonzero, alpha_chosen, monomial, top_features, weights_raw, intercept_raw}`
  - Extract `format_monomial(weights_raw, intercept_raw, geom_names) -> str` from `build_activation_multiset.py:222-264`
  - **Verify**: Call `fit_lasso_single_node` with a synthetic (100, 5) matrix and (100,) target -> returns dict with all expected keys and r2_cv > 0. Call `train_motif_classifier` with synthetic activated/background lists -> returns dict with rules string and f1_cv float.

- [ ] **1.5** Write unit test `tests/test_analysis/test_geometry_module.py`
  - Test `compute_protein_geometry` with a real small PDB from `pdb_cache/`: returns dict with 56 keys, values are finite floats
  - Test `ca_backbone`: real PDB -> (N, 3) array with N > 10
  - Test `extract_local_feature_vector`: returns 44-dim vector, all finite
  - Test `fit_lasso_single_node` with synthetic data: known linear relationship (y = 2*x1 + noise) should give r2_cv > 0.3
  - Test `train_motif_classifier` with synthetic data: 50 activated + 150 background with separable features should give f1_cv > 0.5
  - Test `superpose_fragments` with 10 identical fragments: mean_rmsd should be ~0
  - Test `format_monomial`: known weights -> expected string
  - **Verify**: `conda run -n interplm pytest tests/test_analysis/test_geometry_module.py -v` passes.

### Phase 2: Pipeline Config & Stage 6a

- [ ] **2.1** Add geometry config fields to `PipelineConfig` in `proteinlens/analysis/feature_pipeline/config.py`
  - `geometry_min_active_proteins: int = 300` — min activated proteins for Lasso
  - `geometry_min_activated_positions: int = 200` — min activated residue positions for GBM
  - `geometry_fragment_half_w: int = 10` — half-window for fragments (total = 21)
  - `geometry_act_quantile: float = 0.80` — activation quantile threshold
  - `geometry_frag_top_k: int = 100` — max fragments for superposition
  - `geometry_bg_ratio: int = 3` — background-to-activated ratio
  - `geometry_lasso_cv_folds: int = 5`
  - `geometry_classifier_cv_folds: int = 5`
  - `geometry_top_proteins_for_plots: int = 5` — per node, precompute plot data for top N
  - **Verify**: `PipelineConfig(sae_dir="x", output_dir="y")` instantiates without errors.

- [ ] **2.2** Add geometry path properties to `PipelineConfig`
  - `geometry_protein_features_path` -> `output_dir / "geometry_protein_features.npz"`
  - `geometry_residue_profiles_dir` -> `output_dir / "geometry_residue_profiles"` (mkdir)
  - `geometry_enrichment_dir` -> `output_dir / "geometry_enrichment"` (mkdir)
  - **Verify**: Access each property, confirm directories are created.

- [ ] **2.3** Create `proteinlens/analysis/feature_pipeline/geometry_features.py` (Stage 6a)
  - Public function: `run_geometry_features(config: PipelineConfig) -> None`
  - Iterate all `.pdb` files in `config.pdb_cache_dir`
  - For each protein:
    - Call `compute_protein_geometry(pdb_text)` -> 56-dim vector; skip on failure
    - Call `ca_backbone(pdb_text)` + `detect_alpha_helices_from_ca(ca)` + `compute_residue_profiles(ca, helices)` -> save to `geometry_residue_profiles/{acc}.npz` (keys: `ca`, each profile field, `sequence` from `sequences.json`)
  - Save `geometry_protein_features.npz` with keys: `accessions` (str[]), `geometry_matrix` (N, 56), `feature_names` (str[])
  - Resumable: skip accessions that already have `.npz` in `geometry_residue_profiles_dir`; rebuild full protein features matrix at end from all profiles
  - Print summary: "Computed geometry for N proteins (M skipped/failed, K total PDBs)"
  - **Verify**: Run on existing `feature_data/`. Check `.npz` files appear in `geometry_residue_profiles/`. Check `geometry_protein_features.npz` loads with correct shapes. Re-run — should skip all.

- [ ] **2.4** Register Stage 6a in `scripts/run_feature_pipeline.py`
  - Add `_run_stage_geometry_features` following exact pattern of existing stage runners (check `is_stage_complete`, run, `mark_stage_complete`)
  - Add `("geometry_features", _run_stage_geometry_features)` to `STAGES` list
  - **Verify**: `python scripts/run_feature_pipeline.py --help` shows the new stage name. `--stage geometry_features` works.

- [ ] **2.5** Write unit test `tests/test_feature_pipeline/test_geometry_features.py`
  - Test with 3-5 real PDBs from `pdb_cache/`: verify output `.npz` files have expected keys/shapes
  - Test resumability: run twice, second run should skip all proteins
  - Test failure handling: create a corrupt PDB file, verify it's skipped gracefully
  - **Verify**: `conda run -n interplm pytest tests/test_feature_pipeline/test_geometry_features.py -v` passes.

### Phase 3: Stage 6b (Protein-Level Lasso)

- [ ] **3.1** Create `proteinlens/analysis/feature_pipeline/geometry_protein_enrichment.py`
  - Public function: `run_geometry_protein_enrichment(config: PipelineConfig) -> None`
  - Load `geometry_protein_features.npz` (accessions + 56-dim matrix)
  - Load `protein_feature_maxes.npy` memmap + `pipeline_state.json` (for accession-to-index mapping)
  - Build index: map accessions in geometry matrix to their rows in the memmap
  - For each SAE node (0..num_features-1):
    - Skip if `feature_max == 0`
    - Get max activation values for proteins that have geometry
    - Filter to proteins with activation > 0 (active)
    - Skip if `n_active < config.geometry_min_active_proteins`
    - Call `fit_lasso_single_node(X_geom[active], y_act[active], GEOM_FEATURE_NAMES)`
    - If result is None (no signal), skip
    - Write JSON to `geometry_enrichment/{feat:04d}.json` with `geometric_protein_level` section
  - Write protein-level entries to `geometry_enrichment/summary.json`
  - Print progress every 500 nodes and final summary
  - **Verify**: Run on existing `feature_data/`. Check JSON files appear. Load 5 JSONs and verify `geometric_protein_level` has all fields: `r2_cv`, `pearson_r`, `monomial`, `n_samples`, `n_nonzero`, `top_features`. Verify `r2_cv` values are in [-1, 1] range.

- [ ] **3.2** Register Stage 6b in `scripts/run_feature_pipeline.py`
  - Add `_run_stage_geometry_protein_enrichment` + append to `STAGES`
  - **Verify**: `--stage geometry_protein_enrichment` runs after 6a.

- [ ] **3.3** Write unit test `tests/test_feature_pipeline/test_geometry_protein_enrichment.py`
  - Create synthetic `geometry_protein_features.npz` (100 proteins, 10 features) and synthetic `protein_feature_maxes.npy` (100 proteins, 20 nodes) with known linear relationships for 2-3 nodes
  - Verify: nodes with strong signal have `r2_cv > 0`, nodes with random data are skipped or have `r2_cv ~ 0`
  - Verify: `monomial` string contains expected feature names
  - Verify: nodes with fewer than `min_active_proteins` active are skipped
  - Verify: output JSON schema matches expected structure
  - **Verify**: `conda run -n interplm pytest tests/test_feature_pipeline/test_geometry_protein_enrichment.py -v` passes.

### Phase 4: Stage 6c (Residue-Level GBM + Plot Data)

- [ ] **4.1** Create `proteinlens/analysis/feature_pipeline/geometry_residue_enrichment.py`
  - Public function: `run_geometry_residue_enrichment(config: PipelineConfig) -> None`
  - Load `feature_max_activations.npy`, `pipeline_state.json`, `protein_feature_maxes.npy` memmap
  - For each SAE node:
    - Skip if `feature_max == 0`
    - Identify proteins where node fires (activation > 0 in memmap)
    - Filter to proteins that have both per-residue activations (`.npz` in `residue_activations/` or `interpro_residue_activations/`) AND geometry residue profiles (`.npz` in `geometry_residue_profiles/`)
    - Skip if fewer than `config.geometry_min_activated_positions` total activated residue positions
    - Build `protein_data` list (same format as `build_residue_motifs.py` expects): `{act_matrix, ca, profiles, n_residues, accession, sequence}`
    - Call `collect_node_fragments(protein_data, node_idx, ...)`
    - Call `superpose_fragments(activated, top_k=config.geometry_frag_top_k)`
    - Call `train_motif_classifier(activated, background, ACTIVE_GEOM_NAMES, ...)`
    - Call `compute_concordance_metrics(protein_data, node_idx, tree, threshold, geom_threshold, half_w)`
    - **Precompute plot data** for top `config.geometry_top_proteins_for_plots` proteins (by max activation):
      - `ca_backbone`: `[[x,y,z], ...]` from `geometry_residue_profiles/{acc}.npz`
      - `sae_activation_profile`: per-residue SAE activations for this node
      - `geom_prob_profile`: run GBM `predict_proba` at each position (using stored profiles)
      - `activated_positions`: list of `{position, activation}` for positions above threshold
      - `top_feature_traces`: top-2 features by importance, per-residue values (null outside window range)
      - `concordance_labels`: per-residue: `"agree"` (both active), `"fp"` (geom only), `"fn"` (sae only), `"tn"` (neither)
    - Save motif template as PDB text in `motif_superposition.mean_structure_pdb`
    - Merge into existing `geometry_enrichment/{feat:04d}.json` (add `geometric_residue_level` + `plot_data` alongside `geometric_protein_level` from Stage 6b)
  - Update `geometry_enrichment/summary.json` with residue-level fields
  - **Verify**: Run on existing `feature_data/`. Load 5 JSONs, verify `geometric_residue_level` has: `tree_f1_cv`, `gbm_auc_cv`, `rules`, `concordance`, `motif_superposition`. Verify `plot_data.top_proteins` has <= 5 entries, each with `ca_backbone` (list of [x,y,z]), `geom_prob_profile` (list of floats, same length as sequence), `concordance_labels` (list of strings).

- [ ] **4.2** Register Stage 6c in `scripts/run_feature_pipeline.py`
  - Add `_run_stage_geometry_residue_enrichment` + append to `STAGES`
  - **Verify**: `--stage geometry_residue_enrichment` runs after 6b.

- [ ] **4.3** Write unit test `tests/test_feature_pipeline/test_geometry_residue_enrichment.py`
  - **Test fragment collection**: Create synthetic `protein_data` with known activations, verify `collect_node_fragments` returns correct activated/background counts
  - **Test classifier with separable data**: 50 activated fragments with high curvature + 150 background with low curvature -> `f1_cv > 0.5`, `gbm_auc_cv > 0.6`
  - **Test concordance with known agreement**: Synthetic data where GBM perfectly predicts SAE -> `spearman_r > 0.8`
  - **Test plot data structure**: Verify `ca_backbone` is list of 3-element lists, `geom_prob_profile` length == sequence length, `concordance_labels` only contains valid values (`"agree"`, `"fp"`, `"fn"`, `"tn"`)
  - **Test motif superposition**: 10 identical fragments -> `mean_rmsd < 0.1`
  - **Test edge case**: Node with too few activated positions -> skipped gracefully
  - **Verify**: `conda run -n interplm pytest tests/test_feature_pipeline/test_geometry_residue_enrichment.py -v` passes.

### Phase 5: Integration & Verification

- [ ] **5.1** End-to-end test with existing pipeline output
  - Run all 3 stages sequentially on existing `feature_data/`
  - Verify all stages complete without error
  - Verify `geometry_enrichment/` has JSON files
  - Verify `geometry_enrichment/summary.json` exists and has entries
  - Spot-check 5 feature JSONs for complete structure

- [ ] **5.2** Verify plot data completeness for frontend
  - For 3 features with residue-level results, verify each top protein entry has:
    - `ca_backbone`: N points, each `[x, y, z]` with finite floats
    - `sae_activation_profile`: N values, all >= 0
    - `geom_prob_profile`: N values, all in [0, 1]
    - `activated_positions`: non-empty, positions within [0, N)
    - `top_feature_traces`: 2 keys, each with N values (may contain null at edges)
    - `concordance_labels`: N strings, each one of `["agree", "fp", "fn", "tn"]`
  - Verify a mean_structure_pdb string can be parsed as valid PDB text

- [ ] **5.3** Verify resumability
  - Run Stage 6a twice — second run should report "0 new proteins computed"
  - Run Stage 6b twice — second run should be a no-op (check `is_stage_complete`)

---

## Geometry Enrichment Output JSON Schema

Each `geometry_enrichment/{feat:04d}.json`:

```json
{
  "feature_id": 42,
  "feature_max_activation": 8.5,

  "geometric_protein_level": {
    "r2_cv": 0.45,
    "r2": 0.52,
    "r2_adj": 0.44,
    "pearson_r": 0.67,
    "alpha_chosen": 0.012,
    "monomial": "y_hat = 0.34*hairpin_score - 0.19*avg_curvature + 0.003",
    "n_samples": 1523,
    "n_nonzero": 4,
    "top_features": [
      {"feature": "hairpin_score", "weight": 0.34, "abs_weight": 0.34},
      {"feature": "avg_curvature", "weight": -0.19, "abs_weight": 0.19}
    ]
  },

  "geometric_residue_level": {
    "tree_f1_cv": 0.68,
    "gbm_auc_cv": 0.81,
    "rf_auc_cv": 0.78,
    "lpo_auc": 0.76,
    "rules": "|--- curvature_mean > 0.5000\n|   |--- ...",
    "optimal_threshold": 0.62,
    "activation_threshold": 3.2,
    "n_activated": 450,
    "n_background": 1350,
    "n_unique_proteins": 87,
    "feature_importances": {"curvature_mean": 0.23, "contact_density_8A": 0.15},
    "concordance": {
      "spearman_r": 0.42,
      "residue_auroc": 0.75,
      "avg_precision": 0.38,
      "cosine_sim": 0.55,
      "f1": 0.55,
      "precision": 0.60,
      "recall": 0.51,
      "n_residues": 12500,
      "n_proteins": 87
    },
    "motif_superposition": {
      "mean_rmsd": 2.3,
      "std_rmsd": 0.8,
      "n_fragments": 100,
      "per_position_flexibility": [0.5, 0.4, 0.3, "...21 values..."],
      "mean_structure_pdb": "REMARK  Motif template...\nATOM      1  CA  ALA A   1  ..."
    }
  },

  "plot_data": {
    "top_proteins": [
      {
        "accession": "P12345",
        "sequence": "MKTL...",
        "ca_backbone": [[1.2, 3.4, 5.6], [1.3, 3.5, 5.7], "..."],
        "sae_activation_profile": [0.0, 0.1, 3.2, 0.0, "..."],
        "geom_prob_profile": [0.1, 0.2, 0.8, 0.1, "..."],
        "activated_positions": [
          {"position": 15, "activation": 3.2},
          {"position": 42, "activation": 2.8}
        ],
        "top_feature_traces": {
          "curvature_mean": [null, null, 0.3, 0.4, "...", null],
          "contact_density_8A": [null, null, 2.1, 1.8, "...", null]
        },
        "concordance_labels": ["tn", "tn", "agree", "fn", "tn", "..."]
      }
    ]
  }
}
```

`geometry_enrichment/summary.json`:

```json
{
  "n_features_protein_level": 2048,
  "n_features_residue_level": 1500,
  "n_features_skipped": 3072,
  "n_proteins_with_geometry": 18500,
  "features": {
    "42": {
      "protein_r2_cv": 0.45,
      "protein_pearson_r": 0.67,
      "residue_gbm_auc_cv": 0.81,
      "residue_concordance_spearman": 0.42,
      "motif_rmsd": 2.3
    }
  }
}
```

---

## Geometry Enrichment Files Summary

| Action | File | What |
|--------|------|------|
| Create | `proteinlens/analysis/geometry/__init__.py` | Package init |
| Create | `proteinlens/analysis/geometry/protein_features.py` | 56-dim protein geometry (`compute_protein_geometry`, `GEOM_FEATURE_NAMES`) |
| Create | `proteinlens/analysis/geometry/residue_features.py` | 44-dim residue features (`ca_backbone`, `compute_residue_profiles`, `extract_local_feature_vector`, `LOCAL_GEOM_NAMES`) |
| Create | `proteinlens/analysis/geometry/classifiers.py` | Models (`fit_lasso_single_node`, `train_motif_classifier`, `collect_node_fragments`, `superpose_fragments`, `compute_concordance_metrics`, `kabsch_align`) |
| Modify | `proteinlens/analysis/feature_pipeline/config.py` | Geometry config fields + 3 path properties |
| Create | `proteinlens/analysis/feature_pipeline/geometry_features.py` | Stage 6a: compute geometry for all proteins with PDBs |
| Create | `proteinlens/analysis/feature_pipeline/geometry_protein_enrichment.py` | Stage 6b: LassoCV per node |
| Create | `proteinlens/analysis/feature_pipeline/geometry_residue_enrichment.py` | Stage 6c: GBM + concordance + plot data |
| Modify | `scripts/run_feature_pipeline.py` | Register 3 new stages in STAGES list |
| Create | `tests/test_analysis/test_geometry_module.py` | Unit tests for geometry extraction module |
| Create | `tests/test_feature_pipeline/test_geometry_features.py` | Tests for Stage 6a |
| Create | `tests/test_feature_pipeline/test_geometry_protein_enrichment.py` | Tests for Stage 6b |
| Create | `tests/test_feature_pipeline/test_geometry_residue_enrichment.py` | Tests for Stage 6c |

## Key Source Files to Extract From

| Source | Functions to Extract |
|--------|---------------------|
| `protein_results/build_activation_dataset.py:133` | `GEOM_FEATURE_NAMES` (56 names) |
| `protein_results/build_activation_dataset.py:451` | `compute_geometry` -> `compute_protein_geometry` |
| `protein_results/build_activation_multiset.py:88` | `fit_linear_regressors` -> `fit_lasso_single_node` (single-node, no loop) |
| `protein_results/build_activation_multiset.py:222` | `format_monomial` |
| `protein_results/build_residue_motifs.py:131-206` | `LOCAL_GEOM_NAMES`, `ACTIVE_GEOM_NAMES`, `FEATURE_GROUPS`, `select_features` |
| `protein_results/build_residue_motifs.py:~270-500` | `compute_residue_profiles`, `extract_local_feature_vector` |
| `protein_results/build_residue_motifs.py:628` | `collect_node_fragments` |
| `protein_results/build_residue_motifs.py:738` | `superpose_fragments` |
| `protein_results/build_residue_motifs.py:806` | `train_motif_classifier` (adapt: accept `feature_names` param) |
| `protein_results/build_residue_motifs.py:1856` | `compute_concordance_metrics` |
| `protein_results/pdb_plotter.py:17-55` | `ca_backbone` (strip matplotlib dependency) |
| `protein_results/pdb_plotter.py` | `detect_alpha_helices_from_ca` |
| `protein_results/kabsch_top_alignment.py` | `kabsch_align`, `compute_rmsd` |
| `protein_results/geometry/compute_geometric_features.py` | `ca_curvature_profile`, `ca_torsion_profile`, `local_planarity_profile`, `tangent_vectors`, `writhe`, etc. |

---

# Parallelize Stage 6c (Residue-Level GBM) with multiprocessing

## Context

Stage 6c (`geometry_residue_enrichment.py`) processes ~5000 SAE nodes sequentially,
training a GBM classifier + computing concordance + precomputing plot data per node.
With 75k proteins this takes ~12-24 hours single-threaded. Each node is independent —
same read-only protein data, fixed random seeds, isolated model instances — making this
embarrassingly parallel.

**Goal**: Add `multiprocessing.Pool` parallelism to Stage 6c while guaranteeing
**byte-identical results** to the serial version.

## Parallelization Safety Analysis

**Safe (verified):**
- Per-node JSON writes — each node writes to its own `{ni:04d}.json` file
- Shared data (`all_protein_data`, `act_memmap`, `feature_maxes`) is read-only
- Random seeds are fixed per-node: `default_rng(42)` in fragment collection,
  `random_state=42` in all sklearn models
- Numpy operations are deterministic
- `ACTIVE_FEATURE_MASK` / `ACTIVE_GEOM_NAMES` globals are never mutated during 6c

**Must fix:**
1. **`summary.json`** — currently read-modify-write in the main loop. With parallel
   workers, concurrent writes would cause data loss. Fix: each worker returns its
   summary dict entry; merge all entries into summary.json after all workers finish.
2. **`n_jobs=-1` in RandomForestClassifier** — each worker's RF would spawn
   num_cores threads, causing N × num_cores thread contention. Fix: set `n_jobs=1`
   when running in parallel mode.

## Implementation Checklist

### 1. Extract per-node logic into a standalone function

**File**: `proteinlens/analysis/feature_pipeline/geometry_residue_enrichment.py`

- [ ] **1.1** Create `_process_single_node(args) -> dict | None` that encapsulates
  the body of the current `for ni in range(n_features)` loop (lines ~401-500).
  - Input: a single tuple/dict with `(ni, all_protein_data, act_memmap, feature_maxes,
    enrichment_dir, half_w, config_params)` — all serializable or shared-memory-safe.
  - Output: a dict `{"feature_id": ni, "summary_entry": {...}}` on success, or `None`
    if skipped.
  - Must NOT touch `summary.json` — only write the per-feature `{ni:04d}.json`.
  - Must NOT reference `config` directly (not picklable) — pass needed scalar values
    (thresholds, cv_folds, top_k, etc.) as plain args.
  - **Verify**: Call `_process_single_node` on one node in isolation — output JSON
    matches what the serial loop produces for that node.

### 2. Handle the `n_jobs` contention

**File**: `proteinlens/analysis/geometry/classifiers.py`

- [ ] **2.1** Add a `parallel_mode` parameter (default `False`) to `train_motif_classifier`.
  When `True`, set `n_jobs=1` on the RandomForestClassifier instead of `n_jobs=-1`.
  All other logic unchanged.
  - **Verify**: `train_motif_classifier(..., parallel_mode=False)` produces identical
    results to current code. `parallel_mode=True` produces identical results but RF
    uses 1 thread.

### 3. Add parallel dispatch to `run_geometry_residue_enrichment`

**File**: `proteinlens/analysis/feature_pipeline/geometry_residue_enrichment.py`

- [ ] **3.1** Add `geometry_n_workers: int = 1` to `PipelineConfig` in `config.py`.
  `1` means serial (current behavior). `> 1` enables multiprocessing.
  - **Verify**: `PipelineConfig(sae_dir="x", output_dir="y")` instantiates with
    `geometry_n_workers=1`.

- [ ] **3.2** In `run_geometry_residue_enrichment`, after pre-loading protein data:
  - Build the list of node indices to process (skipping dead/insufficient nodes).
  - If `config.geometry_n_workers == 1`: run the existing serial loop (calling
    `_process_single_node` for each).
  - If `config.geometry_n_workers > 1`: use `multiprocessing.Pool(n_workers)` with
    `pool.map(_process_single_node, args_list)`.
  - Collect all returned summary entries and write `summary.json` once at the end.
  - **Verify**: Run with `n_workers=1` and `n_workers=4` on the same data — both
    produce identical per-feature JSONs (byte-for-byte diff).

- [ ] **3.3** Use `fork` start method (Linux default) so pre-loaded `all_protein_data`
  is shared via copy-on-write, not serialized. Add an explicit check:
  ```python
  import multiprocessing as mp
  mp.set_start_method("fork", force=True)
  ```
  Only set if `n_workers > 1`. Document that this requires Linux (macOS defaults
  to `spawn` which would serialize all data).
  - **Verify**: Memory usage with 4 workers is ~1.5x single-worker, not 4x.

### 4. Wire up the CLI

**File**: `scripts/run_feature_pipeline.py`

- [ ] **4.1** Add `--geometry-workers` CLI arg (default 1), pass to
  `PipelineConfig(geometry_n_workers=...)`.
  - **Verify**: `python scripts/run_feature_pipeline.py --help` shows the new arg.

### 5. Determinism test

**File**: `tests/test_feature_pipeline/test_geometry_residue_enrichment.py`

- [ ] **5.1** Add `test_parallel_matches_serial`: create synthetic protein data
  (5 proteins, 50 residues, 10 SAE nodes with signal), run `_process_single_node`
  on 3 nodes serially, then run the same 3 nodes via `Pool(2)`. Assert all output
  JSONs are byte-identical.
  - **Verify**: `pytest tests/test_feature_pipeline/test_geometry_residue_enrichment.py::test_parallel_matches_serial -v` passes.

- [ ] **5.2** Add `test_summary_json_complete_after_parallel`: run parallel dispatch
  on 5 nodes, verify summary.json contains entries for all 5 (no lost writes).

### 6. Run full regression

- [ ] **6.1** Run the full test suite: `conda run -n interplm pytest tests/ -v`.
  All tests pass (including the 37 geometry tests).
  - **Verify**: 0 failures.

- [ ] **6.2** Run Stage 6c on `feature_data_test_500/` with `--geometry-workers 4`.
  Compare a sample of 10 per-feature JSONs against the serial run — must be identical.

## Files Modified

| File | Change |
|------|--------|
| `proteinlens/analysis/feature_pipeline/geometry_residue_enrichment.py` | Extract `_process_single_node`, add parallel dispatch |
| `proteinlens/analysis/geometry/classifiers.py` | Add `parallel_mode` param to `train_motif_classifier` |
| `proteinlens/analysis/feature_pipeline/config.py` | Add `geometry_n_workers` field |
| `scripts/run_feature_pipeline.py` | Add `--geometry-workers` CLI arg |
| `tests/test_feature_pipeline/test_geometry_residue_enrichment.py` | Add determinism + summary completeness tests |

## Key Constraint

**`n_workers=1` must produce byte-identical output to the current serial code.**
The refactoring (extracting `_process_single_node`) must not change any computation,
ordering, or formatting. The only behavioral difference when `n_workers > 1` is
execution order of nodes — but since nodes are independent and results are written
to separate files, final output is identical.

---

# Web Console for SAE Feature Visualizer

## Context

We have a trained ReLUSAE (5120 features, 320D, ESM2-8M layer 3) and a feature analysis pipeline producing rich per-feature data: top activating sequences with per-residue activations, InterPro annotation enrichment (F1 scores), geometry enrichment (LassoCV R2, GBM AUC, concordance, motif superposition), and precomputed plot data.

Two data directories exist:
- `feature_data_test_20/` — all stages complete, but geometry enrichment empty (too few proteins for signal)
- `feature_data_test_500/` — still running (partial interpro enrichment, no geometry yet)

Goal: localhost web console to browse all this data. Two pages: homepage dashboard + per-feature detail page.

---

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | **FastAPI + uvicorn** | Python ecosystem, async, trivial file serving |
| Frontend | **Vanilla HTML/JS** (no build step) | Only 2 pages, zero framework overhead |
| Feature table | **AG Grid Community** (CDN) | Row virtualization handles 5120 rows |
| 3D protein viz | **3Dmol.js** (CDN) | ~500KB, loads PDB in 3 lines, per-residue coloring API |
| Charts | **Plotly.js** (CDN) | Already used in project's Python side |
| Styling | **Pico CSS** (CDN) | Classless minimal CSS |

No npm, no webpack, no node_modules. All JS from CDN.

---

## File Structure

```
proteinlens/viz/
  __init__.py
  server.py              # FastAPI app + CLI: python -m proteinlens.viz.server --data-dir ... --port 8050
  api.py                 # API route handlers
  index_builder.py       # Startup: merge summary files into single feature index
  static/
    index.html           # Homepage: model card + feature table
    feature.html         # Feature detail page
    css/style.css         # Custom styles (minimal)
    js/
      homepage.js         # AG Grid setup, stats rendering, row click nav
      feature_detail.js   # Orchestrates feature page sections
      mol_viewer.js       # 3Dmol.js wrapper: load PDB, color by activation
      sequence_strip.js   # Canvas-based residue activation strip + domain overlay
      profile_plots.js    # Plotly charts for geometry overlay/concordance
```

---

## API Endpoints

```
GET  /                              → index.html
GET  /feature/{feature_id}          → feature.html (JS reads ID from URL)

GET  /api/stats                     → dataset_stats.json + SAE config.yaml merged
GET  /api/index                     → Feature table data (5120 rows, ~200KB)
GET  /api/feature/{id}              → features/XXXX.json
GET  /api/feature/{id}/interpro     → interpro_enrichment/XXXX.json (404 if missing)
GET  /api/feature/{id}/geometry     → geometry_enrichment/XXXX.json (404 if missing)
GET  /api/pdb/{accession}           → PDB file from pdb_cache/ (text/plain)
GET  /api/interpro/{accession}      → interpro_cache/{accession}.json (for domain overlays)
```

---

## Data Sources (all read dynamically, nothing hardcoded)

### Homepage stats header reads:
- `dataset_stats.json` → esm_model, esm_layer, organism_taxid, total_proteins, total_clusters, activation_threshold, num_features, sae_dir
- `trained_models/.../config.yaml` (path from dataset_stats.json.sae_dir) → dictionary_size, expansion_factor, activation_dim, l1_penalty, lr, steps, wandb_name

### Homepage pipeline status badges read:
- `pipeline_state.json` → completed_stages list
- File counts: `len(glob(features/*.json))`, `len(glob(interpro_enrichment/*.json))`, `len(glob(geometry_enrichment/*.json))`

### Feature table index (built at startup) merges:
- `survey_coverage.json` → per-feature n/pct proteins/clusters activated
- `feature_max_activations.npy` → max_activation per feature
- `interpro_enrichment/summary.json` (if exists) → best protein/residue F1 per feature
- `geometry_enrichment/summary.json` (if exists) → r2_cv, gbm_auc per feature
- Falls back to scanning per-feature files if summaries missing/empty. Missing data → `null`.

### Feature detail page reads:
- `features/XXXX.json` → top_sequences, activation_bins, coverage, per_residue_activations
- `interpro_enrichment/XXXX.json` → F1 scores, thresholds, interpretation, confusion matrix
- `geometry_enrichment/XXXX.json` → r2_cv, monomial, GBM AUC, concordance, motif, plot_data
- `interpro_cache/{accession}.json` → domain boundaries for sequence strip overlay
- `pdb_cache/AF-{accession}-F1-model_v*.pdb` → 3D structure

---

## Page 1: Homepage

### Header Section
- **SAE Model Card**: architecture (ReLUSAE), dict_size, expansion_factor, activation_dim, l1_penalty, wandb_name, training steps, lr
- **Dataset Card**: ESM model, layer, organism taxid, total proteins, total clusters, activation_threshold
- **Pipeline Status Badges**: "Features: X/5120", "InterPro: X/5120", "Geometry: X/5120" + list of completed stages

### Feature Table (AG Grid)
Columns:
- feature_id
- max_activation
- pct_proteins_activated
- pct_clusters_activated (cluster-adjusted, accounts for duplication)
- interpro_protein_best_f1 + annotation name
- interpro_residue_best_f1
- geometry_protein_r2_cv
- geometry_residue_gbm_auc_cv

All sortable + filterable. Null = "—", sorts to bottom. Color-coded: green intensity proportional to F1/R2 values. Row click → `/feature/{feature_id}`.

---

## Page 2: Feature Detail

JS fetches `/api/feature/{id}`, `/api/feature/{id}/interpro`, `/api/feature/{id}/geometry` in parallel. Missing enrichment → 404 → section hidden.

### Section 1: Summary Stats
Grid of cards:
- **Coverage**: "Activates on X/Y proteins (Z%), X/Y clusters (Z%)"
  - pct_clusters_activated accounts for duplication via clustering
- **InterPro protein-level**: best annotation name, F1, threshold (normalized + absolute), precision, recall, TP/FP/FN counts, interpretation string
- **InterPro residue-level**: best annotation name, F1, threshold, residue counts
- **Geometry protein-level**: R2_cv, pearson_r, monomial formula, top features with weights
- **Geometry residue-level**: GBM AUC, Tree F1, concordance (Spearman r, residue AUROC, F1, precision, recall)
- **Motif superposition**: mean RMSD, n_fragments, per-position flexibility
- Hidden sections display "Pending" or "Not enough data" when 404

### Section 2: Top 5 Most Activating Proteins
For each protein in `top_sequences[:5]`:
- **Label**: accession, max_activation, sequence_length
- **Sequence strip** (canvas): residues colored white→red by per_residue_activations (normalized to feature max)
  - Hover tooltip: residue letter, position (1-based), activation value
  - **InterPro domain overlay**: thin colored bar below strip showing domain boundaries for the best-matching annotation (fetched from `/api/interpro/{accession}`)
- **3D viewer** (3Dmol.js, 400×300): cartoon representation, per-residue coloring white→red by activation
  - Lazy-loaded via IntersectionObserver
  - If `pdb_available: false` → "No structure available" placeholder

### Section 3: Activation Bins (collapsible)
Four `<details>` sections (0.75-1.0, 0.5-0.75, 0.25-0.5, 0.0-0.25):
- Header: bin range + protein count
- Same sequence strip + domain overlay + 3D viewer per protein
- **Lazy**: PDB fetch + 3Dmol init only on `<details>` open

### Section 4: Geometry Plots (only if geometry data exists)
For each protein in `plot_data.top_proteins`:
- **Activation vs Geometry overlay** (Plotly): dual Y-axis, SAE activation (red) vs geom_prob (blue), shaded activated regions
- **Concordance strip** (canvas): agree=green, fp=orange, fn=purple, tn=gray
- **Top feature traces**: Plotly traces from `top_feature_traces` dict

**Motif superposition 3D view** (3Dmol.js): mean_structure_pdb colored by per_position_flexibility (blue=rigid → red=flexible)

---

## Handling Partial/Missing Data

- Enrichment file missing → API returns 404 → JS hides section, shows "Pending" badge
- `pdb_available: false` → placeholder in 3D viewer spot
- Pipeline status badges show completion counts from file system scan
- Works for test_20 (no geometry), test_500 (partial interpro), and future full runs

---

## Performance

- Homepage index: single ~200KB JSON built once at startup (no per-feature file reads for table)
- AG Grid virtualizes: ~50 DOM nodes for 5120 rows
- Feature JSONs: 100-500KB each, loaded on demand
- PDB files: `Cache-Control: max-age=86400` (immutable)
- 3Dmol.js: lazy init, max ~5-8 active WebGL contexts
- Bin sections: lazy-load PDBs only when expanded

---

## Implementation Order

### Phase 1: Server Skeleton
- `proteinlens/viz/__init__.py`
- `proteinlens/viz/server.py`: FastAPI app, CLI args (`--data-dir`, `--sae-dir` defaulting from dataset_stats, `--port`), uvicorn startup, static file mount
- `proteinlens/viz/index_builder.py`: reads survey_coverage, feature_max_activations.npy, enrichment summaries → builds feature index list + pipeline status counts
- `proteinlens/viz/api.py`: all 7 API endpoints

### Phase 2: Homepage
- `static/index.html`: page skeleton with CDN imports (AG Grid, Pico CSS)
- `static/js/homepage.js`: fetch `/api/stats` + `/api/index`, render model/dataset cards, init AG Grid with sorting/filtering/coloring, row click handler
- `static/css/style.css`: card layout, color scales, table tweaks

### Phase 3: Feature Detail — Stats + Sequences
- `static/feature.html`: page skeleton
- `static/js/feature_detail.js`: fetch 3 endpoints, render summary stat cards, create sequence strips for top 5
- `static/js/sequence_strip.js`: canvas renderer for residue activation strip + InterPro domain overlay bar

### Phase 4: 3D Viewers
- `static/js/mol_viewer.js`: 3Dmol.js wrapper — `createViewer()`, load PDB from `/api/pdb/`, `colorByFunction` mapping residue index → activation → white-to-red
- Wire into feature_detail.js with IntersectionObserver for lazy loading
- Limit active WebGL contexts (~8 max)

### Phase 5: Bins
- Add collapsible `<details>` sections to feature_detail.js
- Lazy PDB/viewer init on toggle open

### Phase 6: Geometry Plots
- `static/js/profile_plots.js`: Plotly dual-axis chart (SAE activation vs geom_prob), concordance canvas strip, top feature trace overlays
- Motif 3Dmol.js viewer colored by flexibility
- Only rendered when geometry endpoint returns 200

### Phase 7: Polish
- Loading spinners during fetch
- Error/empty states ("No enrichment data", "Pipeline still running")
- Responsive card layout

---

## Dependencies

```yaml
# Add to environment.yml pip section
- fastapi>=0.115.0
- uvicorn[standard]>=0.30.0
- pyyaml  # if not already present (for reading config.yaml)
```

No frontend deps — all CDN.

---

## Verification

1. `conda run -n interplm python -m proteinlens.viz.server --data-dir feature_data_test_20 --port 8050`
2. `http://localhost:8050/` — homepage loads with 5120-row table, model+dataset cards, pipeline status showing all stages complete
3. Sort by pct_proteins_activated descending — high-coverage features at top
4. Click feature 0 → detail page with stats, sequence strips (white→red coloring), 3D viewers load
5. InterPro domain overlay visible on sequence strips for features with enrichment
6. Expand "0.75-1.0" bin → proteins load lazily with PDB viewers
7. Switch to `--data-dir feature_data_test_500` → interpro columns populate for ~354 features, geometry columns show "—"
8. When geometry enrichment completes → restart server → geometry section appears on feature pages, geometry columns populate in table

---

## Excluded Pipeline Artifacts (with justification)

| Artifact | Why excluded |
|----------|-------------|
| `survey_top20.json` | Same data already in features/XXXX.json top_sequences |
| `selection.json` | Internal pipeline artifact — result already in feature JSONs |
| `interpro_selection.json` | Internal pipeline artifact — results in enrichment JSONs |
| `protein_feature_maxes.npy` | Internal (N×5120 matrix) — relevant data surfaced in feature JSONs + coverage |
| `residue_activations/*.npz` | Raw numpy — already in feature JSONs as per_residue_activations |
| `interpro_residue_activations/*.npz` | Same — surfaced in enrichment results |
| `geometry_protein_features.npz` | 55-dim vectors — relevant info in enrichment JSONs (monomial, top_features) |
| `geometry_residue_profiles/*.npz` | Raw profiles — precomputed in plot_data.top_feature_traces |
| `swissprot_human.fasta` | Sequences already in features/XXXX.json and sequences.json |

---

## Pipeline Refactoring: OOM Resilience & Parallelism

### Problem

The pipeline gets OOM-killed repeatedly. Most stages resume fine, but **assembly (Stage 4) restarts from scratch every time** — killed at 14% (699/5120 features after 1h41m). Survey and collection process proteins sequentially despite being embarrassingly parallel.

### Changes

1. **Assembly resumability** — skip features whose JSON already exists on disk
2. **Assembly memory reduction** — reduce NPZ cache from 500→100, periodic clearing
3. **Survey batching** — batch ESM+SAE inference using `extract_embeddings_with_boundaries()`
4. **Collection batching** — same pattern as survey
5. **Parallel PDB downloads** — `ThreadPoolExecutor` for AlphaFold fetches

### Files to Modify

| File | Changes |
|------|---------|
| `proteinlens/analysis/feature_pipeline/assembly.py` | Resume logic, configurable cache, periodic clearing |
| `proteinlens/analysis/feature_pipeline/survey.py` | Batched ESM+SAE with `extract_embeddings_with_boundaries()` |
| `proteinlens/analysis/feature_pipeline/collection.py` | Batched inference + ThreadPoolExecutor for PDBs |
| `proteinlens/analysis/feature_pipeline/config.py` | 4 new config params |
| `scripts/run_feature_pipeline.py` | 4 new CLI flags |

### Checklist: Independently Verifiable Objectives

Each item is a discrete, testable objective. Items marked [REGRESSION] verify exact numerical reproduction of existing behaviour.

#### Assembly Resumability (Change 1a)

- [ ] **A1.** `run_assembly()` skips feature indices whose `features/{NNNN}.json` already exists on disk
- [ ] **A2.** Accessions from skipped feature JSONs are correctly collected into `all_referenced_accessions` so that `sequences.json` is complete
- [ ] **A3.** Corrupt/truncated feature JSONs (e.g. from a mid-write kill) are detected and re-assembled rather than silently skipped
- [ ] **A4.** [REGRESSION] Running assembly from scratch (empty `features/` dir) produces byte-identical output to the current code for all `features/*.json`, `sequences.json`, and `dataset_stats.json`
- [ ] **A5.** [REGRESSION] Running assembly after deleting a subset of feature JSONs produces outputs identical to a full fresh run

#### Assembly Memory Reduction (Change 1b)

- [ ] **A6.** `MAX_NPZ_CACHE` reads from `config.assembly_npz_cache_size` (default 100)
- [ ] **A7.** NPZ cache is cleared every 256 features to prevent memory creep
- [ ] **A8.** [REGRESSION] Reducing cache size does not change any feature JSON output (same inputs → same outputs regardless of cache size)
- [ ] **A9.** CLI flag `--assembly-cache-size` correctly sets the config value

#### Survey Batching (Change 2)

- [ ] **S1.** Survey processes proteins in batches of `config.survey_batch_size` using `extract_embeddings_with_boundaries()`
- [ ] **S2.** Proteins are sorted by sequence length before batching to minimize padding
- [ ] **S3.** SAE normalization (`_normalize_input_and_get_norms`) is applied to the concatenated batch before `encode()`, identical to the per-protein path
- [ ] **S4.** Per-protein max is computed by splitting the concatenated activations at the correct boundaries — no off-by-one errors
- [ ] **S5.** Checkpoint logic still flushes memmap and saves state every `survey_checkpoint_every` proteins
- [ ] **S6.** Resume after kill works: already-processed proteins are skipped, new proteins are batched
- [ ] **S7.** [REGRESSION] `protein_feature_maxes.npy` from batched survey is numerically identical (within float32 tolerance ≤ 1e-6 relative) to single-protein baseline on the same input
- [ ] **S8.** [REGRESSION] `feature_max_activations.npy`, `survey_top20.json`, `survey_coverage.json` derived outputs match baseline
- [ ] **S9.** Test: run survey with `batch_size=1` and `batch_size=8` on 50 proteins, assert `np.allclose(memmap_bs1, memmap_bs8, atol=1e-5)`
- [ ] **S10.** CLI flag `--survey-batch-size` correctly sets the config value

#### Collection Batching (Change 3a)

- [ ] **C1.** Collection processes proteins in batches using `extract_embeddings_with_boundaries()` + concatenated SAE encode
- [ ] **C2.** Each protein's `.npz` file is saved individually after splitting the batch activations
- [ ] **C3.** Existing `.npz` files are still skipped (resumability preserved)
- [ ] **C4.** [REGRESSION] For each protein, the `activations` array in the `.npz` is numerically identical (within float32 tolerance ≤ 1e-6 relative) to the single-protein baseline
- [ ] **C5.** Test: run collection with `batch_size=1` and `batch_size=8` on 20 selected proteins, compare every `.npz` file with `np.allclose(a, b, atol=1e-5)`
- [ ] **C6.** CLI flag `--collection-batch-size` correctly sets the config value

#### Collection PDB Parallelism (Change 3b)

- [ ] **C7.** PDB downloads use `ThreadPoolExecutor` with `config.collection_pdb_workers` workers
- [ ] **C8.** Each thread uses its own `requests.Session` (no shared mutable state)
- [ ] **C9.** [REGRESSION] Same set of PDB files are fetched as sequential baseline (order may differ, content identical)
- [ ] **C10.** Failures in one thread do not crash the pool — failed PDBs are counted in `n_pdb_failed`
- [ ] **C11.** CLI flag `--collection-pdb-workers` correctly sets the config value

#### Config & CLI (Change 4)

- [ ] **CF1.** `PipelineConfig` has 4 new fields: `assembly_npz_cache_size`, `survey_batch_size`, `collection_batch_size`, `collection_pdb_workers` with documented defaults
- [ ] **CF2.** All 4 CLI flags are wired through to config in `run_feature_pipeline.py`
- [ ] **CF3.** Default values produce identical behaviour to current code (except assembly cache size, which is intentionally reduced)

#### Regression Test Suite

- [ ] **T1.** Write `tests/test_pipeline_regression.py` with a fixture that runs the pipeline on a small dataset (~10 proteins)
- [ ] **T2.** Test compares batched vs single-protein survey output: `np.allclose` on the full memmap
- [ ] **T3.** Test compares batched vs single-protein collection output: `np.allclose` on each `.npz`
- [ ] **T4.** Test verifies assembly resume: run assembly, delete half the feature JSONs, re-run, compare outputs to full fresh run
- [ ] **T5.** Test verifies assembly cache size does not affect output: run with cache=10 and cache=500, compare all feature JSONs
- [ ] **T6.** All tests use real ESM + SAE models (no mocks, per project conventions)

---

## Assembly Parallelization (Stage 4)

### Problem

Assembly (Stage 4) processes 5120 features sequentially.  Each feature
requires loading `.npz` files from CephFS (networked storage), extracting
one column per protein, and serializing to JSON.  On the 10k-protein run
this takes ~4 s/feature → **~5.7 hours** for 5120 features.  Features
are fully independent: each writes its own `features/NNNN.json`, reads
from the same upstream files, and touches a disjoint set of output paths.

### Design

Use `multiprocessing.Pool` (not threads — JSON serialization is
CPU-bound and the GIL would prevent true parallelism) to process
features in parallel.  Each worker:

1. Loads its own copy of the upstream data (selection, survey_top,
   survey_coverage, sequences, cluster_map, protein_maxes memmap).
   The memmap is opened in read-only mode (`mode="r"`) so the OS
   can share physical pages across workers via copy-on-write.
2. Maintains its own NPZ cache (no cross-process sharing needed).
3. Writes `features/NNNN.json` files directly (no conflicts since
   each feature index is assigned to exactly one worker).
4. Returns a set of referenced accessions and a set of corrupt NPZ
   accessions back to the parent.

The parent process:

1. Partitions `range(num_features)` into contiguous chunks, one per
   worker.  Contiguous chunks improve NPZ cache hit rates because
   nearby features tend to share the same top-activating proteins.
2. Dispatches chunks via `pool.map()` or `pool.starmap()`.
3. Merges the returned accession sets from all workers.
4. Writes `sequences.json` and `dataset_stats.json` (these depend on
   the merged accession set, so they must happen in the parent).

### Why multiprocessing and not threading

- `_build_protein_entry` does CPU-intensive work: `np.load` (zlib
  decompression), numpy column slicing, Python list conversion with
  `float()` rounding, and `json.dump`.  All of these hold the GIL.
- The memmap backing file is opened read-only by all workers, so the
  OS can share physical pages.  Each worker's ~1 GB of Python dicts
  (sequences, selection, etc.) is duplicated, but with 4 workers
  that's ~4 GB — well within 64 GB.

### Resumability interaction

The existing per-feature resumability (A1–A3) works unchanged:
each worker checks `out_path.exists()` and skips valid JSONs.  Since
feature indices are partitioned (not interleaved), a partially
completed parallel run can be resumed by any number of workers — the
already-written JSONs are skipped regardless of which worker wrote them.

### Files to Modify

| File | Changes |
|------|---------|
| `proteinlens/analysis/feature_pipeline/assembly.py` | Extract worker function, add `Pool` dispatch, merge step |
| `proteinlens/analysis/feature_pipeline/config.py` | 1 new field: `assembly_workers` |
| `scripts/run_feature_pipeline.py` | 1 new CLI flag: `--assembly-workers` |

### Checklist: Independently Verifiable Objectives

Each item is a discrete, testable objective.  Items marked [REGRESSION]
verify exact numerical reproduction of existing behaviour.

#### Config & CLI

- [ ] **P1.** `PipelineConfig` has a new field `assembly_workers: int = 1`
      with docstring explaining that `1` means single-process (original
      behaviour) and `>1` uses multiprocessing.
- [ ] **P2.** CLI flag `--assembly-workers` correctly sets the config value.
- [ ] **P3.** Default value `1` produces identical behaviour to current
      sequential code — no multiprocessing overhead, no fork.

#### Worker function

- [ ] **P4.** Extract a top-level (picklable) worker function
      `_assemble_feature_range(args)` that:
      - Accepts a tuple of `(feature_indices, config, ...)` or a
        serialisable argument bundle (dataclass or dict).
      - Loads upstream data independently (no shared mutable state).
      - Opens the protein_maxes memmap in read-only mode (`mode="r"`).
      - Maintains its own NPZ cache and corrupt-accession set.
      - Processes each feature index in its range using the existing
        `_assemble_single_feature` function (no changes to that function).
      - Respects the existing resumability logic (skip valid JSONs).
      - Returns `(referenced_accessions: Set[str],
        corrupt_accessions: Set[str], n_assembled: int, n_skipped: int)`.
- [ ] **P5.** The worker function is defined at module level (not nested
      inside `run_assembly`) so that `multiprocessing` can pickle it.
- [ ] **P6.** The worker function does NOT write `sequences.json` or
      `dataset_stats.json` — only per-feature JSONs.

#### Dispatch and merge

- [ ] **P7.** `run_assembly` partitions `range(num_features)` into
      `assembly_workers` contiguous chunks (e.g. worker 0 gets features
      0–1279, worker 1 gets 1280–2559, etc. for 4 workers).
- [ ] **P8.** When `assembly_workers == 1`, the worker function is called
      directly (no `Pool`, no fork) to preserve identical single-process
      behaviour and avoid serialisation overhead.
- [ ] **P9.** When `assembly_workers > 1`, a `multiprocessing.Pool` is
      created with `assembly_workers` processes.  The pool uses
      `pool.map()` or `pool.starmap()` — NOT `pool.apply_async` with
      manual result collection (simpler, less error-prone).
- [ ] **P10.** After all workers complete, the parent merges:
      - Union of all `referenced_accessions` sets.
      - Union of all `corrupt_accessions` sets.
      - Sum of all `n_assembled` and `n_skipped` counts.
- [ ] **P11.** `sequences.json` and `dataset_stats.json` are written by
      the parent process after the merge, using the merged accession set.
      Output is identical regardless of worker count.

#### Error handling

- [ ] **P12.** If any worker raises an exception, the pool is terminated
      and the exception is re-raised in the parent with a clear message
      identifying which feature range failed.
- [ ] **P13.** A worker crash (e.g. OOM kill) does not leave the pool
      hung — `pool.map` propagates the error.  Already-written feature
      JSONs from other workers are preserved on disk (resumable).

#### Regression tests

- [ ] **P14.** [REGRESSION] Running assembly with `workers=1` and
      `workers=4` on the same upstream data produces byte-identical
      `features/*.json`, `sequences.json`, and `dataset_stats.json`.
- [ ] **P15.** [REGRESSION] Running assembly with `workers=2` after
      deleting a subset of feature JSONs produces outputs identical to
      a full fresh run with `workers=1` (resume + parallelism interaction).
- [ ] **P16.** Test uses the existing 25-protein regression fixture from
      `test_pipeline_regression.py` (real ESM + SAE, no mocks).
- [ ] **P17.** Test verifies that `assembly_workers=1` does NOT fork
      (no `multiprocessing.Pool` created) by checking that the worker
      function is called directly.

---

## Parallelise Stages 5b, 5c, 6a, 6b, 6c

### Context

Stages 5b, 5c, 6a, 6b, and 6c are embarrassingly parallel but currently run
sequentially.  Each stage's per-unit work (per-protein or per-feature) is fully
independent.  Assembly (Stage 4) already uses `multiprocessing.Pool` with a
`workers` config param and a module-level worker function — we follow that
pattern exactly.

**Correctness constraint:** parallel outputs must be **byte-identical** to
serial outputs for the same inputs.  All five stages produce deterministic
outputs given their inputs (no random seeds, no floating-point reduction-order
sensitivity), so this is achievable as long as we avoid introducing
order-dependent state.

### Config changes

File: `proteinlens/analysis/feature_pipeline/config.py`

- [ ] **C1.** Add `interpro_fetch_workers: int = 1` — number of concurrent
      threads for Stage 5b API requests.
- [ ] **C2.** Add `interpro_enrichment_workers: int = 1` — number of
      processes for Stage 5c F1 computation.
- [ ] **C3.** Add `geometry_features_workers: int = 1` — number of processes
      for Stage 6a per-protein geometry extraction.
- [ ] **C4.** Add `geometry_protein_enrichment_workers: int = 1` — number of
      processes for Stage 6b per-feature LassoCV.
- [ ] **C5.** Add `geometry_residue_enrichment_workers: int = 1` — number of
      processes for Stage 6c per-feature GBM + Kabsch.

---

### Stage 5b: InterPro Fetch (concurrent HTTP)

File: `proteinlens/analysis/feature_pipeline/interpro_api.py`

5b is **I/O-bound** (HTTP requests), so use `concurrent.futures.ThreadPoolExecutor`
rather than `multiprocessing.Pool`.

- [ ] **5b-1.** Extract the per-protein fetch+count logic from the loop in
      `run_interpro_fetch` into a module-level helper
      `_fetch_single(acc, cache_dir, session_factory, rate_limiter) -> bool`
      that returns `True` if annotations were found.  Each thread must create
      its own `requests.Session` (sessions are not thread-safe).
- [ ] **5b-2.** Remove the existing `RateLimiter` usage.  The user has
      confirmed the API does not enforce strict rate limiting, so the
      rate limiter is unnecessary for concurrent requests.
- [ ] **5b-3.** When `interpro_fetch_workers == 1`, call `_fetch_single` in
      a simple loop (no executor created) — preserves identical serial
      behaviour.
- [ ] **5b-4.** When `interpro_fetch_workers > 1`, use
      `ThreadPoolExecutor(max_workers=N)` with `executor.map(...)`.
      Wrap in tqdm for progress.
- [ ] **5b-5.** Aggregate `n_with_annotations` / `n_empty` counts from
      returned booleans after all futures complete.  Final print + wandb log
      unchanged.
- [ ] **5b-6.** Cache files are keyed by accession (one file per protein),
      so concurrent writes to **different** files are safe on any filesystem.
      No locking needed.

---

### Stage 5c: InterPro Enrichment (parallel features)

File: `proteinlens/analysis/feature_pipeline/interpro_enrichment.py`

5c is **CPU-bound** (F1 threshold sweeps over numpy arrays).  Follow the
assembly pattern: `multiprocessing.Pool` with a module-level worker function.

- [ ] **5c-1.** Extract the per-feature body of the loop in
      `run_interpro_enrichment` (lines 93–194) into a module-level function
      `_enrich_feature_range(args) -> dict` that processes a contiguous range
      of feature indices.  It must:
      - Accept a single tuple `(feat_range, config)` (picklable).
      - Load shared read-only data (memmap, global_max, interpro_selection,
        pipeline_state) inside the worker (each process gets its own copy;
        memmap is read-only so the OS shares physical pages via
        copy-on-write).
      - Write per-feature JSON files to `interpro_enrichment/{feat:04d}.json`.
      - Return a dict with `n_analyzed`, `n_skipped`, and a partial
        `summary_features` dict for its range.
- [ ] **5c-2.** Partition features into `interpro_enrichment_workers`
      contiguous chunks (same chunking logic as assembly: `np.array_split`).
- [ ] **5c-3.** When `workers == 1`, call the worker function directly (no
      fork).  When `workers > 1`, use `multiprocessing.Pool.map`.
- [ ] **5c-4.** Parent merges returned `summary_features` dicts and
      `n_analyzed`/`n_skipped` counts, then writes `summary.json` and
      logs to wandb.
- [ ] **5c-5.** Per-feature JSON output must be identical to serial: same
      key ordering (guaranteed by insertion order in `json.dump` with
      `indent=2`), same floating-point rounding (all values go through
      `round(..., 4)` already).

---

### Stage 6a: Geometry Features (parallel proteins)

File: `proteinlens/analysis/feature_pipeline/geometry_features.py`

6a is **CPU-bound** (PDB parsing + geometry math).  Parallelise over proteins.

- [ ] **6a-1.** Extract the per-protein body (lines 99–153) into a
      module-level function `_compute_single_protein(args) -> dict | None`
      that accepts `(pdb_path, sequences, profiles_dir, existing_set)`.
      Returns a dict `{"acc": str, "geom": dict}` on success, or `None`
      on skip/failure.  Writes the `.npz` file to `profiles_dir` directly
      (each protein writes its own file — no collision).
- [ ] **6a-2.** When `workers == 1`, process in a simple loop.  When
      `workers > 1`, use `multiprocessing.Pool.imap_unordered` for progress
      tracking with tqdm.
- [ ] **6a-3.** After all proteins are processed (serial or parallel), the
      parent rebuilds the protein-level geometry matrix from all `.npz`
      profiles (lines 171–198 — unchanged).  This final aggregation step
      is inherently serial but fast.
- [ ] **6a-4.** `sequences` dict must be passed to each worker.  It's
      read-only and small (accession -> sequence string).  Pass via the
      args tuple.

---

### Stage 6b: Protein-Level Geometry Enrichment (parallel features)

File: `proteinlens/analysis/feature_pipeline/geometry_protein_enrichment.py`

6b is **CPU-bound** (LassoCV per feature).  Parallelise over features.

- [ ] **6b-1.** Extract the per-feature body (lines 120–191) into a
      module-level function `_fit_feature_range(args) -> dict` that
      processes a contiguous range of feature indices.  It must:
      - Accept `(feat_range, config)`.
      - Load shared data inside the worker: `geometry_protein_features.npz`,
        pipeline state, feature maxes, and the activation memmap (read-only).
      - Compute the `valid_geom_indices` / `memmap_rows` / `X_geom` /
        `geom_valid` arrays once per worker (same for all features).
      - Write per-feature JSON to `geometry_enrichment/{feat:04d}.json`.
      - Return `n_fitted`, `n_skipped_*` counts and a partial
        `summary_features` dict.
- [ ] **6b-2.** Partition features into contiguous chunks
      (`np.array_split`).
- [ ] **6b-3.** When `workers == 1`, call directly.  When `workers > 1`,
      use `multiprocessing.Pool.map`.
- [ ] **6b-4.** Parent merges summary dicts and counts, writes
      `summary.json`.
- [ ] **6b-5.** Per-feature JSON must handle the "load existing JSON if
      present" merge (6c may have written it).  Since 6b and 6c write to
      the same files, they must NOT run in parallel with each other — only
      parallelise **within** each stage.  Document this ordering constraint
      in a comment in the pipeline runner.

---

### Stage 6c: Residue-Level Geometry Enrichment (parallel features)

File: `proteinlens/analysis/feature_pipeline/geometry_residue_enrichment.py`

6c is **CPU-bound** (GBM training + Kabsch alignment + plot precomputation).
The heaviest stage — most to gain from parallelism.

- [ ] **6c-1.** The `_preload_all_protein_data` call (line 384) must remain
      in the parent process.  The returned dict is large but read-only.
      With `multiprocessing.Pool` (fork), child processes inherit it via
      copy-on-write.  Store it in a module-level variable set before the
      pool is created, so workers can access it without pickling.
- [ ] **6c-2.** Extract the per-feature body (lines 401–538) into a
      module-level function `_enrich_feature_range(args) -> dict` that
      processes a contiguous range of feature indices.  It must:
      - Accept `(feat_range, config_dict)` where `config_dict` contains
        the serialisable config values needed (not the full PipelineConfig
        if it's not picklable — test this).
      - Access the module-level `_ALL_PROTEIN_DATA` and read-only memmap.
      - Write per-feature JSON to `geometry_enrichment/{feat:04d}.json`.
      - Return `n_fitted`, `n_skipped_*` counts and a partial summary dict.
- [ ] **6c-3.** Partition features into contiguous chunks.  When
      `workers == 1`, call directly.  When `workers > 1`, use
      `multiprocessing.Pool.map`.
- [ ] **6c-4.** Parent merges summary dicts and writes `summary.json`.
- [ ] **6c-5.** `_precompute_plot_data` uses sklearn's `predict_proba` —
      the GBM is trained inside each worker and never shared across
      processes, so this is safe.
- [ ] **6c-6.** Ensure `_filter_proteins_for_node` accesses the memmap
      (read-only) — no mutation, safe across fork.

---

### Testing

All tests below use the existing regression fixture infrastructure from
`test_pipeline_regression.py` (real ESM + SAE, no mocks) unless noted.

#### Unit tests

File: `tests/test_feature_pipeline/test_parallel_stages.py`

- [ ] **T1.** Test `_fetch_single` (5b) returns correct boolean and writes
      cache file.  Use a local HTTP mock server (`responses` library or
      `unittest.mock.patch` on `requests.Session.get`) — this is the one
      place mocking is acceptable since we're testing concurrency logic,
      not the API.  Actually: use `responses` library to mock HTTP since
      the user prefers real APIs but the InterPro API is external and
      rate-limited.  Mark with `@pytest.mark.unit`.
- [ ] **T2.** Test `_enrich_feature_range` (5c) on a synthetic 3-feature
      dataset: hand-computed F1 values must match.  Verifies the worker
      function in isolation.
- [ ] **T3.** Test `_compute_single_protein` (6a) on a real PDB file from
      `pdb_cache/` (if available in fixture) or a minimal synthetic PDB
      string.  Verify `.npz` output contains expected keys.
- [ ] **T4.** Test `_fit_feature_range` (6b) on a synthetic geometry matrix
      + activation vector.  Verify JSON output schema and that LassoCV
      coefficients are deterministic.
- [ ] **T5.** Test `_enrich_feature_range` (6c) on a minimal synthetic
      dataset with 2 features.  Verify JSON output contains expected keys
      (`geometric_residue_level`, `plot_data`).

#### Regression tests

- [ ] **T6.** [REGRESSION] Stage 5b: `interpro_fetch_workers=1` and
      `interpro_fetch_workers=4` produce identical cache files (same set
      of `interpro_cache/*.json` with identical content).
- [ ] **T7.** [REGRESSION] Stage 5c: `interpro_enrichment_workers=1` and
      `interpro_enrichment_workers=4` produce byte-identical
      `interpro_enrichment/*.json` and `summary.json`.
- [ ] **T8.** [REGRESSION] Stage 6a: `geometry_features_workers=1` and
      `geometry_features_workers=4` produce identical
      `geometry_residue_profiles/*.npz` and `geometry_protein_features.npz`.
      Compare numpy arrays with `np.testing.assert_array_equal` (exact,
      not approximate — these computations are deterministic).
- [ ] **T9.** [REGRESSION] Stage 6b: `geometry_protein_enrichment_workers=1`
      and `workers=4` produce byte-identical
      `geometry_enrichment/*.json` and `summary.json`.
- [ ] **T10.** [REGRESSION] Stage 6c:
      `geometry_residue_enrichment_workers=1` and `workers=4` produce
      byte-identical `geometry_enrichment/*.json`.  Note: 6c must run
      **after** 6b in both cases (ordering constraint from 6b-5).
- [ ] **T11.** [REGRESSION] `workers=1` path does NOT create a
      `multiprocessing.Pool` or `ThreadPoolExecutor` — verify via
      `unittest.mock.patch` on the pool constructor.
- [ ] **T12.** [REGRESSION] End-to-end: run full pipeline stages 5a→5b→5c
      and 6a→6b→6c with `workers=1`, then again with `workers=2`.  All
      output files identical.  Use the 25-protein regression fixture.

---

## Per-Feature Resumability for Stages 5c, 6b, 6c

### Context

Stages 5c, 6b, and 6c recompute **all** features from scratch if the
pipeline crashes mid-stage.  Other stages (e.g., assembly/stage 4) already
skip features whose output JSON exists on disk.  This section adds the same
per-feature skip-if-exists logic to 5c, 6b, and 6c so they resume from
where they left off.

All three stages use `multiprocessing.Pool` for parallelism.  The resume
check goes **inside each worker function** (not the parent), so it works
identically for `workers=1` and `workers>1`.  Each worker processes a
disjoint chunk of feature indices writing to separate files — no race
conditions.

### Files to modify

- `proteinlens/analysis/feature_pipeline/interpro_enrichment.py`
- `proteinlens/analysis/feature_pipeline/geometry_protein_enrichment.py`
- `proteinlens/analysis/feature_pipeline/geometry_residue_enrichment.py`
- `tests/test_feature_pipeline/test_pipeline_regression.py`

### Checklist

#### Stage 5c — `interpro_enrichment.py`

- [ ] **R1.** In `_enrich_feature_range()`, add counter `n_resumed = 0`
      alongside `n_analyzed` and `n_skipped` (line ~108).
- [ ] **R2.** At the top of the `for feat_idx` loop (before the
      `feat_max == 0` check), compute `out_path` and insert a resume
      guard:
      - If `out_path` exists, attempt `json.load()`.
      - Validate `existing["feature_id"] == feat_idx`.
      - If valid: extract summary entry (same logic as lines 198–211),
        add to `summary_features`, increment `n_resumed`, `continue`.
      - On `json.JSONDecodeError` / `KeyError` / `OSError`: fall through
        to recompute (corrupt file).
      **Verify:** a feature with a valid JSON on disk is never recomputed.
- [ ] **R3.** Add `"n_resumed": n_resumed` to the dict returned by
      `_enrich_feature_range()` (line ~213).
- [ ] **R4.** In `run_interpro_enrichment()`, accumulate `n_resumed`
      from worker results.  Add `"n_features_resumed"` to `summary.json`.
      Include resumed count in the print statement and `wandb_utils.log`.
      **Verify:** `n_resumed + n_analyzed + n_skipped == num_features`.

#### Stage 6b — `geometry_protein_enrichment.py`

- [ ] **R5.** In `_fit_feature_range()`, add counter `n_resumed = 0`
      alongside existing counters (line ~128).
- [ ] **R6.** At the top of the `for loop_i, ni` loop (after the logging
      block, before `feature_maxes[ni] == 0` check), compute `feat_path`
      and insert a resume guard:
      - If `feat_path` exists, attempt `json.loads()`.
      - Validate `feat_json["feature_id"] == ni` **AND**
        `"geometric_protein_level" in feat_json`.  (The key check is
        critical — 6c may have created the file without 6b's key.)
      - If valid: extract summary (same as lines 197–200), increment
        `n_resumed`, `continue`.
      - On decode/key/IO error: fall through to recompute.
      **Verify:** a file that exists but lacks `"geometric_protein_level"`
      is NOT skipped.
- [ ] **R7.** Add `"n_resumed": n_resumed` to the dict returned by
      `_fit_feature_range()` (line ~203).
- [ ] **R8.** In `run_geometry_protein_enrichment()`, accumulate
      `n_resumed`.  Add `"n_features_resumed"` to `summary.json`.
      Update the `logger.info` message.
      **Verify:** `n_resumed + n_fitted + n_skipped_* == n_features`.

#### Stage 6c — `geometry_residue_enrichment.py`

- [ ] **R9.** In `_enrich_feature_range()`, add counter `n_resumed = 0`
      alongside existing counters (line ~413).
- [ ] **R10.** At the top of the `for loop_i, ni` loop (after the logging
      block, before `feature_maxes[ni] == 0` check), compute `feat_path`
      and insert a resume guard:
      - If `feat_path` exists, attempt `json.loads()`.
      - Validate `feat_json["feature_id"] == ni` **AND**
        `"geometric_residue_level" in feat_json`.
      - If valid: extract summary (same as lines 552–557), increment
        `n_resumed`, `continue`.
      - On decode/key/IO error: fall through to recompute.
      **Verify:** a file that exists but lacks `"geometric_residue_level"`
      (e.g., written by 6b only) is NOT skipped.
- [ ] **R11.** Add `"n_resumed": n_resumed` to the dict returned by
      `_enrich_feature_range()` (line ~561).
- [ ] **R12.** In `run_geometry_residue_enrichment()`, accumulate
      `n_resumed`.  Add `"n_features_residue_resumed"` to `summary.json`.
      Update the `logger.info` message.

#### Tests

- [ ] **R13.** [REGRESSION] Stage 5c resumability: run 5c fresh on the
      test fixture, copy output directory, delete 50 % of per-feature
      JSONs, re-run 5c, assert every per-feature JSON is byte-identical
      to the fresh run and `summary.json` is identical.
- [ ] **R14.** [REGRESSION] Stage 6b resumability: same pattern — fresh
      run, delete half, re-run, assert byte-identical outputs.
- [ ] **R15.** [REGRESSION] Stage 6c resumability: same pattern.
      Additionally verify that 6b's `"geometric_protein_level"` key is
      preserved in every JSON (not clobbered by 6c's resume path).
- [ ] **R16.** [UNIT] Stage 5c: write a valid per-feature JSON to disk
      before calling `_enrich_feature_range` on that feature index.
      Assert `n_resumed == 1` and the feature was not recomputed.
      Then write a truncated/corrupt JSON and assert it IS recomputed.
- [ ] **R17.** [UNIT] Stage 6b: write a JSON with only
      `"geometric_residue_level"` (no `"geometric_protein_level"`).
      Assert the feature is NOT skipped by 6b (it must recompute its
      own key).
- [ ] **R18.** [UNIT] Stage 6c: write a JSON with only
      `"geometric_protein_level"` (no `"geometric_residue_level"`).
      Assert the feature is NOT skipped by 6c.
- [ ] **R19.** [REGRESSION] Parallel resume: run 5c with `workers=1`
      fresh, delete 50 % of outputs, resume with `workers=2`.  Assert
      outputs byte-identical to fresh `workers=1` run.  Repeat for 6b
      and 6c.

---

## Stage 6c: Lazy Loading Refactor

### Context

Stage 6c's `_preload_all_protein_data()` loads ALL ~32k protein geometry profiles + per-residue activation matrices into a single in-memory dict before processing any features. With 50k proteins × ~500 residues × 5094 features × float32, this requires ~200-300GB RAM, causing OOMKilled even with a 256Gi pod limit. The preload is unnecessary because:

- **Filtering uses only `memmap_row` (an int)** — `_filter_proteins_for_node()` iterates all proteins but only checks `pdata["memmap_row"]` against the activation memmap
- **Heavy arrays (`act_matrix`, `ca`, `profiles`) are only accessed for the filtered subset** — typically dozens of proteins per feature, not all 50k

### Approach: Two-phase loading

Split into lightweight metadata (preloaded once, shared via fork/COW) + heavy arrays (loaded from disk per-feature, only for filtered proteins).

**Memory impact**: ~300GB → ~50MB peak (metadata dict + one feature's worth of protein arrays at a time).

**Output impact**: Byte-identical. The protein dicts passed to downstream functions (`collect_node_fragments`, `train_motif_classifier`, `compute_concordance_metrics`, `_precompute_plot_data`) have the exact same structure, types, and values.

### Implementation Checklist

#### L1. Refactor `_preload_all_protein_data()` → `_preload_protein_metadata()`
**File**: `proteinlens/analysis/feature_pipeline/geometry_residue_enrichment.py` (lines 89-182)

- [ ] Rename function to `_preload_protein_metadata`
- [ ] Keep all existing eligibility checks (file existence for both geom + act paths, `n >= 20`)
- [ ] Still load both files to compute `n = min(len(ca), act_matrix.shape[0])` for the eligibility check, then discard the arrays
- [ ] Return lightweight dicts: `{"accession", "memmap_row", "geom_path" (str), "act_path" (str), "n_residues" (int)}`
- [ ] Keep the same logging (`"Pre-loaded %d proteins..."`)

**Verify**: Function returns same number of proteins as before (same eligibility filtering).

#### L2. Add `_load_protein_heavy(meta: dict) -> dict`
**File**: `proteinlens/analysis/feature_pipeline/geometry_residue_enrichment.py`

- [ ] New function that takes a metadata dict and returns the full protein dict (same structure as old preload)
- [ ] Loads `np.load(meta["geom_path"])` and `np.load(meta["act_path"])`
- [ ] Slices all arrays to `[:meta["n_residues"]]`
- [ ] Reconstructs `profiles` dict with same keys: `curvature`, `torsion`, `planarity`, `tangents`, `helix_mask`, `categories`
- [ ] Extracts `sequence` with same logic (line 164-165)
- [ ] Returns dict with keys: `accession`, `act_matrix`, `ca`, `profiles`, `n_residues`, `sequence`, `memmap_row` — identical to old preload output

**Verify**: For any protein, `_load_protein_heavy(meta)` returns arrays that are `np.array_equal` to what `_preload_all_protein_data` produced.

#### L3. Rename module-level variable
**File**: `proteinlens/analysis/feature_pipeline/geometry_residue_enrichment.py` (line 62)

- [ ] Rename `_ALL_PROTEIN_DATA` → `_ALL_PROTEIN_META` (and update docstring at lines 59-62)
- [ ] Update all references: line 395 (`all_protein_data = _ALL_PROTEIN_DATA`), line 659 (assignment), line 661 (empty check), line 676, line 742 (cleanup)

#### L4. Update `_filter_proteins_for_node()` (lines 185-209)
**File**: `proteinlens/analysis/feature_pipeline/geometry_residue_enrichment.py`

- [ ] Change parameter name/type: `all_protein_data` → `all_protein_meta`
- [ ] Returns list of metadata dicts (same filtering logic, just `meta["memmap_row"]` instead of `pdata["memmap_row"]`)
- [ ] Update docstring

#### L5. Update `_enrich_feature_range()` worker (lines 368-598)
**File**: `proteinlens/analysis/feature_pipeline/geometry_residue_enrichment.py`

- [ ] Line 395: `all_protein_meta = _ALL_PROTEIN_META`
- [ ] Lines 469-471: `protein_meta = _filter_proteins_for_node(all_protein_meta, ni, act_memmap)`
- [ ] **Add new step after filtering**: `protein_data = [_load_protein_heavy(m) for m in protein_meta]`
- [ ] All subsequent code (lines 477-598) remains unchanged — it receives the same `protein_data` list of dicts

**Verify**: Worker function produces identical JSON output for all features.

#### L6. Update `run_geometry_residue_enrichment()` entry point (lines 610-748)
**File**: `proteinlens/analysis/feature_pipeline/geometry_residue_enrichment.py`

- [ ] Line 659: Call `_preload_protein_metadata` instead of `_preload_all_protein_data`
- [ ] Line 659: `_ALL_PROTEIN_META = _preload_protein_metadata(config, acc_to_idx)`
- [ ] Line 661: `if not _ALL_PROTEIN_META:`
- [ ] Line 676: `_ALL_PROTEIN_META = {}`
- [ ] Line 742: `_ALL_PROTEIN_META = {}`
- [ ] Update `global _ALL_PROTEIN_DATA` → `global _ALL_PROTEIN_META` (line 631)

#### L7. Revert float16 change
**File**: `proteinlens/analysis/feature_pipeline/geometry_residue_enrichment.py` (line 143)

- [ ] Revert `.astype(np.float16)` back to original — no longer needed, and would change numerical results

#### L8. Update existing test: `TestEnrichFeatureRange6c` (T5)
**File**: `tests/test_feature_pipeline/test_parallel_stages.py` (lines 523-657)

- [ ] Line 536: Import `_ALL_PROTEIN_META` instead of `_ALL_PROTEIN_DATA`
- [ ] Line 620-621: Import `_preload_protein_metadata` instead of `_preload_all_protein_data`
- [ ] Line 624: Call `_preload_protein_metadata(config, acc_to_idx)`
- [ ] Line 627-628: Set `gre_mod._ALL_PROTEIN_META = metadata`
- [ ] Line 633: Restore `gre_mod._ALL_PROTEIN_META = old_data`
- [ ] All assertions (lines 635-656) remain unchanged — they verify JSON output structure

**Verify**: Test passes, same assertions hold.

#### L9. Update existing test: `TestGeometryResidueEnrichmentResumeGuard` (R18)
**File**: `tests/test_feature_pipeline/test_parallel_stages.py` (lines 1360-1476)

- [ ] Same mechanical import/variable name changes as L8
- [ ] Lines 1371-1373: Import new names
- [ ] Line 1444: Call `_preload_protein_metadata`
- [ ] Lines 1445-1446: Set `_ALL_PROTEIN_META`
- [ ] Line 1476: Restore
- [ ] All assertions (lines 1460-1474) remain unchanged

**Verify**: Test passes, resume guard still correctly detects missing `geometric_residue_level` key.

#### L10. Update existing test: `TestGeometryResidueEnrichmentRegression` (T10)
**File**: `tests/test_feature_pipeline/test_parallel_stages.py` (lines 899-1043)

- [ ] Lines 911-912 import `_preload_all_protein_data` — remove unused import (the test calls `run_geometry_residue_enrichment` which uses the new path internally)

**Verify**: Serial vs parallel JSON comparison still passes with same tolerances.

#### L11. Update existing test: `TestGeometryResidueEnrichmentResumability` (R15)
**File**: `tests/test_feature_pipeline/test_parallel_stages.py` (lines 1607-1759)

- [ ] Check if it imports `_preload_all_protein_data` — if so, update/remove unused import

**Verify**: Resume test passes, byte-identical for non-deleted JSONs.

#### L12. NEW regression test: `TestLazyLoadingEquivalence`
**File**: `tests/test_feature_pipeline/test_parallel_stages.py`

- [ ] **test_lazy_load_matches_preload**:
  - Create synthetic on-disk data (same pattern as T5: 10 proteins, 80 residues, 5 features)
  - Inline the old `_preload_all_protein_data` logic as a test helper
  - Call old preload to get `full_data: dict[str, dict]`
  - Call `_preload_protein_metadata` to get metadata, then `_load_protein_heavy(meta)` for each protein
  - Assert `np.array_equal` for every array field: `act_matrix`, `ca`, `profiles["curvature"]`, `profiles["torsion"]`, `profiles["planarity"]`, `profiles["tangents"]`, `profiles["helix_mask"]`, `profiles["categories"]`
  - Assert `==` for scalar fields: `accession`, `n_residues`, `sequence`, `memmap_row`
  - Assert same set of accessions loaded (same eligibility filtering)

#### L13. NEW regression test: `TestLazyLoadingEndToEnd`
**File**: `tests/test_feature_pipeline/test_parallel_stages.py`

- [ ] **test_lazy_produces_identical_json_output**:
  - Create synthetic data (10 proteins, 80 residues, 5 features, rng seed 42)
  - Run full `run_geometry_residue_enrichment(config)` with workers=1
  - Capture all output JSONs
  - Compare against a "golden" reference run (inline old preload code in test, run same stage, compare JSON outputs)
  - For single-worker: assert byte-identical JSON (no GBM non-determinism since same process, same seed)

#### L14. NEW regression test: `TestLazyLoadingMemoryBound`
**File**: `tests/test_feature_pipeline/test_parallel_stages.py`

- [ ] **test_metadata_dict_is_lightweight**:
  - Create synthetic data (100 proteins, 200 residues, 10 features)
  - Call `_preload_protein_metadata`
  - Assert each value in returned dict has NO numpy arrays (only str, int fields)
  - Assert `sys.getsizeof` of the full dict is < 1MB (sanity check that heavy data isn't leaking)

### Verification

```bash
# Run ALL 6c tests (existing + new)
conda run -n interplm pytest tests/test_feature_pipeline/test_geometry_residue_enrichment.py -v

# Run parallel/regression tests for 6c
conda run -n interplm pytest tests/test_feature_pipeline/test_parallel_stages.py -v -k "6c or residue_enrichment or LazyLoading"

# Run full test suite to check nothing else broke
conda run -n interplm pytest tests/test_feature_pipeline/ -v
```

All tests must pass. Single-worker JSON output must be byte-identical to the old preload approach.

---

## Stage 7: Sequence Motif F1 Enrichment

### Context

Discover short amino acid motifs (k-mers) that predict SAE feature activation at the residue level. For each of the 5120 features, find the tripeptide whose presence at a position best predicts high activation. Result: a new "Motif F1" column in the feature table, analogous to InterPro F1 and Geometry AUC.

### Design Decisions

| Decision | Choice | Justification |
|----------|--------|---------------|
| k-mer length | k=3 (tripeptides) | 8000 possible 3-mers; with ~20K residues/feature, common tripeptides appear dozens of times. k=5 (3.2M) needs aggressive filtering and has insufficient statistical power at this sample size. |
| Activation threshold | Sweep 50 thresholds from 0 to feat_max | Consistent with InterPro enrichment. Finds optimal threshold per motif rather than relying on a single arbitrary cutoff. |
| Minimum count | 5 occurrences | k-mers appearing <5 times produce unreliable F1. Count reported alongside for user judgement. |
| Edge handling | Skip positions within floor(k/2) of termini | Avoids artificial padding motifs. Loses only 2 residues per protein for k=3. |
| Multiple testing | Report n_kmers_tested, no F1 adjustment | Same approach as InterPro enrichment — descriptive best-of screen, not inferential. |
| F1 direction | k-mer present → residue activated | Natural direction: motifs whose presence signals activation. |
| Data source | `features/NNNN.json` (top_sequences + activation_bins) | Already has per-residue activations and sequences. No GPU or external API needed. |

### Performance

Vectorized: pre-compute `activated_matrix` of shape `(50, n_residues)`, then for each k-mer's index set compute TP via column slicing. Estimated <5 minutes for all 5120 features on CPU.

### Files to Modify/Create

| File | Action |
|------|--------|
| `proteinlens/analysis/feature_pipeline/config.py` | Add motif config params + path property |
| `proteinlens/analysis/feature_pipeline/motif_enrichment.py` | **New** — core enrichment module |
| `scripts/run_feature_pipeline.py` | Register Stage 7 |
| `proteinlens/viz/index_builder.py` | Read motif summary, add columns |
| `proteinlens/viz/api.py` | Add `/api/feature/{id}/motif` endpoint |
| `proteinlens/viz/static/js/homepage.js` | Add "Motif F1" and "Best Motif" columns |
| `tests/test_feature_pipeline/test_motif_enrichment.py` | **New** — unit + integration tests |

### Checklist

#### 7.1 Config: `proteinlens/analysis/feature_pipeline/config.py`

- [ ] **7.1.1** Add `motif_kmer_k: int = 3` parameter with docstring
- [ ] **7.1.2** Add `motif_min_count: int = 5` parameter with docstring
- [ ] **7.1.3** Add `motif_f1_threshold_steps: int = 50` parameter with docstring
- [ ] **7.1.4** Add `motif_top_n: int = 10` parameter — number of top motifs to keep per feature
- [ ] **7.1.5** Add `motif_enrichment_dir` path property returning `self.output_dir / "motif_enrichment"` with `mkdir(parents=True, exist_ok=True)`, same pattern as `interpro_enrichment_dir` (line 228)

#### 7.2 Core module: `proteinlens/analysis/feature_pipeline/motif_enrichment.py`

- [ ] **7.2.1** Create module with docstring explaining Stage 7. Follow docstring style of `interpro_enrichment.py` lines 1-27.

- [ ] **7.2.2** Implement `_extract_kmers_with_activations(sequence, activations, k)`.
  - For each position `i` from `k//2` to `len(sequence) - k//2`, extract k-mer `sequence[i - k//2 : i + k//2 + 1]` and pair with `activations[i]`.
  - Skip positions where any character in the k-mer is not in `ACDEFGHIKLMNPQRSTVWY`.
  - Assert `len(sequence) == len(activations)` at entry.

- [ ] **7.2.3** Implement `_pool_proteins_for_feature(feature_data)`.
  - Extract all proteins from `top_sequences` and all bins in `activation_bins`.
  - Return `(accession, sequence, per_residue_activations)` tuples.
  - Deduplicate by accession. Skip entries where `per_residue_activations is None`.

- [ ] **7.2.4** Implement vectorized F1 computation.
  - Pre-compute `thresholds = np.linspace(0, feat_max, n_steps + 1)[1:]`.
  - Pre-compute `activated_matrix = all_activations[None, :] > thresholds[:, None]` — shape `(n_thresholds, N)`.
  - Pre-compute `n_activated = activated_matrix.sum(axis=1)`.
  - For each k-mer with indices `idx` and `len(idx) >= min_count`:
    - `tp = activated_matrix[:, idx].sum(axis=1)`
    - `fp = len(idx) - tp`, `fn = n_activated - tp`
    - Compute precision, recall, F1 vectorized over thresholds.
    - Pick threshold with max F1.

- [ ] **7.2.5** Implement `_analyze_feature(feature_data, feat_max, config) -> dict | None`.
  1. Pool proteins → extract k-mers → build `kmer_indices` dict.
  2. Run vectorized F1 computation.
  3. Return result dict or `None` if no eligible k-mers.

- [ ] **7.2.6** Implement `run_motif_enrichment(config) -> None`.
  - Load `feature_max_activations.npy`. Iterate features with `tqdm`.
  - Skip `feat_max == 0`. Resumable: skip if output JSON exists.
  - Write per-feature JSON + summary.json. Log wandb metrics.

- [ ] **7.2.7** Per-feature JSON schema:
  ```json
  {
    "feature_id": 42,
    "feature_max_activation": 1.5,
    "n_proteins_evaluated": 56,
    "n_total_residues": 23320,
    "n_unique_kmers_tested": 1847,
    "k": 3,
    "top_motifs": [
      {
        "motif": "GKT",
        "best_f1": 0.723,
        "best_threshold": 0.45,
        "best_threshold_normalized": 0.30,
        "precision_at_best": 0.68,
        "recall_at_best": 0.77,
        "n_occurrences": 34,
        "n_true_positives": 26,
        "n_false_positives": 12,
        "n_false_negatives": 8,
        "interpretation": "Motif GKT predicts activation > 0.45 (30% of max) with F1=0.72"
      }
    ]
  }
  ```

- [ ] **7.2.8** Summary JSON schema:
  ```json
  {
    "n_features_analyzed": 4800,
    "n_features_skipped": 320,
    "k": 3,
    "features": {
      "42": {
        "best_motif": "GKT",
        "best_motif_f1": 0.723,
        "n_kmers_tested": 1847
      }
    }
  }
  ```

#### 7.3 Pipeline orchestration: `scripts/run_feature_pipeline.py`

- [ ] **7.3.1** Add `_run_stage_motif_enrichment(config, state)` following pattern of `_run_stage_geometry_residue_enrichment` (line 215-227).
- [ ] **7.3.2** Append `("motif_enrichment", _run_stage_motif_enrichment)` to `STAGES` list after `geometry_residue_enrichment` (line 243).

#### 7.4 Index builder: `proteinlens/viz/index_builder.py`

- [ ] **7.4.1** Load `motif_enrichment/summary.json` using same pattern as InterPro summary load. Add fallback scanner `_scan_motif_files()`.
- [ ] **7.4.2** Add two keys to the per-feature row dict:
  - `"motif_best_f1"` from `summary.features[fid].best_motif_f1`
  - `"motif_best_name"` from `summary.features[fid].best_motif`
- [ ] **7.4.3** Add `"motif_count"` to `build_pipeline_status()`.

#### 7.5 API: `proteinlens/viz/api.py`

- [ ] **7.5.1** Add `GET /api/feature/{feature_id}/motif` endpoint, serving `motif_enrichment/{feature_id:04d}.json`. Follow pattern of `get_feature_interpro` (line 83-93).

#### 7.6 Frontend: `proteinlens/viz/static/js/homepage.js`

- [ ] **7.6.1** Add "Motif F1" column: `field: "motif_best_f1"`, `headerName: "Motif F1"`, `width: 110`, `valueFormatter: nullFormatter(3)`, `cellStyle: greenScale(1.0)`, `comparator: nullBottomComparator`, `filter: "agNumberColumnFilter"`.
- [ ] **7.6.2** Add "Best Motif" column: `field: "motif_best_name"`, `headerName: "Best Motif"`, `width: 110`, `filter: "agTextColumnFilter"`.

#### 7.7 Tests: `tests/test_feature_pipeline/test_motif_enrichment.py`

- [ ] **7.7.1** Test `_extract_kmers_with_activations`: `"ACDEF"` with k=3 → 3 k-mers at center positions 1,2,3.
- [ ] **7.7.2** Test edge case: sequence shorter than k returns empty list.
- [ ] **7.7.3** Test `_pool_proteins_for_feature`: deduplication when same accession in top and bin.
- [ ] **7.7.4** Test perfect separation: motif "AAA" at only high-activation positions → F1=1.0.
- [ ] **7.7.5** Test min_count filtering: motif appearing 3 times with min_count=5 is excluded.
- [ ] **7.7.6** End-to-end: synthetic data in tmp_path, run `run_motif_enrichment`, verify output files and schema.
- [ ] **7.7.7** Verify summary.json structure; features with no eligible k-mers absent.

### Verification

```bash
# Run motif enrichment on feature_data_cluster
conda run -n interplm python -c "
from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.motif_enrichment import run_motif_enrichment
config = PipelineConfig(output_dir='feature_data_cluster')
run_motif_enrichment(config)
"

# Check output
ls feature_data_cluster/motif_enrichment/ | head -20
python -c "import json; d=json.load(open('feature_data_cluster/motif_enrichment/summary.json')); print(f'Analyzed: {d[\"n_features_analyzed\"]}, Skipped: {d[\"n_features_skipped\"]}')"

# Run tests
conda run -n interplm pytest tests/test_feature_pipeline/test_motif_enrichment.py -v

# Verify viz integration
conda run -n interplm python -m proteinlens.viz.server --data-dir feature_data_cluster --port 8080
# Open http://localhost:8080 and verify "Motif F1" and "Best Motif" columns appear
```
