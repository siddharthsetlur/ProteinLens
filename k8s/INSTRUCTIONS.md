# ProteinLens Feature Pipeline -- Kubernetes Deployment

## Prerequisites

- `kubectl` configured for your cluster with `kubectl pvcsync` plugin installed
- Docker installed and logged in to your registry
- The `trained_models/fiery-sweep/` directory available locally (contains `config.yaml`, `ae.pt`, etc.)

Replace `<REGISTRY>` with your Docker registry and `<NAMESPACE>` with your Kubernetes namespace throughout.

---

## 1. Build and Push the Docker Image

Build from the **project root** (not the `k8s/` directory), using the pipeline-specific Dockerfile:

```bash
cd <PROJECT_ROOT>
docker build -t <REGISTRY>/proteinlens-pipeline:latest -f k8s/Dockerfile.pipeline .
docker push <REGISTRY>/proteinlens-pipeline:latest
```

---

## 2. Create the PVC

```bash
kubectl -n <NAMESPACE> create -f k8s/pvc.yaml
```

Verify it was created:

```bash
kubectl -n <NAMESPACE> get pvc proteinlens-pipeline-pvc
```

---

## 3. Transfer SAE Model Files to the PVC

The pipeline expects the trained SAE at `/data/trained_models/fiery-sweep/` on the PVC.

### 3a. Start the pvcsync tunnel

```bash
kubectl pvcsync proteinlens-pipeline-pvc up
```

Note the **PORTNUMBER** from the output.

### 3b. Upload the model files via rsync

```bash
rsync -rvP \
  -e 'ssh -J <USER>@<GATEWAY_HOST>,<USER>@<INTERNAL_HOST> -p PORTNUMBER' \
  <PROJECT_ROOT>/trained_models/fiery-sweep/ \
  root@localhost:/data/trained_models/fiery-sweep/
```

Replace `PORTNUMBER` with the port from step 3a.

---

## 4. Launch the Job

```bash
kubectl -n <NAMESPACE> create -f k8s/job.yaml
```

Since the job uses `generateName`, note the actual job name from the output (e.g., `proteinlens-pipeline-abc12`).

---

## 5. Monitor the Job

### Find the pod

```bash
kubectl -n <NAMESPACE> get pods | grep proteinlens-pipeline
```

### Stream logs

```bash
kubectl -n <NAMESPACE> logs -f <POD_NAME>
```

### Check job status

```bash
kubectl -n <NAMESPACE> describe job <JOB_NAME>
```

---

## 6. Copy Results Back from the PVC

Once the job completes, the pipeline output will be at `/data/feature_data/` on the PVC.

### 6a. Start the pvcsync tunnel (if not already running)

```bash
kubectl pvcsync proteinlens-pipeline-pvc up
```

Note the **PORTNUMBER** from the output.

### 6b. Download results via rsync

```bash
rsync -rvP \
  -e 'ssh -J <USER>@<GATEWAY_HOST>,<USER>@<INTERNAL_HOST> -p PORTNUMBER' \
  root@localhost:/data/feature_data/ \
  <PROJECT_ROOT>/feature_data_cluster/
```

---

## 7. Cleanup

### Delete completed jobs

```bash
kubectl -n <NAMESPACE> get jobs | grep proteinlens-pipeline
kubectl -n <NAMESPACE> delete job <JOB_NAME>
```

### Delete the PVC (when no longer needed)

WARNING: This permanently deletes all data on the volume.

```bash
kubectl -n <NAMESPACE> delete pvc proteinlens-pipeline-pvc
```

---

## Running Individual Stages

To run only a specific stage (e.g., `survey`), edit the `args` field in `job.yaml`:

```yaml
args:
- python scripts/run_feature_pipeline.py --sae-dir /data/trained_models/fiery-sweep --output-dir /data/feature_data --device cuda --stage survey
```

The pipeline has built-in checkpointing, so if a job fails partway through and you relaunch, it will skip already-completed stages automatically.

## Running with Limited Proteins (Testing)

Add `--max-proteins 50` to the args for a quick test run:

```yaml
args:
- python scripts/run_feature_pipeline.py --sae-dir /data/trained_models/fiery-sweep --output-dir /data/feature_data --device cuda --max-proteins 50
```
