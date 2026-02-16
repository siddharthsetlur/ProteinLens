from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from pdb_plotter import *
import yaml
from geometry.compute_geometric_features import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# ---------- helpers to load/search batches ----------

def list_batch_paths(batch_dir: str | Path, first: int = 0, last: int = 21) -> list[Path]:
    """Return [batch-0.yaml ... batch-21.yaml] that actually exist, in order."""
    batch_dir = Path(batch_dir)
    return [p for i in range(first, last + 1) if (p := batch_dir / f"batch_{i}.yaml").is_file()]

@lru_cache(maxsize=None)
def _load_yaml_cached(path: Path) -> dict:
    """Load a YAML file once (cached)."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def _normalize_pdb_text(pdb_text: str) -> str:
    # Guard against Windows newlines or soft-wrap escapes in some dumps
    return pdb_text.replace("\r\n", "\n").replace("\\\n", "")

def find_pdb_in_batches(entry_key: str, batch_paths: list[Path]) -> str:
    """Return the PDB text for 'entry_key' by scanning the provided batch files."""
    for path in batch_paths:
        data = _load_yaml_cached(path)
        if entry_key in data:
            val = data[entry_key]
            if isinstance(val, dict) and "pdb" in val and val["pdb"]:
                return _normalize_pdb_text(val["pdb"])
            raise ValueError(f"Entry '{entry_key}' found in {path.name} but has no 'pdb' text.")
    raise KeyError(f"Entry '{entry_key}' not found in any batch YAML.")

# ---------- group loader (your '0: - P20937 ...' file) ----------

def load_groups(groups_yaml_path: str | Path) -> dict[int, list[str]]:
    """
    groups
    """
    data = yaml.safe_load(Path(groups_yaml_path).read_text(encoding="utf-8")) or {}
    groups: dict[int, list[str]] = {}
    for k, v in data.items():
        if isinstance(k, str) and k.isdigit():
            k = int(k)
        groups[int(k)] = list(v or [])
    return groups

# ---------- main: plot all groups using your ca_backbone(...) ----------

def plot_groups_backbones(
    groups_yaml_path: str | Path,
    batch_dir: str | Path,
    chain_id: str | None = "A",
    first_batch: int = 0,
    last_batch: int = 21,
    skip_missing: bool = True,
):
    """
    For each group (0..N) and each accession in that group, find the PDB text
    in any batch-x.yaml and plot the Ca backbone using ca_backbone(...).

    - chain_id: pass None to take the first chain, or " " for blank chain IDs.
    - skip_missing: if False, raise when an accession can't be found.
    """
    groups = load_groups(groups_yaml_path)
    batch_paths = list_batch_paths(batch_dir, first_batch, last_batch)
    full_dataset_wr = []
    full_dataset_tor = []
    full_dataset_cur = []
    full_coords = []

    if not batch_paths:
        raise FileNotFoundError(f"No batch YAMLs found in {batch_dir} (expected batch-{first_batch}..batch-{last_batch}.yaml)")

    for gid in range(0, 500):
        print(gid)
        datawr = []
        datacur = []
        datator = []
        coords = []
        accessions = groups[gid]
        if not accessions:
            print(f"[group {gid}] (empty)")
            continue

        print(f"[group {gid}] {len(accessions)} accession(s)")
        for acc in accessions:
            try:
                pdb_text = find_pdb_in_batches(acc, tuple(batch_paths))  # tuple so lru_cache keys nicely
                print(f"  - plotting {acc}")
                ca = ca_backbone(pdb_text, chain_id=chain_id)
                wr_d= writhe(ca, ca)
                wr = np.sum(wr_d)
                cur = average_curvature(ca)
                tor = average_torsion(ca)
                datawr.append(wr)
                datator.append(tor)
                datacur.append(cur)

            except Exception as e:
                msg = f"  ! {acc}: {e}"
                if skip_missing:
                    print(msg)
                    continue
                raise
        full_dataset_wr.append(datawr)
        full_dataset_cur.append(datacur)
        full_dataset_tor.append(datator)

    return full_dataset_wr, full_dataset_cur, full_dataset_tor


full_dataset_wr, full_dataset_cur, full_dataset_tor = plot_groups_backbones(
    groups_yaml_path="Per_feature_max_examples.yaml",
    batch_dir="results",
    chain_id=None,          # or None for first chain
    first_batch=0,
    last_batch=21,
    skip_missing=True
)

for x in range(0, len(full_dataset_wr)):
    plt.hist(full_dataset_wr[x], label=f"{x}")

plt.show()

for x in range(0, len(full_dataset_cur)):
    plt.hist(full_dataset_cur[x], label=f"{x}")

plt.show()

for x in range(0, len(full_dataset_tor)):
    plt.hist(full_dataset_tor[x], label=f"{x}")

plt.show()