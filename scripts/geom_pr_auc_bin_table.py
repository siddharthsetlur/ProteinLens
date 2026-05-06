"""Cross-layer table of geom_pr_auc bin occupancy.

Bins are aligned with pipeline thresholds:
  - < 0.10        : ~chance (PR-AUC baseline ≈ activated-residue prevalence)
  - 0.10 – 0.30   : weak lift, below the is_geometry_primary cutoff (0.3)
  - 0.30 – 0.60   : strong (passes is_geometry_primary)
  - ≥ 0.60        : very strong (top-decile in layer 4)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/s2721407/Desktop/ProteinLens/trained_models")
LAYERS = [
    ("layer_2", "firm-sweep-3"),
    ("layer_4", "frosty-sweep-15"),
    ("layer_6", "major-sweep-15"),
]
SAE_SIZE = 10240
BIN_EDGES = [0.0, 0.10, 0.30, 0.60, 1.0001]
BIN_LABELS = ["<0.10", "0.10–0.30", "0.30–0.60", "≥0.60"]
ALL_LABELS = ["dead/unfit"] + BIN_LABELS


def load_pr_auc(layer_dir: Path) -> np.ndarray:
    with (layer_dir / "analysis" / "geometry_primary_analysis.json").open() as f:
        d = json.load(f)
    vals = []
    for v in d["features"].values():
        prauc = v.get("geom_pr_auc")
        if prauc is None:
            continue
        vals.append(float(prauc))
    return np.array(vals)


def main() -> None:
    rows = []
    for layer, sweep in LAYERS:
        prauc = load_pr_auc(ROOT / layer / sweep)
        n_eval = len(prauc)
        n_dead = SAE_SIZE - n_eval
        bin_counts, _ = np.histogram(prauc, bins=BIN_EDGES)
        all_counts = np.concatenate(([n_dead], bin_counts))
        pcts = 100.0 * all_counts / SAE_SIZE
        rows.append((layer, sweep, n_eval, n_dead, all_counts, pcts, prauc))

    layer_w = max(len(r[0]) for r in rows)
    sweep_w = max(len(r[1]) for r in rows)

    print(f"All percentages computed over the full SAE dictionary (n = {SAE_SIZE}).\n")
    header = (
        f"{'layer':<{layer_w}}  {'sweep':<{sweep_w}}  " + "  ".join(f"{lbl:>11}" for lbl in ALL_LABELS)
    )
    print(header)
    print("-" * len(header))
    for layer, sweep, _, _, _, pcts, _ in rows:
        cells = "  ".join(f"{p:>10.2f}%" for p in pcts)
        print(f"{layer:<{layer_w}}  {sweep:<{sweep_w}}  {cells}")

    print()
    print("Raw counts (out of 10240):")
    print(f"{'layer':<{layer_w}}  " + "  ".join(f"{lbl:>11}" for lbl in ALL_LABELS))
    for layer, _, _, _, all_counts, _, _ in rows:
        cells = "  ".join(f"{c:>11d}" for c in all_counts)
        print(f"{layer:<{layer_w}}  {cells}")

    print()
    print("Mean / median geom_pr_auc (over evaluated features only):")
    for layer, _, n_eval, _, _, _, prauc in rows:
        print(f"  {layer} (n_eval={n_eval}): mean={prauc.mean():.4f}  "
              f"median={np.median(prauc):.4f}  max={prauc.max():.4f}")

    out_csv = Path("/home/s2721407/Desktop/ProteinLens/scripts/geom_pr_auc_bin_table.csv")
    with out_csv.open("w") as f:
        f.write(
            "layer,sweep,sae_size,n_evaluated,n_dead,"
            + ",".join(f"pct_{lbl}" for lbl in ALL_LABELS) + "\n"
        )
        for layer, sweep, n_eval, n_dead, _, pcts, _ in rows:
            f.write(
                f"{layer},{sweep},{SAE_SIZE},{n_eval},{n_dead},"
                + ",".join(f"{p:.3f}" for p in pcts) + "\n"
            )
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
