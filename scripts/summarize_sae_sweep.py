#!/usr/bin/env python
"""Print the finished runs of an SAE sweep project side by side.

Selection of the SAE for the annotation pipeline is a judgement call across
% loss recovered, L0 sparsity, variance explained and the dead-feature fraction
(and how these compare with the ESM2-35M reference), so this only tabulates;
it does not pick.

Usage:
    python scripts/summarize_sae_sweep.py --project proteinlens-sae-amplify120m
"""

import argparse
import os

import wandb

REF_35M = {  # esm-35m-sae final summaries (15360 features, 500k steps), for scale
    "cgvpk5vp lr1e-4 l1.05": {"l0": 457.2, "var_explained": 0.945},
    "6i3n3yaf lr3e-4 l1.05": {"l0": 376.8, "var_explained": 0.941},
    "92sr4v8m lr1e-4 l1.08": {"l0": 130.7, "var_explained": 0.901},
    "fjqtwmgu lr3e-4 l1.08": {"l0": 116.7, "var_explained": 0.898},
}


def get(summary, *keys):
    for k in keys:
        v = summary
        for part in k.split("."):
            v = v.get(part) if isinstance(v, dict) else None
        if v is not None:
            return v
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    args = parser.parse_args()

    api = wandb.Api()
    rows = []
    for r in api.runs(f"{args.entity}/{args.project}"):
        if r.job_type == "smoke" or r.config.get("learning_rate") is None:
            continue
        s = r.summary._json_dict
        rows.append((
            r.id, r.name, r.state, r.config.get("learning_rate"), r.config.get("l1_coefficient"),
            r.config.get("hidden_size"), s.get("_step"),
            get(s, "performance/fidelity.pct_loss_recovered", "performance/pct_loss_recovered"),
            get(s, "performance/l0_sparsity"), get(s, "performance/variance_explained"),
            (s.get("_runtime") or 0) / 3600,
        ))

    fmt = "{:<9} {:<18} {:<9} {:>7} {:>5} {:>6} {:>7} {:>9} {:>7} {:>8} {:>6}"
    print(fmt.format("id", "name", "state", "lr", "l1", "dict", "step", "loss_rec%", "L0", "var_exp", "hours"))
    for row in sorted(rows, key=lambda x: (x[3] or 0, x[4] or 0)):
        print(fmt.format(*[("-" if v is None else (f"{v:.3g}" if isinstance(v, float) else v)) for v in row]))

    print("\nESM2-35M L6 reference (final, 15360 features):")
    for k, v in REF_35M.items():
        print(f"  {k}: L0={v['l0']}  var_explained={v['var_explained']}")
    print("\nNo run is auto-selected: weigh loss recovered against L0 and variance explained "
          "(and the smoke-compare run's dead_frac / norm ratio) and pick one.")


if __name__ == "__main__":
    main()
