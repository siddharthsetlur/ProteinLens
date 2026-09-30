#!/usr/bin/env python
"""Progress, ETA and projected cost of a running SAE sweep, from W&B.

For the agent runbook (runpod/AGENT_RUNBOOK_amplify120m_sae.md): one call per
monitoring tick. Per run: step, steps/s, ETA to --steps, latest metrics, GPU
util/memory from the run's system metrics. Overall: remaining wall-clock (the
slowest unfinished run), spend so far and projected total at --price $/h, and
a PASS/WARN/STOP verdict against --budget.

Usage:
    python scripts/sweep_progress.py --project proteinlens-sae-amplify120m \
        --pod-started 2026-10-01T09:00:00Z --price 1.59 --budget 60
"""

import argparse
import os
from datetime import datetime, timezone

import wandb


def parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--steps", type=int, default=500_000, help="steps per run")
    parser.add_argument("--n-runs", type=int, default=4, help="grid points in the sweep")
    parser.add_argument("--pod-started", required=True, help="pod creation time, RFC3339")
    parser.add_argument("--price", type=float, required=True, help="pod $/h")
    parser.add_argument("--budget", type=float, required=True, help="hard budget for this pod, $")
    parser.add_argument("--warn-frac", type=float, default=0.8)
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    elapsed_h = (now - parse_iso(args.pod_started)).total_seconds() / 3600
    spend = elapsed_h * args.price

    api = wandb.Api()
    runs = [r for r in api.runs(f"{args.entity}/{args.project}")
            if r.job_type != "smoke" and r.config.get("learning_rate") is not None]

    print(f"{'run':<9} {'state':<9} {'lr':>7} {'l1':>5} {'step':>7} {'steps/s':>8} {'eta_h':>6} "
          f"{'L0':>7} {'var_exp':>8} {'loss_rec':>8} {'gpu%':>5} {'gpu_GB':>7}")
    remaining_h, running, done = 0.0, 0, 0
    for r in runs:
        s = r.summary._json_dict
        step, rt = s.get("_step") or 0, s.get("_runtime") or 0
        rate = step / rt if rt else 0.0
        eta = (args.steps - step) / rate / 3600 if rate and r.state == "running" else 0.0
        if r.state == "running":
            running += 1
            remaining_h = max(remaining_h, eta)
        elif r.state == "finished":
            done += 1
        fid = s.get("performance/fidelity") or {}
        loss_rec = fid.get("pct_loss_recovered") if isinstance(fid, dict) else s.get("performance/fidelity.pct_loss_recovered")
        try:
            sysm = r.history(stream="system", pandas=True).tail(5)
            gpu = sysm["system.gpu.0.gpu"].mean() if "system.gpu.0.gpu" in sysm else float("nan")
            mem = sysm["system.gpu.0.memoryAllocatedBytes"].max() / 1e9 if "system.gpu.0.memoryAllocatedBytes" in sysm else float("nan")
        except Exception:
            gpu = mem = float("nan")
        f = lambda v, fmt: "-" if v is None else format(v, fmt)
        print(f"{r.id:<9} {r.state:<9} {r.config.get('learning_rate'):>7g} {r.config.get('l1_coefficient'):>5g} "
              f"{step:>7} {rate:>8.2f} {eta:>6.2f} {f(s.get('performance/l0_sparsity'), '7.1f')} "
              f"{f(s.get('performance/variance_explained'), '8.3f')} {f(loss_rec, '8.1f')} {gpu:>5.0f} {mem:>7.1f}")

    not_started = max(0, args.n_runs - len(runs))
    # Runs not yet started will take about as long as the slowest running one, after it.
    if not_started and running:
        remaining_h += not_started * remaining_h / running
    projected = spend + remaining_h * args.price
    verdict = ("STOP" if spend >= args.budget or projected > args.budget
               else "WARN" if projected > args.warn_frac * args.budget else "PASS")
    print(f"\nruns: {done} finished, {running} running, {not_started} not started | "
          f"pod elapsed {elapsed_h:.2f} h, spend ${spend:.2f}, remaining ~{remaining_h:.2f} h, "
          f"projected ${projected:.2f} vs budget ${args.budget:.2f} -> {verdict}")
    if verdict == "STOP":
        print("ACTION: stop the pod now (pod-action stop), then report to the user.")
    raise SystemExit({"PASS": 0, "WARN": 0, "STOP": 2}[verdict])


if __name__ == "__main__":
    main()
