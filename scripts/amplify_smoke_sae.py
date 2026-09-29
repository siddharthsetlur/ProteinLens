#!/usr/bin/env python
"""Short ReLU-SAE run on AMPLIFY activations: tier-1 gate before the full sweep.

Trains with the same trainer settings as relu_sweep.py (normalize_to_sqrt_d=False,
10% warmup, decay from 80%) for a few thousand steps at one (lr, L1) grid point,
with the AMPLIFY %-loss-recovered eval, and no W&B. Then reports L0, variance
explained and the dead-feature fraction on a held-out activation shard.

AMPLIFY L12 residual-stream norms are ~3.5x ESM2-35M L6's, which weakens a fixed
L1 penalty relative to MSE; this is the check that the sweep's L1 grid does not
give degenerate L0 before committing GPU-days to it.

Usage:
    python scripts/amplify_smoke_sae.py --embd-dir .../layer_12 \
        --eval-seqs eval_sequences.txt --heldout .../shard_4/activations.pt \
        --lr 3e-4 --l1 0.08 --steps 5000 --out /data/smoke/sae_lr3e-4_l1_0.08
"""

import argparse
from pathlib import Path

import torch
import yaml

from proteinlens.sae.inference import load_sae
from proteinlens.train.configs import (
    CheckpointConfig,
    DataloaderConfig,
    ReLUTrainerConfig,
    TrainingRunConfig,
    WandbConfig,
)
from proteinlens.train.fidelity_amplify import AmplifyFidelityConfig
from proteinlens.train.training_run import SAETrainingRun


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--embd-dir", type=Path, required=True)
    parser.add_argument("--eval-seqs", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True,
                        help="activations.pt of a shard not in --embd-dir")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l1", type=float, default=0.08)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--hidden-size", type=int, default=20480)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--model", default="chandar-lab/AMPLIFY_120M")
    args = parser.parse_args()

    run = SAETrainingRun.from_config(TrainingRunConfig(
        dataloader_cfg=DataloaderConfig(plm_embd_dir=args.embd_dir, batch_size=args.batch_size),
        trainer_cfg=ReLUTrainerConfig(
            activation_dim=640,
            dictionary_size=args.hidden_size,
            lr=args.lr,
            l1_penalty=args.l1,
            warmup_steps=int(0.1 * args.steps),
            decay_start=int(0.8 * args.steps),
            steps=args.steps,
            normalize_to_sqrt_d=False,
        ),
        eval_cfg=AmplifyFidelityConfig(
            eval_seq_path=args.eval_seqs,
            model_name=args.model,
            layer_idx=args.layer,
            eval_steps=args.steps,
            eval_batch_size=32,
        ),
        wandb_cfg=WandbConfig(use_wandb=False),
        checkpoint_cfg=CheckpointConfig(save_dir=args.out, save_steps=args.steps, max_ckpts_to_keep=1),
    ))
    run.run()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = load_sae(args.out, device=device)
    x = torch.load(args.heldout)[:200_000].to(device)
    with torch.no_grad():
        f = sae.encode(x)
        r = sae.decode(f)
    l0 = (f > 0).sum(1).float().mean().item()
    var_expl = (1 - ((x - r) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()).item()
    dead = ((f > 0).sum(0) == 0).float().mean().item()

    fidelity = yaml.unsafe_load((args.out / "final_evaluation.yaml").read_text())["fidelity"]
    print(
        f"SMOKE lr={args.lr:g} l1={args.l1:g} steps={args.steps}: "
        f"L0={l0:.1f} var_explained={var_expl:.3f} dead_frac={dead:.3f} "
        f"pct_loss_recovered={float(fidelity['pct_loss_recovered']):.1f}"
    )


if __name__ == "__main__":
    main()
