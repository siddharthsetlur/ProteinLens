#!/usr/bin/env python
"""Compare AMPLIFY-120M L12 activations with ESM2-35M L6 on the same proteins.

The reference for the AMPLIFY SAE sweep is the ESM2-35M run. This script gives
that comparison on identical proteins:

  * residual-stream scale for both models: token-norm mean/std, median per-dim
    |x|, share of squared norm in the largest dimension. A fixed L1 penalty is
    weaker on a larger-scale stream, so this is what the AMPLIFY L1 grid must be
    read against.
  * the published ESM2-35M SAE (cgvpk5vp, pulled from HF) on the ESM2-35M
    activations: L0, variance explained, dead fraction. This is the target the
    5k-step AMPLIFY smoke SAEs (scripts/amplify_smoke_sae.py) are compared to.

Only the 35M SAE (~60 MB) is fetched from HF; the ESM2-35M activations are
recomputed on the fly (the 35M model is tiny), so nothing large is downloaded.

Usage:
    python scripts/compare_amplify_esm35m.py --fasta subset.fasta --n 1000 \
        --out /workspace/smoke/compare.json [--wandb-project ...]
"""

import argparse
import json
import os
from pathlib import Path

import torch
from Bio import SeqIO
from huggingface_hub import snapshot_download

from proteinlens.embedders import ESM, Amplify
from proteinlens.sae.inference import load_sae

ESM35M = "facebook/esm2_t12_35M_UR50D"
ESM35M_LAYER = 6
ESM35M_SAE_REPO = "Sidd2010/proteinlens-sae-esm2-35m"
ESM35M_SAE_PATH = "trained_models/esm2_35m/layer_6/cgvpk5vp"
AMPLIFY = "chandar-lab/AMPLIFY_120M"
AMPLIFY_LAYER = 12


def stream_stats(h: torch.Tensor) -> dict:
    n = h.norm(dim=-1)
    per_dim = h.abs().mean(0)
    top = int(per_dim.argmax())
    return {
        "tokens": int(h.shape[0]),
        "norm_mean": n.mean().item(),
        "norm_std": n.std().item(),
        "per_dim_abs_median": per_dim.median().item(),
        "top_dim": top,
        "top_dim_abs_mean": per_dim[top].item(),
        "top_dim_sq_norm_share": ((h[:, top] ** 2).sum() / (h ** 2).sum()).item(),
    }


def sae_stats(sae, x: torch.Tensor) -> dict:
    with torch.no_grad():
        f = sae.encode(x)
        r = sae.decode(f)
    return {
        "dict_size": int(sae.dict_size),
        "l0": (f > 0).sum(1).float().mean().item(),
        "variance_explained": (1 - ((x - r) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()).item(),
        "dead_frac": ((f > 0).sum(0) == 0).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--wandb-project", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = [str(r.seq) for _, r in zip(range(args.n), SeqIO.parse(args.fasta, "fasta"))]
    print(f"{len(seqs)} proteins, {sum(map(len, seqs))} residues, device {device}")

    result = {"n_proteins": len(seqs), "n_residues": sum(map(len, seqs))}

    amp = Amplify(model_name=AMPLIFY, device=device)
    h_amp = amp.extract_embeddings(seqs, AMPLIFY_LAYER, batch_size=args.batch_size).to(device)
    result["amplify_120m_l12"] = stream_stats(h_amp)
    del amp, h_amp
    torch.cuda.empty_cache() if device == "cuda" else None

    esm = ESM(model_name=ESM35M, device=device)
    h_esm = esm.extract_embeddings(seqs, ESM35M_LAYER, batch_size=args.batch_size).to(device)
    result["esm2_35m_l6"] = stream_stats(h_esm)

    sae_root = snapshot_download(ESM35M_SAE_REPO, allow_patterns=[f"{ESM35M_SAE_PATH}/*"])
    sae = load_sae(Path(sae_root) / ESM35M_SAE_PATH, device=device)
    result["esm2_35m_l6_sae_cgvpk5vp"] = sae_stats(sae, h_esm)

    result["norm_ratio_amplify_over_esm35m"] = (
        result["amplify_120m_l12"]["norm_mean"] / result["esm2_35m_l6"]["norm_mean"]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=os.environ.get("WANDB_ENTITY"),
                         name="smoke-compare-esm35m", job_type="smoke", config=vars(args))
        flat = {f"{k}/{kk}": vv for k, v in result.items() if isinstance(v, dict) for kk, vv in v.items()}
        flat["norm_ratio_amplify_over_esm35m"] = result["norm_ratio_amplify_over_esm35m"]
        run.summary.update(flat)
        run.finish()


if __name__ == "__main__":
    main()
