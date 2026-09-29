#!/usr/bin/env python
"""Tier-1 cluster check for AMPLIFY before the full run.

On CUDA the AMPLIFY remote code switches from torch SDPA to xformers
memory_efficient_attention, which the local CPU tests never exercise. This
script embeds the first N proteins of a FASTA and checks:

1. GPU single-sequence (survey/collection path, no mask) vs CPU single-sequence.
2. GPU batched (extraction path: padded to a multiple of 8, additive mask) vs
   GPU single-sequence.
3. Throughput of both GPU paths, to size the full survey and extraction.

Exits non-zero if either comparison exceeds the tolerance.

Usage:
    python scripts/check_amplify_gpu.py --fasta subset.fasta --n 200 --layer 12
"""

import argparse
import sys
import time

import torch
from Bio import SeqIO

from proteinlens.embedders import Amplify


def embed_single(emb, seqs, layer):
    return [torch.from_numpy(emb.embed_single_sequence(s, layer)) for s in seqs]


def split(flat, seqs):
    return list(torch.split(flat, [len(s) for s in seqs]))


def max_rel_diff(a_list, b_list):
    """Max over proteins of max|a-b| / max|b| (activations have O(100) outlier dims)."""
    return max(((a - b).abs().max() / b.abs().max()).item() for a, b in zip(a_list, b_list))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default="chandar-lab/AMPLIFY_120M")
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    seqs = [str(r.seq) for _, r in zip(range(args.n), SeqIO.parse(args.fasta, "fasta"))]
    n_res = sum(len(s) for s in seqs)
    print(f"{len(seqs)} proteins, {n_res} residues, layer {args.layer}")

    gpu = Amplify(model_name=args.model, device="cuda")
    print(f"revision {gpu.revision}")

    torch.cuda.synchronize()
    t0 = time.time()
    gpu_single = embed_single(gpu, seqs, args.layer)
    torch.cuda.synchronize()
    t_single = time.time() - t0

    t0 = time.time()
    gpu_batched = split(
        gpu.extract_embeddings(seqs, args.layer, batch_size=args.batch_size), seqs
    )
    torch.cuda.synchronize()
    t_batched = time.time() - t0

    del gpu
    torch.cuda.empty_cache()
    cpu = Amplify(model_name=args.model, device="cpu")
    cpu_single = embed_single(cpu, seqs, args.layer)

    d_gpu_cpu = max_rel_diff(gpu_single, cpu_single)
    d_batch = max_rel_diff(gpu_batched, gpu_single)
    abs_gpu_cpu = max((a - b).abs().max().item() for a, b in zip(gpu_single, cpu_single))

    print(f"GPU single vs CPU single : max rel diff {d_gpu_cpu:.2e} (max abs {abs_gpu_cpu:.2e})")
    print(f"GPU batched vs GPU single: max rel diff {d_batch:.2e}")
    print(f"GPU single : {len(seqs) / t_single:.1f} proteins/s, {n_res / t_single:.0f} residues/s")
    print(f"GPU batched: {len(seqs) / t_batched:.1f} proteins/s, {n_res / t_batched:.0f} residues/s "
          f"(batch {args.batch_size})")

    ok = d_gpu_cpu < args.tol and d_batch < args.tol
    print("PASS" if ok else f"FAIL (tol {args.tol})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
