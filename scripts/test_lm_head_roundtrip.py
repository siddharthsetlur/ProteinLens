#!/usr/bin/env python
"""
Null test: ESM2 lm_head roundtrip (batch mode)
===============================================

  Input S → ESM2 → lm_head logits → argmax → S'

Reports sequence identity between S and S' across many proteins.
Optionally folds both with ESMFold and compares pLDDT.

Usage:
  # Run on built-in 50-protein benchmark:
  python scripts/test_lm_head_roundtrip.py --batch

  # Single accession:
  python scripts/test_lm_head_roundtrip.py --accession P00698

  # Comma-separated list:
  python scripts/test_lm_head_roundtrip.py --accessions P00698,P62988,P02144

  # File with one accession per line:
  python scripts/test_lm_head_roundtrip.py --accessions-file my_list.txt

  # Also fold with ESMFold (slow; loads ~3 GB model):
  python scripts/test_lm_head_roundtrip.py --batch --fold

  # Save CSV summary:
  python scripts/test_lm_head_roundtrip.py --batch --output-dir results/roundtrip
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.intervene_and_fold import (
    fetch_sequence,
    fold_sequence,
    load_esmfold,
    logits_to_sequence,
)

DEFAULT_ESM_MODEL = "facebook/esm2_t6_8M_UR50D"

# ── Built-in benchmark set ─────────────────────────────────────────────────────
# 50 diverse, well-characterised human/model-organism proteins, ~50–700 aa.
# Covers enzymes, signalling, transport, immune, structural, and metabolic roles.
BENCHMARK_ACCESSIONS = [
    # Small / model proteins
    "P62988",  # Ubiquitin                      76 aa
    "P62805",  # Histone H4                    102 aa
    "P01308",  # Insulin (preproinsulin)        110 aa
    "P61769",  # Beta-2-microglobulin          119 aa
    "P00698",  # Lysozyme C (chicken)          129 aa
    "P00167",  # Cytochrome b5                 134 aa
    "Q16695",  # Histone H3.3                  136 aa
    "P69905",  # Hemoglobin alpha              142 aa
    "P68871",  # Hemoglobin beta               147 aa
    "P00441",  # SOD1                          154 aa
    "P02144",  # Myoglobin                     154 aa
    # Immune / signalling
    "P61586",  # RhoA                          193 aa
    "P00492",  # HPRT1                         218 aa
    "P01375",  # TNF-alpha                     233 aa
    "P10415",  # BCL-2                         239 aa
    "P00760",  # Trypsin (bovine)              247 aa
    "P07477",  # Trypsin-1 (human)             247 aa
    "P07327",  # ADH1A                         374 aa (alcohol dehydrogenase)
    "P00491",  # PNP                           289 aa
    "P60174",  # Triosephosphate isomerase     286 aa
    "P62136",  # PP1-alpha                     330 aa
    "P00338",  # LDHA                          332 aa
    "P04406",  # GAPDH                         335 aa
    "P00480",  # OTC                           354 aa
    "P04637",  # p53                           393 aa
    "P07550",  # Beta-2 adrenergic receptor    413 aa
    "P01009",  # Alpha-1-antitrypsin           418 aa
    "P00558",  # PGK1                          417 aa
    "P18031",  # PTP1B                         435 aa
    "P68104",  # EEF1A1 (EF-Tu)               462 aa
    # Metabolic enzymes
    "P00352",  # ALDH1A1                       501 aa
    "P00390",  # Glutathione reductase         522 aa
    "P00367",  # Glutamate dehydrogenase       558 aa
    "P02769",  # Serum albumin (bovine)        607 aa
    "P02787",  # Transferrin                   698 aa
    # Kinases / phosphatases
    "P06493",  # CDK1                          297 aa
    "P24941",  # CDK2                          298 aa
    "P27361",  # ERK2 (MAPK1)                 360 aa
    "P28482",  # ERK1 (MAPK3)                 379 aa
    "P31749",  # AKT1                          480 aa
    # Chaperones / folding
    "P08107",  # HSP70 (HSPA1A)               641 aa
    "P07900",  # HSP90-alpha                   732 aa  (truncated to 1024 if needed)
    # Proteases
    "P00734",  # Prothrombin                   622 aa
    "P00742",  # Factor X                      488 aa
    # Transcription / DNA repair
    "P15056",  # BRAF                          766 aa  (long but classic)
    "P00533",  # EGFR                         1210 aa  (will be truncated to 1024)
    # Structural
    "P02452",  # Collagen alpha-1(I)          1464 aa  (will be truncated)
    "P60709",  # Actin (beta)                  375 aa
    "P68133",  # Actin (alpha skeletal)        377 aa
    "P10809",  # HSP60 (HSPD1)                573 aa
]


# ══════════════════════════════════════════════════════════════════
#  Core roundtrip logic (single sequence)
# ══════════════════════════════════════════════════════════════════

def roundtrip_sequence(seq, esm, tok, device, max_len=1024):
    """Run ESM2 → lm_head → argmax for one sequence.

    Returns (seq_prime, n_same, seq_len).
    """
    if len(seq) > max_len:
        seq = seq[:max_len]

    seq_len = len(seq)
    inputs = tok(seq, return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = esm(**inputs).logits  # (1, L+2, vocab)

    seq_prime = logits_to_sequence(logits, tok, seq_len, temperature=0.0)
    n_same = sum(a == b for a, b in zip(seq, seq_prime))
    return seq_prime, n_same, seq_len


# ══════════════════════════════════════════════════════════════════
#  Batch runner
# ══════════════════════════════════════════════════════════════════

def run_batch(accessions, esm, tok, device, fold=False, out_dir=None):
    """Run roundtrip on a list of accessions. Returns list of result dicts."""
    fold_tok = fold_mdl = None
    if fold:
        fold_device = "cuda" if torch.cuda.is_available() else "cpu"
        fold_tok, fold_mdl = load_esmfold(fold_device)

    results = []

    col_w = 12
    header = (
        f"{'Accession':<{col_w}}  {'Len':>5}  {'Identity':>10}  {'Mutations':>10}"
        + ("  {'pLDDT(S)':>10}  {'pLDDT(S')':>10}  {'ΔPLDDT':>8}" if fold else "")
    )
    print(f"\n{header}")
    print("─" * len(header))

    for acc in accessions:
        row = {"accession": acc, "error": None}
        try:
            seq = fetch_sequence(acc)
            seq_prime, n_same, seq_len = roundtrip_sequence(seq, esm, tok, device)
            pct = 100.0 * n_same / seq_len
            row.update({"seq_len": seq_len, "n_same": n_same, "identity_pct": pct,
                        "n_mut": seq_len - n_same, "seq": seq, "seq_prime": seq_prime})

            line = f"{acc:<{col_w}}  {seq_len:>5}  {pct:>9.1f}%  {seq_len - n_same:>10}"

            if fold:
                orig_pdb, orig_plddt = fold_sequence(seq, fold_tok, fold_mdl)
                rt_pdb, rt_plddt = fold_sequence(seq_prime, fold_tok, fold_mdl)
                delta = rt_plddt - orig_plddt
                row.update({"plddt_orig": orig_plddt, "plddt_rt": rt_plddt, "plddt_delta": delta})
                line += f"  {orig_plddt:>10.2f}  {rt_plddt:>10.2f}  {delta:>+8.2f}"

                if out_dir:
                    sub = out_dir / acc
                    sub.mkdir(parents=True, exist_ok=True)
                    (sub / "original.pdb").write_text(orig_pdb)
                    (sub / "roundtrip.pdb").write_text(rt_pdb)

            print(line)

        except Exception as e:
            row["error"] = str(e)
            print(f"{acc:<{col_w}}  ERROR: {e}")

        results.append(row)

    return results


def print_summary(results):
    good = [r for r in results if not r.get("error")]
    if not good:
        print("\nNo successful results.")
        return

    identities = [r["identity_pct"] for r in good]
    print(f"\n{'─' * 50}")
    print(f"  Summary ({len(good)}/{len(results)} proteins succeeded)")
    print(f"  Mean identity:   {sum(identities)/len(identities):.1f}%")
    print(f"  Min identity:    {min(identities):.1f}%  ({good[identities.index(min(identities))]['accession']})")
    print(f"  Max identity:    {max(identities):.1f}%  ({good[identities.index(max(identities))]['accession']})")

    if "plddt_delta" in good[0]:
        deltas = [r["plddt_delta"] for r in good]
        print(f"  Mean |ΔpLDDT|:  {sum(abs(d) for d in deltas)/len(deltas):.2f}")
    print(f"{'─' * 50}\n")


def save_csv(results, path):
    keys = ["accession", "seq_len", "n_same", "n_mut", "identity_pct",
            "plddt_orig", "plddt_rt", "plddt_delta", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"CSV saved to {path}")


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="ESM2 lm_head roundtrip null test (batch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--batch", action="store_true",
                     help="Run on built-in 50-protein benchmark set")
    inp.add_argument("--accession", default=None,
                     help="Single UniProt accession (e.g. P00698)")
    inp.add_argument("--accessions", default=None,
                     help="Comma-separated accessions (e.g. P00698,P62988)")
    inp.add_argument("--accessions-file", default=None,
                     help="File with one accession per line")

    p.add_argument("--esm-model", default=DEFAULT_ESM_MODEL,
                   help=f"ESM2 model (default: {DEFAULT_ESM_MODEL})")
    p.add_argument("--fold", action="store_true",
                   help="Also fold with ESMFold and compare pLDDT (slow)")
    p.add_argument("--output-dir", default=None,
                   help="Save CSV summary (and PDBs if --fold) here")

    args = p.parse_args()

    # ── Resolve accession list ─────────────────────────────────────
    if args.batch:
        accessions = BENCHMARK_ACCESSIONS
    elif args.accession:
        accessions = [args.accession]
    elif args.accessions:
        accessions = [a.strip() for a in args.accessions.split(",") if a.strip()]
    else:
        lines = Path(args.accessions_file).read_text().splitlines()
        accessions = [l.strip() for l in lines if l.strip() and not l.startswith("#")]

    # ── Load ESM2 ─────────────────────────────────────────────────
    from transformers import AutoTokenizer, EsmForMaskedLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading ESM2: {args.esm_model} …")
    tok = AutoTokenizer.from_pretrained(args.esm_model, clean_up_tokenization_spaces=True)
    esm = EsmForMaskedLM.from_pretrained(args.esm_model).to(device).eval()
    print(f"  {sum(p.numel() for p in esm.parameters()):,} parameters")
    print(f"\nRunning roundtrip on {len(accessions)} sequences …")

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = run_batch(accessions, esm, tok, device, fold=args.fold, out_dir=out_dir)
    print_summary(results)

    if out_dir:
        save_csv(results, out_dir / "roundtrip_summary.csv")


if __name__ == "__main__":
    main()
