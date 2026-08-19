#!/usr/bin/env python
"""
Protein SAE Intervention & Folding Pipeline
============================================

Use interpretable SAE features to steer protein sequences, then fold with ESMFold.

Pipeline:
  1. Input protein sequence
  2. Extract ESM2 layer-3 hidden states
  3. Encode with SAE → 5120 interpretable features per residue
  4. Apply user interventions (scale, set, zero, add features)
  5. Decode modified features → modified hidden states
  6. Inject modified hidden states at layer 3, forward through remaining ESM2 layers
  7. Decode modified logits → steered amino acid sequence
  8. (Optional) Fold original & steered sequences with ESMFold → PDB files

Usage:
  # Inspect top features for a protein by UniProt accession:
  python scripts/intervene_and_fold.py --accession P00698

  # Intervene and fold:
  python scripts/intervene_and_fold.py \\
      --accession P00698 \\
      --interventions "5:scale:3.0,42:zero" \\
      --fold --output-dir results/my_intervention

  # With a raw sequence instead of accession:
  python scripts/intervene_and_fold.py \\
      --sequence "MKFLILLFNILCLFPVLAADNH..." \\
      --interventions "5:scale:3.0,42:zero"

  # With intervention YAML file:
  python scripts/intervene_and_fold.py \\
      --accession P00698 \\
      --intervention-file my_interventions.yaml \\
      --fold --output-dir results/my_intervention

Intervention shorthand format:
  <feature>:<action>[:<value>][@<positions>]

  Examples:
    5:scale:2.0          Scale feature 5 by 2× at all positions
    10:zero              Zero out feature 10 everywhere
    15:set:5.0           Set feature 15 to 5.0
    20:add:1.0@10-20     Add 1.0 to feature 20 at residue positions 10–20
    5:scale:3.0@5,10,15  Scale feature 5 by 3× at positions 5, 10, 15

Intervention YAML format:
  interventions:
    - feature: 5
      action: scale
      value: 2.0
    - feature: 10
      action: zero
      positions: [10, 11, 12, 13, 14]
"""

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F
import yaml

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TORCH", "1")

# ── Project imports ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proteinlens.sae.inference import load_sae
from proteinlens.utils import get_device
from proteinlens.analysis.feature_clusters import FeatureClusters

# ── Constants ─────────────────────────────────────────────────────
DEFAULT_ESM_MODEL = "facebook/esm2_t6_8M_UR50D"
DEFAULT_SAE_DIR = Path(__file__).resolve().parent.parent / "trained_models" / "fiery-sweep"
DEFAULT_ESM_LAYER = 3
ESMFOLD_MODEL = "facebook/esmfold_v1"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
MAX_SEQ_LEN = 1024


# ══════════════════════════════════════════════════════════════════
#  Sequence Fetching
# ══════════════════════════════════════════════════════════════════

def fetch_sequence(accession: str) -> str:
    """Fetch a protein sequence from UniProt by accession ID.

    Matches the pattern used in build_activation_dataset.py.
    """
    url = UNIPROT_FASTA_URL.format(acc=accession)
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        raise ValueError(
            f"Could not fetch sequence for '{accession}' from UniProt "
            f"(HTTP {r.status_code}). Check the accession is valid."
        )
    lines = r.text.strip().split("\n")
    seq = "".join(l.strip() for l in lines if not l.startswith(">"))
    if not seq:
        raise ValueError(f"Empty sequence returned for accession '{accession}'.")
    return seq


# ══════════════════════════════════════════════════════════════════
#  Intervention Specification
# ══════════════════════════════════════════════════════════════════

@dataclass
class FeatureIntervention:
    """Specification for a single feature intervention.

    Attributes:
        feature_idx:  SAE dictionary index (0–5119 for the fiery-sweep model).
        action:       One of "scale", "set", "zero", "add".
        value:        Scalar operand (ignored for "zero").
        positions:    0-indexed residue positions, or None for all residues.
    """
    feature_idx: int
    action: str
    value: float = 0.0
    positions: Optional[List[int]] = None

    # ── apply ────────────────────────────────────────────────────
    def apply(self, features: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Mutate *features* in-place and return it.

        Args:
            features: (seq_len, dict_size) – residue-only SAE activations.
            seq_len:  number of residues.
        """
        if self.positions is not None:
            pos = torch.tensor(self.positions, device=features.device, dtype=torch.long)
            pos = pos[(pos >= 0) & (pos < seq_len)]
        else:
            pos = None  # means "all"

        fidx = self.feature_idx

        if self.action == "scale":
            if pos is None:
                features[:, fidx] *= self.value
            else:
                features[pos, fidx] *= self.value
        elif self.action == "set":
            if pos is None:
                features[:, fidx] = self.value
            else:
                features[pos, fidx] = self.value
        elif self.action == "zero":
            if pos is None:
                features[:, fidx] = 0.0
            else:
                features[pos, fidx] = 0.0
        elif self.action == "add":
            if pos is None:
                features[:, fidx] += self.value
            else:
                features[pos, fidx] += self.value
        else:
            raise ValueError(f"Unknown intervention action: {self.action!r}")

        return features

    def __str__(self):
        pos_str = f" @ positions {self.positions}" if self.positions else " @ all positions"
        if self.action == "zero":
            return f"Feature {self.feature_idx}: zero{pos_str}"
        return f"Feature {self.feature_idx}: {self.action} {self.value}{pos_str}"


# ── Parsers ───────────────────────────────────────────────────────

def _parse_position_spec(spec: str) -> List[int]:
    """Parse '5,10-15,20' → [5, 10, 11, 12, 13, 14, 15, 20]."""
    positions: List[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            positions.extend(range(int(lo), int(hi) + 1))
        else:
            positions.append(int(part))
    return positions


def parse_shorthand_interventions(spec: str) -> List[FeatureIntervention]:
    """Parse shorthand string.

    Format:  ``<feat>:<action>[:<value>][@<positions>], …``
    Examples:  ``5:scale:2.0,10:zero,15:add:1.0@10-20``
    """
    interventions: List[FeatureIntervention] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue

        positions = None
        if "@" in item:
            item, pos_spec = item.rsplit("@", 1)
            positions = _parse_position_spec(pos_spec)

        parts = item.split(":")
        feature_idx = int(parts[0])
        action = parts[1]
        value = float(parts[2]) if len(parts) > 2 else (0.0 if action == "zero" else 1.0)

        interventions.append(FeatureIntervention(
            feature_idx=feature_idx, action=action,
            value=value, positions=positions,
        ))
    return interventions


def load_interventions_from_yaml(path: str) -> List[FeatureIntervention]:
    """Load from a YAML file with an ``interventions`` key."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        FeatureIntervention(
            feature_idx=item["feature"],
            action=item["action"],
            value=item.get("value", 0.0 if item["action"] == "zero" else 1.0),
            positions=item.get("positions"),
        )
        for item in data.get("interventions", [])
    ]


# ══════════════════════════════════════════════════════════════════
#  Model Loading
# ══════════════════════════════════════════════════════════════════

def load_pipeline_models(sae_dir: str, esm_model_name: str, device: str):
    """Return ``(tokenizer, esm_model, sae)``."""
    from transformers import AutoTokenizer, EsmForMaskedLM

    print(f"  Loading ESM model: {esm_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        esm_model_name, clean_up_tokenization_spaces=True,
    )
    esm_model = EsmForMaskedLM.from_pretrained(esm_model_name).to(device)
    esm_model.eval()

    print(f"  Loading SAE from {sae_dir}")
    sae = load_sae(sae_dir, device=device)
    sae.eval()

    return tokenizer, esm_model, sae


def load_esmfold(device: str = "cpu"):
    """Return ``(fold_tokenizer, fold_model)``."""
    from transformers import AutoTokenizer, EsmForProteinFolding

    print(f"  Loading ESMFold ({ESMFOLD_MODEL}) – this may take a minute …")
    tok = AutoTokenizer.from_pretrained(ESMFOLD_MODEL)
    model = EsmForProteinFolding.from_pretrained(
        ESMFOLD_MODEL, low_cpu_mem_usage=True,
    )
    # Float16 on CUDA, float32 elsewhere
    if device == "cuda" and torch.cuda.is_available():
        model = model.half().to(device)
    else:
        model = model.float().to("cpu")
    model.eval()
    print(f"  ESMFold loaded on {next(model.parameters()).device}")
    return tok, model


# ══════════════════════════════════════════════════════════════════
#  Core Pipeline Steps
# ══════════════════════════════════════════════════════════════════

def extract_hidden_states(
    esm_model, tokenizer, sequence: str, layer_idx: int, device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the sequence through ESM and return (logits, hidden, token_ids, attn_mask).

    Shapes (batch=1, L = len(sequence)):
        logits:     (1, L+2, vocab)
        hidden:     (1, L+2, d_model)      – includes CLS / EOS positions
        token_ids:  (1, L+2)
        attn_mask:  (1, L+2)
    """
    inputs = tokenizer(sequence, return_tensors="pt", padding=False)
    token_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = esm_model(
            token_ids, attention_mask=attn_mask, output_hidden_states=True,
        )
    return outputs.logits, outputs.hidden_states[layer_idx], token_ids, attn_mask


def encode_with_sae(sae, hidden: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Encode residue-only hidden states through the SAE.

    Handles ``normalize_to_sqrt_d`` transparently: if the SAE was trained with
    √d-normalisation we normalise before encoding and return the norms so they
    can be undone after decoding.

    Returns:
        features:        (seq_len, dict_size)
        original_norms:  (seq_len, 1) or None
    """
    residue_hidden = hidden[0, 1:seq_len + 1, :]  # strip CLS / EOS
    with torch.no_grad():
        normalised, original_norms = sae._normalize_input_and_get_norms(residue_hidden)
        features = sae.encode(normalised)
    return features, original_norms


def decode_and_build_hidden(
    sae, features: torch.Tensor, orig_hidden: torch.Tensor,
    seq_len: int, original_norms: Optional[torch.Tensor],
) -> torch.Tensor:
    """Decode SAE features → modified hidden states ready for injection.

    CLS and EOS positions are kept from the *original* hidden states; only
    residue positions (1 … seq_len) are replaced with the SAE reconstruction.
    """
    with torch.no_grad():
        decoded = sae.decode(features)
        decoded = sae._unnormalize_output(decoded, original_norms)

    modified = orig_hidden.clone()
    modified[0, 1:seq_len + 1, :] = decoded
    return modified


def run_esm_from_layer(
    esm_model, modified_hidden: torch.Tensor, token_ids: torch.Tensor,
    attn_mask: torch.Tensor, from_layer: int,
) -> torch.Tensor:
    """Run ESM2 layers ``[from_layer, …, end]`` then LM head.

    This mirrors the direct hidden-state resume path used in the test suite and
    avoids the NNsight patching route, which can stall on newer PyTorch stacks.
    """
    with torch.no_grad():
        x = modified_hidden
        ext_mask = esm_model.esm.get_extended_attention_mask(
            attn_mask, token_ids.shape,
        )
        for layer_module in esm_model.esm.encoder.layer[from_layer:]:
            layer_out = layer_module(x, ext_mask)
            x = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        x = esm_model.esm.encoder.emb_layer_norm_after(x)
        return esm_model.lm_head(x)


def inject_and_get_logits(
    esm_model, token_ids, attn_mask,
    layer_idx: int, modified_hidden: torch.Tensor,
) -> torch.Tensor:
    """Resume the ESM forward pass from ``layer_idx`` using *modified_hidden*."""
    return run_esm_from_layer(
        esm_model, modified_hidden, token_ids, attn_mask, layer_idx,
    )


def logits_to_sequence(
    logits: torch.Tensor, tokenizer, seq_len: int, temperature: float = 0.0,
) -> str:
    """Convert MLM logits to an amino-acid string.

    Args:
        temperature:  0 → greedy argmax;  >0 → softmax sampling.
    """
    residue_logits = logits[0, 1:seq_len + 1, :]   # skip CLS / EOS

    if temperature <= 0:
        token_ids = residue_logits.argmax(dim=-1)
    else:
        probs = F.softmax(residue_logits / temperature, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

    tokens = tokenizer.convert_ids_to_tokens(token_ids.cpu().tolist())
    return "".join(tokens)


# ══════════════════════════════════════════════════════════════════
#  ESMFold
# ══════════════════════════════════════════════════════════════════

def fold_sequence(sequence: str, fold_tokenizer, fold_model) -> Tuple[str, float]:
    """Fold a sequence with ESMFold → ``(pdb_string, mean_pLDDT)``."""
    from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

    dev = next(fold_model.parameters()).device
    inputs = fold_tokenizer(
        [sequence], return_tensors="pt", add_special_tokens=False, padding=False,
    )
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = fold_model(**inputs)

    positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    np_out = {k: v.cpu().numpy() for k, v in outputs.items() if isinstance(v, torch.Tensor)}
    positions_np = positions.cpu().numpy()

    aa = np_out["aatype"][0]
    resid = np.arange(1, len(aa) + 1)
    pdb_str = to_pdb(OFProtein(
        aatype=aa,
        atom_positions=positions_np[0],
        atom_mask=np_out["atom37_atom_exists"][0],
        residue_index=resid,
        b_factors=np_out["plddt"][0],
        chain_index=np.zeros_like(resid),
    ))
    mean_plddt = float(np_out["plddt"][0].mean())
    return pdb_str, mean_plddt


# ══════════════════════════════════════════════════════════════════
#  Reporting Helpers
# ══════════════════════════════════════════════════════════════════

def print_top_features(
    features: torch.Tensor, top_k: int = 20, motif_summary: dict = None,
):
    """Show the most active SAE features across the sequence."""
    mean_act = features.mean(dim=0)
    max_act = features.max(dim=0).values
    top_idx = mean_act.argsort(descending=True)[:top_k]

    n_res = features.shape[0]
    print(f"\n{'─' * 90}")
    print(f"  Top {top_k} SAE Features  (across {n_res} residues)")
    print(f"{'─' * 90}")
    header = f"  {'Feat':>6}  {'Mean':>9}  {'Max':>9}  {'Active%':>8}  Description"
    print(header)
    print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*40}")

    for idx in top_idx:
        i = idx.item()
        m = mean_act[idx].item()
        mx = max_act[idx].item()
        pct = (features[:, idx] > 0).float().mean().item() * 100

        desc = ""
        if motif_summary:
            node = motif_summary.get(str(i)) or motif_summary.get(i)
            if node:
                if "decision_tree_rules" in node:
                    desc = str(node["decision_tree_rules"])[:50]
                elif "enrichment" in node:
                    best = max(node["enrichment"].items(), key=lambda kv: kv[1], default=("", 0))
                    desc = f"enriched: {best[0]} ({best[1]:.1f}×)"

        print(f"  {i:>6}  {m:>9.4f}  {mx:>9.4f}  {pct:>7.1f}%  {desc}")
    print()


def print_sequence_diff(orig: str, steered: str):
    """Pretty-print a residue-by-residue diff."""
    n_mut = sum(a != b for a, b in zip(orig, steered))
    pct = 100 * n_mut / len(orig)

    print(f"\n{'─' * 90}")
    print(f"  Sequence Comparison  ({n_mut}/{len(orig)} residues changed – {pct:.1f}%)")
    print(f"{'─' * 90}")

    BLK = 60
    for s in range(0, len(orig), BLK):
        e = min(s + BLK, len(orig))
        ob = orig[s:e]
        sb = steered[s:e]
        dm = "".join("|" if a != b else " " for a, b in zip(ob, sb))

        label = f"{s}–{e-1}"
        print(f"  [{label:>11}]  Original: {ob}")
        print(f"  {' '*13}  Diff:     {dm}")
        print(f"  {' '*13}  Steered:  {sb}")
        print()

    if 0 < n_mut <= 40:
        print("  Mutations:")
        for i, (a, b) in enumerate(zip(orig, steered)):
            if a != b:
                print(f"    pos {i}: {a} → {b}")
        print()


# ══════════════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════════════

def run_pipeline(args):
    device = args.device or get_device()
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("  ProteinLens – SAE Intervention & Folding Pipeline")
    print("=" * 90)

    # ── Load models ───────────────────────────────────────────────
    tokenizer, esm_model, sae = load_pipeline_models(
        args.sae_dir, args.esm_model, device,
    )

    # ── Resolve input → sequence ──────────────────────────────────
    accession = None
    if args.accession:
        accession = args.accession.strip()
        print(f"\n  Fetching sequence for UniProt accession: {accession} …")
        sequence = fetch_sequence(accession)
    elif args.sequence:
        sequence = args.sequence
    else:
        raise SystemExit("Error: provide --accession or --sequence")

    sequence = sequence.strip().upper()
    if len(sequence) > MAX_SEQ_LEN:
        print(f"  ⚠ Sequence length {len(sequence)} exceeds {MAX_SEQ_LEN}, truncating.")
        sequence = sequence[:MAX_SEQ_LEN]
    seq_len = len(sequence)
    trunc = sequence[:50] + ("…" if seq_len > 50 else "")
    if accession:
        print(f"  Input: {accession}  ({seq_len} residues): {trunc}")
    else:
        print(f"  Input sequence  ({seq_len} residues): {trunc}")

    # ── Step 1: ESM forward → layer-3 hidden states ──────────────
    print(f"\n[1/7] Extracting ESM2 layer-{args.layer} hidden states …")
    orig_logits, orig_hidden, token_ids, attn_mask = extract_hidden_states(
        esm_model, tokenizer, sequence, args.layer, device,
    )
    print(f"      Hidden-state tensor: {list(orig_hidden.shape)}")

    # ── Step 2: SAE encode ────────────────────────────────────────
    print(f"\n[2/7] Encoding with SAE ({sae.activation_dim} → {sae.dict_size}) …")
    features, original_norms = encode_with_sae(sae, orig_hidden, seq_len)
    l0 = (features > 0).float().sum(dim=-1).mean().item()
    print(f"      Feature matrix: {list(features.shape)}")
    print(f"      Mean L0 (active features / residue): {l0:.1f}")

    # ── Load motif summary (optional) ────────────────────────────
    motif_summary = None
    candidate_paths = [
        Path(__file__).resolve().parent.parent / "protein_results" / "residue_motifs" / "motif_summary.yaml",
        Path(__file__).resolve().parent.parent / "protein_results" / "residue_motifs_21" / "motif_summary.yaml",
    ]
    for p in candidate_paths:
        if p.exists():
            try:
                with open(p) as fh:
                    motif_summary = yaml.full_load(fh)
                print(f"      Loaded motif descriptions from {p.name}")
            except Exception as exc:
                print(f"      ⚠ Could not load {p.name}: {exc}")
            break

    print_top_features(features, top_k=args.top_k, motif_summary=motif_summary)

    # ── Step 3: Parse & apply interventions ───────────────────────
    interventions: List[FeatureIntervention] = []
    if args.interventions:
        interventions = parse_shorthand_interventions(args.interventions)
    elif args.intervention_file:
        interventions = load_interventions_from_yaml(args.intervention_file)

    # Cluster-based interventions (additive with any individual interventions above)
    if args.cluster_idx is not None:
        if not args.cluster_file:
            raise SystemExit("Error: --cluster-idx requires --cluster-file")
        fc = FeatureClusters.from_file(args.cluster_file)
        cluster_positions = (
            _parse_position_spec(args.cluster_positions)
            if args.cluster_positions else None
        )
        cluster_ivs = fc.make_interventions(
            args.cluster_idx,
            action=args.cluster_action,
            value=args.cluster_value,
            positions=cluster_positions,
        )
        print(
            f"[3a/7] Cluster {args.cluster_idx}: {len(cluster_ivs)} features, "
            f"action={args.cluster_action}"
            + (f" value={args.cluster_value}" if args.cluster_action != "zero" else "")
        )
        interventions = interventions + cluster_ivs

        # Show top proteins for the cluster whenever max_examples is provided
        if args.max_examples:
            with open(args.max_examples) as fh:
                max_ex = yaml.safe_load(fh)
            top_prots = fc.get_top_proteins(
                args.cluster_idx, max_ex, n_per_feature=args.cluster_top_n
            )
            print(f"\n  Top proteins for cluster {args.cluster_idx} "
                  f"(n_per_feature={args.cluster_top_n}):")
            for i, prot in enumerate(top_prots[:20], 1):
                print(f"    {i:2d}. {prot}")

    if not interventions:
        # ── Inspect mode (no interventions) ───────────────────────
        print("[3/7] No interventions specified – running in inspect mode.")
        print("      Use --interventions or --intervention-file to steer.\n")
        print("      Examples:")
        print("        --interventions '5:scale:3.0'           # amplify feature 5 by 3×")
        print("        --interventions '10:zero'               # ablate feature 10")
        print("        --interventions '5:scale:2.0,10:zero'   # combine multiple")
        print("        --interventions '5:set:4.0@10-30'       # set feature 5 at positions 10–30\n")

        # Show SAE reconstruction quality as a sanity check
        with torch.no_grad():
            recon = sae.decode(features)
            recon = sae._unnormalize_output(recon, original_norms)
        residue_orig = orig_hidden[0, 1:seq_len + 1, :]
        mse = ((residue_orig - recon) ** 2).mean().item()
        total_var = residue_orig.var(dim=0).sum().item()
        resid_var = (residue_orig - recon).var(dim=0).sum().item()
        ve = 1 - resid_var / total_var
        print(f"      SAE reconstruction MSE:          {mse:.6f}")
        print(f"      SAE variance explained:          {ve:.4f}")

        # Show what ESM predicts from the original logits
        orig_pred = logits_to_sequence(orig_logits, tokenizer, seq_len, temperature=0.0)
        n_same = sum(a == b for a, b in zip(sequence, orig_pred))
        print(f"\n      Original sequence:  {sequence[:60]}{'…' if seq_len > 60 else ''}")
        print(f"      ESM2 MLM argmax:    {orig_pred[:60]}{'…' if seq_len > 60 else ''}")
        print(f"      ESM2 recovery:      {n_same}/{seq_len} ({100*n_same/seq_len:.1f}%)\n")
        return

    # ── Apply interventions ───────────────────────────────────────
    print(f"[3/7] Applying {len(interventions)} intervention(s) …")
    features_mod = features.clone()
    for iv in interventions:
        print(f"      → {iv}")
        features_mod = iv.apply(features_mod, seq_len)

    diff = (features_mod - features).abs()
    n_feat_changed = int((diff.sum(dim=0) > 0).sum().item())
    n_pos_changed = int((diff.sum(dim=1) > 0).sum().item())
    print(f"      Touched {n_feat_changed} feature(s) across {n_pos_changed} position(s)")

    # ── Step 4: SAE decode → modified hidden states ───────────────
    print(f"\n[4/7] Decoding modified features → hidden states …")
    modified_hidden = decode_and_build_hidden(
        sae, features_mod, orig_hidden, seq_len, original_norms,
    )
    delta_norm = (modified_hidden - orig_hidden).norm(dim=-1).mean().item()
    print(f"      Mean Δ‖hidden‖ (L2): {delta_norm:.4f}")

    # ── Step 5: Inject & forward ──────────────────────────────────
    print(f"\n[5/7] Injecting at layer {args.layer}, forwarding through layers {args.layer+1}–6 …")
    modified_logits = inject_and_get_logits(
        esm_model, token_ids, attn_mask,
        args.layer, modified_hidden,
    )

    # ── Step 6: Decode steered sequence ───────────────────────────
    print(f"\n[6/7] Decoding steered sequence (temperature={args.temperature}) …")
    steered_seq = logits_to_sequence(
        modified_logits, tokenizer, seq_len, temperature=args.temperature,
    )
    print_sequence_diff(sequence, steered_seq)

    # ── Step 7: ESMFold ───────────────────────────────────────────
    if args.fold:
        print(f"[7/7] Folding with ESMFold …")
        fold_device = args.fold_device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            fold_tok, fold_mdl = load_esmfold(fold_device)

            print(f"      Folding original sequence …")
            orig_pdb, orig_plddt = fold_sequence(sequence, fold_tok, fold_mdl)
            print(f"      Original  mean pLDDT: {orig_plddt:.2f}")

            print(f"      Folding steered sequence …")
            st_pdb, st_plddt = fold_sequence(steered_seq, fold_tok, fold_mdl)
            print(f"      Steered   mean pLDDT: {st_plddt:.2f}")

            if output_dir:
                for name, pdb_str in [("original.pdb", orig_pdb), ("steered.pdb", st_pdb)]:
                    p = output_dir / name
                    p.write_text(pdb_str)
                    print(f"      Saved {p}")

        except Exception as exc:
            print(f"\n      ⚠ ESMFold error: {exc}")
            print("      ESMFold is ~3 GB; try --fold-device cpu or omit --fold.")
            print("      The steered sequence can still be folded externally (e.g. ColabFold).")
    else:
        print(f"[7/7] Skipping ESMFold (pass --fold to enable)")

    # ── Save results ──────────────────────────────────────────────
    if output_dir:
        # YAML summary
        results = {
            "accession": accession,
            "original_sequence": sequence,
            "steered_sequence": steered_seq,
            "esm_model": args.esm_model,
            "sae_dir": str(args.sae_dir),
            "layer": args.layer,
            "temperature": args.temperature,
            "n_residues": seq_len,
            "n_mutations": sum(a != b for a, b in zip(sequence, steered_seq)),
            "mean_hidden_delta_l2": float(delta_norm),
            "interventions": [
                {"feature_idx": iv.feature_idx, "action": iv.action,
                 "value": iv.value, "positions": iv.positions}
                for iv in interventions
            ],
        }
        (output_dir / "results.yaml").write_text(
            yaml.dump(results, default_flow_style=False, sort_keys=False)
        )

        # FASTA
        fasta = f">original\n{sequence}\n>steered\n{steered_seq}\n"
        (output_dir / "sequences.fasta").write_text(fasta)

        # Feature activations (before & after)
        np.savez(
            output_dir / "features.npz",
            original=features.cpu().numpy(),
            modified=features_mod.cpu().numpy(),
        )

        print(f"\n      Saved results to {output_dir}/")

    print(f"\n{'=' * 90}")
    print("  Pipeline complete!")
    print(f"{'=' * 90}\n")


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="ProteinLens – SAE Intervention & Folding Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Inspect features for a protein by UniProt accession:
  python scripts/intervene_and_fold.py --accession P00698

  # Scale feature 5 by 3× and fold:
  python scripts/intervene_and_fold.py \\
      --accession P00698 \\
      --interventions "5:scale:3.0" \\
      --fold --output-dir results/experiment_01

  # Multiple interventions, temperature sampling:
  python scripts/intervene_and_fold.py \\
      --accession P00698 \\
      --interventions "5:scale:2.0,10:zero,15:add:1.0@10-20" \\
      --temperature 0.5 --output-dir results/experiment_02

  # Or provide a raw sequence directly:
  python scripts/intervene_and_fold.py --sequence "MKFLIL..."
""",

    )

    # Input (one of these is required)
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--accession", default=None,
                     help="UniProt accession ID (e.g. P00698). Sequence fetched automatically.")
    inp.add_argument("--sequence", default=None,
                     help="Raw amino-acid sequence (fallback if no accession)")

    # Interventions (mutually exclusive)
    iv = p.add_mutually_exclusive_group()
    iv.add_argument("--interventions", default=None,
                    help="Shorthand: 'feature:action:value[@pos], …'")
    iv.add_argument("--intervention-file", default=None,
                    help="Path to YAML intervention spec")

    # Models
    p.add_argument("--sae-dir", default=str(DEFAULT_SAE_DIR),
                   help=f"Trained SAE directory (default: {DEFAULT_SAE_DIR.name})")
    p.add_argument("--esm-model", default=DEFAULT_ESM_MODEL,
                   help=f"ESM model name (default: {DEFAULT_ESM_MODEL})")
    p.add_argument("--layer", type=int, default=DEFAULT_ESM_LAYER,
                   help=f"ESM layer to hook (default: {DEFAULT_ESM_LAYER})")

    # Generation
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy)")

    # Folding
    p.add_argument("--fold", action="store_true",
                   help="Fold with ESMFold after steering")
    p.add_argument("--fold-device", default=None,
                   help="Device for ESMFold (default: cuda or cpu)")

    # Cluster-based interventions
    cl = p.add_argument_group("cluster interventions")
    cl.add_argument("--cluster-file", default=None,
                    help="Path to clusters YAML produced by cluster_sae_features.py")
    cl.add_argument("--cluster-idx", type=int, default=None,
                    help="Cluster index to intervene on (requires --cluster-file)")
    cl.add_argument("--cluster-action", default="zero",
                    choices=["scale", "set", "zero", "add"],
                    help="Action to apply to all features in the cluster (default: zero)")
    cl.add_argument("--cluster-value", type=float, default=1.0,
                    help="Scalar value for scale/set/add cluster action (default: 1.0)")
    cl.add_argument("--cluster-positions", default=None,
                    help="Position spec for cluster interventions, e.g. '10-20,30' "
                         "(default: all positions)")
    cl.add_argument("--max-examples", default=None,
                    help="Path to Per_feature_max_examples.yaml for cluster top-protein display")
    cl.add_argument("--cluster-top-n", type=int, default=3,
                    help="n_per_feature when showing top proteins for a cluster (default: 3)")

    # Output
    p.add_argument("--output-dir", default=None,
                   help="Save results here (PDBs, FASTA, YAML, features)")
    p.add_argument("--device", default=None,
                   help="Device for ESM2 + SAE (default: auto)")
    p.add_argument("--top-k", type=int, default=20,
                   help="Top features to display in inspect mode")

    args = p.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
