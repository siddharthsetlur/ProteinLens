#!/usr/bin/env python
"""
Geometry-Primary SAE Steering Experiments
==========================================

Demonstrates that steering geometry-primary SAE features up/down produces
interpretable, directional changes in the folded structure's geometry.

For each experiment:
  1. Load a protein where the target feature is highly active
  2. Encode ESM2 hidden states through the SAE
  3. Apply interventions (clamp / scale) at various multipliers
  4. Decode modified features, inject back into ESM2, get steered sequence
  5. Fold steered sequence with ESMFold
  6. Measure geometric properties and compare to baseline

Usage:
  # Smoke test (no folding):
  python scripts/run_steering_experiments.py --no-fold --output-dir results/steering_test

  # Single feature:
  python scripts/run_steering_experiments.py --features 3093 --output-dir results/steering_single

  # Full run:
  python scripts/run_steering_experiments.py --output-dir results/steering_full
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

# ── Project imports ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proteinlens.sae.inference import load_sae
from proteinlens.utils import get_device
from proteinlens.analysis.geometry.residue_features import (
    ca_backbone,
    detect_alpha_helices_from_ca,
    compute_residue_profiles,
    extract_local_feature_vector,
    LOCAL_GEOM_NAMES,
)
from proteinlens.analysis.geometry.protein_features import compute_protein_geometry


# ══════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════

DEFAULT_ESM_MODEL = "facebook/esm2_t6_8M_UR50D"
DEFAULT_SAE_DIR = Path(__file__).resolve().parent.parent / "trained_models" / "fiery-sweep"
DEFAULT_ESM_LAYER = 3
ESMFOLD_MODEL = "facebook/esmfold_v1"

MULTIPLIERS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
N_RANDOM_CONTROLS = 3
RANDOM_SEED = 42
HALF_W = 5  # half-window for per-residue geometry

DATA_DIR = Path(__file__).resolve().parent.parent / "feature_data_cluster"


# ══════════════════════════════════════════════════════════════════
#  Experiment Configuration
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    feature_idx: int
    accession: str
    max_activation: float
    target_metric: str  # name in LOCAL_GEOM_NAMES
    predicted_direction: str  # "decrease" or "increase"
    structural_category: str


EXPERIMENTS = [
    ExperimentConfig(
        feature_idx=1403,
        accession="P37510",
        max_activation=9.97,
        target_metric="narrow_end_to_end_ratio",
        predicted_direction="decrease",
        structural_category="Local compactness (loop/turn vs extended)",
    ),
    ExperimentConfig(
        feature_idx=3514,
        accession="P23205",
        max_activation=20.10,
        target_metric="curv_centre_third",
        predicted_direction="decrease",
        structural_category="Backbone curvature",
    ),
    ExperimentConfig(
        feature_idx=3192,
        accession="O32268",
        max_activation=20.89,
        target_metric="contact_density_12A",
        predicted_direction="decrease",
        structural_category="Contact density (packing)",
    ),
]


def get_geometry_primary_feature_ids() -> set[int]:
    """Load the set of 204 geometry-primary feature IDs."""
    with open(DATA_DIR / "geometry_primary_analysis.json") as f:
        data = json.load(f)
    gp_ids = set()
    for fid, fdata in data["features"].items():
        if fdata.get("is_geometry_primary", False):
            gp_ids.add(int(fid))
    return gp_ids


def select_random_control_features(
    n: int, exclude: set[int], total_features: int = 5120
) -> list[int]:
    """Select random features that are NOT geometry-primary."""
    rng = np.random.RandomState(RANDOM_SEED)
    candidates = [i for i in range(total_features) if i not in exclude]
    return rng.choice(candidates, size=n, replace=False).tolist()


# ══════════════════════════════════════════════════════════════════
#  Model Loading
# ══════════════════════════════════════════════════════════════════

def load_models(sae_dir: str, esm_model_name: str, device: str):
    """Load ESM2 + SAE (no NNsight -- uses PyTorch hooks for intervention)."""
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

    print(f"  SAE: {sae.activation_dim}D -> {sae.dict_size} features, "
          f"normalize_to_sqrt_d={sae.normalize_to_sqrt_d}")

    return tokenizer, esm_model, sae


def load_esmfold(device: str):
    """Load ESMFold for structure prediction (local model)."""
    from transformers import AutoTokenizer, EsmForProteinFolding

    print(f"  Loading ESMFold ({ESMFOLD_MODEL})")
    tok = AutoTokenizer.from_pretrained(ESMFOLD_MODEL)
    model = EsmForProteinFolding.from_pretrained(
        ESMFOLD_MODEL, low_cpu_mem_usage=True,
    )
    if device == "cuda" and torch.cuda.is_available():
        model = model.half().to(device)
    else:
        model = model.float().to("cpu")
    model.eval()
    print(f"  ESMFold loaded on {next(model.parameters()).device}")
    return tok, model


# ══════════════════════════════════════════════════════════════════
#  Core Intervention Pipeline (rebuilt from low-level components)
# ══════════════════════════════════════════════════════════════════

def esm_forward_with_hidden(esm_model, token_ids, attn_mask, layer):
    """Run ESM2 and return (logits, hidden_states_at_layer).

    Uses output_hidden_states=True. hidden_states[layer] gives the output
    of the layer-th transformer layer (hidden_states[0] = embeddings).
    """
    with torch.no_grad():
        out = esm_model(token_ids, attention_mask=attn_mask, output_hidden_states=True)
    return out.logits, out.hidden_states[layer]


def esm_forward_with_injection(esm_model, token_ids, attn_mask, layer, modified_hidden):
    """Run ESM2 with modified hidden states injected at the given layer.

    Uses a forward pre-hook on the target encoder layer to replace its input
    with our modified hidden states. This avoids NNsight entirely.

    For layer N (1-indexed as hidden_states index), we hook the input of
    encoder.layer[N-1] (which receives hidden_states[N-1] and produces
    hidden_states[N]). But we want to REPLACE hidden_states[N], so we
    hook encoder.layer[N] to replace ITS input.

    ESM2-8M has 6 transformer layers indexed 0-5.
    hidden_states[0] = embedding output
    hidden_states[k] = output of encoder.layer[k-1], k=1..6

    To inject at hidden_states[layer], we want to replace the input to
    encoder.layer[layer] (if layer < n_layers) or to emb_layer_norm_after
    (if layer == n_layers).
    """
    n_layers = len(esm_model.esm.encoder.layer)

    handle = None
    if layer < n_layers:
        target_module = esm_model.esm.encoder.layer[layer]

        def hook_fn(module, args):
            # args is a tuple; args[0] is the hidden_states tensor
            new_args = (modified_hidden,) + args[1:]
            return new_args

        handle = target_module.register_forward_pre_hook(hook_fn)
    elif layer == n_layers:
        target_module = esm_model.esm.encoder.emb_layer_norm_after

        def hook_fn(module, args):
            return (modified_hidden.squeeze(0) if modified_hidden.dim() == 3 else modified_hidden,) + args[1:]

        handle = target_module.register_forward_pre_hook(hook_fn)
    else:
        raise ValueError(f"layer {layer} out of range for {n_layers}-layer model")

    try:
        with torch.no_grad():
            out = esm_model(token_ids, attention_mask=attn_mask, output_hidden_states=True)
        return out.logits
    finally:
        if handle is not None:
            handle.remove()


def extract_and_encode(tokenizer, esm_model, sae, sequence, layer, device):
    """Run ESM2 forward pass, extract hidden states, encode with SAE.

    Returns:
        features: (seq_len, dict_size) SAE feature activations
        orig_hidden_full: (1, L+2, d_model) full hidden states including CLS/EOS
        token_ids: (1, L+2) input token IDs
        attn_mask: (1, L+2) attention mask
        original_norms: norms for unnormalization (None if no sqrt_d)
    """
    seq_len = len(sequence)

    # Tokenize
    inputs = tokenizer(sequence, return_tensors="pt", padding=False)
    token_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    # Forward pass -> get hidden states at target layer
    logits, hidden = esm_forward_with_hidden(esm_model, token_ids, attn_mask, layer)

    # hidden shape: (1, L+2, d_model) where L+2 includes CLS and EOS
    orig_hidden_full = hidden

    # Strip CLS (pos 0) and EOS (pos seq_len+1) to get residue-only hidden states
    residue_hidden = hidden[0, 1:seq_len + 1, :]  # (seq_len, d_model)

    # Normalize if SAE uses sqrt_d (fiery-sweep: no-op)
    with torch.no_grad():
        normalized, original_norms = sae._normalize_input_and_get_norms(residue_hidden)
        features = sae.encode(normalized)  # (seq_len, dict_size)

    return features, orig_hidden_full, token_ids, attn_mask, original_norms


def apply_intervention(features, feature_idx, action, value, positions=None):
    """Apply a single feature intervention. Returns a NEW tensor.

    Args:
        positions: list of residue indices to modify, or None for all.
    """
    modified = features.clone()
    if positions is not None:
        pos = torch.tensor(positions, dtype=torch.long, device=features.device)
    else:
        pos = None

    if action == "set":
        if pos is None:
            modified[:, feature_idx] = value
        else:
            modified[pos, feature_idx] = value
    elif action == "scale":
        if pos is None:
            modified[:, feature_idx] *= value
        else:
            modified[pos, feature_idx] *= value
    elif action == "zero":
        if pos is None:
            modified[:, feature_idx] = 0.0
        else:
            modified[pos, feature_idx] = 0.0
    elif action == "add":
        if pos is None:
            modified[:, feature_idx] += value
        else:
            modified[pos, feature_idx] += value
    else:
        raise ValueError(f"Unknown action: {action}")
    return modified


def decode_and_inject(
    sae, modified_features, orig_hidden_full, seq_len, original_norms,
    esm_model, token_ids, attn_mask, layer,
):
    """Decode SAE features -> modified hidden states, inject into ESM2."""
    with torch.no_grad():
        decoded = sae.decode(modified_features)  # (seq_len, d_model)
        decoded = sae._unnormalize_output(decoded, original_norms)

    # Build full hidden tensor: keep CLS/EOS from original, replace residues
    modified_hidden = orig_hidden_full.clone()
    modified_hidden[0, 1:seq_len + 1, :] = decoded

    # Inject via PyTorch hook and get steered logits
    steered_logits = esm_forward_with_injection(
        esm_model, token_ids, attn_mask, layer, modified_hidden,
    )

    return steered_logits


def logits_to_sequence(logits, tokenizer, seq_len):
    """Convert MLM logits to amino acid sequence (greedy argmax)."""
    residue_logits = logits[0, 1:seq_len + 1, :]
    token_ids = residue_logits.argmax(dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(token_ids.cpu().tolist())
    return "".join(tokens)


def run_single_intervention(
    tokenizer, esm_model, sae, sequence, layer, device,
    feature_idx, action, value,
):
    """Run one intervention end-to-end. Returns steered sequence + metadata."""
    seq_len = len(sequence)

    features, orig_hidden_full, token_ids, attn_mask, original_norms = \
        extract_and_encode(tokenizer, esm_model, sae, sequence, layer, device)

    modified_features = apply_intervention(features, feature_idx, action, value)

    steered_logits = decode_and_inject(
        sae, modified_features, orig_hidden_full, seq_len, original_norms,
        esm_model, token_ids, attn_mask, layer,
    )

    steered_seq = logits_to_sequence(steered_logits, tokenizer, seq_len)

    n_mutations = sum(a != b for a, b in zip(sequence, steered_seq))
    pct_changed = 100.0 * n_mutations / seq_len

    return {
        "steered_sequence": steered_seq,
        "n_mutations": n_mutations,
        "pct_changed": round(pct_changed, 2),
    }


# ══════════════════════════════════════════════════════════════════
#  Folding & Geometric Measurement
# ══════════════════════════════════════════════════════════════════

def fold_sequence_local(sequence, fold_tokenizer, fold_model):
    """Fold a sequence with local ESMFold -> (pdb_string, mean_pLDDT)."""
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


ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"


def fold_sequence_api(sequence, max_retries=3):
    """Fold a sequence via ESM Metagenomic Atlas API -> (pdb_string, mean_pLDDT).

    No local model needed. Rate-limited, max ~400 residues.
    """
    import requests
    import time

    for attempt in range(max_retries):
        try:
            r = requests.post(ESMFOLD_API_URL, data=sequence, timeout=120)
            if r.status_code == 200:
                break
            if r.status_code in (429, 502, 503, 504) and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"(API {r.status_code}, retry in {wait}s) ", end="", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"ESMFold API returned HTTP {r.status_code}: {r.text[:200]}")
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"(API timeout, retry in {wait}s) ", end="", flush=True)
                time.sleep(wait)
                continue
            raise

    pdb_text = r.text

    # Extract pLDDT from B-factor column of ATOM records
    plddts = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and " CA " in line:
            try:
                plddts.append(float(line[60:66].strip()))
            except (ValueError, IndexError):
                pass
    mean_plddt = float(np.mean(plddts)) if plddts else 0.0

    return pdb_text, mean_plddt


def measure_geometry(pdb_text, target_metric_name, sequence=None):
    """Measure per-residue and whole-protein geometry from a PDB structure.

    Returns dict with target metric stats and whole-protein metrics.
    """
    ca = ca_backbone(pdb_text)
    helices = detect_alpha_helices_from_ca(ca)
    profiles = compute_residue_profiles(ca, helices)

    n = len(ca)
    target_idx = LOCAL_GEOM_NAMES.index(target_metric_name)

    # Whole-protein geometry (55 features)
    whole_protein = compute_protein_geometry(pdb_text)

    # Extract per-residue feature vectors
    target_values = []
    for pos in range(HALF_W, n - HALF_W):
        vec = extract_local_feature_vector(profiles, ca, pos, half_w=HALF_W, sequence=sequence)
        if vec is not None:
            target_values.append(float(vec[target_idx]))

    if not target_values:
        return {
            "target_mean": float("nan"),
            "target_median": float("nan"),
            "target_std": float("nan"),
            "target_n_residues": 0,
            "whole_protein": whole_protein,
        }
    target_arr = np.array(target_values)

    return {
        "target_mean": float(np.mean(target_arr)),
        "target_median": float(np.median(target_arr)),
        "target_std": float(np.std(target_arr)),
        "target_n_residues": len(target_values),
        "whole_protein": whole_protein,
    }


# ══════════════════════════════════════════════════════════════════
#  3D Visualization
# ══════════════════════════════════════════════════════════════════

def plot_steering_3d(
    pdb_original, pdb_steered, activations_original, activations_steered,
    feature_idx, multiplier, style, config, save_path,
):
    """Plot side-by-side 3D backbones colored by SAE feature activation.

    Left: original structure with original activations.
    Right: steered structure with steered activations.
    Color: white (0) → red (max activation).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig = plt.figure(figsize=(16, 7))

    # Shared color scale across both panels
    all_acts = np.concatenate([activations_original, activations_steered])
    vmax = max(float(np.max(all_acts)), 1e-6)
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.YlOrRd  # yellow-orange-red: inactive=yellow, active=red

    panels = [
        ("Original", pdb_original, activations_original),
        (f"Steered ({style} x{multiplier})", pdb_steered, activations_steered),
    ]

    for panel_idx, (title, pdb_text, activations) in enumerate(panels):
        ax = fig.add_subplot(1, 2, panel_idx + 1, projection="3d")

        try:
            ca = ca_backbone(pdb_text)
        except Exception:
            ax.set_title(f"{title}\n(fold failed)")
            continue

        n = len(ca)
        n_act = len(activations)

        # Draw each backbone segment: dark grey if inactive, colored if active
        for i in range(n - 1):
            act_val = float(activations[i]) if i < n_act else 0.0
            if act_val > 0:
                color = cmap(norm(act_val))
                lw = 3.5
                alpha = 1.0
            else:
                color = "#333333"
                lw = 1.8
                alpha = 0.7
            ax.plot(
                ca[i:i+2, 0], ca[i:i+2, 1], ca[i:i+2, 2],
                color=color, linewidth=lw, alpha=alpha,
            )

        # Mark N/C termini
        ax.scatter(*ca[0], color="#27ae60", s=70, zorder=5,
                   edgecolors="black", linewidths=0.6, label="N-term")
        ax.scatter(*ca[-1], color="#8e44ad", s=70, zorder=5,
                   edgecolors="black", linewidths=0.6, label="C-term")

        # Mark centre of each active region with a dot
        active_pos = [i for i in range(min(n, n_act)) if activations[i] > 0]
        if active_pos:
            act_coords = ca[active_pos]
            ax.scatter(
                act_coords[:, 0], act_coords[:, 1], act_coords[:, 2],
                c=[cmap(norm(float(activations[p]))) for p in active_pos],
                s=35, zorder=6, alpha=0.9,
                edgecolors="black", linewidths=0.3,
            )

        ax.set_xlabel("X (A)", fontsize=7)
        ax.set_ylabel("Y (A)", fontsize=7)
        ax.set_zlabel("Z (A)", fontsize=7)
        ax.tick_params(labelsize=6)

        mean_act = float(np.mean(activations)) if n_act > 0 else 0
        n_active = len(active_pos)
        ax.set_title(f"{title}\n{n} residues, {n_active} active, mean act={mean_act:.2f}", fontsize=9)
        ax.legend(fontsize=6, loc="upper left")

    # Shared colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.5, pad=0.08)
    cbar.set_label(f"Feature {feature_idx} activation", fontsize=8)

    fig.suptitle(
        f"Feature {feature_idx}: {config.structural_category}\n"
        f"Target metric: {config.target_metric}",
        fontsize=11, y=1.02,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_per_residue_geom_metric(pdb_text, target_metric_name, sequence=None):
    """Return per-residue array of the target geometric metric.

    Positions near the termini (within HALF_W) get NaN.
    """
    ca = ca_backbone(pdb_text)
    helices = detect_alpha_helices_from_ca(ca)
    profiles = compute_residue_profiles(ca, helices)
    n = len(ca)
    target_idx = LOCAL_GEOM_NAMES.index(target_metric_name)

    values = np.full(n, np.nan)
    for pos in range(HALF_W, n - HALF_W):
        vec = extract_local_feature_vector(profiles, ca, pos, half_w=HALF_W, sequence=sequence)
        if vec is not None:
            values[pos] = float(vec[target_idx])
    return values


def plot_residue_profiles(
    activations_orig, activations_steered,
    geom_orig, geom_steered,
    feature_idx, multiplier, style, config, save_path,
):
    """Plot per-residue SAE activation and geometric metric, original vs steered.

    Two rows, two columns:
      Top-left:  Original SAE activation vs residue
      Top-right: Steered SAE activation vs residue
      Bottom-left:  Original geometric metric vs residue
      Bottom-right: Steered geometric metric vs residue
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_orig = len(activations_orig)
    n_steer = len(activations_steered)

    fig, axes = plt.subplots(2, 2, figsize=(16, 7), sharex="col")

    xs_orig = np.arange(n_orig)
    xs_steer = np.arange(n_steer)

    # ── Top row: SAE activation ──
    for col, (ax, acts, xs, label) in enumerate([
        (axes[0, 0], activations_orig, xs_orig, "Original"),
        (axes[0, 1], activations_steered, xs_steer, f"Steered ({style} x{multiplier})"),
    ]):
        ax.fill_between(xs, 0, acts, color="#2980b9", alpha=0.3)
        ax.plot(xs, acts, color="#2980b9", linewidth=0.8)
        # Highlight active positions
        active_mask = acts > 0
        if active_mask.any():
            ax.scatter(xs[active_mask], acts[active_mask],
                       color="#e74c3c", s=12, zorder=5, alpha=0.8)
        ax.set_ylabel(f"Feature {feature_idx}\nactivation", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=7)

    # ── Bottom row: geometric metric ──
    for col, (ax, geom, xs, label) in enumerate([
        (axes[1, 0], geom_orig, xs_orig, "Original"),
        (axes[1, 1], geom_steered, xs_steer, f"Steered ({style} x{multiplier})"),
    ]):
        valid = ~np.isnan(geom)
        ax.plot(xs[valid], geom[valid], color="#27ae60", linewidth=1.0)
        ax.fill_between(xs[valid], 0, geom[valid], color="#27ae60", alpha=0.15)
        ax.set_ylabel(f"{config.target_metric}", fontsize=8)
        ax.set_xlabel("Residue position", fontsize=8)
        ax.tick_params(labelsize=7)

    # Shared y-limits for each row
    act_max = max(float(np.max(activations_orig)), float(np.max(activations_steered)), 1e-6)
    axes[0, 0].set_ylim(-0.05 * act_max, act_max * 1.1)
    axes[0, 1].set_ylim(-0.05 * act_max, act_max * 1.1)

    geom_all = np.concatenate([geom_orig[~np.isnan(geom_orig)], geom_steered[~np.isnan(geom_steered)]])
    if len(geom_all) > 0:
        gmin, gmax = float(geom_all.min()), float(geom_all.max())
        pad = max((gmax - gmin) * 0.1, 0.01)
        axes[1, 0].set_ylim(gmin - pad, gmax + pad)
        axes[1, 1].set_ylim(gmin - pad, gmax + pad)

    fig.suptitle(
        f"Feature {feature_idx}: {config.structural_category}\n"
        f"Target metric: {config.target_metric} (predicted: {config.predicted_direction})",
        fontsize=11,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════
#  Experiment Runner
# ══════════════════════════════════════════════════════════════════

def run_feature_experiment(
    tokenizer, esm_model, sae, fold_mode, fold_models,
    config: ExperimentConfig, sequence: str, control_features: list[int],
    layer: int, device: str, do_fold: bool, output_dir: Path = None,
):
    """Run all interventions for one feature. Returns list of result dicts."""
    results = []
    seq_len = len(sequence)

    print(f"\n{'=' * 80}")
    print(f"  Feature {config.feature_idx}: {config.structural_category}")
    print(f"  Protein: {config.accession} ({seq_len} residues)")
    print(f"  Target metric: {config.target_metric} (predicted: {config.predicted_direction})")
    print(f"{'=' * 80}")

    # Pre-encode once (reused for all interventions on same protein)
    features, orig_hidden_full, token_ids, attn_mask, original_norms = \
        extract_and_encode(tokenizer, esm_model, sae, sequence, layer, device)

    l0 = (features > 0).float().sum(dim=-1).mean().item()
    feat_act = features[:, config.feature_idx]
    n_active = (feat_act > 0).sum().item()
    mean_act = feat_act[feat_act > 0].mean().item() if n_active > 0 else 0.0
    print(f"  SAE L0: {l0:.1f} active features/residue")
    print(f"  Feature {config.feature_idx}: active at {n_active}/{seq_len} residues, "
          f"mean activation (where active): {mean_act:.3f}")

    # Identify positions where the target feature naturally fires
    feat_act = features[:, config.feature_idx]
    active_mask = feat_act > 0
    active_positions = active_mask.nonzero(as_tuple=True)[0].tolist()
    active_values = feat_act[active_mask].tolist()

    def _run_condition(feat_idx, action, value, style_label, is_control,
                       control_idx=None, positions=None, multiplier_label=None):
        """Run a single condition: intervene, fold, measure."""
        modified = apply_intervention(features, feat_idx, action, value, positions=positions)

        steered_logits = decode_and_inject(
            sae, modified, orig_hidden_full, seq_len, original_norms,
            esm_model, token_ids, attn_mask, layer,
        )
        steered_seq = logits_to_sequence(steered_logits, tokenizer, seq_len)
        n_mut = sum(a != b for a, b in zip(sequence, steered_seq))
        pct = 100.0 * n_mut / seq_len

        row = {
            "feature_idx": feat_idx,
            "experiment_feature": config.feature_idx,
            "action": action,
            "value": round(float(value), 4),
            "style": style_label,
            "is_control": is_control,
            "control_idx": control_idx,
            "multiplier": multiplier_label if multiplier_label is not None else round(float(value), 4),
            "steered_sequence": steered_seq,
            "n_mutations": n_mut,
            "pct_changed": round(pct, 2),
        }

        if do_fold:
            if fold_mode == "api":
                pdb_text, plddt = fold_sequence_api(steered_seq)
            else:
                fold_tok, fold_model = fold_models
                pdb_text, plddt = fold_sequence_local(steered_seq, fold_tok, fold_model)
            geom = measure_geometry(pdb_text, config.target_metric, sequence=steered_seq)
            row["plddt"] = round(plddt, 2)
            row["target_mean"] = round(geom["target_mean"], 6)
            row["target_median"] = round(geom["target_median"], 6)
            row["target_std"] = round(geom["target_std"], 6)
            row["target_n_residues"] = geom["target_n_residues"]
            row["whole_protein"] = geom["whole_protein"]
            row["_pdb_text"] = pdb_text  # kept for plotting, stripped before save

        # Store steered activations for the target feature (for plotting)
        row["_activations"] = modified[:, config.feature_idx].cpu().numpy()

        return row

    # Fold original sequence for "before" comparison panel
    original_pdb = None
    original_activations = features[:, config.feature_idx].cpu().numpy()
    if do_fold:
        print(f"  Folding original sequence ...", end=" ", flush=True)
        if fold_mode == "api":
            original_pdb, orig_plddt = fold_sequence_api(sequence)
        else:
            fold_tok, fold_model = fold_models
            original_pdb, orig_plddt = fold_sequence_local(sequence, fold_tok, fold_model)
        print(f"pLDDT={orig_plddt:.2f}")

    # --- Style 1: Scale at active positions only ---
    for mult in MULTIPLIERS:
        print(f"  [{config.feature_idx}] scale@active mult={mult} ...", end=" ", flush=True)
        row = _run_condition(
            config.feature_idx, "scale", mult, "scale",
            is_control=False, positions=active_positions, multiplier_label=mult,
        )
        print(f"{row['n_mutations']} mutations ({row['pct_changed']}%)"
              + (f", pLDDT={row.get('plddt', 'N/A')}" if do_fold else ""))
        results.append(row)

    # --- Style 2: Expand — activate feature in a window around each active site ---
    expand_hw = HALF_W  # expand by ±5 residues around each active position
    for mult in MULTIPLIERS:
        if mult == 0.0:
            # Zero the feature at active positions (ablation)
            print(f"  [{config.feature_idx}] expand mult=0.0 (ablate) ...", end=" ", flush=True)
            row = _run_condition(
                config.feature_idx, "zero", 0, "expand",
                is_control=False, positions=active_positions, multiplier_label=0.0,
            )
        else:
            # Set feature to max_act * mult at expanded window around each active site
            expand_positions = set()
            for pos in active_positions:
                for p in range(max(0, pos - expand_hw), min(seq_len, pos + expand_hw + 1)):
                    expand_positions.add(p)
            expand_positions = sorted(expand_positions)
            set_val = config.max_activation * mult
            print(f"  [{config.feature_idx}] expand mult={mult} ({len(expand_positions)} pos, set={set_val:.2f}) ...",
                  end=" ", flush=True)
            row = _run_condition(
                config.feature_idx, "set", set_val, "expand",
                is_control=False, positions=expand_positions, multiplier_label=mult,
            )
        print(f"{row['n_mutations']} mutations ({row['pct_changed']}%)"
              + (f", pLDDT={row.get('plddt', 'N/A')}" if do_fold else ""))
        results.append(row)

    # --- Random control features: scale at the SAME active positions ---
    for ci, ctrl_feat in enumerate(control_features):
        for mult in MULTIPLIERS:
            # Expand window around the same positions, set to max_act * mult
            expand_positions = set()
            for pos in active_positions:
                for p in range(max(0, pos - expand_hw), min(seq_len, pos + expand_hw + 1)):
                    expand_positions.add(p)
            expand_positions = sorted(expand_positions)
            if mult == 0.0:
                print(f"  [ctrl {ctrl_feat}] expand mult=0.0 ...", end=" ", flush=True)
                row = _run_condition(
                    ctrl_feat, "zero", 0, "expand",
                    is_control=True, control_idx=ci, positions=expand_positions,
                    multiplier_label=0.0,
                )
            else:
                set_val = config.max_activation * mult
                print(f"  [ctrl {ctrl_feat}] expand mult={mult} ...", end=" ", flush=True)
                row = _run_condition(
                    ctrl_feat, "set", set_val, "expand",
                    is_control=True, control_idx=ci, positions=expand_positions,
                    multiplier_label=mult,
                )
            print(f"{row['n_mutations']} mutations"
                  + (f", pLDDT={row.get('plddt', 'N/A')}" if do_fold else ""))
            results.append(row)

    # --- Generate plots for expand conditions ---
    if do_fold and original_pdb is not None:
        plot_dir = (output_dir or Path("results/steering")) / "plots"

        # Compute original geometric profile once
        try:
            geom_orig = compute_per_residue_geom_metric(original_pdb, config.target_metric, sequence=sequence)
        except Exception as e:
            print(f"  Warning: geometry extraction failed for original: {e}")
            geom_orig = None

        target_expand_rows = [r for r in results if not r["is_control"] and r["style"] == "expand"]
        for row in target_expand_rows:
            mult = row["multiplier"]
            pdb_steered = row.get("_pdb_text")
            acts_steered = row.get("_activations")
            if pdb_steered is None or acts_steered is None:
                continue

            # 3D backbone comparison
            try:
                save_3d = plot_dir / f"feat{config.feature_idx}_3d_{mult:.1f}.png"
                plot_steering_3d(
                    original_pdb, pdb_steered,
                    original_activations, acts_steered,
                    config.feature_idx, mult, "expand", config, save_3d,
                )
                print(f"  Saved 3D plot: {save_3d}")
            except Exception as e:
                print(f"  Warning: 3D plot failed for mult={mult}: {e}")

            # Per-residue profile comparison
            if geom_orig is not None:
                try:
                    steered_seq = row.get("steered_sequence", sequence)
                    geom_steered = compute_per_residue_geom_metric(
                        pdb_steered, config.target_metric, sequence=steered_seq,
                    )
                    save_prof = plot_dir / f"feat{config.feature_idx}_profile_{mult:.1f}.png"
                    plot_residue_profiles(
                        original_activations, acts_steered,
                        geom_orig, geom_steered,
                        config.feature_idx, mult, "expand", config, save_prof,
                    )
                    print(f"  Saved profile plot: {save_prof}")
                except Exception as e:
                    print(f"  Warning: profile plot failed for mult={mult}: {e}")

    return results


# ══════════════════════════════════════════════════════════════════
#  Analysis & Reporting
# ══════════════════════════════════════════════════════════════════

def analyze_results(all_results, experiments):
    """Compute dose-response statistics and print report."""
    summary = {}

    for config in experiments:
        feat_id = config.feature_idx
        feat_rows = [r for r in all_results if r["experiment_feature"] == feat_id]

        # Target feature, clamp style
        target_scale = [r for r in feat_rows if not r["is_control"] and r["style"] == "scale"]
        target_expand = [r for r in feat_rows if not r["is_control"] and r["style"] == "expand"]
        controls = [r for r in feat_rows if r["is_control"]]

        print(f"\n{'=' * 80}")
        print(f"  Feature {feat_id}: {config.structural_category}")
        print(f"  Target metric: {config.target_metric} (predicted: {config.predicted_direction})")
        print(f"{'=' * 80}")

        has_geometry = "target_mean" in target_expand[0] if target_expand else False

        # --- Dose-response table ---
        print(f"\n  {'Mult':>6} {'Style':>6} {'Muts':>5} {'%Chg':>6}", end="")
        if has_geometry:
            print(f" {'pLDDT':>6} {'MetricMean':>11} {'Delta':>8}", end="")
        print()
        print(f"  {'-'*6} {'-'*6} {'-'*5} {'-'*6}", end="")
        if has_geometry:
            print(f" {'-'*6} {'-'*11} {'-'*8}", end="")
        print()

        # Get baseline (mult=1.0, expand)
        baseline_metric = None
        if has_geometry:
            baseline_rows = [r for r in target_expand if abs(r["multiplier"] - 1.0) < 0.01]
            if baseline_rows:
                baseline_metric = baseline_rows[0]["target_mean"]

        for row in sorted(target_scale + target_expand, key=lambda r: (r["style"], r["multiplier"])):
            print(f"  {row['multiplier']:>6.1f} {row['style']:>6} {row['n_mutations']:>5} {row['pct_changed']:>5.1f}%", end="")
            if has_geometry:
                delta = row["target_mean"] - baseline_metric if baseline_metric is not None else 0
                print(f" {row.get('plddt', 0):>6.2f} {row['target_mean']:>11.6f} {delta:>+8.6f}", end="")
            print()

        # --- Spearman correlation (clamp) ---
        feat_summary = {
            "feature_idx": feat_id,
            "structural_category": config.structural_category,
            "target_metric": config.target_metric,
            "predicted_direction": config.predicted_direction,
        }

        if has_geometry and len(target_expand) >= 3:
            mults = [r["multiplier"] for r in target_expand]
            metrics = [r["target_mean"] for r in target_expand]
            rho, pval = stats.spearmanr(mults, metrics)

            direction_correct = (
                (config.predicted_direction == "decrease" and rho < 0) or
                (config.predicted_direction == "increase" and rho > 0)
            )

            print(f"\n  Spearman (expand, all): rho={rho:.4f}, p={pval:.4f}")
            print(f"  Direction: {'CORRECT' if direction_correct else 'WRONG'} "
                  f"(predicted={config.predicted_direction}, observed={'decrease' if rho < 0 else 'increase'})")

            feat_summary["spearman_rho"] = round(rho, 4)
            feat_summary["spearman_p"] = round(pval, 4)
            feat_summary["direction_correct"] = direction_correct

            # Also compute Spearman on moderate range only (mult <= 2.0, where pLDDT is reasonable)
            moderate_clamp = [r for r in target_expand if r["multiplier"] <= 2.0]
            if len(moderate_clamp) >= 3:
                m_mults = [r["multiplier"] for r in moderate_clamp]
                m_metrics = [r["target_mean"] for r in moderate_clamp]
                m_rho, m_pval = stats.spearmanr(m_mults, m_metrics)
                m_dir_correct = (
                    (config.predicted_direction == "decrease" and m_rho < 0) or
                    (config.predicted_direction == "increase" and m_rho > 0)
                )
                print(f"  Spearman (expand, mult<=2): rho={m_rho:.4f}, p={m_pval:.4f}, "
                      f"direction={'CORRECT' if m_dir_correct else 'WRONG'}")
                feat_summary["spearman_rho_moderate"] = round(m_rho, 4)
                feat_summary["direction_correct_moderate"] = m_dir_correct

            # Control correlations
            ctrl_rhos = []
            for ci in range(N_RANDOM_CONTROLS):
                ctrl_rows = [r for r in controls if r["control_idx"] == ci]
                if len(ctrl_rows) >= 3:
                    c_mults = [r["multiplier"] for r in ctrl_rows]
                    c_metrics = [r["target_mean"] for r in ctrl_rows]
                    c_rho, _ = stats.spearmanr(c_mults, c_metrics)
                    ctrl_rhos.append(c_rho)

            if ctrl_rhos:
                ctrl_mean = np.mean(ctrl_rhos)
                ctrl_std = np.std(ctrl_rhos)
                print(f"  Controls: mean_rho={ctrl_mean:.4f} +/- {ctrl_std:.4f}")
                separated = abs(rho) > abs(ctrl_mean) + 2 * ctrl_std
                print(f"  Separation: {'YES' if separated else 'NO'} "
                      f"(|target_rho|={abs(rho):.4f} vs threshold={abs(ctrl_mean) + 2*ctrl_std:.4f})")
                feat_summary["ctrl_rho_mean"] = round(ctrl_mean, 4)
                feat_summary["ctrl_rho_std"] = round(ctrl_std, 4)
                feat_summary["ctrl_rhos"] = [round(r, 4) for r in ctrl_rhos]
                feat_summary["separated_from_controls"] = separated

        # Scale style Spearman
        if has_geometry and len(target_scale) >= 3:
            mults_s = [r["multiplier"] for r in target_scale]
            metrics_s = [r["target_mean"] for r in target_scale]
            rho_s, pval_s = stats.spearmanr(mults_s, metrics_s)
            print(f"  Spearman (scale): rho={rho_s:.4f}, p={pval_s:.4f}")
            feat_summary["spearman_rho_scale"] = round(rho_s, 4)

        summary[feat_id] = feat_summary

    return summary


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SAE Geometry Steering Experiments")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if omitted)")
    parser.add_argument("--output-dir", type=str, default="results/steering", help="Output directory")
    parser.add_argument("--no-fold", action="store_true", help="Skip ESMFold (sequence-only mode)")
    parser.add_argument("--fold-api", action="store_true",
                        help="Use ESM Atlas API for folding (no local model needed)")
    parser.add_argument("--features", type=str, default=None,
                        help="Comma-separated feature IDs to test (default: all 3)")
    parser.add_argument("--sae-dir", type=str, default=str(DEFAULT_SAE_DIR))
    parser.add_argument("--esm-model", type=str, default=DEFAULT_ESM_MODEL)
    parser.add_argument("--layer", type=int, default=DEFAULT_ESM_LAYER)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = args.device or get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter experiments if --features specified
    experiments = EXPERIMENTS
    if args.features:
        requested = set(int(x) for x in args.features.split(","))
        experiments = [e for e in EXPERIMENTS if e.feature_idx in requested]
        if not experiments:
            print(f"Error: no matching experiments for features {requested}")
            sys.exit(1)

    # Load sequences
    print("\nLoading sequences ...")
    with open(DATA_DIR / "sequences.json") as f:
        all_sequences = json.load(f)

    for exp in experiments:
        if exp.accession not in all_sequences:
            print(f"Error: {exp.accession} not found in sequences.json")
            sys.exit(1)

    # Select control features
    gp_ids = get_geometry_primary_feature_ids()
    control_features = select_random_control_features(N_RANDOM_CONTROLS, exclude=gp_ids)
    print(f"Control features (non-geometry-primary): {control_features}")

    # Load models
    print("\nLoading models ...")
    tokenizer, esm_model, sae = load_models(args.sae_dir, args.esm_model, device)

    fold_mode = None
    fold_models = None
    if not args.no_fold:
        if args.fold_api:
            fold_mode = "api"
            print("\n  Using ESMFold API (api.esmatlas.com)")
        else:
            fold_mode = "local"
            print("\nLoading ESMFold ...")
            fold_models = load_esmfold(device)

    # Run experiments
    print(f"\n{'#' * 80}")
    print(f"  Running {len(experiments)} experiments x {len(MULTIPLIERS)} multipliers x 2 styles")
    print(f"  + {N_RANDOM_CONTROLS} random controls per experiment")
    fold_label = "NO" if args.no_fold else (f"API" if args.fold_api else "LOCAL")
    print(f"  Fold: {fold_label}")
    print(f"{'#' * 80}")

    all_results = []
    for config in experiments:
        sequence = all_sequences[config.accession]
        results = run_feature_experiment(
            tokenizer, esm_model, sae, fold_mode, fold_models,
            config, sequence, control_features,
            args.layer, device, do_fold=not args.no_fold,
            output_dir=output_dir,
        )
        all_results.extend(results)

    # Analysis
    print(f"\n\n{'#' * 80}")
    print(f"  ANALYSIS")
    print(f"{'#' * 80}")
    summary = analyze_results(all_results, experiments)

    # Save results (strip steered sequences from main results to keep file small)
    results_for_save = []
    for r in all_results:
        row = dict(r)
        row.pop("steered_sequence", None)  # too large for JSON
        row.pop("_pdb_text", None)  # internal, not for saving
        row.pop("_activations", None)  # internal numpy array
        # Convert whole_protein dict values to be JSON-safe
        if "whole_protein" in row and row["whole_protein"] is not None:
            row["whole_protein"] = {k: round(v, 6) for k, v in row["whole_protein"].items()}
        results_for_save.append(row)

    with open(output_dir / "steering_results.json", "w") as f:
        json.dump(results_for_save, f, indent=2)

    # Convert numpy types for JSON serialization
    def _json_safe(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _clean_for_json(d):
        if isinstance(d, dict):
            return {k: _clean_for_json(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_clean_for_json(v) for v in d]
        return _json_safe(d)

    with open(output_dir / "steering_summary.json", "w") as f:
        json.dump(_clean_for_json(summary), f, indent=2)

    print(f"\n  Results saved to {output_dir}/steering_results.json")
    print(f"  Summary saved to {output_dir}/steering_summary.json")

    # Final verdict
    if any("spearman_rho" in s for s in summary.values()):
        n_correct = sum(1 for s in summary.values() if s.get("direction_correct", False))
        n_correct_mod = sum(1 for s in summary.values() if s.get("direction_correct_moderate", False))
        n_separated = sum(1 for s in summary.values() if s.get("separated_from_controls", False))
        n_total = len(summary)
        print(f"\n  Direction correct (all multipliers): {n_correct}/{n_total}")
        print(f"  Direction correct (moderate, mult<=2): {n_correct_mod}/{n_total}")
        print(f"  Separated from controls: {n_separated}/{n_total}")


if __name__ == "__main__":
    main()
