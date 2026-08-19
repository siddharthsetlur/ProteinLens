#!/usr/bin/env python3
"""Benchmark contact-focused hidden-state SAE ablations with ESM2 predict_contacts.

This script keeps the amino-acid sequence fixed and measures counterfactual
changes through the ESM2 contact predictor directly:

1. Select geometry-primary SAE features whose top descriptor is contact-related.
2. Choose a top-activating protein and local activated patch for each feature.
3. Build either a grouped local lesion or a single-node target-only lesion,
   plus a matched control bundle with similar activation magnitude.
4. Zero the chosen SAE features at the patch inside one hidden layer.
5. Resume the remaining ESM2 layers from the modified hidden state.
6. Stitch the unchanged early-layer attentions with the recomputed later-layer
   attentions and run the built-in ``predict_contacts`` readout.
7. Score how strongly local contact metrics shift relative to matched controls.

Compared with the structure-module benchmark, this avoids any folding stack and
tests whether contact-sensitive latent features causally affect the model's own
contact-prediction behaviour.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TORCH", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

import benchmark_geometry_causal_ablation as bgca  # noqa: E402
from intervene_and_fold import decode_and_build_hidden, load_pipeline_models  # noqa: E402
from proteinlens.analysis.feature_clusters import _get_decoder_weights  # noqa: E402
from proteinlens.analysis.geometry.residue_features import FEATURE_GROUPS  # noqa: E402


CONTACT_FEATURE_NAMES = list(FEATURE_GROUPS["contact"])
TARGET_CONTACT_METRIC_MAP = {
    "contact_density_8A": "patch_expected_contacts_per_residue",
    "contact_density_12A": "patch_expected_contacts_per_residue",
    "long_range_contacts_8A": "patch_expected_long_contacts_per_residue",
    "long_range_contacts_12A": "patch_expected_long_contacts_per_residue",
    "max_seq_sep_contact_8A": "patch_max_seq_sep_prob_ge_threshold",
    "mean_seq_sep_contact_8A": "patch_weighted_mean_seq_sep",
    "contact_order_local": "patch_fraction_long_contact_mass",
    # This is only a proxy, since contact probabilities do not recover minimum
    # Euclidean distance directly.
    "min_spatial_dist_long": "patch_max_long_contact_prob",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark grouped or single-node hidden-state SAE lesions on the "
            "ESM2 contact predictor for contact-related geometry features."
        )
    )
    p.add_argument("--data-dir", type=Path, default=Path("feature_data_cluster"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/contact_predictor_ablation"),
    )
    p.add_argument("--sae-dir", default=str(ROOT / "trained_models" / "fiery-sweep"))
    p.add_argument("--esm-model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--device", default=None, help="Device for ESM2 + SAE (default: auto)")
    p.add_argument("--feature-ids", nargs="+", type=int, default=None)
    p.add_argument(
        "--feature-ids-file",
        type=Path,
        default=None,
        help="Optional newline- or comma-separated file of feature ids to run.",
    )
    p.add_argument(
        "--accessions",
        nargs="+",
        default=None,
        help="Optional subset of protein accessions to include.",
    )
    p.add_argument(
        "--accessions-file",
        type=Path,
        default=None,
        help="Optional newline- or comma-separated file of protein accessions to include.",
    )
    p.add_argument("--max-features", type=int, default=6)
    p.add_argument(
        "--proteins-per-feature",
        type=int,
        default=1,
        help="Number of top-activating proteins to evaluate per selected feature.",
    )
    p.add_argument(
        "--top-sequence-pool",
        type=int,
        default=20,
        help="How many top_sequences entries to scan per feature while building case candidates.",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional total cap on expanded cases after feature/protein selection.",
    )
    p.add_argument("--max-seq-len", type=int, default=350)
    p.add_argument("--min-concordance-f1", type=float, default=0.60)
    p.add_argument("--min-geom-pr-auc", type=float, default=0.80)
    p.add_argument("--max-position-f1", type=float, default=0.10)
    p.add_argument(
        "--contact-feature-names",
        nargs="+",
        choices=CONTACT_FEATURE_NAMES,
        default=list(CONTACT_FEATURE_NAMES),
        help="Only benchmark features whose top geometric descriptor matches one of these contact metrics.",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=5,
        help="Number of SAE features to ablate together. Use 1, or pass --single-node-only, for a target-only lesion.",
    )
    p.add_argument(
        "--single-node-only",
        action="store_true",
        help="Ablate only the selected target SAE feature while keeping a matched single-feature control.",
    )
    p.add_argument(
        "--intervention-radius",
        type=int,
        default=2,
        help="Expand activated positions by this radius for the actual lesion.",
    )
    p.add_argument(
        "--context-radius",
        type=int,
        default=6,
        help="Expand activated positions by this radius when identifying redundant features.",
    )
    p.add_argument(
        "--min-seq-sep-short",
        type=int,
        default=3,
        help="Minimum sequence separation counted as a non-trivial contact.",
    )
    p.add_argument(
        "--min-seq-sep-long",
        type=int,
        default=12,
        help="Minimum sequence separation counted as long-range for contact-order metrics.",
    )
    p.add_argument(
        "--contact-prob-threshold",
        type=float,
        default=0.5,
        help="Threshold used for the max-sequence-separation contact proxy.",
    )
    p.add_argument(
        "--save-contact-maps",
        action="store_true",
        help="Save original and candidate contact maps as compressed .npz files.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve and save the selected cases; do not load models or run interventions.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def resolve_runtime_device(requested: Optional[str]) -> str:
    """Resolve a safe torch device for this runtime.

    The benchmark only needs one device flag, so we allow explicit requests but
    gracefully fall back when a backend is unavailable.
    """

    if requested is None:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends.mps, "is_available", lambda: False)():
            return "mps"
        return "cpu"

    requested = str(requested).strip().lower()
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print(
            "  Requested --device cuda, but this PyTorch build has no CUDA support. "
            "Falling back to cpu."
        )
        return "cpu"
    if requested == "mps":
        if getattr(torch.backends.mps, "is_available", lambda: False)():
            return "mps"
        print(
            "  Requested --device mps, but MPS is unavailable in this runtime. "
            "Falling back to cpu."
        )
        return "cpu"
    return requested


def select_contact_cases(args: argparse.Namespace) -> list[bgca.CaseSpec]:
    gp = bgca.load_json(args.data_dir / "geometry_primary_analysis.json")["features"]
    feat_dir = args.data_dir / "features"
    geo_dir = args.data_dir / "geometry_enrichment"

    explicit_feature_ids = bgca.resolve_requested_feature_ids(args)
    explicit = set(explicit_feature_ids)
    allowed_accessions = bgca.resolve_requested_accessions(args)
    allowed_contact_names = set(args.contact_feature_names)

    feature_case_pools: list[dict[str, Any]] = []

    for fid_str, row in gp.items():
        fid = int(fid_str)
        top_geom = row.get("top_geometric_feature")
        if top_geom not in allowed_contact_names:
            continue
        if explicit and fid not in explicit:
            continue
        if not explicit:
            if not row.get("is_geometry_primary"):
                continue
            if row.get("concordance_f1", 0.0) < args.min_concordance_f1:
                continue
            if row.get("geom_pr_auc", 0.0) < args.min_geom_pr_auc:
                continue
            if row.get("position_f1", 1.0) > args.max_position_f1:
                continue

        cases = bgca._build_feature_case_candidates(  # noqa: SLF001
            fid=fid,
            row=row,
            args=args,
            feat_dir=feat_dir,
            geo_dir=geo_dir,
            allowed_accessions=allowed_accessions,
        )
        if not cases:
            continue

        feature_case_pools.append(
            {
                "feature_id": fid,
                "top_geometric_feature": cases[0].top_geometric_feature,
                "concordance_f1": cases[0].concordance_f1,
                "geom_pr_auc": cases[0].geom_pr_auc,
                "sequence_length": min(case.sequence_length for case in cases),
                "cases": cases,
            }
        )

    if explicit_feature_ids:
        pools_by_fid = {pool["feature_id"]: pool for pool in feature_case_pools}
        selected: list[bgca.CaseSpec] = []
        for fid in explicit_feature_ids:
            selected.extend(pools_by_fid.get(fid, {}).get("cases", []))
        if args.max_cases is not None:
            return selected[: args.max_cases]
        return selected

    by_geom: dict[str, list[dict[str, Any]]] = {}
    for pool in feature_case_pools:
        by_geom.setdefault(pool["top_geometric_feature"], []).append(pool)

    for pools in by_geom.values():
        pools.sort(
            key=lambda pool: (
                -pool["concordance_f1"],
                -pool["geom_pr_auc"],
                pool["sequence_length"],
                pool["feature_id"],
            )
        )

    geom_order = sorted(
        by_geom,
        key=lambda name: (
            -by_geom[name][0]["concordance_f1"],
            -by_geom[name][0]["geom_pr_auc"],
            by_geom[name][0]["sequence_length"],
            name,
        ),
    )

    selected_pools: list[dict[str, Any]] = []
    while len(selected_pools) < args.max_features:
        added = False
        for name in geom_order:
            if not by_geom[name]:
                continue
            selected_pools.append(by_geom[name].pop(0))
            added = True
            if len(selected_pools) >= args.max_features:
                break
        if not added:
            break

    selected_cases: list[bgca.CaseSpec] = []
    for pool in selected_pools:
        selected_cases.extend(pool["cases"])
    if args.max_cases is not None:
        return selected_cases[: args.max_cases]
    return selected_cases


def extract_hidden_states_and_attentions(
    esm_model,
    tokenizer,
    sequence: str,
    layer_idx: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run ESM2 once and return logits, one hidden state, tokens, mask, and attentions."""

    if layer_idx < 0 or layer_idx >= esm_model.config.num_hidden_layers:
        raise ValueError(
            f"Layer index {layer_idx} is out of range for {esm_model.config.num_hidden_layers} ESM layers."
        )

    inputs = tokenizer(sequence, return_tensors="pt", padding=False)
    token_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = esm_model(
            token_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    if outputs.attentions is None:
        raise RuntimeError(
            "ESM2 did not return attentions. Ensure the model uses the 'eager' "
            "attention implementation before running this benchmark."
        )

    hidden_states = tuple(state.detach() for state in outputs.hidden_states)
    attentions = tuple(attn.detach() for attn in outputs.attentions)
    return outputs.logits.detach(), hidden_states[layer_idx], token_ids, attn_mask, attentions


def resume_esm_attentions_from_hidden(
    esm_model,
    modified_hidden: torch.Tensor,
    token_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    from_layer: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Resume ESM2 from an overridden hidden state and collect later-layer attentions.

    ``from_layer`` follows the same convention as the existing intervention
    scripts: it indexes ``outputs.hidden_states[from_layer]``, so the resumed
    encoder pass starts at ``encoder.layer[from_layer]``.
    """

    hidden_states = modified_hidden.detach()
    resumed_attentions: list[torch.Tensor] = []

    with torch.no_grad():
        ext_mask = esm_model.esm.get_extended_attention_mask(attn_mask, token_ids.shape)
        for layer_module in esm_model.esm.encoder.layer[from_layer:]:
            hidden_states_ln = layer_module.attention.LayerNorm(hidden_states)
            attn_output, attn_weights = layer_module.attention.self(
                hidden_states_ln,
                attention_mask=ext_mask,
                head_mask=None,
            )
            attention_output = layer_module.attention.output(attn_output, hidden_states)
            attention_output_ln = layer_module.LayerNorm(attention_output)
            intermediate_output = layer_module.intermediate(attention_output_ln)
            hidden_states = layer_module.output(intermediate_output, attention_output)
            resumed_attentions.append(attn_weights.detach())

        final_hidden = esm_model.esm.encoder.emb_layer_norm_after(hidden_states)

    return final_hidden.detach(), tuple(resumed_attentions)


def predict_contacts_from_attention_stack(
    esm_model,
    token_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    attentions: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Run the built-in contact head on a full per-layer attention stack."""

    if not attentions:
        raise ValueError("Need at least one attention tensor to predict contacts.")
    attns = torch.stack(tuple(attentions), dim=1)
    mask = attn_mask.to(dtype=attns.dtype)
    attns = attns * mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    attns = attns * mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)
    with torch.no_grad():
        contacts = esm_model.esm.contact_head(token_ids, attns)
    return contacts.detach()


def predict_contacts_with_hidden_override(
    esm_model,
    original_attentions: tuple[torch.Tensor, ...],
    modified_hidden: torch.Tensor,
    token_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    from_layer: int,
) -> dict[str, Any]:
    """Predict contacts after resuming ESM2 from a modified hidden state."""

    final_hidden, resumed_attentions = resume_esm_attentions_from_hidden(
        esm_model,
        modified_hidden,
        token_ids,
        attn_mask,
        from_layer,
    )
    stitched_attentions = tuple(original_attentions[:from_layer]) + tuple(resumed_attentions)
    if len(stitched_attentions) != esm_model.config.num_hidden_layers:
        raise RuntimeError(
            "Stitched attention stack has the wrong length: "
            f"expected {esm_model.config.num_hidden_layers}, got {len(stitched_attentions)}."
        )
    contacts = predict_contacts_from_attention_stack(
        esm_model,
        token_ids,
        attn_mask,
        stitched_attentions,
    )
    logits = esm_model.lm_head(final_hidden).detach()
    return {
        "contacts": contacts,
        "attentions": stitched_attentions,
        "logits": logits,
    }


def summarize_patch_contact_map(
    contact_map: np.ndarray,
    patch_positions: list[int],
    *,
    min_seq_sep_short: int,
    min_seq_sep_long: int,
    prob_threshold: float,
) -> dict[str, float]:
    """Summarize a contact map using per-patch expected contact statistics."""

    cm = np.asarray(contact_map, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Expected a square 2D contact map, got shape {cm.shape}.")

    n = cm.shape[0]
    patch = sorted({int(pos) for pos in patch_positions if 0 <= int(pos) < n})
    if not patch:
        return {
            "patch_expected_contacts_per_residue": float("nan"),
            "patch_expected_long_contacts_per_residue": float("nan"),
            "patch_fraction_long_contact_mass": float("nan"),
            "patch_weighted_mean_seq_sep": float("nan"),
            "patch_max_seq_sep_prob_ge_threshold": float("nan"),
            "patch_max_long_contact_prob": float("nan"),
            "patch_n_positions": 0.0,
        }

    short_masses: list[float] = []
    long_masses: list[float] = []
    fraction_long: list[float] = []
    max_spans: list[float] = []
    max_long_probs: list[float] = []
    total_short_mass = 0.0
    weighted_sep_num = 0.0

    residue_index = np.arange(n, dtype=int)
    for pos in patch:
        seps = np.abs(residue_index - pos)
        short_mask = seps >= min_seq_sep_short
        long_mask = seps >= min_seq_sep_long

        short_probs = cm[pos, short_mask]
        long_probs = cm[pos, long_mask]
        short_mass = float(short_probs.sum())
        long_mass = float(long_probs.sum())

        short_masses.append(short_mass)
        long_masses.append(long_mass)
        fraction_long.append(long_mass / short_mass if short_mass > 1e-8 else 0.0)
        max_long_probs.append(float(long_probs.max()) if long_probs.size else 0.0)

        qualifying = short_mask & (cm[pos] >= prob_threshold)
        qualifying_seps = seps[qualifying]
        max_spans.append(float(qualifying_seps.max()) if qualifying_seps.size else 0.0)

        total_short_mass += short_mass
        if short_mass > 1e-8:
            weighted_sep_num += float((cm[pos, short_mask] * seps[short_mask]).sum())

    weighted_mean_sep = weighted_sep_num / total_short_mass if total_short_mass > 1e-8 else 0.0
    return {
        "patch_expected_contacts_per_residue": float(np.mean(short_masses)),
        "patch_expected_long_contacts_per_residue": float(np.mean(long_masses)),
        "patch_fraction_long_contact_mass": float(np.mean(fraction_long)),
        "patch_weighted_mean_seq_sep": float(weighted_mean_sep),
        "patch_max_seq_sep_prob_ge_threshold": float(np.mean(max_spans)),
        "patch_max_long_contact_prob": float(np.mean(max_long_probs)),
        "patch_n_positions": float(len(patch)),
    }


def resolve_target_contact_metric(case: bgca.CaseSpec) -> tuple[str, bool]:
    metric_name = TARGET_CONTACT_METRIC_MAP.get(
        case.top_geometric_feature,
        "patch_expected_contacts_per_residue",
    )
    is_proxy = case.top_geometric_feature == "min_spatial_dist_long"
    return metric_name, is_proxy


def compute_contact_map_metrics(
    original_contact_map: np.ndarray,
    candidate_contact_map: np.ndarray,
    case: bgca.CaseSpec,
    *,
    min_seq_sep_short: int,
    min_seq_sep_long: int,
    prob_threshold: float,
) -> dict[str, float | str | bool]:
    """Compare original vs candidate contact maps around the activated patch."""

    orig = np.asarray(original_contact_map, dtype=float)
    cand = np.asarray(candidate_contact_map, dtype=float)
    if orig.shape != cand.shape:
        raise ValueError(f"Contact-map shape mismatch: {orig.shape} vs {cand.shape}.")

    target_metric_name, target_metric_is_proxy = resolve_target_contact_metric(case)
    orig_summary = summarize_patch_contact_map(
        orig,
        case.patch_positions,
        min_seq_sep_short=min_seq_sep_short,
        min_seq_sep_long=min_seq_sep_long,
        prob_threshold=prob_threshold,
    )
    cand_summary = summarize_patch_contact_map(
        cand,
        case.patch_positions,
        min_seq_sep_short=min_seq_sep_short,
        min_seq_sep_long=min_seq_sep_long,
        prob_threshold=prob_threshold,
    )

    n = orig.shape[0]
    patch = sorted({int(pos) for pos in case.patch_positions if 0 <= int(pos) < n})
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    patch_row_mask = np.zeros((n, n), dtype=bool)
    if patch:
        patch_row_mask[patch, :] = True
    patch_short_mask = patch_row_mask & (sep >= min_seq_sep_short)
    patch_long_mask = patch_row_mask & (sep >= min_seq_sep_long)

    tri_i, tri_j = np.triu_indices(n, k=1)
    global_short_mask = np.abs(tri_i - tri_j) >= min_seq_sep_short
    diff = cand - orig

    out: dict[str, float | str | bool] = {
        "target_contact_metric_name": target_metric_name,
        "target_contact_metric_is_proxy": target_metric_is_proxy,
    }

    for key, orig_value in orig_summary.items():
        cand_value = float(cand_summary[key])
        orig_value = float(orig_value)
        out[f"{key}_original"] = orig_value
        out[f"{key}_candidate"] = cand_value
        out[f"{key}_delta"] = cand_value - orig_value

    target_orig = float(orig_summary[target_metric_name])
    target_cand = float(cand_summary[target_metric_name])
    out["target_contact_metric_original"] = target_orig
    out["target_contact_metric_candidate"] = target_cand
    out["target_contact_metric_delta"] = target_cand - target_orig
    out["target_contact_metric_abs_delta"] = abs(target_cand - target_orig)

    if np.any(patch_short_mask):
        out["patch_contact_l1_delta"] = float(np.mean(np.abs(diff[patch_short_mask])))
        out["patch_contact_signed_delta"] = float(np.mean(diff[patch_short_mask]))
    else:
        out["patch_contact_l1_delta"] = float("nan")
        out["patch_contact_signed_delta"] = float("nan")

    if np.any(patch_long_mask):
        out["patch_long_contact_l1_delta"] = float(np.mean(np.abs(diff[patch_long_mask])))
        out["patch_long_contact_signed_delta"] = float(np.mean(diff[patch_long_mask]))
    else:
        out["patch_long_contact_l1_delta"] = float("nan")
        out["patch_long_contact_signed_delta"] = float("nan")

    if np.any(global_short_mask):
        global_diff = diff[tri_i[global_short_mask], tri_j[global_short_mask]]
        out["global_contact_l1_delta"] = float(np.mean(np.abs(global_diff)))
        out["global_contact_signed_delta"] = float(np.mean(global_diff))
    else:
        out["global_contact_l1_delta"] = float("nan")
        out["global_contact_signed_delta"] = float("nan")

    return out


def summarize_contact_case(
    case: bgca.CaseSpec,
    lesion: bgca.InterventionSpec,
    control: bgca.InterventionSpec,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def aggregate(label: str, key: str) -> float:
        vals = [
            float(r[key])
            for r in rows
            if r["intervention_label"] == label
            and key in r
            and np.isfinite(float(r[key]))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    metric_name = rows[0].get("target_contact_metric_name", "") if rows else ""
    metric_is_proxy = bool(rows[0].get("target_contact_metric_is_proxy", False)) if rows else False

    return {
        "case_dir": f"f{case.feature_id:04d}_{case.accession}",
        "feature_id": case.feature_id,
        "accession": case.accession,
        "protein_rank": case.protein_rank,
        "selection_source": case.selection_source,
        "max_activation": case.max_activation,
        "mean_activation": case.mean_activation,
        "top_geometric_feature": case.top_geometric_feature,
        "structural_category": case.structural_category,
        "intervention_mode": "single_node" if len(lesion.feature_ids) == 1 else "grouped",
        "patch_positions": case.patch_positions,
        "lesion_feature_ids": lesion.feature_ids,
        "lesion_feature_count": len(lesion.feature_ids),
        "control_feature_ids": control.feature_ids,
        "control_feature_count": len(control.feature_ids),
        "target_contact_metric_name": metric_name,
        "target_contact_metric_is_proxy": metric_is_proxy,
        "group_mean_target_contact_abs_delta": aggregate("group_lesion", "target_contact_metric_abs_delta"),
        "control_mean_target_contact_abs_delta": aggregate("matched_control", "target_contact_metric_abs_delta"),
        "group_mean_patch_contact_l1_delta": aggregate("group_lesion", "patch_contact_l1_delta"),
        "control_mean_patch_contact_l1_delta": aggregate("matched_control", "patch_contact_l1_delta"),
        "group_mean_patch_long_contact_l1_delta": aggregate("group_lesion", "patch_long_contact_l1_delta"),
        "control_mean_patch_long_contact_l1_delta": aggregate("matched_control", "patch_long_contact_l1_delta"),
        "group_mean_global_contact_l1_delta": aggregate("group_lesion", "global_contact_l1_delta"),
        "control_mean_global_contact_l1_delta": aggregate("matched_control", "global_contact_l1_delta"),
        "group_mean_kl_patch": aggregate("group_lesion", "mean_kl_patch"),
        "control_mean_kl_patch": aggregate("matched_control", "mean_kl_patch"),
        "paired_target_contact_abs_delta_margin": (
            aggregate("group_lesion", "target_contact_metric_abs_delta")
            - aggregate("matched_control", "target_contact_metric_abs_delta")
        ),
        "paired_patch_contact_l1_margin": (
            aggregate("group_lesion", "patch_contact_l1_delta")
            - aggregate("matched_control", "patch_contact_l1_delta")
        ),
        "paired_global_contact_l1_margin": (
            aggregate("group_lesion", "global_contact_l1_delta")
            - aggregate("matched_control", "global_contact_l1_delta")
        ),
    }


def run_case(
    case: bgca.CaseSpec,
    args: argparse.Namespace,
    tokenizer,
    esm_model,
    sae,
    decoder_weights: torch.Tensor,
    root_out: Path,
    case_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_dir = root_out / f"{case_index:02d}_f{case.feature_id:04d}_{case.accession}"
    case_dir.mkdir(parents=True, exist_ok=True)
    bgca.save_json(case_dir / "case_spec.json", asdict(case))
    bgca.write_fasta(case_dir / "original_sequence.fasta", [("original", case.sequence)])

    orig_logits, orig_hidden, token_ids, attn_mask, original_attentions = extract_hidden_states_and_attentions(
        esm_model,
        tokenizer,
        case.sequence,
        args.layer,
        bgca.auto_device(args.device),
    )
    seq_len = len(case.sequence)
    with torch.no_grad():
        residue_hidden = orig_hidden[0, 1 : seq_len + 1, :]
        normalised, original_norms = sae._normalize_input_and_get_norms(residue_hidden)
        features = sae.encode(normalised)

    lesion_spec, control_spec = bgca.build_intervention_specs(
        features,
        decoder_weights,
        case,
        args.group_size,
    )
    bgca.save_json(
        case_dir / "interventions.json",
        {
            "group_lesion": asdict(lesion_spec),
            "matched_control": asdict(control_spec),
        },
    )

    original_contacts = predict_contacts_from_attention_stack(
        esm_model,
        token_ids,
        attn_mask,
        original_attentions,
    )[0].detach().cpu().float().numpy()
    if args.save_contact_maps:
        np.savez_compressed(
            case_dir / "original_contacts.npz",
            contacts=original_contacts,
            patch_positions=np.asarray(case.patch_positions, dtype=int),
        )

    all_rows: list[dict[str, Any]] = []
    for spec in [lesion_spec, control_spec]:
        intervention_dir = case_dir / spec.label
        intervention_dir.mkdir(parents=True, exist_ok=True)

        features_mod = bgca.apply_zero_intervention(features, spec.feature_ids, spec.positions)
        modified_hidden = decode_and_build_hidden(
            sae,
            features_mod,
            orig_hidden,
            seq_len,
            original_norms,
        )

        hidden_delta = (
            (modified_hidden - orig_hidden)[0, 1 : seq_len + 1]
            .norm(dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        local_hidden_delta = float(np.mean(hidden_delta[spec.positions])) if spec.positions else 0.0
        context_hidden_delta = (
            float(np.mean(hidden_delta[case.context_positions]))
            if case.context_positions
            else 0.0
        )

        print(f"    {spec.label}: predicting contacts from direct hidden override ...")
        contact_pred = predict_contacts_with_hidden_override(
            esm_model,
            original_attentions,
            modified_hidden,
            token_ids,
            attn_mask,
            args.layer,
        )
        candidate_contacts = contact_pred["contacts"][0].detach().cpu().float().numpy()
        logit_effects = bgca.compute_logit_effects(
            orig_logits,
            contact_pred["logits"],
            case.patch_positions,
            case.lesion_positions,
        )
        contact_metrics = compute_contact_map_metrics(
            original_contacts,
            candidate_contacts,
            case,
            min_seq_sep_short=args.min_seq_sep_short,
            min_seq_sep_long=args.min_seq_sep_long,
            prob_threshold=args.contact_prob_threshold,
        )

        sample_dir = intervention_dir / "sample_01"
        sample_dir.mkdir(parents=True, exist_ok=True)
        bgca.write_fasta(sample_dir / "sequence.fasta", [("original", case.sequence)])
        if args.save_contact_maps:
            np.savez_compressed(
                sample_dir / "contacts.npz",
                candidate_contacts=candidate_contacts,
                delta_contacts=(candidate_contacts - original_contacts),
                patch_positions=np.asarray(case.patch_positions, dtype=int),
            )

        row: dict[str, Any] = {
            "case_dir": case_dir.name,
            "feature_id": case.feature_id,
            "accession": case.accession,
            "protein_rank": case.protein_rank,
            "selection_source": case.selection_source,
            "max_activation": case.max_activation,
            "mean_activation": case.mean_activation,
            "intervention_label": spec.label,
            "intervention_mode": "single_node" if len(spec.feature_ids) == 1 else "grouped",
            "intervention_feature_count": len(spec.feature_ids),
            "sample_index": 1,
            "sequence": case.sequence,
            "decode_mode": "direct_hidden_override",
            "n_mutations": 0,
            "mutation_positions": [],
            "mutable_positions": [],
            "patch_positions": case.patch_positions,
            "lesion_positions": case.lesion_positions,
            "feature_ids_zeroed": spec.feature_ids,
            "top_geometric_feature": case.top_geometric_feature,
            "local_hidden_delta": local_hidden_delta,
            "context_hidden_delta": context_hidden_delta,
            **logit_effects,
            **contact_metrics,
        }
        bgca.save_json(sample_dir / "metrics.json", row)
        all_rows.append(row)

    case_summary = summarize_contact_case(case, lesion_spec, control_spec, all_rows)
    bgca.save_json(case_dir / "case_summary.json", case_summary)
    return all_rows, case_summary


def main() -> None:
    args = parse_args()
    if args.min_seq_sep_short < 1:
        raise SystemExit("--min-seq-sep-short must be at least 1.")
    if args.min_seq_sep_long < args.min_seq_sep_short:
        raise SystemExit("--min-seq-sep-long must be at least --min-seq-sep-short.")
    if not 0.0 <= args.contact_prob_threshold <= 1.0:
        raise SystemExit("--contact-prob-threshold must lie in [0, 1].")
    args.device = resolve_runtime_device(args.device)
    bgca.set_seed(args.seed)
    intervention_mode = bgca.resolve_intervention_mode(args)
    requested_feature_ids = bgca.resolve_requested_feature_ids(args)
    requested_accessions = sorted(bgca.resolve_requested_accessions(args))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = select_contact_cases(args)
    if not cases:
        raise SystemExit("No contact-focused benchmark cases satisfied the selection criteria.")

    print()
    print("=" * 90)
    print("  ProteinLens — Contact Predictor Ablation Benchmark")
    print("=" * 90)
    print(f"  Selected {len(cases)} case(s) across {len({case.feature_id for case in cases})} feature(s)")
    print(f"  Contact descriptors: {', '.join(args.contact_feature_names)}")
    print(f"  ESM model: {args.esm_model}")
    if intervention_mode == "single_node":
        print("  Intervention mode: single-node lesion (target feature only)")
    else:
        print(f"  Intervention mode: grouped lesion ({args.group_size} features)")
    print(f"  Proteins per feature: {args.proteins_per_feature}")
    if requested_accessions:
        print(f"  Accession filter size: {len(requested_accessions)}")
    for idx, case in enumerate(cases, start=1):
        patch_str = ",".join(str(p) for p in case.patch_positions)
        print(
            f"    {idx:2d}. f/{case.feature_id:04d}  {case.accession}  "
            f"{case.top_geometric_feature}  len={case.sequence_length}  "
            f"rank={case.protein_rank}  patch={patch_str}"
        )

    case_rows = [asdict(case) for case in cases]
    bgca.save_table(args.output_dir / "selected_cases.csv", case_rows)
    bgca.save_json(args.output_dir / "selected_cases.json", case_rows)

    manifest = {
        "esm_model": args.esm_model,
        "sae_dir": args.sae_dir,
        "layer": args.layer,
        "group_size": args.group_size,
        "single_node_only": bool(args.single_node_only),
        "intervention_mode": intervention_mode,
        "feature_ids_requested": requested_feature_ids,
        "accessions_requested": requested_accessions,
        "proteins_per_feature": args.proteins_per_feature,
        "top_sequence_pool": args.top_sequence_pool,
        "max_cases": args.max_cases,
        "contact_feature_names": list(args.contact_feature_names),
        "min_seq_sep_short": args.min_seq_sep_short,
        "min_seq_sep_long": args.min_seq_sep_long,
        "contact_prob_threshold": args.contact_prob_threshold,
        "save_contact_maps": bool(args.save_contact_maps),
        "dry_run": bool(args.dry_run),
        "n_cases": len(cases),
        "n_unique_features": len({case.feature_id for case in cases}),
    }

    if args.dry_run:
        (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\n  Dry run complete. Saved selected cases to {args.output_dir}")
        print()
        return

    print(f"\n  Loading ESM2 + SAE on {args.device} ...")
    tokenizer, esm_model, sae = load_pipeline_models(args.sae_dir, args.esm_model, args.device)
    if hasattr(esm_model, "set_attn_implementation"):
        esm_model.set_attn_implementation("eager")
    decoder_weights = _get_decoder_weights(sae)

    if not hasattr(esm_model.esm, "contact_head"):
        raise RuntimeError(
            "The loaded ESM model does not expose a contact head, so predict_contacts cannot be used."
        )

    per_sample_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        print(f"\n[{case_index}/{len(cases)}] Running contact ablation for f/{case.feature_id:04d} on {case.accession} ...")
        rows, summary = run_case(
            case,
            args,
            tokenizer,
            esm_model,
            sae,
            decoder_weights,
            args.output_dir,
            case_index,
        )
        per_sample_rows.extend(rows)
        case_summaries.append(summary)

    bgca.save_table(args.output_dir / "per_sample_metrics.csv", per_sample_rows)
    bgca.save_json(args.output_dir / "per_sample_metrics.json", per_sample_rows)
    bgca.save_table(args.output_dir / "case_summaries.csv", case_summaries)
    bgca.save_json(args.output_dir / "case_summaries.json", case_summaries)

    manifest["dry_run"] = False
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n  Saved contact-predictor ablation outputs to {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
