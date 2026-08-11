#!/usr/bin/env python3
"""Benchmark grouped or single-node local SAE lesions for geometry-primary features.

This script builds a causal-ablation benchmark around the same local
geometry descriptors used in the feature-analysis pipeline.

For each selected geometry-primary feature, it:

1. Chooses a top activating protein and a local activated patch from
   ``feature_data_cluster/geometry_enrichment``.
2. Builds either a grouped lesion bundle consisting of the target feature plus
   a few locally redundant features on that patch, or a single-node lesion
   containing just the target feature.
3. Builds a same-size matched control bundle with similar local activation
   magnitude but low similarity to the target bundle.
4. Applies each intervention at the local patch, resumes the ESM forward
   pass, and samples local counterfactual sequences while keeping the rest
   of the protein fixed.
5. Folds the resulting sequences with ESMFold.
6. Scores local geometry deltas, backbone RMSD, and confidence changes,
   then writes per-sample and per-feature summaries.

The goal is to test whether geometry-relevant latent interventions produce
larger local geometric changes than matched controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Keep transformers on the torch path only.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TORCH", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

from intervene_and_fold import (  # noqa: E402
    decode_and_build_hidden,
    extract_hidden_states,
    inject_and_get_logits,
    load_esmfold,
    load_pipeline_models,
)
from proteinlens.analysis.feature_clusters import _get_decoder_weights  # noqa: E402
from proteinlens.analysis.geometry.classifiers import (  # noqa: E402
    compute_rmsd,
    kabsch_align,
)
from proteinlens.analysis.geometry.residue_features import (  # noqa: E402
    LOCAL_GEOM_NAMES,
    ca_backbone,
    compute_residue_profiles,
    detect_alpha_helices_from_ca,
    extract_local_feature_vector,
)


AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class CaseSpec:
    feature_id: int
    accession: str
    sequence: str
    sequence_length: int
    structural_category: str
    top_geometric_feature: str
    concordance_f1: float
    geom_pr_auc: float
    position_f1: float
    activation_threshold: float
    patch_positions: list[int]
    lesion_positions: list[int]
    context_positions: list[int]
    protein_rank: int = 1
    selection_source: str = "top_sequence"
    max_activation: float = float("nan")
    mean_activation: float = float("nan")


@dataclass
class InterventionSpec:
    label: str
    feature_ids: list[int]
    positions: list[int]
    diagnostics: list[dict[str, float]]


@dataclass
class GeometrySignatureSpec:
    feature_names: list[str]
    raw_importances: list[float]
    weights: list[float]
    scales: list[float]
    original_values: list[float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark grouped or single-node causal lesions for geometry-primary SAE features."
    )
    p.add_argument("--data-dir", type=Path, default=Path("feature_data_cluster"))
    p.add_argument("--output-dir", type=Path, default=Path("results/geometry_causal_benchmark"))
    p.add_argument("--sae-dir", default=str(ROOT / "trained_models" / "fiery-sweep"))
    p.add_argument("--esm-model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--device", default=None, help="Device for ESM2 + SAE (default: auto)")
    p.add_argument("--fold-device", default=None, help="Device for ESMFold (default: auto)")
    p.add_argument(
        "--original-structure-source",
        choices=["esmfold", "feature_data"],
        default="esmfold",
        help="Use ESMFold for the original sequence or reuse CA backbones stored in feature_data_cluster.",
    )
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
    p.add_argument("--min-concordance-f1", type=float, default=0.85)
    p.add_argument("--min-geom-pr-auc", type=float, default=0.90)
    p.add_argument("--max-position-f1", type=float, default=0.05)
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
        "--mutation-radius",
        type=int,
        default=6,
        help="Only these local residues are allowed to change during decoding.",
    )
    p.add_argument("--local-half-window", type=int, default=5)
    p.add_argument(
        "--signature-top-k",
        type=int,
        default=5,
        help="Number of top geometry descriptors to include in the feature-specific signature score.",
    )
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--top-k-aa", type=int, default=5)
    p.add_argument(
        "--disable-greedy-first",
        action="store_true",
        help="Do not force the first decoded sequence to use greedy argmax decoding.",
    )
    p.add_argument("--skip-fold", action="store_true", help="Generate sequences but do not fold them.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve and save the selected cases; do not load models or run interventions.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_token_file(path: Optional[Path]) -> list[str]:
    if path is None:
        return []
    tokens: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [token.strip() for token in line.replace(",", " ").split()]
        tokens.extend(token for token in parts if token)
    return tokens


def resolve_requested_feature_ids(args: argparse.Namespace) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for value in list(args.feature_ids or []):
        fid = int(value)
        if fid not in seen:
            seen.add(fid)
            ordered.append(fid)
    for token in _read_token_file(getattr(args, "feature_ids_file", None)):
        fid = int(token)
        if fid not in seen:
            seen.add(fid)
            ordered.append(fid)
    return ordered


def resolve_requested_accessions(args: argparse.Namespace) -> set[str]:
    accessions: set[str] = set()
    for accession in list(getattr(args, "accessions", None) or []):
        accession = accession.strip()
        if accession:
            accessions.add(accession)
    for token in _read_token_file(getattr(args, "accessions_file", None)):
        accession = token.strip()
        if accession:
            accessions.add(accession)
    return accessions


def resolve_intervention_mode(args: argparse.Namespace) -> str:
    if args.group_size < 1:
        raise SystemExit("--group-size must be at least 1.")
    if getattr(args, "proteins_per_feature", 1) < 1:
        raise SystemExit("--proteins-per-feature must be at least 1.")
    if getattr(args, "top_sequence_pool", 1) < 1:
        raise SystemExit("--top-sequence-pool must be at least 1.")
    if getattr(args, "single_node_only", False):
        args.group_size = 1
    return "single_node" if args.group_size == 1 else "grouped"


def auto_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def expand_positions(positions: list[int], radius: int, n_residues: int) -> list[int]:
    expanded: set[int] = set()
    for pos in positions:
        for idx in range(max(0, pos - radius), min(n_residues, pos + radius + 1)):
            expanded.add(idx)
    return sorted(expanded)


def _float_or_default(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _patch_positions_from_rows(
    seq_row: dict[str, Any],
    protein_row: Optional[dict[str, Any]],
    threshold: float,
) -> list[int]:
    patch_positions: list[int] = []
    if protein_row is not None:
        patch_positions = [
            int(item["position"])
            for item in protein_row.get("activated_positions", [])
            if "position" in item
        ]
    if not patch_positions:
        activations = seq_row.get("per_residue_activations", [])
        patch_positions = [idx for idx, value in enumerate(activations) if value >= threshold]
        if not patch_positions and activations:
            patch_positions = [int(np.argmax(np.asarray(activations, dtype=float)))]
    return sorted(set(patch_positions))


def _build_feature_case_candidates(
    *,
    fid: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    feat_dir: Path,
    geo_dir: Path,
    allowed_accessions: set[str],
) -> list[CaseSpec]:
    feat_path = feat_dir / f"{fid:04d}.json"
    geo_path = geo_dir / f"{fid:04d}.json"
    if not feat_path.exists() or not geo_path.exists():
        return []

    feat_json = load_json(feat_path)
    geo_json = load_json(geo_path)
    top_sequences = list(feat_json.get("top_sequences", []))[: max(1, args.top_sequence_pool)]
    if not top_sequences:
        return []

    top_geom = row.get("top_geometric_feature")
    if top_geom not in LOCAL_GEOM_NAMES:
        return []

    top_proteins = geo_json.get("plot_data", {}).get("top_proteins", [])
    protein_lookup = {
        protein.get("accession"): protein
        for protein in top_proteins
        if protein.get("accession")
    }
    threshold = float(geo_json.get("geometric_residue_level", {}).get("activation_threshold", 0.0))

    candidates: list[CaseSpec] = []
    seen_accessions: set[str] = set()

    for protein_rank, seq_row in enumerate(top_sequences, start=1):
        accession = str(seq_row.get("accession") or "").strip()
        if not accession or accession in seen_accessions:
            continue
        if allowed_accessions and accession not in allowed_accessions:
            continue

        protein_row = protein_lookup.get(accession)
        sequence = (protein_row or {}).get("sequence") or seq_row.get("sequence")
        if not sequence or len(sequence) > args.max_seq_len:
            continue

        patch_positions = _patch_positions_from_rows(seq_row, protein_row, threshold)
        if not patch_positions:
            continue

        seen_accessions.add(accession)
        candidates.append(
            CaseSpec(
                feature_id=fid,
                accession=accession,
                sequence=sequence,
                sequence_length=len(sequence),
                structural_category=row.get("structural_category", top_geom),
                top_geometric_feature=top_geom,
                concordance_f1=float(row.get("concordance_f1", 0.0)),
                geom_pr_auc=float(row.get("geom_pr_auc", 0.0)),
                position_f1=float(row.get("position_f1", 1.0)),
                activation_threshold=threshold,
                patch_positions=patch_positions,
                lesion_positions=expand_positions(patch_positions, args.intervention_radius, len(sequence)),
                context_positions=expand_positions(patch_positions, args.context_radius, len(sequence)),
                protein_rank=protein_rank,
                selection_source="top_protein" if protein_row is not None else "top_sequence",
                max_activation=_float_or_default(seq_row.get("max_activation")),
                mean_activation=_float_or_default(seq_row.get("mean_activation")),
            )
        )
        if len(candidates) >= max(1, args.proteins_per_feature):
            break

    return candidates


def select_cases(args: argparse.Namespace) -> list[CaseSpec]:
    gp = load_json(args.data_dir / "geometry_primary_analysis.json")["features"]
    feat_dir = args.data_dir / "features"
    geo_dir = args.data_dir / "geometry_enrichment"

    explicit_feature_ids = resolve_requested_feature_ids(args)
    explicit = set(explicit_feature_ids)
    allowed_accessions = resolve_requested_accessions(args)

    feature_case_pools: list[dict[str, Any]] = []

    for fid_str, row in gp.items():
        fid = int(fid_str)
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

        cases = _build_feature_case_candidates(
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
                "structural_category": cases[0].structural_category,
                "concordance_f1": cases[0].concordance_f1,
                "geom_pr_auc": cases[0].geom_pr_auc,
                "sequence_length": min(case.sequence_length for case in cases),
                "cases": cases,
            }
        )

    if explicit_feature_ids:
        pools_by_fid = {pool["feature_id"]: pool for pool in feature_case_pools}
        selected: list[CaseSpec] = []
        for fid in explicit_feature_ids:
            selected.extend(pools_by_fid.get(fid, {}).get("cases", []))
        if args.max_cases is not None:
            return selected[: args.max_cases]
        return selected

    by_cat: dict[str, list[dict[str, Any]]] = {}
    for pool in feature_case_pools:
        by_cat.setdefault(pool["structural_category"], []).append(pool)

    for pools in by_cat.values():
        pools.sort(
            key=lambda pool: (
                -pool["concordance_f1"],
                -pool["geom_pr_auc"],
                pool["sequence_length"],
                pool["feature_id"],
            )
        )

    cat_order = sorted(
        by_cat,
        key=lambda cat: (
            -by_cat[cat][0]["concordance_f1"],
            -by_cat[cat][0]["geom_pr_auc"],
            by_cat[cat][0]["sequence_length"],
            cat,
        ),
    )

    selected_pools: list[dict[str, Any]] = []
    while len(selected_pools) < args.max_features:
        added = False
        for cat in cat_order:
            if not by_cat[cat]:
                continue
            selected_pools.append(by_cat[cat].pop(0))
            added = True
            if len(selected_pools) >= args.max_features:
                break
        if not added:
            break

    selected_cases: list[CaseSpec] = []
    for pool in selected_pools:
        selected_cases.extend(pool["cases"])
    if args.max_cases is not None:
        return selected_cases[: args.max_cases]
    return selected_cases


def allowed_amino_acid_ids(tokenizer) -> list[int]:
    ids = []
    for token in AA_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            raise ValueError(f"Could not map amino-acid token '{token}' to a tokenizer id.")
        ids.append(int(token_id))
    return ids


def build_intervention_specs(
    feature_matrix: torch.Tensor,
    decoder_weights: torch.Tensor,
    case: CaseSpec,
    group_size: int,
) -> tuple[InterventionSpec, InterventionSpec]:
    feats = feature_matrix.detach().cpu().numpy()
    d_sae = feats.shape[1]
    lesion_idx = np.asarray(case.lesion_positions, dtype=int)
    context_idx = np.asarray(case.context_positions, dtype=int)
    target = case.feature_id

    context_slice = feats[context_idx]
    lesion_slice = feats[lesion_idx]
    target_context = context_slice[:, target]
    target_norm = float(np.linalg.norm(target_context))

    context_norms = np.linalg.norm(context_slice, axis=0)
    context_dot = context_slice.T @ target_context
    denom = context_norms * max(target_norm, 1e-8)
    context_cos = np.divide(context_dot, denom, out=np.zeros(d_sae, dtype=float), where=denom > 0)

    context_mean = context_slice.mean(axis=0)
    lesion_mean = lesion_slice.mean(axis=0)
    context_active = (context_slice > 0).mean(axis=0)
    lesion_active = (lesion_slice > 0).mean(axis=0)
    global_active = (feats > 0).mean(axis=0)

    w_dec = F.normalize(decoder_weights.float(), dim=1).cpu().numpy()
    decoder_cos = w_dec @ w_dec[target]

    target_scale = float(max(context_mean[target], 1e-6))
    rel_mean = np.clip(context_mean / target_scale, 0.0, 2.0) / 2.0
    group_score = 0.45 * np.maximum(context_cos, 0.0) + 0.35 * np.maximum(decoder_cos, 0.0) + 0.20 * rel_mean
    group_score[target] = -np.inf

    ranked = np.argsort(group_score)[::-1]
    lesion_group = [target]
    for idx in ranked:
        if len(lesion_group) >= group_size:
            break
        if context_mean[idx] <= 0:
            continue
        lesion_group.append(int(idx))

    if len(lesion_group) < group_size:
        fallback = np.argsort(context_mean)[::-1]
        for idx in fallback:
            idx = int(idx)
            if idx == target or idx in lesion_group or context_mean[idx] <= 0:
                continue
            lesion_group.append(idx)
            if len(lesion_group) >= group_size:
                break

    def match_distance(src: int, cand: int) -> float:
        return (
            1.20 * abs(math.log1p(float(lesion_mean[cand])) - math.log1p(float(lesion_mean[src])))
            + 0.50 * abs(math.log1p(float(context_mean[cand])) - math.log1p(float(context_mean[src])))
            + 0.45 * abs(float(lesion_active[cand]) - float(lesion_active[src]))
            + 0.30 * abs(float(context_active[cand]) - float(context_active[src]))
            + 0.15 * abs(float(global_active[cand]) - float(global_active[src]))
        )

    control_group: list[int] = []
    used = set(lesion_group)
    similarity_thresholds = [0.10, 0.20, 0.35, 0.50, 0.75]

    for src in lesion_group:
        chosen = None
        for thr in similarity_thresholds:
            candidates = []
            for cand in range(d_sae):
                if cand in used or lesion_mean[cand] <= 0:
                    continue
                if max(decoder_cos[cand], 0.0) > thr:
                    continue
                candidates.append(cand)
            if candidates:
                chosen = min(
                    candidates,
                    key=lambda cand: match_distance(src, cand) + 0.35 * max(context_cos[cand], 0.0),
                )
                break
        if chosen is None:
            remaining = [
                cand
                for cand in range(d_sae)
                if cand not in used and lesion_mean[cand] > 0
            ]
            if remaining:
                chosen = min(
                    remaining,
                    key=lambda cand: (
                        match_distance(src, cand)
                        + 0.75 * max(context_cos[cand], 0.0)
                        + 1.50 * max(decoder_cos[cand], 0.0)
                    ),
                )
        if chosen is not None:
            chosen = int(chosen)
            control_group.append(chosen)
            used.add(chosen)

    lesion_diag = [
        {
            "feature_id": float(fid),
            "context_mean": float(context_mean[fid]),
            "lesion_mean": float(lesion_mean[fid]),
            "context_cosine_to_target": float(context_cos[fid]),
            "decoder_cosine_to_target": float(decoder_cos[fid]),
        }
        for fid in lesion_group
    ]
    control_diag = [
        {
            "feature_id": float(fid),
            "context_mean": float(context_mean[fid]),
            "lesion_mean": float(lesion_mean[fid]),
            "context_cosine_to_target": float(context_cos[fid]),
            "decoder_cosine_to_target": float(decoder_cos[fid]),
        }
        for fid in control_group
    ]

    return (
        InterventionSpec(
            label="group_lesion",
            feature_ids=lesion_group,
            positions=list(case.lesion_positions),
            diagnostics=lesion_diag,
        ),
        InterventionSpec(
            label="matched_control",
            feature_ids=control_group,
            positions=list(case.lesion_positions),
            diagnostics=control_diag,
        ),
    )


def apply_zero_intervention(
    features: torch.Tensor,
    feature_ids: list[int],
    positions: list[int],
) -> torch.Tensor:
    out = features.clone()
    pos = torch.tensor(positions, device=features.device, dtype=torch.long)
    pos = pos[(pos >= 0) & (pos < out.shape[0])]
    if len(pos) == 0:
        return out
    out[pos[:, None], torch.tensor(feature_ids, device=features.device, dtype=torch.long)] = 0.0
    return out


def apply_scaled_intervention(
    features: torch.Tensor,
    feature_ids: list[int],
    positions: list[int],
    scale: float,
) -> torch.Tensor:
    out = features.clone()
    pos = torch.tensor(positions, device=features.device, dtype=torch.long)
    pos = pos[(pos >= 0) & (pos < out.shape[0])]
    if len(pos) == 0:
        return out
    feature_idx = torch.tensor(feature_ids, device=features.device, dtype=torch.long)
    out[pos[:, None], feature_idx] = out[pos[:, None], feature_idx] * float(scale)
    return out


def compute_logit_effects(
    orig_logits: torch.Tensor,
    mod_logits: torch.Tensor,
    patch_positions: list[int],
    mutable_positions: list[int],
) -> dict[str, float]:
    if orig_logits.ndim != 3 or mod_logits.ndim != 3:
        return {}

    length = min(orig_logits.shape[1], mod_logits.shape[1]) - 2
    if length <= 0:
        return {}

    orig_lp = torch.log_softmax(orig_logits[0, 1 : length + 1], dim=-1)
    mod_lp = torch.log_softmax(mod_logits[0, 1 : length + 1], dim=-1)
    orig_p = orig_lp.exp()
    mod_p = mod_lp.exp()

    kl = (orig_p * (orig_lp - mod_lp)).sum(dim=-1)
    orig_ent = -(orig_p * orig_lp).sum(dim=-1)
    mod_ent = -(mod_p * mod_lp).sum(dim=-1)

    orig_ids = orig_logits[0, 1 : length + 1].argmax(dim=-1)
    mod_ids = mod_logits[0, 1 : length + 1].argmax(dim=-1)

    patch_idx = torch.tensor(
        [p for p in patch_positions if 0 <= p < length],
        dtype=torch.long,
        device=kl.device,
    )
    mutable_idx = torch.tensor(
        [p for p in mutable_positions if 0 <= p < length],
        dtype=torch.long,
        device=kl.device,
    )

    out = {
        "mean_kl_patch": 0.0,
        "mean_entropy_delta_patch": 0.0,
        "argmax_changes_patch": 0.0,
        "argmax_changes_mutable": 0.0,
    }
    if len(patch_idx) > 0:
        out["mean_kl_patch"] = float(kl[patch_idx].mean().item())
        out["mean_entropy_delta_patch"] = float((mod_ent - orig_ent)[patch_idx].mean().item())
        out["argmax_changes_patch"] = float((orig_ids[patch_idx] != mod_ids[patch_idx]).sum().item())
    if len(mutable_idx) > 0:
        out["argmax_changes_mutable"] = float((orig_ids[mutable_idx] != mod_ids[mutable_idx]).sum().item())
    return out


def sample_local_sequences(
    logits: torch.Tensor,
    original_sequence: str,
    mutable_positions: list[int],
    tokenizer,
    aa_token_ids: list[int],
    num_samples: int,
    temperature: float,
    top_k_aa: int,
    seed: int,
    greedy_first: bool,
) -> list[dict[str, Any]]:
    residue_logits = logits[0, 1 : len(original_sequence) + 1].detach().cpu()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    valid_positions = [p for p in mutable_positions if 0 <= p < len(original_sequence)]
    aa_ids = torch.tensor(aa_token_ids, dtype=torch.long)
    unique: dict[str, dict[str, Any]] = {}
    max_attempts = max(10, num_samples * 20)

    def choose_token_id(logits_row: torch.Tensor, greedy: bool) -> int:
        aa_logits = logits_row[aa_ids]
        if greedy or temperature <= 0:
            idx = int(torch.argmax(aa_logits).item())
            return int(aa_ids[idx].item())

        scaled = aa_logits / temperature
        if top_k_aa > 0 and top_k_aa < len(aa_ids):
            top_vals, top_idx = torch.topk(scaled, k=top_k_aa)
            probs = torch.softmax(top_vals, dim=-1)
            chosen = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
            return int(aa_ids[top_idx[chosen]].item())

        probs = torch.softmax(scaled, dim=-1)
        chosen = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        return int(aa_ids[chosen].item())

    for attempt in range(max_attempts):
        greedy = greedy_first and attempt == 0
        seq = list(original_sequence)
        for pos in valid_positions:
            token_id = choose_token_id(residue_logits[pos], greedy=greedy)
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            seq[pos] = token
        steered = "".join(seq)
        if steered in unique:
            continue
        changed = [idx for idx, (a, b) in enumerate(zip(original_sequence, steered)) if a != b]
        unique[steered] = {
            "sequence": steered,
            "mutations": changed,
            "n_mutations": len(changed),
            "decode_mode": "greedy" if greedy else "sampled",
        }
        if len(unique) >= num_samples:
            break

    if not unique:
        unique[original_sequence] = {
            "sequence": original_sequence,
            "mutations": [],
            "n_mutations": 0,
            "decode_mode": "fallback",
        }

    return list(unique.values())[:num_samples]


def ca_to_pdb_text(ca: np.ndarray) -> str:
    lines = []
    for idx, (x, y, z) in enumerate(np.asarray(ca, dtype=float), start=1):
        lines.append(
            f"ATOM  {idx:5d}  CA  ALA A{idx:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           C"
        )
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def fold_sequence_with_details(sequence: str, fold_tokenizer, fold_model) -> dict[str, Any]:
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
    from transformers.models.esm.openfold_utils.protein import to_pdb

    dev = next(fold_model.parameters()).device
    inputs = fold_tokenizer(
        [sequence], return_tensors="pt", add_special_tokens=False, padding=False
    )
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = fold_model(**inputs)

    positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    np_out = {k: v.detach().cpu().numpy() for k, v in outputs.items() if isinstance(v, torch.Tensor)}
    positions_np = positions.detach().cpu().numpy()
    plddt = np_out["plddt"][0].astype(float)

    aa = np_out["aatype"][0]
    resid = np.arange(1, len(aa) + 1)
    pdb_text = to_pdb(
        OFProtein(
            aatype=aa,
            atom_positions=positions_np[0],
            atom_mask=np_out["atom37_atom_exists"][0],
            residue_index=resid,
            b_factors=np_out["plddt"][0],
            chain_index=np.zeros_like(resid),
        )
    )

    return {
        "pdb_text": pdb_text,
        "plddt": plddt.tolist(),
        "mean_plddt": float(plddt.mean()),
    }


def build_structure_record(
    *,
    sequence: str,
    pdb_text: Optional[str] = None,
    ca: Optional[np.ndarray] = None,
    plddt: Optional[list[float]] = None,
    source: str,
) -> dict[str, Any]:
    if pdb_text is not None:
        ca_arr = ca_backbone(pdb_text)
    elif ca is not None:
        ca_arr = np.asarray(ca, dtype=float)
    else:
        raise ValueError("Either pdb_text or ca must be provided.")

    helices = detect_alpha_helices_from_ca(ca_arr)
    profiles = compute_residue_profiles(ca_arr, helices)
    plddt_arr = None if plddt is None else np.asarray(plddt, dtype=float)
    return {
        "sequence": sequence,
        "ca": ca_arr,
        "profiles": profiles,
        "plddt": plddt_arr,
        "mean_plddt": None if plddt_arr is None else float(plddt_arr.mean()),
        "source": source,
        "pdb_text": pdb_text if pdb_text is not None else ca_to_pdb_text(ca_arr),
    }


def compute_local_geometry_vector(
    structure: dict[str, Any],
    positions: list[int],
    half_w: int,
) -> np.ndarray | None:
    vecs: list[np.ndarray] = []
    for pos in positions:
        vec = extract_local_feature_vector(
            structure["profiles"],
            structure["ca"],
            pos,
            half_w=half_w,
            sequence=structure["sequence"],
        )
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=float)
        if np.all(np.isfinite(arr)):
            vecs.append(arr)
    if not vecs:
        return None
    return np.mean(np.vstack(vecs), axis=0)


def compute_local_geometry_value(
    structure: dict[str, Any],
    positions: list[int],
    feature_name: str,
    half_w: int,
) -> float:
    idx = LOCAL_GEOM_NAMES.index(feature_name)
    vec = compute_local_geometry_vector(structure, positions, half_w)
    if vec is None:
        return float("nan")
    return float(vec[idx])


def _parse_feature_importances(raw: Any) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    if isinstance(raw, dict):
        for name, value in raw.items():
            if name in LOCAL_GEOM_NAMES:
                items.append((str(name), float(value)))
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("feature") or item.get("feature_name")
            value = item.get("importance") or item.get("value") or item.get("weight")
            if name in LOCAL_GEOM_NAMES and value is not None:
                items.append((str(name), float(value)))
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items


def _compute_descriptor_scales(
    structure: dict[str, Any],
    feature_names: list[str],
    half_w: int,
) -> list[float]:
    valid_vecs: list[np.ndarray] = []
    for pos in range(half_w, len(structure["ca"]) - half_w):
        vec = extract_local_feature_vector(
            structure["profiles"],
            structure["ca"],
            pos,
            half_w=half_w,
            sequence=structure["sequence"],
        )
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=float)
        if np.all(np.isfinite(arr)):
            valid_vecs.append(arr)
    if not valid_vecs:
        return [1e-3 for _ in feature_names]

    mat = np.vstack(valid_vecs)
    scales: list[float] = []
    for name in feature_names:
        idx = LOCAL_GEOM_NAMES.index(name)
        scale = float(np.std(mat[:, idx]))
        if not np.isfinite(scale) or scale < 1e-3:
            scale = 1e-3
        scales.append(scale)
    return scales


def build_geometry_signature_spec(
    geo_json: dict[str, Any],
    case: CaseSpec,
    original_structure: dict[str, Any],
    top_k: int,
    half_w: int,
) -> GeometrySignatureSpec:
    importances = _parse_feature_importances(
        geo_json.get("geometric_residue_level", {}).get("feature_importances", {})
    )
    importance_map = {name: value for name, value in importances}

    selected: list[str] = []
    if case.top_geometric_feature in LOCAL_GEOM_NAMES:
        selected.append(case.top_geometric_feature)
    for name, value in importances:
        if value <= 0 or name in selected:
            continue
        selected.append(name)
        if len(selected) >= max(1, top_k):
            break
    if not selected:
        selected = [case.top_geometric_feature]
    selected = selected[: max(1, top_k)]

    raw_importances = [float(max(importance_map.get(name, 0.0), 0.0)) for name in selected]
    if sum(raw_importances) <= 0:
        raw_importances = [1.0 for _ in selected]
    weights_arr = np.asarray(raw_importances, dtype=float)
    weights = (weights_arr / max(weights_arr.sum(), 1e-8)).tolist()

    original_vec = compute_local_geometry_vector(original_structure, case.patch_positions, half_w)
    if original_vec is None:
        original_values = [float("nan") for _ in selected]
    else:
        original_values = [float(original_vec[LOCAL_GEOM_NAMES.index(name)]) for name in selected]

    return GeometrySignatureSpec(
        feature_names=selected,
        raw_importances=raw_importances,
        weights=weights,
        scales=_compute_descriptor_scales(original_structure, selected, half_w),
        original_values=original_values,
    )


def compute_geometry_signature_metrics(
    original: dict[str, Any],
    candidate: dict[str, Any],
    positions: list[int],
    half_w: int,
    signature_spec: Optional[GeometrySignatureSpec],
) -> tuple[dict[str, float], Optional[dict[str, Any]]]:
    if signature_spec is None or not signature_spec.feature_names:
        return {}, None

    orig_vec = compute_local_geometry_vector(original, positions, half_w)
    cand_vec = compute_local_geometry_vector(candidate, positions, half_w)
    if orig_vec is None or cand_vec is None:
        return {}, None

    original_values: dict[str, float] = {}
    candidate_values: dict[str, float] = {}
    deltas: dict[str, float] = {}
    standardized_abs_deltas: dict[str, float] = {}
    weighted_abs_contributions: dict[str, float] = {}
    weighted_signed_contributions: dict[str, float] = {}

    shift = 0.0
    signed_shift = 0.0
    max_component_shift = 0.0

    for name, weight, scale in zip(
        signature_spec.feature_names,
        signature_spec.weights,
        signature_spec.scales,
    ):
        idx = LOCAL_GEOM_NAMES.index(name)
        orig_val = float(orig_vec[idx])
        cand_val = float(cand_vec[idx])
        delta = cand_val - orig_val
        scaled_delta = delta / max(scale, 1e-8)
        abs_scaled_delta = abs(scaled_delta)
        weighted_abs = float(weight * abs_scaled_delta)
        weighted_signed = float(weight * scaled_delta)

        original_values[name] = orig_val
        candidate_values[name] = cand_val
        deltas[name] = delta
        standardized_abs_deltas[name] = float(abs_scaled_delta)
        weighted_abs_contributions[name] = weighted_abs
        weighted_signed_contributions[name] = weighted_signed

        shift += weighted_abs
        signed_shift += weighted_signed
        max_component_shift = max(max_component_shift, abs_scaled_delta)

    summary = {
        "geometry_signature_shift": float(shift),
        "geometry_signature_signed_shift": float(signed_shift),
        "geometry_signature_max_component_shift": float(max_component_shift),
    }
    details = {
        "feature_names": list(signature_spec.feature_names),
        "raw_importances": list(signature_spec.raw_importances),
        "weights": list(signature_spec.weights),
        "scales": list(signature_spec.scales),
        "original_values": original_values,
        "candidate_values": candidate_values,
        "deltas": deltas,
        "standardized_abs_deltas": standardized_abs_deltas,
        "weighted_abs_contributions": weighted_abs_contributions,
        "weighted_signed_contributions": weighted_signed_contributions,
    }
    return summary, details


def compute_structure_metrics(
    original: dict[str, Any],
    candidate: dict[str, Any],
    case: CaseSpec,
    local_half_window: int,
    signature_spec: Optional[GeometrySignatureSpec] = None,
) -> tuple[dict[str, float], Optional[dict[str, Any]]]:
    orig_ca = original["ca"]
    cand_ca = candidate["ca"]
    n = min(len(orig_ca), len(cand_ca))
    span_start = max(0, min(case.patch_positions) - local_half_window)
    span_end = min(n, max(case.patch_positions) + local_half_window + 1)

    orig_span = orig_ca[span_start:span_end]
    cand_span = cand_ca[span_start:span_end]
    if len(orig_span) >= 2 and len(cand_span) >= 2:
        aligned_local = kabsch_align(cand_span, orig_span)
        local_rmsd = compute_rmsd(aligned_local, orig_span)
    else:
        local_rmsd = float("nan")

    aligned_global = kabsch_align(cand_ca[:n], orig_ca[:n])
    global_rmsd = compute_rmsd(aligned_global, orig_ca[:n])

    target_orig = compute_local_geometry_value(
        original, case.patch_positions, case.top_geometric_feature, local_half_window
    )
    target_cand = compute_local_geometry_value(
        candidate, case.patch_positions, case.top_geometric_feature, local_half_window
    )

    helix_orig = float(original["profiles"]["helix_mask"][span_start:span_end].mean())
    helix_cand = float(candidate["profiles"]["helix_mask"][span_start:span_end].mean())

    plddt_orig = None
    plddt_cand = None
    if original["plddt"] is not None and len(original["plddt"]) >= span_end:
        plddt_orig = float(original["plddt"][span_start:span_end].mean())
    if candidate["plddt"] is not None and len(candidate["plddt"]) >= span_end:
        plddt_cand = float(candidate["plddt"][span_start:span_end].mean())

    out = {
        "global_ca_rmsd": float(global_rmsd),
        "local_ca_rmsd": float(local_rmsd),
        "target_feature_original": float(target_orig),
        "target_feature_candidate": float(target_cand),
        "target_feature_delta": float(target_cand - target_orig),
        "target_feature_abs_delta": float(abs(target_cand - target_orig)),
        "local_helix_fraction_original": float(helix_orig),
        "local_helix_fraction_candidate": float(helix_cand),
        "local_helix_fraction_delta": float(helix_cand - helix_orig),
    }
    if plddt_orig is not None:
        out["local_mean_plddt_original"] = plddt_orig
    if plddt_cand is not None:
        out["local_mean_plddt_candidate"] = plddt_cand
    if plddt_orig is not None and plddt_cand is not None:
        out["local_mean_plddt_delta"] = plddt_cand - plddt_orig
    signature_summary, signature_details = compute_geometry_signature_metrics(
        original,
        candidate,
        case.patch_positions,
        local_half_window,
        signature_spec,
    )
    out.update(signature_summary)
    return out, signature_details


def write_fasta(path: Path, entries: list[tuple[str, str]]) -> None:
    chunks = []
    for name, sequence in entries:
        chunks.append(f">{name}")
        chunks.append(sequence)
    path.write_text("\n".join(chunks) + "\n")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")


def summarize_case(
    case: CaseSpec,
    lesion: InterventionSpec,
    control: InterventionSpec,
    rows: list[dict[str, Any]],
    signature_spec: Optional[GeometrySignatureSpec] = None,
) -> dict[str, Any]:
    def aggregate(label: str, key: str) -> float:
        vals = [float(r[key]) for r in rows if r["intervention_label"] == label and key in r and not math.isnan(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

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
        "geometry_signature_feature_names": [] if signature_spec is None else signature_spec.feature_names,
        "geometry_signature_weights": [] if signature_spec is None else signature_spec.weights,
        "group_mean_target_abs_delta": aggregate("group_lesion", "target_feature_abs_delta"),
        "control_mean_target_abs_delta": aggregate("matched_control", "target_feature_abs_delta"),
        "group_mean_signature_shift": aggregate("group_lesion", "geometry_signature_shift"),
        "control_mean_signature_shift": aggregate("matched_control", "geometry_signature_shift"),
        "group_mean_local_rmsd": aggregate("group_lesion", "local_ca_rmsd"),
        "control_mean_local_rmsd": aggregate("matched_control", "local_ca_rmsd"),
        "group_mean_local_plddt_delta": aggregate("group_lesion", "local_mean_plddt_delta"),
        "control_mean_local_plddt_delta": aggregate("matched_control", "local_mean_plddt_delta"),
        "paired_target_abs_delta_margin": (
            aggregate("group_lesion", "target_feature_abs_delta")
            - aggregate("matched_control", "target_feature_abs_delta")
        ),
        "paired_signature_shift_margin": (
            aggregate("group_lesion", "geometry_signature_shift")
            - aggregate("matched_control", "geometry_signature_shift")
        ),
        "paired_local_rmsd_margin": (
            aggregate("group_lesion", "local_ca_rmsd")
            - aggregate("matched_control", "local_ca_rmsd")
        ),
    }


def run_case(
    case: CaseSpec,
    args: argparse.Namespace,
    tokenizer,
    esm_model,
    sae,
    decoder_weights: torch.Tensor,
    fold_assets: Optional[tuple[Any, Any]],
    aa_token_ids: list[int],
    root_out: Path,
    case_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_dir = root_out / f"{case_index:02d}_f{case.feature_id:04d}_{case.accession}"
    case_dir.mkdir(parents=True, exist_ok=True)
    save_json(case_dir / "case_spec.json", asdict(case))
    write_fasta(case_dir / "original_sequence.fasta", [("original", case.sequence)])
    geo_json = load_json(args.data_dir / "geometry_enrichment" / f"{case.feature_id:04d}.json")

    orig_logits, orig_hidden, token_ids, attn_mask = extract_hidden_states(
        esm_model,
        tokenizer,
        case.sequence,
        args.layer,
        auto_device(args.device),
    )
    seq_len = len(case.sequence)
    with torch.no_grad():
        residue_hidden = orig_hidden[0, 1 : seq_len + 1, :]
        normalised, original_norms = sae._normalize_input_and_get_norms(residue_hidden)
        features = sae.encode(normalised)

    lesion_spec, control_spec = build_intervention_specs(
        features, decoder_weights, case, args.group_size
    )
    save_json(
        case_dir / "interventions.json",
        {
            "group_lesion": asdict(lesion_spec),
            "matched_control": asdict(control_spec),
        },
    )

    if args.original_structure_source == "feature_data":
        protein_row = next(
            (
                p
                for p in geo_json.get("plot_data", {}).get("top_proteins", [])
                if p.get("accession") == case.accession
            ),
            None,
        )
        if protein_row is None:
            raise ValueError(
                f"Could not find accession {case.accession} inside geometry_enrichment/{case.feature_id:04d}.json"
            )
        original_structure = build_structure_record(
            sequence=case.sequence,
            ca=np.asarray(protein_row["ca_backbone"], dtype=float),
            source="feature_data",
        )
    else:
        if fold_assets is None:
            raise RuntimeError("Folding is required for original_structure_source='esmfold'.")
        fold_tokenizer, fold_model = fold_assets
        print("    Folding original sequence with ESMFold …")
        folded = fold_sequence_with_details(case.sequence, fold_tokenizer, fold_model)
        (case_dir / "original_esmfold.pdb").write_text(folded["pdb_text"])
        original_structure = build_structure_record(
            sequence=case.sequence,
            pdb_text=folded["pdb_text"],
            plddt=folded["plddt"],
            source="esmfold",
        )

    signature_spec = build_geometry_signature_spec(
        geo_json,
        case,
        original_structure,
        args.signature_top_k,
        args.local_half_window,
    )
    save_json(case_dir / "geometry_signature_spec.json", asdict(signature_spec))

    mutable_positions = expand_positions(case.patch_positions, args.mutation_radius, seq_len)

    all_rows: list[dict[str, Any]] = []
    for intervention_idx, spec in enumerate([lesion_spec, control_spec]):
        intervention_dir = case_dir / spec.label
        intervention_dir.mkdir(parents=True, exist_ok=True)

        features_mod = apply_zero_intervention(features, spec.feature_ids, spec.positions)
        modified_hidden = decode_and_build_hidden(
            sae, features_mod, orig_hidden, seq_len, original_norms
        )
        modified_logits = inject_and_get_logits(
            esm_model, token_ids, attn_mask, args.layer, modified_hidden
        )

        hidden_delta = (modified_hidden - orig_hidden)[0, 1 : seq_len + 1].norm(dim=-1).detach().cpu().numpy()
        local_hidden_delta = float(np.mean(hidden_delta[spec.positions])) if spec.positions else 0.0
        context_hidden_delta = float(np.mean(hidden_delta[case.context_positions])) if case.context_positions else 0.0
        logit_effects = compute_logit_effects(
            orig_logits,
            modified_logits,
            case.patch_positions,
            mutable_positions,
        )

        samples = sample_local_sequences(
            modified_logits,
            case.sequence,
            mutable_positions,
            tokenizer,
            aa_token_ids,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k_aa=args.top_k_aa,
            seed=args.seed + 1000 * case.feature_id + 17 * intervention_idx,
            greedy_first=not args.disable_greedy_first,
        )
        write_fasta(
            intervention_dir / "sequences.fasta",
            [(f"{spec.label}_{i+1}", sample["sequence"]) for i, sample in enumerate(samples)],
        )

        for sample_idx, sample in enumerate(samples, start=1):
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
                "sample_index": sample_idx,
                "sequence": sample["sequence"],
                "decode_mode": sample["decode_mode"],
                "n_mutations": sample["n_mutations"],
                "mutation_positions": sample["mutations"],
                "mutable_positions": mutable_positions,
                "patch_positions": case.patch_positions,
                "lesion_positions": case.lesion_positions,
                "feature_ids_zeroed": spec.feature_ids,
                "top_geometric_feature": case.top_geometric_feature,
                "local_hidden_delta": local_hidden_delta,
                "context_hidden_delta": context_hidden_delta,
                **logit_effects,
            }

            sample_dir = intervention_dir / f"sample_{sample_idx:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            write_fasta(
                sample_dir / "sequence.fasta",
                [(f"{spec.label}_{sample_idx}", sample["sequence"])],
            )

            signature_details = None
            if not args.skip_fold:
                if fold_assets is None:
                    raise RuntimeError("Folding requested but ESMFold assets were not loaded.")
                fold_tokenizer, fold_model = fold_assets
                print(
                    f"    {spec.label}: folding sample {sample_idx}/{len(samples)} "
                    f"({sample['n_mutations']} mutation(s)) …"
                )
                folded = fold_sequence_with_details(sample["sequence"], fold_tokenizer, fold_model)
                (sample_dir / "structure.pdb").write_text(folded["pdb_text"])
                candidate_structure = build_structure_record(
                    sequence=sample["sequence"],
                    pdb_text=folded["pdb_text"],
                    plddt=folded["plddt"],
                    source="esmfold",
                )
                row["mean_plddt"] = folded["mean_plddt"]
                structure_metrics, signature_details = compute_structure_metrics(
                    original_structure,
                    candidate_structure,
                    case,
                    args.local_half_window,
                    signature_spec,
                )
                row.update(structure_metrics)

            metrics_payload = dict(row)
            if signature_details is not None:
                metrics_payload["geometry_signature"] = signature_details
            save_json(sample_dir / "metrics.json", metrics_payload)
            all_rows.append(row)

    case_summary = summarize_case(case, lesion_spec, control_spec, all_rows, signature_spec)
    save_json(case_dir / "case_summary.json", case_summary)
    return all_rows, case_summary


def save_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    intervention_mode = resolve_intervention_mode(args)
    requested_feature_ids = resolve_requested_feature_ids(args)
    requested_accessions = sorted(resolve_requested_accessions(args))

    device = auto_device(args.device)
    fold_device = auto_device(args.fold_device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = select_cases(args)
    if not cases:
        raise SystemExit("No benchmark cases satisfied the selection criteria.")

    print()
    print("=" * 90)
    print("  ProteinLens — Geometry Causal Ablation Benchmark")
    print("=" * 90)
    print(f"  Selected {len(cases)} case(s) across {len({case.feature_id for case in cases})} feature(s)")
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
            f"{case.structural_category}  len={case.sequence_length}  "
            f"rank={case.protein_rank}  patch={patch_str}"
        )

    case_rows = [asdict(case) for case in cases]
    save_table(args.output_dir / "selected_cases.csv", case_rows)
    save_json(args.output_dir / "selected_cases.json", case_rows)
    if args.dry_run:
        save_json(
            args.output_dir / "run_manifest.json",
            {
                "esm_model": args.esm_model,
                "sae_dir": args.sae_dir,
                "layer": args.layer,
                "original_structure_source": args.original_structure_source,
                "group_size": args.group_size,
                "single_node_only": bool(args.single_node_only),
                "intervention_mode": intervention_mode,
                "feature_ids_requested": requested_feature_ids,
                "accessions_requested": requested_accessions,
                "proteins_per_feature": args.proteins_per_feature,
                "top_sequence_pool": args.top_sequence_pool,
                "max_cases": args.max_cases,
                "dry_run": True,
                "n_cases": len(cases),
                "n_unique_features": len({case.feature_id for case in cases}),
            },
        )
        print(f"\n  Dry run complete. Saved selected cases to {args.output_dir}")
        print()
        return

    tokenizer, esm_model, sae = load_pipeline_models(args.sae_dir, args.esm_model, device)
    decoder_weights = _get_decoder_weights(sae)
    aa_token_ids = allowed_amino_acid_ids(tokenizer)

    fold_assets = None
    if not args.skip_fold:
        print(f"\n  Loading ESMFold on {fold_device} …")
        fold_assets = load_esmfold(fold_device)

    per_sample_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases, start=1):
        print(f"\n[{case_index}/{len(cases)}] Running f/{case.feature_id:04d} on {case.accession} …")
        rows, summary = run_case(
            case,
            args,
            tokenizer,
            esm_model,
            sae,
            decoder_weights,
            fold_assets,
            aa_token_ids,
            args.output_dir,
            case_index,
        )
        per_sample_rows.extend(rows)
        case_summaries.append(summary)

    save_table(args.output_dir / "per_sample_metrics.csv", per_sample_rows)
    save_json(args.output_dir / "per_sample_metrics.json", per_sample_rows)
    save_table(args.output_dir / "case_summaries.csv", case_summaries)
    save_json(args.output_dir / "case_summaries.json", case_summaries)
    save_json(
        args.output_dir / "run_manifest.json",
        {
            "esm_model": args.esm_model,
            "sae_dir": args.sae_dir,
            "layer": args.layer,
            "original_structure_source": args.original_structure_source,
            "group_size": args.group_size,
            "single_node_only": bool(args.single_node_only),
            "intervention_mode": intervention_mode,
            "feature_ids_requested": requested_feature_ids,
            "accessions_requested": requested_accessions,
            "proteins_per_feature": args.proteins_per_feature,
            "top_sequence_pool": args.top_sequence_pool,
            "max_cases": args.max_cases,
            "dry_run": False,
            "n_cases": len(cases),
            "n_unique_features": len({case.feature_id for case in cases}),
        },
    )

    print(f"\n  Saved benchmark outputs to {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
