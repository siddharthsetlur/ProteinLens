#!/usr/bin/env python3
"""Run scaled single-neuron contact-predictor ablations for case-study examples.

This script is the contact-prediction analogue of the structure-only scale
sweep. It is designed for visual case studies after a larger parity run:

1. Load finished contact causal-parity results.
2. Pick the top-k strongest positive target-vs-control cases, or explicit
   feature/accession pairs.
3. Keep the amino-acid sequence fixed.
4. Scale the target SAE feature from 1.0 down to 0.0 at the lesion patch.
5. Resume the remaining ESM2 layers and run ``predict_contacts``.
6. Save per-scale contact maps plus compact dose-response summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TORCH", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

import benchmark_contact_predictor_ablation as bcpa  # noqa: E402
import benchmark_geometry_causal_ablation as bgca  # noqa: E402
from intervene_and_fold import decode_and_build_hidden, load_pipeline_models  # noqa: E402
from proteinlens.analysis.feature_clusters import _get_decoder_weights  # noqa: E402


DEFAULT_ABLATION_SCALES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run scaled single-neuron case-study sweeps on the ESM2 contact "
            "predictor using top cases from a finished parity run."
        )
    )
    p.add_argument(
        "--source-results-dir",
        type=Path,
        default=Path("results/contact_predictor_ablation_1000"),
        help="Finished contact-parity run used to choose case-study examples.",
    )
    p.add_argument("--data-dir", type=Path, default=Path("feature_data_cluster"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/contact_predictor_scale_sweep_case_studies"),
    )
    p.add_argument("--sae-dir", default=str(ROOT / "trained_models" / "fiery-sweep"))
    p.add_argument("--esm-model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--device", default=None, help="Device for ESM2 + SAE (default: auto)")
    p.add_argument(
        "--case-specs",
        nargs="+",
        default=None,
        help="Explicit case list as feature_id:accession tokens, e.g. 4803:Q45389 371:Q4JCM0.",
    )
    p.add_argument(
        "--top-k-cases",
        type=int,
        default=2,
        help="Total number of case-study examples to auto-select from the source parity run.",
    )
    p.add_argument(
        "--selection-mode",
        choices=["balanced_signed_target_delta", "top_sort_key"],
        default="balanced_signed_target_delta",
        help=(
            "How to auto-select case-study examples when --case-specs is not given. "
            "'balanced_signed_target_delta' picks half the cases with the largest "
            "positive signed target deltas and half with the most negative signed "
            "target deltas. 'top_sort_key' reproduces the older single-direction "
            "ranking by --sort-key."
        ),
    )
    p.add_argument(
        "--sort-key",
        default="paired_target_contact_abs_delta_margin",
        help="Case-summary key used when --selection-mode=top_sort_key.",
    )
    p.add_argument(
        "--include-proxy",
        action="store_true",
        help="Allow proxy-metric cases like min_spatial_dist_long during auto-selection.",
    )
    p.add_argument(
        "--ablation-scales",
        nargs="+",
        type=float,
        default=list(DEFAULT_ABLATION_SCALES),
        help="Multiplicative scales applied to the target feature at the lesion positions.",
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
        "--dry-run",
        action="store_true",
        help="Only resolve and save the selected case-study cases.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-8:
        return float("nan")
    return float(numerator / denominator)


def _scale_label(scale: float) -> str:
    text = f"{float(scale):.2f}".replace("-", "m").replace(".", "p")
    return f"scale_{text}"


def _parse_case_token(token: str) -> tuple[int, str]:
    token = str(token).strip()
    if ":" not in token:
        raise SystemExit(
            f"Invalid --case-specs entry '{token}'. Expected feature_id:accession, "
            "for example 4803:Q45389."
        )
    fid_str, accession = token.split(":", 1)
    return int(fid_str), accession.strip()


def _load_source_payloads(
    source_results_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, str], bgca.CaseSpec]]:
    case_summaries_path = source_results_dir / "case_summaries.json"
    selected_cases_path = source_results_dir / "selected_cases.json"
    per_sample_metrics_path = source_results_dir / "per_sample_metrics.json"
    if not case_summaries_path.exists():
        raise FileNotFoundError(f"Could not find {case_summaries_path}")
    if not selected_cases_path.exists():
        raise FileNotFoundError(f"Could not find {selected_cases_path}")
    if not per_sample_metrics_path.exists():
        raise FileNotFoundError(f"Could not find {per_sample_metrics_path}")

    summaries = list(bgca.load_json(case_summaries_path))
    selected_rows = list(bgca.load_json(selected_cases_path))
    per_sample_rows = list(bgca.load_json(per_sample_metrics_path))
    case_map: dict[tuple[int, str], bgca.CaseSpec] = {}
    for row in selected_rows:
        key = (int(row["feature_id"]), str(row["accession"]))
        case_map[key] = bgca.CaseSpec(**row)
    return summaries, per_sample_rows, case_map


def _attach_signed_source_metrics(
    summaries: list[dict[str, Any]],
    per_sample_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, list[float]]] = {}
    for row in per_sample_rows:
        key = (int(row["feature_id"]), str(row["accession"]))
        label = str(row.get("intervention_label", ""))
        bucket = grouped.setdefault(
            key,
            {
                "group_signed": [],
                "control_signed": [],
                "group_abs": [],
                "control_abs": [],
            },
        )
        signed = _safe_float(row.get("target_contact_metric_delta"))
        abs_delta = _safe_float(row.get("target_contact_metric_abs_delta"))
        if label == "group_lesion":
            if np.isfinite(signed):
                bucket["group_signed"].append(signed)
            if np.isfinite(abs_delta):
                bucket["group_abs"].append(abs_delta)
        elif label == "matched_control":
            if np.isfinite(signed):
                bucket["control_signed"].append(signed)
            if np.isfinite(abs_delta):
                bucket["control_abs"].append(abs_delta)

    enriched: list[dict[str, Any]] = []
    for row in summaries:
        key = (int(row["feature_id"]), str(row["accession"]))
        bucket = grouped.get(key, {})

        def _mean(values: list[float]) -> float:
            return float(np.mean(values)) if values else float("nan")

        payload = dict(row)
        payload["source_group_mean_target_contact_delta"] = _mean(bucket.get("group_signed", []))
        payload["source_control_mean_target_contact_delta"] = _mean(bucket.get("control_signed", []))
        payload["source_group_mean_target_contact_abs_delta"] = _mean(bucket.get("group_abs", []))
        payload["source_control_mean_target_contact_abs_delta"] = _mean(bucket.get("control_abs", []))
        payload["source_paired_target_contact_delta_margin"] = (
            payload["source_group_mean_target_contact_delta"]
            - payload["source_control_mean_target_contact_delta"]
        )
        enriched.append(payload)
    return enriched


def _select_cases_from_source(
    args: argparse.Namespace,
) -> tuple[list[bgca.CaseSpec], list[dict[str, Any]]]:
    summaries, per_sample_rows, case_map = _load_source_payloads(args.source_results_dir)
    summaries = _attach_signed_source_metrics(summaries, per_sample_rows)
    summary_map = {
        (int(row["feature_id"]), str(row["accession"])): row
        for row in summaries
    }

    if args.case_specs:
        requested = [_parse_case_token(token) for token in args.case_specs]
        cases: list[bgca.CaseSpec] = []
        source_rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for key in requested:
            case = case_map.get(key)
            row = summary_map.get(key)
            if case is None or row is None:
                missing.append(f"{key[0]}:{key[1]}")
                continue
            cases.append(case)
            source_rows.append(row)
        if missing:
            raise SystemExit(
                "Could not resolve the following explicit case-study cases in "
                f"{args.source_results_dir}: {', '.join(missing)}"
            )
        return cases, source_rows

    candidate_rows: list[dict[str, Any]] = []
    for row in summaries:
        key = (int(row["feature_id"]), str(row["accession"]))
        if key not in case_map:
            continue
        if not args.include_proxy and bool(row.get("target_contact_metric_is_proxy", False)):
            continue
        candidate_rows.append(row)

    if args.selection_mode == "top_sort_key":
        ranked_rows: list[dict[str, Any]] = []
        for row in candidate_rows:
            score = _safe_float(row.get(args.sort_key))
            if not np.isfinite(score):
                continue
            ranked_rows.append(row)

        ranked_rows.sort(
            key=lambda row: (
                _safe_float(row.get(args.sort_key), float("-inf")),
                _safe_float(row.get("group_mean_target_contact_abs_delta"), float("-inf")),
            ),
            reverse=True,
        )
        ranked_rows = ranked_rows[: max(0, args.top_k_cases)]
        if not ranked_rows:
            raise SystemExit(
                "No case-study examples were selected from the source results. "
                "Try --include-proxy or an explicit --case-specs list."
            )

        cases = [case_map[(int(row["feature_id"]), str(row["accession"]))] for row in ranked_rows]
        return cases, ranked_rows

    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for row in candidate_rows:
        signed_delta = _safe_float(row.get("source_group_mean_target_contact_delta"))
        if not np.isfinite(signed_delta):
            continue
        if signed_delta > 0:
            positives.append(row)
        elif signed_delta < 0:
            negatives.append(row)

    positives.sort(
        key=lambda row: (
            _safe_float(row.get("source_group_mean_target_contact_delta"), float("-inf")),
            _safe_float(row.get("source_group_mean_target_contact_abs_delta"), float("-inf")),
        ),
        reverse=True,
    )
    negatives.sort(
        key=lambda row: (
            _safe_float(row.get("source_group_mean_target_contact_delta"), float("inf")),
            -_safe_float(row.get("source_group_mean_target_contact_abs_delta"), float("-inf")),
        ),
    )

    n_total = max(0, int(args.top_k_cases))
    n_positive = n_total // 2
    n_negative = n_total // 2
    if n_total % 2 == 1:
        n_positive += 1

    selected_rows = positives[:n_positive] + negatives[:n_negative]
    selected_keys = {(int(row["feature_id"]), str(row["accession"])) for row in selected_rows}

    if len(selected_rows) < n_total:
        leftovers = [
            row
            for row in (positives[n_positive:] + negatives[n_negative:])
            if (int(row["feature_id"]), str(row["accession"])) not in selected_keys
        ]
        leftovers.sort(
            key=lambda row: abs(_safe_float(row.get("source_group_mean_target_contact_delta"), 0.0)),
            reverse=True,
        )
        selected_rows.extend(leftovers[: max(0, n_total - len(selected_rows))])

    if not selected_rows:
        raise SystemExit(
            "No case-study examples were selected from the source results. "
            "Try --include-proxy or an explicit --case-specs list."
        )

    selected_rows.sort(
        key=lambda row: _safe_float(row.get("source_group_mean_target_contact_delta"), 0.0),
        reverse=True,
    )
    selected_rows = selected_rows[:n_total]
    cases = [case_map[(int(row["feature_id"]), str(row["accession"]))] for row in selected_rows]
    return cases, selected_rows


def _baseline_row(
    *,
    case: bgca.CaseSpec,
    case_dir: Path,
    target_summary_row: dict[str, Any],
    original_contacts: np.ndarray,
    original_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_dir": case_dir.name,
        "feature_id": case.feature_id,
        "accession": case.accession,
        "protein_rank": case.protein_rank,
        "selection_source": case.selection_source,
        "max_activation": case.max_activation,
        "mean_activation": case.mean_activation,
        "intervention_label": "scaled_target",
        "intervention_mode": "single_node_contact_scale_sweep",
        "intervention_feature_count": 1,
        "sample_index": 1,
        "sequence": case.sequence,
        "decode_mode": "direct_hidden_override",
        "n_mutations": 0,
        "mutation_positions": [],
        "mutable_positions": [],
        "patch_positions": case.patch_positions,
        "lesion_positions": case.lesion_positions,
        "feature_ids_scaled": [case.feature_id],
        "feature_scale": 1.0,
        "ablation_scale": 1.0,
        "ablation_strength": 0.0,
        "top_geometric_feature": case.top_geometric_feature,
        "source_sort_key": str(target_summary_row.get("source_sort_key", "")),
        "source_sort_value": _safe_float(target_summary_row.get("source_sort_value")),
        "source_target_margin": _safe_float(target_summary_row.get("paired_target_contact_abs_delta_margin")),
        "source_patch_margin": _safe_float(target_summary_row.get("paired_patch_contact_l1_margin")),
        "source_global_margin": _safe_float(target_summary_row.get("paired_global_contact_l1_margin")),
        "n_residues": int(original_contacts.shape[0]),
        "local_hidden_delta": 0.0,
        "context_hidden_delta": 0.0,
        "mean_kl_patch": 0.0,
        "mean_entropy_delta_patch": 0.0,
        "argmax_changes_patch": 0.0,
        "argmax_changes_mutable": 0.0,
        **original_metrics,
    }


def _summarize_scale_case(
    rows: list[dict[str, Any]],
    case: bgca.CaseSpec,
) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            _safe_float(row.get("ablation_strength"), 0.0),
            _safe_float(row.get("ablation_scale"), 1.0),
        ),
    )
    strengths = np.asarray([float(row["ablation_strength"]) for row in ordered], dtype=float)
    target_metric_name = str(ordered[0].get("target_contact_metric_name", ""))
    target_metric_is_proxy = bool(ordered[0].get("target_contact_metric_is_proxy", False))

    def _metric_arr(key: str) -> np.ndarray:
        return np.asarray([_safe_float(row.get(key)) for row in ordered], dtype=float)

    def _slope(values: np.ndarray) -> float:
        mask = np.isfinite(values) & np.isfinite(strengths)
        if mask.sum() < 2:
            return float("nan")
        x = strengths[mask]
        y = values[mask]
        denom = float(np.var(x))
        if denom <= 1e-12:
            return float("nan")
        return float(np.cov(x, y, bias=True)[0, 1] / denom)

    def _monotonic(values: np.ndarray, tol: float = 1e-6) -> bool:
        filtered = values[np.isfinite(values)]
        if filtered.size < 2:
            return False
        return bool(np.all(np.diff(filtered) >= -tol))

    target_abs = _metric_arr("target_contact_metric_abs_delta")
    target_signed = _metric_arr("target_contact_metric_delta")
    patch_l1 = _metric_arr("patch_contact_l1_delta")
    patch_long_l1 = _metric_arr("patch_long_contact_l1_delta")
    global_l1 = _metric_arr("global_contact_l1_delta")
    kl_patch = _metric_arr("mean_kl_patch")
    hidden = _metric_arr("local_hidden_delta")

    full_row = max(ordered, key=lambda row: _safe_float(row.get("ablation_strength"), 0.0))
    return {
        "case_dir": f"f{case.feature_id:04d}_{case.accession}",
        "feature_id": case.feature_id,
        "accession": case.accession,
        "protein_rank": case.protein_rank,
        "selection_source": case.selection_source,
        "top_geometric_feature": case.top_geometric_feature,
        "structural_category": case.structural_category,
        "target_contact_metric_name": target_metric_name,
        "target_contact_metric_is_proxy": target_metric_is_proxy,
        "n_scales": len(ordered),
        "scales": [float(row["ablation_scale"]) for row in ordered],
        "ablation_strengths": [float(row["ablation_strength"]) for row in ordered],
        "target_abs_slope": _slope(target_abs),
        "target_signed_slope": _slope(target_signed),
        "patch_contact_l1_slope": _slope(patch_l1),
        "patch_long_contact_l1_slope": _slope(patch_long_l1),
        "global_contact_l1_slope": _slope(global_l1),
        "kl_patch_slope": _slope(kl_patch),
        "local_hidden_delta_slope": _slope(hidden),
        "target_abs_monotonic": _monotonic(target_abs),
        "patch_contact_l1_monotonic": _monotonic(patch_l1),
        "global_contact_l1_monotonic": _monotonic(global_l1),
        "full_ablation_scale": float(full_row["ablation_scale"]),
        "full_ablation_target_abs_delta": _safe_float(full_row.get("target_contact_metric_abs_delta")),
        "full_ablation_target_signed_delta": _safe_float(full_row.get("target_contact_metric_delta")),
        "full_ablation_patch_contact_l1_delta": _safe_float(full_row.get("patch_contact_l1_delta")),
        "full_ablation_patch_long_contact_l1_delta": _safe_float(full_row.get("patch_long_contact_l1_delta")),
        "full_ablation_global_contact_l1_delta": _safe_float(full_row.get("global_contact_l1_delta")),
        "full_ablation_mean_kl_patch": _safe_float(full_row.get("mean_kl_patch")),
        "full_ablation_local_hidden_delta": _safe_float(full_row.get("local_hidden_delta")),
        "full_ablation_target_efficiency": _safe_divide(
            _safe_float(full_row.get("target_contact_metric_abs_delta")),
            _safe_float(full_row.get("local_hidden_delta")),
        ),
        "full_ablation_patch_efficiency": _safe_divide(
            _safe_float(full_row.get("patch_contact_l1_delta")),
            _safe_float(full_row.get("local_hidden_delta")),
        ),
        "source_sort_key": str(full_row.get("source_sort_key", "")),
        "source_sort_value": _safe_float(full_row.get("source_sort_value")),
        "source_target_margin": _safe_float(full_row.get("source_target_margin")),
        "source_patch_margin": _safe_float(full_row.get("source_patch_margin")),
        "source_global_margin": _safe_float(full_row.get("source_global_margin")),
    }


def _run_case(
    case: bgca.CaseSpec,
    source_row: dict[str, Any],
    args: argparse.Namespace,
    tokenizer,
    esm_model,
    sae,
    root_out: Path,
    case_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_dir = root_out / f"{case_index:02d}_f{case.feature_id:04d}_{case.accession}"
    case_dir.mkdir(parents=True, exist_ok=True)
    bgca.save_json(case_dir / "case_spec.json", asdict(case))
    bgca.save_json(case_dir / "source_case_summary.json", source_row)
    bgca.write_fasta(case_dir / "original_sequence.fasta", [("original", case.sequence)])

    orig_logits, orig_hidden, token_ids, attn_mask, original_attentions = bcpa.extract_hidden_states_and_attentions(
        esm_model,
        tokenizer,
        case.sequence,
        args.layer,
        args.device,
    )
    seq_len = len(case.sequence)
    with torch.no_grad():
        residue_hidden = orig_hidden[0, 1 : seq_len + 1, :]
        normalised, original_norms = sae._normalize_input_and_get_norms(residue_hidden)
        features = sae.encode(normalised)

    original_contacts = bcpa.predict_contacts_from_attention_stack(
        esm_model,
        token_ids,
        attn_mask,
        original_attentions,
    )[0].detach().cpu().float().numpy()
    np.savez_compressed(
        case_dir / "original_contacts.npz",
        contacts=original_contacts,
        patch_positions=np.asarray(case.patch_positions, dtype=int),
        lesion_positions=np.asarray(case.lesion_positions, dtype=int),
    )

    interventions_payload = {
        "target_feature_id": case.feature_id,
        "positions": list(case.lesion_positions),
        "ablation_scales": [float(scale) for scale in args.ablation_scales],
    }
    bgca.save_json(case_dir / "interventions.json", interventions_payload)

    original_metrics = bcpa.compute_contact_map_metrics(
        original_contacts,
        original_contacts,
        case,
        min_seq_sep_short=args.min_seq_sep_short,
        min_seq_sep_long=args.min_seq_sep_long,
        prob_threshold=args.contact_prob_threshold,
    )
    baseline = _baseline_row(
        case=case,
        case_dir=case_dir,
        target_summary_row=source_row,
        original_contacts=original_contacts,
        original_metrics=original_metrics,
    )

    all_rows: list[dict[str, Any]] = []
    for scale_idx, scale in enumerate(args.ablation_scales, start=1):
        scale = float(scale)
        scale_dir = case_dir / _scale_label(scale)
        scale_dir.mkdir(parents=True, exist_ok=True)
        bgca.write_fasta(scale_dir / "sequence.fasta", [("original", case.sequence)])

        if np.isclose(scale, 1.0):
            row = dict(baseline)
            row["sample_index"] = scale_idx
            row["ablation_scale"] = scale
            row["ablation_strength"] = 1.0 - scale
            bgca.save_json(scale_dir / "metrics.json", row)
            np.savez_compressed(
                scale_dir / "contacts.npz",
                candidate_contacts=original_contacts,
                delta_contacts=np.zeros_like(original_contacts),
                patch_positions=np.asarray(case.patch_positions, dtype=int),
                lesion_positions=np.asarray(case.lesion_positions, dtype=int),
            )
            all_rows.append(row)
            continue

        features_mod = bgca.apply_scaled_intervention(features, [case.feature_id], case.lesion_positions, scale)
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
        local_hidden_delta = float(np.mean(hidden_delta[case.lesion_positions])) if case.lesion_positions else 0.0
        context_hidden_delta = (
            float(np.mean(hidden_delta[case.context_positions]))
            if case.context_positions
            else 0.0
        )

        print(f"    target scale={scale:.2f}: predicting contacts from direct hidden override ...")
        contact_pred = bcpa.predict_contacts_with_hidden_override(
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
        contact_metrics = bcpa.compute_contact_map_metrics(
            original_contacts,
            candidate_contacts,
            case,
            min_seq_sep_short=args.min_seq_sep_short,
            min_seq_sep_long=args.min_seq_sep_long,
            prob_threshold=args.contact_prob_threshold,
        )

        row: dict[str, Any] = {
            "case_dir": case_dir.name,
            "feature_id": case.feature_id,
            "accession": case.accession,
            "protein_rank": case.protein_rank,
            "selection_source": case.selection_source,
            "max_activation": case.max_activation,
            "mean_activation": case.mean_activation,
            "intervention_label": "scaled_target",
            "intervention_mode": "single_node_contact_scale_sweep",
            "intervention_feature_count": 1,
            "sample_index": scale_idx,
            "sequence": case.sequence,
            "decode_mode": "direct_hidden_override",
            "n_mutations": 0,
            "mutation_positions": [],
            "mutable_positions": [],
            "patch_positions": case.patch_positions,
            "lesion_positions": case.lesion_positions,
            "feature_ids_scaled": [case.feature_id],
            "feature_scale": scale,
            "ablation_scale": scale,
            "ablation_strength": 1.0 - scale,
            "top_geometric_feature": case.top_geometric_feature,
            "source_sort_key": str(source_row.get("source_sort_key", "")),
            "source_sort_value": _safe_float(source_row.get("source_sort_value")),
            "source_target_margin": _safe_float(source_row.get("paired_target_contact_abs_delta_margin")),
            "source_patch_margin": _safe_float(source_row.get("paired_patch_contact_l1_margin")),
            "source_global_margin": _safe_float(source_row.get("paired_global_contact_l1_margin")),
            "n_residues": int(candidate_contacts.shape[0]),
            "local_hidden_delta": local_hidden_delta,
            "context_hidden_delta": context_hidden_delta,
            **logit_effects,
            **contact_metrics,
        }
        bgca.save_json(scale_dir / "metrics.json", row)
        np.savez_compressed(
            scale_dir / "contacts.npz",
            candidate_contacts=candidate_contacts,
            delta_contacts=(candidate_contacts - original_contacts),
            patch_positions=np.asarray(case.patch_positions, dtype=int),
            lesion_positions=np.asarray(case.lesion_positions, dtype=int),
        )
        all_rows.append(row)

    case_summary = _summarize_scale_case(all_rows, case)
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
    if not args.ablation_scales:
        raise SystemExit("Please provide at least one scale via --ablation-scales.")
    args.ablation_scales = sorted({float(scale) for scale in args.ablation_scales}, reverse=True)
    if any(scale < 0.0 or scale > 1.0 for scale in args.ablation_scales):
        raise SystemExit("All --ablation-scales must lie between 0.0 and 1.0.")

    args.device = bcpa.resolve_runtime_device(args.device)
    bgca.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases, source_rows = _select_cases_from_source(args)
    requested_case_specs = [f"{case.feature_id}:{case.accession}" for case in cases]
    source_rows_for_save: list[dict[str, Any]] = []
    for row in source_rows:
        payload = dict(row)
        if args.selection_mode == "balanced_signed_target_delta":
            payload["source_sort_key"] = "source_group_mean_target_contact_delta"
            payload["source_sort_value"] = _safe_float(row.get("source_group_mean_target_contact_delta"))
        else:
            payload["source_sort_key"] = args.sort_key
            payload["source_sort_value"] = _safe_float(row.get(args.sort_key))
        source_rows_for_save.append(payload)

    print()
    print("=" * 90)
    print("  ProteinLens — Contact Predictor Case-Study Scale Sweep")
    print("=" * 90)
    print(f"  Selected {len(cases)} case-study case(s)")
    print(f"  Source results: {args.source_results_dir}")
    print(f"  ESM model: {args.esm_model}")
    print(f"  Selection mode: {args.selection_mode}")
    print(f"  Ablation scales: {', '.join(f'{scale:.2f}' for scale in args.ablation_scales)}")
    for idx, (case, row) in enumerate(zip(cases, source_rows_for_save, strict=True), start=1):
        patch_str = ",".join(str(p) for p in case.patch_positions)
        score = _safe_float(row.get("source_sort_value"))
        print(
            f"    {idx:2d}. f/{case.feature_id:04d}  {case.accession}  "
            f"{case.top_geometric_feature}  rank={case.protein_rank}  "
            f"patch={patch_str}  {row.get('source_sort_key')}={score:.6f}"
        )

    case_rows = [asdict(case) for case in cases]
    bgca.save_table(args.output_dir / "selected_cases.csv", case_rows)
    bgca.save_json(args.output_dir / "selected_cases.json", case_rows)
    bgca.save_json(args.output_dir / "selected_source_case_summaries.json", source_rows_for_save)

    manifest = {
        "source_results_dir": str(args.source_results_dir),
        "data_dir": str(args.data_dir),
        "esm_model": args.esm_model,
        "sae_dir": args.sae_dir,
        "layer": args.layer,
        "device": args.device,
        "intervention_mode": "single_node_contact_scale_sweep",
        "case_specs_requested": list(args.case_specs or []),
        "case_specs_selected": requested_case_specs,
        "top_k_cases": args.top_k_cases,
        "selection_mode": args.selection_mode,
        "sort_key": args.sort_key,
        "include_proxy": bool(args.include_proxy),
        "ablation_scales": [float(scale) for scale in args.ablation_scales],
        "min_seq_sep_short": args.min_seq_sep_short,
        "min_seq_sep_long": args.min_seq_sep_long,
        "contact_prob_threshold": args.contact_prob_threshold,
        "save_contact_maps": True,
        "dry_run": bool(args.dry_run),
        "n_cases": len(cases),
        "n_unique_features": len({case.feature_id for case in cases}),
    }

    if args.dry_run:
        (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\n  Dry run complete. Saved selected case-study cases to {args.output_dir}")
        print()
        return

    print(f"\n  Loading ESM2 + SAE on {args.device} ...")
    tokenizer, esm_model, sae = load_pipeline_models(args.sae_dir, args.esm_model, args.device)
    if hasattr(esm_model, "set_attn_implementation"):
        esm_model.set_attn_implementation("eager")
    _ = _get_decoder_weights(sae)

    if not hasattr(esm_model.esm, "contact_head"):
        raise RuntimeError(
            "The loaded ESM model does not expose a contact head, so predict_contacts cannot be used."
        )

    per_sample_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case_index, (case, source_row) in enumerate(zip(cases, source_rows_for_save, strict=True), start=1):
        print(f"\n[{case_index}/{len(cases)}] Running contact scale sweep for f/{case.feature_id:04d} on {case.accession} ...")
        rows, summary = _run_case(
            case,
            source_row,
            args,
            tokenizer,
            esm_model,
            sae,
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

    print(f"\n  Saved contact scale-sweep case studies to {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
