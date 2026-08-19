#!/usr/bin/env python3
"""Generate scatter summaries for contact-predictor ablation runs."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm, colors
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

import generate_paper_figures as paper  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/contact_predictor_ablation_run1"),
        help="Directory containing finished contact ablation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_figures_preview/contact_predictor_ablation"),
        help="Directory where summary figures will be written.",
    )
    parser.add_argument(
        "--annotate-top-k",
        type=int,
        default=8,
        help="How many strongest positive-margin cases to annotate.",
    )
    parser.add_argument(
        "--label-min-margin",
        type=float,
        default=0.0,
        help="Only annotate cases with paired target-contact margin at least this large.",
    )
    parser.add_argument(
        "--feature-rank-file",
        type=Path,
        default=Path("results/contact_top_feature_ids_ranked.txt"),
        help="Newline- or comma-separated ranked feature-id list used for continuous point colors.",
    )
    parser.add_argument(
        "--symlog",
        action="store_true",
        help="Use symlog axes for the parity scatter to compress heavy tails.",
    )
    parser.add_argument(
        "--symlog-linthresh",
        type=float,
        default=0.05,
        help="Linear threshold used when --symlog is enabled.",
    )
    parser.add_argument(
        "--aggregate-by",
        choices=["case", "feature"],
        default="case",
        help="Plot one point per protein case or average rows into one point per feature.",
    )
    parser.add_argument(
        "--color-by",
        choices=["rank", "signed_margin", "geometry_pr_auc", "fixed_red"],
        default="rank",
        help="Color points either by feature rank or by target-minus-control contact shift.",
    )
    parser.add_argument(
        "--geometry-analysis-file",
        type=Path,
        default=Path("feature_data_cluster/geometry_primary_analysis.json"),
        help="Feature-level geometry analysis JSON used to attach geometry PR-AUC scores.",
    )
    parser.add_argument(
        "--max-control-context-cosine",
        type=float,
        default=None,
        help=(
            "If set, keep only cases whose matched-control intervention has "
            "max context cosine to the target at or below this threshold."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def case_label(row: dict[str, Any]) -> str:
    return f"f{int(row['feature_id']):04d}"


def aggregation_label(aggregate_by: str) -> str:
    return "Feature-averaged" if aggregate_by == "feature" else "Per-protein"


def axis_prefix(aggregate_by: str) -> str:
    return "Feature-mean" if aggregate_by == "feature" else ""


def load_ranked_feature_ids(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find feature rank file: {path}")
    tokens = path.read_text().replace(",", "\n").splitlines()
    ranked: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        stripped = token.strip()
        if not stripped:
            continue
        fid = int(stripped)
        if fid in seen:
            continue
        seen.add(fid)
        ranked.append(fid)
    if not ranked:
        raise SystemExit(f"No feature ids found in {path}")
    return ranked


def canonical_case_key(value: str) -> str:
    parts = str(value).split("_", 1)
    if parts and parts[0].isdigit() and len(parts) > 1:
        return parts[1]
    return str(value)


def attach_feature_ranks(rows: list[dict[str, Any]], rank_file: Path) -> tuple[colors.Normalize, colors.Colormap]:
    ranked_ids = load_ranked_feature_ids(rank_file)
    rank_map = {fid: idx + 1 for idx, fid in enumerate(ranked_ids)}

    next_rank = len(rank_map) + 1
    missing_ids = sorted({int(row["feature_id"]) for row in rows if int(row["feature_id"]) not in rank_map})
    for fid in missing_ids:
        rank_map[fid] = next_rank
        next_rank += 1

    for row in rows:
        row["feature_rank"] = int(rank_map[int(row["feature_id"])])

    present_ranks = [int(row["feature_rank"]) for row in rows]
    vmin = min(present_ranks) if present_ranks else 1
    vmax = max(present_ranks) if present_ranks else max(1, len(rank_map))
    return colors.Normalize(vmin=vmin, vmax=vmax), matplotlib.colormaps["viridis_r"]


def attach_control_similarity_metrics(rows: list[dict[str, Any]], results_dir: Path) -> None:
    similarity_by_case: dict[str, dict[str, float]] = {}
    for path in results_dir.glob("*/interventions.json"):
        payload = load_json(path)
        diagnostics = list(payload.get("matched_control", {}).get("diagnostics", []))
        if not diagnostics:
            continue
        context_vals = np.asarray(
            [float(diag.get("context_cosine_to_target", float("nan"))) for diag in diagnostics],
            dtype=float,
        )
        decoder_vals = np.asarray(
            [float(diag.get("decoder_cosine_to_target", float("nan"))) for diag in diagnostics],
            dtype=float,
        )
        context_vals = context_vals[np.isfinite(context_vals)]
        decoder_vals = decoder_vals[np.isfinite(decoder_vals)]
        case_key = canonical_case_key(path.parent.name)
        similarity_by_case[case_key] = {
            "matched_control_context_cosine_max": float(np.max(context_vals)) if context_vals.size else float("nan"),
            "matched_control_context_cosine_mean": float(np.mean(context_vals)) if context_vals.size else float("nan"),
            "matched_control_decoder_cosine_max": float(np.max(decoder_vals)) if decoder_vals.size else float("nan"),
            "matched_control_decoder_cosine_mean": float(np.mean(decoder_vals)) if decoder_vals.size else float("nan"),
        }

    for row in rows:
        key = canonical_case_key(str(row.get("case_dir", "")))
        metrics = similarity_by_case.get(key, {})
        for metric_key, metric_value in metrics.items():
            row[metric_key] = metric_value


def attach_signed_margin_colors(
    rows: list[dict[str, Any]],
    *,
    robust_quantile: float = 0.95,
) -> tuple[colors.Normalize, colors.Colormap]:
    signed_margins = np.asarray(
        [float(row.get("target_control_signed_margin", 0.0)) for row in rows],
        dtype=float,
    )
    finite = signed_margins[np.isfinite(signed_margins)]
    abs_finite = np.abs(finite)
    vmax = float(np.quantile(abs_finite, robust_quantile)) if abs_finite.size else 1.0
    if vmax <= 1e-8:
        vmax = 1.0
    cmap = LinearSegmentedColormap.from_list(
        "contact_parity_diverging",
        [
            paper.PALETTE["primary"],
            "#DCEBFF",
            "#F8FAFC",
            "#FDE2E2",
            paper.PALETTE["accent"],
        ],
    )
    return colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), cmap


def attach_geometry_pr_auc(rows: list[dict[str, Any]], analysis_file: Path) -> None:
    if not analysis_file.exists():
        raise FileNotFoundError(f"Could not find geometry analysis file: {analysis_file}")
    payload = load_json(analysis_file)
    feature_payload = dict(payload.get("features", {}))
    for row in rows:
        info = feature_payload.get(str(int(row["feature_id"])), {})
        row["geometry_pr_auc"] = float(info.get("geom_pr_auc", float("nan")))


def attach_geometry_pr_auc_colors(rows: list[dict[str, Any]]) -> tuple[colors.Normalize, colors.Colormap]:
    values = np.asarray([float(row.get("geometry_pr_auc", float("nan"))) for row in rows], dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.max(finite)) if finite.size else 1.0
    vmax = max(1.0, vmax)
    cmap = LinearSegmentedColormap.from_list(
        "geometry_pr_auc_sequential",
        [
            paper.PALETTE["primary"],
            "#9EC5FE",
            "#E9EEF6",
            "#FCA5A5",
            paper.PALETTE["accent"],
        ],
    )
    return colors.Normalize(vmin=0.0, vmax=vmax), cmap


def attach_plot_metrics(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        target = float(row.get("group_mean_target_contact_abs_delta", 0.0))
        control = float(row.get("control_mean_target_contact_abs_delta", 0.0))
        signed_margin = target - control
        row["target_control_signed_margin"] = signed_margin
        row["target_control_abs_margin"] = abs(signed_margin)


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray(
        [
            float(row[key])
            for row in rows
            if key in row and row[key] is not None and np.isfinite(float(row[key]))
        ],
        dtype=float,
    )
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def aggregate_rows_by_feature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[int(row["feature_id"])].append(row)

    numeric_keys = [
        "protein_rank",
        "max_activation",
        "mean_activation",
        "lesion_feature_count",
        "control_feature_count",
        "group_mean_target_contact_abs_delta",
        "control_mean_target_contact_abs_delta",
        "group_mean_patch_contact_l1_delta",
        "control_mean_patch_contact_l1_delta",
        "group_mean_patch_long_contact_l1_delta",
        "control_mean_patch_long_contact_l1_delta",
        "group_mean_global_contact_l1_delta",
        "control_mean_global_contact_l1_delta",
        "group_mean_kl_patch",
        "control_mean_kl_patch",
        "paired_target_contact_abs_delta_margin",
        "paired_patch_contact_l1_margin",
        "paired_global_contact_l1_margin",
        "matched_control_context_cosine_max",
        "matched_control_context_cosine_mean",
        "matched_control_decoder_cosine_max",
        "matched_control_decoder_cosine_mean",
        "geometry_pr_auc",
    ]

    aggregated_rows: list[dict[str, Any]] = []
    for feature_id, feature_rows in sorted(grouped.items()):
        first = feature_rows[0]
        aggregate = dict(first)
        aggregate["case_dir"] = f"f{feature_id:04d}"
        aggregate["accession"] = "feature_mean"
        aggregate["selection_source"] = "feature_mean"
        aggregate["n_cases_aggregated"] = len(feature_rows)
        aggregate["protein_rank"] = _mean_numeric(feature_rows, "protein_rank")
        aggregate["max_activation"] = _mean_numeric(feature_rows, "max_activation")
        aggregate["mean_activation"] = _mean_numeric(feature_rows, "mean_activation")

        # Keep a representative patch site for reference while averaging all numeric effects.
        patch_positions: list[int] = []
        for row in feature_rows:
            patch_positions.extend(int(pos) for pos in row.get("patch_positions", []))
        aggregate["patch_positions"] = sorted(set(patch_positions))

        lesion_ids: list[int] = []
        control_ids: list[int] = []
        for row in feature_rows:
            lesion_ids.extend(int(fid) for fid in row.get("lesion_feature_ids", []))
            control_ids.extend(int(fid) for fid in row.get("control_feature_ids", []))
        aggregate["lesion_feature_ids"] = sorted(set(lesion_ids))
        aggregate["control_feature_ids"] = sorted(set(control_ids))

        for key in numeric_keys:
            if any(key in row for row in feature_rows):
                aggregate[key] = _mean_numeric(feature_rows, key)

        aggregate["target_contact_metric_is_proxy"] = bool(
            sum(bool(row.get("target_contact_metric_is_proxy", False)) for row in feature_rows)
            >= math.ceil(len(feature_rows) / 2)
        )
        aggregated_rows.append(aggregate)
    return aggregated_rows


def rank_for_labels(rows: list[dict[str, Any]], top_k: int, min_margin: float) -> list[dict[str, Any]]:
    ranked = [
        row for row in rows
        if float(row.get("paired_target_contact_abs_delta_margin", float("-inf"))) >= min_margin
    ]
    ranked.sort(
        key=lambda row: (
            float(row.get("paired_target_contact_abs_delta_margin", 0.0)),
            float(row.get("group_mean_target_contact_abs_delta", 0.0)),
        ),
        reverse=True,
    )
    return ranked[: max(0, top_k)]


def rank_for_abs_margin_labels(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    ranked = list(rows)
    ranked.sort(
        key=lambda row: (
            float(row.get("target_control_abs_margin", 0.0)),
            -float(row.get("feature_rank", float("inf"))),
        ),
        reverse=True,
    )
    return ranked[: max(0, top_k)]


def apply_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "axes.facecolor": paper.PALETTE["background"],
            "figure.facecolor": "white",
            "axes.edgecolor": paper.PALETTE["grid"],
            "grid.color": paper.PALETTE["grid"],
            "grid.linewidth": 0.8,
            "axes.labelcolor": paper.PALETTE["ink"],
            "xtick.color": paper.PALETTE["ink"],
            "ytick.color": paper.PALETTE["ink"],
            "text.color": paper.PALETTE["ink"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
        }
    )


def proxy_marker(row: dict[str, Any]) -> str:
    return "X" if bool(row.get("target_contact_metric_is_proxy", False)) else "o"


def scatter_by_proxy(
    ax,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    color_key: str,
    cmap: colors.Colormap,
    norm: colors.Normalize,
    fixed_color: str | None = None,
    alpha: float = 0.82,
    size: float = 42,
) -> None:
    for marker, proxy_value in (("o", False), ("X", True)):
        subset = [row for row in rows if bool(row.get("target_contact_metric_is_proxy", False)) == proxy_value]
        if not subset:
            continue
        x = np.asarray([float(row[x_key]) for row in subset], dtype=float)
        y = np.asarray([float(row[y_key]) for row in subset], dtype=float)
        scatter_kwargs = {
            "x": x,
            "y": y,
            "s": size,
            "alpha": alpha,
            "marker": marker,
            "edgecolors": "white",
            "linewidths": 0.55,
            "rasterized": len(subset) > 200,
            "zorder": 3,
        }
        if fixed_color is not None:
            ax.scatter(color=fixed_color, **scatter_kwargs)
        else:
            color_values = np.asarray([float(row[color_key]) for row in subset], dtype=float)
            ax.scatter(
                c=color_values,
                cmap=cmap,
                norm=norm,
                **scatter_kwargs,
            )


def add_annotation_labels(
    ax,
    rows_to_label: list[dict[str, Any]],
    x_key: str,
    y_key: str,
) -> None:
    for idx, row in enumerate(rows_to_label):
        x = float(row[x_key])
        y = float(row[y_key])
        dx = 5 if idx % 2 == 0 else -5
        dy = 4 + 4 * (idx % 3)
        ax.annotate(
            case_label(row),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color=paper.PALETTE["ink"],
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": paper.PALETTE["grid"],
                "lw": 0.7,
                "alpha": 0.9,
            },
            zorder=5,
        )


def format_stats_text(rows: list[dict[str, Any]], *, aggregate_by: str) -> str:
    eps = 1e-8
    wins = sum(
        float(row.get("group_mean_target_contact_abs_delta", float("nan")))
        > float(row.get("control_mean_target_contact_abs_delta", float("nan")))
        for row in rows
    )
    margins = np.asarray(
        [float(row.get("paired_target_contact_abs_delta_margin", 0.0)) for row in rows],
        dtype=float,
    )
    ratios = np.asarray(
        [
            (float(row.get("group_mean_target_contact_abs_delta", 0.0)) + eps)
            / (float(row.get("control_mean_target_contact_abs_delta", 0.0)) + eps)
            for row in rows
        ],
        dtype=float,
    )
    lines = [f"n = {len(rows)}"]
    if aggregate_by == "feature":
        protein_total = int(sum(int(row.get("n_cases_aggregated", 1)) for row in rows))
        lines.append(f"n proteins = {protein_total}")
    lines.extend(
        [
            f"Above diagonal = {wins}/{len(rows)}",
            f"Median target/control = {np.median(ratios):.2f}x",
            f"Median margin = {np.median(margins):.3f}",
        ]
    )
    return "\n".join(lines)


def add_rank_colorbar(
    fig,
    ax,
    *,
    cmap: colors.Colormap,
    norm: colors.Normalize,
    label: str,
    invert: bool,
) -> None:
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.055)
    cbar.set_label(label)
    if invert:
        cbar.ax.invert_yaxis()
    cbar.outline.set_edgecolor(paper.PALETTE["grid"])
    cbar.outline.set_linewidth(0.8)


def add_rank_median_line(
    ax,
    rows: list[dict[str, Any]],
    *,
    n_bins: int = 20,
) -> None:
    sorted_rows = sorted(rows, key=lambda row: float(row.get("feature_rank", float("inf"))))
    if not sorted_rows:
        return

    x_vals: list[float] = []
    y_vals: list[float] = []
    total = len(sorted_rows)
    for idx in range(max(1, n_bins)):
        start = idx * total // max(1, n_bins)
        end = (idx + 1) * total // max(1, n_bins)
        subset = sorted_rows[start:end]
        if not subset:
            continue
        x_vals.append(float(np.median([float(row["feature_rank"]) for row in subset])))
        y_vals.append(float(np.median([float(row["target_control_abs_margin"]) for row in subset])))

    if not x_vals:
        return

    ax.plot(
        x_vals,
        y_vals,
        color=paper.PALETTE["primary"],
        linewidth=1.8,
        alpha=0.95,
        label="Binned median",
        zorder=4,
    )


def draw_target_vs_control_panel(
    ax,
    rows: list[dict[str, Any]],
    annotate_top_k: int,
    min_margin: float,
    *,
    aggregate_by: str,
    cmap: colors.Colormap,
    norm: colors.Normalize,
    color_key: str,
    fixed_color: str | None,
    symlog: bool,
    symlog_linthresh: float,
) -> None:
    scatter_by_proxy(
        ax,
        rows,
        "control_mean_target_contact_abs_delta",
        "group_mean_target_contact_abs_delta",
        color_key=color_key,
        cmap=cmap,
        norm=norm,
        fixed_color=fixed_color,
    )

    values = []
    for row in rows:
        values.append(float(row["control_mean_target_contact_abs_delta"]))
        values.append(float(row["group_mean_target_contact_abs_delta"]))
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    vmax = float(finite.max()) if finite.size else 1.0
    diagonal = np.linspace(0.0, vmax * 1.03, 200)
    ax.plot(
        diagonal,
        diagonal,
        color=paper.PALETTE["muted"],
        linewidth=1.2,
        linestyle="--",
        zorder=1,
    )
    ax.fill_between(
        diagonal,
        diagonal,
        vmax * 1.03,
        color=paper.PALETTE["accent"],
        alpha=0.05,
        zorder=0,
    )
    ax.fill_between(
        diagonal,
        0.0,
        diagonal,
        color=paper.PALETTE["primary"],
        alpha=0.05,
        zorder=0,
    )

    if symlog:
        ax.set_xscale("symlog", linthresh=symlog_linthresh)
        ax.set_yscale("symlog", linthresh=symlog_linthresh)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    prefix = axis_prefix(aggregate_by)
    x_label = f"{prefix} matched-control shift" if prefix else "Matched-control contact shift"
    y_label = f"{prefix} target-neuron shift" if prefix else "Target-neuron contact shift"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{aggregation_label(aggregate_by)} Contact Causal Parity")

    rows_to_label = rank_for_labels(rows, annotate_top_k, min_margin)
    add_annotation_labels(
        ax,
        rows_to_label,
        "control_mean_target_contact_abs_delta",
        "group_mean_target_contact_abs_delta",
    )

    stats_text = format_stats_text(rows, aggregate_by=aggregate_by)
    ax.text(
        0.03,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.28",
            "fc": "white",
            "ec": paper.PALETTE["grid"],
            "lw": 0.8,
            "alpha": 0.96,
        },
    )
    ax.set_aspect("equal", adjustable="box")
    sns.despine(ax=ax)


def draw_rank_vs_abs_margin_panel(
    ax,
    rows: list[dict[str, Any]],
    annotate_top_k: int,
    *,
    aggregate_by: str,
    cmap: colors.Colormap,
    norm: colors.Normalize,
    color_key: str,
    fixed_color: str | None,
    symlog: bool,
    symlog_linthresh: float,
) -> None:
    scatter_by_proxy(
        ax,
        rows,
        "feature_rank",
        "target_control_abs_margin",
        color_key=color_key,
        cmap=cmap,
        norm=norm,
        fixed_color=fixed_color,
        alpha=0.78,
        size=34,
    )
    show_median_line = aggregate_by != "feature"
    if show_median_line:
        add_rank_median_line(ax, rows)

    if symlog:
        ax.set_yscale("symlog", linthresh=symlog_linthresh)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Neuron rank (1 = strongest)")
    ylabel = (
        "Feature-mean |target - control| shift"
        if aggregate_by == "feature"
        else "|Target - control| contact shift"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(f"{aggregation_label(aggregate_by)} Rank Vs Distance From Diagonal")

    rows_to_label = rank_for_abs_margin_labels(rows, annotate_top_k)
    add_annotation_labels(
        ax,
        rows_to_label,
        "feature_rank",
        "target_control_abs_margin",
    )

    median_abs = float(np.median(np.asarray([float(row["target_control_abs_margin"]) for row in rows], dtype=float)))
    ax.text(
        0.03,
        0.97,
        f"n = {len(rows)}\nMedian |margin| = {median_abs:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.28",
            "fc": "white",
            "ec": paper.PALETTE["grid"],
            "lw": 0.8,
            "alpha": 0.96,
        },
    )
    if show_median_line:
        ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            fontsize=8.5,
        )
    sns.despine(ax=ax)


def plot_target_vs_control(
    rows: list[dict[str, Any]],
    output_dir: Path,
    annotate_top_k: int,
    min_margin: float,
    *,
    aggregate_by: str,
    cmap: colors.Colormap,
    norm: colors.Normalize,
    color_key: str,
    colorbar_label: str,
    invert_colorbar: bool,
    show_colorbar: bool,
    fixed_color: str | None,
    symlog: bool,
    symlog_linthresh: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    draw_target_vs_control_panel(
        ax,
        rows,
        annotate_top_k,
        min_margin,
        aggregate_by=aggregate_by,
        cmap=cmap,
        norm=norm,
        color_key=color_key,
        fixed_color=fixed_color,
        symlog=symlog,
        symlog_linthresh=symlog_linthresh,
    )
    if show_colorbar:
        add_rank_colorbar(
            fig,
            ax,
            cmap=cmap,
            norm=norm,
            label=colorbar_label,
            invert=invert_colorbar,
        )
    fig.tight_layout()

    pdf_path = output_dir / "contact_causal_parity_scatter.pdf"
    png_path = output_dir / "contact_causal_parity_scatter.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_parity_with_rank_effect(
    rows: list[dict[str, Any]],
    output_dir: Path,
    annotate_top_k: int,
    min_margin: float,
    *,
    aggregate_by: str,
    cmap: colors.Colormap,
    norm: colors.Normalize,
    color_key: str,
    colorbar_label: str,
    invert_colorbar: bool,
    show_colorbar: bool,
    fixed_color: str | None,
    symlog: bool,
    symlog_linthresh: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.9), constrained_layout=True)
    draw_target_vs_control_panel(
        axes[0],
        rows,
        annotate_top_k,
        min_margin,
        aggregate_by=aggregate_by,
        cmap=cmap,
        norm=norm,
        color_key=color_key,
        fixed_color=fixed_color,
        symlog=symlog,
        symlog_linthresh=symlog_linthresh,
    )
    draw_rank_vs_abs_margin_panel(
        axes[1],
        rows,
        annotate_top_k,
        aggregate_by=aggregate_by,
        cmap=cmap,
        norm=norm,
        color_key=color_key,
        fixed_color=fixed_color,
        symlog=symlog,
        symlog_linthresh=symlog_linthresh,
    )
    if show_colorbar:
        add_rank_colorbar(
            fig,
            axes,
            cmap=cmap,
            norm=norm,
            label=colorbar_label,
            invert=invert_colorbar,
        )

    pdf_path = output_dir / "contact_causal_parity_and_rank_effect_scatter.pdf"
    png_path = output_dir / "contact_causal_parity_and_rank_effect_scatter.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_specificity_vs_global_drift(
    rows: list[dict[str, Any]],
    output_dir: Path,
    annotate_top_k: int,
    min_margin: float,
    *,
    aggregate_by: str,
    cmap: colors.Colormap,
    norm: colors.Normalize,
    color_key: str,
    colorbar_label: str,
    invert_colorbar: bool,
    show_colorbar: bool,
    fixed_color: str | None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 5.6))
    scatter_by_proxy(
        ax,
        rows,
        "group_mean_global_contact_l1_delta",
        "paired_target_contact_abs_delta_margin",
        color_key=color_key,
        cmap=cmap,
        norm=norm,
        fixed_color=fixed_color,
    )
    ax.axhline(0.0, color=paper.PALETTE["muted"], linestyle="--", linewidth=1.1, zorder=1)

    rows_to_label = rank_for_labels(rows, annotate_top_k, min_margin)
    add_annotation_labels(
        ax,
        rows_to_label,
        "group_mean_global_contact_l1_delta",
        "paired_target_contact_abs_delta_margin",
    )

    positive = sum(float(row.get("paired_target_contact_abs_delta_margin", 0.0)) > 0 for row in rows)
    med_global = np.median(
        np.asarray([float(row.get("group_mean_global_contact_l1_delta", 0.0)) for row in rows], dtype=float)
    )
    med_margin = np.median(
        np.asarray([float(row.get("paired_target_contact_abs_delta_margin", 0.0)) for row in rows], dtype=float)
    )
    ax.text(
        0.03,
        0.97,
        f"n = {len(rows)}\nPositive margin = {positive}/{len(rows)}\nMedian global drift = {med_global:.4f}\nMedian margin = {med_margin:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.28",
            "fc": "white",
            "ec": paper.PALETTE["grid"],
            "lw": 0.8,
            "alpha": 0.96,
        },
    )

    ax.annotate(
        "More specific\ncontact effect",
        xy=(0.02, 0.93),
        xytext=(0.16, 0.82),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": paper.PALETTE["ink"], "lw": 0.9},
        fontsize=9,
        ha="left",
        va="center",
    )

    prefix = aggregation_label(aggregate_by)
    ax.set_xlabel(f"{prefix} target lesion global contact-map drift")
    ax.set_ylabel(f"{prefix} target minus matched-control contact shift")
    ax.set_title(f"{prefix} Specificity Vs Global Drift")
    if show_colorbar:
        add_rank_colorbar(
            fig,
            ax,
            cmap=cmap,
            norm=norm,
            label=colorbar_label,
            invert=invert_colorbar,
        )
    sns.despine(ax=ax)
    fig.tight_layout()

    pdf_path = output_dir / "contact_specificity_vs_global_drift_scatter.pdf"
    png_path = output_dir / "contact_specificity_vs_global_drift_scatter.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    aggregate_by: str,
    max_control_context_cosine: float | None,
    n_rows_before_filter: int,
) -> None:
    eps = 1e-8
    lines = []
    lines.append(f"aggregation_mode: {aggregate_by}")
    lines.append(f"n_rows_before_filters: {n_rows_before_filter}")
    if max_control_context_cosine is not None:
        lines.append(f"max_control_context_cosine: {max_control_context_cosine}")
    if aggregate_by == "feature":
        protein_total = int(sum(int(row.get("n_cases_aggregated", 1)) for row in rows))
        lines.append(f"n_features: {len(rows)}")
        lines.append(f"n_underlying_proteins: {protein_total}")
        lines.append(
            f"median_proteins_per_feature: {np.median(np.asarray([float(row.get('n_cases_aggregated', 1)) for row in rows], dtype=float)):.1f}"
        )
    else:
        lines.append(f"n_cases: {len(rows)}")
    wins = sum(
        float(row.get("group_mean_target_contact_abs_delta", float("nan")))
        > float(row.get("control_mean_target_contact_abs_delta", float("nan")))
        for row in rows
    )
    lines.append(f"n_above_diagonal_target_vs_control: {wins}")
    lines.append(f"fraction_above_diagonal_target_vs_control: {wins / max(len(rows), 1):.3f}")

    ratios = np.asarray(
        [
            (float(row.get("group_mean_target_contact_abs_delta", 0.0)) + eps)
            / (float(row.get("control_mean_target_contact_abs_delta", 0.0)) + eps)
            for row in rows
        ],
        dtype=float,
    )
    margins = np.asarray(
        [float(row.get("paired_target_contact_abs_delta_margin", 0.0)) for row in rows],
        dtype=float,
    )
    patch_margins = np.asarray(
        [float(row.get("paired_patch_contact_l1_margin", 0.0)) for row in rows],
        dtype=float,
    )
    global_margins = np.asarray(
        [float(row.get("paired_global_contact_l1_margin", 0.0)) for row in rows],
        dtype=float,
    )
    lines.append(f"median_target_control_ratio: {np.median(ratios):.6f}")
    lines.append(f"median_target_contact_margin: {np.median(margins):.6f}")
    lines.append(f"median_patch_contact_l1_margin: {np.median(patch_margins):.6f}")
    lines.append(f"median_global_contact_l1_margin: {np.median(global_margins):.6f}")

    ranked = sorted(
        rows,
        key=lambda row: float(row.get("paired_target_contact_abs_delta_margin", 0.0)),
        reverse=True,
    )
    lines.append("")
    header = "top_features_by_target_margin:" if aggregate_by == "feature" else "top_cases_by_target_margin:"
    lines.append(header)
    for row in ranked[:10]:
        line = (
            f"  {case_label(row)} {row.get('accession')} {row.get('top_geometric_feature')}"
            f" margin={float(row.get('paired_target_contact_abs_delta_margin', 0.0)):.6f}"
            f" ratio={((float(row.get('group_mean_target_contact_abs_delta', 0.0)) + eps) / (float(row.get('control_mean_target_contact_abs_delta', 0.0)) + eps)):.3f}"
        )
        if aggregate_by == "feature":
            line += f" n_proteins={int(row.get('n_cases_aggregated', 1))}"
        lines.append(line)

    (output_dir / "contact_ablation_scatter_summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    apply_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_summaries_path = args.results_dir / "case_summaries.json"
    if not case_summaries_path.exists():
        raise FileNotFoundError(f"Could not find {case_summaries_path}")

    rows = list(load_json(case_summaries_path))
    if not rows:
        raise SystemExit(f"No rows found in {case_summaries_path}")
    n_rows_before_filter = len(rows)
    attach_control_similarity_metrics(rows, args.results_dir)

    if args.max_control_context_cosine is not None:
        filtered_rows = [
            row
            for row in rows
            if np.isfinite(float(row.get("matched_control_context_cosine_max", float("nan"))))
            and float(row.get("matched_control_context_cosine_max", float("nan"))) <= args.max_control_context_cosine
        ]
        rows = filtered_rows
        if not rows:
            raise SystemExit(
                "No rows remain after applying "
                f"--max-control-context-cosine {args.max_control_context_cosine}."
            )

    if args.aggregate_by == "feature":
        rows = aggregate_rows_by_feature(rows)

    attach_plot_metrics(rows)
    attach_geometry_pr_auc(rows, args.geometry_analysis_file)
    rank_norm, rank_cmap = attach_feature_ranks(rows, args.feature_rank_file)
    fixed_color: str | None = None
    if args.color_by == "signed_margin":
        norm, cmap = attach_signed_margin_colors(rows)
        color_key = "target_control_signed_margin"
        colorbar_label = "Target - control contact shift"
        invert_colorbar = False
        show_colorbar = True
    elif args.color_by == "geometry_pr_auc":
        norm, cmap = attach_geometry_pr_auc_colors(rows)
        color_key = "geometry_pr_auc"
        colorbar_label = "Feature geometry PR-AUC"
        invert_colorbar = False
        show_colorbar = True
    elif args.color_by == "fixed_red":
        norm, cmap = rank_norm, rank_cmap
        color_key = "feature_rank"
        colorbar_label = ""
        invert_colorbar = False
        show_colorbar = False
        fixed_color = paper.PALETTE["accent"]
    else:
        norm, cmap = rank_norm, rank_cmap
        color_key = "feature_rank"
        colorbar_label = "Neuron rank (1 = strongest)"
        invert_colorbar = True
        show_colorbar = True

    plot_target_vs_control(
        rows,
        args.output_dir,
        annotate_top_k=args.annotate_top_k,
        min_margin=args.label_min_margin,
        aggregate_by=args.aggregate_by,
        cmap=cmap,
        norm=norm,
        color_key=color_key,
        colorbar_label=colorbar_label,
        invert_colorbar=invert_colorbar,
        show_colorbar=show_colorbar,
        fixed_color=fixed_color,
        symlog=args.symlog,
        symlog_linthresh=args.symlog_linthresh,
    )
    plot_specificity_vs_global_drift(
        rows,
        args.output_dir,
        annotate_top_k=args.annotate_top_k,
        min_margin=args.label_min_margin,
        aggregate_by=args.aggregate_by,
        cmap=cmap,
        norm=norm,
        color_key=color_key,
        colorbar_label=colorbar_label,
        invert_colorbar=invert_colorbar,
        show_colorbar=show_colorbar,
        fixed_color=fixed_color,
    )
    plot_parity_with_rank_effect(
        rows,
        args.output_dir,
        annotate_top_k=args.annotate_top_k,
        min_margin=args.label_min_margin,
        aggregate_by=args.aggregate_by,
        cmap=cmap,
        norm=norm,
        color_key=color_key,
        colorbar_label=colorbar_label,
        invert_colorbar=invert_colorbar,
        show_colorbar=show_colorbar,
        fixed_color=fixed_color,
        symlog=args.symlog,
        symlog_linthresh=args.symlog_linthresh,
    )
    write_summary(
        rows,
        args.output_dir,
        aggregate_by=args.aggregate_by,
        max_control_context_cosine=args.max_control_context_cosine,
        n_rows_before_filter=n_rows_before_filter,
    )

    print(f"Wrote contact ablation summary figures to {args.output_dir}")


if __name__ == "__main__":
    main()
