#!/usr/bin/env python3
"""Generate case-study figures for contact-predictor scale sweeps."""

from __future__ import annotations

import argparse
import json
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

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

import generate_paper_figures as paper  # noqa: E402


TARGET_METRIC_LABELS = {
    "patch_expected_contacts_per_residue": "Expected contacts / residue",
    "patch_expected_long_contacts_per_residue": "Expected long-range contacts / residue",
    "patch_fraction_long_contact_mass": "Fraction long-range contact mass",
    "patch_weighted_mean_seq_sep": "Weighted mean sequence separation",
    "patch_max_seq_sep_prob_ge_threshold": "Max sequence-separation contact span",
    "patch_max_long_contact_prob": "Max long-range contact probability",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/contact_predictor_scale_sweep_case_studies"),
        help="Completed contact scale-sweep run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_figures_preview/contact_scale_sweep_case_studies"),
        help="Directory where case-study figures will be written.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def nice_target_metric(name: str) -> str:
    return TARGET_METRIC_LABELS.get(name, name.replace("_", " "))


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
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
        }
    )


def _case_slug(case_summary: dict[str, Any]) -> str:
    return paper.slugify(f"f{int(case_summary['feature_id']):04d}_{case_summary['accession']}_contact_scale_sweep")


def _list_case_dirs(results_dir: Path) -> list[Path]:
    case_dirs = [path for path in sorted(results_dir.iterdir()) if path.is_dir() and "_f" in path.name]
    if not case_dirs:
        raise SystemExit(f"No case directories found in {results_dir}")
    return case_dirs


def _load_case_payload(case_dir: Path, min_seq_sep_short: int) -> dict[str, Any]:
    case_summary = load_json(case_dir / "case_summary.json")
    case_spec = load_json(case_dir / "case_spec.json")
    source_summary = load_json(case_dir / "source_case_summary.json")
    original_npz = np.load(case_dir / "original_contacts.npz")
    original_contacts = np.asarray(original_npz["contacts"], dtype=float)
    patch_positions = [int(pos) for pos in case_spec.get("patch_positions", [])]

    scale_dirs = [path for path in sorted(case_dir.iterdir()) if path.is_dir() and path.name.startswith("scale_")]
    rows: list[dict[str, Any]] = []
    contacts_by_scale: list[np.ndarray] = []
    for scale_dir in scale_dirs:
        row = load_json(scale_dir / "metrics.json")
        rows.append(row)
        contacts_npz = np.load(scale_dir / "contacts.npz")
        contacts_by_scale.append(np.asarray(contacts_npz["candidate_contacts"], dtype=float))

    order = np.argsort([float(row["ablation_strength"]) for row in rows])
    ordered_rows = [rows[idx] for idx in order]
    ordered_contacts = [contacts_by_scale[idx] for idx in order]

    return {
        "case_dir": case_dir,
        "case_summary": case_summary,
        "case_spec": case_spec,
        "source_summary": source_summary,
        "original_contacts": original_contacts,
        "patch_positions": patch_positions,
        "rows": ordered_rows,
        "contacts": ordered_contacts,
        "min_seq_sep_short": int(min_seq_sep_short),
    }


def _patch_contact_profile(
    contact_map: np.ndarray,
    patch_positions: list[int],
    min_seq_sep_short: int,
) -> np.ndarray:
    cm = np.asarray(contact_map, dtype=float)
    n = cm.shape[0]
    patch = [int(pos) for pos in patch_positions if 0 <= int(pos) < n]
    if not patch:
        return np.full(n, np.nan, dtype=float)

    sums = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)
    indices = np.arange(n, dtype=int)
    for pos in patch:
        valid = np.abs(indices - pos) >= min_seq_sep_short
        sums[valid] += cm[pos, valid]
        counts[valid] += 1.0
    out = np.full(n, np.nan, dtype=float)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    return out


def _add_patch_markers(ax, patch_positions: list[int], n_residues: int) -> None:
    valid = sorted({int(pos) for pos in patch_positions if 0 <= int(pos) < n_residues})
    for pos in valid:
        ax.axvline(pos, color="white", alpha=0.18, linewidth=0.6, zorder=4)


def _plot_case_figure(payload: dict[str, Any], output_dir: Path) -> None:
    case_summary = payload["case_summary"]
    source_summary = payload["source_summary"]
    rows = payload["rows"]
    contacts = payload["contacts"]
    patch_positions = payload["patch_positions"]
    original_contacts = payload["original_contacts"]
    min_seq_sep_short = int(payload["min_seq_sep_short"])
    n_residues = int(original_contacts.shape[0])

    target_metric_name = str(case_summary.get("target_contact_metric_name", ""))
    x = np.asarray([float(row["ablation_strength"]) for row in rows], dtype=float)
    target_delta = np.asarray([float(row["target_contact_metric_delta"]) for row in rows], dtype=float)
    patch_l1 = np.asarray([float(row["patch_contact_l1_delta"]) for row in rows], dtype=float)
    global_l1 = np.asarray([float(row["global_contact_l1_delta"]) for row in rows], dtype=float)
    target_candidate = np.asarray([float(row["target_contact_metric_candidate"]) for row in rows], dtype=float)
    target_original = float(rows[0]["target_contact_metric_original"])

    original_profile = _patch_contact_profile(original_contacts, patch_positions, min_seq_sep_short)
    candidate_profiles = np.asarray(
        [_patch_contact_profile(cm, patch_positions, min_seq_sep_short) for cm in contacts],
        dtype=float,
    )
    delta_profiles = candidate_profiles - original_profile[None, :]

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.7), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.plot(x, target_delta, color=paper.PALETTE["primary"], linewidth=2.0, label="Target metric delta")
    ax1.plot(x, patch_l1, color=paper.PALETTE["secondary_dark"], linewidth=1.8, label="Patch contact L1")
    ax1.plot(x, global_l1, color=paper.PALETTE["accent"], linewidth=1.8, label="Global contact L1")
    ax1.axhline(0.0, color=paper.PALETTE["muted"], linestyle="--", linewidth=1.0)
    ax1.set_xlabel("Ablation strength")
    ax1.set_ylabel("Effect size")
    ax1.set_title("Dose Response")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=8.5)

    ax2.plot(x, target_candidate, color=paper.PALETTE["secondary_dark"], linewidth=2.0)
    ax2.axhline(target_original, color=paper.PALETTE["muted"], linestyle="--", linewidth=1.1)
    ax2.set_xlabel("Ablation strength")
    ax2.set_ylabel(nice_target_metric(target_metric_name))
    ax2.set_title("Target Contact Metric")

    abs_vmax = float(np.nanmax(candidate_profiles)) if np.isfinite(candidate_profiles).any() else 1.0
    abs_vmax = max(abs_vmax, 1e-6)
    im_abs = ax3.imshow(
        np.ma.masked_invalid(candidate_profiles),
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=abs_vmax,
    )
    ax3.set_title("Patch Contact Profile Across Sweep")
    ax3.set_xlabel("Residue index")
    ax3.set_ylabel("Ablation strength (%)")
    ax3.set_yticks(np.arange(len(rows)))
    ax3.set_yticklabels([f"{int(round(100 * float(row['ablation_strength'])))}" for row in rows])
    _add_patch_markers(ax3, patch_positions, n_residues)
    cbar_abs = fig.colorbar(im_abs, ax=ax3, pad=0.02, fraction=0.046)
    cbar_abs.set_label("Mean patch contact probability")

    delta_vmax = float(np.nanmax(np.abs(delta_profiles))) if np.isfinite(delta_profiles).any() else 1.0
    delta_vmax = max(delta_vmax, 1e-6)
    im_delta = ax4.imshow(
        np.ma.masked_invalid(delta_profiles),
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-delta_vmax,
        vmax=delta_vmax,
    )
    ax4.set_title("Patch Contact Profile Delta")
    ax4.set_xlabel("Residue index")
    ax4.set_ylabel("Ablation strength (%)")
    ax4.set_yticks(np.arange(len(rows)))
    ax4.set_yticklabels([f"{int(round(100 * float(row['ablation_strength'])))}" for row in rows])
    _add_patch_markers(ax4, patch_positions, n_residues)
    cbar_delta = fig.colorbar(im_delta, ax=ax4, pad=0.02, fraction=0.046)
    cbar_delta.set_label("Delta contact probability")

    title = (
        f"f{int(case_summary['feature_id']):04d} · {case_summary['accession']} · "
        f"{case_summary['top_geometric_feature']}"
    )
    subtitle = (
        f"source margin={float(source_summary.get('paired_target_contact_abs_delta_margin', 0.0)):.3f} · "
        f"full-ablation target delta={float(case_summary.get('full_ablation_target_signed_delta', 0.0)):.3f}"
    )
    fig.suptitle(f"{title}\n{subtitle}", y=1.02, fontsize=13.5, fontweight="bold")

    for ax in axes.ravel():
        sns.despine(ax=ax)

    slug = _case_slug(case_summary)
    pdf_path = output_dir / f"{slug}.pdf"
    png_path = output_dir / f"{slug}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(payloads: list[dict[str, Any]], output_dir: Path) -> None:
    if not payloads:
        return

    n_cases = len(payloads)
    fig, axes = plt.subplots(n_cases, 2, figsize=(12.8, 3.9 * n_cases), constrained_layout=True)
    if n_cases == 1:
        axes = np.asarray([axes], dtype=object)

    for row_axes, payload in zip(axes, payloads, strict=True):
        ax_line, ax_heat = row_axes
        case_summary = payload["case_summary"]
        source_summary = payload["source_summary"]
        rows = payload["rows"]
        contacts = payload["contacts"]
        patch_positions = payload["patch_positions"]
        original_contacts = payload["original_contacts"]
        min_seq_sep_short = int(payload["min_seq_sep_short"])

        x = np.asarray([float(row["ablation_strength"]) for row in rows], dtype=float)
        target_delta = np.asarray([float(row["target_contact_metric_delta"]) for row in rows], dtype=float)
        patch_l1 = np.asarray([float(row["patch_contact_l1_delta"]) for row in rows], dtype=float)
        global_l1 = np.asarray([float(row["global_contact_l1_delta"]) for row in rows], dtype=float)

        original_profile = _patch_contact_profile(original_contacts, patch_positions, min_seq_sep_short)
        candidate_profiles = np.asarray(
            [_patch_contact_profile(cm, patch_positions, min_seq_sep_short) for cm in contacts],
            dtype=float,
        )
        delta_profiles = candidate_profiles - original_profile[None, :]

        ax_line.plot(x, target_delta, color=paper.PALETTE["primary"], linewidth=2.0, label="Target delta")
        ax_line.plot(x, patch_l1, color=paper.PALETTE["secondary_dark"], linewidth=1.7, label="Patch L1")
        ax_line.plot(x, global_l1, color=paper.PALETTE["accent"], linewidth=1.7, label="Global L1")
        ax_line.axhline(0.0, color=paper.PALETTE["muted"], linestyle="--", linewidth=1.0)
        ax_line.set_xlabel("Ablation strength")
        ax_line.set_ylabel("Effect size")
        ax_line.set_title(
            f"f{int(case_summary['feature_id']):04d} · {case_summary['accession']}\n"
            f"source margin={float(source_summary.get('paired_target_contact_abs_delta_margin', 0.0)):.3f}"
        )
        ax_line.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=8)

        delta_vmax = float(np.nanmax(np.abs(delta_profiles))) if np.isfinite(delta_profiles).any() else 1.0
        delta_vmax = max(delta_vmax, 1e-6)
        im = ax_heat.imshow(
            np.ma.masked_invalid(delta_profiles),
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-delta_vmax,
            vmax=delta_vmax,
        )
        ax_heat.set_title("Patch contact delta profile")
        ax_heat.set_xlabel("Residue index")
        ax_heat.set_ylabel("Ablation strength (%)")
        ax_heat.set_yticks(np.arange(len(rows)))
        ax_heat.set_yticklabels([f"{int(round(100 * float(row['ablation_strength'])))}" for row in rows])
        _add_patch_markers(ax_heat, patch_positions, int(original_contacts.shape[0]))
        cbar = fig.colorbar(im, ax=ax_heat, pad=0.02, fraction=0.046)
        cbar.set_label("Delta")

        sns.despine(ax=ax_line)
        sns.despine(ax=ax_heat)

    pdf_path = output_dir / "contact_scale_sweep_case_study_overview.pdf"
    png_path = output_dir / "contact_scale_sweep_case_study_overview.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    manifest_path = args.results_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not find {manifest_path}")
    manifest = load_json(manifest_path)
    min_seq_sep_short = int(manifest.get("min_seq_sep_short", 3))

    payloads = [
        _load_case_payload(case_dir, min_seq_sep_short=min_seq_sep_short)
        for case_dir in _list_case_dirs(args.results_dir)
    ]

    for payload in payloads:
        _plot_case_figure(payload, args.output_dir)
    _plot_overview(payloads, args.output_dir)

    summary_lines = [
        f"n_cases: {len(payloads)}",
        f"results_dir: {args.results_dir}",
    ]
    for payload in payloads:
        case_summary = payload["case_summary"]
        summary_lines.append(
        f"f{int(case_summary['feature_id']):04d} {case_summary['accession']} "
            f"{case_summary['top_geometric_feature']} "
            f"full_ablation_target_signed_delta={float(case_summary.get('full_ablation_target_signed_delta', 0.0)):.6f}"
        )
    (args.output_dir / "contact_scale_sweep_case_study_summary.txt").write_text("\n".join(summary_lines) + "\n")

    print(f"Wrote contact scale-sweep case-study figures to {args.output_dir}")


if __name__ == "__main__":
    main()
