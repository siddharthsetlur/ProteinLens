#!/usr/bin/env python3
"""Generate paper-ready PDF figures from existing feature_data_cluster artifacts.

This script turns the JSON payloads already consumed by the web interface into
publication-friendly PDF plots organized into interpretable folder names.

Outputs:
  - {output_dir}/overview/*.pdf
  - {output_dir}/case_studies/<group>/<rank>_<family_slug>/*.pdf
  - {output_dir}/cross_family/<rank>_<feature_slug>/*.pdf
  - {output_dir}/manifest.json

Defaults are tuned to the strongest families/features currently surfaced by the
dashboard, but can be overridden via CLI flags.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None
    ImageOps = None
try:
    from Bio.Align import PairwiseAligner
except Exception:  # pragma: no cover - optional dependency fallback
    PairwiseAligner = None

DEFAULT_CASE_FAMILY_CODES = [
    "IPR003594",  # Histidine kinase/HSP90-like ATPase domain
    "IPR004358",  # Signal transduction histidine kinase-related protein, C-terminal
    "IPR013087",  # Zinc finger C2H2-type
]
DEFAULT_SEQUENCE_MOTIF_CASE_FAMILY_CODES = [
    "IPR042099",  # ANL, N-terminal domain
    "IPR005218",  # GroEL-like equatorial domain
    "IPR034466",  # S-adenosyl-L-methionine-dependent methyltransferase superfamily
]

CASE_STUDY_GROUP_SPECS = (
    {
        "key": "geometry_primary_dominant",
        "dirname": "geometry_primary_dominant",
        "label": "Geometry primary/dominant",
        "default_codes": DEFAULT_CASE_FAMILY_CODES,
    },
    {
        "key": "sequence_motif_primary_dominant",
        "dirname": "sequence_motif_primary_dominant",
        "label": "Sequence motif primary/dominant",
        "default_codes": DEFAULT_SEQUENCE_MOTIF_CASE_FAMILY_CODES,
    },
)

DEFAULT_CROSS_FEATURE_IDS = [1403, 1595, 235, 3958]
DEFAULT_GEOMETRY_DOMINANT_FEATURE_IDS = [1403, 2531, 235]
DEFAULT_SEQUENCE_DOMINANT_FEATURE_IDS: list[int] = []
DEFAULT_BIOLOGY_DOMINANT_FEATURE_IDS: list[int] = []
DEFAULT_MAX_CASE_SUMMARY_FAMILIES = 40

OVERVIEW_DIRNAME = "overview"
CASE_DIRNAME = "case_studies"
CROSS_DIRNAME = "cross_family"
GEOMETRY_DOMINANT_DIRNAME = "geometry_dominant"
SEQUENCE_DOMINANT_DIRNAME = "sequence_dominant"
BIOLOGY_DOMINANT_DIRNAME = "biology_dominant"

SPOTLIGHT_MODE_SPECS = (
    {
        "mode": "geometry",
        "manifest_key": "geometry_dominant",
        "dirname": GEOMETRY_DOMINANT_DIRNAME,
        "label": "Geometry dominant",
        "default_ids": DEFAULT_GEOMETRY_DOMINANT_FEATURE_IDS,
        "ids_arg": "geometry_dominant_feature_ids",
        "max_arg": "max_geometry_dominant_features",
        "all_arg": "all_geometry_dominant_features",
    },
    {
        "mode": "sequence_motif",
        "manifest_key": "sequence_dominant",
        "dirname": SEQUENCE_DOMINANT_DIRNAME,
        "label": "Sequence motif dominant",
        "default_ids": DEFAULT_SEQUENCE_DOMINANT_FEATURE_IDS,
        "ids_arg": "sequence_dominant_feature_ids",
        "max_arg": "max_sequence_dominant_features",
        "all_arg": "all_sequence_dominant_features",
    },
    {
        "mode": "biology",
        "manifest_key": "biology_dominant",
        "dirname": BIOLOGY_DOMINANT_DIRNAME,
        "label": "Biology dominant",
        "default_ids": DEFAULT_BIOLOGY_DOMINANT_FEATURE_IDS,
        "ids_arg": "biology_dominant_feature_ids",
        "max_arg": "max_biology_dominant_features",
        "all_arg": "all_biology_dominant_features",
    },
)

PALETTE = {
    "ink": "#1F2937",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "background": "#F8FAFC",
    "primary": "#2563EB",
    "primary_light": "#93C5FD",
    "secondary": "#D97706",
    "secondary_dark": "#B45309",
    "secondary_light": "#FDE68A",
    "accent": "#DC2626",
    "accent_dark": "#991B1B",
    "success": "#16A34A",
    "muted_fill": "#CBD5E1",
    "slate": "#334155",
}

STRUCTURE_HIGHLIGHT_COLOR = PALETTE["accent"]
STRUCTURE_HIGHLIGHT_COLOR_INT = int(STRUCTURE_HIGHLIGHT_COLOR.lstrip("#"), 16)

Q_VALUE_LABELS = {
    "position_f1_padj": "Position",
    "interpro_res_f1_padj": "InterPro residue",
    "cath_res_f1_padj": "CATH residue",
}
GEOMETRY_Q_KEYS = ("geometry_prauc_padj",)
SEQ_MOTIF_Q_KEYS = ("motif_pr_auc_padj", "motif_f1_padj")
EXCLUDED_BEST_ANNOTATION_Q_KEYS = set(GEOMETRY_Q_KEYS) | set(SEQ_MOTIF_Q_KEYS)

CHROME_CANDIDATES = [
    os.environ.get("CHROME_BIN", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

INTERPRO_PROTEIN_URL = (
    "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/"
    "{accession}?page_size=200"
)
INTERPRO_FETCH_TIMEOUT_SEC = 8.0
CASE_STUDY_MEMBER_LIMIT = 5
CASE_STUDY_MEMBER_SPECIAL_CASES = {
    "IPR013087": {
        "keep_top": 4,
        "include_feature_ids": [5368],
    },
}
CASE_DOMAIN_SHADE_COLOR = "#D1D5DB"
CASE_SHARED_STRUCTURE_GRID_COLS = 3
CASE_SHARED_STRUCTURE_GRID_ROWS = 2
CASE_SHARED_STRUCTURE_VIEWER_WIDTH = 520
CASE_SHARED_STRUCTURE_VIEWER_HEIGHT = 480
CASE_SHARED_STRUCTURE_GAP = 20
CASE_SHARED_STRUCTURE_ROW_GAP = 30
CASE_SHARED_STRUCTURE_PADDING = 24
CASE_SHARED_STRUCTURE_CARD_OVERHEAD = 176
CASE_SHARED_STRUCTURE_GLOBAL_ZOOM = 0.97
CASE_STUDY_ACTIVATION_SPAN_REDUNDANCY_WEIGHT = 0.6
CASE_STUDY_ACTIVATION_REDUNDANCY_THRESHOLD = 0.35
GEOMETRY_DOMINANT_VIEWER_WIDTH = 520
GEOMETRY_DOMINANT_VIEWER_HEIGHT = 400
GEOMETRY_DOMINANT_PROTEIN_GLOBAL_ZOOM = 0.94
# Previous runs used a Chrome device scale factor of 2; 10 yields 5x the
# linear pixel dimensions for the PNG assets while preserving the same layout.
ASSET_RENDER_DEVICE_SCALE_FACTOR = 10


@dataclass
class RenderConfig:
    base_output_dir: Path
    export_clean_variants: bool = True
    clean_root_name: str = "panel_ready"
    skip_structure_renders: bool = False


RENDER_CONFIG: RenderConfig | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("feature_data_cluster"),
        help="Path to a built feature data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_figures"),
        help="Directory where PDF plots will be written",
    )
    parser.add_argument(
        "--case-family-codes",
        nargs="*",
        default=None,
        help="Explicit InterPro case-study family codes to plot",
    )
    parser.add_argument(
        "--cross-feature-ids",
        nargs="*",
        type=int,
        default=None,
        help="Explicit cross-family feature IDs to plot",
    )
    parser.add_argument(
        "--max-case-families",
        type=int,
        default=3,
        help="Number of case-study families to plot when no explicit codes are given",
    )
    parser.add_argument(
        "--max-cross-features",
        type=int,
        default=4,
        help="Number of cross-family features to plot when no explicit IDs are given",
    )
    parser.add_argument(
        "--all-case-families",
        action="store_true",
        help="Plot every family present in case_study_families.json",
    )
    parser.add_argument(
        "--all-cross-features",
        action="store_true",
        help="Plot every feature marked cross-family in cross_family_geometry.json",
    )
    parser.add_argument(
        "--geometry-dominant-feature-ids",
        nargs="*",
        type=int,
        default=None,
        help="Explicit geometry-dominant spotlight feature IDs to plot",
    )
    parser.add_argument(
        "--max-geometry-dominant-features",
        type=int,
        default=3,
        help="Number of geometry-dominant spotlights to produce when no explicit IDs are given",
    )
    parser.add_argument(
        "--all-geometry-dominant-features",
        action="store_true",
        help="Plot every geometry-dominant feature that passes the spotlight filters",
    )
    parser.add_argument(
        "--sequence-dominant-feature-ids",
        nargs="*",
        type=int,
        default=None,
        help="Explicit sequence-motif-dominant spotlight feature IDs to plot",
    )
    parser.add_argument(
        "--max-sequence-dominant-features",
        type=int,
        default=0,
        help="Number of sequence-motif-dominant spotlights to produce when no explicit IDs are given",
    )
    parser.add_argument(
        "--all-sequence-dominant-features",
        action="store_true",
        help="Plot every sequence-motif-dominant feature that passes the spotlight filters",
    )
    parser.add_argument(
        "--biology-dominant-feature-ids",
        nargs="*",
        type=int,
        default=None,
        help="Explicit biology-dominant spotlight feature IDs to plot",
    )
    parser.add_argument(
        "--max-biology-dominant-features",
        type=int,
        default=0,
        help="Number of biology-dominant spotlights to produce when no explicit IDs are given",
    )
    parser.add_argument(
        "--all-biology-dominant-features",
        action="store_true",
        help="Plot every biology-dominant feature that passes the spotlight filters",
    )
    parser.add_argument(
        "--no-clean-variants",
        action="store_true",
        help="Disable mirrored title-free panel-ready exports",
    )
    parser.add_argument(
        "--clean-root-name",
        type=str,
        default="panel_ready",
        help="Folder name used for mirrored title-free exports",
    )
    parser.add_argument(
        "--skip-structure-renders",
        action="store_true",
        help=(
            "Disable headless Chrome / 3Dmol structure snapshots and use "
            "Matplotlib-only fallback panels."
        ),
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Skip overview plots.",
    )
    parser.add_argument(
        "--skip-case-studies",
        action="store_true",
        help="Skip case-study bundles.",
    )
    parser.add_argument(
        "--skip-cross-family",
        action="store_true",
        help="Skip cross-family bundles.",
    )
    parser.add_argument(
        "--max-case-summary-families",
        type=int,
        default=DEFAULT_MAX_CASE_SUMMARY_FAMILIES,
        help=(
            "Maximum number of families shown in the overview case-family "
            "summary. Use 0 to plot all families."
        ),
    )
    return parser.parse_args()


def set_paper_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.08)
    sns.set_palette("colorblind")
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.facecolor": "white",
        "axes.edgecolor": PALETTE["slate"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10.2,
        "legend.frameon": False,
        "legend.fontsize": 8.6,
        "legend.title_fontsize": 8.8,
        "axes.linewidth": 0.8,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.55,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
    })


def slugify(text: str, max_len: int = 72) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    if not slug:
        slug = "untitled"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug


def wrap(text: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def polish_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color=PALETTE["grid"], linewidth=0.7, alpha=0.7)
    ax.tick_params(length=3.0, width=0.8, colors=PALETTE["ink"])
    ax.xaxis.label.set_color(PALETTE["ink"])
    ax.yaxis.label.set_color(PALETTE["ink"])


def save_pdf(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)

    if RENDER_CONFIG and RENDER_CONFIG.export_clean_variants:
        relative = path.relative_to(RENDER_CONFIG.base_output_dir)
        clean_path = RENDER_CONFIG.base_output_dir / RENDER_CONFIG.clean_root_name / relative
        clean_path.parent.mkdir(parents=True, exist_ok=True)

        title_cache = []
        for ax in fig.axes:
            title_cache.append((
                ax.get_title(loc="left"),
                ax.get_title(loc="center"),
                ax.get_title(loc="right"),
            ))
            ax.set_title("", loc="left")
            ax.set_title("", loc="center")
            ax.set_title("", loc="right")

        suptitle_text = None
        if getattr(fig, "_suptitle", None) is not None:
            suptitle_text = fig._suptitle.get_text()
            fig._suptitle.set_text("")

        fig.savefig(clean_path)

        for ax, (left, center, right) in zip(fig.axes, title_cache):
            if left:
                ax.set_title(left, loc="left")
            if center:
                ax.set_title(center, loc="center")
            if right:
                ax.set_title(right, loc="right")

        if suptitle_text is not None and getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_text(suptitle_text)

    plt.close(fig)
    return str(path)


def mirror_generated_asset(path: Path) -> None:
    if not RENDER_CONFIG or not RENDER_CONFIG.export_clean_variants:
        return
    try:
        relative = path.relative_to(RENDER_CONFIG.base_output_dir)
    except ValueError:
        return
    clean_path = RENDER_CONFIG.base_output_dir / RENDER_CONFIG.clean_root_name / relative
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, clean_path)


def normalize_structure_card_border(path: Path, color: str) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return

    with Image.open(path) as img:
        base = img.convert("RGBA")

    w, h = base.size
    top_left_cleanup = max(44, int(min(w, h) * 0.060))
    bottom_right_cleanup = max(22, int(min(w, h) * 0.032))
    padded_border = max(20, int(min(w, h) * 0.026))
    stroke = max(4, int(min(w, h) * 0.005))
    radius = max(20, int(min(w, h) * 0.04))

    cleaned = base.copy()
    draw = ImageDraw.Draw(cleaned)
    white = (255, 255, 255, 255)
    draw.rectangle((0, 0, w, top_left_cleanup), fill=white)
    draw.rectangle((0, 0, top_left_cleanup, h), fill=white)
    draw.rectangle((w - bottom_right_cleanup, 0, w, h), fill=white)
    draw.rectangle((0, h - bottom_right_cleanup, w, h), fill=white)

    canvas = Image.new("RGBA", (w + 2 * padded_border, h + 2 * padded_border), white)
    canvas.paste(cleaned, (padded_border, padded_border))
    draw_canvas = ImageDraw.Draw(canvas)
    rgb = tuple(int(round(channel * 255)) for channel in to_rgb(color))
    inset = padded_border + stroke
    draw_canvas.rounded_rectangle(
        (
            inset,
            inset,
            canvas.width - inset,
            canvas.height - inset,
        ),
        radius=radius,
        outline=rgb + (255,),
        width=stroke,
    )
    canvas.save(path)


def add_png_margin(path: Path, margin: int = 18, color: str = "white") -> None:
    if Image is None or ImageOps is None or margin <= 0 or not path.exists():
        return
    try:
        with Image.open(path) as img:
            expanded = ImageOps.expand(img, border=margin, fill=color)
            expanded.save(path)
    except Exception:
        return


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def first_available_float(mapping: dict[str, Any], keys: list[str], default: float = 0.0) -> tuple[float, str | None]:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return safe_float(mapping[key], default), key
    return default, None


def is_valid_q(value: Any) -> bool:
    try:
        q_value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(q_value) and q_value > 0.0


def q_field_label(key: str) -> str:
    if key in Q_VALUE_LABELS:
        return Q_VALUE_LABELS[key]
    stem = re.sub(r"_padj$", "", key)
    stem = stem.replace("_", " ")
    return stem.title()


def q_values_except_seq_motif_and_geometry(info: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in info.items():
        if not key.endswith("_padj") or key in EXCLUDED_BEST_ANNOTATION_Q_KEYS:
            continue
        q_value = safe_float(value, math.nan)
        if is_valid_q(q_value):
            values[q_field_label(key)] = q_value
    return values


def best_q_value(q_values: dict[str, float]) -> tuple[float, str | None]:
    valid = [(label, value) for label, value in q_values.items() if is_valid_q(value)]
    if not valid:
        return math.nan, None
    label, value = min(valid, key=lambda item: item[1])
    return value, label


class FigureData:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.coverage = load_json(data_dir / "survey_coverage.json")
        self.geometry_primary = load_json(data_dir / "geometry_primary_analysis.json")
        self.cross_family = load_json(data_dir / "cross_family_geometry.json")
        self.case_studies = load_json(data_dir / "case_study_families.json")
        self.geometry_summary = load_json(data_dir / "geometry_enrichment" / "summary.json")
        self.interpro_summary = load_json(data_dir / "interpro_enrichment" / "summary.json")

        self.annotation_score_label = "Best raw annotation F1"
        self.motif_metric_key = "motif_f1"
        self.geometry_primary_rows = self._build_geometry_primary_rows()
        self.has_q_values = any(
            is_valid_q(row.get("geometry_q"))
            and is_valid_q(row.get("best_non_motif_annotation_q"))
            for row in self.geometry_primary_rows
        )
        self.cross_family_rows = self._build_cross_family_rows()

    def _build_geometry_primary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        interpro_features = self.interpro_summary.get("features", {})
        geometry_features = self.geometry_summary.get("features", {})
        null_thresholds = self.geometry_primary.get("null_thresholds", {})
        motif_null, motif_null_key = first_available_float(
            null_thresholds,
            ["motif_f1", "motif_pr_auc"],
            1.0,
        )
        if not motif_null:
            motif_null = 1.0
        self.motif_metric_key = motif_null_key or "motif_f1"
        if self.motif_metric_key == "motif_pr_auc":
            self.annotation_score_label = "Best raw annotation score"
        position_null = safe_float(null_thresholds.get("position_f1"), 1.0) or 1.0
        interpro_res_null = safe_float(null_thresholds.get("interpro_res_f1"), 1.0) or 1.0
        cath_res_null = safe_float(null_thresholds.get("cath_res_f1"), 1.0) or 1.0

        for fid_str, info in self.geometry_primary.get("features", {}).items():
            ip = interpro_features.get(fid_str, {})
            geom = geometry_features.get(fid_str, {})
            cov = self.coverage.get(fid_str, {})
            motif_f1, motif_metric_key = first_available_float(
                info,
                ["motif_f1", "motif_pr_auc"],
                0.0,
            )
            if motif_metric_key == "motif_pr_auc":
                self.motif_metric_key = "motif_pr_auc"
                self.annotation_score_label = "Best raw annotation score"
            geometry_q, _ = first_available_float(info, list(GEOMETRY_Q_KEYS), math.nan)
            seq_motif_q, _ = first_available_float(info, list(SEQ_MOTIF_Q_KEYS), math.nan)
            non_motif_annotation_q_values = q_values_except_seq_motif_and_geometry(info)
            best_non_motif_annotation_q, best_non_motif_annotation_q_source = best_q_value(
                non_motif_annotation_q_values
            )
            position_f1 = safe_float(info.get("position_f1"), 0.0)
            interpro_res_f1 = safe_float(info.get("interpro_res_f1"), 0.0)
            source_scores = {
                "Sequence Motif": motif_f1,
                "Position": position_f1,
                "InterPro Residue": interpro_res_f1,
            }
            source_scores_norm = {
                "Sequence Motif": motif_f1 / motif_null,
                "Position": position_f1 / position_null,
                "InterPro Residue": interpro_res_f1 / interpro_res_null,
            }
            cath_res_f1 = None
            if "cath_res_f1" in info:
                cath_res_f1 = safe_float(info.get("cath_res_f1"), 0.0)
                source_scores["CATH Residue"] = cath_res_f1
                source_scores_norm["CATH Residue"] = cath_res_f1 / cath_res_null
            best_annotation_f1 = max(source_scores.values())
            winning_sources = [
                name for name, value in source_scores.items()
                if math.isclose(value, best_annotation_f1, rel_tol=0.0, abs_tol=1e-9)
            ]
            best_annotation_source = winning_sources[0] if len(winning_sources) == 1 else "Tie"
            best_annotation_f1_norm = max(source_scores_norm.values())
            winning_sources_norm = [
                name for name, value in source_scores_norm.items()
                if math.isclose(value, best_annotation_f1_norm, rel_tol=0.0, abs_tol=1e-9)
            ]
            best_annotation_source_norm = winning_sources_norm[0] if len(winning_sources_norm) == 1 else "Tie"
            rows.append({
                "feature_id": int(fid_str),
                "geom_pr_auc": float(info.get("geom_pr_auc", 0.0)),
                "composite_score": float(info.get("composite_score", 0.0)),
                "concordance_f1": float(info.get("concordance_f1", 0.0)),
                "best_seq_f1": float(info.get("best_seq_f1", 0.0)),
                "best_annotation_f1": best_annotation_f1,
                "best_annotation_source": best_annotation_source,
                "best_annotation_f1_norm": best_annotation_f1_norm,
                "best_annotation_source_norm": best_annotation_source_norm,
                "motif_f1_norm": motif_f1 / motif_null,
                "position_f1_norm": position_f1 / position_null,
                "interpro_res_f1_norm": interpro_res_f1 / interpro_res_null,
                "cath_res_f1_norm": cath_res_f1 / cath_res_null if cath_res_f1 is not None else math.nan,
                "motif_f1": motif_f1,
                "motif_metric_key": motif_metric_key or self.motif_metric_key,
                "position_f1": position_f1,
                "interpro_res_f1": interpro_res_f1,
                "cath_res_f1": cath_res_f1,
                "geometry_q": geometry_q,
                "seq_motif_q": seq_motif_q,
                "non_motif_annotation_q_values": non_motif_annotation_q_values,
                "best_non_motif_annotation_q": best_non_motif_annotation_q,
                "best_non_motif_annotation_q_source": best_non_motif_annotation_q_source,
                "interpro_protein_f1": float(ip.get("top_protein_f1", 0.0) or 0.0),
                "interpro_protein_name": ip.get("top_protein_annotation_name", ""),
                "motif_rmsd": float(geom.get("motif_rmsd", math.nan)),
                "coverage_pct": float(cov.get("pct_proteins_activated", math.nan)),
                "is_geometry_primary": bool(info.get("is_geometry_primary", False)),
                "structural_category": info.get("structural_category", ""),
            })
        return rows

    def _build_cross_family_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for feat in self.cross_family.get("features", []):
            fid_str = str(feat["feature_id"])
            cov = self.coverage.get(fid_str, {})
            merged = dict(feat)
            merged["coverage_pct"] = float(cov.get("pct_proteins_activated", math.nan))
            rows.append(merged)
        return rows

    @lru_cache(maxsize=None)
    def feature_json(self, fid: int) -> dict[str, Any]:
        return load_json(self.data_dir / "features" / f"{fid:04d}.json")

    @lru_cache(maxsize=None)
    def feature_sequence_entries(self, fid: int) -> dict[str, dict[str, Any]]:
        return feature_sequence_entry_map(self.feature_json(fid))

    @lru_cache(maxsize=None)
    def geometry_json(self, fid: int) -> dict[str, Any]:
        return load_json(self.data_dir / "geometry_enrichment" / f"{fid:04d}.json")

    @lru_cache(maxsize=None)
    def interpro_json(self, fid: int) -> dict[str, Any]:
        return load_json(self.data_dir / "interpro_enrichment" / f"{fid:04d}.json")

    @lru_cache(maxsize=None)
    def cath_json(self, fid: int) -> dict[str, Any]:
        path = self.data_dir / "cath_enrichment" / f"{fid:04d}.json"
        if not path.exists():
            return {}
        return load_json(path)

    @lru_cache(maxsize=None)
    def motif_pwm_json(self, fid: int) -> dict[str, Any]:
        path = self.data_dir / "motif_pwm_enrichment" / f"{fid:04d}.json"
        if not path.exists():
            return {}
        return load_json(path)


def select_case_families(data: FigureData, args: argparse.Namespace) -> list[dict[str, Any]]:
    families = data.case_studies.get("families", [])
    if args.case_family_codes:
        want = set(args.case_family_codes)
        return [fam for fam in families if fam["annotation_code"] in want]
    if args.all_case_families:
        return families

    preferred = []
    family_map = {fam["annotation_code"]: fam for fam in families}
    for code in DEFAULT_CASE_FAMILY_CODES:
        if code in family_map:
            preferred.append(family_map[code])
    if len(preferred) >= args.max_case_families:
        return preferred[:args.max_case_families]

    chosen_codes = {fam["annotation_code"] for fam in preferred}
    for fam in families:
        if fam["annotation_code"] in chosen_codes:
            continue
        preferred.append(fam)
        if len(preferred) >= args.max_case_families:
            break
    return preferred


def case_family_set_payload(data: FigureData, family_set_key: str) -> dict[str, Any] | None:
    family_sets = data.case_studies.get("family_sets", {})
    payload = family_sets.get(family_set_key)
    if isinstance(payload, dict):
        return payload
    return None


def select_case_families_for_set(
    data: FigureData,
    args: argparse.Namespace,
    family_set_key: str,
    default_codes: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    payload = case_family_set_payload(data, family_set_key)
    families = list(payload.get("families", [])) if payload else []
    if not families:
        return []

    if args.case_family_codes:
        want = set(args.case_family_codes)
        return [family for family in families if family["annotation_code"] in want]
    if args.all_case_families:
        return families

    preferred_codes = list(default_codes or [])
    preferred: list[dict[str, Any]] = []
    family_map = {family["annotation_code"]: family for family in families}
    for code in preferred_codes:
        if code in family_map:
            preferred.append(family_map[code])
    if len(preferred) >= args.max_case_families:
        return preferred[:args.max_case_families]

    chosen_codes = {family["annotation_code"] for family in preferred}
    for family in families:
        if family["annotation_code"] in chosen_codes:
            continue
        preferred.append(family)
        if len(preferred) >= args.max_case_families:
            break
    return preferred


def select_cross_features(data: FigureData, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [row for row in data.cross_family_rows if row.get("is_cross_family")]
    if args.cross_feature_ids:
        wanted = set(args.cross_feature_ids)
        return [row for row in rows if row["feature_id"] in wanted]
    if args.all_cross_features:
        return sorted(rows, key=lambda r: r["composite_score"], reverse=True)

    row_map = {row["feature_id"]: row for row in rows}
    preferred = []
    for fid in DEFAULT_CROSS_FEATURE_IDS:
        if fid in row_map:
            preferred.append(row_map[fid])
    if len(preferred) >= args.max_cross_features:
        return preferred[:args.max_cross_features]

    chosen = {row["feature_id"] for row in preferred}
    remainder = sorted(rows, key=lambda r: (r["composite_score"], -(r.get("motif_rmsd_per_pos") or 99.0)), reverse=True)
    for row in remainder:
        if row["feature_id"] in chosen:
            continue
        preferred.append(row)
        if len(preferred) >= args.max_cross_features:
            break
    return preferred


def motif_metric_label(row: dict[str, Any]) -> str:
    if str(row.get("motif_metric_key", "")) == "motif_pr_auc":
        return "Seq motif PR-AUC"
    return "Seq motif F1"


def motif_pr_auc_value(value: Any) -> float:
    if isinstance(value, dict):
        return safe_float(value.get("pr_auc"), math.nan)
    return safe_float(value, math.nan)


def flatten_cath_residue_entries(residue_level: Any) -> list[dict[str, Any]]:
    if isinstance(residue_level, list):
        return [entry for entry in residue_level if isinstance(entry, dict)]
    if isinstance(residue_level, dict):
        entries: list[dict[str, Any]] = []
        for value in residue_level.values():
            if isinstance(value, list):
                entries.extend(entry for entry in value if isinstance(entry, dict))
        return entries
    return []


def best_sequence_motif_entry(data: FigureData, fid: int) -> dict[str, Any]:
    payload = data.motif_pwm_json(fid)
    motifs = payload.get("top_motifs") or payload.get("motifs") or []
    best: dict[str, Any] | None = None
    best_key = (-math.inf, -math.inf)
    for motif in motifs:
        if not isinstance(motif, dict):
            continue
        pr_auc = motif_pr_auc_value(motif.get("pr_auc"))
        best_f1 = safe_float(motif.get("best_f1"), math.nan)
        key = (
            pr_auc if np.isfinite(pr_auc) else -math.inf,
            best_f1 if np.isfinite(best_f1) else -math.inf,
        )
        if key > best_key:
            best = motif
            best_key = key
    if best is None:
        return {}
    return {
        "sequence_motif_consensus": str(best.get("consensus", "") or "").strip(),
        "sequence_motif_pr_auc": motif_pr_auc_value(best.get("pr_auc")),
        "sequence_motif_best_f1": safe_float(best.get("best_f1"), math.nan),
        "sequence_motif_id": best.get("motif_id"),
        "sequence_motif_width": best.get("width"),
        "sequence_motif_source": "probabilistic_meme_pwm",
    }


def best_interpro_residue_entry(data: FigureData, fid: int) -> dict[str, Any]:
    entries = data.interpro_json(fid).get("residue_level", [])
    if not isinstance(entries, list) or not entries:
        return {}
    best = max(entries, key=lambda entry: safe_float(entry.get("best_f1"), -math.inf))
    return best if isinstance(best, dict) else {}


def best_cath_residue_entry(data: FigureData, fid: int) -> dict[str, Any]:
    entries = flatten_cath_residue_entries(data.cath_json(fid).get("residue_level"))
    if not entries:
        return {}
    best = max(entries, key=lambda entry: safe_float(entry.get("best_f1"), -math.inf))
    return best if isinstance(best, dict) else {}


def best_biology_annotation(data: FigureData, fid: int) -> dict[str, Any]:
    interpro = best_interpro_residue_entry(data, fid)
    cath = best_cath_residue_entry(data, fid)
    interpro_score = safe_float(interpro.get("best_f1"), math.nan)
    cath_score = safe_float(cath.get("best_f1"), math.nan)

    if np.isfinite(cath_score) and cath_score > interpro_score:
        level = str(cath.get("cath_level", "") or "").strip()
        code = str(cath.get("cath_label", "") or "").strip()
        label = str(cath.get("description", "") or "").strip() or f"CATH {level}:{code}".strip()
        return {
            "biology_source": "CATH",
            "biology_score": cath_score,
            "biology_code": code,
            "biology_level": level,
            "biology_label": label,
        }

    if np.isfinite(interpro_score):
        code = str(interpro.get("annotation_code", "") or "").strip()
        label = str(interpro.get("annotation_name", "") or "").strip() or code
        return {
            "biology_source": "InterPro",
            "biology_score": interpro_score,
            "biology_code": code,
            "biology_level": "",
            "biology_label": label,
        }

    return {
        "biology_source": None,
        "biology_score": math.nan,
        "biology_code": "",
        "biology_level": "",
        "biology_label": "",
    }


def dominant_biology_score(row: dict[str, Any]) -> float:
    return max(
        safe_float(row.get("interpro_res_f1"), 0.0),
        safe_float(row.get("cath_res_f1"), 0.0),
    )


def dominance_mode_for_row(
    row: dict[str, Any],
    *,
    geom_threshold: float,
    motif_threshold: float,
    interpro_threshold: float,
    cath_threshold: float,
) -> str | None:
    geom_score = safe_float(row.get("geom_pr_auc"), 0.0)
    motif_score = safe_float(row.get("motif_f1"), 0.0)
    position_score = safe_float(row.get("position_f1"), 0.0)
    interpro_score = safe_float(row.get("interpro_res_f1"), 0.0)
    cath_score = safe_float(row.get("cath_res_f1"), 0.0)
    bio_score = max(interpro_score, cath_score)
    bio_threshold = interpro_threshold if interpro_score >= cath_score else cath_threshold

    eligible: list[tuple[str, float]] = []
    if geom_score > geom_threshold and geom_score >= max(motif_score, position_score, bio_score):
        eligible.append(("geometry", geom_score))
    if motif_score > motif_threshold and motif_score >= max(geom_score, position_score, bio_score):
        eligible.append(("sequence_motif", motif_score))
    if bio_score > bio_threshold and bio_score >= max(geom_score, motif_score, position_score):
        eligible.append(("biology", bio_score))
    if not eligible:
        return None
    return max(eligible, key=lambda item: item[1])[0]


def residue_annotatable_mode_label(mode: str | None) -> str:
    if mode == "sequence_motif":
        return "Sequence motif dominant"
    if mode == "geometry":
        return "Geometry dominant"
    if mode == "biology":
        return "Biology dominant"
    return "Residue annotatable"


def residue_annotatable_spotlight_score(row: dict[str, Any]) -> float:
    mode = str(row.get("selection_mode", "") or "")
    geom_score = safe_float(row.get("geom_pr_auc"), 0.0)
    motif_score = safe_float(row.get("motif_f1"), 0.0)
    position_score = safe_float(row.get("position_f1"), 0.0)
    biology_score = dominant_biology_score(row)
    category = row.get("structural_category", "") or ""
    motif_rmsd = row.get("motif_rmsd")
    if motif_rmsd is None or not np.isfinite(motif_rmsd):
        motif_rmsd = 2.0

    coverage_pct = row.get("coverage_pct")
    if coverage_pct is None or not np.isfinite(coverage_pct):
        coverage_pct = 15.0

    category_bonus = 0.0
    if mode == "geometry" and any(token in category for token in ["Local", "Curvature", "Torsion", "Planarity", "Tangent"]):
        category_bonus += 0.05
    if mode == "geometry" and ("Global compactness" in category or "frac_" in category):
        category_bonus -= 0.05

    dominant_signal_map = {
        "geometry": geom_score,
        "sequence_motif": motif_score,
        "biology": biology_score,
    }
    runner_up_map = {
        "geometry": max(motif_score, biology_score, position_score),
        "sequence_motif": max(geom_score, biology_score, position_score),
        "biology": max(geom_score, motif_score, position_score),
    }
    dominant_signal = dominant_signal_map.get(mode, 0.0)
    runner_up_signal = runner_up_map.get(mode, 0.0)
    dominance_margin = max(dominant_signal - runner_up_signal, 0.0)
    protein_penalty_weight = 0.25 if mode == "geometry" else 0.08

    return (
        dominant_signal
        + 0.30 * dominance_margin
        + 0.20 * float(row.get("composite_score", 0.0))
        - 0.08 * position_score
        - protein_penalty_weight * float(row.get("interpro_protein_f1", 0.0))
        - 0.04 * float(motif_rmsd)
        - 0.04 * min(float(coverage_pct) / 10.0, 2.0)
        + category_bonus
    )


def spotlight_mode_spec(mode: str) -> dict[str, Any]:
    for spec in SPOTLIGHT_MODE_SPECS:
        if spec["mode"] == mode:
            return spec
    raise KeyError(f"Unknown spotlight mode: {mode}")


def spotlight_detail_line(row: dict[str, Any]) -> str | None:
    mode = str(row.get("selection_mode", "") or "")
    if mode == "sequence_motif":
        consensus = str(row.get("sequence_motif_consensus", "") or "").strip()
        if consensus:
            return f"MEME motif: {consensus}"
        return "MEME motif label unavailable"
    if mode == "biology":
        source = str(row.get("biology_source", "") or "").strip()
        code = str(row.get("biology_code", "") or "").strip()
        label = str(row.get("biology_label", "") or "").strip()
        if source and code:
            return f"{source} residue: {code} {label}".strip()
        if source and label:
            return f"{source} residue: {label}"
        return None
    top_feature = str(row.get("top_geometric_feature", "") or "").strip()
    if top_feature:
        return f"Top geometry: {top_feature}"
    return None


def spotlight_title(row: dict[str, Any]) -> str:
    mode = str(row.get("selection_mode", "") or "")
    if mode == "biology":
        biology_label = str(row.get("biology_label", "") or "").strip()
        if biology_label:
            return textwrap.shorten(biology_label, width=78, placeholder="...")
    fallback = (
        row.get("interpro_protein_name")
        or row.get("structural_category")
        or f"{residue_annotatable_mode_label(mode)} candidate"
    )
    return textwrap.shorten(str(fallback), width=78, placeholder="...")


def spotlight_output_prefix(mode: str) -> str:
    if mode == "sequence_motif":
        return "sequence_dominant"
    if mode == "biology":
        return "biology_dominant"
    return "geometry_dominant"


def spotlight_diversity_key(row: dict[str, Any], mode: str) -> tuple[str | None, str | None]:
    if mode == "sequence_motif":
        return (
            str(row.get("interpro_protein_name") or row.get("structural_category") or ""),
            str(row.get("structural_category") or ""),
        )
    if mode == "biology":
        return (
            str(row.get("best_annotation_source") or ""),
            str(row.get("interpro_protein_name") or row.get("structural_category") or ""),
        )
    return (str(row.get("structural_category") or ""), None)


def enrich_spotlight_feature_details(data: FigureData, row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    fid = int(enriched["feature_id"])
    enriched.update(best_sequence_motif_entry(data, fid))
    enriched.update(best_biology_annotation(data, fid))
    return enriched


def spotlight_renderability(data: FigureData, fid: int) -> dict[str, Any]:
    geometry = data.geometry_json(fid)
    motif = geometry.get("geometric_residue_level", {}).get("motif_superposition", {})
    top_proteins = geometry.get("plot_data", {}).get("top_proteins", [])
    return {
        "has_top_proteins": bool(top_proteins),
        "motif_n_fragments": int(motif.get("n_fragments", 0) or 0),
        "motif_rmsd": float(motif.get("mean_rmsd", math.nan) or math.nan),
        "motif_has_template": bool(str(motif.get("mean_structure_pdb", "") or "").strip()),
    }


def spotlight_candidate_is_renderable(data: FigureData, row: dict[str, Any]) -> bool:
    renderability = spotlight_renderability(data, int(row["feature_id"]))
    return (
        renderability["has_top_proteins"]
        and renderability["motif_has_template"]
        and renderability["motif_n_fragments"] >= 20
        and float(row.get("coverage_pct", math.inf)) < 15.0
    )


def build_residue_spotlight_candidates(data: FigureData) -> list[dict[str, Any]]:
    geom_threshold = safe_float(data.geometry_primary.get("geom_pr_auc_threshold"), 0.3)
    motif_threshold = safe_float(data.geometry_primary.get("null_thresholds", {}).get(data.motif_metric_key), 1.0)
    interpro_threshold = safe_float(data.geometry_primary.get("null_thresholds", {}).get("interpro_res_f1"), 1.0)
    cath_threshold = safe_float(data.geometry_primary.get("null_thresholds", {}).get("cath_res_f1"), 1.0)
    rows = list(data.geometry_primary_rows)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["selection_mode"] = dominance_mode_for_row(
            enriched,
            geom_threshold=geom_threshold,
            motif_threshold=motif_threshold,
            interpro_threshold=interpro_threshold,
            cath_threshold=cath_threshold,
        )
        if not enriched["selection_mode"]:
            continue
        enriched["spotlight_score"] = residue_annotatable_spotlight_score(enriched)
        candidates.append(enriched)

    return [row for row in candidates if float(row.get("coverage_pct", math.inf)) < 15.0]


def select_spotlight_features(
    data: FigureData,
    args: argparse.Namespace,
    *,
    mode: str,
) -> list[dict[str, Any]]:
    spec = spotlight_mode_spec(mode)
    candidates = [row for row in build_residue_spotlight_candidates(data) if row.get("selection_mode") == mode]
    explicit_ids = getattr(args, spec["ids_arg"])
    max_features = int(getattr(args, spec["max_arg"]))
    include_all = bool(getattr(args, spec["all_arg"]))

    if explicit_ids:
        candidate_map = {
            row["feature_id"]: row
            for row in candidates
            if spotlight_candidate_is_renderable(data, row)
        }
        ordered = []
        for fid in explicit_ids:
            if fid in candidate_map:
                row = dict(candidate_map[fid])
                row.update(spotlight_renderability(data, fid))
                ordered.append(row)
        return [enrich_spotlight_feature_details(data, row) for row in ordered]
    if include_all:
        return [
            enrich_spotlight_feature_details(data, row)
            for row in (
                dict(row, **spotlight_renderability(data, row["feature_id"]))
                for row in sorted(candidates, key=lambda row: row["spotlight_score"], reverse=True)
                if spotlight_candidate_is_renderable(data, row)
            )
        ]
    if max_features <= 0:
        return []

    remainder = sorted(candidates, key=lambda row: row["spotlight_score"], reverse=True)
    preferred: list[dict[str, Any]] = []
    chosen_ids: set[int] = set()
    for fid in spec["default_ids"]:
        for row in remainder:
            if row["feature_id"] != fid or fid in chosen_ids:
                continue
            preferred.append(row)
            chosen_ids.add(fid)
            break
    if len(preferred) >= max_features:
        return preferred[:max_features]

    seen_diversity_keys = {spotlight_diversity_key(row, mode) for row in preferred}
    for row in remainder:
        fid = row["feature_id"]
        if fid in chosen_ids:
            continue
        if not spotlight_candidate_is_renderable(data, row):
            continue
        diversity_key = spotlight_diversity_key(row, mode)
        if diversity_key in seen_diversity_keys and len(preferred) < max_features:
            continue
        enriched_row = dict(row)
        enriched_row.update(spotlight_renderability(data, fid))
        preferred.append(enriched_row)
        chosen_ids.add(fid)
        seen_diversity_keys.add(diversity_key)
        if len(preferred) >= max_features:
            break

    if len(preferred) < max_features:
        for row in remainder:
            fid = row["feature_id"]
            if fid in chosen_ids:
                continue
            if not spotlight_candidate_is_renderable(data, row):
                continue
            enriched_row = dict(row)
            enriched_row.update(spotlight_renderability(data, fid))
            preferred.append(enriched_row)
            chosen_ids.add(fid)
            if len(preferred) >= max_features:
                break
    return [enrich_spotlight_feature_details(data, row) for row in preferred]


def select_geometry_dominant_features(data: FigureData, args: argparse.Namespace) -> list[dict[str, Any]]:
    return select_spotlight_features(data, args, mode="geometry")


def plot_geometry_primary_selection(data: FigureData, out_dir: Path) -> str:
    rows = data.geometry_primary_rows
    background = [row for row in rows if not row["is_geometry_primary"]]
    primary = [row for row in rows if row["is_geometry_primary"]]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(
        [row["geom_pr_auc"] for row in background],
        [row["best_seq_f1"] for row in background],
        s=12,
        color=PALETTE["muted_fill"],
        alpha=0.28,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        [row["geom_pr_auc"] for row in primary],
        [row["best_seq_f1"] for row in primary],
        s=28,
        color=PALETTE["secondary"],
        edgecolors=PALETTE["secondary_dark"],
        linewidths=0.3,
        alpha=0.92,
        rasterized=True,
    )
    ax.axvline(data.geometry_primary.get("geom_pr_auc_threshold", 0.3), color=PALETTE["muted"], linestyle="--", linewidth=1.0)
    ax.set_xlabel("Geometry PR-AUC")
    ax.set_ylabel("Best sequence-level F1")
    ax.set_title("Geometry-primary selection", fontsize=12, pad=8)
    polish_axes(ax)
    sns.despine(ax=ax)
    return save_pdf(fig, out_dir / "01_geometry_primary_selection_scatter.pdf")


def plot_structural_category_counts(data: FigureData, out_dir: Path) -> str:
    all_counts = data.cross_family.get("structural_categories_all", {})
    cross_counts = data.cross_family.get("structural_categories_cross_family", {})
    categories = list(all_counts.keys())
    categories.sort(key=lambda c: all_counts[c], reverse=True)

    y = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(7.0, max(4.5, 0.35 * len(categories) + 1.0)))
    ax.barh(y - 0.18, [all_counts[c] for c in categories], height=0.34, color=PALETTE["primary"], label="All geometry-primary")
    ax.barh(y + 0.18, [cross_counts.get(c, 0) for c in categories], height=0.34, color=PALETTE["secondary"], label="Cross-family subset")
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(c, 28) for c in categories], fontsize=9)
    ax.set_xlabel("Number of features")
    ax.set_title("Structural categories represented by geometry-primary features", fontsize=12, pad=8)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    polish_axes(ax, grid_axis="x")
    sns.despine(ax=ax, left=False, bottom=False)
    return save_pdf(fig, out_dir / "02_structural_category_counts.pdf")


def plot_cross_family_zone_scatter(data: FigureData, out_dir: Path) -> str:
    rows = data.cross_family_rows
    cross = [row for row in rows if row["is_cross_family"]]
    other = [row for row in rows if not row["is_cross_family"]]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.axvspan(0.3, 0.7, color=PALETTE["secondary_light"], alpha=0.24, zorder=0)
    ax.scatter(
        [row["best_interpro_protein_f1"] for row in other],
        [row["composite_score"] for row in other],
        s=18,
        color=PALETTE["primary_light"],
        alpha=0.34,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        [row["best_interpro_protein_f1"] for row in cross],
        [row["composite_score"] for row in cross],
        s=34,
        color=PALETTE["accent"],
        edgecolors="white",
        linewidths=0.4,
        alpha=0.9,
        rasterized=True,
    )
    ax.axvline(0.3, color=PALETTE["muted"], linestyle="--", linewidth=0.9)
    ax.axvline(0.7, color=PALETTE["muted"], linestyle="--", linewidth=0.9)
    ax.set_xlabel("Best InterPro protein F1")
    ax.set_ylabel("Geometry composite score")
    ax.set_title("Cross-family features occupy an intermediate family-match regime", fontsize=12, pad=8)
    polish_axes(ax)
    sns.despine(ax=ax)
    return save_pdf(fig, out_dir / "03_cross_family_zone_scatter.pdf")


def plot_residue_f1_distribution(data: FigureData, out_dir: Path) -> str:
    rows = data.cross_family_rows
    vals = [row["best_interpro_residue_f1"] for row in rows]
    null_threshold = data.cross_family.get("global_stats", {}).get("null_thresholds", {}).get("interpro_res_f1", 0.2)

    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    sns.histplot(vals, bins=28, color=PALETTE["primary"], edgecolor="white", linewidth=0.4, ax=ax)
    ax.axvline(null_threshold, color=PALETTE["accent"], linestyle="--", linewidth=1.2)
    ax.text(null_threshold + 0.005, ax.get_ylim()[1] * 0.94, f"null p95 = {null_threshold:.3f}", color=PALETTE["accent_dark"], fontsize=9)
    ax.set_xlabel("Best InterPro residue-level F1")
    ax.set_ylabel("Count")
    ax.set_title("Residue-level family annotations mostly fail for geometry-primary features", fontsize=12, pad=8)
    polish_axes(ax, grid_axis="y")
    sns.despine(ax=ax)
    return save_pdf(fig, out_dir / "04_residue_interpro_f1_distribution.pdf")


def plot_motif_rmsd_vs_interpro(data: FigureData, out_dir: Path) -> str:
    rows = [row for row in data.cross_family_rows if row.get("motif_rmsd_per_pos") is not None]
    cross = [row for row in rows if row["is_cross_family"]]
    other = [row for row in rows if not row["is_cross_family"]]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(
        [row["best_interpro_protein_f1"] for row in other],
        [row["motif_rmsd_per_pos"] for row in other],
        s=18,
        color=PALETTE["muted"],
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        [row["best_interpro_protein_f1"] for row in cross],
        [row["motif_rmsd_per_pos"] for row in cross],
        s=36,
        color=PALETTE["secondary"],
        edgecolors="white",
        linewidths=0.4,
        alpha=0.92,
        rasterized=True,
    )
    ax.set_xlabel("Best InterPro protein F1")
    ax.set_ylabel("Motif RMSD per position (A)")
    ax.set_title("Tighter motifs do not automatically imply weaker family association", fontsize=12, pad=8)
    polish_axes(ax)
    sns.despine(ax=ax)
    return save_pdf(fig, out_dir / "05_motif_rmsd_vs_interpro_protein_f1.pdf")


def plot_concordance_vs_rmsd(data: FigureData, out_dir: Path) -> str:
    rows = [row for row in data.cross_family_rows if row.get("motif_rmsd_per_pos") is not None]
    cross = [row for row in rows if row["is_cross_family"]]
    other = [row for row in rows if not row["is_cross_family"]]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(
        [row["motif_rmsd_per_pos"] for row in other],
        [row["concordance_prauc"] for row in other],
        s=18,
        color=PALETTE["primary_light"],
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        [row["motif_rmsd_per_pos"] for row in cross],
        [row["concordance_prauc"] for row in cross],
        s=34,
        color=PALETTE["accent"],
        edgecolors="white",
        linewidths=0.4,
        alpha=0.92,
        rasterized=True,
    )
    ax.set_xlabel("Motif RMSD per position (A)")
    ax.set_ylabel("Concordance PR-AUC")
    ax.set_title("Tighter structural motifs tend to yield higher geometry concordance", fontsize=12, pad=8)
    polish_axes(ax)
    sns.despine(ax=ax)
    return save_pdf(fig, out_dir / "06_concordance_vs_motif_rmsd.pdf")


def plot_rmsd_variation_distribution(data: FigureData, out_dir: Path) -> str:
    rows = [
        row for row in data.cross_family_rows
        if row.get("motif_rmsd_per_pos") is not None
        and math.isfinite(float(row["motif_rmsd_per_pos"]))
    ]
    vals = np.array([float(row["motif_rmsd_per_pos"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    if vals.size == 0:
        ax.text(
            0.5,
            0.5,
            "No RMSD data available for geometry-primary features.",
            ha="center",
            va="center",
            color=PALETTE["muted"],
            fontsize=10.0,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return save_pdf(fig, out_dir / "13_rmsd_variation_distribution.pdf")

    upper = max(0.28, math.ceil(float(vals.max()) / 0.02) * 0.02)
    bins = np.arange(0.0, upper + 0.0201, 0.02)

    ax.axvspan(0.0, 0.10, color=PALETTE["success"], alpha=0.08, linewidth=0)
    ax.axvspan(0.10, 0.20, color=PALETTE["secondary"], alpha=0.08, linewidth=0)
    ax.axvspan(0.20, upper, color=PALETTE["accent"], alpha=0.05, linewidth=0)

    sns.histplot(
        vals,
        bins=bins,
        color=PALETTE["primary"],
        edgecolor="white",
        linewidth=0.45,
        ax=ax,
    )

    median = float(np.median(vals))
    ax.axvline(median, color=PALETTE["accent_dark"], linestyle="--", linewidth=1.2)
    ax.text(
        median + 0.004,
        ax.get_ylim()[1] * 0.94,
        f"median = {median:.3f}",
        color=PALETTE["accent_dark"],
        fontsize=9.0,
        ha="left",
        va="top",
    )

    low = int(np.sum(vals < 0.10))
    mid = int(np.sum((vals >= 0.10) & (vals < 0.20)))
    high = int(np.sum(vals >= 0.20))
    total = int(vals.size)
    summary = "\n".join([
        f"< 0.10 A/pos: {low}/{total} ({100 * low / total:.0f}%)",
        f"0.10-0.20 A/pos: {mid}/{total} ({100 * mid / total:.0f}%)",
        f">= 0.20 A/pos: {high}/{total} ({100 * high / total:.0f}%)",
    ])
    ax.text(
        0.98,
        0.97,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        color=PALETTE["ink"],
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": PALETTE["grid"],
            "linewidth": 0.8,
            "alpha": 0.96,
        },
    )

    ax.set_xlim(0.0, upper)
    ax.set_xlabel("Motif RMSD per position (A)")
    ax.set_ylabel("Count")
    ax.set_title("Low RMSD is common, but not required, among geometry-primary features", fontsize=12, pad=8)
    polish_axes(ax, grid_axis="y")
    sns.despine(ax=ax)
    return save_pdf(fig, out_dir / "13_rmsd_variation_distribution.pdf")


def plot_coverage_vs_score(data: FigureData, out_dir: Path) -> str:
    rows = [row for row in data.geometry_primary_rows if row["is_geometry_primary"]]
    cross_ids = {row["feature_id"] for row in data.cross_family_rows if row["is_cross_family"]}

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(
        [row["coverage_pct"] for row in rows if row["feature_id"] not in cross_ids],
        [row["composite_score"] for row in rows if row["feature_id"] not in cross_ids],
        s=20,
        color=PALETTE["primary_light"],
        alpha=0.5,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        [row["coverage_pct"] for row in rows if row["feature_id"] in cross_ids],
        [row["composite_score"] for row in rows if row["feature_id"] in cross_ids],
        s=36,
        color=PALETTE["secondary"],
        edgecolors="white",
        linewidths=0.4,
        alpha=0.92,
        rasterized=True,
    )
    ax.axvline(20.0, color=PALETTE["muted"], linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Percent of proteins activated")
    ax.set_ylabel("Geometry composite score")
    ax.set_title("Most geometry-primary features are sparse, but dense outliers remain", fontsize=12, pad=8)
    polish_axes(ax)
    sns.despine(ax=ax)
    return save_pdf(fig, out_dir / "07_coverage_vs_geometry_score.pdf")


def q_axis_floor(values: list[float]) -> float:
    valid = [float(value) for value in values if is_valid_q(value)]
    if not valid:
        return 1e-3
    min_q = min(valid)
    return max(1e-300, 10 ** math.floor(math.log10(min_q)))


def plot_q_comparison(
    data: FigureData,
    out_dir: Path,
    *,
    x_key: str,
    x_label: str,
    title: str,
    output_name: str,
) -> str:
    rows = [
        row
        for row in data.geometry_primary_rows
        if is_valid_q(row.get(x_key))
        and is_valid_q(row.get("best_non_motif_annotation_q"))
    ]

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    if not rows:
        ax.text(
            0.5,
            0.5,
            "Adjusted q-values are not available for these artifacts.",
            ha="center",
            va="center",
            color=PALETTE["muted"],
            fontsize=10.0,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return save_pdf(fig, out_dir / output_name)

    source_order = ["Position", "InterPro residue", "CATH residue"]
    source_palette = {
        "Position": PALETTE["primary"],
        "InterPro residue": PALETTE["success"],
        "CATH residue": PALETTE["accent"],
    }
    fallback_color = PALETTE["muted"]

    lower = q_axis_floor(
        [row[x_key] for row in rows]
        + [row["best_non_motif_annotation_q"] for row in rows]
    )
    upper = 1.05

    for source in source_order:
        subset = [
            row
            for row in rows
            if row.get("best_non_motif_annotation_q_source") == source
        ]
        if not subset:
            continue
        ax.scatter(
            [row[x_key] for row in subset],
            [row["best_non_motif_annotation_q"] for row in subset],
            s=16,
            alpha=0.58,
            linewidths=0,
            color=source_palette[source],
            label=source,
            rasterized=True,
        )

    other = [
        row
        for row in rows
        if row.get("best_non_motif_annotation_q_source") not in set(source_order)
    ]
    if other:
        ax.scatter(
            [row[x_key] for row in other],
            [row["best_non_motif_annotation_q"] for row in other],
            s=16,
            alpha=0.50,
            linewidths=0,
            color=fallback_color,
            label="Other",
            rasterized=True,
        )

    ax.plot(
        [lower, 1.0],
        [lower, 1.0],
        color=PALETTE["slate"],
        linestyle="--",
        linewidth=0.95,
        alpha=0.82,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Best non-motif, non-geometry annotation q")
    ax.set_title(title, fontsize=12, pad=8)
    ax.text(
        0.02,
        0.98,
        f"n = {len(rows)}\nlower q = stronger evidence",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        color=PALETTE["muted"],
    )
    ax.legend(loc="lower right", title="Best annotation q from")
    polish_axes(ax)
    sns.despine(ax=ax)
    return save_pdf(fig, out_dir / output_name)


def plot_geometry_q_vs_best_annotation_q(data: FigureData, out_dir: Path) -> str:
    return plot_q_comparison(
        data,
        out_dir,
        x_key="geometry_q",
        x_label="Geometry PR-AUC adjusted q",
        title="Geometry significance vs best non-motif annotation significance",
        output_name="09_geometry_q_vs_best_annotation_q.pdf",
    )


def plot_seq_motif_q_vs_best_annotation_q(data: FigureData, out_dir: Path) -> str:
    return plot_q_comparison(
        data,
        out_dir,
        x_key="seq_motif_q",
        x_label="Sequence motif adjusted q",
        title="Sequence motif significance vs best non-motif annotation significance",
        output_name="10_seq_motif_q_vs_best_annotation_q.pdf",
    )


def select_case_summary_families(
    families: list[dict[str, Any]],
    max_families: int,
) -> tuple[list[dict[str, Any]], bool]:
    if max_families <= 0 or len(families) <= max_families:
        return families, False

    default_codes = set(DEFAULT_CASE_FAMILY_CODES)
    ranked = sorted(
        families,
        key=lambda fam: (
            int(fam.get("n_nodes", 0)),
            int(fam.get("n_unique_top_geom", 0)),
            bool(fam.get("geom_diverse", False)),
            -safe_float(fam.get("mean_cosine_similarity"), 0.0),
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    for family in ranked:
        code = str(family.get("annotation_code", ""))
        if code in default_codes:
            selected.append(family)
            seen_codes.add(code)

    for family in ranked:
        if len(selected) >= max_families:
            break
        code = str(family.get("annotation_code", ""))
        if code in seen_codes:
            continue
        selected.append(family)
        seen_codes.add(code)

    selected.sort(
        key=lambda fam: (
            int(fam.get("n_nodes", 0)),
            int(fam.get("n_unique_top_geom", 0)),
            -safe_float(fam.get("mean_cosine_similarity"), 0.0),
        ),
        reverse=True,
    )
    return selected, True


def plot_case_family_summary(
    data: FigureData,
    out_dir: Path,
    max_families: int = DEFAULT_MAX_CASE_SUMMARY_FAMILIES,
) -> str:
    all_families = data.case_studies.get("families", [])
    families, truncated = select_case_summary_families(all_families, max_families)
    y_labels = [wrap(fam["annotation_name"], 26) for fam in families]
    y = np.arange(len(families))
    sizes = [90 + 45 * fam["n_nodes"] for fam in families]
    colors = [fam["n_unique_top_geom"] for fam in families]

    fig, ax = plt.subplots(figsize=(7.2, max(4.2, 0.55 * len(families) + 1.0)))
    scatter = ax.scatter(
        [fam["mean_cosine_similarity"] for fam in families],
        y,
        s=sizes,
        c=colors,
        cmap="flare",
        edgecolors=PALETTE["slate"],
        linewidths=0.4,
    )
    for yi, fam in zip(y, families):
        ax.text(
            fam["mean_cosine_similarity"] + 0.012,
            yi,
            f"{fam['n_nodes']} nodes",
            va="center",
            fontsize=8.5,
            color=PALETTE["slate"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Mean pairwise cosine similarity of geometry importances")
    title = "Case-study families show multiple geometry modes within one annotation"
    if truncated:
        title = f"{title} (top {len(families)} of {len(all_families)})"
    ax.set_title(title, fontsize=12, pad=8)
    ax.invert_yaxis()
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Number of unique top geometry features")
    polish_axes(ax, grid_axis="x")
    sns.despine(ax=ax, left=False, bottom=False)
    return save_pdf(fig, out_dir / "08_case_family_summary.pdf")


def plot_geom_prauc_vs_best_annotation_density(data: FigureData, out_dir: Path) -> str:
    rows = data.geometry_primary_rows
    x = np.array([row["geom_pr_auc"] for row in rows], dtype=float)
    y = np.array([row["best_annotation_f1"] for row in rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    density = np.ones_like(x)
    if x.size > 1:
        try:
            density = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
        except Exception:
            density = np.ones_like(x)

    order = np.argsort(density)
    x = x[order]
    y = y[order]
    density = density[order]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    scatter = ax.scatter(
        x,
        y,
        c=density,
        cmap="viridis",
        s=16,
        alpha=0.82,
        linewidths=0,
        rasterized=True,
    )
    ax.axvline(
        data.geometry_primary.get("geom_pr_auc_threshold", 0.3),
        color=PALETTE["muted"],
        linestyle="--",
        linewidth=1.0,
    )
    ax.set_xlabel("Geometry PR-AUC")
    ax.set_ylabel(data.annotation_score_label)
    ax.set_title(f"Geometry PR-AUC vs {data.annotation_score_label.lower()}", fontsize=12, pad=8)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("KDE density")
    polish_axes(ax)
    sns.despine(ax=ax)
    return save_pdf(fig, out_dir / "09_geom_prauc_vs_best_annotation_f1_density.pdf")


def plot_geom_prauc_vs_best_annotation_source(data: FigureData, out_dir: Path) -> str:
    rows = data.geometry_primary_rows
    fig, ax = plt.subplots(figsize=(6.6, 4.9))

    source_order = ["Sequence Motif", "Position", "InterPro Residue", "CATH Residue", "Tie"]
    source_palette = {
        "Sequence Motif": PALETTE["secondary_dark"],
        "Position": PALETTE["primary"],
        "InterPro Residue": PALETTE["success"],
        "CATH Residue": PALETTE["accent"],
        "Tie": PALETTE["muted"],
    }
    source_label_map = {
        "Sequence Motif": "Sequence motif",
        "Position": "Position",
        "InterPro Residue": "InterPro residue",
        "CATH Residue": "CATH residue",
        "Tie": "Tie",
    }

    for source in source_order:
        subset = [row for row in rows if row["best_annotation_source"] == source]
        if not subset:
            continue
        ax.scatter(
            [row["geom_pr_auc"] for row in subset],
            [row["best_annotation_f1"] for row in subset],
            s=16,
            alpha=0.64,
            color=source_palette[source],
            label=source_label_map[source],
            linewidths=0,
            rasterized=True,
        )

    ax.axvline(
        data.geometry_primary.get("geom_pr_auc_threshold", 0.3),
        color=PALETTE["muted"],
        linestyle="--",
        linewidth=1.0,
    )
    ax.set_xlabel("Geometry PR-AUC")
    ax.set_ylabel(data.annotation_score_label)
    ax.set_title(f"Geometry PR-AUC vs source of {data.annotation_score_label.lower()}", fontsize=12, pad=8)
    ax.legend(loc="upper right", title="Best raw score from")
    polish_axes(ax)
    sns.despine(ax=ax)
    return save_pdf(fig, out_dir / "10_geom_prauc_vs_best_annotation_f1_source.pdf")


def plot_geom_f1_vs_best_annotation_density(data: FigureData, out_dir: Path) -> str:
    rows = data.geometry_primary_rows
    x = np.array([row["concordance_f1"] for row in rows], dtype=float)
    y = np.array([row["best_annotation_f1"] for row in rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    density = np.ones_like(x)
    if x.size > 1:
        try:
            density = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
        except Exception:
            density = np.ones_like(x)

    order = np.argsort(density)
    x = x[order]
    y = y[order]
    density = density[order]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    scatter = ax.scatter(
        x,
        y,
        c=density,
        cmap="viridis",
        s=16,
        alpha=0.82,
        linewidths=0,
        rasterized=True,
    )
    ax.axvline(0.35, color=PALETTE["muted"], linestyle="--", linewidth=1.0)
    ax.text(
        0.355,
        0.98,
        "0.35",
        color=PALETTE["muted"],
        fontsize=8.8,
        ha="left",
        va="top",
        transform=ax.get_xaxis_transform(),
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Geometry concordance F1")
    ax.set_ylabel(data.annotation_score_label)
    ax.set_title(f"Geometry F1 vs {data.annotation_score_label.lower()}", fontsize=12, pad=8)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("KDE density")
    polish_axes(ax)
    sns.despine(ax=ax)
    return save_pdf(fig, out_dir / "11_geom_f1_vs_best_annotation_f1_density.pdf")


def plot_geom_f1_vs_best_annotation_source(data: FigureData, out_dir: Path) -> str:
    rows = data.geometry_primary_rows
    fig, ax = plt.subplots(figsize=(6.6, 4.9))

    source_order = ["Sequence Motif", "Position", "InterPro Residue", "CATH Residue", "Tie"]
    source_palette = {
        "Sequence Motif": PALETTE["secondary_dark"],
        "Position": PALETTE["primary"],
        "InterPro Residue": PALETTE["success"],
        "CATH Residue": PALETTE["accent"],
        "Tie": PALETTE["muted"],
    }
    source_label_map = {
        "Sequence Motif": "Sequence motif",
        "Position": "Position",
        "InterPro Residue": "InterPro residue",
        "CATH Residue": "CATH residue",
        "Tie": "Tie",
    }

    for source in source_order:
        subset = [row for row in rows if row["best_annotation_source"] == source]
        if not subset:
            continue
        ax.scatter(
            [row["concordance_f1"] for row in subset],
            [row["best_annotation_f1"] for row in subset],
            s=16,
            alpha=0.64,
            color=source_palette[source],
            label=source_label_map[source],
            linewidths=0,
            rasterized=True,
        )

    ax.axvline(0.35, color=PALETTE["muted"], linestyle="--", linewidth=1.0)
    ax.text(
        0.355,
        0.98,
        "0.35",
        color=PALETTE["muted"],
        fontsize=8.8,
        ha="left",
        va="top",
        transform=ax.get_xaxis_transform(),
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Geometry concordance F1")
    ax.set_ylabel(data.annotation_score_label)
    ax.set_title(f"Geometry F1 vs source of {data.annotation_score_label.lower()}", fontsize=12, pad=8)
    ax.legend(loc="upper right", title="Best raw score from")
    polish_axes(ax)
    sns.despine(ax=ax)
    return save_pdf(fig, out_dir / "12_geom_f1_vs_best_annotation_f1_source.pdf")


def make_overview_plots(
    data: FigureData,
    output_dir: Path,
    max_case_summary_families: int = DEFAULT_MAX_CASE_SUMMARY_FAMILIES,
) -> list[str]:
    out_dir = output_dir / OVERVIEW_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        plot_geometry_primary_selection(data, out_dir),
        plot_structural_category_counts(data, out_dir),
        plot_cross_family_zone_scatter(data, out_dir),
        plot_residue_f1_distribution(data, out_dir),
        plot_motif_rmsd_vs_interpro(data, out_dir),
        plot_concordance_vs_rmsd(data, out_dir),
        plot_coverage_vs_score(data, out_dir),
        plot_case_family_summary(data, out_dir, max_families=max_case_summary_families),
    ]
    if data.has_q_values:
        plots.extend([
            plot_geometry_q_vs_best_annotation_q(data, out_dir),
            plot_seq_motif_q_vs_best_annotation_q(data, out_dir),
        ])
    else:
        plots.extend([
            plot_geom_prauc_vs_best_annotation_density(data, out_dir),
            plot_geom_prauc_vs_best_annotation_source(data, out_dir),
        ])
    plots.extend([
        plot_geom_f1_vs_best_annotation_density(data, out_dir),
        plot_geom_f1_vs_best_annotation_source(data, out_dir),
        plot_rmsd_variation_distribution(data, out_dir),
    ])
    return plots


def activation_residue_jaccard(a: list[int], b: list[int]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 0.0
    return float(len(set_a & set_b) / max(1, len(set_a | set_b)))


def activation_span_overlap(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    start_a, end_a = min(a), max(a)
    start_b, end_b = min(b), max(b)
    overlap = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
    union = max(end_a, end_b) - min(start_a, start_b) + 1
    return float(overlap / max(1, union))


def activation_trace_redundancy(entry_a: dict[str, Any], entry_b: dict[str, Any]) -> float:
    activation_a = entry_a.get("per_residue_activations", []) or []
    activation_b = entry_b.get("per_residue_activations", []) or []
    highlighted_a = highlighted_residues(activation_a)
    highlighted_b = highlighted_residues(activation_b)
    hot_a = top_hot_residues(activation_a)
    hot_b = top_hot_residues(activation_b)
    highlighted_jaccard = activation_residue_jaccard(highlighted_a, highlighted_b)
    hot_jaccard = activation_residue_jaccard(hot_a, hot_b)
    span_overlap = activation_span_overlap(highlighted_a, highlighted_b)
    return max(
        highlighted_jaccard,
        hot_jaccard,
        CASE_STUDY_ACTIVATION_SPAN_REDUNDANCY_WEIGHT * span_overlap,
    )


def best_shared_accession_for_feature_ids(
    data: FigureData,
    feature_ids: list[int],
) -> tuple[str | None, dict[int, dict[str, Any]]]:
    if not feature_ids:
        return None, {}

    accession_to_entries: dict[str, dict[int, dict[str, Any]]] = {}
    for fid in feature_ids:
        for accession, entry in data.feature_sequence_entries(fid).items():
            accession_to_entries.setdefault(accession, {})[fid] = entry

    best_accession = None
    best_shared_entries: dict[int, dict[str, Any]] = {}
    best_key = None
    for accession, shared_entries in accession_to_entries.items():
        shared_count = len(shared_entries)
        total_activation = sum(
            safe_float(entry.get("max_activation"), 0.0)
            for entry in shared_entries.values()
        )
        key = (shared_count, total_activation)
        if best_key is None or key > best_key:
            best_key = key
            best_accession = accession
            best_shared_entries = shared_entries

    return best_accession, best_shared_entries


def candidate_activation_diversity_on_accession(
    data: FigureData,
    selected_feature_ids: list[int],
    candidate_feature_id: int,
    accession: str,
) -> tuple[float, float]:
    candidate_entry = data.feature_sequence_entries(candidate_feature_id).get(accession)
    if candidate_entry is None:
        return 0.0, 0.0

    overlaps = [
        activation_trace_redundancy(candidate_entry, data.feature_sequence_entries(fid)[accession])
        for fid in selected_feature_ids
        if accession in data.feature_sequence_entries(fid)
    ]
    if not overlaps:
        return 1.0, 1.0

    max_redundancy = max(overlaps)
    mean_redundancy = float(sum(overlaps) / len(overlaps))
    return 1.0 - max_redundancy, 1.0 - mean_redundancy


def select_case_study_members(
    data: FigureData,
    family: dict[str, Any],
    max_members: int = CASE_STUDY_MEMBER_LIMIT,
) -> list[dict[str, Any]]:
    members = list(family.get("members") or [])
    if len(members) <= max_members:
        return members

    return select_diverse_case_study_members(data, family, max_members=max_members)


def case_study_member_similarity_matrix(members: list[dict[str, Any]]) -> np.ndarray:
    feature_names = sorted({
        feature_name
        for member in members
        for feature_name in (member.get("feature_importances") or {}).keys()
    })
    if not feature_names:
        return np.eye(len(members), dtype=float)

    matrix = np.zeros((len(members), len(feature_names)), dtype=float)
    for row_idx, member in enumerate(members):
        importances = member.get("feature_importances") or {}
        for col_idx, feature_name in enumerate(feature_names):
            matrix[row_idx, col_idx] = safe_float(importances.get(feature_name), 0.0)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = matrix / norms
    return normed @ normed.T


def select_diverse_case_study_members(
    data: FigureData,
    family: dict[str, Any],
    max_members: int = CASE_STUDY_MEMBER_LIMIT,
) -> list[dict[str, Any]]:
    members = list(family.get("members") or [])
    if len(members) <= max_members:
        return members

    similarity = case_study_member_similarity_matrix(members)
    special_case = CASE_STUDY_MEMBER_SPECIAL_CASES.get(str(family.get("annotation_code", ""))) or {}

    selected_indices: list[int] = []
    selected_ids: set[int] = set()
    protected_ids: set[int] = set()

    keep_top = max(1, min(max_members, int(special_case.get("keep_top", 1) or 1)))
    for idx, member in enumerate(members[:keep_top]):
        selected_indices.append(idx)
        fid = int(member["feature_id"])
        selected_ids.add(fid)
        protected_ids.add(fid)

    for fid in special_case.get("include_feature_ids", []):
        if len(selected_indices) >= max_members or fid in selected_ids:
            continue
        match_idx = next((idx for idx, member in enumerate(members) if int(member["feature_id"]) == fid), None)
        if match_idx is not None:
            selected_indices.append(match_idx)
            selected_ids.add(fid)
            protected_ids.add(fid)

    while len(selected_indices) < max_members:
        best_idx = None
        best_key = None
        for idx, member in enumerate(members):
            fid = int(member["feature_id"])
            if fid in selected_ids:
                continue

            if selected_indices:
                distances = [1.0 - float(similarity[idx, chosen_idx]) for chosen_idx in selected_indices]
                min_distance = min(distances)
                mean_distance = float(sum(distances) / len(distances))
            else:
                min_distance = 1.0
                mean_distance = 1.0

            key = (
                min_distance,
                mean_distance,
                safe_float(member.get("geom_pr_auc"), 0.0),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx

        if best_idx is None:
            break
        selected_indices.append(best_idx)
        selected_ids.add(int(members[best_idx]["feature_id"]))

    selected_members = [members[idx] for idx in selected_indices[:max_members]]
    return refine_case_study_members_by_activation_diversity(
        data,
        members,
        selected_members,
        protected_ids=protected_ids,
    )


def refine_case_study_members_by_activation_diversity(
    data: FigureData,
    all_members: list[dict[str, Any]],
    selected_members: list[dict[str, Any]],
    protected_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    protected = set(protected_ids or set())
    feature_names = sorted({
        feature_name
        for member in all_members
        for feature_name in (member.get("feature_importances") or {}).keys()
    })
    if not feature_names:
        return selected_members

    member_vectors: dict[int, np.ndarray] = {}
    for member in all_members:
        fid = int(member["feature_id"])
        vec = np.array([
            safe_float((member.get("feature_importances") or {}).get(feature_name), 0.0)
            for feature_name in feature_names
        ], dtype=float)
        vec /= max(np.linalg.norm(vec), 1e-12)
        member_vectors[fid] = vec

    selected = list(selected_members)
    for _ in range(len(selected)):
        selected_feature_ids = [int(member["feature_id"]) for member in selected]
        accession, shared_entries = best_shared_accession_for_feature_ids(data, selected_feature_ids)
        if accession is None:
            break

        worst_pair: tuple[float, int, int] | None = None
        for i, member_a in enumerate(selected):
            fid_a = int(member_a["feature_id"])
            entry_a = shared_entries.get(fid_a)
            if entry_a is None:
                continue
            for member_b in selected[i + 1:]:
                fid_b = int(member_b["feature_id"])
                entry_b = shared_entries.get(fid_b)
                if entry_b is None:
                    continue
                redundancy = activation_trace_redundancy(entry_a, entry_b)
                if worst_pair is None or redundancy > worst_pair[0]:
                    worst_pair = (redundancy, fid_a, fid_b)

        if worst_pair is None or worst_pair[0] <= CASE_STUDY_ACTIVATION_REDUNDANCY_THRESHOLD:
            break

        _, fid_a, fid_b = worst_pair
        removable_fid = None
        for candidate_fid in sorted(
            [fid_a, fid_b],
            key=lambda fid: (
                fid in protected,
                safe_float(next(
                    member.get("geom_pr_auc", 0.0)
                    for member in selected
                    if int(member["feature_id"]) == fid
                ), 0.0),
            ),
        ):
            if candidate_fid not in protected:
                removable_fid = candidate_fid
                break
        if removable_fid is None:
            break

        removable_idx = next(
            idx for idx, member in enumerate(selected)
            if int(member["feature_id"]) == removable_fid
        )
        base_selected = [member for member in selected if int(member["feature_id"]) != removable_fid]
        base_ids = [int(member["feature_id"]) for member in base_selected]
        best_replacement = None
        best_key = None
        selected_ids = {int(member["feature_id"]) for member in selected}
        for member in all_members:
            fid = int(member["feature_id"])
            if fid in selected_ids or accession not in data.feature_sequence_entries(fid):
                continue

            activation_distance, mean_activation_distance = candidate_activation_diversity_on_accession(
                data,
                base_ids,
                fid,
                accession,
            )
            distances = [
                1.0 - float(np.dot(member_vectors[fid], member_vectors[base_fid]))
                for base_fid in base_ids
            ]
            min_geom_distance = min(distances) if distances else 1.0
            mean_geom_distance = float(sum(distances) / len(distances)) if distances else 1.0
            if (1.0 - activation_distance) > CASE_STUDY_ACTIVATION_REDUNDANCY_THRESHOLD:
                continue
            key = (
                min_geom_distance,
                mean_geom_distance,
                safe_float(member.get("geom_pr_auc"), 0.0),
                activation_distance,
                mean_activation_distance,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_replacement = member

        if best_replacement is None:
            break

        selected[removable_idx] = best_replacement

    return selected


def prepare_case_study_family_for_plotting(
    data: FigureData,
    family: dict[str, Any],
    max_members: int = CASE_STUDY_MEMBER_LIMIT,
) -> dict[str, Any]:
    prepared = dict(family)
    prepared["members"] = select_case_study_members(data, family, max_members=max_members)
    return prepared


def is_sequence_motif_case_family(family: dict[str, Any]) -> bool:
    return str(family.get("member_label_mode", "")) == "sequence_motif"


def case_member_descriptor(family: dict[str, Any], member: dict[str, Any]) -> str:
    if is_sequence_motif_case_family(family):
        consensus = str(member.get("sequence_motif_consensus", "") or "").strip()
        if consensus:
            return consensus
        return "sequence motif"
    return str(member.get("top_geometric_feature", "geometry") or "geometry")


def case_member_secondary_motif_label(family: dict[str, Any], member: dict[str, Any]) -> str | None:
    if is_sequence_motif_case_family(family):
        return None
    if not bool(member.get("sequence_motif_is_strong")):
        return None
    consensus = str(member.get("sequence_motif_consensus", "") or "").strip()
    if not consensus:
        return None
    return consensus


def render_sequence_motif_markup(label: str, letter_probs: list[float] | None) -> str:
    probabilities = list(letter_probs or [])
    spans: list[str] = []
    for idx, residue in enumerate(label):
        prob = probabilities[idx] if idx < len(probabilities) else math.nan
        if math.isfinite(prob):
            font_size = 12.0 + 18.0 * max(0.0, min(1.0, prob))
            opacity = 0.6 + 0.4 * max(0.0, min(1.0, prob))
        else:
            font_size = 14.0
            opacity = 0.85
        spans.append(
            f'<span class="motif-letter" style="font-size:{font_size:.1f}px; opacity:{opacity:.3f}">'
            f"{html.escape(residue)}</span>"
        )
    return "".join(spans)


def render_shared_structure_card(panel: dict[str, Any], idx: int) -> str:
    title = html.escape(str(panel.get("title", "")))
    color = html.escape(str(panel.get("color", "#000000")))
    subtitle = html.escape(str(panel.get("subtitle", "")))
    secondary_subtitle = str(panel.get("secondary_subtitle", "") or "").strip()
    secondary_markup = ""
    if secondary_subtitle:
        secondary_markup = (
            '<div class="subtitle-secondary">'
            f"{render_sequence_motif_markup(secondary_subtitle, panel.get('secondary_letter_probs'))}"
            "</div>"
        )
    return (
        '<div class="card">'
        f'<div class="title" style="color:{color}">{title}</div>'
        '<div class="subtitle">'
        f'<div class="subtitle-primary">{subtitle}</div>'
        f"{secondary_markup}"
        "</div>"
        f'<div class="viewer-shell" style="border:3px solid {color}"><div class="viewer" id="viewer-{idx}"></div></div>'
        "</div>"
    )


def case_member_color_map(members: list[dict[str, Any]]) -> dict[int, tuple[float, float, float]]:
    palette = sns.color_palette("colorblind", n_colors=max(3, len(members)))
    return {
        int(member["feature_id"]): palette[idx]
        for idx, member in enumerate(members)
    }


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []

    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end < start:
            continue
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
            continue
        merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def normalize_interval_bounds(
    intervals: list[tuple[int, int]],
    sequence_length: int,
) -> list[tuple[int, int]]:
    clipped: list[tuple[int, int]] = []
    for start, end in intervals:
        clipped_start = max(1, int(start))
        clipped_end = min(sequence_length, int(end))
        if clipped_end < clipped_start:
            continue
        clipped.append((clipped_start, clipped_end))
    return merge_intervals(clipped)


def interpro_cache_candidates(data_dir: Path, accession: str) -> list[Path]:
    candidates: list[Path] = []
    for root, dirname in (
        (data_dir, "interpro_cache"),
        (data_dir.parent, "interpro_cache"),
        (RENDER_CONFIG.base_output_dir if RENDER_CONFIG else None, "_interpro_cache"),
        (RENDER_CONFIG.base_output_dir if RENDER_CONFIG else None, "interpro_cache"),
    ):
        if root is None:
            continue
        candidates.append(Path(root) / dirname / f"{accession}.json")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate)
    return deduped


def load_cached_interpro_domains(cache_path: Path) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(cache_path.read_text())
    except Exception:
        return None
    domains = payload.get("domains")
    if not isinstance(domains, list):
        return None
    return [domain for domain in domains if isinstance(domain, dict)]


def extract_interpro_domains_from_response(response_json: dict[str, Any]) -> list[dict[str, Any]]:
    domains: list[dict[str, Any]] = []
    for entry in response_json.get("results", []):
        metadata = entry.get("metadata") or {}
        accession = str(metadata.get("accession", "") or "")
        name = str(metadata.get("name", "") or "")
        entry_type = str(metadata.get("type", "") or "")
        fragments: list[tuple[int, int]] = []

        for protein in entry.get("proteins", []):
            for location in protein.get("entry_protein_locations", []):
                for fragment in location.get("fragments", []):
                    start = fragment.get("start")
                    end = fragment.get("end")
                    if start is None or end is None:
                        continue
                    try:
                        start_i = int(start)
                        end_i = int(end)
                    except (TypeError, ValueError):
                        continue
                    if end_i < start_i:
                        continue
                    fragments.append((start_i, end_i))

        for start, end in merge_intervals(fragments):
            domains.append({
                "interpro_accession": accession,
                "interpro_name": name,
                "type": entry_type,
                "member_db": "",
                "member_accession": "",
                "start": start,
                "end": end,
            })
    return domains


def fetch_interpro_domains(accession: str) -> list[dict[str, Any]]:
    domains: list[dict[str, Any]] = []
    next_url = INTERPRO_PROTEIN_URL.format(accession=accession)
    seen_urls: set[str] = set()

    while next_url and next_url not in seen_urls:
        seen_urls.add(next_url)
        request = Request(
            next_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "ProteinLensPaperFigures/1.0",
            },
        )
        try:
            with urlopen(request, timeout=INTERPRO_FETCH_TIMEOUT_SEC) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                return []
            return []
        except (URLError, TimeoutError, OSError, ValueError):
            return []

        domains.extend(extract_interpro_domains_from_response(payload))
        next_url = payload.get("next")
    return domains


def load_or_fetch_interpro_domains(accession: str, data_dir: Path) -> list[dict[str, Any]]:
    cache_candidates = interpro_cache_candidates(data_dir, accession)
    for cache_path in cache_candidates:
        if not cache_path.exists():
            continue
        cached = load_cached_interpro_domains(cache_path)
        if cached is not None:
            return cached

    domains = fetch_interpro_domains(accession)
    if not domains:
        return []

    if cache_candidates:
        cache_path = cache_candidates[0]
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({
                "accession": accession,
                "domains": domains,
            }, indent=2))
        except Exception:
            pass
    return domains


def get_annotation_fragments(
    data: FigureData,
    accession: str,
    annotation_code: str,
    annotation_name: str,
) -> list[tuple[int, int]]:
    domains = load_or_fetch_interpro_domains(accession, data.data_dir)
    fragments: list[tuple[int, int]] = []
    for domain in domains:
        code = str(domain.get("interpro_accession", "") or "")
        name = str(domain.get("interpro_name", "") or "")
        if annotation_code and code == annotation_code:
            fragments.append((int(domain.get("start", 0) or 0), int(domain.get("end", 0) or 0)))
            continue
        if annotation_name and name == annotation_name:
            fragments.append((int(domain.get("start", 0) or 0), int(domain.get("end", 0) or 0)))
    return merge_intervals(fragments)


def highlighted_residues(activation: list[float]) -> list[int]:
    arr = np.asarray(activation, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return []
    finite = arr[np.isfinite(arr)]
    max_val = float(np.nanmax(finite))
    if max_val <= 0:
        return []
    threshold = max(float(np.quantile(finite, 0.95)), 0.2 * max_val)
    return [
        idx + 1
        for idx, value in enumerate(arr)
        if np.isfinite(value) and value >= threshold
    ]


def select_active_geometry_features(members: list[dict[str, Any]], max_features: int = 14) -> list[str]:
    feature_max = {}
    for member in members:
        for fname, value in (member.get("feature_importances") or {}).items():
            feature_max[fname] = max(feature_max.get(fname, 0.0), value)
    active = [fname for fname, val in feature_max.items() if val > 0.02]
    active.sort(
        key=lambda fname: sum((member.get("feature_importances") or {}).get(fname, 0.0) for member in members),
        reverse=True,
    )
    return active[:max_features]


def build_importance_matrix(members: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    matrix = np.zeros((len(members), len(feature_names)))
    for i, member in enumerate(members):
        imps = member.get("feature_importances") or {}
        for j, fname in enumerate(feature_names):
            matrix[i, j] = imps.get(fname, 0.0)
    return matrix


def plot_family_importance_heatmap(family: dict[str, Any], family_dir: Path) -> str:
    members = family["members"]
    member_colors = case_member_color_map(members)
    feature_names = select_active_geometry_features(members)
    matrix = build_importance_matrix(members, feature_names)

    fig, ax = plt.subplots(figsize=(max(6.4, 0.42 * len(feature_names) + 2.2), max(3.2, 0.6 * len(members) + 1.4)))
    sns.heatmap(
        matrix,
        cmap=sns.color_palette("rocket_r", as_cmap=True),
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Importance"},
        xticklabels=[wrap(name, 16) for name in feature_names],
        yticklabels=[f"f/{m['feature_id']} ({case_member_descriptor(family, m)})" for m in members],
        ax=ax,
    )
    ax.set_xlabel("Geometry descriptor")
    ax.set_ylabel("SAE feature")
    ax.set_title(f"{family['annotation_name']}: geometry importance heatmap", fontsize=12, pad=8)
    for tick, member in zip(ax.get_yticklabels(), members):
        tick.set_color(to_hex(member_colors[int(member["feature_id"])]))
    polish_axes(ax, grid_axis="both")
    return save_pdf(fig, family_dir / "importance_heatmap.pdf")


def plot_family_cosine_heatmap(family: dict[str, Any], family_dir: Path) -> str:
    members = family["members"]
    member_colors = case_member_color_map(members)
    feature_names = select_active_geometry_features(members, max_features=24)
    matrix = build_importance_matrix(members, feature_names)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = matrix / norms
    cosine = normed @ normed.T

    labels = [f"f/{m['feature_id']}" for m in members]
    fig, ax = plt.subplots(figsize=(max(3.6, 0.8 * len(members) + 1.5), max(3.2, 0.8 * len(members) + 1.3)))
    sns.heatmap(
        cosine,
        cmap=sns.color_palette("crest", as_cmap=True),
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Cosine similarity"},
        ax=ax,
    )
    ax.set_title(f"{family['annotation_name']}: pairwise geometry similarity", fontsize=12, pad=8)
    for tick, member in zip(ax.get_xticklabels(), members):
        tick.set_color(to_hex(member_colors[int(member["feature_id"])]))
    for tick, member in zip(ax.get_yticklabels(), members):
        tick.set_color(to_hex(member_colors[int(member["feature_id"])]))
    polish_axes(ax, grid_axis="both")
    return save_pdf(fig, family_dir / "pairwise_cosine_heatmap.pdf")


def plot_family_metric_scatter(family: dict[str, Any], family_dir: Path) -> str:
    members = family["members"]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))

    x = [m["interpro_res_f1"] for m in members]
    y = [m["geom_pr_auc"] for m in members]
    sizes = [100 + 20 * m["pct_proteins_activated"] for m in members]
    member_colors = case_member_color_map(members)
    colors = [member_colors[int(member["feature_id"])] for member in members]
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.9, edgecolors="#334155", linewidths=0.4)

    for color, member in zip(colors, members):
        ax.text(
            member["interpro_res_f1"] + 0.01,
            member["geom_pr_auc"] + 0.007,
            f"f/{member['feature_id']}\n{case_member_descriptor(family, member)}",
            fontsize=8.2,
            color=color,
        )

    ax.set_xlabel("InterPro residue-level F1")
    ax.set_ylabel("Geometry PR-AUC")
    ax.set_title(f"{family['annotation_name']}: node-level metric tradeoffs", fontsize=12, pad=8)
    polish_axes(ax)
    sns.despine(ax=ax)
    return save_pdf(fig, family_dir / "member_metric_scatter.pdf")


def feature_sequence_entry_map(feature_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entry_map: dict[str, dict[str, Any]] = {}

    def consider(entry: dict[str, Any]) -> None:
        accession = str(entry.get("accession", "") or "")
        if not accession:
            return
        current = entry_map.get(accession)
        if current is None or float(entry.get("max_activation", 0.0) or 0.0) > float(current.get("max_activation", 0.0) or 0.0):
            entry_map[accession] = entry

    for entry in feature_data.get("top_sequences", []):
        if isinstance(entry, dict):
            consider(entry)

    for items in (feature_data.get("activation_bins") or {}).values():
        if not isinstance(items, list):
            continue
        for entry in items:
            if isinstance(entry, dict):
                consider(entry)
    return entry_map


def pick_shared_accession(
    data: FigureData,
    family: dict[str, Any],
) -> tuple[str | None, list[tuple[int, dict[str, Any] | None]]]:
    accession_to_entries: dict[str, dict[int, dict[str, Any]]] = {}
    per_feature_entries: dict[int, dict[str, dict[str, Any]]] = {}

    for member in family["members"]:
        fid = int(member["feature_id"])
        feature_entries = feature_sequence_entry_map(data.feature_json(fid))
        per_feature_entries[fid] = feature_entries
        for accession, entry in feature_entries.items():
            accession_to_entries.setdefault(accession, {})[fid] = entry

    if not accession_to_entries:
        return None, []

    best_accession = None
    best_key = (-1, -1.0)
    for accession, entry_map in accession_to_entries.items():
        shared_count = len(entry_map)
        total_activation = sum(float(entry.get("max_activation", 0.0) or 0.0) for entry in entry_map.values())
        key = (shared_count, total_activation)
        if key > best_key:
            best_accession = accession
            best_key = key

    if best_accession is None:
        return None, []

    ordered_entries: list[tuple[int, dict[str, Any] | None]] = []
    for member in family["members"]:
        fid = int(member["feature_id"])
        ordered_entries.append((fid, per_feature_entries.get(fid, {}).get(best_accession)))
    return best_accession, ordered_entries


def find_chrome_binary() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if candidate and Path(candidate).exists():
            return candidate
    return shutil.which("google-chrome") or shutil.which("chrome") or shutil.which("chromium")


def structure_renders_enabled() -> bool:
    return not (RENDER_CONFIG and RENDER_CONFIG.skip_structure_renders)


def top_hot_residues(activation: list[float], max_hot: int = 24) -> list[int]:
    arr = np.asarray(activation, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return []
    finite = arr[np.isfinite(arr)]
    max_val = float(np.nanmax(finite))
    if max_val <= 0:
        return []
    threshold = max(float(np.quantile(finite, 0.985)), 0.2 * max_val)
    hot = [idx + 1 for idx, value in enumerate(arr) if np.isfinite(value) and value >= threshold]
    if len(hot) > max_hot:
        order = np.argsort(arr)[::-1][:max_hot]
        hot = sorted(int(idx) + 1 for idx in order if np.isfinite(arr[idx]))
    return hot


def render_shared_structure_strip(
    accession: str,
    panel_specs: list[dict[str, Any]],
    output_png: Path,
) -> bool:
    if not structure_renders_enabled():
        return False
    chrome = find_chrome_binary()
    if chrome is None or not panel_specs:
        return False

    viewer_width = CASE_SHARED_STRUCTURE_VIEWER_WIDTH
    viewer_height = CASE_SHARED_STRUCTURE_VIEWER_HEIGHT
    gap = CASE_SHARED_STRUCTURE_GAP
    row_gap = CASE_SHARED_STRUCTURE_ROW_GAP
    padding = CASE_SHARED_STRUCTURE_PADDING
    if len(panel_specs) <= 5:
        grid_cols = len(panel_specs)
        grid_rows = 1
    else:
        grid_cols = CASE_SHARED_STRUCTURE_GRID_COLS
        grid_rows = max(
            CASE_SHARED_STRUCTURE_GRID_ROWS,
            math.ceil(len(panel_specs) / grid_cols),
        )
    card_overhead = CASE_SHARED_STRUCTURE_CARD_OVERHEAD
    window_width = grid_cols * viewer_width + max(0, grid_cols - 1) * gap + 2 * padding
    window_height = grid_rows * (viewer_height + card_overhead) + max(0, grid_rows - 1) * row_gap + 2 * padding

    payload = []
    for panel in panel_specs:
        activation = [round(float(v), 6) for v in panel["activation"]]
        payload.append({
            "featureId": int(panel["feature_id"]) if panel.get("feature_id") is not None else None,
            "title": panel.get("title") or (f"f/{int(panel['feature_id'])}" if panel.get("feature_id") is not None else "Panel"),
            "subtitle": panel["subtitle"],
            "secondary_subtitle": panel.get("secondary_subtitle"),
            "secondary_letter_probs": panel.get("secondary_letter_probs") or [],
            "color": panel["color"],
            "activation": activation,
            "hotResidues": top_hot_residues(activation),
            "highlightResidues": highlighted_residues(activation),
            "renderMode": panel.get("render_mode", "activation"),
            "missingReason": panel.get("missing_reason"),
            "annotationCode": panel.get("annotation_code"),
            "annotationName": panel.get("annotation_name"),
            "annotationFragments": panel.get("annotation_fragments") or [],
        })

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>INIT</title>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: white;
      font-family: "DejaVu Sans", sans-serif;
    }}
    .wrap {{
      padding: {padding}px;
      display: grid;
      grid-template-columns: repeat({grid_cols}, {viewer_width}px);
      column-gap: {gap}px;
      row-gap: {row_gap}px;
      align-items: start;
      justify-content: start;
      padding-bottom: {padding + 40}px;
    }}
    .card {{
      width: {viewer_width}px;
      display: flex;
      flex-direction: column;
      align-items: stretch;
    }}
    .title {{
      font-size: 24px;
      font-weight: 800;
      text-align: center;
      margin-bottom: 4px;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      margin-bottom: 10px;
      min-height: 78px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
      gap: 6px;
    }}
    .subtitle-primary {{
      font-size: 14px;
      color: #475569;
      text-align: center;
      line-height: 1.2;
      font-weight: 500;
    }}
    .subtitle-secondary {{
      min-height: 28px;
      display: flex;
      align-items: flex-end;
      justify-content: center;
      gap: 1px;
      flex-wrap: nowrap;
      color: #B45309;
    }}
    .motif-letter {{
      display: inline-block;
      line-height: 0.92;
      font-weight: 800;
      letter-spacing: -0.03em;
    }}
    .viewer-shell {{
      border-radius: 16px;
      overflow: hidden;
      background: white;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
      margin-bottom: 28px;
    }}
    .viewer {{
      width: {viewer_width}px;
      height: {viewer_height}px;
      position: relative;
      overflow: hidden;
      background: white;
    }}
    .viewer canvas {{
      position: absolute !important;
      inset: 0 !important;
    }}
    .viewer-note {{
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 18px;
      color: #64748B;
      font-size: 18px;
      font-weight: 600;
      line-height: 1.35;
    }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"></script>
</head>
<body>
  <div class="wrap">
    {"".join(render_shared_structure_card(panel, idx) for idx, panel in enumerate(payload))}
  </div>
  <script>
    const ACCESSION = {json.dumps(accession)};
    const PANELS = {json.dumps(payload)};
    const INTERPRO_URL = `https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/${{ACCESSION}}?page_size=200`;
    const HIGHLIGHT_COLOR = {STRUCTURE_HIGHLIGHT_COLOR_INT};

    function focusSelection(panel) {{
      const residues = (panel.highlightResidues && panel.highlightResidues.length)
        ? panel.highlightResidues
        : panel.hotResidues;
      if (!residues || !residues.length) return null;
      const minResi = Math.min(...residues);
      const maxResi = Math.max(...residues);
      const start = Math.max(1, minResi - 14);
      const end = Math.min(panel.activation.length, maxResi + 14);
      return {{ resi: `${{start}}-${{end}}` }};
    }}

    function unionFocusSelection(panels) {{
      const all = [];
      for (const panel of panels) {{
        const residues = (panel.highlightResidues && panel.highlightResidues.length)
          ? panel.highlightResidues
          : panel.hotResidues;
        if (residues && residues.length) {{
          all.push(...residues);
        }}
      }}
      if (!all.length) return null;
      const minResi = Math.min(...all);
      const maxResi = Math.max(...all);
      const start = Math.max(1, minResi - 10);
      const end = Math.min(panels[0].activation.length, maxResi + 10);
      return {{ resi: `${{start}}-${{end}}` }};
    }}

    async function fetchInterproEntries(accession) {{
      let url = INTERPRO_URL;
      const results = [];
      let guard = 0;
      while (url && guard < 12) {{
        const res = await fetch(url, {{
          headers: {{
            "Accept": "application/json"
          }}
        }});
        if (!res.ok) throw new Error(`interpro ${{res.status}}`);
        const data = await res.json();
        results.push(...(data.results || []));
        url = data.next || null;
        guard += 1;
      }}
      return results;
    }}

    function expandAnnotationFragments(fragments) {{
      const residues = [];
      for (const fragment of (fragments || [])) {{
        if (!Array.isArray(fragment) || fragment.length < 2) continue;
        const start = Number(fragment[0] || 0);
        const end = Number(fragment[1] || 0);
        if (!start || !end || end < start) continue;
        for (let resi = start; resi <= end; resi++) {{
          residues.push(resi);
        }}
      }}
      return Array.from(new Set(residues)).sort((a, b) => a - b);
    }}

    function extractAnnotationResidues(entries, annotationCode, annotationName) {{
      const residues = [];
      for (const entry of entries) {{
        const metadata = entry.metadata || {{}};
        const code = metadata.accession || "";
        const name = metadata.name || "";
        const matches = (annotationCode && code === annotationCode) ||
          (annotationName && name === annotationName);
        if (!matches) continue;

        for (const protein of (entry.proteins || [])) {{
          for (const loc of (protein.entry_protein_locations || [])) {{
            for (const frag of (loc.fragments || [])) {{
              const start = Number(frag.start || 0);
              const end = Number(frag.end || 0);
              if (!start || !end || end < start) continue;
              for (let resi = start; resi <= end; resi++) {{
                residues.push(resi);
              }}
            }}
          }}
        }}
      }}
      return Array.from(new Set(residues)).sort((a, b) => a - b);
    }}

    function residueSelection(residues, sequenceLength, padding = 10) {{
      if (!residues || !residues.length) return null;
      const minResi = Math.min(...residues);
      const maxResi = Math.max(...residues);
      const start = Math.max(1, minResi - padding);
      const end = Math.min(sequenceLength, maxResi + padding);
      return {{ resi: `${{start}}-${{end}}` }};
    }}

    async function fetchPdb(accession) {{
      const pred = await fetch(`https://alphafold.ebi.ac.uk/api/prediction/${{accession}}`);
      if (!pred.ok) throw new Error(`prediction ${{pred.status}}`);
      const predJson = await pred.json();
      const pdbUrl = predJson[0].pdbUrl;
      const pdbRes = await fetch(pdbUrl);
      if (!pdbRes.ok) throw new Error(`pdb ${{pdbRes.status}}`);
      return await pdbRes.text();
    }}

    (async () => {{
      try {{
        const pdb = await fetchPdb(ACCESSION);
        const needsAnnotationFetch = PANELS.some(
          panel => panel.renderMode === "annotation" && (!panel.annotationFragments || !panel.annotationFragments.length)
        );
        const interproEntries = needsAnnotationFetch ? await fetchInterproEntries(ACCESSION) : [];
        let masterView = null;
        for (let idx = 0; idx < PANELS.length; idx++) {{
          const panel = PANELS[idx];
          const viewer = $3Dmol.createViewer(`viewer-${{idx}}`, {{ backgroundColor: "white", antialias: true }});
          viewer.addModel(pdb, "pdb");

          if (panel.renderMode === "annotation") {{
            const annotationResidues = (panel.annotationFragments && panel.annotationFragments.length)
              ? expandAnnotationFragments(panel.annotationFragments)
              : extractAnnotationResidues(interproEntries, panel.annotationCode, panel.annotationName);
            const residueSet = new Set(annotationResidues);
            viewer.setStyle({{}}, {{
              cartoon: {{
                colorfunc: function(atom) {{
                  return residueSet.has(atom.resi) ? HIGHLIGHT_COLOR : 0xE5E7EB;
                }}
              }}
            }});
          }} else if (panel.renderMode === "missing") {{
            viewer.setStyle({{}}, {{
              cartoon: {{
                color: "#D1D5DB"
              }}
            }});
            const note = document.createElement("div");
            note.className = "viewer-note";
            note.textContent = panel.missingReason || "No cached trace for this protein";
            document.getElementById(`viewer-${{idx}}`).appendChild(note);
          }} else {{
            const highlightResidues = (panel.highlightResidues && panel.highlightResidues.length)
              ? panel.highlightResidues
              : panel.hotResidues;
            const residueSet = new Set(highlightResidues || []);
            viewer.setStyle({{}}, {{
              cartoon: {{
                colorfunc: function(atom) {{
                  return residueSet.has(atom.resi) ? HIGHLIGHT_COLOR : 0xE5E7EB;
                }}
              }}
            }});
          }}
          viewer.resize();
          if (masterView) {{
            viewer.setView(masterView);
          }} else {{
            // Fit the full protein into the larger export card so highlighted
            // regions stay visible without clipping the surrounding fold.
            viewer.center();
            viewer.zoomTo();
            viewer.zoom({CASE_SHARED_STRUCTURE_GLOBAL_ZOOM});
            masterView = viewer.getView();
          }}
          viewer.render();
        }}

        await new Promise(resolve => setTimeout(resolve, 600));
        document.title = "READY";
      }} catch (err) {{
        document.body.innerHTML = `<pre>${{String(err)}}</pre>`;
        document.title = "ERR";
      }}
    }})();
  </script>
</body>
</html>
"""

    with tempfile.TemporaryDirectory(prefix="proteinlens_chrome_") as tmpdir:
        html_path = Path(tmpdir) / "render.html"
        screenshot_path = Path(tmpdir) / "render.png"
        html_path.write_text(html)

        chrome_flags = [
            chrome,
            "--headless",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--enable-unsafe-swiftshader",
            "--use-angle=swiftshader",
            "--use-gl=swiftshader",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--force-device-scale-factor={ASSET_RENDER_DEVICE_SCALE_FACTOR}",
            f"--window-size={window_width},{window_height}",
            "--run-all-compositor-stages-before-draw",
            "--dump-dom",
            "--virtual-time-budget=25000",
            f"file://{html_path}",
        ]
        try:
            dom_result = subprocess.run(
                chrome_flags,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return False
        if dom_result.returncode != 0 or "<title>READY</title>" not in dom_result.stdout:
            return False

        screenshot_cmd = [
            chrome,
            "--headless",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--enable-unsafe-swiftshader",
            "--use-angle=swiftshader",
            "--use-gl=swiftshader",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--force-device-scale-factor={ASSET_RENDER_DEVICE_SCALE_FACTOR}",
            f"--window-size={window_width},{window_height}",
            "--run-all-compositor-stages-before-draw",
            f"--screenshot={screenshot_path}",
            "--virtual-time-budget=25000",
            f"file://{html_path}",
        ]
        try:
            shot_result = subprocess.run(
                screenshot_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return False
        if shot_result.returncode != 0 or not screenshot_path.exists():
            return False

        output_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(screenshot_path, output_png)
        return True


def render_cross_structure_strip(
    panel_specs: list[dict[str, Any]],
    output_png: Path,
) -> bool:
    if not structure_renders_enabled():
        return False
    chrome = find_chrome_binary()
    if chrome is None or not panel_specs:
        return False

    viewer_width = 430
    viewer_height = 400
    gap = 18
    padding = 18
    window_width = len(panel_specs) * viewer_width + max(0, len(panel_specs) - 1) * gap + 2 * padding
    window_height = viewer_height + 220

    payload = []
    for panel in panel_specs:
        activation = [round(float(v), 6) for v in panel["activation"]]
        payload.append({
            "accession": panel["accession"],
            "title": panel["title"],
            "subtitle": panel["subtitle"],
            "color": panel["color"],
            "activation": activation,
            "hotResidues": top_hot_residues(activation),
        })

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>INIT</title>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: white;
      font-family: "DejaVu Sans", sans-serif;
    }}
    .wrap {{
      padding: {padding}px;
      display: grid;
      grid-template-columns: repeat({len(payload)}, {viewer_width}px);
      gap: {gap}px;
      align-items: start;
      justify-content: start;
      padding-bottom: 88px;
    }}
    .card {{
      width: {viewer_width}px;
      display: flex;
      flex-direction: column;
      align-items: stretch;
    }}
    .title {{
      font-size: 24px;
      font-weight: 800;
      text-align: center;
      margin-bottom: 4px;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      font-size: 14px;
      color: #475569;
      text-align: center;
      min-height: 36px;
      margin-bottom: 10px;
      line-height: 1.2;
      font-weight: 500;
    }}
    .viewer-shell {{
      border-radius: 16px;
      overflow: hidden;
      background: white;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
      margin-bottom: 28px;
    }}
    .viewer {{
      width: {viewer_width}px;
      height: {viewer_height}px;
      position: relative;
      overflow: hidden;
      background: white;
    }}
    .viewer canvas {{
      position: absolute !important;
      inset: 0 !important;
    }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"></script>
</head>
<body>
  <div class="wrap">
    {"".join(
        f'<div class="card"><div class="title" style="color:{panel["color"]}">{panel["title"]}</div>'
        f'<div class="subtitle">{panel["subtitle"]}</div>'
        f'<div class="viewer-shell" style="border:3px solid {panel["color"]}"><div class="viewer" id="viewer-{idx}"></div></div></div>'
        for idx, panel in enumerate(payload)
    )}
  </div>
  <script>
    const PANELS = {json.dumps(payload)};

    function activationColorHex(value, maxValue) {{
      const norm = maxValue > 0 ? Math.min(Math.max(value / maxValue, 0), 1) : 0;
      const r = 255;
      const g = Math.round(255 - norm * (255 - 38));
      const b = Math.round(255 - norm * (255 - 38));
      return (r << 16) | (g << 8) | b;
    }}

    async function fetchPdb(accession) {{
      const pred = await fetch(`https://alphafold.ebi.ac.uk/api/prediction/${{accession}}`);
      if (!pred.ok) throw new Error(`prediction ${{pred.status}}`);
      const predJson = await pred.json();
      const pdbUrl = predJson[0].pdbUrl;
      const pdbRes = await fetch(pdbUrl);
      if (!pdbRes.ok) throw new Error(`pdb ${{pdbRes.status}}`);
      return await pdbRes.text();
    }}

    (async () => {{
      try {{
        for (let idx = 0; idx < PANELS.length; idx++) {{
          const panel = PANELS[idx];
          const pdb = await fetchPdb(panel.accession);
          const viewer = $3Dmol.createViewer(`viewer-${{idx}}`, {{ backgroundColor: "white", antialias: true }});
          viewer.addModel(pdb, "pdb");

          const maxAct = Math.max(...panel.activation, 0);
          const colorMap = {{}};
          for (let i = 0; i < panel.activation.length; i++) {{
            colorMap[i + 1] = activationColorHex(panel.activation[i], maxAct);
          }}

          viewer.setStyle({{}}, {{
            cartoon: {{
              colorfunc: function(atom) {{
                return colorMap[atom.resi] ?? 0xE5E7EB;
              }}
            }}
          }});
          viewer.resize();
          // Cross-family panels compare whole-protein context across accessions,
          // so keep the full structure in frame and let color carry the local hit.
          viewer.center();
          viewer.zoomTo();
          viewer.zoom(0.96);
          viewer.render();
        }}

        await new Promise(resolve => setTimeout(resolve, 600));
        document.title = "READY";
      }} catch (err) {{
        document.body.innerHTML = `<pre>${{String(err)}}</pre>`;
        document.title = "ERR";
      }}
    }})();
  </script>
</body>
</html>
"""

    with tempfile.TemporaryDirectory(prefix="proteinlens_chrome_") as tmpdir:
        html_path = Path(tmpdir) / "render.html"
        screenshot_path = Path(tmpdir) / "render.png"
        html_path.write_text(html)

        chrome_flags = [
            chrome,
            "--headless",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--enable-unsafe-swiftshader",
            "--use-angle=swiftshader",
            "--use-gl=swiftshader",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--force-device-scale-factor={ASSET_RENDER_DEVICE_SCALE_FACTOR}",
            f"--window-size={window_width},{window_height}",
            "--run-all-compositor-stages-before-draw",
            "--dump-dom",
            "--virtual-time-budget=25000",
            f"file://{html_path}",
        ]
        dom_result = subprocess.run(chrome_flags, capture_output=True, text=True, check=False)
        if dom_result.returncode != 0 or "<title>READY</title>" not in dom_result.stdout:
            return False

        screenshot_cmd = [
            chrome,
            "--headless",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--enable-unsafe-swiftshader",
            "--use-angle=swiftshader",
            "--use-gl=swiftshader",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--force-device-scale-factor={ASSET_RENDER_DEVICE_SCALE_FACTOR}",
            f"--window-size={window_width},{window_height}",
            "--run-all-compositor-stages-before-draw",
            f"--screenshot={screenshot_path}",
            "--virtual-time-budget=25000",
            f"file://{html_path}",
        ]
        shot_result = subprocess.run(screenshot_cmd, capture_output=True, text=True, check=False)
        if shot_result.returncode != 0 or not screenshot_path.exists():
            return False

        output_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(screenshot_path, output_png)
        return True


def render_cross_structure_panel(
    accession: str,
    activation: list[float],
    color: str,
    output_png: Path,
) -> bool:
    if not structure_renders_enabled():
        return False
    chrome = find_chrome_binary()
    if chrome is None or not accession:
        return False

    viewer_width = 420
    viewer_height = 340
    padding = 22
    safe_margin = 26
    window_width = viewer_width + 2 * padding + safe_margin
    window_height = viewer_height + 2 * padding + safe_margin
    activation_values = [round(float(v), 6) for v in activation]

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>INIT</title>
    <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: white;
      font-family: "DejaVu Sans", sans-serif;
    }}
    .wrap {{
      display: inline-block;
      padding: {padding}px {padding + 16}px {padding + 16}px {padding}px;
    }}
    .viewer-shell {{
      border-radius: 16px;
      overflow: hidden;
      background: white;
      border: 3px solid {color};
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
    }}
    .viewer {{
      width: {viewer_width}px;
      height: {viewer_height}px;
      position: relative;
      overflow: hidden;
      background: white;
    }}
    .viewer canvas {{
      position: absolute !important;
      inset: 0 !important;
    }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"></script>
</head>
<body>
  <div class="wrap">
    <div class="viewer-shell"><div class="viewer" id="viewer"></div></div>
  </div>
  <script>
    const ACCESSION = {json.dumps(accession)};
    const ACTIVATION = {json.dumps(activation_values)};

    function activationColorHex(value, maxValue) {{
      const norm = maxValue > 0 ? Math.min(Math.max(value / maxValue, 0), 1) : 0;
      const r = 255;
      const g = Math.round(255 - norm * (255 - 38));
      const b = Math.round(255 - norm * (255 - 38));
      return (r << 16) | (g << 8) | b;
    }}

    async function fetchPdb(accession) {{
      const pred = await fetch(`https://alphafold.ebi.ac.uk/api/prediction/${{accession}}`);
      if (!pred.ok) throw new Error(`prediction ${{pred.status}}`);
      const predJson = await pred.json();
      const pdbUrl = predJson[0].pdbUrl;
      const pdbRes = await fetch(pdbUrl);
      if (!pdbRes.ok) throw new Error(`pdb ${{pdbRes.status}}`);
      return await pdbRes.text();
    }}

    (async () => {{
      try {{
        const pdb = await fetchPdb(ACCESSION);
        const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white", antialias: true }});
        viewer.addModel(pdb, "pdb");

        const maxAct = Math.max(...ACTIVATION, 0);
        const colorMap = {{}};
        for (let i = 0; i < ACTIVATION.length; i++) {{
          colorMap[i + 1] = activationColorHex(ACTIVATION[i], maxAct);
        }}

        viewer.setStyle({{}}, {{
          cartoon: {{
            colorfunc: function(atom) {{
              return colorMap[atom.resi] ?? 0xE5E7EB;
            }}
          }}
        }});
        viewer.resize();
        viewer.center();
        viewer.zoomTo();
        viewer.zoom(0.96);
        viewer.render();

        await new Promise(resolve => setTimeout(resolve, 600));
        document.title = "READY";
      }} catch (err) {{
        document.body.innerHTML = `<pre>${{String(err)}}</pre>`;
        document.title = "ERR";
      }}
    }})();
  </script>
</body>
</html>
"""

    with tempfile.TemporaryDirectory(prefix="proteinlens_chrome_") as tmpdir:
        html_path = Path(tmpdir) / "render.html"
        screenshot_path = Path(tmpdir) / "render.png"
        html_path.write_text(html)

        chrome_flags = [
            chrome,
            "--headless",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--enable-unsafe-swiftshader",
            "--use-angle=swiftshader",
            "--use-gl=swiftshader",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--force-device-scale-factor={ASSET_RENDER_DEVICE_SCALE_FACTOR}",
            f"--window-size={window_width},{window_height}",
            "--run-all-compositor-stages-before-draw",
            "--dump-dom",
            "--virtual-time-budget=25000",
            f"file://{html_path}",
        ]
        dom_result = subprocess.run(chrome_flags, capture_output=True, text=True, check=False)
        if dom_result.returncode != 0 or "<title>READY</title>" not in dom_result.stdout:
            return False

        screenshot_cmd = [
            chrome,
            "--headless",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--enable-unsafe-swiftshader",
            "--use-angle=swiftshader",
            "--use-gl=swiftshader",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--force-device-scale-factor={ASSET_RENDER_DEVICE_SCALE_FACTOR}",
            f"--window-size={window_width},{window_height}",
            "--run-all-compositor-stages-before-draw",
            f"--screenshot={screenshot_path}",
            "--virtual-time-budget=25000",
            f"file://{html_path}",
        ]
        shot_result = subprocess.run(screenshot_cmd, capture_output=True, text=True, check=False)
        if shot_result.returncode != 0 or not screenshot_path.exists():
            return False

        output_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(screenshot_path, output_png)
        return True


def render_geometry_dominant_pair(
    accession: str,
    activation: list[float],
    motif_pdb: str,
    output_png: Path,
) -> bool:
    if not structure_renders_enabled():
        return False
    chrome = find_chrome_binary()
    if chrome is None or not accession or not motif_pdb:
        return False

    viewer_width = GEOMETRY_DOMINANT_VIEWER_WIDTH
    viewer_height = GEOMETRY_DOMINANT_VIEWER_HEIGHT
    gap = 22
    padding = 18
    window_width = 2 * viewer_width + gap + 2 * padding
    window_height = viewer_height + 210

    activation_values = [round(float(v), 6) for v in activation]
    hot_residues = top_hot_residues(activation_values)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>INIT</title>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: white;
      font-family: "DejaVu Sans", sans-serif;
    }}
    .wrap {{
      padding: {padding}px;
      display: grid;
      grid-template-columns: {viewer_width}px {viewer_width}px;
      gap: {gap}px;
      align-items: start;
      justify-content: start;
      padding-bottom: 72px;
    }}
    .card {{
      width: {viewer_width}px;
      display: flex;
      flex-direction: column;
      align-items: stretch;
    }}
    .title {{
      font-size: 22px;
      font-weight: 800;
      text-align: center;
      margin-bottom: 6px;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      font-size: 14px;
      color: #475569;
      text-align: center;
      min-height: 34px;
      margin-bottom: 10px;
      line-height: 1.2;
      font-weight: 500;
    }}
    .viewer-shell {{
      border-radius: 16px;
      overflow: hidden;
      background: white;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
      margin-bottom: 24px;
    }}
    .viewer {{
      width: {viewer_width}px;
      height: {viewer_height}px;
      position: relative;
      overflow: hidden;
      background: white;
    }}
    .viewer canvas {{
      position: absolute !important;
      inset: 0 !important;
    }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"></script>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="title" style="color:{PALETTE["primary"]}">Representative Protein</div>
      <div class="subtitle">{accession}</div>
      <div class="viewer-shell" style="border:3px solid {PALETTE["primary"]}"><div class="viewer" id="protein-viewer"></div></div>
    </div>
    <div class="card">
      <div class="title" style="color:{PALETTE["accent"]}">Mean Geometry Template</div>
      <div class="subtitle">Kabsch-aligned C-alpha mean structure</div>
      <div class="viewer-shell" style="border:3px solid {PALETTE["accent"]}"><div class="viewer" id="motif-viewer"></div></div>
    </div>
  </div>
  <script>
    const ACCESSION = {json.dumps(accession)};
    const ACTIVATION = {json.dumps(activation_values)};
    const HOT_RESIDUES = {json.dumps(hot_residues)};
    const MOTIF_PDB = {json.dumps(motif_pdb)};

    function activationColorHex(value, maxValue) {{
      const norm = maxValue > 0 ? Math.min(Math.max(value / maxValue, 0), 1) : 0;
      const r = 255;
      const g = Math.round(255 - norm * (255 - 38));
      const b = Math.round(255 - norm * (255 - 38));
      return (r << 16) | (g << 8) | b;
    }}

    function parseCaCoords(pdbText) {{
      const coords = [];
      const lines = pdbText.split(/\\r?\\n/);
      for (const line of lines) {{
        if (!line.startsWith("ATOM")) continue;
        const atomName = line.slice(12, 16).trim();
        if (atomName !== "CA") continue;
        coords.push({{
          x: parseFloat(line.slice(30, 38)),
          y: parseFloat(line.slice(38, 46)),
          z: parseFloat(line.slice(46, 54)),
        }});
      }}
      return coords.filter(pt => Number.isFinite(pt.x) && Number.isFinite(pt.y) && Number.isFinite(pt.z));
    }}

    async function fetchPdb(accession) {{
      const pred = await fetch(`https://alphafold.ebi.ac.uk/api/prediction/${{accession}}`);
      if (!pred.ok) throw new Error(`prediction ${{pred.status}}`);
      const predJson = await pred.json();
      const pdbUrl = predJson[0].pdbUrl;
      const pdbRes = await fetch(pdbUrl);
      if (!pdbRes.ok) throw new Error(`pdb ${{pdbRes.status}}`);
      return await pdbRes.text();
    }}

    (async () => {{
      try {{
        const pdb = await fetchPdb(ACCESSION);
        const proteinViewer = $3Dmol.createViewer("protein-viewer", {{ backgroundColor: "white", antialias: true }});
        proteinViewer.addModel(pdb, "pdb");
        const maxAct = Math.max(...ACTIVATION, 0);
        const colorMap = {{}};
        for (let i = 0; i < ACTIVATION.length; i++) {{
          colorMap[i + 1] = activationColorHex(ACTIVATION[i], maxAct);
        }}
        proteinViewer.setStyle({{}}, {{
          cartoon: {{
            colorfunc: function(atom) {{
              return colorMap[atom.resi] ?? 0xE5E7EB;
            }}
          }}
        }});
        proteinViewer.resize();
        // Keep the full representative protein in frame for spotlight exports.
        proteinViewer.center();
        proteinViewer.zoomTo();
        proteinViewer.zoom({GEOMETRY_DOMINANT_PROTEIN_GLOBAL_ZOOM});
        proteinViewer.render();

        const motifViewer = $3Dmol.createViewer("motif-viewer", {{ backgroundColor: "white", antialias: true }});
        const motifCoords = parseCaCoords(MOTIF_PDB);
        for (let i = 0; i < motifCoords.length - 1; i++) {{
          motifViewer.addCylinder({{
            start: motifCoords[i],
            end: motifCoords[i + 1],
            radius: 0.28,
            color: "#DC2626",
            fromCap: 1,
            toCap: 1
          }});
        }}
        for (const pt of motifCoords) {{
          motifViewer.addSphere({{
            center: pt,
            radius: 0.34,
            color: "#FCA5A5"
          }});
        }}
        motifViewer.resize();
        motifViewer.zoomTo();
        motifViewer.zoom(1.24);
        motifViewer.render();

        await new Promise(resolve => setTimeout(resolve, 600));
        document.title = "READY";
      }} catch (err) {{
        document.body.innerHTML = `<pre>${{String(err)}}</pre>`;
        document.title = "ERR";
      }}
    }})();
  </script>
</body>
</html>
"""

    with tempfile.TemporaryDirectory(prefix="proteinlens_chrome_") as tmpdir:
        html_path = Path(tmpdir) / "render.html"
        screenshot_path = Path(tmpdir) / "render.png"
        html_path.write_text(html)

        chrome_flags = [
            chrome,
            "--headless",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--enable-unsafe-swiftshader",
            "--use-angle=swiftshader",
            "--use-gl=swiftshader",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--force-device-scale-factor={ASSET_RENDER_DEVICE_SCALE_FACTOR}",
            f"--window-size={window_width},{window_height}",
            "--run-all-compositor-stages-before-draw",
            "--dump-dom",
            "--virtual-time-budget=25000",
            f"file://{html_path}",
        ]
        dom_result = subprocess.run(chrome_flags, capture_output=True, text=True, check=False)
        if dom_result.returncode != 0 or "<title>READY</title>" not in dom_result.stdout:
            return False

        screenshot_cmd = [
            chrome,
            "--headless",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--enable-unsafe-swiftshader",
            "--use-angle=swiftshader",
            "--use-gl=swiftshader",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--force-device-scale-factor={ASSET_RENDER_DEVICE_SCALE_FACTOR}",
            f"--window-size={window_width},{window_height}",
            "--run-all-compositor-stages-before-draw",
            f"--screenshot={screenshot_path}",
            "--virtual-time-budget=25000",
            f"file://{html_path}",
        ]
        shot_result = subprocess.run(screenshot_cmd, capture_output=True, text=True, check=False)
        if shot_result.returncode != 0 or not screenshot_path.exists():
            return False

        output_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(screenshot_path, output_png)
        return True


def draw_shared_overlay_axes(
    ax: plt.Axes,
    family: dict[str, Any],
    entries: list[tuple[int, dict[str, Any]]],
    member_map: dict[int, dict[str, Any]],
    color_map: dict[int, tuple[float, float, float]],
    annotation_fragments: list[tuple[int, int]],
    seq_len: int,
) -> None:
    x = np.arange(1, seq_len + 1)
    legend_handles: list[Any] = []

    if annotation_fragments:
        for start, end in annotation_fragments:
            ax.axvspan(
                start - 0.5,
                end + 0.5,
                color=CASE_DOMAIN_SHADE_COLOR,
                alpha=0.42,
                linewidth=0,
                zorder=0,
            )
        legend_handles.append(Patch(
            facecolor=CASE_DOMAIN_SHADE_COLOR,
            edgecolor="none",
            alpha=0.42,
            label=f"{family['annotation_code']} residue domain",
        ))

    for fid, seq_entry in entries:
        member = member_map.get(fid, {})
        line, = ax.plot(
            x,
            seq_entry.get("per_residue_activations", []),
            color=color_map[fid],
            linewidth=1.6,
            label=f"f/{fid} ({case_member_descriptor(family, member)})",
        )
        legend_handles.append(line)

    ax.set_xlabel("Residue position")
    ax.set_ylabel("SAE activation")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)


def plot_family_shared_overlay(data: FigureData, family: dict[str, Any], family_dir: Path) -> str | None:
    accession, entries = pick_shared_accession(data, family)
    available_entries = [(fid, seq_entry) for fid, seq_entry in entries if seq_entry is not None]
    if accession is None or len(available_entries) < 2:
        return None

    seq_data = available_entries[0][1]
    assert seq_data is not None
    seq_len = len(seq_data.get("per_residue_activations", []))
    annotation_fragments = normalize_interval_bounds(
        get_annotation_fragments(
            data,
            accession,
            family["annotation_code"],
            family["annotation_name"],
        ),
        seq_len,
    )

    member_map = {int(member["feature_id"]): member for member in family["members"]}
    member_colors = case_member_color_map(family["members"])
    color_map = {
        int(fid): member_colors[int(fid)]
        for fid, _ in entries
        if int(fid) in member_colors
    }

    panel_specs = []
    for fid, seq_entry in entries:
        member = member_map.get(fid, {})
        if seq_entry is None:
            continue
        secondary_motif = case_member_secondary_motif_label(family, member)
        panel_specs.append({
            "feature_id": fid,
            "title": f"f/{fid}",
            "subtitle": case_member_descriptor(family, member),
            "secondary_subtitle": secondary_motif,
            "secondary_letter_probs": member.get("sequence_motif_letter_probs", []) if secondary_motif else [],
            "activation": seq_entry.get("per_residue_activations", []),
            "color": to_hex(color_map[fid]),
            "render_mode": "activation",
        })

    annotation_signal = [0.0] * seq_len
    panel_specs.append({
        "feature_id": None,
        "title": family["annotation_code"],
        "subtitle": f"InterPro: {family['annotation_name']}",
        "activation": annotation_signal,
        "color": PALETTE["slate"],
        "render_mode": "annotation",
        "annotation_code": family["annotation_code"],
        "annotation_name": family["annotation_name"],
        "annotation_fragments": annotation_fragments,
    })

    asset_path = family_dir / "_render_assets" / f"shared_protein_structures_{accession}.png"
    render_shared_structure_strip(accession, panel_specs, asset_path)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    draw_shared_overlay_axes(
        ax,
        family,
        available_entries,
        member_map,
        color_map,
        annotation_fragments,
        seq_len,
    )
    ax.set_title(f"{family['annotation_name']}: shared protein overlay ({accession})", fontsize=12, pad=8)
    polish_axes(ax)
    sns.despine(ax=ax)
    return save_pdf(fig, family_dir / f"shared_protein_overlay_{accession}.pdf")


def make_case_family_bundle(data: FigureData, family: dict[str, Any], family_dir: Path) -> dict[str, str]:
    plotted_family = prepare_case_study_family_for_plotting(data, family)
    outputs = {
        "importance_heatmap": plot_family_importance_heatmap(plotted_family, family_dir),
        "pairwise_cosine_heatmap": plot_family_cosine_heatmap(plotted_family, family_dir),
        "member_metric_scatter": plot_family_metric_scatter(plotted_family, family_dir),
    }
    shared = plot_family_shared_overlay(data, plotted_family, family_dir)
    if shared:
        outputs["shared_protein_overlay"] = shared
    return outputs


def select_cross_representative_proteins(geometry: dict[str, Any], max_proteins: int = 3) -> list[dict[str, Any]]:
    representatives: list[dict[str, Any]] = []
    seen_accessions: set[str] = set()
    for protein in geometry.get("plot_data", {}).get("top_proteins", []):
        accession = protein.get("accession")
        activation = protein.get("sae_activation_profile") or []
        if not accession or accession in seen_accessions or not activation:
            continue
        representatives.append(protein)
        seen_accessions.add(accession)
        if len(representatives) >= max_proteins:
            break
    return representatives


def activated_position_indices(protein: dict[str, Any]) -> list[int]:
    positions: list[int] = []
    for item in protein.get("activated_positions", []):
        if isinstance(item, dict):
            pos = item.get("position")
        else:
            pos = item
        if isinstance(pos, int):
            positions.append(pos + 1)
    if positions:
        return sorted(set(positions))

    activation = protein.get("sae_activation_profile") or []
    return top_hot_residues([float(v) for v in activation], max_hot=8)


def format_residue_span_label(positions: list[int]) -> str:
    if not positions:
        return "active residues unavailable"
    if len(positions) == 1:
        return f"active residue {positions[0]}"
    return f"active residues {positions[0]}-{positions[-1]}"


def plot_cross_feature_representative_structures(
    data: FigureData,
    fid: int,
    feature_dir: Path,
) -> str | None:
    geometry = data.geometry_json(fid)
    proteins = select_cross_representative_proteins(geometry, max_proteins=3)
    if len(proteins) < 2:
        return None

    palette = sns.color_palette("colorblind", n_colors=len(proteins))
    feature_importances = geometry.get("geometric_residue_level", {}).get("feature_importances", {})

    ymax = max(max((protein.get("sae_activation_profile") or [0.0])) for protein in proteins)
    ymax = max(1.0, float(ymax) * 1.08)
    geom_ymax = max(max((protein.get("geom_prob_profile") or [0.0])) for protein in proteins)
    geom_ymax = max(1.0, float(geom_ymax) * 1.06)

    n_cols = len(proteins) + 1
    fig = plt.figure(figsize=(max(11.0, 2.8 * len(proteins) + 3.0), 5.6))
    grid = fig.add_gridspec(
        2,
        n_cols,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0] * len(proteins) + [0.95],
        hspace=0.22,
        wspace=0.18,
    )

    for idx, (color, protein) in enumerate(zip(palette, proteins)):
        ax = fig.add_subplot(grid[0, idx])
        activation = np.asarray(protein.get("sae_activation_profile", []), dtype=float)
        x = np.arange(1, len(activation) + 1)
        ax.fill_between(x, activation, color=color, alpha=0.22, linewidth=0)
        ax.plot(x, activation, color=color, linewidth=1.7)

        positions = activated_position_indices(protein)
        if positions:
            scatter_x = [pos for pos in positions if 1 <= pos <= len(activation)]
            scatter_y = [float(activation[pos - 1]) for pos in scatter_x]
            ax.scatter(scatter_x, scatter_y, color=PALETTE["accent_dark"], s=18, zorder=3)

        ax.set_ylim(0, ymax)
        if idx == 0:
            ax.set_ylabel("SAE activation")
        else:
            ax.set_ylabel("")
        ax.set_title(protein["accession"], color=to_hex(color), fontsize=10.8, pad=6)
        ax.tick_params(labelbottom=False)
        polish_axes(ax, grid_axis="x")
        sns.despine(ax=ax)

        ax_geom = fig.add_subplot(grid[1, idx], sharex=ax)
        geom_prob = np.asarray(protein.get("geom_prob_profile", []), dtype=float)
        gx = np.arange(1, len(geom_prob) + 1)
        ax_geom.fill_between(gx, geom_prob, color=color, alpha=0.18, linewidth=0)
        ax_geom.plot(gx, geom_prob, color=color, linewidth=1.7)
        if positions:
            geom_scatter_x = [pos for pos in positions if 1 <= pos <= len(geom_prob)]
            geom_scatter_y = [float(geom_prob[pos - 1]) for pos in geom_scatter_x]
            ax_geom.scatter(geom_scatter_x, geom_scatter_y, color=PALETTE["accent_dark"], s=18, zorder=3)
        ax_geom.set_ylim(0, geom_ymax)
        ax_geom.set_xlabel("Residue position", labelpad=8)
        if idx == 0:
            ax_geom.set_ylabel("Geometry\nprobability")
        else:
            ax_geom.set_ylabel("")
        polish_axes(ax_geom, grid_axis="x")
        sns.despine(ax=ax_geom)

    ax_blank_top = fig.add_subplot(grid[0, -1])
    ax_blank_top.set_axis_off()
    ax_profile = fig.add_subplot(grid[:, -1], projection="polar")
    plot_geometry_profile_dial(ax_profile, feature_importances)

    fig.suptitle(
        f"f/{fid}: representative proteins with shared local activation",
        fontsize=12,
        x=0.01,
        ha="left",
        y=0.985,
    )
    return save_pdf(fig, feature_dir / "representative_protein_panel.pdf")


def save_cross_representative_structure_cards(
    data: FigureData,
    fid: int,
    feature_dir: Path,
) -> dict[str, str]:
    geometry = data.geometry_json(fid)
    proteins = select_cross_representative_proteins(geometry, max_proteins=3)
    if not proteins:
        return {}

    palette = sns.color_palette("colorblind", n_colors=len(proteins))
    out_dir = feature_dir / "structure_cards"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    for color, protein in zip(palette, proteins):
        accession = protein["accession"]
        out_path = out_dir / f"representative_structure_{accession}.png"
        rendered = render_cross_structure_panel(
            accession,
            protein.get("sae_activation_profile", []),
            to_hex(color),
            out_path,
        )
        if not rendered:
            legacy_asset = feature_dir / "_render_assets" / f"representative_structure_{accession}.png"
            if legacy_asset.exists():
                shutil.copy2(legacy_asset, out_path)
                rendered = True

        if rendered and out_path.exists():
            normalize_structure_card_border(out_path, to_hex(color))
            mirror_generated_asset(out_path)
            outputs[accession] = str(out_path)
    return outputs


@lru_cache(maxsize=1)
def sequence_identity_aligner() -> Any | None:
    if PairwiseAligner is None:
        return None
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.5
    return aligner


def global_sequence_identity(seq_a: str, seq_b: str) -> float:
    if not seq_a or not seq_b:
        return 0.0

    aligner = sequence_identity_aligner()
    if aligner is None:
        overlap = max(len(seq_a), len(seq_b), 1)
        matches = sum(1 for a, b in zip(seq_a, seq_b) if a == b)
        return 100.0 * matches / overlap

    alignment = aligner.align(seq_a, seq_b)[0]
    coords = np.asarray(alignment.coordinates, dtype=int)
    matches = 0
    for idx in range(coords.shape[1] - 1):
        a0, a1 = coords[0, idx], coords[0, idx + 1]
        b0, b1 = coords[1, idx], coords[1, idx + 1]
        if (a1 - a0) <= 0 or (b1 - b0) <= 0:
            continue
        span = min(a1 - a0, b1 - b0)
        for offset in range(span):
            if seq_a[a0 + offset] == seq_b[b0 + offset]:
                matches += 1
    return 100.0 * matches / max(len(seq_a), len(seq_b), 1)


def representative_identity_matrix(proteins: list[dict[str, Any]]) -> np.ndarray:
    n = len(proteins)
    matrix = np.zeros((n, n), dtype=float)
    sequences = [protein.get("sequence", "") for protein in proteins]
    for i in range(n):
        matrix[i, i] = 100.0
        for j in range(i + 1, n):
            identity = global_sequence_identity(sequences[i], sequences[j])
            matrix[i, j] = identity
            matrix[j, i] = identity
    return matrix


def plot_cross_feature_identity_context(
    data: FigureData,
    fid: int,
    feature_dir: Path,
    feature_row: dict[str, Any],
) -> str | None:
    geometry = data.geometry_json(fid)
    proteins = select_cross_representative_proteins(geometry, max_proteins=3)
    if len(proteins) < 2:
        return None

    matrix = representative_identity_matrix(proteins)
    labels = [
        f"{protein['accession']}\n{len(protein.get('sequence', ''))} aa"
        for protein in proteins
    ]
    off_diag = matrix[np.triu_indices_from(matrix, k=1)]

    families = sorted(
        feature_row.get("interpro_families", []),
        key=lambda entry: entry.get("f1", 0.0),
        reverse=True,
    )[:5]

    fig = plt.figure(figsize=(8.9, 4.7))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.34)

    ax_heat = fig.add_subplot(grid[0, 0])
    cmap = sns.light_palette(PALETTE["primary"], as_cmap=True)
    sns.heatmap(
        matrix,
        ax=ax_heat,
        cmap=cmap,
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        square=True,
        linewidths=1.0,
        linecolor="white",
        cbar_kws={"label": "% identity", "shrink": 0.84},
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"fontsize": 8.7, "color": PALETTE["ink"]},
    )
    ax_heat.set_title("Pairwise global sequence identity", fontsize=11, pad=8)
    ax_heat.tick_params(axis="x", labelrotation=0, labelsize=8.4, pad=4)
    ax_heat.tick_params(axis="y", labelrotation=0, labelsize=8.4)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    min_identity = float(np.nanmin(off_diag)) if off_diag.size else 100.0
    max_identity = float(np.nanmax(off_diag)) if off_diag.size else 100.0
    ax_heat.text(
        0.0,
        -0.21,
        f"Range across representatives: {min_identity:.0f}-{max_identity:.0f}% identity",
        transform=ax_heat.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=PALETTE["muted"],
    )

    ax_bar = fig.add_subplot(grid[0, 1])
    if families:
        labels_bar = [
            wrap(f"{entry.get('code', '')}: {entry.get('name', '')}", 30)
            for entry in families
        ]
        values = [float(entry.get("f1", 0.0)) for entry in families]
        colors = [PALETTE["secondary"]] + [PALETTE["secondary_light"]] * (len(values) - 1)
        y = np.arange(len(values))
        ax_bar.barh(y, values, color=colors, edgecolor="none")
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(labels_bar, fontsize=8.6)
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, max(0.7, max(values) * 1.10))
        ax_bar.set_xlabel("Protein-level F1")
        ax_bar.set_title("Top coarse InterPro contexts", fontsize=11, pad=8)
        counts = (
            f"{int(feature_row.get('n_families_above_03', 0))} families >= 0.3 F1\n"
            f"{int(feature_row.get('n_families_above_05', 0))} families >= 0.5 F1"
        )
        ax_bar.text(
            0.98,
            0.98,
            counts,
            transform=ax_bar.transAxes,
            ha="right",
            va="top",
            fontsize=8.3,
            color=PALETTE["muted"],
        )
        polish_axes(ax_bar, grid_axis="x")
        sns.despine(ax=ax_bar, left=False, bottom=False)
    else:
        ax_bar.text(
            0.5,
            0.5,
            "No InterPro family breakdown available",
            ha="center",
            va="center",
            fontsize=10,
            color=PALETTE["muted"],
            transform=ax_bar.transAxes,
        )
        ax_bar.set_axis_off()

    fig.suptitle(
        f"f/{fid}: sequence dissimilarity and coarse family context",
        fontsize=12,
        x=0.01,
        ha="left",
        y=0.99,
    )
    fig.subplots_adjust(bottom=0.20)
    return save_pdf(fig, feature_dir / "representative_identity_context.pdf")


def plot_interpro_family_breakdown(feature_row: dict[str, Any], feature_dir: Path) -> str:
    families = sorted(feature_row.get("interpro_families", []), key=lambda entry: entry.get("f1", 0.0), reverse=True)[:8]
    labels = [wrap(entry["name"], 26) for entry in families]
    y = np.arange(len(families))
    width = 0.23

    fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.5 * len(families) + 1.0)))
    ax.barh(y - width, [entry["f1"] for entry in families], height=width, color=PALETTE["primary"], label="F1")
    ax.barh(y, [entry["precision"] for entry in families], height=width, color=PALETTE["success"], label="Precision")
    ax.barh(y + width, [entry["recall"] for entry in families], height=width, color=PALETTE["accent"], label="Recall")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.02)
    ax.set_title(f"f/{feature_row['feature_id']}: InterPro family breakdown", fontsize=12, pad=8)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    polish_axes(ax, grid_axis="x")
    sns.despine(ax=ax, left=False, bottom=False)
    return save_pdf(fig, feature_dir / "interpro_family_breakdown.pdf")


def plot_cross_feature_importances(data: FigureData, fid: int, feature_dir: Path, feature_row: dict[str, Any]) -> str:
    geometry = data.geometry_json(fid)
    importances = geometry["geometric_residue_level"].get("feature_importances", {})
    top_items = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:12]
    labels = [wrap(name, 22) for name, _ in reversed(top_items)]
    values = [value for _, value in reversed(top_items)]

    fig, ax = plt.subplots(figsize=(6.4, max(3.6, 0.35 * len(top_items) + 1.0)))
    ax.barh(labels, values, color=PALETTE["primary"])
    ax.set_xlabel("Importance")
    ax.set_title(f"f/{fid}: top geometry descriptors", fontsize=12, pad=8)
    polish_axes(ax, grid_axis="x")
    sns.despine(ax=ax, left=False, bottom=False)
    return save_pdf(fig, feature_dir / "geometry_importances.pdf")


GEOMETRY_PROFILE_LABELS = [
    "Curvature",
    "Torsion",
    "Planarity",
    "Alignment",
    "Compactness",
    "Contacts",
    "Composition",
]


def aggregate_geometry_profile(importances: dict[str, float]) -> dict[str, float]:
    totals = {label: 0.0 for label in GEOMETRY_PROFILE_LABELS}
    for name, value in importances.items():
        lname = name.lower()
        if "contact" in lname:
            totals["Contacts"] += float(value)
        elif "end_to_end_ratio" in lname or "spatial_dist" in lname or "compact" in lname:
            totals["Compactness"] += float(value)
        elif "tangent_alignment" in lname or lname.startswith("tangent"):
            totals["Alignment"] += float(value)
        elif "plan" in lname:
            totals["Planarity"] += float(value)
        elif "tors" in lname:
            totals["Torsion"] += float(value)
        elif "curv" in lname or "curvature" in lname:
            totals["Curvature"] += float(value)
        elif lname.startswith("frac_"):
            totals["Composition"] += float(value)
        else:
            totals["Compactness"] += float(value)

    total_mass = sum(totals.values())
    if total_mass <= 0:
        return totals
    return {label: value / total_mass for label, value in totals.items()}


def plot_geometry_profile_dial(
    ax: plt.Axes,
    importances: dict[str, float],
) -> None:
    box = ax.get_position()
    ax.set_position([
        box.x0 + box.width * 0.08,
        box.y0 + box.height * 0.10,
        box.width * 0.80,
        box.height * 0.80,
    ])

    profile = aggregate_geometry_profile(importances)
    labels = GEOMETRY_PROFILE_LABELS
    values = np.array([profile[label] for label in labels], dtype=float)
    theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    theta_closed = np.concatenate([theta, theta[:1]])
    values_closed = np.concatenate([values, values[:1]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(theta_closed, values_closed, color=PALETTE["primary"], linewidth=1.9)
    ax.fill(theta_closed, values_closed, color=PALETTE["primary_light"], alpha=0.24)

    tick_labels = [
        "Curvature",
        "Torsion",
        "Planarity",
        "Alignment",
        "Compact.",
        "Contacts",
        "Comp.",
    ]
    ax.set_xticks(theta)
    ax.set_xticklabels(tick_labels, fontsize=7.6, color=PALETTE["ink"])
    ax.tick_params(axis="x", pad=10)
    ax.set_ylim(0, max(0.55, float(np.nanmax(values)) * 1.18 if values.size else 0.55))
    ax.set_yticks([0.15, 0.30, 0.45])
    ax.set_yticklabels(["0.15", "0.30", "0.45"], fontsize=6.8, color=PALETTE["muted"])
    ax.grid(color=PALETTE["grid"], linewidth=0.8, alpha=0.8)
    ax.spines["polar"].set_color(PALETTE["grid"])
    ax.spines["polar"].set_linewidth(1.0)
    ax.set_title("")
    ax.text(
        0.5,
        1.02,
        "Geometric profile",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.0,
        color=PALETTE["ink"],
        fontweight="semibold",
    )

    top_labels = sorted(profile.items(), key=lambda item: item[1], reverse=True)[:2]
    top_text = " + ".join(f"{label} {value:.0%}" for label, value in top_labels if value > 0)
    if top_text:
        ax.text(
            0.5,
            0.95,
            top_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.6,
            color=PALETTE["muted"],
        )


def plot_cross_feature_flexibility(data: FigureData, fid: int, feature_dir: Path) -> str:
    geometry = data.geometry_json(fid)
    motif = geometry["geometric_residue_level"]["motif_superposition"]
    flexibility = motif.get("per_position_flexibility", [])
    x = np.arange(1, len(flexibility) + 1)

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(x, flexibility, color=PALETTE["accent"], linewidth=1.8)
    ax.fill_between(x, flexibility, color="#FECACA", alpha=0.5)
    ax.set_xlabel("Motif position")
    ax.set_ylabel("Per-position flexibility (A)")
    ax.set_title(
        f"f/{fid}: motif flexibility (mean RMSD {motif.get('mean_rmsd', math.nan):.2f} A over {len(flexibility)} residues)",
        fontsize=12,
        pad=8,
    )
    polish_axes(ax, grid_axis="y")
    sns.despine(ax=ax)
    return save_pdf(fig, feature_dir / "motif_flexibility.pdf")


def clean_trace(values: list[Any]) -> np.ndarray:
    arr = np.array([np.nan if value is None else float(value) for value in values], dtype=float)
    return arr


def plot_cross_feature_top_protein_overlay(data: FigureData, fid: int, feature_dir: Path) -> str | None:
    geometry = data.geometry_json(fid)
    top_proteins = geometry.get("plot_data", {}).get("top_proteins", [])
    if not top_proteins:
        return None

    protein = top_proteins[0]
    accession = protein["accession"]
    activation = np.array(protein.get("sae_activation_profile", []), dtype=float)
    geom_prob = np.array(protein.get("geom_prob_profile", []), dtype=float)
    x = np.arange(1, len(activation) + 1)

    fig, ax1 = plt.subplots(figsize=(7.4, 3.8))
    ax1.fill_between(x, activation, color=PALETTE["secondary_light"], alpha=0.4, linewidth=0)
    ax1.plot(x, activation, color=PALETTE["secondary_dark"], linewidth=1.5, label="SAE activation")
    ax1.set_xlabel("Residue position")
    ax1.set_ylabel("SAE activation", color=PALETTE["secondary_dark"])
    ax1.tick_params(axis="y", colors=PALETTE["secondary_dark"])

    ax2 = ax1.twinx()
    ax2.plot(x, geom_prob, color=PALETTE["primary"], linewidth=1.6, label="Geometry probability")
    ax2.set_ylabel("Geometry probability", color=PALETTE["primary"])
    ax2.tick_params(axis="y", colors=PALETTE["primary"])

    activated_positions = protein.get("activated_positions", [])
    if activated_positions:
        scatter_x = []
        scatter_y = []
        for item in activated_positions:
            if isinstance(item, dict):
                pos = item.get("position")
                y_val = item.get("activation")
            else:
                pos = item
                y_val = None
            if pos is None or pos < 0 or pos >= len(activation):
                continue
            scatter_x.append(pos + 1)
            scatter_y.append(float(y_val) if y_val is not None else float(activation[pos]))
        ax1.scatter(
            scatter_x,
            scatter_y,
            color=PALETTE["accent_dark"],
            s=18,
            zorder=3,
        )

    ax1.set_title(f"f/{fid}: top protein activation overlay ({accession})", fontsize=12, pad=8)
    polish_axes(ax1, grid_axis="x")
    return save_pdf(fig, feature_dir / f"top_protein_overlay_{accession}.pdf")


def plot_cross_feature_descriptor_traces(data: FigureData, fid: int, feature_dir: Path) -> str | None:
    geometry = data.geometry_json(fid)
    top_proteins = geometry.get("plot_data", {}).get("top_proteins", [])
    if not top_proteins:
        return None

    protein = top_proteins[0]
    traces = protein.get("top_feature_traces") or {}
    if not traces:
        return None

    cleaned = {name: clean_trace(values) for name, values in traces.items()}
    valid_names = [name for name, arr in cleaned.items() if np.isfinite(arr).sum() > 0]
    if not valid_names:
        return None

    selected_names = valid_names[:2]
    activation = np.array(protein.get("sae_activation_profile", []), dtype=float)
    x = np.arange(1, len(activation) + 1)

    def normalize(arr: np.ndarray) -> np.ndarray:
        finite = np.isfinite(arr)
        if not finite.any():
            return np.full_like(arr, np.nan)
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)
        if math.isclose(min_val, max_val):
            out = np.zeros_like(arr)
            out[~finite] = np.nan
            return out
        return (arr - min_val) / (max_val - min_val)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(x, normalize(activation), color=PALETTE["secondary_dark"], linewidth=1.8, label="SAE activation (norm)")
    palette = sns.color_palette("colorblind", n_colors=len(selected_names))
    for color, name in zip(palette, selected_names):
        ax.plot(x, normalize(cleaned[name]), color=color, linewidth=1.5, label=f"{name} (norm)")
    ax.set_xlabel("Residue position")
    ax.set_ylabel("Normalized value")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"f/{fid}: descriptor traces on top protein ({protein['accession']})", fontsize=12, pad=8)
    ax.legend(loc="upper right", fontsize=8)
    polish_axes(ax, grid_axis="x")
    sns.despine(ax=ax)
    return save_pdf(fig, feature_dir / f"descriptor_traces_{protein['accession']}.pdf")


def format_metric_value(value: float | None, kind: str = "score") -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    if kind == "percent":
        return f"{value:.2f}%"
    if kind == "rmsd":
        return f"{value:.2f} A"
    if kind == "count":
        return f"{int(value)}"
    return f"{value:.3f}"


def draw_metric_chip(ax: plt.Axes, x: float, y: float, label: str, value: str, color: str) -> None:
    ax.text(
        x,
        y,
        f"{label}\n{value}",
        ha="left",
        va="top",
        fontsize=9.2,
        color=PALETTE["ink"],
        linespacing=1.2,
        bbox={
            "boxstyle": "round,pad=0.38,rounding_size=0.14",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 1.1,
        },
    )


def plot_residue_annotatable_spotlight(
    data: FigureData,
    feature_row: dict[str, Any],
    feature_dir: Path,
) -> str | None:
    fid = feature_row["feature_id"]
    mode = str(feature_row.get("selection_mode", "") or "geometry")
    geometry = data.geometry_json(fid)
    top_proteins = geometry.get("plot_data", {}).get("top_proteins", [])
    if not top_proteins:
        return None

    protein = top_proteins[0]
    accession = protein.get("accession")
    activation = clean_trace(protein.get("sae_activation_profile", []))
    geom_prob = clean_trace(protein.get("geom_prob_profile", []))
    if activation.size == 0 or geom_prob.size == 0:
        return None

    seq_len = len(activation)
    x = np.arange(1, seq_len + 1)
    motif = geometry.get("geometric_residue_level", {}).get("motif_superposition", {})
    feature_importances = geometry.get("geometric_residue_level", {}).get("feature_importances", {})
    motif_pdb = motif.get("mean_structure_pdb", "")

    asset_path = feature_dir / "_render_assets" / f"spotlight_pair_f{fid}.png"
    rendered_pair = render_geometry_dominant_pair(accession, activation.tolist(), motif_pdb, asset_path)

    fig = plt.figure(figsize=(8.4, 8.55 if rendered_pair else 6.3))
    if rendered_pair:
        grid = fig.add_gridspec(5, 1, height_ratios=[0.74, 1.0, 1.0, 0.26, 1.55], hspace=0.18)
    else:
        grid = fig.add_gridspec(3, 1, height_ratios=[0.74, 1.0, 1.0], hspace=0.26)

    ax_stats = fig.add_subplot(grid[0, 0])
    ax_stats.set_axis_off()

    title = spotlight_title(feature_row)
    ax_stats.text(
        0.0,
        1.03,
        f"f/{fid}: {title}",
        fontsize=12.4,
        fontweight="semibold",
        color=PALETTE["ink"],
        ha="left",
        va="top",
        transform=ax_stats.transAxes,
    )
    ax_stats.text(
        0.0,
        0.80,
        f"Category: {feature_row.get('structural_category', 'n/a')}",
        fontsize=9.4,
        color=PALETTE["muted"],
        ha="left",
        va="top",
        transform=ax_stats.transAxes,
    )
    ax_stats.text(
        0.0,
        0.63,
        f"Selected by: {residue_annotatable_mode_label(feature_row.get('selection_mode'))}",
        fontsize=9.4,
        color=PALETTE["muted"],
        ha="left",
        va="top",
        transform=ax_stats.transAxes,
    )
    detail_line = spotlight_detail_line(feature_row)
    if detail_line:
        ax_stats.text(
            0.0,
            0.46,
            textwrap.shorten(detail_line, width=95, placeholder="..."),
            fontsize=9.1,
            color=PALETTE["muted"],
            ha="left",
            va="top",
            transform=ax_stats.transAxes,
        )

    chip_positions = [0.0, 0.18, 0.36, 0.54, 0.72, 0.90]
    chip_specs = [
        ("Geom PR-AUC", format_metric_value(feature_row.get("geom_pr_auc")), PALETTE["primary"]),
        (motif_metric_label(feature_row), format_metric_value(feature_row.get("motif_f1")), PALETTE["secondary_dark"]),
        ("InterPro res F1", format_metric_value(feature_row.get("interpro_res_f1")), PALETTE["success"]),
        ("CATH res F1", format_metric_value(feature_row.get("cath_res_f1")), PALETTE["slate"]),
        ("Motif RMSD", format_metric_value(motif.get("mean_rmsd"), "rmsd"), PALETTE["accent"]),
        ("Coverage", format_metric_value(feature_row.get("coverage_pct"), "percent"), PALETTE["muted"]),
    ]
    for xpos, (label, value, color) in zip(chip_positions, chip_specs):
        draw_metric_chip(ax_stats, xpos, 0.22, label, value, color)

    activated_positions = activated_position_indices(protein)
    ax_act = fig.add_subplot(grid[1, 0])
    ax_act.plot(x, activation, color=PALETTE["secondary_dark"], linewidth=1.7)
    if activated_positions:
        scatter_x = [pos for pos in activated_positions if 1 <= pos <= len(activation)]
        scatter_y = [float(activation[pos - 1]) for pos in scatter_x]
        ax_act.scatter(scatter_x, scatter_y, color=PALETTE["accent_dark"], s=20, zorder=3)
    ax_act.set_xlim(1, seq_len)
    ax_act.set_ylabel("SAE activation")
    ax_act.set_title(f"Representative protein response ({accession})", fontsize=11, pad=6)
    polish_axes(ax_act, grid_axis="x")
    sns.despine(ax=ax_act)

    ax_geom = fig.add_subplot(grid[2, 0], sharex=ax_act)
    ax_geom.plot(x, geom_prob, color=PALETTE["primary"], linewidth=1.7)
    if activated_positions:
        scatter_x = [pos for pos in activated_positions if 1 <= pos <= len(geom_prob)]
        scatter_y = [float(geom_prob[pos - 1]) for pos in scatter_x]
        ax_geom.scatter(scatter_x, scatter_y, color=PALETTE["accent_dark"], s=20, zorder=3)
    ax_geom.set_xlim(1, seq_len)
    finite_geom = geom_prob[np.isfinite(geom_prob)]
    geom_max = float(np.nanmax(finite_geom)) if finite_geom.size else 1.0
    ax_geom.set_ylim(-0.02, max(1.02, geom_max * 1.06))
    ax_geom.set_xlabel("Residue position", labelpad=9)
    ax_geom.set_ylabel("Geometry\nprobability")
    ax_geom.set_title("Geometry classifier response", fontsize=11, pad=6)
    polish_axes(ax_geom, grid_axis="x")
    sns.despine(ax=ax_geom)

    if rendered_pair:
        ax_spacer = fig.add_subplot(grid[3, 0])
        ax_spacer.set_axis_off()
        bottom_grid = grid[4, 0].subgridspec(1, 2, width_ratios=[2.05, 1.0], wspace=0.12)
        ax_pair = fig.add_subplot(bottom_grid[0, 0])
        ax_pair.imshow(plt.imread(asset_path))
        ax_pair.set_axis_off()
        ax_pair.set_anchor("N")
        ax_profile = fig.add_subplot(bottom_grid[0, 1], projection="polar")
        plot_geometry_profile_dial(ax_profile, feature_importances)

    prefix = spotlight_output_prefix(mode)
    return save_pdf(fig, feature_dir / f"{prefix}_spotlight_{accession}.pdf")


def make_cross_feature_bundle(data: FigureData, feature_row: dict[str, Any], feature_dir: Path) -> dict[str, str]:
    fid = feature_row["feature_id"]
    outputs = {
        "interpro_family_breakdown": plot_interpro_family_breakdown(feature_row, feature_dir),
        "geometry_importances": plot_cross_feature_importances(data, fid, feature_dir, feature_row),
        "motif_flexibility": plot_cross_feature_flexibility(data, fid, feature_dir),
    }
    representatives = plot_cross_feature_representative_structures(data, fid, feature_dir)
    if representatives:
        outputs["representative_protein_panel"] = representatives
    structure_cards = save_cross_representative_structure_cards(data, fid, feature_dir)
    if structure_cards:
        outputs["representative_structure_cards"] = structure_cards
    identity_context = plot_cross_feature_identity_context(data, fid, feature_dir, feature_row)
    if identity_context:
        outputs["representative_identity_context"] = identity_context
    overlay = plot_cross_feature_top_protein_overlay(data, fid, feature_dir)
    if overlay:
        outputs["top_protein_overlay"] = overlay
    traces = plot_cross_feature_descriptor_traces(data, fid, feature_dir)
    if traces:
        outputs["descriptor_traces"] = traces
    return outputs


def make_spotlight_bundle(data: FigureData, feature_row: dict[str, Any], feature_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    spotlight = plot_residue_annotatable_spotlight(data, feature_row, feature_dir)
    if spotlight:
        outputs["spotlight"] = spotlight
    return outputs


def main() -> None:
    global RENDER_CONFIG

    args = parse_args()
    set_paper_style()

    data = FigureData(args.data_dir.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    RENDER_CONFIG = RenderConfig(
        base_output_dir=output_dir,
        export_clean_variants=not args.no_clean_variants,
        clean_root_name=args.clean_root_name,
        skip_structure_renders=args.skip_structure_renders,
    )

    manifest: dict[str, Any] = {
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(output_dir),
        "panel_ready_root": (
            str(output_dir / args.clean_root_name)
            if not args.no_clean_variants
            else None
        ),
        "overview": [],
        "case_studies": {},
        "cross_family": {},
        "geometry_dominant": {},
        "sequence_dominant": {},
        "biology_dominant": {},
        "defaults": {
            "case_family_codes": DEFAULT_CASE_FAMILY_CODES,
            "case_family_codes_by_group": {
                spec["key"]: list(spec["default_codes"])
                for spec in CASE_STUDY_GROUP_SPECS
            },
            "cross_feature_ids": DEFAULT_CROSS_FEATURE_IDS,
            "geometry_dominant_feature_ids": DEFAULT_GEOMETRY_DOMINANT_FEATURE_IDS,
            "sequence_dominant_feature_ids": DEFAULT_SEQUENCE_DOMINANT_FEATURE_IDS,
            "biology_dominant_feature_ids": DEFAULT_BIOLOGY_DOMINANT_FEATURE_IDS,
        },
        "render_options": {
            "export_clean_variants": not args.no_clean_variants,
            "clean_root_name": args.clean_root_name,
            "skip_structure_renders": args.skip_structure_renders,
            "skip_overview": args.skip_overview,
            "skip_case_studies": args.skip_case_studies,
            "skip_cross_family": args.skip_cross_family,
            "max_case_summary_families": args.max_case_summary_families,
            "max_geometry_dominant_features": args.max_geometry_dominant_features,
            "max_sequence_dominant_features": args.max_sequence_dominant_features,
            "max_biology_dominant_features": args.max_biology_dominant_features,
        },
    }

    case_bundle_count = 0
    if not args.skip_overview:
        manifest["overview"] = make_overview_plots(
            data,
            output_dir,
            max_case_summary_families=args.max_case_summary_families,
        )

    if not args.skip_case_studies:
        case_root = output_dir / CASE_DIRNAME
        family_sets = data.case_studies.get("family_sets", {})
        if family_sets:
            for spec in CASE_STUDY_GROUP_SPECS:
                selected = select_case_families_for_set(
                    data,
                    args,
                    family_set_key=spec["key"],
                    default_codes=spec["default_codes"],
                )
                if not selected:
                    continue

                payload = case_family_set_payload(data, spec["key"]) or {}
                group_manifest: dict[str, Any] = {
                    "label": spec["label"],
                    "directory": str(case_root / spec["dirname"]),
                    "n_available_families": len(payload.get("families", [])),
                    "selection": payload.get("selection"),
                    "families": {},
                }
                manifest["case_studies"][spec["key"]] = group_manifest

                for rank, family in enumerate(selected, start=1):
                    family_context = dict(family)
                    family_context["case_study_group_key"] = spec["key"]
                    family_context["member_label_mode"] = (
                        "sequence_motif"
                        if spec["key"] == "sequence_motif_primary_dominant"
                        else "geometry"
                    )
                    family_slug = slugify(family["annotation_name"])
                    family_dir = case_root / spec["dirname"] / f"{rank:02d}_{family['annotation_code'].lower()}_{family_slug}"
                    outputs = make_case_family_bundle(data, family_context, family_dir)
                    plotted_family = prepare_case_study_family_for_plotting(data, family_context)
                    group_manifest["families"][family["annotation_code"]] = {
                        "annotation_name": family["annotation_name"],
                        "n_nodes": family["n_nodes"],
                        "plotted_n_nodes": len(plotted_family["members"]),
                        "plotted_feature_ids": [member["feature_id"] for member in plotted_family["members"]],
                        "directory": str(family_dir),
                        "plots": outputs,
                    }
                    case_bundle_count += 1
        else:
            case_families = select_case_families(data, args)
            for rank, family in enumerate(case_families, start=1):
                family_context = dict(family)
                family_context["member_label_mode"] = "geometry"
                family_slug = slugify(family["annotation_name"])
                family_dir = case_root / f"{rank:02d}_{family['annotation_code'].lower()}_{family_slug}"
                outputs = make_case_family_bundle(data, family_context, family_dir)
                plotted_family = prepare_case_study_family_for_plotting(data, family_context)
                manifest["case_studies"][family["annotation_code"]] = {
                    "annotation_name": family["annotation_name"],
                    "n_nodes": family["n_nodes"],
                    "plotted_n_nodes": len(plotted_family["members"]),
                    "plotted_feature_ids": [member["feature_id"] for member in plotted_family["members"]],
                    "directory": str(family_dir),
                    "plots": outputs,
                }
                case_bundle_count += 1

    if not args.skip_cross_family:
        cross_features = select_cross_features(data, args)
        cross_root = output_dir / CROSS_DIRNAME
        for rank, feature_row in enumerate(cross_features, start=1):
            fid = feature_row["feature_id"]
            feature_slug = slugify(feature_row.get("best_interpro_protein_name") or feature_row.get("structural_category") or f"feature_{fid}")
            feature_dir = cross_root / f"{rank:02d}_feature_{fid}_{feature_slug}"
            outputs = make_cross_feature_bundle(data, feature_row, feature_dir)
            manifest["cross_family"][str(fid)] = {
                "best_interpro_protein_name": feature_row.get("best_interpro_protein_name"),
                "structural_category": feature_row.get("structural_category"),
                "directory": str(feature_dir),
                "plots": outputs,
            }

    spotlight_counts: dict[str, int] = {}
    for spec in SPOTLIGHT_MODE_SPECS:
        selected = select_spotlight_features(data, args, mode=spec["mode"])
        spotlight_root = output_dir / spec["dirname"]
        for rank, feature_row in enumerate(selected, start=1):
            fid = feature_row["feature_id"]
            feature_slug = slugify(
                feature_row.get("biology_label")
                or feature_row.get("interpro_protein_name")
                or feature_row.get("structural_category")
                or f"feature_{fid}"
            )
            feature_dir = spotlight_root / f"{rank:02d}_feature_{fid}_{feature_slug}"
            outputs = make_spotlight_bundle(data, feature_row, feature_dir)
            manifest[spec["manifest_key"]][str(fid)] = {
                "best_interpro_protein_name": feature_row.get("interpro_protein_name"),
                "structural_category": feature_row.get("structural_category"),
                "selection_mode": feature_row.get("selection_mode"),
                "sequence_motif_consensus": feature_row.get("sequence_motif_consensus"),
                "biology_source": feature_row.get("biology_source"),
                "biology_code": feature_row.get("biology_code"),
                "biology_label": feature_row.get("biology_label"),
                "directory": str(feature_dir),
                "plots": outputs,
                "spotlight_score": feature_row.get("spotlight_score"),
            }
        spotlight_counts[spec["manifest_key"]] = len(selected)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote overview plots: {len(manifest['overview'])}")
    print(f"Wrote case-study bundles: {case_bundle_count}")
    print(f"Wrote cross-family bundles: {len(manifest['cross_family'])}")
    print(f"Wrote geometry-dominant bundles: {spotlight_counts.get('geometry_dominant', 0)}")
    print(f"Wrote sequence-dominant bundles: {spotlight_counts.get('sequence_dominant', 0)}")
    print(f"Wrote biology-dominant bundles: {spotlight_counts.get('biology_dominant', 0)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
