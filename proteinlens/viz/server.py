"""
FastAPI server for the GeoPedia SAE feature visualizer.

Multi-layer mode (recommended)::

    python -m proteinlens.viz.server --analysis-root analysis --port 8050

``--analysis-root`` should contain one subdirectory per layer (the subdir name
must match ``l<N>`` or ``layer_<N>``, e.g. ``l2``, ``l4``, ``l6``). Each
subdirectory is treated as a full SAE analysis dir (the same shape that
``--analysis-dir`` consumes in single-layer mode).

Single-layer mode (back-compat)::

    python -m proteinlens.viz.server \\
        --analysis-dir trained_models/layer_4/frosty-sweep-15/analysis \\
        --port 8050

In single-layer mode, the layer number is read from
``dataset_stats.json["esm_layer"]``. The same routes work; the SPA notices
only one layer is registered and adapts.

Startup sequence:
  1. Parse CLI args.
  2. Discover layer subdirs (root) or wrap the single dir.
  3. For each, run ``index_builder`` to merge enrichment summaries +
     permutation q-values into the in-memory feature index.
  4. Configure the API module with the per-layer state dict + a landing
     payload (per-layer summary stats + paper-described feature highlights).
  5. Mount static files (HTML/JS/CSS) and start uvicorn.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from proteinlens.viz import api
from proteinlens.viz.api import LayerState
from proteinlens.viz.index_builder import (
    Q_SIGNIFICANCE_THRESHOLD,
    build_feature_index,
    build_method_coverage,
    build_pipeline_status,
    build_stats,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

_LAYER_SUBDIR_RE = re.compile(r"^(?:l|layer_)(\d+)$", re.IGNORECASE)


# Paper-named features for the landing page. Numeric scores come from the
# real per-layer feature index at startup; only the curated paper metadata
# is hard-coded here.
_FEATURED_BLUEPRINT: list[dict[str, Any]] = [
    {"feature_id": 894, "layer": 4, "kind": "geom",
     "title": "Geometry-primary feature (Fig. 3)",
     "desc": ("f/894 — annotation only recoverable through the geometric "
              "pipeline. High GBM PR-AUC; no significant database / sequence "
              "match. The canonical geometry-primary case in the paper.")},
    {"feature_id": 4714, "layer": 4, "kind": "geom",
     "title": "Contact-related, ablation-causal (Fig. 5)",
     "desc": ("f/4714 — strong geometry-primary, contact-related feature used "
              "in the ablation experiment (linear response between ablation "
              "strength and contact-density target).")},
    {"feature_id": 6775, "layer": 4, "kind": "both",
     "title": "Histidine kinase / HSP90-like ATPase — variant A",
     "desc": ("f/6775 (Q15120). One of four layer-4 features sharing the "
              "HSP90-like ATPase residue annotation; geometry distinguishes "
              "it from f/5508, f/8254, f/9608.")},
    {"feature_id": 5508, "layer": 4, "kind": "both",
     "title": "HSP90-like ATPase — variant B",
     "desc": ("f/5508 (Q15120). Same residue label as f/6775 but lights up a "
              "different geometric sub-structure within the domain.")},
    {"feature_id": 8254, "layer": 4, "kind": "both",
     "title": "HSP90-like ATPase — variant C",
     "desc": ("f/8254 (Q15120). Third member of the HSP90-like ATPase cluster.")},
    {"feature_id": 9608, "layer": 4, "kind": "both",
     "title": "HSP90-like ATPase — variant D",
     "desc": ("f/9608 (Q15120). Fourth HSP90-like ATPase variant — biology "
              "says one label, geometry says four.")},
    {"feature_id": 8518, "layer": 4, "kind": "geom",
     "title": "Metagenomic-firing feature (Fig. 4)",
     "desc": ("f/8518 — geometry-significant on Swiss-Prot and fires on "
              "NMPfams metagenomic clusters. Bridges Swiss-Prot annotation "
              "onto unannotated sequences.")},
    {"feature_id": 670, "layer": 4, "kind": "both",
     "title": "Schematic figure feature (Fig. 1)",
     "desc": ("f/670 — the feature used in the paper's overview schematic to "
              "walk through the SAE → annotation pipeline.")},
]


def _is_sig(q: float | None) -> bool:
    return q is not None and q < Q_SIGNIFICANCE_THRESHOLD


def _load_one_layer(analysis_dir: Path) -> LayerState:
    """Build the in-memory state for a single analysis dir."""
    stats = build_stats(analysis_dir)
    pipeline_status = build_pipeline_status(analysis_dir)
    feature_index = build_feature_index(analysis_dir)
    method_coverage = build_method_coverage(feature_index)

    layer_num = stats.get("dataset", {}).get("esm_layer")
    if layer_num is None:
        raise SystemExit(
            f"dataset_stats.json missing 'esm_layer' field in {analysis_dir}"
        )

    logger.info(
        "Layer %d: %d features; %d annotated (%.1f%%) [from %s]",
        layer_num,
        len(feature_index),
        method_coverage["total_annotated_n"],
        method_coverage["total_annotated_pct"],
        analysis_dir,
    )
    return LayerState(
        layer=int(layer_num),
        analysis_dir=analysis_dir,
        stats=stats,
        feature_index=feature_index,
        pipeline_status=pipeline_status,
        method_coverage=method_coverage,
    )


def _discover_layers(root: Path) -> dict[int, Path]:
    """Find layer subdirs under ``root``: ``l2``, ``layer_4``, etc."""
    out: dict[int, Path] = {}
    if not root.is_dir():
        raise SystemExit(f"Analysis root not found: {root}")
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = _LAYER_SUBDIR_RE.match(child.name)
        if not m:
            continue
        layer = int(m.group(1))
        if not (child / "dataset_stats.json").exists():
            logger.warning("Skipping %s — no dataset_stats.json", child)
            continue
        out[layer] = child
    if not out:
        raise SystemExit(
            f"No layer subdirs found under {root} (expected names like 'l2', 'layer_4')"
        )
    return out


def _build_landing(layers: dict[int, LayerState]) -> dict[str, Any]:
    """Build the cross-layer landing payload from the per-layer states."""
    layer_summaries: list[dict[str, Any]] = []
    for L in sorted(layers):
        s = layers[L]
        idx = s.feature_index
        total = len(idx)

        def pct(k: int) -> float:
            n = sum(1 for r in idx if _is_sig(r.get(f"m{k}_q")))
            return round(100.0 * n / total, 2) if total else 0.0

        # Categories: geometry-only (only m7), bio-only (any of m1..m6 only),
        # both, or none.
        cats = {"geom_only": 0, "bio_only": 0, "both": 0, "none": 0}
        for r in idx:
            geom = _is_sig(r.get("m7_q"))
            bio = any(_is_sig(r.get(f"m{k}_q")) for k in range(1, 7))
            if geom and bio:
                cats["both"] += 1
            elif geom:
                cats["geom_only"] += 1
            elif bio:
                cats["bio_only"] += 1
            else:
                cats["none"] += 1

        any_n = total - cats["none"]
        bio_n = sum(1 for r in idx if any(_is_sig(r.get(f"m{k}_q")) for k in range(1, 7)))
        geom_n = sum(1 for r in idx if _is_sig(r.get("m7_q")))

        layer_summaries.append({
            "layer": L,
            "num_features": total,
            "novel_geom_count": cats["geom_only"],
            "any_pct": round(100.0 * any_n / total, 2) if total else 0.0,
            "bio_pct": round(100.0 * bio_n / total, 2) if total else 0.0,
            "geom_pct": round(100.0 * geom_n / total, 2) if total else 0.0,
            "both_pct": round(100.0 * cats["both"] / total, 2) if total else 0.0,
            # Per-method (paper Table 1 columns)
            "interpro_prot_pct": pct(1),
            "interpro_res_pct": pct(2),
            "cath_prot_pct": pct(3),
            "cath_res_pct": pct(4),
            "seq_pos_pct": pct(5),
            "seq_motif_pct": pct(6),
            "geometric_pct": pct(7),
            "total_annotated_pct": round(100.0 * any_n / total, 2) if total else 0.0,
            "category_counts": cats,
        })
    sae_cfg = layers[next(iter(layers))].stats.get("sae", {}) if layers else {}
    dataset_cfg = layers[next(iter(layers))].stats.get("dataset", {}) if layers else {}
    return {
        "layers": layer_summaries,
        "sae": sae_cfg,
        "dataset": {
            "total_proteins": dataset_cfg.get("total_proteins"),
            "total_clusters": dataset_cfg.get("total_clusters"),
            "esm_model": dataset_cfg.get("esm_model"),
        },
    }


def _build_featured(layers: dict[int, LayerState]) -> list[dict[str, Any]]:
    """Look up real per-feature scores for each paper-named feature."""
    out: list[dict[str, Any]] = []
    for entry in _FEATURED_BLUEPRINT:
        L = entry["layer"]
        s = layers.get(L)
        if s is None:
            continue
        match = s.feature_by_id.get(entry["feature_id"]) or next(
            (r for r in s.feature_index if r.get("feature_id") == entry["feature_id"]),
            None,
        )
        if match is None:
            logger.warning(
                "Featured feature L%s/f%s not in index", L, entry["feature_id"]
            )
            continue
        bio_scores = [match.get(f"m{k}_score") or 0.0 for k in range(1, 7)]
        out.append({
            **entry,
            "geom_score": match.get("m7_score"),
            "q_geom": match.get("m7_q"),
            "bio_best": max(bio_scores) if bio_scores else None,
            "pct_proteins": match.get("pct_proteins_activated"),
            "structural_category": match.get("structural_category"),
        })
    return out


def create_app(layer_states: dict[int, LayerState]) -> FastAPI:
    """
    Build the FastAPI app for one or more layers.

    The SPA (static/index.html) is served for every non-API, non-static
    GET, so client-side routing in the SPA "just works" for paths like
    ``/layer/4`` or ``/feature/123``.
    """
    app = FastAPI(title="GeoPedia — SAE feature atlas", docs_url=None, redoc_url=None)

    landing = _build_landing(layer_states)
    featured = _build_featured(layer_states)
    api.configure(layer_states, landing=landing, featured=featured)
    app.include_router(api.router)

    spa_index = STATIC_DIR / "index.html"

    @app.get("/", response_class=FileResponse)
    def root_page():
        return FileResponse(spa_index)

    # Static mount (CSS/JS/images). Mounted before the catch-all so it wins.
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # SPA catch-all for client-side routes. Anything that didn't match an API
    # route or /static gets the SPA shell, which reads window.location to
    # pick the initial view.
    @app.get("/{full_path:path}", response_class=FileResponse)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            # Should never reach here (FastAPI routing tries APIs first), but
            # if it does we don't want to serve the SPA for an unknown API path.
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Unknown API endpoint")
        return FileResponse(spa_index)

    return app


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. Either ``--analysis-root`` or ``--analysis-dir`` is required."""
    parser = argparse.ArgumentParser(
        description="GeoPedia — SAE feature atlas (FastAPI viz server)",
    )
    parser.add_argument(
        "--analysis-root",
        type=str,
        help=(
            "Path to a directory containing per-layer analysis subdirs "
            "(subdir names must match l<N> or layer_<N>, e.g. analysis/l2)."
        ),
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        help=(
            "Single-layer fallback: path to an SAE run's analysis directory "
            "(e.g., trained_models/layer_4/frosty-sweep-15/analysis/)."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Deprecated alias for --analysis-dir.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the server on (default: 8050)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    return parser.parse_args()


def _resolve_layers(args: argparse.Namespace) -> dict[int, LayerState]:
    """Pick the right input mode and return the layer-state dict."""
    if args.analysis_root:
        root = Path(args.analysis_root).resolve()
        layer_dirs = _discover_layers(root)
        logger.info("Discovered layers under %s: %s", root, sorted(layer_dirs.keys()))
        return {L: _load_one_layer(p) for L, p in sorted(layer_dirs.items())}

    raw = args.analysis_dir or args.data_dir
    if not raw:
        raise SystemExit(
            "Either --analysis-root <root with l<N>/ subdirs> or "
            "--analysis-dir <single layer's analysis dir> is required."
        )
    analysis_dir = Path(raw).resolve()
    if not analysis_dir.is_dir():
        raise SystemExit(f"Analysis directory not found: {analysis_dir}")
    if not (analysis_dir / "dataset_stats.json").exists():
        raise SystemExit(f"Not a valid analysis directory (no dataset_stats.json): {analysis_dir}")
    state = _load_one_layer(analysis_dir)
    return {state.layer: state}


def main() -> None:
    """Entry point for ``python -m proteinlens.viz.server``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    layer_states = _resolve_layers(args)
    app = create_app(layer_states)

    logger.info("Starting GeoPedia at http://%s:%d", args.host, args.port)
    logger.info("Layers loaded: %s", sorted(layer_states.keys()))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
