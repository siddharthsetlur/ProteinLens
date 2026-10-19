"""
FastAPI server for the GeoPedia SAE feature visualizer.

Multi-layer mode (recommended)::

    python -m proteinlens.viz.server --analysis-root analysis --port 8050

``--analysis-root`` should contain one subdirectory per SAE run. Each
subdirectory is treated as a full SAE analysis dir (the same shape that
``--analysis-dir`` consumes in single-layer mode):

  * ``l<N>`` / ``layer_<N>`` (e.g. ``l2``, ``l4``, ``l6``) are the paper's
    ESM2-8M layers. Their run id is the bare layer number, so URLs such as
    ``/layer/4`` and ``/feature/4/4714`` are unchanged.
  * Any other subdir with a ``dataset_stats.json`` (e.g. ``esm2-35m-l6``,
    ``amplify-120m-l12``) is a generalization model. Its run id is the subdir
    name, and its display label comes from ``dataset_stats.json``
    (``esm_model`` + ``esm_layer``).

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
_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# HF model id -> short display name.
_MODEL_NAMES = {
    "facebook/esm2_t6_8M_UR50D": "ESM2-8M",
    "facebook/esm2_t12_35M_UR50D": "ESM2-35M",
    "chandar-lab/AMPLIFY_120M": "AMPLIFY-120M",
}


# Paper-named features for the landing page. Numeric scores come from the
# real per-layer feature index at startup; only the curated paper metadata
# is hard-coded here.
_FEATURED_BLUEPRINT: list[dict[str, Any]] = [
    {"feature_id": 894, "layer": "4", "kind": "geom",
     "title": "Geometry-primary feature (Fig. 3)",
     "desc": ("f/894 — annotation only recoverable through the geometric "
              "pipeline. High GBM PR-AUC; no significant database / sequence "
              "match. The canonical geometry-primary case in the paper.")},
    {"feature_id": 4714, "layer": "4", "kind": "geom",
     "title": "Contact-related, ablation-causal (Fig. 5)",
     "desc": ("f/4714 — strong geometry-primary, contact-related feature used "
              "in the ablation experiment (linear response between ablation "
              "strength and contact-density target).")},
    {"feature_id": 6775, "layer": "4", "kind": "both",
     "title": "Histidine kinase / HSP90-like ATPase — variant A",
     "desc": ("f/6775 (Q15120). One of four layer-4 features sharing the "
              "HSP90-like ATPase residue annotation; geometry distinguishes "
              "it from f/5508, f/8254, f/9608.")},
    {"feature_id": 5508, "layer": "4", "kind": "both",
     "title": "HSP90-like ATPase — variant B",
     "desc": ("f/5508 (Q15120). Same residue label as f/6775 but lights up a "
              "different geometric sub-structure within the domain.")},
    {"feature_id": 8254, "layer": "4", "kind": "both",
     "title": "HSP90-like ATPase — variant C",
     "desc": ("f/8254 (Q15120). Third member of the HSP90-like ATPase cluster.")},
    {"feature_id": 9608, "layer": "4", "kind": "both",
     "title": "HSP90-like ATPase — variant D",
     "desc": ("f/9608 (Q15120). Fourth HSP90-like ATPase variant — biology "
              "says one label, geometry says four.")},
    {"feature_id": 8518, "layer": "4", "kind": "geom",
     "title": "Metagenomic-firing feature (Fig. 4)",
     "desc": ("f/8518 — geometry-significant on Swiss-Prot and fires on "
              "NMPfams metagenomic clusters. Bridges Swiss-Prot annotation "
              "onto unannotated sequences.")},
    {"feature_id": 670, "layer": "4", "kind": "both",
     "title": "Schematic figure feature (Fig. 1)",
     "desc": ("f/670 — the feature used in the paper's overview schematic to "
              "walk through the SAE → annotation pipeline.")},
]


def _is_sig(q: float | None) -> bool:
    return q is not None and q < Q_SIGNIFICANCE_THRESHOLD


def _load_one_layer(analysis_dir: Path, run_id: str | None, primary: bool) -> LayerState:
    """Build the in-memory state for a single analysis dir.

    ``run_id`` None means "use the bare layer number" (paper ESM2-8M runs and
    single-dir mode).
    """
    stats = build_stats(analysis_dir)
    pipeline_status = build_pipeline_status(analysis_dir)
    feature_index = build_feature_index(analysis_dir)
    method_coverage = build_method_coverage(feature_index)

    layer_num = stats.get("dataset", {}).get("esm_layer")
    if layer_num is None:
        raise SystemExit(
            f"dataset_stats.json missing 'esm_layer' field in {analysis_dir}"
        )

    model_id = stats.get("dataset", {}).get("esm_model") or ""
    model = _MODEL_NAMES.get(model_id, model_id.split("/")[-1])
    layer_num = int(layer_num)
    if run_id is None:
        run_id = str(layer_num)
    if primary:
        label, short = f"Layer {layer_num}", f"L{layer_num}"
    else:
        label, short = f"{model} · Layer {layer_num}", f"{model} L{layer_num}"

    logger.info(
        "%s: %d features; %d annotated (%.1f%%) [from %s]",
        label,
        len(feature_index),
        method_coverage["total_annotated_n"],
        method_coverage["total_annotated_pct"],
        analysis_dir,
    )
    return LayerState(
        layer=run_id,
        label=label,
        short=short,
        model=model,
        esm_layer=layer_num,
        primary=primary,
        analysis_dir=analysis_dir,
        stats=stats,
        feature_index=feature_index,
        pipeline_status=pipeline_status,
        method_coverage=method_coverage,
    )


def _discover_layers(root: Path) -> list[tuple[str | None, Path, bool]]:
    """Find run subdirs under ``root``, in display order.

    Returns ``(run_id, path, primary)`` tuples: the paper's ``l<N>`` /
    ``layer_<N>`` dirs first (by layer number, run_id None = bare layer
    number), then every other subdir by name, keyed by that name.
    """
    if not root.is_dir():
        raise SystemExit(f"Analysis root not found: {root}")
    paper: list[tuple[int, Path]] = []
    other: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "dataset_stats.json").exists():
            continue
        m = _LAYER_SUBDIR_RE.match(child.name)
        if m:
            paper.append((int(m.group(1)), child))
        elif _RUN_ID_RE.match(child.name):
            other.append((child.name, child))
        else:
            logger.warning("Skipping %s — name is not a valid run id", child)
    if not paper and not other:
        raise SystemExit(
            f"No analysis subdirs found under {root} (expected names like 'l2', "
            "'layer_4' or 'esm2-35m-l6', each with a dataset_stats.json)"
        )
    return ([(None, p, True) for _, p in sorted(paper)]
            + [(name, p, False) for name, p in other])


def _build_landing(layers: dict[str, LayerState]) -> dict[str, Any]:
    """Build the cross-layer landing payload from the per-layer states."""
    layer_summaries: list[dict[str, Any]] = []
    for L, s in layers.items():
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
            "label": s.label,
            "short": s.short,
            "model": s.model,
            "esm_layer": s.esm_layer,
            "primary": s.primary,
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


def _build_featured(layers: dict[str, LayerState]) -> list[dict[str, Any]]:
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


def create_app(layer_states: dict[str, LayerState]) -> FastAPI:
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


def _resolve_layers(args: argparse.Namespace) -> dict[str, LayerState]:
    """Pick the right input mode and return the layer-state dict."""
    if args.analysis_root:
        root = Path(args.analysis_root).resolve()
        runs = _discover_layers(root)
        logger.info("Discovered runs under %s: %s", root, [p.name for _, p, _ in runs])
        states = [_load_one_layer(p, run_id, primary) for run_id, p, primary in runs]
        out = {s.layer: s for s in states}
        if len(out) != len(states):
            raise SystemExit(f"Duplicate run ids under {root}: {[s.layer for s in states]}")
        return out

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
    state = _load_one_layer(analysis_dir, None, True)
    return {state.layer: state}


def main() -> None:
    """Entry point for ``python -m proteinlens.viz.server``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    layer_states = _resolve_layers(args)
    app = create_app(layer_states)

    logger.info("Starting GeoPedia at http://%s:%d", args.host, args.port)
    logger.info("Runs loaded: %s", list(layer_states))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
