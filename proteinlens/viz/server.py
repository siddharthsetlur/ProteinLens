"""
FastAPI server for the SAE Feature Visualizer web console.

Usage:
    python -m proteinlens.viz.server \\
        --analysis-dir trained_models/layer_4/frosty-sweep-15/analysis \\
        --port 8050

Startup sequence:
  1. Parse CLI args (--analysis-dir, --port, --host)
  2. Run index_builder to merge enrichment summaries + permutation q-values
     into the in-memory feature index
  3. Configure API routes with the prebuilt data
  4. Mount static files (HTML/JS/CSS) and start uvicorn

The analysis dir is self-contained: it holds geometry_primary_analysis.json,
permutation_null/, per-feature enrichment JSONs (interpro_enrichment/,
cath_enrichment/, motif_pwm_enrichment/, position_enrichment/,
geometry_enrichment/), case-study JSONs, sequences, PDB cache, and
dataset_stats.json. One --analysis-dir per SAE run.
"""

import argparse
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from proteinlens.viz import api
from proteinlens.viz.index_builder import (
    build_feature_index,
    build_method_coverage,
    build_pipeline_status,
    build_stats,
)

logger = logging.getLogger(__name__)

# Directory containing static HTML/JS/CSS files
STATIC_DIR = Path(__file__).parent / "static"


def create_app(analysis_dir: Path) -> FastAPI:
    """
    Build and configure the FastAPI application.

    Args:
        analysis_dir: Path to an SAE run's analysis directory
                      (e.g., trained_models/layer_4/frosty-sweep-15/analysis/).

    Returns:
        Configured FastAPI app with all routes registered.
    """
    app = FastAPI(title="SAE Feature Visualizer", docs_url=None, redoc_url=None)

    # --- Build index at startup ---
    logger.info("Building feature index from %s ...", analysis_dir)
    stats = build_stats(analysis_dir)
    pipeline_status = build_pipeline_status(analysis_dir)
    feature_index = build_feature_index(analysis_dir)
    method_coverage = build_method_coverage(feature_index)
    logger.info(
        "Index built: %d features; method coverage: total_annotated=%d/%d (%.1f%%)",
        len(feature_index),
        method_coverage["total_annotated_n"],
        method_coverage["total_features"],
        method_coverage["total_annotated_pct"],
    )

    # Inject prebuilt data into the API module
    api.configure(analysis_dir, stats, feature_index, pipeline_status, method_coverage)

    # --- Register API routes ---
    app.include_router(api.router)

    # --- HTML page routes ---
    @app.get("/", response_class=FileResponse)
    def homepage():
        """Serve the homepage (feature table dashboard)."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/feature/{feature_id}", response_class=FileResponse)
    def feature_page(feature_id: int):
        """Serve the feature detail page. JS reads the feature ID from the URL."""
        return FileResponse(STATIC_DIR / "feature.html")

    @app.get("/geometry-fills-missing-db", response_class=FileResponse)
    def db_silent_geometry_page():
        """Serve the 'Geometry annotates features with missing DB annotations' page."""
        return FileResponse(STATIC_DIR / "cross_family_geometry.html")

    @app.get("/cross-family-geometry", response_class=FileResponse)
    def cross_family_geometry_page():
        """Legacy slug — same page; JS soft-redirects the URL to the new slug."""
        return FileResponse(STATIC_DIR / "cross_family_geometry.html")

    @app.get("/cross-family-geometry/{feature_id}", response_class=FileResponse)
    def cross_family_geometry_detail_page(feature_id: int):
        """Serve a single feature deep-dive within the DB-silent-geometry case study."""
        return FileResponse(STATIC_DIR / "cross_family_detail.html")

    @app.get("/meme-case-studies", response_class=FileResponse)
    def meme_case_studies_page():
        """Serve the MEME case studies list page."""
        return FileResponse(STATIC_DIR / "meme_case_studies.html")

    @app.get("/meme-case-studies/{consensus}", response_class=FileResponse)
    def meme_case_study_detail_page(consensus: str):
        """Serve the MEME case study detail page. JS reads consensus from URL."""
        return FileResponse(STATIC_DIR / "meme_case_study_detail.html")

    @app.get("/subdomain-decomposition", response_class=FileResponse)
    def subdomain_case_study_page():
        """Serve the sub-domain geometric decomposition case study list page."""
        return FileResponse(STATIC_DIR / "subdomain_case_study.html")

    @app.get("/subdomain-decomposition/{source}/{code:path}", response_class=FileResponse)
    def subdomain_detail_page(source: str, code: str):
        """Serve the per-group detail page. JS reads source + code from the URL."""
        return FileResponse(STATIC_DIR / "subdomain_detail.html")

    @app.get("/nmpfam-case-study", response_class=FileResponse)
    def nmpfam_case_study_page():
        """Serve the NMPFams case study overview page."""
        return FileResponse(STATIC_DIR / "nmpfam_case_study.html")

    # Literal /sun must be declared BEFORE the int-param route below, otherwise
    # FastAPI will try to parse "sun" as int and 422.
    @app.get("/nmpfam-case-study/sun", response_class=FileResponse)
    def nmpfam_sun_case_study_page():
        """Serve the unified annotation-transfer case study (the 'sun' view)."""
        return FileResponse(STATIC_DIR / "nmpfam_sun_case_study.html")

    @app.get("/nmpfam-case-study/{feature_id}", response_class=FileResponse)
    def nmpfam_detail_page(feature_id: int):
        """Serve the NMPFams feature detail page."""
        return FileResponse(STATIC_DIR / "nmpfam_detail.html")

    # --- Static file mount (CSS, JS) ---
    # Mounted last so it doesn't shadow the page routes
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the viz server."""
    parser = argparse.ArgumentParser(
        description="SAE Feature Visualizer — localhost web console",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        help=(
            "Path to an SAE run's analysis directory "
            "(e.g., trained_models/layer_4/frosty-sweep-15/analysis/)"
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


def main() -> None:
    """Entry point for `python -m proteinlens.viz.server`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    args = parse_args()
    raw = args.analysis_dir or args.data_dir
    if not raw:
        raise SystemExit("--analysis-dir is required (or the deprecated --data-dir alias)")
    analysis_dir = Path(raw).resolve()

    if not analysis_dir.is_dir():
        raise SystemExit(f"Analysis directory not found: {analysis_dir}")
    if not (analysis_dir / "dataset_stats.json").exists():
        raise SystemExit(
            f"Not a valid analysis directory (missing dataset_stats.json): {analysis_dir}"
        )

    app = create_app(analysis_dir)

    logger.info("Starting server at http://%s:%d", args.host, args.port)
    logger.info("Analysis directory: %s", analysis_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
