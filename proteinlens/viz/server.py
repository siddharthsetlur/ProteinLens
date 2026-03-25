"""
FastAPI server for the SAE Feature Visualizer web console.

Usage:
    python -m proteinlens.viz.server --data-dir feature_data_test_20 --port 8050

Startup sequence:
  1. Parse CLI args (--data-dir, --port, --host)
  2. Run index_builder to merge summary files into in-memory feature index
  3. Configure API routes with the prebuilt data
  4. Mount static files (HTML/JS/CSS) and start uvicorn

The server serves two pages:
  - GET /             -> index.html (homepage with feature table)
  - GET /feature/{id} -> feature.html (detail page, JS reads ID from URL)
"""

import argparse
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from proteinlens.viz import api
from proteinlens.viz.index_builder import build_feature_index, build_pipeline_status, build_stats

logger = logging.getLogger(__name__)

# Directory containing static HTML/JS/CSS files
STATIC_DIR = Path(__file__).parent / "static"


def create_app(data_dir: Path) -> FastAPI:
    """
    Build and configure the FastAPI application.

    Args:
        data_dir: Path to the feature analysis output directory
                  (e.g., feature_data_test_20/).

    Returns:
        Configured FastAPI app with all routes registered.
    """
    app = FastAPI(title="SAE Feature Visualizer", docs_url=None, redoc_url=None)

    # --- Build index at startup ---
    logger.info("Building feature index from %s ...", data_dir)
    stats = build_stats(data_dir)
    pipeline_status = build_pipeline_status(data_dir)
    feature_index = build_feature_index(data_dir)
    logger.info("Index built: %d features", len(feature_index))

    # Inject prebuilt data into the API module
    api.configure(data_dir, stats, feature_index, pipeline_status)

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
        "--data-dir",
        type=str,
        required=True,
        help="Path to the feature analysis output directory (e.g., feature_data_test_20/)",
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
    data_dir = Path(args.data_dir).resolve()

    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")
    if not (data_dir / "dataset_stats.json").exists():
        raise SystemExit(f"Not a valid data directory (missing dataset_stats.json): {data_dir}")

    app = create_app(data_dir)

    logger.info("Starting server at http://%s:%d", args.host, args.port)
    logger.info("Data directory: %s", data_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
