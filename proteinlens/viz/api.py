"""
API route handlers for the SAE feature visualizer.

All endpoints read data files from the configured data directory.
Feature/enrichment data is served directly from JSON files on disk.
PDB and InterPro cache files are served with appropriate content types.

Endpoints:
  GET /api/stats          -> merged dataset + SAE config
  GET /api/index          -> feature table data (built at startup)
  GET /api/feature/{id}   -> per-feature JSON (top sequences, bins, coverage)
  GET /api/feature/{id}/interpro  -> interpro enrichment JSON (404 if missing)
  GET /api/feature/{id}/geometry  -> geometry enrichment JSON (404 if missing)
  GET /api/pdb/{accession}        -> PDB file as text/plain (404 if missing)
  GET /api/interpro/{accession}   -> interpro cache JSON (404 if missing)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# These are set by server.py at startup after building the index
_data_dir: Path = Path(".")
_stats: dict[str, Any] = {}
_feature_index: list[dict[str, Any]] = []
_pipeline_status: dict[str, Any] = {}


def configure(
    data_dir: Path,
    stats: dict[str, Any],
    feature_index: list[dict[str, Any]],
    pipeline_status: dict[str, Any],
) -> None:
    """
    Inject startup-computed data into the API module.

    Called once by server.py after index_builder runs. This avoids
    re-reading files on every request.
    """
    global _data_dir, _stats, _feature_index, _pipeline_status
    _data_dir = data_dir
    _stats = stats
    _feature_index = feature_index
    _pipeline_status = pipeline_status


@router.get("/stats")
def get_stats() -> dict[str, Any]:
    """Return merged dataset stats + SAE config + pipeline status."""
    return {**_stats, "pipeline": _pipeline_status}


@router.get("/index")
def get_index() -> list[dict[str, Any]]:
    """Return the full feature index for the homepage table."""
    return _feature_index


@router.get("/feature/{feature_id}")
def get_feature(feature_id: int) -> dict[str, Any]:
    """
    Return the per-feature JSON containing top_sequences, activation_bins,
    coverage, and per_residue_activations.

    The feature file is named {feature_id:04d}.json under features/.
    """
    fpath = _data_dir / "features" / f"{feature_id:04d}.json"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"Feature {feature_id} not found")
    with open(fpath) as f:
        return json.load(f)


@router.get("/feature/{feature_id}/interpro")
def get_feature_interpro(feature_id: int) -> dict[str, Any]:
    """
    Return interpro enrichment results for a feature.
    Returns 404 if enrichment hasn't been computed yet.
    """
    fpath = _data_dir / "interpro_enrichment" / f"{feature_id:04d}.json"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"InterPro enrichment for feature {feature_id} not found")
    with open(fpath) as f:
        return json.load(f)


@router.get("/feature/{feature_id}/geometry")
def get_feature_geometry(feature_id: int) -> dict[str, Any]:
    """
    Return geometry enrichment results for a feature.
    Returns 404 if enrichment hasn't been computed yet.
    """
    fpath = _data_dir / "geometry_enrichment" / f"{feature_id:04d}.json"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"Geometry enrichment for feature {feature_id} not found")
    with open(fpath) as f:
        return json.load(f)


_ACCESSION_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_accession(accession: str) -> None:
    """Validate that an accession string contains only safe characters (alphanumeric, underscore, hyphen)."""
    if not _ACCESSION_RE.match(accession):
        raise HTTPException(status_code=400, detail=f"Invalid accession format: {accession}")


@router.get("/pdb/{accession}")
async def get_pdb(accession: str) -> Response:
    """
    Serve an AlphaFold PDB file.

    Checks the local pdb_cache directory first. If not found, fetches from
    the AlphaFold API (https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb).

    Returns text/plain with a 24-hour cache header (PDBs are immutable).
    """
    _validate_accession(accession)

    # Try local cache first
    pdb_dir = _data_dir / "pdb_cache"
    if pdb_dir.exists():
        matches = list(pdb_dir.glob(f"AF-{accession}-F1-model_v*.pdb"))
        if matches:
            content = matches[0].read_text()
            return Response(
                content=content,
                media_type="text/plain",
                headers={"Cache-Control": "max-age=86400"},
            )

    # Fetch from AlphaFold API (lookup prediction to get correct PDB URL)
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            pred = await client.get(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}")
            if pred.status_code == 200:
                pdb_url = pred.json()[0]["pdbUrl"]
                resp = await client.get(pdb_url)
                if resp.status_code == 200:
                    return Response(
                        content=resp.text,
                        media_type="text/plain",
                        headers={"Cache-Control": "max-age=86400"},
                    )
    except (httpx.HTTPError, KeyError, IndexError):
        pass

    raise HTTPException(status_code=404, detail=f"PDB for {accession} not found locally or on AlphaFold")


@router.get("/interpro/{accession}")
def get_interpro_cache(accession: str) -> dict[str, Any]:
    """
    Return cached InterPro domain annotations for a protein accession.
    Used by the frontend to draw domain boundary overlays on sequence strips.
    """
    _validate_accession(accession)
    fpath = _data_dir / "interpro_cache" / f"{accession}.json"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"InterPro data for {accession} not found")
    with open(fpath) as f:
        return json.load(f)
