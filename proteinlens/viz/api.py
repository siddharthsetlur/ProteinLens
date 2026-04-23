"""
API route handlers for the SAE feature visualizer.

All endpoints read from a single analysis directory (see ``server.py``).
Per-feature enrichment, q-values, case-study JSONs, PDB cache, and
sequence metadata all live there.

Endpoints:
  GET /api/stats                     -> merged dataset + SAE config
  GET /api/index                     -> feature table data (built at startup)
  GET /api/method-coverage           -> per-method q<0.05 counts + total
  GET /api/feature/{id}              -> per-feature JSON (top sequences, bins)
  GET /api/feature/{id}/significance -> thin row of 7-method score/label/q
  GET /api/feature/{id}/interpro     -> interpro enrichment JSON
  GET /api/feature/{id}/motif        -> MEME/PWM motif enrichment JSON
  GET /api/feature/{id}/position     -> position enrichment JSON
  GET /api/feature/{id}/geometry     -> geometry enrichment JSON
  GET /api/feature/{id}/cath         -> CATH enrichment JSON
  GET /api/feature/{id}/nmpfam       -> NMPFams annotation JSON
  GET /api/pdb/{accession}           -> PDB file (cache then AlphaFold)
  GET /api/interpro/{accession}      -> cached InterPro domain annotations
  GET /api/geometry-primary          -> geometry-primary analysis
  GET /api/cross-family-geometry     -> DB-silent-geometry case study
  GET /api/subdomain-case-study      -> subdomain (granularity) case study
  GET /api/meme-case-study-families  -> MEME case-study list
  GET /api/nmpfam-case-study         -> NMPFams case-study overview
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
_analysis_dir: Path = Path(".")
_stats: dict[str, Any] = {}
_feature_index: list[dict[str, Any]] = []
_pipeline_status: dict[str, Any] = {}
_method_coverage: dict[str, Any] = {}
_feature_by_id: dict[int, dict[str, Any]] = {}


def configure(
    analysis_dir: Path,
    stats: dict[str, Any],
    feature_index: list[dict[str, Any]],
    pipeline_status: dict[str, Any],
    method_coverage: dict[str, Any],
) -> None:
    """
    Inject startup-computed data into the API module.

    Called once by server.py after the index builder runs, so per-request
    handlers can serve precomputed state without re-reading files.
    """
    global _analysis_dir, _stats, _feature_index, _pipeline_status
    global _method_coverage, _feature_by_id
    _analysis_dir = analysis_dir
    _stats = stats
    _feature_index = feature_index
    _pipeline_status = pipeline_status
    _method_coverage = method_coverage
    _feature_by_id = {row["feature_id"]: row for row in feature_index}


@router.get("/stats")
def get_stats() -> dict[str, Any]:
    """Return merged dataset stats + SAE config + pipeline status."""
    return {**_stats, "pipeline": _pipeline_status}


@router.get("/index")
def get_index() -> list[dict[str, Any]]:
    """Return the full feature index for the homepage table."""
    return _feature_index


@router.get("/method-coverage")
def get_method_coverage() -> dict[str, Any]:
    """Return per-method q<0.05 coverage and the union total annotated count."""
    return _method_coverage


_SIG_FIELDS = [
    "feature_id",
    "geometry_radar",
    "top_geometric_feature",
    "structural_category",
    "is_geometry_primary",
]
for _k in range(1, 8):
    _SIG_FIELDS.extend([f"m{_k}_score", f"m{_k}_label", f"m{_k}_q"])


@router.get("/feature/{feature_id}/significance")
def get_feature_significance(feature_id: int) -> dict[str, Any]:
    """Return the 7-method score/label/q-value row for a single feature."""
    row = _feature_by_id.get(feature_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Feature {feature_id} not found")
    return {k: row.get(k) for k in _SIG_FIELDS}


@router.get("/geometry-primary")
def get_geometry_primary() -> dict[str, Any]:
    """Return geometry-primary analysis results (404 if not computed)."""
    fpath = _analysis_dir / "geometry_primary_analysis.json"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Geometry-primary analysis not computed yet")
    with open(fpath) as f:
        return json.load(f)


@router.get("/cross-family-geometry")
def get_cross_family_geometry() -> dict[str, Any]:
    """Return cross-family geometry case study data (404 if not built)."""
    fpath = _analysis_dir / "cross_family_geometry.json"
    if not fpath.exists():
        raise HTTPException(
            status_code=404,
            detail="Cross-family geometry case study not built. "
            "Run scripts/build_cross_family_case_study.py first.",
        )
    with open(fpath) as f:
        return json.load(f)


@router.get("/feature/{feature_id}")
def get_feature(feature_id: int) -> dict[str, Any]:
    """
    Return the per-feature JSON containing top_sequences, activation_bins,
    coverage, and per_residue_activations.

    The feature file is named {feature_id:04d}.json under features/.
    """
    fpath = _analysis_dir / "features" / f"{feature_id:04d}.json"
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
    fpath = _analysis_dir / "interpro_enrichment" / f"{feature_id:04d}.json"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"InterPro enrichment for feature {feature_id} not found")
    with open(fpath) as f:
        return json.load(f)


@router.get("/feature/{feature_id}/motif")
def get_feature_motif(feature_id: int) -> dict[str, Any]:
    """
    Return MEME/PWM motif enrichment results for a feature.
    Returns 404 if enrichment hasn't been computed yet.
    """
    fpath = _analysis_dir / "motif_pwm_enrichment" / f"{feature_id:04d}.json"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"Motif PWM enrichment for feature {feature_id} not found")
    with open(fpath) as f:
        return json.load(f)


@router.get("/feature/{feature_id}/position")
def get_feature_position(feature_id: int) -> dict[str, Any]:
    """
    Return sequence position enrichment results for a feature.
    Returns 404 if enrichment hasn't been computed yet.
    """
    fpath = _analysis_dir / "position_enrichment" / f"{feature_id:04d}.json"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"Position enrichment for feature {feature_id} not found")
    with open(fpath) as f:
        return json.load(f)


@router.get("/feature/{feature_id}/geometry")
def get_feature_geometry(feature_id: int) -> dict[str, Any]:
    """
    Return geometry enrichment results for a feature.
    Returns 404 if enrichment hasn't been computed yet.
    """
    fpath = _analysis_dir / "geometry_enrichment" / f"{feature_id:04d}.json"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"Geometry enrichment for feature {feature_id} not found")
    with open(fpath) as f:
        return json.load(f)


@router.get("/static-plot/{filename}")
def get_static_plot(filename: str) -> Response:
    """Serve a pre-generated scatter plot PNG from static_plots/."""
    if not re.match(r"^[a-z0-9_]+\.png$", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    fpath = _analysis_dir / "static_plots" / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"Plot {filename} not found")
    return Response(
        content=fpath.read_bytes(),
        media_type="image/png",
        headers={"Cache-Control": "max-age=3600"},
    )


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
    pdb_dir = _analysis_dir / "pdb_cache"
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


@router.get("/meme-case-study-families")
def get_meme_case_study_families() -> dict[str, Any]:
    """Return pre-computed MEME/PWM case study families (404 if not built)."""
    fpath = _analysis_dir / "meme_case_study_families.json"
    if not fpath.exists():
        raise HTTPException(
            status_code=404,
            detail="MEME case study families not built. Run scripts/build_meme_case_studies.py",
        )
    with open(fpath) as f:
        return json.load(f)


@router.get("/subdomain-case-study")
def get_subdomain_case_study() -> dict[str, Any]:
    """Return sub-domain geometric decomposition case study (404 if not built)."""
    fpath = _analysis_dir / "subdomain_case_study.json"
    if not fpath.exists():
        raise HTTPException(
            status_code=404,
            detail="Sub-domain case study not built. "
            "Run scripts/build_subdomain_case_study.py first.",
        )
    with open(fpath) as f:
        return json.load(f)


@router.get("/nmpfam-case-study")
def get_nmpfam_case_study() -> dict[str, Any]:
    """Return the NMPFams case study JSON (404 if not built)."""
    fpath = _analysis_dir / "nmpfam_case_study.json"
    if not fpath.exists():
        raise HTTPException(
            status_code=404,
            detail="NMPFams case study not built. Run scripts/build_nmpfam_case_study.py first.",
        )
    with open(fpath) as f:
        return json.load(f)


@router.get("/feature/{feature_id}/cath")
def get_feature_cath(feature_id: int) -> dict[str, Any]:
    """
    Return CATH enrichment results for a feature.
    Returns 404 if enrichment hasn't been computed yet.
    """
    fpath = _analysis_dir / "cath_enrichment" / f"{feature_id:04d}.json"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"CATH enrichment for feature {feature_id} not found")
    with open(fpath) as f:
        return json.load(f)


@router.get("/feature/{feature_id}/nmpfam")
def get_feature_nmpfam(feature_id: int) -> dict[str, Any]:
    """
    Return NMPFams novel protein annotation for a feature.
    Returns 404 if NMPFams analysis hasn't been computed yet.
    """
    fpath = _analysis_dir / "nmpfam" / "nmpfam_enrichment" / f"{feature_id:04d}.json"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"NMPFams annotation for feature {feature_id} not found")
    with open(fpath) as f:
        return json.load(f)


@router.get("/nmpfam-pdb/{family_id}")
async def get_nmpfam_pdb(family_id: str) -> Response:
    """
    Proxy an NMPFams PDB file from the Fleming Institute API.
    Streams on demand — no local caching.
    """
    if not re.match(r"^[A-Za-z0-9_-]+$", family_id):
        raise HTTPException(status_code=400, detail=f"Invalid family ID: {family_id}")
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"https://bib.fleming.gr/NMPFamsDB/data/pdb/{family_id}.pdb")
            if resp.status_code == 200:
                return Response(
                    content=resp.text,
                    media_type="text/plain",
                    headers={"Cache-Control": "max-age=86400"},
                )
    except httpx.HTTPError:
        pass
    raise HTTPException(status_code=404, detail=f"NMPFams PDB for {family_id} not found")


@router.get("/interpro/{accession}")
def get_interpro_cache(accession: str) -> dict[str, Any]:
    """
    Return cached InterPro domain annotations for a protein accession.
    Used by the frontend to draw domain boundary overlays on sequence strips.
    """
    _validate_accession(accession)
    fpath = _analysis_dir / "interpro_cache" / f"{accession}.json"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"InterPro data for {accession} not found")
    with open(fpath) as f:
        return json.load(f)
