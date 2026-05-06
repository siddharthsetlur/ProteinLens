"""
API route handlers for the GeoPedia SAE feature visualizer.

The server can serve **one or many** SAE layers in a single process. Each
layer's analysis dir (with its enrichment JSONs, geometry-primary analysis,
case-study payloads, PDB cache, etc.) is registered under a layer number
at startup; routes that touch per-layer data live under
``/api/layers/{layer}/...``.

Endpoints:
  GET /api/layers                               -> registered layers + summary
  GET /api/landing                              -> landing-page payload (all layers)
  GET /api/featured                             -> paper-described features (real scores)

  GET /api/layers/{L}/stats                     -> dataset stats + SAE config
  GET /api/layers/{L}/index                     -> feature index for the table
  GET /api/layers/{L}/method-coverage           -> per-method q<0.05 counts
  GET /api/layers/{L}/feature/{id}              -> per-feature JSON (top sequences, bins)
  GET /api/layers/{L}/feature/{id}/significance -> 7-method score/label/q row
  GET /api/layers/{L}/feature/{id}/interpro     -> interpro enrichment JSON
  GET /api/layers/{L}/feature/{id}/motif        -> MEME/PWM motif enrichment JSON
  GET /api/layers/{L}/feature/{id}/position     -> position enrichment JSON
  GET /api/layers/{L}/feature/{id}/geometry     -> geometry enrichment JSON
  GET /api/layers/{L}/feature/{id}/cath         -> CATH enrichment JSON
  GET /api/layers/{L}/feature/{id}/nmpfam       -> NMPFams annotation JSON
  GET /api/layers/{L}/geometry-primary          -> geometry-primary analysis
  GET /api/layers/{L}/cross-family-geometry     -> case study 01 payload
  GET /api/layers/{L}/subdomain-case-study      -> case study 02 payload
  GET /api/layers/{L}/nmpfam-case-study         -> case study 03 payload
  GET /api/layers/{L}/meme-case-study-families  -> MEME case-study list
  GET /api/layers/{L}/static-plot/{filename}    -> pre-rendered scatter PNG

Shared resources (queried across all known layers' caches):
  GET /api/pdb/{accession}                      -> AlphaFold PDB
  GET /api/interpro/{accession}                 -> cached InterPro annotations
  GET /api/nmpfam-pdb/{family_id}               -> NMPFams structure (proxy)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@dataclass
class LayerState:
    """Per-layer in-memory state, populated once at startup."""
    layer: int
    analysis_dir: Path
    stats: dict[str, Any]
    feature_index: list[dict[str, Any]]
    pipeline_status: dict[str, Any]
    method_coverage: dict[str, Any]
    feature_by_id: dict[int, dict[str, Any]] = field(default_factory=dict)


# Layer number -> LayerState, populated by configure().
_layers: dict[int, LayerState] = {}

# Optional landing-page summary (set once at startup).
_landing: dict[str, Any] = {}
_featured: list[dict[str, Any]] = []


def configure(
    layer_states: dict[int, LayerState],
    landing: dict[str, Any] | None = None,
    featured: list[dict[str, Any]] | None = None,
) -> None:
    """Inject per-layer states + landing payload from the server bootstrap."""
    global _layers, _landing, _featured
    _layers = dict(layer_states)
    _landing = landing or {}
    _featured = list(featured or [])
    for L, s in _layers.items():
        s.feature_by_id = {row["feature_id"]: row for row in s.feature_index}
    logger.info("API configured for layers: %s", sorted(_layers.keys()))


def _get_layer(layer: int) -> LayerState:
    """Look up a layer state or 404 if it's not registered."""
    s = _layers.get(layer)
    if s is None:
        raise HTTPException(
            status_code=404,
            detail=f"Layer {layer} not loaded. Available: {sorted(_layers.keys())}",
        )
    return s


def _read_json(path: Path) -> Any:
    """Read a JSON file or 404 if missing."""
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}")
    with open(path) as f:
        return json.load(f)


# ─────────────────────────── Top-level routes ───────────────────────────


@router.get("/layers")
def list_layers() -> list[dict[str, Any]]:
    """Return the list of registered layers with a small summary each."""
    return [
        {
            "layer": L,
            "num_features": len(s.feature_index),
            "num_annotated": s.method_coverage.get("total_annotated_n"),
            "pct_annotated": s.method_coverage.get("total_annotated_pct"),
        }
        for L, s in sorted(_layers.items())
    ]


@router.get("/landing")
def get_landing() -> dict[str, Any]:
    """Return the multi-layer landing payload (per-layer summary + SAE cfg)."""
    return _landing


@router.get("/featured")
def get_featured() -> list[dict[str, Any]]:
    """Return the paper-described feature highlights with real per-feature scores."""
    return _featured


# ─────────────────────────── Per-layer routes ───────────────────────────


_SIG_FIELDS = [
    "feature_id",
    "geometry_radar",
    "top_geometric_feature",
    "structural_category",
    "is_geometry_primary",
]
for _k in range(1, 8):
    _SIG_FIELDS.extend([f"m{_k}_score", f"m{_k}_label", f"m{_k}_q"])


@router.get("/layers/{layer}/stats")
def get_stats(layer: int) -> dict[str, Any]:
    """Return merged dataset stats + SAE config + pipeline status for the layer."""
    s = _get_layer(layer)
    return {**s.stats, "pipeline": s.pipeline_status, "layer": layer}


@router.get("/layers/{layer}/index")
def get_index(layer: int) -> list[dict[str, Any]]:
    """Return the full feature index for the layer table view."""
    return _get_layer(layer).feature_index


@router.get("/layers/{layer}/method-coverage")
def get_method_coverage(layer: int) -> dict[str, Any]:
    """Return per-method q<0.05 coverage and the union total annotated count."""
    return _get_layer(layer).method_coverage


@router.get("/layers/{layer}/feature/{feature_id}/significance")
def get_feature_significance(layer: int, feature_id: int) -> dict[str, Any]:
    """Return the 7-method score/label/q-value row for a single feature."""
    s = _get_layer(layer)
    row = s.feature_by_id.get(feature_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Feature {feature_id} not in layer {layer}")
    return {k: row.get(k) for k in _SIG_FIELDS}


@router.get("/layers/{layer}/feature/{feature_id}")
def get_feature(layer: int, feature_id: int) -> dict[str, Any]:
    """Return the per-feature JSON (top_sequences, activation_bins, coverage, ...)."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "features" / f"{feature_id:04d}.json")


@router.get("/layers/{layer}/feature/{feature_id}/interpro")
def get_feature_interpro(layer: int, feature_id: int) -> dict[str, Any]:
    """Return InterPro enrichment results for a feature."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "interpro_enrichment" / f"{feature_id:04d}.json")


@router.get("/layers/{layer}/feature/{feature_id}/motif")
def get_feature_motif(layer: int, feature_id: int) -> dict[str, Any]:
    """Return MEME/PWM motif enrichment results for a feature."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "motif_pwm_enrichment" / f"{feature_id:04d}.json")


@router.get("/layers/{layer}/feature/{feature_id}/position")
def get_feature_position(layer: int, feature_id: int) -> dict[str, Any]:
    """Return sequence position enrichment results for a feature."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "position_enrichment" / f"{feature_id:04d}.json")


@router.get("/layers/{layer}/feature/{feature_id}/geometry")
def get_feature_geometry(layer: int, feature_id: int) -> dict[str, Any]:
    """Return geometry enrichment results for a feature."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "geometry_enrichment" / f"{feature_id:04d}.json")


@router.get("/layers/{layer}/feature/{feature_id}/cath")
def get_feature_cath(layer: int, feature_id: int) -> dict[str, Any]:
    """Return CATH enrichment results for a feature."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "cath_enrichment" / f"{feature_id:04d}.json")


@router.get("/layers/{layer}/feature/{feature_id}/nmpfam")
def get_feature_nmpfam(layer: int, feature_id: int) -> dict[str, Any]:
    """Return NMPFams novel-protein annotation for a feature."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "nmpfam" / "nmpfam_enrichment" / f"{feature_id:04d}.json")


@router.get("/layers/{layer}/geometry-primary")
def get_geometry_primary(layer: int) -> dict[str, Any]:
    """Return geometry-primary analysis results."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "geometry_primary_analysis.json")


@router.get("/layers/{layer}/cross-family-geometry")
def get_cross_family_geometry(layer: int) -> dict[str, Any]:
    """Case study 01: features without DB labels but with significant geometry."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "cross_family_geometry.json")


@router.get("/layers/{layer}/subdomain-case-study")
def get_subdomain_case_study(layer: int) -> dict[str, Any]:
    """Case study 02: shared DB labels split by geometry."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "subdomain_case_study.json")


@router.get("/layers/{layer}/nmpfam-case-study")
def get_nmpfam_case_study(layer: int) -> dict[str, Any]:
    """Case study 03: NMPFams metagenomic transfer (not all layers have this)."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "nmpfam_case_study.json")


@router.get("/layers/{layer}/nmpfam-transfer-summary")
def get_nmpfam_transfer_summary(layer: int) -> dict[str, Any]:
    """Per-feature transfer aggregates (max / median PR-AUC over NMPFam hits)
    plus the per-layer Table 4 stats. Built by
    ``scripts/build_nmpfam_transfer_summary.py``."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "nmpfam_transfer_summary.json")


@router.get("/layers/{layer}/meme-case-study-families")
def get_meme_case_study_families(layer: int) -> dict[str, Any]:
    """Return MEME/PWM case-study families."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "meme_case_study_families.json")


@router.get("/layers/{layer}/nmpfam-annotations")
def get_nmpfam_annotations(layer: int) -> dict[str, Any]:
    """q-value-tiered NMPFam annotations master index."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "nmpfam_annotation" / "annotations.json")


@router.get("/layers/{layer}/nmpfam-annotations/{feature_id}")
def get_nmpfam_annotation_feature(layer: int, feature_id: int) -> dict[str, Any]:
    """Per-feature NMPFam annotation payload with all hit tiers."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "nmpfam_annotation" / f"{feature_id:04d}.json")


@router.get("/layers/{layer}/transfer-metrics/B")
def get_transfer_metric_b(layer: int) -> dict[str, Any]:
    """Metric B — predictive transfer of SwissProt GBM to NMPFams."""
    s = _get_layer(layer)
    return _read_json(s.analysis_dir / "transfer_metrics" / "metric_B.json")


@router.get("/layers/{layer}/static-plot/{filename}")
def get_static_plot(layer: int, filename: str) -> Response:
    """Serve a pre-generated scatter plot PNG from the layer's static_plots/."""
    if not re.match(r"^[a-z0-9_]+\.png$", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    s = _get_layer(layer)
    fpath = s.analysis_dir / "static_plots" / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"Plot {filename} not found")
    return Response(
        content=fpath.read_bytes(),
        media_type="image/png",
        headers={"Cache-Control": "max-age=3600"},
    )


# ───────────────────── Shared / external resource routes ─────────────────────

_ACCESSION_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_accession(accession: str) -> None:
    """Reject non-alphanumeric accession strings to prevent path traversal."""
    if not _ACCESSION_RE.match(accession):
        raise HTTPException(status_code=400, detail=f"Invalid accession format: {accession}")


@router.get("/pdb/{accession}")
async def get_pdb(accession: str) -> Response:
    """
    Serve an AlphaFold PDB file.

    Tries every loaded layer's pdb_cache/ in turn; falls back to the
    AlphaFold REST API if not cached locally.
    """
    _validate_accession(accession)

    for s in _layers.values():
        pdb_dir = s.analysis_dir / "pdb_cache"
        if pdb_dir.exists():
            matches = list(pdb_dir.glob(f"AF-{accession}-F1-model_v*.pdb"))
            if matches:
                content = matches[0].read_text()
                return Response(
                    content=content,
                    media_type="text/plain",
                    headers={"Cache-Control": "max-age=86400"},
                )

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
    """Return cached InterPro domain annotations for a protein accession."""
    _validate_accession(accession)
    for s in _layers.values():
        fpath = s.analysis_dir / "interpro_cache" / f"{accession}.json"
        if fpath.exists():
            with open(fpath) as f:
                return json.load(f)
    raise HTTPException(status_code=404, detail=f"InterPro data for {accession} not found")


@router.get("/nmpfam-pdb/{family_id}")
async def get_nmpfam_pdb(family_id: str) -> Response:
    """Proxy an NMPFams PDB file from the Fleming Institute API (no caching)."""
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
