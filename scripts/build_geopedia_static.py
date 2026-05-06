"""
Bake static JSON files for the GeoPedia (new) viz frontend.

Reads the analysis dirs for layers 2/4/6 and produces a small set of
JSON files under ``proteinlens/viz/static/data/`` so the new SPA can:

  - render the multi-layer landing page (per-layer summary stats),
  - render the feature browser table for any of the three layers
    (without needing the live server to be pointed at that layer),
  - look up real metadata for the paper-named "featured" feature IDs,
  - drive the case-study list pages (geometry-fills-DB, subdomains,
    NMPFam) with real per-layer numbers.

Per-feature deep-dive data (top sequences, activation bins, radar,
NMPFam hits, etc.) is NOT baked here — that flows through the live
``/api/feature/{id}`` endpoints, which are served against the single
analysis dir the user starts the viz with.

Run it once after a new analysis pass:

    python scripts/build_geopedia_static.py

Layer dirs are hard-coded against the canonical local mirror layout
(``trained_models/layer_<N>/<run>/analysis/``). Adjust ``LAYERS`` if
you point at a different run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from proteinlens.viz.index_builder import (
    Q_SIGNIFICANCE_THRESHOLD,
    build_feature_index,
    build_method_coverage,
    build_pipeline_status,
    build_stats,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


REPO = Path(__file__).resolve().parents[1]
TRAINED = REPO / "trained_models"
OUT_DIR = REPO / "proteinlens" / "viz" / "static" / "data"

LAYERS: list[tuple[int, str]] = [
    (2, "firm-sweep-3"),
    (4, "frosty-sweep-15"),
    (6, "major-sweep-15"),
]


# Paper-named features the landing page links to directly.
# Only the feature_id and the curated paper-side metadata is hard-coded;
# all numeric scores are looked up in the per-layer feature index at
# build time (so they always reflect the most recent analysis pass).
FEATURED_BLUEPRINT: list[dict[str, Any]] = [
    {
        "feature_id": 894, "layer": 4, "kind": "geom",
        "title": "Geometry-primary feature (Fig. 3)",
        "desc": (
            "f/894 — annotation only recoverable through the geometric "
            "pipeline. High GBM PR-AUC; no significant database / sequence "
            "match. The canonical geometry-primary case in the paper."
        ),
    },
    {
        "feature_id": 4714, "layer": 4, "kind": "geom",
        "title": "Contact-related, ablation-causal (Fig. 5)",
        "desc": (
            "f/4714 — strong geometry-primary, contact-related feature used "
            "in the ablation experiment (linear response between ablation "
            "strength and contact-density target)."
        ),
    },
    {
        "feature_id": 6775, "layer": 4, "kind": "both",
        "title": "Histidine kinase / HSP90-like ATPase — variant A",
        "desc": (
            "f/6775 (Q15120). One of four layer-4 features sharing the "
            "HSP90-like ATPase residue annotation; geometry distinguishes "
            "it from f/5508, f/8254, f/9608."
        ),
    },
    {
        "feature_id": 5508, "layer": 4, "kind": "both",
        "title": "HSP90-like ATPase — variant B",
        "desc": (
            "f/5508 (Q15120). Same residue label as f/6775 but lights up a "
            "different geometric sub-structure within the domain."
        ),
    },
    {
        "feature_id": 8254, "layer": 4, "kind": "both",
        "title": "HSP90-like ATPase — variant C",
        "desc": (
            "f/8254 (Q15120). Third member of the HSP90-like ATPase cluster; "
            "geometry separates it from siblings on the central β-strand."
        ),
    },
    {
        "feature_id": 9608, "layer": 4, "kind": "both",
        "title": "HSP90-like ATPase — variant D",
        "desc": (
            "f/9608 (Q15120). Fourth HSP90-like ATPase variant — biology "
            "says one label, geometry says four."
        ),
    },
    {
        "feature_id": 8518, "layer": 4, "kind": "geom",
        "title": "Metagenomic-firing feature (Fig. 4)",
        "desc": (
            "f/8518 — geometry-significant on Swiss-Prot and fires on six "
            "NMPfams metagenomic clusters. Bridges Swiss-Prot annotation "
            "onto unannotated sequences."
        ),
    },
    {
        "feature_id": 670, "layer": 4, "kind": "both",
        "title": "Schematic figure feature (Fig. 1)",
        "desc": (
            "f/670 — the feature used in the paper's overview schematic to "
            "walk through the SAE → annotation pipeline."
        ),
    },
]


def _is_sig(q: float | None) -> bool:
    return q is not None and q < Q_SIGNIFICANCE_THRESHOLD


def _per_method_pct(index: list[dict[str, Any]]) -> dict[str, float]:
    """Return the % of features with q<0.05 for each of the 7 methods, plus the union."""
    total = len(index)
    out: dict[str, float] = {}
    for k in range(1, 8):
        n = sum(1 for r in index if _is_sig(r.get(f"m{k}_q")))
        out[f"m{k}_pct"] = round(100.0 * n / total, 2) if total else 0.0
        out[f"m{k}_n"] = n
    n_any = sum(
        1 for r in index if any(_is_sig(r.get(f"m{k}_q")) for k in range(1, 8))
    )
    out["any_pct"] = round(100.0 * n_any / total, 2) if total else 0.0
    out["any_n"] = n_any
    return out


def _category(row: dict[str, Any]) -> str:
    bio = any(_is_sig(row.get(f"m{k}_q")) for k in range(1, 7))
    geom = _is_sig(row.get("m7_q"))
    if geom and bio:
        return "both"
    if geom:
        return "geom_only"
    if bio:
        return "bio_only"
    return "none"


def _category_counts(index: list[dict[str, Any]]) -> dict[str, int]:
    out = {"any": 0, "geom_only": 0, "bio_only": 0, "both": 0, "none": 0, "total": len(index)}
    for r in index:
        c = _category(r)
        out[c] += 1
        if c != "none":
            out["any"] += 1
    return out


def _slim_row(r: dict[str, Any]) -> dict[str, Any]:
    """Return a compact feature row for the layer view table."""
    pct = r.get("pct_proteins_activated")
    if pct is None:
        # Some pipelines emit fraction instead of percent; normalise to percent.
        pct = 0.0
    keep = {
        "feature_id": r["feature_id"],
        "max_act": round(float(r.get("max_activation") or 0), 3),
        "pct_proteins": round(float(pct), 2),
        "structural_category": r.get("structural_category"),
        "geometry_radar": r.get("geometry_radar"),
        "n_nmpfam_hits": r.get("n_nmpfam_hits"),
    }
    for k in range(1, 8):
        keep[f"m{k}_score"] = r.get(f"m{k}_score")
        keep[f"m{k}_label"] = r.get(f"m{k}_label")
        keep[f"m{k}_q"] = r.get(f"m{k}_q")
    return keep


def _summarise_layer(layer: int, run: str, analysis: Path) -> dict[str, Any]:
    """Return per-layer headline numbers + the slim feature index."""
    stats = build_stats(analysis)
    pipeline_status = build_pipeline_status(analysis)
    index = build_feature_index(analysis)
    coverage = build_method_coverage(index)

    pct = _per_method_pct(index)
    cats = _category_counts(index)

    # Pull a few interesting features for the landing's "novel geom" hint
    novel_geom = [
        r["feature_id"] for r in index
        if _is_sig(r.get("m7_q")) and not any(_is_sig(r.get(f"m{k}_q")) for k in range(1, 7))
    ]

    sae_cfg = stats.get("sae", {}) or {}
    return {
        "layer": layer,
        "run": run,
        "num_features": len(index),
        "novel_geom_count": len(novel_geom),
        "any_pct": pct["any_pct"],
        "geom_pct": pct["m7_pct"],
        "bio_pct": max(pct[f"m{k}_pct"] for k in range(1, 7)),
        "interpro_prot_pct": pct["m1_pct"],
        "interpro_res_pct": pct["m2_pct"],
        "cath_prot_pct": pct["m3_pct"],
        "cath_res_pct": pct["m4_pct"],
        "seq_pos_pct": pct["m5_pct"],
        "seq_motif_pct": pct["m6_pct"],
        "geometric_pct": pct["m7_pct"],
        "category_counts": cats,
        "method_coverage": coverage,
        "pipeline_status": pipeline_status,
        "sae": sae_cfg,
        "dataset": stats.get("dataset", {}),
        "_index": index,
    }


def _featured_with_real_scores(featured: list[dict[str, Any]],
                                indexes: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Look up real per-feature scores from the layer indexes for each
    paper-listed feature blueprint."""
    out = []
    for entry in featured:
        L = entry["layer"]
        idx = indexes.get(L) or []
        match = next((r for r in idx if r.get("feature_id") == entry["feature_id"]), None)
        if match is None:
            logger.warning("Featured feature L%s/f%s not found in index", L, entry["feature_id"])
            continue
        e = dict(entry)
        e["geom_score"] = match.get("m7_score")
        e["q_geom"] = match.get("m7_q")
        bio_scores = [match.get(f"m{k}_score") or 0.0 for k in range(1, 7)]
        e["bio_best"] = max(bio_scores) if bio_scores else None
        e["pct_proteins"] = match.get("pct_proteins_activated")
        out.append(e)
    return out


def _maybe_load(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _slim_cross_family(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Trim cross_family_geometry.json down to what the case-study list page needs."""
    if not payload:
        return None
    return {
        "global_stats": payload.get("global_stats", {}),
        "structural_categories_all": payload.get("structural_categories_all", {}),
        "structural_categories_cross_family": payload.get("structural_categories_cross_family", {}),
        # Keep the feature list but only fields we actually render.
        "features": [
            {
                "feature_id": f.get("feature_id"),
                "structural_category": f.get("structural_category"),
                "geom_pr_auc": f.get("geom_pr_auc"),
                "geometry_prauc_padj": f.get("geometry_prauc_padj"),
                "top_geometric_feature": f.get("top_geometric_feature"),
                "is_geometry_primary": f.get("is_geometry_primary"),
                "n_nmpfam_hits": f.get("n_nmpfam_hits"),
            }
            for f in (payload.get("features") or [])
        ],
    }


def _slim_subdomain(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "global_stats": payload.get("global_stats", {}),
        "interpro_groups": payload.get("interpro_groups", []),
        "cath_groups": payload.get("cath_groups", []),
    }


def _slim_nmpfam(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "summary": payload.get("summary", {}),
        "triple_features": payload.get("triple_features", [])[:200],
        "broader_gp_features": payload.get("broader_gp_features", [])[:400],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries: dict[int, dict[str, Any]] = {}
    indexes: dict[int, list[dict[str, Any]]] = {}

    for layer, run in LAYERS:
        analysis = TRAINED / f"layer_{layer}" / run / "analysis"
        if not analysis.is_dir():
            logger.warning("Skipping layer %d — analysis dir missing: %s", layer, analysis)
            continue
        logger.info("Reading layer %d (%s) ...", layer, run)
        s = _summarise_layer(layer, run, analysis)
        idx = s.pop("_index")
        summaries[layer] = s
        indexes[layer] = idx

        # Per-layer slim index for the table view
        slim = [_slim_row(r) for r in idx]
        slim_path = OUT_DIR / f"layer_{layer}_index.json"
        slim_path.write_text(json.dumps(slim, separators=(",", ":")))
        logger.info("  wrote %s (%d rows)", slim_path.name, len(slim))

        # Per-layer case-study slim payloads
        cf = _slim_cross_family(_maybe_load(analysis / "cross_family_geometry.json"))
        if cf is not None:
            (OUT_DIR / f"cross_family_layer_{layer}.json").write_text(
                json.dumps(cf, separators=(",", ":"))
            )
        sub = _slim_subdomain(_maybe_load(analysis / "subdomain_case_study.json"))
        if sub is not None:
            (OUT_DIR / f"subdomain_layer_{layer}.json").write_text(
                json.dumps(sub, separators=(",", ":"))
            )
        nmp = _slim_nmpfam(_maybe_load(analysis / "nmpfam_case_study.json"))
        if nmp is not None:
            (OUT_DIR / f"nmpfam_layer_{layer}.json").write_text(
                json.dumps(nmp, separators=(",", ":"))
            )

    # Cross-layer summary: drives the landing page layer picker.
    layers_payload = {
        "layers": [summaries[L] for L, _ in LAYERS if L in summaries],
        "sae": (summaries[next(iter(summaries))]["sae"] if summaries else {}),
    }
    (OUT_DIR / "layers.json").write_text(
        json.dumps(layers_payload, separators=(",", ":"))
    )
    logger.info("wrote layers.json (%d layers)", len(summaries))

    # Featured features with real numeric scores
    featured = _featured_with_real_scores(FEATURED_BLUEPRINT, indexes)
    (OUT_DIR / "featured.json").write_text(json.dumps(featured, separators=(",", ":")))
    logger.info("wrote featured.json (%d features)", len(featured))


if __name__ == "__main__":
    main()
