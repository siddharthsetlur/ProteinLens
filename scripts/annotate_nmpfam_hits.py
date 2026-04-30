#!/usr/bin/env python3
"""Annotate NMPFam hits using BH q-values on the parent SAE feature.

For a curated set of clean SAE features, walks the analysis directory and
assigns each NMPFam hit a proposed biological label, a confidence tier, and
a structured rationale. Outputs:

    {analysis_dir}/nmpfam_annotation/annotations.json    master index
    {analysis_dir}/nmpfam_annotation/{feature_id}.json   per-feature payload

Confidence tiers are driven by BH-corrected q-values (q<0.05) from the
permutation-null analysis, combined with NMPFam normalised activation.
``n_sig`` counts how many of the seven annotation methods (InterPro
protein+residue, CATH protein+residue, sequence position, MEME motif,
geometry) achieve q<0.05 for that feature.

    A  n_sig >= 6  AND  norm_act >= 0.70     most methods agree; strong hit
    B  n_sig >= 4  AND  norm_act >= 0.55     multi-modal support; solid hit
    C  n_sig >= 2  AND  norm_act >= 0.50     minority of methods; weak hit
    D  otherwise  (above the feature's SAE activation threshold only)

Sub-flags (also q-value driven, independent of tier):

    site_specific   m2_q < 0.05                (InterPro residue-level)
    scaffold        m1_q < 0.05 AND m2_q >= 0.05  (protein fold only, no residue)
    motif_driven    m6_q < 0.05                (MEME PR-AUC)
    fold_only       m7_q < 0.05 AND m1_q >= 0.05  (geometry only)

Usage:
    python scripts/annotate_nmpfam_hits.py \\
        --analysis-dir trained_models/layer_4/frosty-sweep-15/analysis
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

Q_SIG = 0.05

# Method key order matches the feature-index {m1..m7}_q fields.
METHODS = [
    ("m1", "InterPro protein"),
    ("m2", "InterPro residue"),
    ("m3", "CATH protein"),
    ("m4", "CATH residue"),
    ("m5", "Sequence position"),
    ("m6", "MEME motif"),
    ("m7", "Geometry"),
]


# Curated SAE features shown on /nmpfam-case-study/sun.
CURATED_FEATURES: list[dict[str, Any]] = [
    # A. Fold transfer
    {"fid": 10235, "section": "fold",     "label_override": "Peptidase S1, PA clan (trypsin fold)"},
    {"fid": 10213, "section": "fold",     "label_override": "VOC superfamily (glyoxalase / bleomycin)"},
    {"fid": 10120, "section": "fold",     "label_override": "β-lactamase / transpeptidase-like"},
    {"fid": 10051, "section": "fold",     "label_override": "Pectin-lyase fold (parallel β-helix)"},
    {"fid": 10091, "section": "fold",     "label_override": "GNAT N-acetyltransferase"},
    # B. Residue-site transfer
    {"fid": 10077, "section": "residue",  "label_override": "CBS domain (energy/redox sensor)"},
    {"fid": 10084, "section": "residue",  "label_override": "OmpR/PhoB response-regulator DBD"},
    {"fid": 10216, "section": "residue",  "label_override": "Cyclic-nucleotide-binding domain"},
    {"fid": 10179, "section": "residue",  "label_override": "MscS mechanosensitive channel"},
    {"fid": 9987,  "section": "residue",  "label_override": "LRR superfamily"},
    # C. Repeat scaffolds
    {"fid": 9914,  "section": "scaffold", "label_override": "TPR helical-repeat superfamily"},
    {"fid": 10118, "section": "scaffold", "label_override": "WD40 / YVTN β-propeller"},
    {"fid": 10151, "section": "scaffold", "label_override": "Ankyrin repeat superfamily"},
    # D. Sequence motif
    {"fid": 10114, "section": "motif",    "label_override": "Bacterial OM / lipo-anchor signal"},
]


ENZYME_FOLDS = {
    "Peptidase S1, PA clan",
    "Pectin lyase fold",
    "Glyoxalase/Bleomycin resistance protein/Dihydroxybiphenyl dioxygenase",
    "Beta-lactamase/transpeptidase-like",
    "Acyl-CoA N-acyltransferase",
    "Ribonuclease H-like superfamily",
}


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Malformed JSON at %s", path)
        return None


def _bh(pvals: list[float | None]) -> list[float | None]:
    """BH q-value correction; None passes through."""
    n = len(pvals)
    idx = [(i, p) for i, p in enumerate(pvals) if p is not None]
    if not idx:
        return [None] * n
    idx.sort(key=lambda x: x[1])
    m = len(idx)
    out: list[float | None] = [None] * n
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        orig, p = idx[rank]
        q = min(1.0, p * m / (rank + 1))
        if q < running_min:
            running_min = q
        out[orig] = running_min
    return out


def load_all_qvalues(analysis_dir: Path) -> dict[int, dict[str, float | None]]:
    """Read permutation_null/* and BH-correct per method, mirroring index_builder.

    Returns ``{feature_id: {method_key: q_value}}``.
    """
    pn_dir = analysis_dir / "permutation_null"
    if not pn_dir.is_dir():
        raise FileNotFoundError(f"Missing {pn_dir}; run the permutation-null stage.")

    p_by_method: dict[str, list[tuple[int, float | None]]] = {
        "motif_f1":             [],
        "position_f1":          [],
        "interpro_res_f1":      [],
        "cath_res_f1":          [],
        "geometry_prauc":       [],
        "interpro_protein_f1":  [],
        "cath_protein_f1":      [],
        "pwm_f1":               [],
        "pwm_pr_auc":           [],
    }
    fids: list[int] = []

    for path in sorted(pn_dir.glob("*.json")):
        d = read_json(path)
        if not d:
            continue
        fid = int(d["feature_id"])
        fids.append(fid)
        pvs = d.get("p_values", {}) or {}
        for method in p_by_method:
            val = pvs.get(method)
            p_by_method[method].append((fid, float(val) if val is not None else None))

    # BH-correct each method independently
    q_by_method: dict[str, dict[int, float]] = {}
    for method, pairs in p_by_method.items():
        pvals = [p for _, p in pairs]
        qvals = _bh(pvals)
        q_by_method[method] = {fid: q for (fid, _), q in zip(pairs, qvals) if q is not None}

    # Collapse to per-feature q-values in the m1..m7 schema
    out: dict[int, dict[str, float | None]] = {}
    for fid in fids:
        out[fid] = {
            "m1": q_by_method["interpro_protein_f1"].get(fid),
            "m2": q_by_method["interpro_res_f1"].get(fid),
            "m3": q_by_method["cath_protein_f1"].get(fid),
            "m4": q_by_method["cath_res_f1"].get(fid),
            "m5": q_by_method["position_f1"].get(fid),
            # PWM uses PR-AUC on SwissProt; the legacy motif_f1 is mirrored as a fallback
            "m6": q_by_method["pwm_pr_auc"].get(fid) or q_by_method["motif_f1"].get(fid),
            "m7": q_by_method["geometry_prauc"].get(fid),
        }
    return out


def load_evidence(analysis_dir: Path, fid: int, q_row: dict[str, float | None]) -> dict:
    """Assemble evidence streams + per-method q-values for one feature."""
    ev: dict[str, Any] = {"feature_id": fid}

    ipr = read_json(analysis_dir / "interpro_enrichment" / f"{fid:04d}.json") or {}
    mot = read_json(analysis_dir / "motif_pwm_enrichment"  / f"{fid:04d}.json") or {}
    geo = read_json(analysis_dir / "geometry_enrichment"   / f"{fid}.json")     or {}
    nmp = read_json(analysis_dir / "nmpfam"   / "nmpfam_enrichment" / f"{fid:04d}.json") or {}

    pl_top = (ipr.get("protein_level") or [{}])[0]
    rl_top = (ipr.get("residue_level") or [{}])[0]
    ev["interpro"] = {
        "protein_name": pl_top.get("annotation_name"),
        "protein_code": pl_top.get("annotation_code"),
        "protein_f1":   pl_top.get("best_f1", 0.0) or 0.0,
        "residue_name": rl_top.get("annotation_name"),
        "residue_code": rl_top.get("annotation_code"),
        "residue_f1":   rl_top.get("best_f1", 0.0) or 0.0,
    }

    mt = (mot.get("motifs") or [{}])[0]
    ev["meme"] = {
        "consensus": mt.get("consensus"),
        "e_value":   mt.get("e_value"),
    }

    grl = geo.get("geometric_residue_level", {}) or {}
    conc = grl.get("concordance") or {}
    fi = grl.get("feature_importances") or {}
    top_feats = sorted(fi.items(), key=lambda kv: -kv[1])[:4]
    rule_first = (grl.get("rules") or "").split("\n")[0].strip()
    ev["geometry"] = {
        "res_auroc":     conc.get("residue_auroc"),
        "top_feats":     [(k, round(v, 3)) for k, v in top_feats],
        "first_rule":    rule_first,
    }

    ev["qvalues"] = {mk: (round(v, 4) if v is not None else None) for mk, v in q_row.items()}
    ev["n_sig_methods"] = sum(1 for v in q_row.values() if v is not None and v < Q_SIG)
    ev["sig_methods"] = [name for (mk, name) in METHODS if q_row.get(mk) is not None and q_row[mk] < Q_SIG]

    ev["feature_global_max"]      = nmp.get("feature_global_max")
    ev["activation_threshold"]    = nmp.get("activation_threshold_sae")
    ev["n_nmpfam_hits"]           = nmp.get("n_nmpfam_hits", 0)
    ev["nmpfam_hits_raw"]         = nmp.get("nmpfam_hits", [])
    return ev


def tier_for_hit(n_sig: int, norm_act: float) -> str:
    if n_sig >= 6 and norm_act >= 0.70:
        return "A"
    if n_sig >= 4 and norm_act >= 0.55:
        return "B"
    if n_sig >= 2 and norm_act >= 0.50:
        return "C"
    return "D"


TIER_DESCRIPTIONS = {
    "A": "Strong multi-modal — ≥6 of 7 annotation methods significant at q<0.05 and the NMPFam fires ≥70% of the SwissProt global max. Safe to propose the label.",
    "B": "Solid — ≥4 of 7 methods significant at q<0.05 and ≥55% max activation. Propose the label with the caveats listed.",
    "C": "Weak — only 2–3 methods are q-significant, or the NMPFam activation is middling. Hypothesis, not an annotation.",
    "D": "Speculative — above the feature's SAE activation threshold but most methods are not q-significant. Do not use as an annotation.",
}


def _sub_flags(q: dict[str, float | None]) -> dict[str, bool]:
    sig = lambda mk: q.get(mk) is not None and q[mk] < Q_SIG
    return {
        "site_specific": sig("m2"),
        "scaffold":      sig("m1") and not sig("m2"),
        "motif_driven":  sig("m6"),
        "fold_only":     sig("m7") and not sig("m1"),
    }


def _label(tier: str, base_name: str, flags: dict[str, bool]) -> str:
    prefix = {"A": "Candidate", "B": "Candidate", "C": "Possible", "D": "Speculative"}[tier]
    if flags["motif_driven"] and not flags["site_specific"] and tier in ("C", "D"):
        return f"{base_name} — motif match, fold uncertain"
    if flags["scaffold"]:
        return f"{prefix} {base_name} scaffold"
    if flags["site_specific"]:
        return f"{prefix} {base_name} (homologous functional site)"
    if flags["fold_only"]:
        return f"{prefix} {base_name} (structural-motif match)"
    return f"{prefix} {base_name}"


def _caveats(ev: dict, tier: str, flags: dict[str, bool]) -> list[str]:
    out: list[str] = []
    ipr = ev["interpro"]
    base_name = ipr["protein_name"] or "unannotated"

    q = ev["qvalues"]
    non_sig = [name for (mk, name) in METHODS
               if q.get(mk) is not None and q[mk] >= Q_SIG]
    if non_sig and tier in ("A", "B"):
        out.append("q ≥ 0.05 for: " + ", ".join(non_sig))

    if base_name in ENZYME_FOLDS and not flags["site_specific"]:
        out.append("Fold transfer only — enzymatic subclass / catalytic residues not directly transferred.")
    if flags["scaffold"]:
        out.append("Repeat-scaffold match — functional partner / cargo of the scaffold is not inferred.")
    if tier in ("C", "D"):
        out.append("Low confidence under q-value tiering — hypothesis, not annotation.")
    if flags["fold_only"]:
        out.append("Only geometry is q-significant — the InterPro label rides on the fold class, not a sequence match.")
    return out


def annotate_feature(ev: dict, entry: dict) -> dict:
    fid = ev["feature_id"]
    ipr = ev["interpro"]
    flags = _sub_flags(ev["qvalues"])
    base_name = entry.get("label_override") or ipr["protein_name"] or "unannotated fold"

    gmax = ev["feature_global_max"] or 0.0
    n_sig = ev["n_sig_methods"]
    hits_out: list[dict] = []
    for h in ev["nmpfam_hits_raw"]:
        max_act = h.get("max_sae_activation") or h.get("max_activation") or 0.0
        norm_act = (max_act / gmax) if gmax else 0.0
        tier = tier_for_hit(n_sig, norm_act)

        hits_out.append({
            "family_id": h["family_id"],
            "nmpfams_url": h.get("nmpfams_url") or f"https://bib.fleming.gr/NMPFamsDB/family/{h['family_id']}",
            "category": h.get("category"),
            "sequence_count": h.get("sequence_count"),
            "n_residues": h.get("n_residues") or h.get("sequence_length"),
            "max_activation": round(max_act, 3),
            "normalized_activation": round(norm_act, 3),
            "confidence_tier": tier,
            "proposed_label": _label(tier, base_name, flags),
            "rationale": {
                "source_feature": fid,
                "source_interpro_protein": ipr["protein_name"],
                "source_interpro_code":    ipr["protein_code"],
                "n_sig_methods":           n_sig,
                "sig_methods":             ev["sig_methods"],
                "qvalues":                 ev["qvalues"],
                "source_meme_consensus":   ev["meme"].get("consensus"),
                "source_geometry_rule":    ev["geometry"].get("first_rule"),
                "source_geometry_resAUROC": round(ev["geometry"].get("res_auroc") or 0.0, 3),
                "mean_geom_prob_at_active": h.get("mean_geom_prob_at_active"),
                "n_concordance_agree":     h.get("n_agree"),
                "n_concordance_sae_only":  h.get("n_sae_only"),
                "n_concordance_geom_only": h.get("n_geom_only"),
            },
            "sub_flags": flags,
            "caveats": _caveats(ev, tier, flags),
        })

    hits_out.sort(key=lambda r: (-{"A": 3, "B": 2, "C": 1, "D": 0}[r["confidence_tier"]], -r["normalized_activation"]))

    return {
        "feature_id": fid,
        "section": entry.get("section"),
        "base_label": base_name,
        "interpro": ipr,
        "meme": ev["meme"],
        "geometry": ev["geometry"],
        "qvalues": ev["qvalues"],
        "n_sig_methods": n_sig,
        "sig_methods": ev["sig_methods"],
        "sub_flags": flags,
        "tier_counts": {
            tier: sum(1 for h in hits_out if h["confidence_tier"] == tier)
            for tier in ("A", "B", "C", "D")
        },
        "n_hits": len(hits_out),
        "activation_threshold": ev["activation_threshold"],
        "feature_global_max": ev["feature_global_max"],
        "hits": hits_out,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis-dir", type=Path, required=True)
    args = ap.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    if not analysis_dir.is_dir():
        raise SystemExit(f"Not a directory: {analysis_dir}")

    out_dir = analysis_dir / "nmpfam_annotation"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading permutation-null q-values …")
    qvals = load_all_qvalues(analysis_dir)
    logger.info("Loaded q-values for %d features", len(qvals))

    by_feature: list[dict] = []
    for entry in CURATED_FEATURES:
        fid = entry["fid"]
        q_row = qvals.get(fid) or {mk: None for mk, _ in METHODS}
        ev = load_evidence(analysis_dir, fid, q_row)
        if not ev["nmpfam_hits_raw"]:
            logger.warning("Feature %d has no NMPFam hits — skipping", fid)
            continue
        record = annotate_feature(ev, entry)
        (out_dir / f"{fid:04d}.json").write_text(json.dumps(record, indent=2))
        by_feature.append({
            "feature_id": record["feature_id"],
            "section": record["section"],
            "base_label": record["base_label"],
            "n_hits": record["n_hits"],
            "n_sig_methods": record["n_sig_methods"],
            "sig_methods": record["sig_methods"],
            "tier_counts": record["tier_counts"],
        })
        logger.info("F%d (%s): n_sig=%d/7  hits=%d  tiers=%s",
                    fid, record["base_label"], record["n_sig_methods"],
                    record["n_hits"], record["tier_counts"])

    index = {"tier_descriptions": TIER_DESCRIPTIONS, "features": by_feature}
    (out_dir / "annotations.json").write_text(json.dumps(index, indent=2))
    logger.info("Wrote %d feature records to %s", len(by_feature), out_dir)


if __name__ == "__main__":
    main()
