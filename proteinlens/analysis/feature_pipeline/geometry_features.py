"""Stage 6a: Compute geometric features for all proteins with PDB files.

For each AlphaFold PDB in ``config.pdb_cache_dir``:

1. Compute the 55-dim protein-level geometry vector
   (writhe, curvature, helix stats, etc.).
2. Compute per-residue geometric profiles (curvature, torsion, planarity,
   tangent vectors, structural categories) and save to a per-protein ``.npz``
   in ``geometry_residue_profiles/``.
3. Assemble all protein-level vectors into a single
   ``geometry_protein_features.npz`` matrix.

The stage is **resumable**: proteins that already have a ``.npz`` in the
residue profiles directory are skipped. The full protein-level matrix is
rebuilt at the end from all available profiles.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.geometry.protein_features import (
    GEOM_FEATURE_NAMES,
    compute_protein_geometry,
)
from proteinlens.analysis.geometry.residue_features import (
    ca_backbone,
    compute_residue_profiles,
    detect_alpha_helices_from_ca,
)

logger = logging.getLogger(__name__)


def _extract_accession_from_pdb_filename(pdb_path: Path) -> str | None:
    """Extract the UniProt accession from an AlphaFold PDB filename.

    Expected format: ``AF-{ACCESSION}-F1-model_v*.pdb``

    Parameters
    ----------
    pdb_path : Path
        Path to an AlphaFold PDB file.

    Returns
    -------
    str or None
        The accession string, or None if the filename doesn't match the
        expected pattern.
    """
    name = pdb_path.stem  # e.g. "AF-P12345-F1-model_v4"
    parts = name.split("-")
    if len(parts) >= 3 and parts[0] == "AF":
        return parts[1]
    return None


def run_geometry_features(config: PipelineConfig) -> None:
    """Stage 6a entry point: compute geometry for all proteins with PDBs.

    Iterates all ``.pdb`` files in ``config.pdb_cache_dir``, computes
    protein-level and residue-level geometric features, and saves results.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration with paths and geometry parameters.
    """
    profiles_dir = config.geometry_residue_profiles_dir  # creates dir
    pdb_cache = config.pdb_cache_dir

    # Load sequences if available (for amino acid composition features)
    sequences: dict[str, str] = {}
    if config.sequences_path.exists():
        sequences = json.loads(config.sequences_path.read_text())

    # Discover all PDB files
    pdb_files = sorted(pdb_cache.glob("*.pdb"))
    n_total = len(pdb_files)

    if n_total == 0:
        logger.warning("No PDB files found in %s", pdb_cache)
        return

    # Check which accessions already have residue profiles (for resumability)
    existing = {p.stem for p in profiles_dir.glob("*.npz")}

    # Process each protein
    n_computed = 0
    n_skipped = 0
    n_failed = 0
    protein_geom: dict[str, dict[str, float]] = {}

    for i, pdb_path in enumerate(pdb_files):
        acc = _extract_accession_from_pdb_filename(pdb_path)
        if acc is None:
            n_failed += 1
            continue

        # Resumability: skip if residue profile already exists
        if acc in existing:
            n_skipped += 1
            # Still load the protein-level geometry from the saved profile
            # so we can rebuild the full matrix at the end
            continue

        pdb_text = pdb_path.read_text()

        # -- Protein-level geometry (55-dim vector) --
        geom = compute_protein_geometry(pdb_text)
        if geom is None:
            n_failed += 1
            continue

        # -- Residue-level profiles --
        try:
            ca = ca_backbone(pdb_text, chain_id=None)
        except Exception:
            n_failed += 1
            continue

        if ca is None or len(ca) < 10:
            n_failed += 1
            continue

        helices = detect_alpha_helices_from_ca(ca)
        profiles = compute_residue_profiles(ca, helices)

        # Get sequence if available
        seq = sequences.get(acc, "")

        # Save residue-level profile as .npz
        np.savez_compressed(
            profiles_dir / f"{acc}.npz",
            ca=ca,
            curvature=profiles["curvature"],
            torsion=profiles["torsion"],
            planarity=profiles["planarity"],
            tangents=profiles["tangents"],
            helix_mask=profiles["helix_mask"],
            categories=profiles["categories"],
            sequence=np.array([seq]),  # store as array for npz compat
            # Also store the protein-level geometry vector for matrix assembly
            protein_geometry=np.array([geom[name] for name in GEOM_FEATURE_NAMES]),
        )

        protein_geom[acc] = geom
        n_computed += 1

        if (i + 1) % 500 == 0:
            logger.info(
                "Progress: %d/%d PDBs (computed=%d, skipped=%d, failed=%d)",
                i + 1, n_total, n_computed, n_skipped, n_failed,
            )

    # -- Rebuild full protein-level geometry matrix from ALL profiles --
    # (includes both newly computed and previously existing)
    all_profiles = sorted(profiles_dir.glob("*.npz"))
    accessions: list[str] = []
    geom_rows: list[np.ndarray] = []

    for npz_path in all_profiles:
        acc = npz_path.stem
        try:
            data = np.load(npz_path, allow_pickle=True)
            if "protein_geometry" in data:
                accessions.append(acc)
                geom_rows.append(data["protein_geometry"])
        except Exception as e:
            logger.warning("Failed to load %s: %s", npz_path, e)

    if geom_rows:
        geometry_matrix = np.vstack(geom_rows)  # (N, 55)
        np.savez_compressed(
            config.geometry_protein_features_path,
            accessions=np.array(accessions),
            geometry_matrix=geometry_matrix,
            feature_names=np.array(GEOM_FEATURE_NAMES),
        )
        logger.info(
            "Saved geometry_protein_features.npz: %d proteins x %d features",
            geometry_matrix.shape[0], geometry_matrix.shape[1],
        )
    else:
        logger.warning("No valid geometry profiles found.")

    logger.info(
        "Computed geometry for %d proteins (%d skipped/resumed, %d failed, %d total PDBs)",
        n_computed, n_skipped, n_failed, n_total,
    )
