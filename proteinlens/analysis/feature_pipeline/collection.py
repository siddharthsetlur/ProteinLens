"""Stage 3 — Per-residue activation collection and PDB fetching.

For each protein selected in Stage 2, we:

1. Re-embed with ESM2 to get per-residue embeddings.
2. Run the full SAE encode to get a ``(seq_len, num_features)``
   activation matrix.
3. Save the activation matrix as a compressed ``.npz`` file
   (one per protein).
4. Download the AlphaFold PDB structure and cache it.

The per-residue activations are stored because the visualiser needs
them to colour individual residues by activation strength for any
feature.  The ``.npz`` format with default compression typically
achieves 3-5x compression on sparse activation matrices.

This stage is **resumable**: proteins that already have both a ``.npz``
and a ``.pdb`` file are skipped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import requests
import torch
from tqdm import tqdm

from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.data_acquisition import _parse_fasta
from proteinlens.embedders.esm import ESM
from proteinlens.sae.inference import load_sae
from proteinlens.utils import get_device

# AlphaFold API endpoint for discovering the PDB download URL
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"


# ===================================================================
# Public API
# ===================================================================


def run_collection(config: PipelineConfig) -> None:
    """Execute the per-residue collection stage (Stage 3).

    Reads the selection JSON from Stage 2, loads ESM + SAE, and
    processes each selected protein.

    For each protein, writes:
    - ``residue_activations/{accession}.npz`` — compressed numpy
      archive containing key ``"activations"`` with shape
      ``(seq_len, num_features)``, dtype float32.
    - ``pdb_cache/{accession}.pdb`` (or ``AF-{acc}-F1-model_v*.pdb``)
      — AlphaFold predicted structure (best-effort; missing PDBs are
      noted but do not block the pipeline).

    Args:
        config: Pipeline configuration.

    Raises:
        FileNotFoundError: If the selection JSON or FASTA does not exist.
    """
    # ── Load selection ──
    with open(config.selection_path, "r") as f:
        selection = json.load(f)
    selected_accessions: List[str] = selection["all_selected_accessions"]

    # ── Load sequences ──
    _, sequences = _parse_fasta(config.fasta_path)

    # ── Determine which proteins still need activation computation ──
    todo_npz = _get_remaining_proteins(
        selected_accessions, sequences, config
    )
    # Also find proteins that have .npz but are missing a PDB (retry)
    todo_pdb_only = [
        acc for acc in selected_accessions
        if acc in sequences
        and acc not in todo_npz
        and not _has_pdb(acc, config.pdb_cache_dir)
    ]
    print(
        f"[collection] {len(selected_accessions)} selected, "
        f"{len(todo_npz)} need activations, "
        f"{len(todo_pdb_only)} need PDB retry only."
    )

    if not todo_npz and not todo_pdb_only:
        print("[collection] Nothing to do.")
        return

    # ── Load models (only if we have activations to compute) ──
    device = config.device or get_device()
    sae = None
    esm_model = None
    if todo_npz:
        print(f"[collection] Loading SAE from {config.sae_dir} ...")
        sae = load_sae(config.sae_dir, device=device)
        print(f"[collection] Loading ESM model {config.esm_model_name} ...")
        esm_model = ESM(model_name=config.esm_model_name, device=device)

    # ── Process proteins needing activations ──
    session = requests.Session()
    n_npz_saved = 0
    n_pdb_saved = 0
    n_pdb_failed = 0

    for acc in tqdm(todo_npz, desc="Collecting per-residue data"):
        seq = sequences[acc]

        # Step 1 — Compute per-residue activations
        npz_path = config.residue_activations_dir / f"{acc}.npz"
        if not npz_path.exists():
            activations = _compute_residue_activations(
                esm_model, sae, seq, config.esm_layer, device
            )
            # Save as compressed npz.
            # Key "activations" holds shape (seq_len, num_features), float32.
            np.savez_compressed(npz_path, activations=activations)
            n_npz_saved += 1

        # Step 2 — Fetch AlphaFold PDB (best-effort)
        if not _has_pdb(acc, config.pdb_cache_dir):
            pdb_text = fetch_alphafold_pdb(acc, config.pdb_cache_dir, session)
            if pdb_text is not None:
                n_pdb_saved += 1
            else:
                n_pdb_failed += 1

    # ── Retry PDB downloads for proteins that already have .npz ──
    for acc in tqdm(todo_pdb_only, desc="Retrying PDB downloads"):
        pdb_text = fetch_alphafold_pdb(acc, config.pdb_cache_dir, session)
        if pdb_text is not None:
            n_pdb_saved += 1
        else:
            n_pdb_failed += 1

    print(
        f"[collection] Done. {n_npz_saved} new .npz files, "
        f"{n_pdb_saved} new PDBs, {n_pdb_failed} PDB fetch failures."
    )


# ===================================================================
# AlphaFold PDB fetching
# ===================================================================


def fetch_alphafold_pdb(
    accession: str,
    cache_dir: Path,
    session: requests.Session,
) -> Optional[str]:
    """Download an AlphaFold PDB for a UniProt accession and cache it.

    Queries the AlphaFold REST API to discover the current PDB URL
    (the version suffix changes between AlphaFold releases), then
    downloads and caches the file.

    Args:
        accession: UniProt accession (e.g. ``"P12345"``).
        cache_dir: Directory to store PDB files.
        session: ``requests.Session`` for connection pooling.

    Returns:
        The PDB file contents as a string, or ``None`` if the fetch
        failed for any reason (no AlphaFold model, network error, etc.).
    """
    # Check cache first (any AlphaFold version)
    cached = list(cache_dir.glob(f"AF-{accession}-F1-model_v*.pdb"))
    if cached:
        return cached[0].read_text()

    # Also check for simple {accession}.pdb
    simple_cache = cache_dir / f"{accession}.pdb"
    if simple_cache.exists():
        return simple_cache.read_text()

    # Query the AlphaFold API to get the PDB download URL
    api_url = ALPHAFOLD_API_URL.format(acc=accession)
    try:
        resp = session.get(api_url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, list):
            if not data:
                return None
            data = data[0]
        pdb_url = data.get("pdbUrl")
        if not pdb_url:
            return None
    except Exception:
        return None

    # Download the PDB
    try:
        resp = session.get(pdb_url, timeout=30)
        if resp.status_code != 200:
            return None
        pdb_text = resp.text
        # Cache with the filename from the URL (includes version)
        fname = pdb_url.rsplit("/", 1)[-1]
        (cache_dir / fname).write_text(pdb_text)
        return pdb_text
    except Exception:
        return None


# ===================================================================
# Internal helpers
# ===================================================================


def _compute_residue_activations(
    esm: ESM,
    sae,
    sequence: str,
    layer: int,
    device: str,
) -> np.ndarray:
    """Embed one protein and run the full SAE encode.

    Args:
        esm: Loaded ESM embedder instance.
        sae: Loaded SAE model (in eval mode).
        sequence: Amino acid sequence string.
        layer: ESM layer to extract.
        device: PyTorch device.

    Returns:
        Numpy array of shape ``(seq_len, num_features)`` with float32
        activation values.  ``seq_len`` equals ``len(sequence)``.
    """
    # embed_single_sequence returns np.ndarray of shape (seq_len, emb_dim)
    embeddings = esm.embed_single_sequence(sequence, layer)
    embeddings_tensor = torch.tensor(embeddings, device=device)

    with torch.no_grad():
        # Apply the SAE's input normalization before encoding.
        # sae.encode() does NOT normalise internally — only forward() does.
        # For SAEs with normalize_to_sqrt_d=False this is a no-op, but
        # skipping it for normalised SAEs would produce wrong activations.
        normed_input, _ = sae._normalize_input_and_get_norms(embeddings_tensor)
        activations = sae.encode(normed_input)

    return activations.cpu().numpy()


def _get_remaining_proteins(
    selected_accessions: List[str],
    sequences: Dict[str, str],
    config: PipelineConfig,
) -> List[str]:
    """Filter the selection to proteins that still need processing.

    A protein is considered "done" if its ``.npz`` file exists.
    (Missing PDBs are tolerated — they are fetched best-effort.)

    Args:
        selected_accessions: Full list of accessions from selection.
        sequences: Dict of accession -> sequence (from FASTA).
        config: Pipeline configuration.

    Returns:
        List of accessions that still need per-residue collection.
    """
    todo = []
    for acc in selected_accessions:
        if acc not in sequences:
            # PM NOTE: This should not happen if the pipeline stages ran
            # in order, because selection only picks from surveyed
            # proteins which all come from the FASTA.  If it does happen,
            # we skip silently rather than crash, since the protein's
            # data simply won't appear in the final output.
            print(
                f"[collection] WARNING: {acc} is selected but not in FASTA — skipping."
            )
            continue
        npz_path = config.residue_activations_dir / f"{acc}.npz"
        if not npz_path.exists():
            todo.append(acc)
    return todo


def _has_pdb(accession: str, cache_dir: Path) -> bool:
    """Check if a PDB file already exists in the cache.

    Looks for both ``AF-{acc}-F1-model_v*.pdb`` and ``{acc}.pdb``.

    Args:
        accession: UniProt accession.
        cache_dir: PDB cache directory.

    Returns:
        True if at least one PDB file for this accession exists.
    """
    if list(cache_dir.glob(f"AF-{accession}-F1-model_v*.pdb")):
        return True
    if (cache_dir / f"{accession}.pdb").exists():
        return True
    return False
