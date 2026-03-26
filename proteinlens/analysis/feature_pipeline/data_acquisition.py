"""Stage 0a — Download reviewed SwissProt sequences as a FASTA file.

Downloads the full SwissProt FASTA via bulk streaming from UniProt,
filtering by max sequence length. Resumable: existing entries in the
FASTA are preserved and only missing sequences are appended.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from proteinlens.analysis.feature_pipeline.config import PipelineConfig

# ---------------------------------------------------------------------------
# UniProt API endpoints
# ---------------------------------------------------------------------------
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"


# ===================================================================
# Public API
# ===================================================================


def fetch_swissprot_accessions(
    organism_taxid: Optional[int] = 9606,
    max_proteins: Optional[int] = None,
) -> List[str]:
    """Query UniProt for reviewed (SwissProt) accessions.

    Paginates through the UniProt streaming endpoint using the ``Link``
    header until all accessions are retrieved (or *max_proteins* is
    reached).

    Args:
        organism_taxid: NCBI taxonomy ID.  Default 9606 = *Homo sapiens*
            (~20 400 proteins).  ``None`` means all organisms (full
            SwissProt, ~570 000 proteins).
        max_proteins: If not ``None``, stop after collecting this many
            accessions.

    Returns:
        List of UniProt accession strings (e.g. ``["P12345", "Q67890", ...]``).

    Raises:
        requests.HTTPError: If the UniProt API returns a non-200 status.
    """
    if organism_taxid is not None:
        print(
            f"[data_acquisition] Querying UniProt for reviewed proteins "
            f"(taxon {organism_taxid}) ..."
        )
        query = f"(reviewed:true) AND (organism_id:{organism_taxid})"
    else:
        print("[data_acquisition] Querying UniProt for ALL reviewed proteins ...")
        query = "(reviewed:true)"
    params = {
        "query": query,
        "format": "list",  # plain text, one accession per line
        "size": 500,
    }

    accessions: List[str] = []
    url: Optional[str] = UNIPROT_SEARCH_URL

    # UniProt returns paginated results; follow the Link header
    while url is not None:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()

        # Each response body is newline-separated accessions
        batch = [a.strip() for a in resp.text.strip().split("\n") if a.strip()]
        accessions.extend(batch)

        # After the first request, params are baked into the pagination URL
        params = None

        # Check for a "next" pagination link
        link_header = resp.headers.get("Link", "")
        url = None
        if 'rel="next"' in link_header:
            match = re.search(r"<([^>]+)>", link_header)
            if match:
                url = match.group(1)

        if max_proteins is not None and len(accessions) >= max_proteins:
            accessions = accessions[:max_proteins]
            break

    print(f"[data_acquisition]   -> {len(accessions)} accessions retrieved.")
    return accessions


def fetch_sequence(
    accession: str, session: requests.Session
) -> Optional[str]:
    """Fetch a single protein sequence from the UniProt FASTA endpoint.

    Args:
        accession: UniProt accession string (e.g. ``"P12345"``).
        session: A ``requests.Session`` for connection pooling.

    Returns:
        The amino-acid sequence as a plain string (no header, no
        newlines), or ``None`` if the fetch fails (404, timeout, etc.).
    """
    url = UNIPROT_FASTA_URL.format(acc=accession)
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        # FASTA format: first line is ">header", remaining lines are sequence
        lines = resp.text.strip().split("\n")
        return "".join(line.strip() for line in lines if not line.startswith(">"))
    except Exception:
        return None


def download_swissprot_fasta(config: PipelineConfig) -> Tuple[List[str], Dict[str, str]]:
    """Download SwissProt sequences via bulk streaming and write to FASTA.

    Uses UniProt's streaming FASTA endpoint to download all sequences in
    one request (streamed in chunks), filtering out sequences longer than
    ``config.max_seq_len``.  Resumable: sequences already in the output
    FASTA are skipped.

    Args:
        config: Pipeline configuration (uses ``organism_taxid``,
            ``max_proteins``, ``fasta_path``, ``max_seq_len``).

    Returns:
        A tuple ``(accessions, sequences)`` where:
        - *accessions* is the ordered list of accession strings in the FASTA.
        - *sequences* is a dict mapping accession -> sequence string.
    """
    from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog

    # Step 1 — read accessions already in the FASTA (resume support)
    existing_accessions, existing_sequences = _parse_fasta(config.fasta_path)
    already_done = set(existing_accessions)
    print(
        f"[data_acquisition] {len(already_done)} accessions already in "
        f"{config.fasta_path.name}, will skip those."
    )

    # Step 2 — build the UniProt query
    if config.organism_taxid is not None:
        query = (
            f"(reviewed:true) AND (organism_id:{config.organism_taxid})"
            f" AND (length:[1 TO {config.max_seq_len}])"
        )
        print(
            f"[data_acquisition] Bulk downloading reviewed proteins "
            f"(taxon {config.organism_taxid}, length <= {config.max_seq_len}) ..."
        )
    else:
        query = f"(reviewed:true) AND (length:[1 TO {config.max_seq_len}])"
        print(
            f"[data_acquisition] Bulk downloading ALL reviewed proteins "
            f"(length <= {config.max_seq_len}) ..."
        )

    # Step 3 — stream the FASTA from UniProt
    params = {"query": query, "format": "fasta"}
    if config.max_proteins is not None:
        params["size"] = config.max_proteins

    resp = requests.get(
        UNIPROT_SEARCH_URL, params=params, stream=True, timeout=60
    )
    resp.raise_for_status()

    # Step 4 — parse the streamed FASTA, filtering and writing incrementally
    n_new = 0
    n_skipped_existing = 0
    n_streamed = 0
    current_acc: Optional[str] = None
    current_seq_parts: List[str] = []
    existing_acc_set = set(existing_accessions)
    max_total = config.max_proteins  # None means no cap

    def _flush_current() -> bool:
        """Flush the current entry. Returns True if we should stop."""
        nonlocal n_new, n_skipped_existing, n_streamed
        if current_acc is None:
            return False
        n_streamed += 1
        if current_acc in existing_acc_set:
            n_skipped_existing += 1
        else:
            _flush_entry(
                current_acc, current_seq_parts, already_done,
                existing_sequences, fasta_fh,
            )
            n_new += 1
        # Stop early if we've reached max_proteins in the FASTA
        if max_total is not None and len(already_done) >= max_total:
            return True
        return False

    with open(config.fasta_path, "a") as fasta_fh:
        done = False
        for line in resp.iter_lines(decode_unicode=True):
            if done:
                break
            if line is None:
                continue
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Flush previous entry
                done = _flush_current()
                if done:
                    break

                # Parse the new header
                header = line[1:].split()[0]
                parts = header.split("|")
                current_acc = parts[1] if len(parts) >= 2 else parts[0]
                current_seq_parts = []

                # Progress logging
                if n_streamed > 0 and n_streamed % 10000 == 0:
                    print(
                        f"[data_acquisition]   streamed {n_streamed} entries "
                        f"({n_new} new, {n_skipped_existing} already had) ..."
                    )
                    wlog({
                        "download/streamed_total": n_streamed,
                        "download/new": n_new,
                        "download/skipped_existing": n_skipped_existing,
                    })
            else:
                current_seq_parts.append(line)

        # Flush last entry
        if not done:
            _flush_current()

    resp.close()

    print(
        f"[data_acquisition] Done. {n_new} new sequences written, "
        f"{n_skipped_existing} already existed. "
        f"Total in FASTA: {len(already_done)}."
    )
    wlog({
        "download/final_new": n_new,
        "download/final_total": len(already_done),
    })

    # Build the final ordered accession list from the FASTA on disk
    final_accessions, final_sequences = _parse_fasta(config.fasta_path)
    return final_accessions, final_sequences


# ===================================================================
# Internal helpers
# ===================================================================


def _flush_entry(
    acc: str,
    seq_parts: List[str],
    already_done: set,
    existing_sequences: Dict[str, str],
    fasta_fh,
) -> None:
    """Write a single FASTA entry if it's not already present."""
    if acc in already_done:
        return
    seq = "".join(seq_parts)
    if not seq:
        return
    fasta_fh.write(f">{acc}\n{seq}\n")
    fasta_fh.flush()
    already_done.add(acc)
    existing_sequences[acc] = seq


def _parse_fasta(fasta_path: Path) -> Tuple[List[str], Dict[str, str]]:
    """Parse a simple FASTA file into accessions and sequences.

    Supports two header formats::

        >P12345
        >sp|P12345|PROT_HUMAN Some description

    Args:
        fasta_path: Path to the FASTA file.  If the file does not exist,
            returns empty results.

    Returns:
        A tuple ``(accession_list, {accession: sequence})``.  The list
        preserves the order of entries in the file.
    """
    if not fasta_path.exists():
        return [], {}

    accessions: List[str] = []
    sequences: Dict[str, str] = {}
    current_acc: Optional[str] = None
    current_seq_parts: List[str] = []

    with open(fasta_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Flush previous entry
                if current_acc is not None:
                    sequences[current_acc] = "".join(current_seq_parts)
                    accessions.append(current_acc)
                # Parse the new header
                header = line[1:].split()[0]
                parts = header.split("|")
                # Handle >sp|P12345|NAME  or  >P12345
                current_acc = parts[1] if len(parts) >= 2 else parts[0]
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    # Flush last entry
    if current_acc is not None:
        sequences[current_acc] = "".join(current_seq_parts)
        accessions.append(current_acc)

    return accessions, sequences
