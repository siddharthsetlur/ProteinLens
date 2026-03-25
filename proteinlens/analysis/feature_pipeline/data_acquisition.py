"""Stage 0a — Download reviewed SwissProt sequences as a FASTA file.

This module fetches accession lists from UniProt and then downloads the
individual sequences, writing them incrementally to a FASTA file.  The
download is **resumable**: accessions already present in the FASTA are
skipped on restart so a killed job picks up where it left off.
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
    """Download SwissProt sequences and write them to a FASTA file.

    This is the main entry point for Stage 0a.  It:

    1. Fetches the list of SwissProt accessions for the configured organism.
    2. Reads any accessions already present in the output FASTA (for
       resumability).
    3. Downloads missing sequences one at a time (with connection pooling)
       and appends them to the FASTA file.

    Args:
        config: Pipeline configuration (uses ``organism_taxid``,
            ``max_proteins``, ``fasta_path``, ``max_seq_len``).

    Returns:
        A tuple ``(accessions, sequences)`` where:
        - *accessions* is the ordered list of accession strings that are
          in the FASTA (i.e. those that were successfully downloaded and
          whose length is <= ``config.max_seq_len``).
        - *sequences* is a dict mapping accession -> sequence string.
    """
    # Step 1 — get the target accession list from UniProt
    all_accessions = fetch_swissprot_accessions(
        organism_taxid=config.organism_taxid,
        max_proteins=config.max_proteins,
    )

    # Step 2 — read accessions that are already in the FASTA (resume support)
    existing_accessions, existing_sequences = _parse_fasta(config.fasta_path)
    already_done = set(existing_accessions)
    print(
        f"[data_acquisition] {len(already_done)} accessions already in "
        f"{config.fasta_path.name}, will skip those."
    )

    # Step 3 — download missing sequences and append to FASTA
    session = requests.Session()
    n_downloaded = 0
    n_skipped_long = 0
    n_failed = 0

    with open(config.fasta_path, "a") as fasta_fh:
        for i, acc in enumerate(all_accessions):
            if acc in already_done:
                continue

            seq = fetch_sequence(acc, session)
            if seq is None:
                n_failed += 1
                continue

            # Skip proteins that exceed the ESM context window
            if len(seq) > config.max_seq_len:
                n_skipped_long += 1
                continue

            # Write FASTA entry: >accession\nsequence\n
            fasta_fh.write(f">{acc}\n{seq}\n")
            fasta_fh.flush()

            existing_sequences[acc] = seq
            already_done.add(acc)
            n_downloaded += 1

            # Progress logging every 500 proteins
            if n_downloaded % 500 == 0:
                print(
                    f"[data_acquisition]   downloaded {n_downloaded} "
                    f"(total in file: {len(already_done)}) ..."
                )

            # Polite rate-limiting: small delay to avoid hammering UniProt
            if n_downloaded % 100 == 0:
                time.sleep(0.5)

    print(
        f"[data_acquisition] Done. {n_downloaded} new, "
        f"{n_skipped_long} skipped (too long), {n_failed} failed. "
        f"Total in FASTA: {len(already_done)}."
    )

    # Build the final ordered accession list from the FASTA on disk
    # (so the order matches what is actually persisted)
    final_accessions, final_sequences = _parse_fasta(config.fasta_path)
    return final_accessions, final_sequences


# ===================================================================
# Internal helpers
# ===================================================================


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
