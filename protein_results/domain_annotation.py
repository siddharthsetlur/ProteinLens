"""
Pfam domain annotation using **pyhmmer** and (optionally) **biotite**.

Scans protein sequences against the Pfam-A HMM database to produce
per-protein domain-count vectors, e.g.:

    {"zf-C2H2": 3, "WD40": 0, "AAA": 1}

These counts serve as categorical/structural features that complement
the continuous geometric features (writhe, curvature, …) already in the
pipeline.  Concatenating the two gives a combined feature matrix:

    [ geometric_features | pfam_domain_counts ]

which is then fed to LassoCV so the regressor can discover nodes that
respond to domain identity, domain multiplicity, or combinations of
geometry + domain architecture.

Dependencies
~~~~~~~~~~~~
    pip install pyhmmer           # required — HMMER3 bindings
    pip install biotite           # optional — cleaner PDB sequence extraction

The Pfam-A HMM database (~270 MB compressed, ~1.4 GB uncompressed) is
downloaded automatically on first use and cached locally.

Usage as a library
~~~~~~~~~~~~~~~~~~
    from domain_annotation import annotate_domains_from_pdb_cache

    domain_matrix, domain_names, domain_counts = \\
        annotate_domains_from_pdb_cache(
            accessions, pdb_cache_dir, pfam_dir,
        )

Usage standalone (for testing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    python domain_annotation.py \\
        --pdb-cache ./alphafold_analysis/pdb_cache \\
        --pfam-dir  ./alphafold_analysis/pfam \\
        --accessions ./alphafold_analysis/processed_accessions.txt
"""

from __future__ import annotations

import gzip
import io
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# pyhmmer (required)
# ---------------------------------------------------------------------------
try:
    import pyhmmer
    from pyhmmer.easel import Alphabet, TextSequence
    from pyhmmer.plan7 import HMMFile
except ImportError:
    sys.exit(
        "pyhmmer is required for domain annotation.\n"
        "Install it with:  pip install pyhmmer"
    )

# ---------------------------------------------------------------------------
# biotite (optional — falls back to regex PDB parser)
# ---------------------------------------------------------------------------
try:
    import biotite.structure.io.pdb as bpdb
    import biotite.structure as bstruc

    HAS_BIOTITE = True
except ImportError:
    HAS_BIOTITE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PFAM_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
)

DEFAULT_MIN_DOMAIN_FREQ = 5    # min proteins a domain must appear in
DEFAULT_E_VALUE = 1e-5         # domain E-value (if not using GA thresholds)

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Download / cache the Pfam HMM database
# ═══════════════════════════════════════════════════════════════════════════

def ensure_pfam_hmm(dest_dir: Path, url: str = PFAM_URL) -> Path:
    """
    Download Pfam-A.hmm.gz and decompress if not already cached.

    Returns the path to the uncompressed ``Pfam-A.hmm`` file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    hmm_path = dest_dir / "Pfam-A.hmm"
    hmm_gz = dest_dir / "Pfam-A.hmm.gz"

    if hmm_path.exists():
        print(f"  [pfam] Using cached Pfam-A.hmm → {hmm_path}")
        return hmm_path

    # Download
    if not hmm_gz.exists():
        print(f"  [pfam] Downloading Pfam-A.hmm.gz (~270 MB) …")
        urllib.request.urlretrieve(url, str(hmm_gz))
        print(f"  [pfam] Downloaded → {hmm_gz}")

    # Decompress
    print(f"  [pfam] Decompressing Pfam-A.hmm.gz → Pfam-A.hmm …")
    with gzip.open(hmm_gz, "rb") as f_in, open(hmm_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"  [pfam] Pfam HMM database ready → {hmm_path}")
    return hmm_path


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Sequence extraction from PDB text
# ═══════════════════════════════════════════════════════════════════════════

def _sequence_from_pdb_biotite(
    pdb_text: str, chain_id: str = "A",
) -> str | None:
    """Extract amino-acid sequence from PDB text using **biotite**."""
    try:
        pdb_file = bpdb.PDBFile.read(io.StringIO(pdb_text))
        atom_array = pdb_file.get_structure(model=1)
        chain = atom_array[atom_array.chain_id == chain_id]
        ca = chain[chain.atom_name == "CA"]
        seq = "".join(THREE_TO_ONE.get(r, "X") for r in ca.res_name)
        return seq if seq else None
    except Exception:
        return None


def _sequence_from_pdb_regex(pdb_text: str) -> str | None:
    """Fallback: extract sequence from PDB ATOM/CA records with regex."""
    seen: set[str] = set()
    seq: list[str] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        if atom != "CA":
            continue
        resseq = line[22:27].strip()
        if resseq in seen:
            continue
        seen.add(resseq)
        resname = line[17:20].strip()
        seq.append(THREE_TO_ONE.get(resname, "X"))
    return "".join(seq) if seq else None


def extract_sequence(pdb_text: str, chain_id: str = "A") -> str | None:
    """
    Extract the amino-acid sequence from a PDB text block.

    Prefers **biotite** (handles multi-model, non-standard residues, etc.)
    and falls back to a simple regex parser if biotite is not installed.
    """
    if HAS_BIOTITE:
        seq = _sequence_from_pdb_biotite(pdb_text, chain_id)
        if seq:
            return seq
    return _sequence_from_pdb_regex(pdb_text)


def _sanitize(seq: str) -> str:
    """Replace non-standard amino acids (U, O, B, …) with X for HMMER."""
    return "".join(c if c in VALID_AA else "X" for c in seq.upper())


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Pfam domain scanning with pyhmmer
# ═══════════════════════════════════════════════════════════════════════════

def scan_domains(
    sequences: dict[str, str],
    hmm_path: Path,
    *,
    cpus: int = 4,
    use_gathering_thresholds: bool = True,
    e_value: float = DEFAULT_E_VALUE,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """
    Scan protein sequences against Pfam HMM profiles using
    ``pyhmmer.hmmsearch``.

    Parameters
    ----------
    sequences
        ``{accession: amino_acid_string}``.
    hmm_path
        Path to ``Pfam-A.hmm`` (uncompressed).
    cpus
        Number of threads for HMMER.
    use_gathering_thresholds
        If True (recommended for Pfam), use the manually-curated gathering
        (GA) bit-score thresholds stored in each HMM profile.  These are the
        gold-standard for deciding domain membership.  If False, use
        *e_value* instead.
    e_value
        Domain-level E-value threshold (only used when
        ``use_gathering_thresholds=False``).

    Returns
    -------
    domain_counts
        ``{accession: {pfam_family_name: n_domain_instances, …}}``.
    all_domains
        Sorted list of every Pfam domain name observed across the dataset.
    """
    alphabet = Alphabet.amino()

    # Build digital sequences
    acc_list = list(sequences.keys())
    digital_seqs = []
    for acc in acc_list:
        sanitised = _sanitize(sequences[acc])
        ts = TextSequence(name=acc.encode(), sequence=sanitised)
        digital_seqs.append(ts.digitize(alphabet))

    print(
        f"  [pfam] Scanning {len(digital_seqs)} sequences "
        f"against Pfam ({hmm_path.name}) …"
    )

    domain_counts: dict[str, dict[str, int]] = {a: {} for a in acc_list}

    # Configure search thresholds
    search_kwargs: dict = dict(cpus=cpus)
    if use_gathering_thresholds:
        search_kwargs["bit_cutoffs"] = "gathering"
    else:
        search_kwargs["E"] = e_value
        search_kwargs["incE"] = e_value

    # Stream HMMs from disk — memory-efficient for ~20 K profiles
    n_hmms = 0
    with HMMFile(str(hmm_path)) as hmm_file:
        for top_hits in pyhmmer.hmmsearch(hmm_file, digital_seqs, **search_kwargs):
            n_hmms += 1
            if n_hmms % 2000 == 0:
                n_hits_so_far = sum(
                    len(v) for v in domain_counts.values()
                )
                print(
                    f"    … {n_hmms:>6d} HMM profiles scanned  "
                    f"({n_hits_so_far} domain hits so far)"
                )

            hmm_name = top_hits.query.name if isinstance(top_hits.query.name, str) else top_hits.query.name.decode()
            for hit in top_hits:
                if not hit.included:
                    continue
                seq_name = hit.name if isinstance(hit.name, str) else hit.name.decode()
                n_dom = sum(1 for d in hit.domains if d.included)
                if n_dom > 0:
                    domain_counts[seq_name][hmm_name] = (
                        domain_counts[seq_name].get(hmm_name, 0) + n_dom
                    )

    # Summary
    all_domains = sorted(
        {d for counts in domain_counts.values() for d in counts}
    )
    n_annotated = sum(1 for v in domain_counts.values() if v)
    total_hits = sum(sum(v.values()) for v in domain_counts.values())
    print(
        f"  [pfam] Done — scanned {n_hmms} Pfam profiles.\n"
        f"         {len(all_domains)} distinct domain families detected\n"
        f"         {n_annotated}/{len(acc_list)} proteins have ≥ 1 domain\n"
        f"         {total_hits} total domain instances"
    )
    return domain_counts, all_domains


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Build the domain-count feature matrix
# ═══════════════════════════════════════════════════════════════════════════

def build_domain_matrix(
    accessions: list[str],
    domain_counts: dict[str, dict[str, int]],
    domain_names: list[str],
) -> np.ndarray:
    """
    Assemble a ``(n_proteins × n_domains)`` matrix of domain instance counts.

    Proteins with no hit for a given domain get 0.
    """
    dom_idx = {name: i for i, name in enumerate(domain_names)}
    mat = np.zeros((len(accessions), len(domain_names)), dtype=np.float64)
    for i, acc in enumerate(accessions):
        for dom, cnt in domain_counts.get(acc, {}).items():
            if dom in dom_idx:
                mat[i, dom_idx[dom]] = cnt
    return mat


def filter_rare_domains(
    domain_matrix: np.ndarray,
    domain_names: list[str],
    min_freq: int = DEFAULT_MIN_DOMAIN_FREQ,
) -> tuple[np.ndarray, list[str]]:
    """
    Drop domain columns that appear in fewer than *min_freq* proteins.

    This prevents the Lasso from fitting noise on ultra-rare domains.
    Returns the filtered matrix and updated name list.
    """
    presence = (domain_matrix > 0).sum(axis=0)
    keep = presence >= min_freq
    filtered_names = [n for n, k in zip(domain_names, keep) if k]
    filtered_mat = domain_matrix[:, keep]
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(
            f"  [pfam] Dropped {n_dropped} rare domains "
            f"(present in < {min_freq} proteins); "
            f"keeping {len(filtered_names)}."
        )
    return filtered_mat, filtered_names


# ═══════════════════════════════════════════════════════════════════════════
# 5.  High-level convenience function
# ═══════════════════════════════════════════════════════════════════════════

def annotate_domains_from_pdb_cache(
    accessions: list[str],
    pdb_cache: Path,
    pfam_dir: Path,
    *,
    cpus: int = 4,
    min_freq: int = DEFAULT_MIN_DOMAIN_FREQ,
    use_gathering_thresholds: bool = True,
    e_value: float = DEFAULT_E_VALUE,
) -> tuple[np.ndarray, list[str], dict[str, dict[str, int]]]:
    """
    End-to-end domain annotation from cached AlphaFold PDB files.

    1. Extracts amino-acid sequences from PDB cache (biotite or regex).
    2. Downloads Pfam-A HMMs if needed.
    3. Scans with ``pyhmmer.hmmsearch``.
    4. Builds and filters the domain-count matrix.

    Parameters
    ----------
    accessions : list[str]
        Protein accessions (matching ``AF-{acc}-F1-model_v*.pdb`` in cache).
    pdb_cache : Path
        Directory containing cached AlphaFold PDB files.
    pfam_dir : Path
        Directory for downloading / caching ``Pfam-A.hmm``.
    cpus : int
        Threads for pyhmmer.
    min_freq : int
        Minimum number of proteins a domain must appear in to be retained
        as a feature column.
    use_gathering_thresholds : bool
        Use Pfam's curated GA bit-score thresholds (recommended).
    e_value : float
        Fallback E-value threshold if not using gathering thresholds.

    Returns
    -------
    domain_matrix : np.ndarray
        ``(n_proteins, n_domains)`` float64 count matrix.
    domain_names : list[str]
        Column names (Pfam family short names).
    domain_counts : dict
        Raw per-protein domain counts.
    """
    # 1. Extract sequences
    print(
        f"  [pfam] Extracting sequences from {len(accessions)} "
        f"cached PDB files …"
    )
    sequences: dict[str, str] = {}
    n_skip = 0
    for acc in accessions:
        cached = list(pdb_cache.glob(f"AF-{acc}-F1-model_v*.pdb"))
        if not cached:
            n_skip += 1
            continue
        pdb_text = cached[0].read_text()
        seq = extract_sequence(pdb_text)
        if seq and len(seq) >= 4:
            sequences[acc] = seq
        else:
            n_skip += 1

    method = "biotite" if HAS_BIOTITE else "regex"
    print(
        f"  [pfam] Extracted {len(sequences)} sequences via {method} "
        f"({n_skip} skipped)."
    )

    if not sequences:
        print("  [pfam] ✘ No sequences — skipping domain annotation.")
        return np.zeros((len(accessions), 0)), [], {}

    # 2. Ensure Pfam-A HMMs are available
    hmm_path = ensure_pfam_hmm(pfam_dir)

    # 3. Scan
    domain_counts, all_domains = scan_domains(
        sequences,
        hmm_path,
        cpus=cpus,
        use_gathering_thresholds=use_gathering_thresholds,
        e_value=e_value,
    )

    if not all_domains:
        print("  [pfam] ✘ No domains found — check thresholds.")
        return np.zeros((len(accessions), 0)), [], domain_counts

    # 4. Build & filter matrix
    raw_mat = build_domain_matrix(accessions, domain_counts, all_domains)
    domain_matrix, domain_names = filter_rare_domains(
        raw_mat, all_domains, min_freq=min_freq,
    )

    return domain_matrix, domain_names, domain_counts


# ═══════════════════════════════════════════════════════════════════════════
# CLI for standalone testing
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan proteins against Pfam-A and produce domain counts."
    )
    parser.add_argument(
        "--pdb-cache", type=Path, required=True,
        help="Directory with cached AF-*-F1-model_v*.pdb files.",
    )
    parser.add_argument(
        "--pfam-dir", type=Path, required=True,
        help="Directory to store/cache Pfam-A.hmm.",
    )
    parser.add_argument(
        "--accessions", type=Path, required=True,
        help="Text file with one UniProt accession per line.",
    )
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: same as --pfam-dir).",
    )
    args = parser.parse_args()

    accessions = args.accessions.read_text().strip().split("\n")
    out = args.output or args.pfam_dir
    out.mkdir(parents=True, exist_ok=True)

    domain_matrix, domain_names, domain_counts = (
        annotate_domains_from_pdb_cache(
            accessions, args.pdb_cache, args.pfam_dir,
            cpus=args.cpus, min_freq=args.min_freq,
        )
    )

    # Save
    np.save(out / "domain_matrix.npy", domain_matrix)
    with open(out / "domain_names.json", "w") as f:
        json.dump(domain_names, f, indent=2)
    sparse = {a: c for a, c in domain_counts.items() if c}
    import yaml
    with open(out / "domain_counts.yaml", "w") as f:
        yaml.dump(sparse, f, default_flow_style=False)

    print(f"\nSaved to {out}/:")
    print(f"  domain_matrix.npy   – {domain_matrix.shape}")
    print(f"  domain_names.json   – {len(domain_names)} domains")
    print(f"  domain_counts.yaml  – {len(sparse)} proteins with ≥ 1 domain")


if __name__ == "__main__":
    main()
