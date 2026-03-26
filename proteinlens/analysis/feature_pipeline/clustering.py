"""Stage 0b — Cluster sequences with MMseqs2.

Runs ``mmseqs easy-cluster`` to group sequences at a given identity
threshold (default 30%).  The resulting cluster map is used downstream
for coverage statistics: we report how many *clusters* (not just
individual proteins) activate each SAE feature, which guards against
inflated counts from large, highly-similar families.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from proteinlens.analysis.feature_pipeline.config import PipelineConfig


# ===================================================================
# Public API
# ===================================================================


def run_mmseqs_clustering(config: PipelineConfig) -> Dict[str, str]:
    """Run MMseqs2 easy-cluster on the downloaded FASTA and persist results.

    This is the main entry point for Stage 0b.  It:

    1. Checks whether MMseqs2 is installed (raises if not).
    2. Runs ``mmseqs easy-cluster`` with ``--min-seq-id`` from config.
    3. Parses the TSV output into a representative -> member mapping.
    4. Writes ``cluster_map.tsv`` into the output directory.

    The TSV file has two columns (tab-separated, no header)::

        representative_accession    member_accession

    Every accession appears exactly once as a member (representatives
    are members of their own cluster).

    Args:
        config: Pipeline configuration (uses ``fasta_path``,
            ``cluster_map_path``, ``mmseqs_min_seq_id``).

    Returns:
        A dict mapping **member accession** -> **representative accession**.
        This lets you look up which cluster any protein belongs to in O(1).

    Raises:
        FileNotFoundError: If the FASTA file does not exist.
        RuntimeError: If MMseqs2 is not installed or the clustering command fails.
    """
    if not config.fasta_path.exists():
        raise FileNotFoundError(
            f"FASTA file not found at {config.fasta_path}. "
            "Run Stage 0a (data_acquisition) first."
        )

    _check_mmseqs_installed()

    # Use a temporary directory for MMseqs2 intermediate files.
    # Only the final cluster_map.tsv is kept.
    with tempfile.TemporaryDirectory(prefix="mmseqs_") as tmp_dir:
        tmp = Path(tmp_dir)
        result_prefix = tmp / "cluster_result"

        # ── Run MMseqs2 easy-cluster ──
        # easy-cluster outputs three files:
        #   cluster_result_cluster.tsv  — representative \t member
        #   cluster_result_all_seqs.fasta
        #   cluster_result_rep_seq.fasta
        cmd = [
            "mmseqs",
            "easy-cluster",
            str(config.fasta_path),
            str(result_prefix),
            str(tmp / "mmseqs_tmp"),
            "--min-seq-id",
            str(config.mmseqs_min_seq_id),
            # Sensible defaults for protein clustering
            "-c",
            "0.8",          # coverage threshold
            "--cov-mode",
            "0",            # bidirectional coverage
        ]
        print(f"[clustering] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"MMseqs2 clustering failed (exit code {result.returncode}).\n"
                f"stderr: {result.stderr}"
            )

        # ── Parse the TSV output ──
        tsv_path = Path(str(result_prefix) + "_cluster.tsv")
        if not tsv_path.exists():
            raise RuntimeError(
                f"Expected MMseqs2 output {tsv_path} not found. "
                f"Files in tmp dir: {list(tmp.iterdir())}"
            )

        member_to_rep = _parse_mmseqs_tsv(tsv_path)

    # ── Persist to the pipeline output directory ──
    _write_cluster_map(member_to_rep, config.cluster_map_path)
    n_members = len(member_to_rep)
    n_clusters = len(set(member_to_rep.values()))
    print(
        f"[clustering] Wrote {config.cluster_map_path} "
        f"({n_members} members, {n_clusters} clusters)."
    )
    from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog

    wlog({"cluster/members": n_members, "cluster/clusters": n_clusters})
    return member_to_rep


def load_cluster_map(config: PipelineConfig) -> Dict[str, str]:
    """Load a previously computed cluster map from disk.

    Args:
        config: Pipeline configuration (uses ``cluster_map_path``).

    Returns:
        Dict mapping member accession -> representative accession.

    Raises:
        FileNotFoundError: If the cluster map TSV does not exist.
    """
    if not config.cluster_map_path.exists():
        raise FileNotFoundError(
            f"Cluster map not found at {config.cluster_map_path}. "
            "Run Stage 0b (clustering) first."
        )
    return _parse_cluster_map_file(config.cluster_map_path)


def get_cluster_representatives(member_to_rep: Dict[str, str]) -> Set[str]:
    """Return the set of unique cluster representative accessions.

    Args:
        member_to_rep: Dict mapping member -> representative.

    Returns:
        Set of representative accession strings.
    """
    return set(member_to_rep.values())


def sample_representative_accessions(
    member_to_rep: Dict[str, str],
    max_proteins: int | None,
) -> Set[str]:
    """Sample cluster representatives, then return all their members.

    If the number of unique cluster representatives exceeds *max_proteins*,
    randomly sample *max_proteins* representatives (deterministic seed).
    Then return the full set of member accessions belonging to those
    selected clusters.

    Args:
        member_to_rep: Dict mapping member -> representative.
        max_proteins: Maximum number of cluster representatives to keep.
            ``None`` means keep all.

    Returns:
        Set of accession strings (all members of the selected clusters).
    """
    import random

    representatives = sorted(set(member_to_rep.values()))
    if max_proteins is not None and len(representatives) > max_proteins:
        rng = random.Random(42)
        representatives = rng.sample(representatives, max_proteins)
        representatives_set = set(representatives)
    else:
        representatives_set = set(representatives)

    # Return all members belonging to selected clusters
    selected = {
        member
        for member, rep in member_to_rep.items()
        if rep in representatives_set
    }
    return selected


def get_clusters(member_to_rep: Dict[str, str]) -> Dict[str, List[str]]:
    """Invert the member->representative map to representative->members.

    Args:
        member_to_rep: Dict mapping member -> representative.

    Returns:
        Dict mapping representative accession -> list of member accessions
        (including the representative itself).
    """
    clusters: Dict[str, List[str]] = defaultdict(list)
    for member, rep in member_to_rep.items():
        clusters[rep].append(member)
    return dict(clusters)


# ===================================================================
# Internal helpers
# ===================================================================


def _check_mmseqs_installed() -> None:
    """Verify that the ``mmseqs`` binary is on PATH.

    Raises:
        RuntimeError: If ``mmseqs`` cannot be found.
    """
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "MMseqs2 is not installed or not on PATH. "
            "Install with: conda install -c bioconda mmseqs2"
        )


def _parse_mmseqs_tsv(tsv_path: Path) -> Dict[str, str]:
    """Parse an MMseqs2 ``_cluster.tsv`` file.

    The file has two tab-separated columns: representative, member.
    Each row says "member belongs to the cluster represented by
    representative".

    Args:
        tsv_path: Path to the ``*_cluster.tsv`` file.

    Returns:
        Dict mapping member accession -> representative accession.
    """
    member_to_rep: Dict[str, str] = {}
    with open(tsv_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                # PM NOTE: MMseqs2 TSV should always have exactly 2 columns.
                # If we ever see a different format, this will fail loudly
                # rather than silently misparse.
                raise ValueError(
                    f"Unexpected MMseqs2 TSV line (expected 2 columns): {line!r}"
                )
            rep, member = parts
            member_to_rep[member] = rep
    return member_to_rep


def _write_cluster_map(
    member_to_rep: Dict[str, str], output_path: Path
) -> None:
    """Write the cluster map as a two-column TSV.

    Format: ``representative\\tmember`` (one row per member, no header).

    Args:
        member_to_rep: Dict mapping member -> representative.
        output_path: Where to write the TSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for member, rep in sorted(member_to_rep.items()):
            fh.write(f"{rep}\t{member}\n")


def _parse_cluster_map_file(path: Path) -> Dict[str, str]:
    """Read back a cluster map TSV written by :func:`_write_cluster_map`.

    Args:
        path: Path to the TSV file.

    Returns:
        Dict mapping member accession -> representative accession.
    """
    member_to_rep: Dict[str, str] = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Bad cluster_map.tsv line (expected 2 columns): {line!r}"
                )
            rep, member = parts
            member_to_rep[member] = rep
    return member_to_rep
