"""Tests for Stage 0b — MMseqs2 clustering (real MMseqs2, no mocks).

These tests require MMseqs2 to be installed.  They operate on small
FASTA files (5-10 sequences) so clustering completes in seconds.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from proteinlens.analysis.feature_pipeline.clustering import (
    _check_mmseqs_installed,
    _parse_mmseqs_tsv,
    get_cluster_representatives,
    get_clusters,
    load_cluster_map,
    run_mmseqs_clustering,
    sample_representative_accessions,
)
from proteinlens.analysis.feature_pipeline.config import PipelineConfig


# Skip all tests if MMseqs2 is not installed
pytestmark = pytest.mark.skipif(
    shutil.which("mmseqs") is None,
    reason="MMseqs2 not installed",
)

# A small FASTA with proteins of varying similarity.
# These are short fragments chosen so that some cluster together at 30%.
SMALL_FASTA = """\
>SEQ1
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
>SEQ2
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
>SEQ3
MGHFTEEDKATITSLWGKVNVEDAGGETLGRLLVVYPWTQRFFDSFGNLSS
>SEQ4
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV
>SEQ5
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV
"""


class TestRunMmseqsClustering:
    """Integration tests for the full clustering workflow."""

    def test_cluster_small_fasta(self):
        """Clustering 5 sequences should produce a valid cluster map."""
        with tempfile.TemporaryDirectory() as tmp:
            config = PipelineConfig(output_dir=Path(tmp))
            # Write the test FASTA
            config.fasta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config.fasta_path, "w") as f:
                f.write(SMALL_FASTA)

            member_to_rep = run_mmseqs_clustering(config)

            # Every sequence should appear as a member
            assert len(member_to_rep) == 5
            for seq_id in ["SEQ1", "SEQ2", "SEQ3", "SEQ4", "SEQ5"]:
                assert seq_id in member_to_rep

            # Identical sequences should share a representative
            assert member_to_rep["SEQ1"] == member_to_rep["SEQ2"]
            assert member_to_rep["SEQ4"] == member_to_rep["SEQ5"]

            # Cluster map TSV should exist on disk
            assert config.cluster_map_path.exists()

    def test_load_cluster_map(self):
        """load_cluster_map should reproduce the original mapping."""
        with tempfile.TemporaryDirectory() as tmp:
            config = PipelineConfig(output_dir=Path(tmp))
            with open(config.fasta_path, "w") as f:
                f.write(SMALL_FASTA)

            original = run_mmseqs_clustering(config)
            loaded = load_cluster_map(config)

            assert loaded == original


class TestClusterUtilities:
    """Tests for helper functions."""

    def test_get_cluster_representatives(self):
        """Should return the unique set of representative accessions."""
        member_to_rep = {"A": "A", "B": "A", "C": "C"}
        reps = get_cluster_representatives(member_to_rep)
        assert reps == {"A", "C"}

    def test_get_clusters(self):
        """Should invert the mapping to representative -> members."""
        member_to_rep = {"A": "A", "B": "A", "C": "C", "D": "C"}
        clusters = get_clusters(member_to_rep)
        assert set(clusters["A"]) == {"A", "B"}
        assert set(clusters["C"]) == {"C", "D"}

    def test_sample_representative_accessions_no_cap(self):
        """With max_proteins=None, all representatives should be returned."""
        member_to_rep = {"A": "A", "B": "A", "C": "C", "D": "C", "E": "E"}
        sampled = sample_representative_accessions(member_to_rep, None)
        # Returns only representatives, not all members
        assert sampled == {"A", "C", "E"}

    def test_sample_representative_accessions_with_cap(self):
        """With max_proteins < num_clusters, should sample exactly that many."""
        # 10 clusters, each with 2 members = 20 total
        member_to_rep = {}
        for i in range(10):
            rep = f"R{i}"
            member_to_rep[rep] = rep
            member_to_rep[f"M{i}"] = rep
        sampled = sample_representative_accessions(member_to_rep, 3)
        # Should have exactly 3 representative proteins
        assert len(sampled) == 3
        # All returned accessions should be cluster representatives
        all_reps = set(member_to_rep.values())
        assert sampled.issubset(all_reps)

    def test_sample_representative_accessions_deterministic(self):
        """Sampling should be deterministic (seed=42)."""
        member_to_rep = {f"P{i}": f"R{i % 20}" for i in range(100)}
        s1 = sample_representative_accessions(member_to_rep, 5)
        s2 = sample_representative_accessions(member_to_rep, 5)
        assert s1 == s2

    def test_parse_mmseqs_tsv(self):
        """Should parse a two-column TSV correctly."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False
        ) as f:
            f.write("REP1\tMEM1\nREP1\tMEM2\nREP2\tMEM3\n")
            tsv_path = Path(f.name)

        result = _parse_mmseqs_tsv(tsv_path)
        assert result == {"MEM1": "REP1", "MEM2": "REP1", "MEM3": "REP2"}
        tsv_path.unlink()
