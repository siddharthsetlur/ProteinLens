"""Tests for Stage 0a — data acquisition (real UniProt API, no mocks).

These tests hit the real UniProt REST API.  They use small queries
(max_proteins=5) to keep runtime and network usage minimal.
"""

import tempfile
from pathlib import Path

import pytest
import requests

from proteinlens.analysis.feature_pipeline.config import PipelineConfig
from proteinlens.analysis.feature_pipeline.data_acquisition import (
    _parse_fasta,
    download_swissprot_fasta,
    fetch_sequence,
    fetch_swissprot_accessions,
)


class TestFetchSwissprotAccessions:
    """Tests for fetching accession lists from UniProt."""

    def test_fetches_human_accessions(self):
        """Should return a non-empty list of accession strings for human."""
        accessions = fetch_swissprot_accessions(
            organism_taxid=9606, max_proteins=10
        )
        assert len(accessions) > 0
        assert len(accessions) <= 10
        # Accessions should be non-empty strings
        for acc in accessions:
            assert isinstance(acc, str)
            assert len(acc) > 0

    def test_max_proteins_cap(self):
        """max_proteins should limit the result count."""
        accessions = fetch_swissprot_accessions(
            organism_taxid=9606, max_proteins=3
        )
        assert len(accessions) <= 3

    def test_ecoli_accessions(self):
        """Should also work for E. coli (taxid 83333)."""
        accessions = fetch_swissprot_accessions(
            organism_taxid=83333, max_proteins=5
        )
        assert len(accessions) > 0


class TestFetchSequence:
    """Tests for fetching individual protein sequences."""

    def test_known_accession(self):
        """P68871 (human hemoglobin beta) should return a valid sequence."""
        session = requests.Session()
        seq = fetch_sequence("P68871", session)
        assert seq is not None
        assert len(seq) > 50  # hemoglobin beta is ~147 aa
        # Should be uppercase amino acids only
        assert seq.isalpha()
        assert seq.isupper()

    def test_nonexistent_accession(self):
        """A made-up accession should return None."""
        session = requests.Session()
        seq = fetch_sequence("ZZZZZZZ_FAKE", session)
        assert seq is None


class TestDownloadSwissprotFasta:
    """Integration test for the full FASTA download workflow."""

    def test_download_small_set(self):
        """Download 5 proteins and verify FASTA integrity."""
        with tempfile.TemporaryDirectory() as tmp:
            config = PipelineConfig(
                output_dir=Path(tmp),
                organism_taxid=9606,
                max_proteins=5,
            )
            accessions, sequences = download_swissprot_fasta(config)

            # Should have downloaded some proteins (not all may succeed)
            assert len(accessions) > 0
            assert len(sequences) == len(accessions)

            # FASTA file should exist
            assert config.fasta_path.exists()

            # Verify parse round-trip
            parsed_acc, parsed_seq = _parse_fasta(config.fasta_path)
            assert parsed_acc == accessions
            assert parsed_seq == sequences

    def test_resumability(self):
        """Re-running should skip already-downloaded proteins."""
        with tempfile.TemporaryDirectory() as tmp:
            config = PipelineConfig(
                output_dir=Path(tmp),
                organism_taxid=9606,
                max_proteins=3,
            )
            # First run
            acc1, seq1 = download_swissprot_fasta(config)
            size1 = config.fasta_path.stat().st_size

            # Second run — should not append duplicates
            acc2, seq2 = download_swissprot_fasta(config)
            size2 = config.fasta_path.stat().st_size

            assert size2 == size1
            assert acc2 == acc1


class TestParseFasta:
    """Tests for the FASTA parser."""

    def test_parse_simple_fasta(self):
        """Should parse a simple two-entry FASTA."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">P12345\nMKTL\nAGEK\n>Q67890\nMVLS\n")
            fasta_path = Path(f.name)

        accessions, sequences = _parse_fasta(fasta_path)
        assert accessions == ["P12345", "Q67890"]
        assert sequences["P12345"] == "MKTLAGEK"
        assert sequences["Q67890"] == "MVLS"
        fasta_path.unlink()

    def test_parse_sp_header_format(self):
        """Should handle >sp|P12345|NAME_HUMAN style headers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">sp|P12345|HBB_HUMAN Hemoglobin\nMVHLT\n")
            fasta_path = Path(f.name)

        accessions, sequences = _parse_fasta(fasta_path)
        assert accessions == ["P12345"]
        assert sequences["P12345"] == "MVHLT"
        fasta_path.unlink()

    def test_parse_nonexistent_file(self):
        """Should return empty results for a missing file."""
        accessions, sequences = _parse_fasta(Path("/nonexistent/file.fasta"))
        assert accessions == []
        assert sequences == {}
