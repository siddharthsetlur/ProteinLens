"""Tests for Stage 5b — InterPro API client (parsing, caching, rate limiting).

Unit tests use a hardcoded sample InterPro API response to verify
correct parsing into InterProDomain objects.  The integration test
(marked ``@pytest.mark.integration``) hits the real EBI API.
"""

import json
import time

import pytest
import requests

from proteinlens.analysis.feature_pipeline.interpro_api import (
    InterProDomain,
    RateLimiter,
    _load_cached,
    _parse_interpro_response,
    _save_cached,
    fetch_interpro_annotations,
)


# ===================================================================
# Sample API response for unit testing
# ===================================================================

# This is a simplified but structurally accurate representation of what
# the InterPro API returns for a protein with 2 InterPro entries.
SAMPLE_INTERPRO_RESPONSE = {
    "results": [
        {
            "metadata": {
                "accession": "IPR000796",
                "name": "Aspartate aminotransferase",
                "type": "Family",
                "member_databases": {
                    "pfam": {"PF00155": {}},
                    "smart": {"SM00432": {}},
                },
            },
            "proteins": [
                {
                    "entry_protein_locations": [
                        {
                            "fragments": [
                                {"start": 45, "end": 210},
                                {"start": 250, "end": 300},
                            ]
                        }
                    ]
                }
            ],
        },
        {
            "metadata": {
                "accession": "IPR004839",
                "name": "Aminotransferase, class I/classII",
                "type": "Domain",
                "member_databases": {},
            },
            "proteins": [
                {
                    "entry_protein_locations": [
                        {
                            "fragments": [
                                {"start": 50, "end": 200},
                            ]
                        }
                    ]
                }
            ],
        },
    ]
}


class TestParseInterProResponse:
    """Tests for _parse_interpro_response."""

    def test_correct_number_of_domains(self):
        """Should produce one domain per (entry, member_db, fragment) triple.

        Entry 1: 2 member DBs (pfam, smart) x 2 fragments = 4 domains
        Entry 2: 0 member DBs, 1 fragment = 1 domain (InterPro-only)
        Total = 5
        """
        domains = _parse_interpro_response("P12345", SAMPLE_INTERPRO_RESPONSE)
        assert len(domains) == 5

    def test_domain_fields_populated(self):
        """Each domain should have all fields filled in correctly."""
        domains = _parse_interpro_response("P12345", SAMPLE_INTERPRO_RESPONSE)

        # Find the Pfam PF00155 domain at position 45-210
        pfam_domain = [
            d for d in domains
            if d.member_accession == "PF00155" and d.start == 45
        ]
        assert len(pfam_domain) == 1
        d = pfam_domain[0]
        assert d.interpro_accession == "IPR000796"
        assert d.interpro_name == "Aspartate aminotransferase"
        assert d.type == "Family"
        assert d.member_db == "pfam"
        assert d.start == 45
        assert d.end == 210

    def test_entry_without_member_db(self):
        """An entry with no member_databases should still produce domains
        with empty member_db and member_accession."""
        domains = _parse_interpro_response("P12345", SAMPLE_INTERPRO_RESPONSE)

        no_member = [d for d in domains if d.interpro_accession == "IPR004839"]
        assert len(no_member) == 1
        assert no_member[0].member_db == ""
        assert no_member[0].member_accession == ""
        assert no_member[0].start == 50
        assert no_member[0].end == 200

    def test_multiple_fragments(self):
        """Multiple fragments under one entry should produce separate domains."""
        domains = _parse_interpro_response("P12345", SAMPLE_INTERPRO_RESPONSE)

        pfam_domains = [d for d in domains if d.member_accession == "PF00155"]
        starts = sorted([d.start for d in pfam_domains])
        assert starts == [45, 250]

    def test_positions_are_1_based(self):
        """Start and end positions should be 1-based as returned by the API."""
        domains = _parse_interpro_response("P12345", SAMPLE_INTERPRO_RESPONSE)
        for d in domains:
            assert d.start >= 1, f"Start {d.start} is not 1-based"
            assert d.end >= d.start, f"End {d.end} < start {d.start}"

    def test_empty_response(self):
        """An empty results list should produce no domains."""
        domains = _parse_interpro_response("NONE", {"results": []})
        assert domains == []


class TestCaching:
    """Tests for cache save/load round-trip."""

    def test_cache_round_trip(self, tmp_path):
        """Saving and loading should produce identical InterProDomain objects."""
        original = [
            InterProDomain(
                interpro_accession="IPR000796",
                interpro_name="Test Domain",
                type="Family",
                member_db="pfam",
                member_accession="PF00155",
                start=10,
                end=100,
            )
        ]
        cache_path = tmp_path / "TEST.json"
        _save_cached(cache_path, "TEST", original)

        loaded = _load_cached(cache_path)
        assert len(loaded) == 1
        assert loaded[0] == original[0]

    def test_cache_empty_domains(self, tmp_path):
        """Caching an empty domain list (404 result) should round-trip cleanly."""
        cache_path = tmp_path / "EMPTY.json"
        _save_cached(cache_path, "EMPTY", [])

        loaded = _load_cached(cache_path)
        assert loaded == []

    def test_fetch_uses_cache(self, tmp_path):
        """Second call should read from cache, not hit the API.

        We verify this by pre-populating the cache with known data and
        checking that fetch_interpro_annotations returns it without
        making any network calls.
        """
        # Pre-populate cache
        known_domain = InterProDomain(
            interpro_accession="IPR999999",
            interpro_name="Cached Domain",
            type="Domain",
            member_db="pfam",
            member_accession="PF99999",
            start=1,
            end=50,
        )
        _save_cached(tmp_path / "CACHED.json", "CACHED", [known_domain])

        # Fetch — should return cached data without network call
        session = requests.Session()
        rate_limiter = RateLimiter(rate=100.0)
        result = fetch_interpro_annotations(
            "CACHED", tmp_path, session, rate_limiter
        )

        assert len(result) == 1
        assert result[0].interpro_accession == "IPR999999"

    def test_404_cached_as_empty(self, tmp_path):
        """A cached 404 result should return empty domains, not re-fetch."""
        # Simulate a 404 cache entry
        _save_cached(tmp_path / "NOTFOUND.json", "NOTFOUND", [])

        session = requests.Session()
        rate_limiter = RateLimiter(rate=100.0)
        result = fetch_interpro_annotations(
            "NOTFOUND", tmp_path, session, rate_limiter
        )

        assert result == []


class TestRateLimiter:
    """Tests for the token-bucket rate limiter."""

    def test_rate_limiter_spacing(self):
        """Calls should be spaced by at least 1/rate seconds."""
        rate = 10.0  # 10 per second = 0.1s spacing
        limiter = RateLimiter(rate=rate)

        t0 = time.monotonic()
        limiter.wait()
        limiter.wait()
        limiter.wait()
        elapsed = time.monotonic() - t0

        # 3 calls at 10/s should take at least 0.2s (2 intervals)
        assert elapsed >= 0.18, (
            f"Expected >= 0.18s for 3 calls at rate {rate}/s, got {elapsed:.3f}s"
        )

    def test_zero_rate_no_blocking(self):
        """A rate of 0 should not block."""
        limiter = RateLimiter(rate=0)
        t0 = time.monotonic()
        for _ in range(10):
            limiter.wait()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1


@pytest.mark.integration
class TestInterProAPIIntegration:
    """Integration tests that hit the real EBI InterPro API.

    Run with: ``pytest -m integration``
    """

    def test_real_protein_p12345(self, tmp_path):
        """Fetch annotations for P12345 (human serum albumin) from the real API."""
        session = requests.Session()
        rate_limiter = RateLimiter(rate=5.0)

        domains = fetch_interpro_annotations(
            "P12345", tmp_path, session, rate_limiter
        )

        # P12345 should have InterPro annotations
        assert len(domains) > 0, "P12345 should have InterPro domains"

        # Check that domain positions are sensible (1-based, within protein)
        for d in domains:
            assert d.start >= 1
            assert d.end >= d.start
            assert d.interpro_accession.startswith("IPR")

        # Verify cache file was created
        cache_path = tmp_path / "P12345.json"
        assert cache_path.exists()

        # Verify cache content is valid JSON
        with open(cache_path, "r") as f:
            cached = json.load(f)
        assert cached["accession"] == "P12345"
        assert len(cached["domains"]) == len(domains)
