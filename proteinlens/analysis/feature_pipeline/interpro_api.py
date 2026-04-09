"""Stage 5b — InterPro API client with caching and rate limiting.

Fetches domain/family annotations from the InterPro REST API for each
protein selected in Stage 5a.  Results are cached as JSON files so that
re-runs skip already-fetched proteins.

**InterPro API response structure:**

The endpoint ``/entry/interpro/protein/uniprot/{accession}`` returns a
list of InterPro entries that match the protein.  Each entry has metadata
(accession, name, type) plus ``proteins`` containing location data.  A
single InterPro entry can map to multiple member database entries (e.g.
Pfam, PROSITE, SMART), each with their own domain boundaries.

We flatten this into a list of ``InterProDomain`` dataclass instances —
one per (InterPro entry, member DB entry, fragment) combination — so that
downstream F1 analysis can treat each domain boundary independently.

**Rate limiting:**

The EBI InterPro API docs recommend a maximum of ~5 requests per second
for programmatic access.  We use a token-bucket rate limiter to enforce
this, with configurable rate via ``config.interpro_api_rate_limit``.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

from proteinlens.analysis.feature_pipeline.config import PipelineConfig

# InterPro API endpoint for protein annotations
INTERPRO_PROTEIN_URL = (
    "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{accession}"
)


# ===================================================================
# Data model
# ===================================================================


@dataclass
class InterProDomain:
    """A single domain/family annotation for a protein.

    Each instance represents one contiguous region on the protein
    sequence that is annotated by a specific InterPro entry and,
    optionally, a member database entry (Pfam, SMART, etc.).

    Attributes:
        interpro_accession: InterPro accession (e.g. "IPR000796").
        interpro_name: Human-readable name (e.g. "Aspartate aminotransferase").
        type: InterPro entry type — one of "Family", "Domain", "Homologous
            superfamily", "Repeat", "Site", "Active site", "Binding site",
            "Conserved site", "PTM".
        member_db: Member database name (e.g. "pfam", "smart", "prosite").
            May be empty string if the annotation comes only from InterPro
            with no member database cross-reference.
        member_accession: Member database accession (e.g. "PF00155").
            May be empty string if no member database.
        start: 1-based inclusive start position on the protein sequence.
        end: 1-based inclusive end position on the protein sequence.
    """

    interpro_accession: str
    interpro_name: str
    type: str
    member_db: str
    member_accession: str
    start: int
    end: int


# ===================================================================
# Rate limiter
# ===================================================================


class RateLimiter:
    """Simple interval-based rate limiter for API calls.

    Ensures that at most ``rate`` calls per second are made.  Each call
    to :meth:`wait` blocks until the minimum interval since the last
    call has elapsed.  This is a fixed-interval limiter (no burst
    capacity), which is appropriate for sequential API access.

    Args:
        rate: Maximum number of requests per second.
    """

    def __init__(self, rate: float) -> None:
        self.rate = rate
        self.min_interval = 1.0 / rate if rate > 0 else 0.0
        self._last_call: float = 0.0

    def wait(self) -> None:
        """Block until a rate-limit token is available."""
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


# ===================================================================
# API fetching
# ===================================================================


def fetch_interpro_annotations(
    accession: str,
    cache_dir: Path,
    session: requests.Session,
    rate_limiter: RateLimiter,
) -> List[InterProDomain]:
    """Fetch InterPro annotations for a single UniProt accession.

    Checks the local cache first.  On cache miss, queries the InterPro
    REST API, parses the response into a list of ``InterProDomain``
    objects, and caches the result.

    On HTTP 404 (protein not in InterPro), caches an empty result so
    we don't re-fetch on subsequent runs.

    On transient errors (5xx, timeout), retries up to 3 times with
    exponential backoff.

    Args:
        accession: UniProt accession (e.g. "P12345").
        cache_dir: Directory for cached JSON files.
        session: ``requests.Session`` for connection pooling.
        rate_limiter: Rate limiter instance.

    Returns:
        List of ``InterProDomain`` objects.  May be empty if the protein
        has no InterPro annotations or if the API returned 404.
    """
    # ── Check cache ──
    cache_path = cache_dir / f"{accession}.json"
    if cache_path.exists():
        return _load_cached(cache_path)

    # ── Fetch from API with retries ──
    url = INTERPRO_PROTEIN_URL.format(accession=accession)
    max_retries = 3
    backoff = 1.0

    for attempt in range(max_retries):
        rate_limiter.wait()
        try:
            resp = session.get(url, timeout=30)

            if resp.status_code == 404:
                # Protein not in InterPro — cache empty result
                _save_cached(cache_path, accession, [])
                return []

            if resp.status_code == 200:
                response_json = resp.json()
                domains = _parse_interpro_response(accession, response_json)

                # Handle pagination: the InterPro API may split results
                # across multiple pages.  Each page has a "next" URL.
                next_url = response_json.get("next")
                while next_url:
                    rate_limiter.wait()
                    page_resp = session.get(next_url, timeout=30)
                    if page_resp.status_code != 200:
                        break
                    page_json = page_resp.json()
                    domains.extend(
                        _parse_interpro_response(accession, page_json)
                    )
                    next_url = page_json.get("next")

                _save_cached(cache_path, accession, domains)
                return domains

            # 5xx or unexpected status — retry
            if resp.status_code >= 500:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                # Final attempt failed
                print(
                    f"[interpro_api] WARNING: {accession} returned "
                    f"HTTP {resp.status_code} after {max_retries} retries"
                )
                _save_cached(cache_path, accession, [])
                return []

            # 4xx (not 404) — don't retry, cache empty
            print(
                f"[interpro_api] WARNING: {accession} returned "
                f"HTTP {resp.status_code}"
            )
            _save_cached(cache_path, accession, [])
            return []

        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            print(
                f"[interpro_api] WARNING: {accession} failed after "
                f"{max_retries} retries: {e}"
            )
            _save_cached(cache_path, accession, [])
            return []

    # Should not reach here, but be safe
    return []


# ===================================================================
# Public orchestrator
# ===================================================================


def run_interpro_fetch(config: PipelineConfig) -> None:
    """Execute the InterPro fetch stage (Stage 5b).

    Loads the InterPro selection JSON from Stage 5a, then fetches
    annotations for all selected proteins that are not already cached.

    Args:
        config: Pipeline configuration.  Requires that Stage 5a
            has completed, providing ``interpro_selection.json``.

    Raises:
        FileNotFoundError: If ``interpro_selection.json`` does not exist.
    """
    with open(config.interpro_selection_path, "r") as f:
        selection = json.load(f)

    all_accessions = selection["all_selected_accessions"]

    # Filter out already-cached accessions
    cache_dir = config.interpro_cache_dir
    todo = [
        acc for acc in all_accessions
        if not (cache_dir / f"{acc}.json").exists()
    ]

    print(
        f"[interpro_fetch] {len(all_accessions)} proteins selected, "
        f"{len(todo)} need fetching, "
        f"{len(all_accessions) - len(todo)} already cached."
    )

    if not todo:
        print("[interpro_fetch] Nothing to do.")
        return

    n_workers = min(int(os.environ.get("PIPELINE_WORKERS", "1")), 4)
    per_worker_rate = config.interpro_api_rate_limit / max(n_workers, 1)

    n_with_annotations = 0
    n_empty = 0

    def _fetch_one(acc: str, session: requests.Session, rl: RateLimiter) -> bool:
        domains = fetch_interpro_annotations(acc, cache_dir, session, rl)
        return bool(domains)

    if n_workers > 1:
        print(f"[interpro_fetch] Using {n_workers} parallel workers "
              f"({per_worker_rate:.1f} req/s each)")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            sessions_and_rls = [
                (requests.Session(), RateLimiter(per_worker_rate))
                for _ in range(n_workers)
            ]
            for i, acc in enumerate(todo):
                s, rl = sessions_and_rls[i % n_workers]
                futures[pool.submit(_fetch_one, acc, s, rl)] = acc
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="Fetching InterPro annotations"):
                if fut.result():
                    n_with_annotations += 1
                else:
                    n_empty += 1
    else:
        session = requests.Session()
        rate_limiter = RateLimiter(config.interpro_api_rate_limit)
        for acc in tqdm(todo, desc="Fetching InterPro annotations"):
            if _fetch_one(acc, session, rate_limiter):
                n_with_annotations += 1
            else:
                n_empty += 1

    print(
        f"[interpro_fetch] Fetched annotations for {len(todo)} proteins "
        f"({n_with_annotations} with annotations, {n_empty} not in InterPro)"
    )
    from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog

    wlog({
        "interpro_fetch/total_fetched": len(todo),
        "interpro_fetch/with_annotations": n_with_annotations,
        "interpro_fetch/empty": n_empty,
    })


# ===================================================================
# Internal helpers
# ===================================================================


def _parse_interpro_response(
    accession: str,
    response_json: dict,
) -> List[InterProDomain]:
    """Parse the InterPro API JSON response into InterProDomain objects.

    The API returns a paginated response with ``results`` as a list of
    InterPro entries.  Each entry has:
    - ``metadata``: accession, name, type
    - ``proteins``: list with one entry per protein, containing
      ``entry_protein_locations`` which has domain boundary fragments
    - ``metadata.member_databases``: dict of member DB cross-references

    We extract one ``InterProDomain`` per (entry, member_db, fragment)
    triple.  If an entry has no member databases, we still create
    domains with empty member_db/member_accession strings.

    Args:
        accession: The UniProt accession we queried for (used only for
            context in error messages).
        response_json: Parsed JSON from the InterPro API.

    Returns:
        Flat list of ``InterProDomain`` objects.
    """
    domains: List[InterProDomain] = []
    results = response_json.get("results", [])

    for entry in results:
        metadata = entry.get("metadata", {})
        ipr_acc = metadata.get("accession", "")
        ipr_name = metadata.get("name", "")
        ipr_type = metadata.get("type", "")

        # Extract domain boundary fragments from the protein locations.
        # The structure is: proteins[0].entry_protein_locations[*].fragments[*]
        # Each fragment has "start" and "end" (1-based, inclusive).
        fragments = _extract_fragments(entry)

        # Extract member database cross-references
        member_dbs = metadata.get("member_databases", {})

        if member_dbs:
            # Create one domain per member DB entry per fragment
            for db_name, db_entries in member_dbs.items():
                # PM NOTE: db_entries is a dict keyed by member accession.
                # Each value may have additional metadata but we only need
                # the accession string (the key).
                if isinstance(db_entries, dict):
                    for member_acc in db_entries:
                        for start, end in fragments:
                            domains.append(InterProDomain(
                                interpro_accession=ipr_acc,
                                interpro_name=ipr_name,
                                type=ipr_type,
                                member_db=db_name,
                                member_accession=member_acc,
                                start=start,
                                end=end,
                            ))
        else:
            # No member databases — still record the InterPro entry itself
            for start, end in fragments:
                domains.append(InterProDomain(
                    interpro_accession=ipr_acc,
                    interpro_name=ipr_name,
                    type=ipr_type,
                    member_db="",
                    member_accession="",
                    start=start,
                    end=end,
                ))

    return domains


def _extract_fragments(entry: dict) -> List[tuple]:
    """Extract (start, end) fragment tuples from an InterPro entry.

    Navigates the nested ``proteins[0].entry_protein_locations[*].fragments[*]``
    structure.  Each fragment has ``start`` and ``end`` keys with 1-based
    inclusive positions.

    Args:
        entry: A single entry dict from the InterPro API ``results`` list.

    Returns:
        List of ``(start, end)`` tuples (1-based inclusive).
        Returns an empty list if no fragments are found, so that
        the entry produces no ``InterProDomain`` objects and does
        not inject phantom residue-level domain labels.
    """
    fragments = []
    proteins = entry.get("proteins", [])

    for protein in proteins:
        locations = protein.get("entry_protein_locations", [])
        for location in locations:
            for frag in location.get("fragments", []):
                start = frag.get("start")
                end = frag.get("end")
                if start is not None and end is not None:
                    fragments.append((int(start), int(end)))

    if not fragments:
        # No parseable fragment positions.  Return empty so this entry
        # contributes no residue-level domain labels.  The InterPro entry
        # is still counted at the protein level (annotation presence is
        # determined by whether _parse_interpro_response yields any
        # InterProDomain objects at all, not by fragment count).
        return []

    return fragments


def _save_cached(
    cache_path: Path,
    accession: str,
    domains: List[InterProDomain],
) -> None:
    """Save parsed InterPro annotations to the cache.

    Args:
        cache_path: Path to the cache JSON file.
        accession: UniProt accession.
        domains: Parsed domain list to cache.
    """
    cache_data = {
        "accession": accession,
        "domains": [asdict(d) for d in domains],
    }
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)


def _load_cached(cache_path: Path) -> List[InterProDomain]:
    """Load InterPro annotations from a cache file.

    Args:
        cache_path: Path to the cached JSON file.

    Returns:
        List of ``InterProDomain`` objects reconstructed from the cache.
    """
    with open(cache_path, "r") as f:
        data = json.load(f)
    return [
        InterProDomain(**d)
        for d in data.get("domains", [])
    ]
