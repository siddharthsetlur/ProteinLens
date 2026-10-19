"""Pipeline state management for resumability.

Persists which proteins have been processed, which stage is complete, etc.
so that a crashed run can be restarted without re-doing finished work.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set



def write_json_atomic(path: Path, obj: Any) -> None:
    """Write ``obj`` as JSON to ``path`` via a PID-suffixed temp file + rename.

    A kill mid-write leaves only the temp file, never a truncated ``path`` —
    the per-feature resume checks treat ``path.exists()`` as "done".
    """
    tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


class PipelineState:
    """Tracks pipeline progress across stages for crash-safe resumability.

    The state is stored as a JSON file on disk.  Each write is atomic
    (write-to-temp then rename) to avoid corruption if the process is
    killed mid-write.

    Attributes:
        state_path: Absolute path to the JSON state file on disk.
        data: The in-memory dict that mirrors what is on disk.

    State schema (``data`` dict)::

        {
            "completed_stages": ["stage_0a", "stage_0b", ...],
            "survey_processed_accessions": ["P12345", ...],
            "survey_processed_count": 1234,
            "total_proteins": 20000,
            "accession_index": { "P12345": 0, "Q67890": 1, ... }
        }

    ``accession_index`` maps each accession to its row in the memmap so
    that the survey pass can be resumed without re-scanning the FASTA.
    """

    def __init__(self, state_path: Path) -> None:
        """Initialise from an existing state file or create an empty one.

        Args:
            state_path: Path where the JSON state file lives (or will be
                created).
        """
        self.state_path = Path(state_path)
        if self.state_path.exists():
            with open(self.state_path, "r") as f:
                self.data: Dict = json.load(f)
        else:
            self.data = {
                "completed_stages": [],
                "survey_processed_accessions": [],
                "survey_processed_count": 0,
                "total_proteins": 0,
                "accession_index": {},
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Atomically persist the current state to disk.

        Writes to a temporary file first, then renames.  This guarantees
        that a concurrent reader always sees a complete JSON file even if
        the process is killed between write and rename.

        The temp file name carries this process's PID so that two pipeline
        processes sharing one ``output_dir`` (the parallel branch jobs —
        InterPro / geometry / sequence) cannot write the same scratch path
        and interleave bytes into each other's JSON.
        """
        tmp_path = self.state_path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.data, f, indent=2)
        tmp_path.rename(self.state_path)

    # ------------------------------------------------------------------
    # Stage tracking
    # ------------------------------------------------------------------

    def is_stage_complete(self, stage_name: str) -> bool:
        """Return True if *stage_name* has already been marked complete.

        Args:
            stage_name: An identifier like ``"stage_0a"`` or ``"survey"``.

        Returns:
            Whether the stage appears in the ``completed_stages`` list.
        """
        return stage_name in self.data.get("completed_stages", [])

    def mark_stage_complete(self, stage_name: str) -> None:
        """Record that *stage_name* finished successfully and persist.

        Before writing, the on-disk ``completed_stages`` list is merged into
        the in-memory one.  Without this merge, two pipeline processes
        sharing an ``output_dir`` would each save the snapshot they loaded at
        startup, and whichever finished last would silently erase the other's
        completion marks — which then re-runs finished stages on resume.
        Only this one list is merged: every other field is written by a
        single stage, so there is nothing to reconcile.

        Args:
            stage_name: An identifier like ``"stage_0a"`` or ``"survey"``.
        """
        completed = self.data.setdefault("completed_stages", [])
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    on_disk = json.load(f).get("completed_stages", [])
            except (json.JSONDecodeError, OSError):
                on_disk = []
            for other in on_disk:
                if other not in completed:
                    completed.append(other)
        if stage_name not in completed:
            completed.append(stage_name)
        self.save()

    # ------------------------------------------------------------------
    # Survey-pass tracking
    # ------------------------------------------------------------------

    def get_survey_processed(self) -> Set[str]:
        """Return the set of accessions that have been surveyed so far.

        Returns:
            A set of UniProt accession strings.
        """
        return set(self.data.get("survey_processed_accessions", []))

    def add_survey_processed(self, accessions: List[str]) -> None:
        """Append newly processed accessions and update the count.

        Does **not** call :meth:`save` — the caller is expected to batch
        updates and call :meth:`save` at checkpoint boundaries.

        Args:
            accessions: List of accession strings just processed.
        """
        self.data.setdefault("survey_processed_accessions", []).extend(accessions)
        self.data["survey_processed_count"] = len(
            self.data["survey_processed_accessions"]
        )

    # ------------------------------------------------------------------
    # Accession index (row mapping for the memmap)
    # ------------------------------------------------------------------

    def set_accession_index(self, index: Dict[str, int]) -> None:
        """Store the accession-to-row mapping and persist.

        Args:
            index: Dict mapping accession string to integer row index in
                the memmap file.
        """
        self.data["accession_index"] = index
        self.save()

    def get_accession_index(self) -> Dict[str, int]:
        """Return the accession-to-row mapping.

        Returns:
            Dict mapping accession string to integer row index.
        """
        return self.data.get("accession_index", {})

    def set_total_proteins(self, n: int) -> None:
        """Record the total number of proteins in the dataset and persist.

        Args:
            n: Total protein count.
        """
        self.data["total_proteins"] = n
        self.save()
