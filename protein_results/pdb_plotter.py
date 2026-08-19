from __future__ import annotations
from io import StringIO
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import yaml 

def _clean_pdb_text(pdb_text: str) -> str:
    """Keep only records the PDBParser understands."""
    return "\n".join(
        line for line in pdb_text.splitlines()
        if line.startswith(("ATOM", "HETATM", "TER", "END"))
    )

def ca_backbone(pdb_text: str, chain_id: str | None ):
    """Plot the Cα backbone from an in-memory PDB text string."""
    cleaned = _clean_pdb_text(pdb_text)
    if not cleaned.strip():
        raise ValueError("No PDB ATOM/HETATM lines found in the input text.")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", StringIO(cleaned))
    model = structure[0]

    # Choose chain
    if chain_id is None:
        chain = next(iter(model))
    else:
        try:
            chain = model[chain_id]
        except KeyError:
            available = [c.id for c in model]
            raise ValueError(f"Chain '{chain_id}' not found. Available chains: {available}")

    # Collect Cα coordinates
    coords = []
    for res in chain:
        hetflag, resseq, icode = res.id
        if hetflag == " " and res.has_id("CA"):
            coords.append(res["CA"].coord)

    if not coords:
        raise ValueError("No Ca atoms found for the selected chain.")

    coords = np.array(coords, dtype=float)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot(coords[:, 0], coords[:, 1], coords[:, 2])
    # ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    # plt.tight_layout()
    # plt.show()
    return coords


def ca_backbone_from_yaml_entry(yaml_path: str | Path, entry_key: str, chain_id: str | None = "A"):
    """
    Load a YAML mapping like:
      A2AIP0: {pdb: "<pdb text>"}
      O35089: {pdb: "<pdb text>"}
    and plot the specified entry.
    """
    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))

    if entry_key not in data:
        available = list(data.keys())
        raise KeyError(f"Entry '{entry_key}' not found. Available keys: {available[:10]}{'...' if len(available)>10 else ''}")

    pdb_text = data[entry_key].get("pdb")
    if not pdb_text:
        raise ValueError(f"Entry '{entry_key}' has no 'pdb' field.")

    # Some ESM/YAML dumps escape soft-wrapped lines with backslash-newline. Normalize just in case:
    pdb_text = pdb_text.replace("\r\n", "\n").replace("\\\n", "")

    ca_backbone(pdb_text, chain_id=chain_id)

# ---- usage ----

def detect_alpha_helices_from_ca(coords: np.ndarray, min_len: int = 6):
    """
    Identify alpha-helical segments using only Ca geometry.
    Returns list of (start_idx, end_idx) with end_idx exclusive.
    """
    n = coords.shape[0]
    if n < min_len:
        return []

    helical = np.zeros(n, dtype=bool)

    # Mark positions that look alpha-helical in a local window
    for i in range(n - 4):
        d_i3 = np.linalg.norm(coords[i]   - coords[i+3])
        d_i4 = np.linalg.norm(coords[i]   - coords[i+4])

        if (4.8 <= d_i3 <= 6.4) and (5.6 <= d_i4 <= 7.4):
            # Mark this local stretch as helical-like
            helical[i:i+5] = True

    # Merge into continuous segments
    helices = []
    i = 0
    while i < n:
        if helical[i]:
            start = i
            while i < n and helical[i]:
                i += 1
            end = i
            if end - start >= min_len:
                helices.append((start, end))
        else:
            i += 1

    return helices