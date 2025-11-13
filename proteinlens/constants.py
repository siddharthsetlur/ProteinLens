"""Constants and configuration for InterPLM."""

import os
from pathlib import Path

# Base directory for InterPLM data (can be overridden by environment variable)
DATA_DIR = Path(os.environ.get("INTERPLM_DATA", Path.home() / "interplm_data"))

# Standard amino acids (canonical 20)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_LIST = list(AMINO_ACIDS)
AMINO_ACID_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AMINO_ACID = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# PDB directory for structure files
PDB_DIR = Path(os.environ.get('INTERPLM_DATA', '.')) / 'pdb'
