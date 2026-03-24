"""Pipeline configuration dataclass.

Centralises all tuneable knobs for the feature data pipeline so that every
stage reads from one consistent source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PipelineConfig:
    """Configuration for the feature data pipeline.

    Attributes:
        sae_dir: Path to the trained SAE model directory. Must contain
            ``ae.pt`` and ``config.yaml``.
        output_dir: Root directory where all pipeline outputs are written.
            Created automatically if it does not exist.
        organism_taxid: NCBI taxonomy ID for the organism whose proteome
            we process.  Default 9606 = *Homo sapiens*.
        max_proteins: Optional cap on the number of proteins to process.
            Useful for local testing (e.g. ``max_proteins=50``).  ``None``
            means process the full proteome.
        esm_model_name: HuggingFace identifier for the ESM model used to
            produce the embeddings that the SAE was trained on.
        esm_layer: Which ESM hidden-layer to extract embeddings from.
            Must match the layer the SAE was trained on.
        max_seq_len: Proteins longer than this are skipped (ESM context
            limit).
        n_top_per_feature: Number of top-activating proteins to keep per
            feature in the survey pass (Stage 1).
        n_per_bin: Number of sample proteins to keep per normalised
            activation bin per feature (Stage 2 selection).
        activation_bins: Normalised activation bin edges expressed as
            fractions of the feature's global max.  Default gives four
            quartile bins: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0].
        activation_threshold: Minimum activation value (absolute, not
            normalised) for a protein to count as "activated" by a feature.
        survey_checkpoint_every: How many proteins to process between
            checkpoint saves in the survey pass.
        embedding_batch_size: Number of sequences to embed in one ESM
            forward pass.
        device: PyTorch device string.  ``None`` means auto-detect via
            ``get_device()``.
        mmseqs_min_seq_id: Minimum sequence identity for MMseqs2
            clustering (0.0 – 1.0).
    """

    # --- Paths ---
    sae_dir: Path = Path("trained_models/fiery-sweep")
    output_dir: Path = Path("feature_data")

    # --- Dataset scope ---
    organism_taxid: int = 9606
    max_proteins: Optional[int] = None

    # --- Model settings (must match how the SAE was trained) ---
    esm_model_name: str = "facebook/esm2_t6_8M_UR50D"
    esm_layer: int = 3
    max_seq_len: int = 1024

    # --- Survey / selection knobs ---
    n_top_per_feature: int = 20
    n_per_bin: int = 10
    activation_bins: List[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    activation_threshold: float = 0.05

    # --- Operational ---
    survey_checkpoint_every: int = 1000
    embedding_batch_size: int = 8
    device: Optional[str] = None
    mmseqs_min_seq_id: float = 0.3

    def __post_init__(self) -> None:
        """Coerce path-like strings and create output directory."""
        self.sae_dir = Path(self.sae_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Derived paths (read-only properties so they stay in sync)
    # -----------------------------------------------------------------

    @property
    def fasta_path(self) -> Path:
        """Path to the downloaded SwissProt FASTA file."""
        return self.output_dir / "swissprot_human.fasta"

    @property
    def cluster_map_path(self) -> Path:
        """Path to the MMseqs2 cluster assignment TSV."""
        return self.output_dir / "cluster_map.tsv"

    @property
    def feature_max_path(self) -> Path:
        """Path to the (num_features,) numpy array of global max activations."""
        return self.output_dir / "feature_max_activations.npy"

    @property
    def protein_feature_maxes_path(self) -> Path:
        """Path to the (n_proteins, num_features) memmap of per-protein maxes."""
        return self.output_dir / "protein_feature_maxes.npy"

    @property
    def survey_top_path(self) -> Path:
        """Path to the survey top-N JSON."""
        return self.output_dir / "survey_top20.json"

    @property
    def survey_coverage_path(self) -> Path:
        """Path to per-feature coverage JSON."""
        return self.output_dir / "survey_coverage.json"

    @property
    def pipeline_state_path(self) -> Path:
        """Path to the resumability checkpoint JSON."""
        return self.output_dir / "pipeline_state.json"

    @property
    def residue_activations_dir(self) -> Path:
        """Directory for per-protein residue activation ``.npz`` files."""
        d = self.output_dir / "residue_activations"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def pdb_cache_dir(self) -> Path:
        """Directory for cached AlphaFold PDB files."""
        d = self.output_dir / "pdb_cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def features_dir(self) -> Path:
        """Directory for per-feature JSON files."""
        d = self.output_dir / "features"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def sequences_path(self) -> Path:
        """Path to the shared sequences JSON."""
        return self.output_dir / "sequences.json"

    @property
    def dataset_stats_path(self) -> Path:
        """Path to the dataset statistics JSON."""
        return self.output_dir / "dataset_stats.json"

    @property
    def selection_path(self) -> Path:
        """Path to the selection results JSON (which proteins to collect)."""
        return self.output_dir / "selection.json"
