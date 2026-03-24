#!/usr/bin/env python3
"""CLI entry point for the feature data pipeline.

Orchestrates all pipeline stages (0a through 4) in sequence, with
support for running individual stages and resuming from checkpoints.

Usage examples::

    # Full pipeline (human proteome, default SAE)
    python scripts/run_feature_pipeline.py

    # Small local test (50 proteins)
    python scripts/run_feature_pipeline.py --max-proteins 50

    # Run only the survey stage
    python scripts/run_feature_pipeline.py --stage survey

    # Custom SAE and output directory
    python scripts/run_feature_pipeline.py \\
        --sae-dir trained_models/my-sae \\
        --output-dir my_feature_data

Stages (in order):
    0a  download    Fetch SwissProt sequences as FASTA
    0b  cluster     Cluster sequences with MMseqs2
    1   survey      Stream all proteins through ESM2 -> SAE
    2   selection   Assign proteins to normalised activation bins
    3   collection  Compute per-residue activations for selected proteins
    4   assembly    Assemble per-feature JSON files
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so that `proteinlens` is importable
# when running the script directly (not via `pip install -e .`).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proteinlens.analysis.feature_pipeline.checkpoint import PipelineState
from proteinlens.analysis.feature_pipeline.config import PipelineConfig


# ===================================================================
# Stage runners (thin wrappers that handle skip-if-done logic)
# ===================================================================


def _run_stage_download(config: PipelineConfig, state: PipelineState) -> None:
    """Stage 0a: Download SwissProt sequences."""
    if state.is_stage_complete("download"):
        print("[pipeline] Stage 0a (download) already complete — skipping.")
        return
    from proteinlens.analysis.feature_pipeline.data_acquisition import (
        download_swissprot_fasta,
    )
    download_swissprot_fasta(config)
    state.mark_stage_complete("download")


def _run_stage_cluster(config: PipelineConfig, state: PipelineState) -> None:
    """Stage 0b: Cluster sequences with MMseqs2."""
    if state.is_stage_complete("cluster"):
        print("[pipeline] Stage 0b (cluster) already complete — skipping.")
        return
    from proteinlens.analysis.feature_pipeline.clustering import (
        run_mmseqs_clustering,
    )
    run_mmseqs_clustering(config)
    state.mark_stage_complete("cluster")


def _run_stage_survey(config: PipelineConfig, state: PipelineState) -> None:
    """Stage 1: Survey pass."""
    if state.is_stage_complete("survey"):
        print("[pipeline] Stage 1 (survey) already complete — skipping.")
        return
    from proteinlens.analysis.feature_pipeline.clustering import load_cluster_map
    from proteinlens.analysis.feature_pipeline.survey import run_survey

    member_to_rep = load_cluster_map(config)
    run_survey(config, state, member_to_rep)


def _run_stage_selection(config: PipelineConfig, state: PipelineState) -> None:
    """Stage 2: Selection."""
    if state.is_stage_complete("selection"):
        print("[pipeline] Stage 2 (selection) already complete — skipping.")
        return
    from proteinlens.analysis.feature_pipeline.selection import run_selection

    run_selection(config)
    state.mark_stage_complete("selection")


def _run_stage_collection(config: PipelineConfig, state: PipelineState) -> None:
    """Stage 3: Per-residue collection."""
    if state.is_stage_complete("collection"):
        print("[pipeline] Stage 3 (collection) already complete — skipping.")
        return
    from proteinlens.analysis.feature_pipeline.collection import run_collection

    run_collection(config)
    state.mark_stage_complete("collection")


def _run_stage_assembly(config: PipelineConfig, state: PipelineState) -> None:
    """Stage 4: Assembly."""
    if state.is_stage_complete("assembly"):
        print("[pipeline] Stage 4 (assembly) already complete — skipping.")
        return
    from proteinlens.analysis.feature_pipeline.assembly import run_assembly

    run_assembly(config)
    state.mark_stage_complete("assembly")


# Map of stage name -> runner function, in execution order
STAGES = [
    ("download", _run_stage_download),
    ("cluster", _run_stage_cluster),
    ("survey", _run_stage_survey),
    ("selection", _run_stage_selection),
    ("collection", _run_stage_collection),
    ("assembly", _run_stage_assembly),
]
STAGE_NAMES = [name for name, _ in STAGES]


# ===================================================================
# CLI
# ===================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace with fields matching PipelineConfig
        constructor arguments plus ``--stage`` for single-stage runs.
    """
    parser = argparse.ArgumentParser(
        description="Run the SAE feature data pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sae-dir",
        type=Path,
        default=Path("trained_models/fiery-sweep"),
        help="Path to trained SAE directory (default: trained_models/fiery-sweep)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("feature_data"),
        help="Output directory for pipeline results (default: feature_data/)",
    )
    parser.add_argument(
        "--organism-taxid",
        type=int,
        default=9606,
        help="NCBI taxonomy ID (default: 9606 for Homo sapiens)",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=None,
        help="Cap on number of proteins to process (default: all)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=STAGE_NAMES,
        help="Run only this stage (default: run all stages in order)",
    )
    parser.add_argument(
        "--esm-model",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace ESM model name",
    )
    parser.add_argument(
        "--esm-layer",
        type=int,
        default=3,
        help="ESM layer to extract embeddings from (default: 3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device (default: auto-detect)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: parse args, build config, run stages."""
    args = parse_args()

    config = PipelineConfig(
        sae_dir=args.sae_dir,
        output_dir=args.output_dir,
        organism_taxid=args.organism_taxid,
        max_proteins=args.max_proteins,
        esm_model_name=args.esm_model,
        esm_layer=args.esm_layer,
        device=args.device,
    )

    state = PipelineState(config.pipeline_state_path)

    print("=" * 60)
    print("Feature Data Pipeline")
    print("=" * 60)
    print(f"  SAE dir:       {config.sae_dir}")
    print(f"  Output dir:    {config.output_dir}")
    print(f"  Organism:      taxid {config.organism_taxid}")
    print(f"  Max proteins:  {config.max_proteins or 'all'}")
    print(f"  ESM model:     {config.esm_model_name}")
    print(f"  ESM layer:     {config.esm_layer}")
    print(f"  Device:        {config.device or 'auto'}")
    print("=" * 60)

    t0 = time.time()

    if args.stage is not None:
        # Run a single stage
        stage_map = dict(STAGES)
        print(f"\n>>> Running single stage: {args.stage}")
        stage_map[args.stage](config, state)
    else:
        # Run all stages in order
        for stage_name, stage_fn in STAGES:
            print(f"\n>>> Stage: {stage_name}")
            stage_fn(config, state)

    elapsed = time.time() - t0
    print(f"\nPipeline finished in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
