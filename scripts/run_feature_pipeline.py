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
    from proteinlens.analysis.feature_pipeline.clustering import (
        load_cluster_map,
        sample_representative_accessions,
    )
    from proteinlens.analysis.feature_pipeline.survey import run_survey

    member_to_rep = load_cluster_map(config)

    # If max_proteins is set, sample cluster representatives for diversity.
    # One representative per cluster -> exactly max_proteins proteins.
    if config.max_proteins is not None:
        sampled = sample_representative_accessions(
            member_to_rep, config.max_proteins
        )
        n_clusters = len(set(member_to_rep.values()))
        print(
            f"[pipeline] Sampled {len(sampled)} cluster representatives from "
            f"{n_clusters} clusters (max_proteins={config.max_proteins})."
        )
        # Filter cluster map to only sampled representatives
        member_to_rep = {m: m for m in sampled}

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


def _run_stage_interpro_selection(
    config: PipelineConfig, state: PipelineState
) -> None:
    """Stage 5a: InterPro stratified selection + per-residue collection."""
    if state.is_stage_complete("interpro_selection"):
        print("[pipeline] Stage 5a (interpro_selection) already complete — skipping.")
        return
    from proteinlens.analysis.feature_pipeline.interpro_selection import (
        run_interpro_selection,
    )

    run_interpro_selection(config)
    state.mark_stage_complete("interpro_selection")


def _run_stage_interpro_fetch(
    config: PipelineConfig, state: PipelineState
) -> None:
    """Stage 5b: Fetch InterPro annotations for selected proteins."""
    if state.is_stage_complete("interpro_fetch"):
        print("[pipeline] Stage 5b (interpro_fetch) already complete — skipping.")
        return
    from proteinlens.analysis.feature_pipeline.interpro_api import (
        run_interpro_fetch,
    )

    run_interpro_fetch(config)
    state.mark_stage_complete("interpro_fetch")


def _run_stage_interpro_enrichment(
    config: PipelineConfig, state: PipelineState
) -> None:
    """Stage 5c: Compute InterPro F1 enrichment scores per feature."""
    if state.is_stage_complete("interpro_enrichment"):
        print("[pipeline] Stage 5c (interpro_enrichment) already complete — skipping.")
        return
    from proteinlens.analysis.feature_pipeline.interpro_enrichment import (
        run_interpro_enrichment,
    )

    run_interpro_enrichment(config)
    state.mark_stage_complete("interpro_enrichment")


def _run_stage_geometry_features(
    config: PipelineConfig, state: PipelineState
) -> None:
    """Stage 6a: Compute geometry for all proteins with PDBs."""
    if state.is_stage_complete("geometry_features"):
        print("[pipeline] Stage 6a (geometry_features) already complete — skipping.")
        return
    from proteinlens.analysis.feature_pipeline.geometry_features import (
        run_geometry_features,
    )

    run_geometry_features(config)
    state.mark_stage_complete("geometry_features")


def _run_stage_geometry_residue_enrichment(
    config: PipelineConfig, state: PipelineState
) -> None:
    """Stage 6c: Residue-level GBM geometry enrichment + plot data."""
    if state.is_stage_complete("geometry_residue_enrichment"):
        print("[pipeline] Stage 6c (geometry_residue_enrichment) already complete — skipping.")
        return
    from proteinlens.analysis.feature_pipeline.geometry_residue_enrichment import (
        run_geometry_residue_enrichment,
    )

    run_geometry_residue_enrichment(config)
    state.mark_stage_complete("geometry_residue_enrichment")


def _run_stage_motif_enrichment(
    config: PipelineConfig, state: PipelineState
) -> None:
    """Stage 7: Sequence motif (k-mer) F1 enrichment per feature."""
    if state.is_stage_complete("motif_enrichment"):
        print("[pipeline] Stage 7 (motif_enrichment) already complete — skipping.")
        return
    from proteinlens.analysis.feature_pipeline.motif_enrichment import (
        run_motif_enrichment,
    )

    run_motif_enrichment(config)
    state.mark_stage_complete("motif_enrichment")


def _run_stage_motif_pwm(
    config: PipelineConfig, state: PipelineState
) -> None:
    """Stage 7b: PWM-based motif discovery (MEME). Optional, gated by config."""
    if not config.motif_pwm_enabled:
        return
    if state.is_stage_complete("motif_pwm"):
        print("[pipeline] Stage 7b (motif_pwm) already complete — skipping.")
        return
    from proteinlens.analysis.feature_pipeline.motif_pwm import (
        run_motif_pwm_enrichment,
    )

    run_motif_pwm_enrichment(config)
    state.mark_stage_complete("motif_pwm")


def _run_stage_position_enrichment(
    config: PipelineConfig, state: PipelineState
) -> None:
    """Stage 8: Sequence position F1 enrichment per feature."""
    if state.is_stage_complete("position_enrichment"):
        print("[pipeline] Stage 8 (position_enrichment) already complete — skipping.")
        return
    from proteinlens.analysis.feature_pipeline.position_enrichment import (
        run_position_enrichment,
    )

    run_position_enrichment(config)
    state.mark_stage_complete("position_enrichment")


# Map of stage name -> runner function, in execution order
STAGES = [
    ("download", _run_stage_download),
    ("cluster", _run_stage_cluster),
    ("survey", _run_stage_survey),
    ("selection", _run_stage_selection),
    ("collection", _run_stage_collection),
    ("assembly", _run_stage_assembly),
    ("interpro_selection", _run_stage_interpro_selection),
    ("interpro_fetch", _run_stage_interpro_fetch),
    ("interpro_enrichment", _run_stage_interpro_enrichment),
    ("geometry_features", _run_stage_geometry_features),
    ("geometry_residue_enrichment", _run_stage_geometry_residue_enrichment),
    ("motif_enrichment", _run_stage_motif_enrichment),
    ("motif_pwm", _run_stage_motif_pwm),
    ("position_enrichment", _run_stage_position_enrichment),
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
        help="NCBI taxonomy ID (default: 9606 for Homo sapiens). Use 0 for all organisms (full SwissProt).",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=None,
        help="Cap on number of proteins to process (default: all)",
    )
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=STAGE_NAMES,
        help="Run only this stage (default: run all stages in order)",
    )
    stage_group.add_argument(
        "--start-stage",
        type=str,
        default=None,
        choices=STAGE_NAMES,
        help="First stage to run (inclusive). Use with --end-stage for a range.",
    )
    parser.add_argument(
        "--end-stage",
        type=str,
        default=None,
        choices=STAGE_NAMES,
        help="Last stage to run (inclusive). Defaults to last stage if omitted.",
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
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable wandb logging for pipeline progress",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="proteinlens-pipeline",
        help="wandb project name (default: proteinlens-pipeline)",
    )
    parser.add_argument(
        "--motif-pwm",
        action="store_true",
        default=False,
        help=(
            "Enable optional Stage 7b PWM motif discovery via MEME. "
            "Requires the `meme` binary on PATH."
        ),
    )
    args = parser.parse_args()
    if args.end_stage and not args.start_stage:
        parser.error("--end-stage requires --start-stage")
    if args.start_stage and args.end_stage:
        start_idx = STAGE_NAMES.index(args.start_stage)
        end_idx = STAGE_NAMES.index(args.end_stage)
        if start_idx > end_idx:
            parser.error(
                f"--start-stage '{args.start_stage}' comes after "
                f"--end-stage '{args.end_stage}'"
            )
    return args


def main() -> None:
    """CLI entry point: parse args, build config, run stages."""
    args = parse_args()

    taxid = args.organism_taxid if args.organism_taxid != 0 else None

    config = PipelineConfig(
        sae_dir=args.sae_dir,
        output_dir=args.output_dir,
        organism_taxid=taxid,
        max_proteins=args.max_proteins,
        esm_model_name=args.esm_model,
        esm_layer=args.esm_layer,
        device=args.device,
        motif_pwm_enabled=args.motif_pwm,
    )

    state = PipelineState(config.pipeline_state_path)

    print("=" * 60)
    print("Feature Data Pipeline")
    print("=" * 60)
    print(f"  SAE dir:       {config.sae_dir}")
    print(f"  Output dir:    {config.output_dir}")
    org_label = f"taxid {config.organism_taxid}" if config.organism_taxid is not None else "all organisms"
    print(f"  Organism:      {org_label}")
    print(f"  Max proteins:  {config.max_proteins or 'all'}")
    print(f"  ESM model:     {config.esm_model_name}")
    print(f"  ESM layer:     {config.esm_layer}")
    print(f"  Device:        {config.device or 'auto'}")
    print("=" * 60)

    # ── Optional wandb init ──
    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            config={
                "sae_dir": str(config.sae_dir),
                "output_dir": str(config.output_dir),
                "organism_taxid": config.organism_taxid,
                "max_proteins": config.max_proteins,
                "esm_model_name": config.esm_model_name,
                "esm_layer": config.esm_layer,
                "device": config.device,
            },
        )

    t0 = time.time()

    from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog

    if args.stage is not None:
        # Run a single stage
        stage_map = dict(STAGES)
        print(f"\n>>> Running single stage: {args.stage}")
        stage_map[args.stage](config, state)
    elif args.start_stage is not None:
        # Run a contiguous range of stages
        start_idx = STAGE_NAMES.index(args.start_stage)
        end_idx = (
            STAGE_NAMES.index(args.end_stage) if args.end_stage else len(STAGES) - 1
        )
        print(
            f"\n>>> Running stages: {args.start_stage} through "
            f"{STAGE_NAMES[end_idx]}"
        )
        for stage_name, stage_fn in STAGES[start_idx : end_idx + 1]:
            print(f"\n>>> Stage: {stage_name}")
            wlog({"stage": stage_name})
            stage_fn(config, state)
    else:
        # Run all stages in order
        for stage_name, stage_fn in STAGES:
            print(f"\n>>> Stage: {stage_name}")
            wlog({"stage": stage_name})
            stage_fn(config, state)

    elapsed = time.time() - t0
    print(f"\nPipeline finished in {elapsed:.1f}s.")

    from proteinlens.analysis.feature_pipeline.wandb_utils import log as wlog

    wlog({"pipeline_elapsed_s": elapsed})
    if args.wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
