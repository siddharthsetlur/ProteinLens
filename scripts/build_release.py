"""
Build the ProteinLens artifact release trees for Hugging Face upload.

The release is split into three bundles, each mapping to one HF repo:

  models  -> HF model repo.   SAE weights + sanitized configs (~80 MB).
             Entry point: "give me the trained SAEs".

  paper   -> HF dataset repo. Exactly the artifacts the table/figure
             generators read (~2.5 GB/layer). Entry point: "regenerate
             Tables 1-4, 7, 8 and Figure 6 and check them against the paper".

  viz     -> HF dataset repo. Everything GeoPedia's index builder and API
             require (~10 GB/layer). Entry point: "launch the visualizer".

Bundle trees mirror the on-disk layout the code already expects
(``trained_models/layer_N/<run>/analysis/...``) so a user can download a
bundle and drop it at the repository root without rewriting any path.

Files are hard-linked when the destination shares a filesystem with the
source and copied otherwise, so building the viz bundle does not duplicate
tens of gigabytes unnecessarily.

Usage
-----
    python scripts/build_release.py --source /mnt/datastore/pvc-raw \
        --out release --bundles models,paper,viz

Then upload each bundle tree with ``hf upload-large-folder`` (see
``docs/data_release.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger("build_release")

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Bundle contents
#
# Each entry is a path relative to a run's ``analysis/`` directory. Directories
# are taken whole; files are taken individually. Anything not listed here is
# deliberately excluded -- see EXCLUDED_WITH_REASON for the rationale.
# ---------------------------------------------------------------------------

# Artifacts read by scripts/paper_tables.py, build_subdomain_case_study.py,
# build_nmpfam_transfer_summary.py and figure6_descriptor_counts.py.
PAPER_ANALYSIS_ITEMS = [
    "permutation_null",             # fixed-score raw p-values: Tables 1-3, Fig 6
    "geometry_classifiers",         # importances: Table 3 cosines, Fig 6
    "motif_pwm_enrichment",         # Table 1 MEME column
    "interpro_enrichment",          # Tables 1, 3
    "position_enrichment",          # Table 1 position column
    "nmpfam/nmpfam_enrichment",     # Table 4 raw per-hit profiles
    "interpro_selection.json",      # Table 3 eligible groups
    "geometry_primary_analysis.json",
    "dataset_stats.json",
    "cluster_map.tsv",
    "selection.json",
]

# Additional artifacts required by proteinlens/viz/index_builder.py and api.py.
VIZ_EXTRA_ANALYSIS_ITEMS = [
    "features",                     # per-feature detail pages (api.py:185)
    "geometry_enrichment",          # radar glyphs (index_builder.py:239)
    "cath_enrichment",              # index_builder.py:232
    "feature_max_activations.npy",  # index_builder.py:195
    "survey_coverage.json",         # index_builder.py:207
    "survey_top20.json",
    "sequences.json",
    "pipeline_state.json",          # index_builder.py:156
    "transfer_metrics",             # api.py:292 reads transfer_metrics/metric_B.json
]

# Excluded on purpose. Recorded so the dataset card can state why.
EXCLUDED_WITH_REASON = {
    "pdb_cache": "AlphaFold structures; api.py falls back to the AlphaFold REST API",
    "interpro_cache": "raw EBI InterPro responses; re-fetchable from a pinned release",
    "swissprot_all.fasta": "UniProt/SwissProt; ship a pinned-release fetch script instead",
    "residue_activations": "pipeline intermediate, not read by tables or viz",
    "geom_refit": "layer-4 refit robustness check, not the paper's primary null",
    "geometry_null_refit": "refit robustness check, not the paper's primary null",
    "protein_feature_maxes.npy": "2 GB raw memmap; opt in with --include-protein-maxes",
}

# Files copied from each run root (not from analysis/).
RUN_ROOT_ITEMS = ["config.yaml", "final_evaluation.yaml"]

# Config keys whose values are machine-specific absolute paths.
PATH_KEYS_TO_SANITIZE = {
    "plm_embd_dir",
    "eval_seq_path",
    "eval_embd_dir",
    "save_dir",
    "zscore_means_file",
    "zscore_vars_file",
}


def load_paper_layers() -> dict[int, dict]:
    """Read canonical layer -> run mapping from paper_manifest.yaml."""
    manifest = yaml.safe_load((REPO_ROOT / "paper_manifest.yaml").read_text())
    return {int(k): v for k, v in manifest["layers"].items()}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 so multi-GB artifacts do not have to fit in memory."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def place(src: Path, dst: Path, mode: str) -> int:
    """
    Materialize ``src`` at ``dst``. Returns the number of bytes placed.

    Hard-linking avoids duplicating the viz bundle's ~31 GB when source and
    destination share a filesystem; it silently falls back to copying across
    device boundaries.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst.stat().st_size
    if mode == "link":
        try:
            os.link(src, dst)
            return dst.stat().st_size
        except OSError:
            pass  # cross-device or unsupported; fall through to copy
    shutil.copy2(src, dst)
    return dst.stat().st_size


def place_item(src_root: Path, rel: str, dst_root: Path, mode: str) -> tuple[int, int]:
    """Place one manifest entry (file or whole directory). Returns (files, bytes)."""
    src = src_root / rel
    if not src.exists():
        logger.warning("MISSING: %s", src)
        return (0, 0)
    if src.is_file():
        return (1, place(src, dst_root / rel, mode))

    n_files = 0
    n_bytes = 0
    for path in sorted(src.rglob("*")):
        if path.is_file():
            n_bytes += place(path, dst_root / rel / path.relative_to(src), mode)
            n_files += 1
    return (n_files, n_bytes)


def sanitize_config(src: Path, dst: Path) -> None:
    """
    Copy a run config with machine-specific absolute paths replaced.

    Hyperparameters are portable and must survive verbatim; recorded input and
    output locations are not, and leak the author's filesystem layout.
    """
    cfg = yaml.safe_load(src.read_text())

    def scrub(node):
        if isinstance(node, dict):
            return {
                k: ("<PATH_REDACTED_ON_RELEASE>" if k in PATH_KEYS_TO_SANITIZE and v else scrub(v))
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [scrub(v) for v in node]
        return node

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(scrub(cfg), sort_keys=False))


def write_memmap_sidecar(npy_path: Path, n_features: int) -> None:
    """
    Describe a raw float32 memmap so downstream users can actually read it.

    ``protein_feature_maxes.npy`` and ``feature_max_activations.npy`` are raw
    memmaps rather than real .npy containers -- ``np.load`` fails on them. The
    sidecar records shape and dtype so a reader can reconstruct the array.
    """
    if not npy_path.exists():
        return
    itemsize = np.dtype("float32").itemsize
    total = npy_path.stat().st_size // itemsize
    n_rows = total // n_features if n_features else 0
    sidecar = {
        "format": "raw memmap, not a .npy container",
        "dtype": "float32",
        "shape": [n_rows, n_features],
        "order": "C",
        "read_with": (
            f"np.memmap(path, dtype='float32', mode='r', shape=({n_rows}, {n_features}))"
        ),
        "note": "np.load() will not work on this file.",
    }
    if n_rows * n_features != total:
        sidecar["warning"] = (
            f"file holds {total} float32 values, not divisible by n_features="
            f"{n_features}; shape is unverified"
        )
    npy_path.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2))


def build_bundle(
    bundle: str,
    source: Path,
    out_root: Path,
    layers: dict[int, dict],
    mode: str,
    include_protein_maxes: bool,
) -> dict:
    """Materialize one bundle tree and return its summary record."""
    dst_root = out_root / bundle
    summary = {"bundle": bundle, "layers": {}, "total_files": 0, "total_bytes": 0}

    if bundle == "paper":
        items = list(PAPER_ANALYSIS_ITEMS)
    elif bundle == "viz":
        items = list(PAPER_ANALYSIS_ITEMS) + list(VIZ_EXTRA_ANALYSIS_ITEMS)
    else:
        items = []
    if include_protein_maxes and bundle in {"paper", "viz"}:
        items.append("protein_feature_maxes.npy")

    for layer, spec in sorted(layers.items()):
        run_rel = Path(spec["sae_dir"])  # trained_models/layer_N/<run>
        src_run = source / run_rel
        if not src_run.exists():
            # Tolerate a datastore laid out as <source>/<run> without the
            # trained_models/layer_N prefix.
            alt = source / run_rel.name
            if alt.exists():
                src_run = alt
            else:
                logger.error("layer %s: run dir not found under %s", layer, source)
                continue

        dst_run = dst_root / run_rel
        n_files = 0
        n_bytes = 0

        if bundle == "models":
            f, b = place_item(src_run, "ae.pt", dst_run, mode)
            n_files += f
            n_bytes += b
            for name in RUN_ROOT_ITEMS:
                if name == "config.yaml" and (src_run / name).exists():
                    sanitize_config(src_run / name, dst_run / name)
                    n_files += 1
                    n_bytes += (dst_run / name).stat().st_size
                else:
                    f, b = place_item(src_run, name, dst_run, mode)
                    n_files += f
                    n_bytes += b
        else:
            src_analysis = src_run / "analysis"
            dst_analysis = dst_run / "analysis"
            for rel in items:
                f, b = place_item(src_analysis, rel, dst_analysis, mode)
                n_files += f
                n_bytes += b
            # config.yaml sits at the run root but the viz reads it via the
            # analysis dir resolution in index_builder.py:110.
            if (src_run / "config.yaml").exists():
                sanitize_config(src_run / "config.yaml", dst_run / "config.yaml")
                n_files += 1

            n_feat = int(spec.get("dictionary_size") or 0)
            write_memmap_sidecar(dst_analysis / "feature_max_activations.npy", n_feat)
            write_memmap_sidecar(dst_analysis / "protein_feature_maxes.npy", n_feat)

        summary["layers"][layer] = {
            "run": spec["sae_run"],
            "files": n_files,
            "bytes": n_bytes,
        }
        summary["total_files"] += n_files
        summary["total_bytes"] += n_bytes
        logger.info(
            "%s / layer %s (%s): %d files, %.2f GB",
            bundle, layer, spec["sae_run"], n_files, n_bytes / 1e9,
        )

    return summary


def emit_upload_patterns(bundle: str, layers: dict[int, dict], include_protein_maxes: bool) -> str:
    """
    Print ``hf upload-large-folder --include`` globs for a bundle.

    This is the zero-copy path: rather than staging a tree, point the uploader
    straight at the datastore root and let it select files by pattern. Note it
    skips config sanitization and the memmap sidecars, which must be generated
    separately and pushed as a follow-up commit (see docs/data_release.md).
    """
    if bundle == "paper":
        items = list(PAPER_ANALYSIS_ITEMS)
    elif bundle == "viz":
        items = list(PAPER_ANALYSIS_ITEMS) + list(VIZ_EXTRA_ANALYSIS_ITEMS)
    else:
        items = []
    if include_protein_maxes and bundle in {"paper", "viz"}:
        items.append("protein_feature_maxes.npy")

    patterns: list[str] = []
    for _layer, spec in sorted(layers.items()):
        run = spec["sae_dir"]
        if bundle == "models":
            patterns.append(f"{run}/ae.pt")
            for name in RUN_ROOT_ITEMS:
                patterns.append(f"{run}/{name}")
        else:
            for rel in items:
                # A trailing /** matches directory contents; plain files match as-is.
                suffix = "" if Path(rel).suffix else "/**"
                patterns.append(f"{run}/analysis/{rel}{suffix}")
            patterns.append(f"{run}/final_evaluation.yaml")

    quoted = " \\\n    ".join(f'"{p}"' for p in patterns)
    return f"--include \\\n    {quoted}"


def write_checksums(bundle_root: Path) -> Path:
    """Emit SHA256SUMS over every file in a bundle tree."""
    lines = []
    for path in sorted(bundle_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256_file(path)}  {path.relative_to(bundle_root)}")
    out = bundle_root / "SHA256SUMS"
    out.write_text("\n".join(lines) + "\n")
    logger.info("wrote %s (%d files)", out, len(lines))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=None,
                    help="Datastore root holding trained_models/layer_N/<run>/ "
                         "(not needed with --emit-patterns)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "release",
                    help="Destination root for the bundle trees")
    ap.add_argument("--bundles", default="models,paper",
                    help="Comma-separated subset of: models,paper,viz")
    ap.add_argument("--mode", choices=["link", "copy"], default="link",
                    help="Hard-link (default) or copy files from the source")
    ap.add_argument("--include-protein-maxes", action="store_true",
                    help="Include the 2 GB protein_feature_maxes.npy memmap")
    ap.add_argument("--checksums", action="store_true",
                    help="Emit SHA256SUMS per bundle (slow on the viz bundle)")
    ap.add_argument("--emit-patterns", action="store_true",
                    help="Print hf upload --include globs instead of staging a tree "
                         "(zero-copy: upload straight from the datastore)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    layers = load_paper_layers()
    requested = [b.strip() for b in args.bundles.split(",") if b.strip()]
    unknown = set(requested) - {"models", "paper", "viz"}
    if unknown:
        logger.error("unknown bundles: %s", ", ".join(sorted(unknown)))
        return 1

    if args.emit_patterns:
        for bundle in requested:
            print(f"\n# ---- bundle: {bundle} ----")
            print(emit_upload_patterns(bundle, layers, args.include_protein_maxes))
        return 0

    if args.source is None:
        logger.error("--source is required unless --emit-patterns is given")
        return 1
    if not args.source.exists():
        logger.error("source does not exist: %s", args.source)
        return 1

    summaries = []
    for bundle in requested:
        summary = build_bundle(
            bundle, args.source, args.out, layers, args.mode, args.include_protein_maxes
        )
        if args.checksums:
            write_checksums(args.out / bundle)
        summaries.append(summary)

    report = {
        "bundles": summaries,
        "excluded_with_reason": EXCLUDED_WITH_REASON,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "build_report.json").write_text(json.dumps(report, indent=2))

    print()
    for s in summaries:
        print(f"  {s['bundle']:<7} {s['total_files']:>8,} files  {s['total_bytes']/1e9:>8.2f} GB")
    print(f"\nwrote {args.out / 'build_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
