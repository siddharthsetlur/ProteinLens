#!/usr/bin/env python3
"""Validate candidate paper artifacts against model/layer metadata.

This verifies identity and emits local fingerprints. It does not declare an
unresolved local snapshot to be the published snapshot; --strict requires a
release snapshot ID and populated checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

FINGERPRINT_FILES = [
    "dataset_stats.json",
    "feature_max_activations.npy",
    "geometry_primary_analysis.json",
    "survey_coverage.json",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_layer(
    layer: int, spec: dict[str, Any], expected_model: str
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    analysis_dir = Path(spec["analysis_dir"])
    sae_dir = Path(spec["sae_dir"])
    stats_path = analysis_dir / "dataset_stats.json"
    config_path = sae_dir / "config.yaml"
    if not stats_path.exists():
        return [f"layer {layer}: missing {stats_path}"], {}
    if not config_path.exists():
        return [f"layer {layer}: missing {config_path}"], {}

    stats = json.loads(stats_path.read_text())
    config = yaml.safe_load(config_path.read_text())
    eval_config = config.get("eval_cfg", {})
    trainer_config = config.get("trainer_cfg", {})

    checks = [
        ("dataset_stats.esm_layer", stats.get("esm_layer"), layer),
        ("config.eval_cfg.layer_idx", eval_config.get("layer_idx"), layer),
        ("dataset_stats.esm_model", stats.get("esm_model"), expected_model),
        (
            "config.eval_cfg.model_name",
            eval_config.get("model_name"),
            expected_model.removeprefix("facebook/"),
        ),
        (
            "dataset_stats.num_features",
            stats.get("num_features"),
            spec["dictionary_size"],
        ),
        (
            "config.trainer_cfg.dictionary_size",
            trainer_config.get("dictionary_size"),
            spec["dictionary_size"],
        ),
    ]
    for field, actual, expected in checks:
        if actual != expected:
            errors.append(
                f"layer {layer}: {field}={actual!r}, expected {expected!r}"
            )

    fingerprints = {}
    for relative in FINGERPRINT_FILES:
        path = analysis_dir / relative
        if path.exists():
            fingerprints[relative] = digest(path)
    fingerprints["../config.yaml"] = digest(config_path)
    expected_hashes = spec.get("sha256") or {}
    for relative, expected in expected_hashes.items():
        actual = fingerprints.get(relative)
        if actual != expected:
            errors.append(
                f"layer {layer}: checksum mismatch for {relative}: "
                f"{actual} != {expected}"
            )
    identity = {
        "analysis_dir": str(analysis_dir),
        "sae_run": spec.get("sae_run"),
        "esm_model": stats.get("esm_model"),
        "esm_layer": stats.get("esm_layer"),
        "dictionary_size": stats.get("num_features"),
        "total_proteins": stats.get("total_proteins"),
        "total_clusters": stats.get("total_clusters"),
        "fingerprints": fingerprints,
    }
    return errors, identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("paper_manifest.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also require a resolved release snapshot and checksums",
    )
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    expected_model = manifest["paper"]["model"]
    errors: list[str] = []
    identities: dict[str, Any] = {}
    for layer_text, spec in manifest["layers"].items():
        layer = int(layer_text)
        layer_errors, identity = check_layer(layer, spec, expected_model)
        errors.extend(layer_errors)
        identities[str(layer)] = identity

    if args.strict:
        release = manifest.get("artifact_release") or {}
        if not release.get("snapshot_id"):
            errors.append("artifact_release.snapshot_id is unresolved")
        for layer_text, spec in manifest["layers"].items():
            if not spec.get("sha256"):
                errors.append(f"layer {layer_text}: no release checksums pinned")

    result = {
        "manifest": str(args.manifest),
        "strict": args.strict,
        "identity_checks_passed": not errors,
        "release_status": manifest.get("artifact_release", {}).get("status"),
        "errors": errors,
        "layers": identities,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
